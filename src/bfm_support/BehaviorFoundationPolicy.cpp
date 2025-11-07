#include "motion_tracking_controller/bfm_support/BehaviorFoundationPolicy.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <rclcpp/logging.hpp>

namespace {
void sanitizeVector(Eigen::VectorXf& vec,
                    const std::string& label,
                    const rclcpp::Logger& logger) {
  if (vec.allFinite()) {
    return;
  }
  RCLCPP_ERROR(logger, "Non-finite entries detected in %s. Clamping to zero.", label.c_str());
  for (int i = 0; i < vec.size(); ++i) {
    if (!std::isfinite(vec[i])) {
      vec[i] = 0.0f;
    }
  }
}
}  // namespace

namespace legged::bfm {

BehaviorFoundationPolicy::BehaviorFoundationPolicy(BfmPaths paths, std::string provider)
    : paths_(std::move(paths)), provider_(std::move(provider)) {}

void BehaviorFoundationPolicy::init() {
  metadata_ = loadMetadata(paths_.metadata);
  try {
    loadObservationStats(paths_.obs_norm, metadata_);
  } catch (const std::exception& e) {
    RCLCPP_WARN(rclcpp::get_logger("BehaviorFoundationPolicy"), "Failed to load observation stats: %s", e.what());
  }

  configureGoalLayout();
  configureControlMask();

  assembler_ = std::make_unique<ObservationAssembler>(metadata_);
  history_.configure(metadata_.history);

  commandNames_.clear();
  commandNames_.push_back("speed");
  observationNames_.clear();
  observationNames_.push_back("behavior_residual_obs");

  actionScale_ = vector_t::Ones(metadata_.action_dim);
  if (!metadata_.action_scale.empty()) {
    actionScale_ = vector_t::Zero(metadata_.action_dim);
    for (Eigen::Index i = 0; i < actionScale_.size(); ++i) {
      const size_t idx = std::min(static_cast<size_t>(i), metadata_.action_scale.size() - 1);
      actionScale_[i] = metadata_.action_scale[idx];
    }
  }

  last_action_ = vector_t::Zero(std::max(1, metadata_.action_dim));
  cached_sp_real_ = Eigen::VectorXf::Zero(std::max(1, metadata_.dim_sp_real));
  cached_sg_masked_ = Eigen::VectorXf::Zero(std::max(1, metadata_.dim_goal));
  const int residual_width = std::max(metadata_.widthOf("residual"), metadata_.residual_dim);
  last_residual_for_obs_ = Eigen::VectorXf::Zero(std::max(1, residual_width));
  const int residual_norm_width = std::max(metadata_.widthOf("residual_norm"), 1);
  last_residual_norm_ = Eigen::VectorXf::Zero(residual_norm_width);

  env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "motion_tracking_bfm_policy");
  session_options_.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
  session_options_.DisableMemPattern();
  session_options_.SetIntraOpNumThreads(1);

  initSessions();
}

void BehaviorFoundationPolicy::reset() {
  history_.configure(metadata_.history);
  cache_ready_ = false;
  ticks_since_reset_ = 0;
  time_step_counter_ = 0;
  last_action_ = vector_t::Zero(std::max(1, metadata_.action_dim));
  last_residual_for_obs_.setZero();
  last_residual_norm_.setZero();
}

vector_t BehaviorFoundationPolicy::prepareObservation() {
  if (!model_) {
    RCLCPP_WARN(rclcpp::get_logger("BehaviorFoundationPolicy"),
                "LeggedModel not set; returning zero observation.");
    cache_ready_ = false;
    return vector_t::Zero(getObservationSize());
  }

  const vector_t& q = model_->getGeneralizedPosition();
  const vector_t& v = model_->getGeneralizedVelocity();
  if (q.size() < static_cast<Eigen::Index>(metadata_.action_dim) ||
      v.size() < static_cast<Eigen::Index>(metadata_.action_dim + 6)) {
    RCLCPP_ERROR(rclcpp::get_logger("BehaviorFoundationPolicy"),
                 "State dimension mismatch (q=%ld, v=%ld, expected >= %d).",
                 q.size(),
                 v.size(),
                 metadata_.action_dim);
    cache_ready_ = false;
    return vector_t::Zero(getObservationSize());
  }

  const Eigen::Matrix3f rot = orientationMatrix();
  const Eigen::Vector3d baseLinWorld = v.head<3>();
  const Eigen::Vector3d baseAngWorld = v.segment<3>(3);
  const Eigen::Vector3f baseLinBody = (rot.transpose().cast<double>() * baseLinWorld).cast<float>();
  const Eigen::Vector3f baseAngBody = baseAngWorld.cast<float>();
  Eigen::Vector3f gravityBody = rot.transpose() * Eigen::Vector3f(0.0f, 0.0f, -1.0f);
  if (gravityBody.norm() > 1e-5f) {
    gravityBody.normalize();
  }

  const Eigen::VectorXf jointPos = q.tail(metadata_.action_dim).cast<float>();
  const Eigen::VectorXf jointVel = v.tail(metadata_.action_dim).cast<float>();

  Eigen::VectorXf features = Eigen::VectorXf::Zero(std::max(1, metadata_.history.featureDim()));
  int featureOffset = 0;
  const auto assignFeature = [&](const Eigen::VectorXf& src, int width) {
    if (width <= 0) {
      return;
    }
    const int remaining = std::max(0, static_cast<int>(features.size()) - featureOffset);
    if (remaining <= 0) {
      featureOffset += width;
      return;
    }
    const int copy = std::min(width, remaining);
    if (copy > 0 && src.size() >= copy) {
      features.segment(featureOffset, copy) = src.head(copy);
    }
    featureOffset += width;
  };

  if (metadata_.history.dim_base_lin_vel > 0) {
    Eigen::VectorXf baseLinVec(baseLinBody.size());
    baseLinVec = baseLinBody;
    assignFeature(baseLinVec, metadata_.history.dim_base_lin_vel);
  }
  assignFeature(jointPos, metadata_.history.dim_joint_pos);
  assignFeature(jointVel, metadata_.history.dim_joint_vel);
  Eigen::VectorXf baseAngVec(baseAngBody.size());
  baseAngVec = baseAngBody;
  assignFeature(baseAngVec, metadata_.history.dim_root_ang_vel);
  Eigen::VectorXf gravityVec(gravityBody.size());
  gravityVec = gravityBody;
  assignFeature(gravityVec, metadata_.history.dim_gravity);

  Eigen::VectorXf lastActionFloat = Eigen::VectorXf::Zero(metadata_.history.dim_last_action);
  if (metadata_.history.dim_last_action > 0 && last_action_.size() > 0) {
    const int copy = std::min(static_cast<int>(lastActionFloat.size()), static_cast<int>(last_action_.size()));
    if (copy > 0) {
      lastActionFloat.head(copy) = last_action_.head(copy).cast<float>();
    }
  }

  Eigen::VectorXf sp_real = history_.push(features, lastActionFloat);

  vector_t cmd = command_manager_ ? command_manager_->getValue() : vector_t();
  double cmdX = cmd.size() > 0 ? cmd[0] : 0.0;
  double cmdY = cmd.size() > 1 ? cmd[1] : 0.0;
  double cmdYaw = cmd.size() > 2 ? cmd[2] : 0.0;
  const double speed_mag = std::hypot(cmdX, cmdY);
  if (speed_mag < 1e-4) {
    cmdX = default_forward_speed_;
    cmdY = 0.0;
  }
  const Eigen::Vector3d commandWorld(cmdX, cmdY, 0.0);
  const Eigen::Quaterniond baseQuat = model_->getBaseRotation();
  const Eigen::Vector2f dirBody = computeDirectionBody(commandWorld, baseQuat);
  const float speed = static_cast<float>(std::hypot(cmdX, cmdY));
  const float yawRate = metadata_.speed_cfg.enable_yaw_command ? static_cast<float>(cmdYaw) : 0.0f;

  Eigen::VectorXf sg_real = buildGoalVector(dirBody, speed, yawRate);
  Eigen::VectorXf sg_masked = sg_real;
  if (control_mask_.size() == sg_real.size()) {
    sg_masked = sg_real.cwiseProduct(control_mask_);
  }

  Eigen::VectorXf baseObs =
      assembleBaseObservation(baseLinBody, baseAngBody, gravityBody, jointPos, jointVel);

  Eigen::VectorXf mu_prior;
  Eigen::VectorXf base_action = runBase(sp_real, sg_masked, mu_prior);
  if (metadata_.widthOf("mu_p") > 0 && mu_prior.size() == 0) {
    mu_prior = Eigen::VectorXf::Zero(metadata_.widthOf("mu_p"));
  }

  ComponentMap components;
  for (const auto& [name, slice] : metadata_.component_slices) {
    components[name] = Eigen::VectorXf::Zero(slice.size());
  }

  const auto assignComponent = [&](const std::string& key, const Eigen::VectorXf& value) {
    auto it = components.find(key);
    if (it == components.end()) {
      return;
    }
    Eigen::VectorXf filled = it->second;
    const int copy = std::min(static_cast<int>(filled.size()), static_cast<int>(value.size()));
    if (copy > 0) {
      filled.head(copy) = value.head(copy);
    }
    it->second = filled;
  };

  assignComponent("base_obs", baseObs);
  assignComponent("sp_real", sp_real);
  assignComponent("sg_real_masked", sg_masked);
  assignComponent("base_action", base_action);
  assignComponent("mu_p", mu_prior);
  assignComponent("residual", last_residual_for_obs_);
  assignComponent("residual_norm", last_residual_norm_);

  Eigen::VectorXf dirComp(2);
  dirComp << dirBody[0], dirBody[1];
  assignComponent("dir_b", dirComp);

  Eigen::VectorXf speedComp(1);
  speedComp[0] = speed;
  assignComponent("speed", speedComp);

  Eigen::VectorXf yawComp(1);
  yawComp[0] = yawRate;
  assignComponent("yaw_rate", yawComp);

  Eigen::VectorXf baseLinVec(baseLinBody.size());
  baseLinVec = baseLinBody;
  assignComponent("base_lin_vel", baseLinVec);

  Eigen::VectorXf baseAngComp(baseAngBody.size());
  baseAngComp = baseAngBody;
  assignComponent("base_ang_vel", baseAngComp);

  Eigen::VectorXf observation;
  try {
    observation = assembler_->assemble(components);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("BehaviorFoundationPolicy"),
                 "Failed to assemble residual observation: %s",
                 e.what());
    cache_ready_ = false;
    return vector_t::Zero(getObservationSize());
  }

  if (observation.size() != metadata_.dim_obs || !observation.allFinite()) {
    RCLCPP_ERROR(rclcpp::get_logger("BehaviorFoundationPolicy"),
                 "Residual observation invalid (size=%ld, expected=%d, finite=%d). Zeroing vector.",
                 static_cast<long>(observation.size()),
                 metadata_.dim_obs,
                 observation.allFinite());
    observation = Eigen::VectorXf::Zero(std::max(1, metadata_.dim_obs));
  }

  if (debug_dump_ && debug_counter_ < 5) {
    RCLCPP_INFO(rclcpp::get_logger("BehaviorFoundationPolicy"),
                "[debug] observation snapshot %d: min=%.5f max=%.5f mean=%.5f",
                debug_counter_,
                observation.minCoeff(),
                observation.maxCoeff(),
                observation.mean());
    for (const auto& [name, slice] : metadata_.component_slices) {
      const auto seg = observation.segment(slice.start, slice.size());
      RCLCPP_INFO(rclcpp::get_logger("BehaviorFoundationPolicy"),
                  "    slice %-20s [%5d:%5d] min=% .4f max=% .4f mean=% .4f",
                  name.c_str(),
                  slice.start,
                  slice.end,
                  seg.minCoeff(),
                  seg.maxCoeff(),
                  seg.mean());
    }
    ++debug_counter_;
  }

  cacheComponents(sp_real, sg_masked, components, observation);

  vector_t out = vector_t::Zero(observation.size());
  out = observation.cast<double>();
  return out;
}

vector_t BehaviorFoundationPolicy::forward(const vector_t& observations) {
  (void)observations;
  if (!cache_ready_) {
    return last_action_;
  }

  const int grace_target = grace_override_ >= 0 ? grace_override_ : metadata_.grace_steps;
  const int grace_remaining = std::max(0, grace_target - ticks_since_reset_);

  StepOut out =
      step(cached_sp_real_, cached_sg_masked_, cached_components_, true, grace_remaining);
  vector_t blended = vector_t::Zero(static_cast<Eigen::Index>(out.blended_action.size()));
  blended = out.blended_action.cast<double>();
  last_action_ = blended;
  last_residual_for_obs_ = out.residual_for_obs;
  last_residual_norm_ = out.residual_norm;
  cache_ready_ = false;
  ++ticks_since_reset_;
  return last_action_;
}

void BehaviorFoundationPolicy::initSessions() {
  if (paths_.residual.empty()) {
    throw std::runtime_error("BehaviorFoundationPolicy requires a residual ONNX path.");
  }

  residual_session_ = std::make_unique<Ort::Session>(*env_, paths_.residual.c_str(), session_options_);
  initSessionIO(*residual_session_, residual_input_names_, residual_input_ptrs_, residual_output_names_,
                residual_output_ptrs_);

  if (!paths_.base.empty()) {
    base_session_ = std::make_unique<Ort::Session>(*env_, paths_.base.c_str(), session_options_);
    initSessionIO(*base_session_, base_input_names_, base_input_ptrs_, base_output_names_, base_output_ptrs_);
  }
}

void BehaviorFoundationPolicy::initSessionIO(Ort::Session& session,
                                             std::vector<std::string>& input_names,
                                             std::vector<const char*>& input_ptrs,
                                             std::vector<std::string>& output_names,
                                             std::vector<const char*>& output_ptrs) {
  Ort::AllocatorWithDefaultOptions allocator;

  const size_t input_count = session.GetInputCount();
  input_names.resize(input_count);
  input_ptrs.resize(input_count);
  for (size_t i = 0; i < input_count; ++i) {
    Ort::AllocatedStringPtr name = session.GetInputNameAllocated(i, allocator);
    input_names[i] = name.get();
    input_ptrs[i] = input_names[i].c_str();
  }

  const size_t output_count = session.GetOutputCount();
  output_names.resize(output_count);
  output_ptrs.resize(output_count);
  for (size_t i = 0; i < output_count; ++i) {
    Ort::AllocatedStringPtr name = session.GetOutputNameAllocated(i, allocator);
    output_names[i] = name.get();
    output_ptrs[i] = output_names[i].c_str();
  }
}

BehaviorFoundationPolicy::StepOut BehaviorFoundationPolicy::step(const Eigen::VectorXf& sp_real,
                                                                 const Eigen::VectorXf& sg_masked,
                                                                 ComponentMap components,
                                                                 bool apply_grace_gate,
                                                                 int grace_remaining) {
  if (!residual_session_) {
    throw std::runtime_error("BehaviorFoundationPolicy: residual session not initialised.");
  }

  Eigen::VectorXf base_action = Eigen::VectorXf::Zero(metadata_.action_dim);
  Eigen::VectorXf mu_prior;
  auto baseIt = components.find("base_action");
  if (baseIt != components.end() && baseIt->second.size() == metadata_.action_dim) {
    base_action = baseIt->second;
  }
  auto muIt = components.find("mu_p");
  if (muIt != components.end() && muIt->second.size() > 0) {
    mu_prior = muIt->second;
  }

  if (base_action.size() != metadata_.action_dim || base_action.isZero(0)) {
    base_action = runBase(sp_real, sg_masked, mu_prior);
    if (baseIt != components.end()) {
      baseIt->second = base_action;
    }
    if (muIt != components.end()) {
      Eigen::VectorXf filled = muIt->second;
      const int copy = std::min(static_cast<int>(filled.size()), static_cast<int>(mu_prior.size()));
      if (copy > 0) {
        filled.head(copy) = mu_prior.head(copy);
      }
      muIt->second = filled;
    }
  }
  if (mu_prior.size() == 0) {
    mu_prior = Eigen::VectorXf::Zero(metadata_.widthOf("mu_p"));
  }

  sanitizeVector(base_action, "base_action", rclcpp::get_logger("BehaviorFoundationPolicy"));

  components["sp_real"] = sp_real;
  components["sg_real_masked"] = sg_masked;
  components["base_action"] = base_action;
  components["mu_p"] = mu_prior;

  const Eigen::VectorXf residual_obs = assembler_->assemble(components);
  Eigen::VectorXf normalized_obs = residual_obs;
  if (metadata_.host_normalize_obs && !metadata_.obs_mean.empty() && !metadata_.obs_var.empty()) {
    normalized_obs = applyNormalization(residual_obs);
  }
  if (!normalized_obs.allFinite()) {
    RCLCPP_ERROR(rclcpp::get_logger("BehaviorFoundationPolicy"),
                 "Normalized residual observation contains non-finite values. Zeroing.");
    normalized_obs.setZero();
  }

  Eigen::VectorXf residual_raw = runResidual(normalized_obs);
  sanitizeVector(residual_raw, "residual_raw", rclcpp::get_logger("BehaviorFoundationPolicy"));
  Eigen::VectorXf residual_clipped =
      residual_raw.cwiseMin(metadata_.residual_clip).cwiseMax(-metadata_.residual_clip);

  if (apply_grace_gate && grace_remaining > 0) {
    residual_clipped.setZero();
  }

  Eigen::VectorXf blended = blendActions(base_action, residual_clipped);
  sanitizeVector(blended, "blended_action", rclcpp::get_logger("BehaviorFoundationPolicy"));

  StepOut out;
  out.base_action = base_action;
  out.residual_raw = residual_raw;
  out.residual_clipped = residual_clipped;
  out.blended_action = blended;
  out.residual_for_obs = residual_clipped.array().tanh().matrix();
  if (out.residual_for_obs.size() > 0) {
    const float rms = std::sqrt(out.residual_for_obs.array().square().mean());
    out.residual_norm = Eigen::VectorXf::Constant(1, rms);
  } else {
    out.residual_norm = Eigen::VectorXf::Zero(1);
  }
  out.mu_p = mu_prior;

  ++time_step_counter_;
  return out;
}

Eigen::VectorXf BehaviorFoundationPolicy::runBase(const Eigen::VectorXf& sp_real,
                                                  const Eigen::VectorXf& sg_masked,
                                                  Eigen::VectorXf& mu_prior) {
  if (!base_session_) {
    mu_prior = Eigen::VectorXf::Zero(metadata_.latent_dim);
    return Eigen::VectorXf::Zero(metadata_.action_dim);
  }

  const Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::vector<Ort::Value> tensors;
  std::vector<const char*> input_names;

  tensors.reserve(base_input_names_.size());
  input_names.reserve(base_input_names_.size());

  const auto make_tensor = [&](const Eigen::VectorXf& vec, std::array<int64_t, 2> shape) {
    return Ort::Value::CreateTensor<float>(mem_info, const_cast<float*>(vec.data()), vec.size(), shape.data(),
                                           shape.size());
  };
  std::vector<std::vector<float>> owned_scalars;

  for (const auto& name : base_input_names_) {
    if (name.find("sp_real") != std::string::npos) {
      std::array<int64_t, 2> shape{1, sp_real.size()};
      tensors.emplace_back(make_tensor(sp_real, shape));
    } else if (name.find("sg_real") != std::string::npos) {
      std::array<int64_t, 2> shape{1, sg_masked.size()};
      tensors.emplace_back(make_tensor(sg_masked, shape));
    } else if (name.find("time") != std::string::npos) {
      owned_scalars.push_back({static_cast<float>(time_step_counter_)});
      std::array<int64_t, 2> shape{1, 1};
      tensors.emplace_back(
          Ort::Value::CreateTensor<float>(mem_info, owned_scalars.back().data(), owned_scalars.back().size(),
                                          shape.data(), shape.size()));
    } else {
      throw std::runtime_error("Unknown base input tensor: " + name);
    }
    input_names.push_back(name.c_str());
  }

  Ort::RunOptions run_options;
  std::vector<Ort::Value> outputs;
  try {
    outputs = base_session_->Run(run_options,
                                 input_names.data(),
                                 tensors.data(),
                                 tensors.size(),
                                 base_output_ptrs_.data(),
                                 base_output_ptrs_.size());
  } catch (const Ort::Exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("BehaviorFoundationPolicy"),
                 "Base ONNX inference failed: %s. Returning zeros.",
                 e.what());
    mu_prior = Eigen::VectorXf::Zero(metadata_.widthOf("mu_p"));
    return Eigen::VectorXf::Zero(metadata_.action_dim);
  }

  Eigen::VectorXf base_action = Eigen::VectorXf::Zero(metadata_.action_dim);
  mu_prior = Eigen::VectorXf::Zero(metadata_.latent_dim);
  for (size_t idx = 0; idx < outputs.size(); ++idx) {
    const auto vector = tensorToVector(outputs[idx]);
    const auto& name = base_output_names_[idx];
    if (name.find("base_action") != std::string::npos) {
      base_action = vector;
    } else if (name.find("mu") != std::string::npos) {
      mu_prior = vector;
    }
  }

  if (base_action.size() == 0 && !outputs.empty()) {
    base_action = tensorToVector(outputs.front());
  }
  return base_action;
}

Eigen::VectorXf BehaviorFoundationPolicy::runResidual(const Eigen::VectorXf& residual_obs) {
  const Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  std::vector<Ort::Value> tensors;
  std::vector<const char*> input_names;
  tensors.reserve(residual_input_names_.size());
  input_names.reserve(residual_input_names_.size());

  const auto make_tensor = [&](const Eigen::VectorXf& vec, std::array<int64_t, 2> shape) {
    return Ort::Value::CreateTensor<float>(mem_info, const_cast<float*>(vec.data()), vec.size(), shape.data(),
                                           shape.size());
  };
  std::vector<std::vector<float>> owned_scalars;

  for (const auto& name : residual_input_names_) {
    if (name.find("obs") != std::string::npos) {
      std::array<int64_t, 2> shape{1, residual_obs.size()};
      tensors.emplace_back(make_tensor(residual_obs, shape));
    } else if (name.find("time") != std::string::npos) {
      owned_scalars.push_back({static_cast<float>(time_step_counter_)});
      std::array<int64_t, 2> shape{1, 1};
      tensors.emplace_back(
          Ort::Value::CreateTensor<float>(mem_info, owned_scalars.back().data(), owned_scalars.back().size(),
                                          shape.data(), shape.size()));
    } else {
      throw std::runtime_error("Unknown residual input tensor: " + name);
    }
    input_names.push_back(name.c_str());
  }

  Ort::RunOptions run_options;
  std::vector<Ort::Value> outputs;
  try {
    outputs = residual_session_->Run(run_options,
                                     input_names.data(),
                                     tensors.data(),
                                     tensors.size(),
                                     residual_output_ptrs_.data(),
                                     residual_output_ptrs_.size());
  } catch (const Ort::Exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("BehaviorFoundationPolicy"),
                 "Residual ONNX inference failed: %s. Returning zeros.",
                 e.what());
    return Eigen::VectorXf::Zero(metadata_.action_dim);
  }
  if (outputs.empty()) {
    throw std::runtime_error("Residual session returned no outputs.");
  }

  return tensorToVector(outputs.front());
}

Eigen::VectorXf BehaviorFoundationPolicy::blendActions(const Eigen::VectorXf& base_action,
                                                       const Eigen::VectorXf& residual) const {
  if (metadata_.residual_mode == "delta_a") {
    return (base_action + residual).array().max(-1.0).min(1.0).matrix();
  }
  return ((1.0 - metadata_.alpha) * base_action.array() + metadata_.alpha * residual.array()).max(-1.0).min(1.0).matrix();
}

Eigen::VectorXf BehaviorFoundationPolicy::tensorToVector(const Ort::Value& value) {
  const auto info = value.GetTensorTypeAndShapeInfo();
  const size_t count = info.GetElementCount();
  const float* data = value.GetTensorData<float>();
  Eigen::VectorXf out(count);
  Eigen::Map<const Eigen::VectorXf> map(data, count);
  out = map;
  return out;
}

void BehaviorFoundationPolicy::cacheComponents(const Eigen::VectorXf& sp_real,
                                                const Eigen::VectorXf& sg_masked,
                                                const ComponentMap& components,
                                                const Eigen::VectorXf& observation) {
  cached_sp_real_ = sp_real;
  cached_sg_masked_ = sg_masked;
  cached_components_ = components;
  cached_observation_ = observation;
  cache_ready_ = true;
}

Eigen::VectorXf BehaviorFoundationPolicy::applyNormalization(const Eigen::VectorXf& observation) const {
  if (!metadata_.host_normalize_obs || metadata_.obs_mean.empty() || metadata_.obs_var.empty()) {
    return observation;
  }
  Eigen::VectorXf normalized = observation;
  const size_t limit = std::min(metadata_.obs_mean.size(), metadata_.obs_var.size());
  for (size_t i = 0; i < std::min(limit, static_cast<size_t>(normalized.size())); ++i) {
    const float variance = metadata_.obs_var[i];
    const float mean = metadata_.obs_mean[i];
    normalized[static_cast<Eigen::Index>(i)] =
        (normalized[static_cast<Eigen::Index>(i)] - mean) / std::sqrt(variance + 1e-6f);
  }
  return normalized;
}

void BehaviorFoundationPolicy::configureGoalLayout() {
  goal_layout_ = {};
  goal_layout_valid_ = false;
  int idx = 0;
  const auto assign = [&](GoalRange& range, int width) {
    const int remaining = std::max(0, metadata_.dim_goal - idx);
    const int clamped = std::min(width, remaining);
    range.start = idx;
    range.end = idx + clamped;
    idx += clamped;
  };

  assign(goal_layout_.root_pos, 3);
  assign(goal_layout_.root_ori, 6);
  assign(goal_layout_.root_lin_vel, 3);
  assign(goal_layout_.root_ang_vel, 3);
  const int remaining_for_keypoints = std::max(0, metadata_.dim_goal - idx - metadata_.action_dim);
  assign(goal_layout_.keypoints, remaining_for_keypoints);
  assign(goal_layout_.joints, metadata_.action_dim);
  goal_layout_valid_ = goal_layout_.root_lin_vel.valid();
}

void BehaviorFoundationPolicy::configureControlMask() {
  control_mask_ = Eigen::VectorXf::Ones(metadata_.dim_goal);
  control_mask_.setZero();
  if (!goal_layout_.root_lin_vel.valid()) {
    return;
  }
  control_mask_.segment(goal_layout_.root_lin_vel.start, goal_layout_.root_lin_vel.size()).setOnes();
  if (metadata_.speed_cfg.enable_yaw_command && goal_layout_.root_ang_vel.valid()) {
    control_mask_.segment(goal_layout_.root_ang_vel.start, goal_layout_.root_ang_vel.size()).setOnes();
  }
}

Eigen::VectorXf BehaviorFoundationPolicy::buildGoalVector(const Eigen::Vector2f& dir_body,
                                                          float speed,
                                                          float yaw_rate) const {
  Eigen::VectorXf goal = Eigen::VectorXf::Zero(metadata_.dim_goal);
  if (goal_layout_.root_lin_vel.valid()) {
    Eigen::Vector3f lin = Eigen::Vector3f::Zero();
    lin.x() = dir_body[0] * speed;
    lin.y() = dir_body[1] * speed;
    const int width = goal_layout_.root_lin_vel.size();
    goal.segment(goal_layout_.root_lin_vel.start, width) = lin.head(width);
  }
  if (goal_layout_.root_ang_vel.valid()) {
    Eigen::Vector3f ang = Eigen::Vector3f::Zero();
    ang.z() = yaw_rate;
    const int width = goal_layout_.root_ang_vel.size();
    goal.segment(goal_layout_.root_ang_vel.start, width) = ang.head(width);
  }
  return goal;
}

Eigen::Vector2f BehaviorFoundationPolicy::computeDirectionBody(const Eigen::Vector3d& command_world,
                                                               const Eigen::Quaterniond& base_quat) const {
  const Eigen::Matrix3d rot = base_quat.toRotationMatrix();
  Eigen::Vector3d body = rot.transpose() * command_world;
  Eigen::Vector2f dir = body.head<2>().cast<float>();
  const float norm = dir.norm();
  if (norm > 1e-6f) {
    dir /= norm;
  } else {
    dir = Eigen::Vector2f::UnitX();
  }
  return dir;
}

Eigen::Matrix3f BehaviorFoundationPolicy::orientationMatrix() const {
  if (!model_) {
    return Eigen::Matrix3f::Identity();
  }
  const auto quat = model_->getBaseRotation();
  Eigen::Quaternionf qf(static_cast<float>(quat.w()),
                        static_cast<float>(quat.x()),
                        static_cast<float>(quat.y()),
                        static_cast<float>(quat.z()));
  return qf.toRotationMatrix();
}

Eigen::VectorXf BehaviorFoundationPolicy::assembleBaseObservation(const Eigen::Vector3f& base_lin_vel,
                                                                  const Eigen::Vector3f& base_ang_vel,
                                                                  const Eigen::Vector3f& gravity,
                                                                  const Eigen::VectorXf& joints,
                                                                  const Eigen::VectorXf& joint_vel) const {
  const int dim = 3 + 3 + 3 + metadata_.action_dim * 3;
  Eigen::VectorXf obs = Eigen::VectorXf::Zero(dim);
  int offset = 0;
  obs.segment(offset, 3) = base_lin_vel;
  offset += 3;
  obs.segment(offset, 3) = base_ang_vel;
  offset += 3;
  obs.segment(offset, 3) = gravity;
  offset += 3;
  const int jointCopy = std::min(metadata_.action_dim, static_cast<int>(joints.size()));
  if (jointCopy > 0) {
    obs.segment(offset, jointCopy) = joints.head(jointCopy);
  }
  offset += metadata_.action_dim;
  const int velCopy = std::min(metadata_.action_dim, static_cast<int>(joint_vel.size()));
  if (velCopy > 0) {
    obs.segment(offset, velCopy) = joint_vel.head(velCopy);
  }
  offset += metadata_.action_dim;
  const int actionCopy = std::min(metadata_.action_dim, static_cast<int>(last_action_.size()));
  if (actionCopy > 0) {
    obs.segment(offset, actionCopy) = last_action_.head(actionCopy).cast<float>();
  }
  return obs;
}

}  // namespace legged::bfm

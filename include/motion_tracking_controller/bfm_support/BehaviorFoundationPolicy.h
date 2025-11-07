#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <legged_model/LeggedModel.h>
#include <legged_rl_controllers/CommandManager.h>
#include <legged_rl_controllers/Policy.h>

#include "motion_tracking_controller/bfm_support/BehaviorMetadata.h"
#include "motion_tracking_controller/bfm_support/HistoryBuffer.h"
#include "motion_tracking_controller/bfm_support/ObservationAssembler.h"

namespace legged::bfm {

class BehaviorFoundationPolicy : public Policy {
 public:
  using SharedPtr = std::shared_ptr<BehaviorFoundationPolicy>;
  using ComponentMap = ObservationAssembler::ComponentMap;

  explicit BehaviorFoundationPolicy(BfmPaths paths, std::string provider = "CPUExecutionProvider");

  size_t getObservationSize() const override { return static_cast<size_t>(metadata_.dim_obs); }
  size_t getActionSize() const override { return static_cast<size_t>(metadata_.action_dim); }

  void init() override;
  void reset() override;
  vector_t forward(const vector_t& observations) override;
  vector_t getLastAction() override { return last_action_; }

  const BehaviorMetadata& metadata() const { return metadata_; }

  void setLeggedModel(const LeggedModel::SharedPtr& model) { model_ = model; }
  void setCommandManager(const CommandManager::SharedPtr& manager) { command_manager_ = manager; }
  void setGraceOverride(int override_steps) { grace_override_ = override_steps; }
  void setDefaultForwardSpeed(double speed) { default_forward_speed_ = speed; }
  void setDebugDump(bool enabled) { debug_dump_ = enabled; debug_counter_ = 0; }

  vector_t prepareObservation();

  struct StepOut {
    Eigen::VectorXf base_action;
    Eigen::VectorXf residual_raw;
    Eigen::VectorXf residual_clipped;
    Eigen::VectorXf blended_action;
    Eigen::VectorXf residual_for_obs;
    Eigen::VectorXf residual_norm;
    Eigen::VectorXf mu_p;
  };

 private:
  struct GoalRange {
    int start{0};
    int end{0};
    int size() const { return std::max(0, end - start); }
    bool valid() const { return end > start; }
  };

  struct GoalLayout {
    GoalRange root_pos;
    GoalRange root_ori;
    GoalRange root_lin_vel;
    GoalRange root_ang_vel;
    GoalRange keypoints;
    GoalRange joints;
  };

  void initSessions();
  void initSessionIO(Ort::Session& session,
                     std::vector<std::string>& input_names,
                     std::vector<const char*>& input_ptrs,
                     std::vector<std::string>& output_names,
                     std::vector<const char*>& output_ptrs);

  Eigen::VectorXf runBase(const Eigen::VectorXf& sp_real, const Eigen::VectorXf& sg_masked, Eigen::VectorXf& mu_prior);
  Eigen::VectorXf runResidual(const Eigen::VectorXf& residual_obs);
  Eigen::VectorXf blendActions(const Eigen::VectorXf& base_action, const Eigen::VectorXf& residual) const;
  Eigen::VectorXf applyNormalization(const Eigen::VectorXf& observation) const;
  static Eigen::VectorXf tensorToVector(const Ort::Value& value);

  void cacheComponents(const Eigen::VectorXf& sp_real,
                       const Eigen::VectorXf& sg_masked,
                       const ComponentMap& components,
                       const Eigen::VectorXf& observation);
  StepOut step(const Eigen::VectorXf& sp_real,
               const Eigen::VectorXf& sg_masked,
               ComponentMap components,
               bool apply_grace_gate,
               int grace_remaining);
  void configureGoalLayout();
  void configureControlMask();
  Eigen::VectorXf buildGoalVector(const Eigen::Vector2f& dir_body, float speed, float yaw_rate) const;
  Eigen::Vector2f computeDirectionBody(const Eigen::Vector3d& command_world, const Eigen::Quaterniond& base_quat) const;
  Eigen::Matrix3f orientationMatrix() const;
  Eigen::VectorXf assembleBaseObservation(const Eigen::Vector3f& base_lin_vel,
                                          const Eigen::Vector3f& base_ang_vel,
                                          const Eigen::Vector3f& gravity,
                                          const Eigen::VectorXf& joints,
                                          const Eigen::VectorXf& joint_vel) const;

  BfmPaths paths_;
  std::string provider_;
  BehaviorMetadata metadata_;
  HistoryBuffer history_;
  std::unique_ptr<ObservationAssembler> assembler_;

  std::unique_ptr<Ort::Env> env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> base_session_;
  std::unique_ptr<Ort::Session> residual_session_;

  std::vector<std::string> base_input_names_;
  std::vector<const char*> base_input_ptrs_;
  std::vector<std::string> base_output_names_;
  std::vector<const char*> base_output_ptrs_;

  std::vector<std::string> residual_input_names_;
  std::vector<const char*> residual_input_ptrs_;
  std::vector<std::string> residual_output_names_;
  std::vector<const char*> residual_output_ptrs_;

  LeggedModel::SharedPtr model_;
  CommandManager::SharedPtr command_manager_;

  vector_t last_action_;
  Eigen::VectorXf last_residual_for_obs_;
  Eigen::VectorXf last_residual_norm_;
  Eigen::VectorXf cached_sp_real_;
  Eigen::VectorXf cached_sg_masked_;
  ComponentMap cached_components_;
  Eigen::VectorXf cached_observation_;
  bool cache_ready_{false};
  GoalLayout goal_layout_;
  bool goal_layout_valid_{false};
  Eigen::VectorXf control_mask_;
  int grace_override_{-1};
  int ticks_since_reset_{0};
  double default_forward_speed_{1.4};
  bool debug_dump_{false};
  int debug_counter_{0};

  int64_t time_step_counter_{0};
};

}  // namespace legged::bfm

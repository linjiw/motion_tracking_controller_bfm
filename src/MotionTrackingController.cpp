#include "motion_tracking_controller/MotionTrackingController.h"

#include "motion_tracking_controller/MotionCommand.h"
#include "motion_tracking_controller/MotionObservation.h"

namespace legged {
controller_interface::CallbackReturn MotionTrackingController::on_init() {
  if (RlController::on_init() != controller_interface::CallbackReturn::SUCCESS) {
    return controller_interface::CallbackReturn::ERROR;
  }

  try {
    auto_declare("motion.start_step", 0);
    auto_declare("policy.mode", std::string("legacy"));
    auto_declare("policy.bfm.residual_path", std::string(""));
    auto_declare("policy.bfm.base_path", std::string(""));
    auto_declare("policy.bfm.metadata_path", std::string(""));
    auto_declare("policy.bfm.obs_norm_path", std::string(""));
    auto_declare("policy.bfm.grace_override", -1);
    auto_declare("policy.bfm.default_forward_speed", 1.4);
    auto_declare("policy.bfm.debug_dump", false);
    auto_declare("command.topic", std::string("/cmd_vel"));
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Exception during init: %s", e.what());
    return CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn MotionTrackingController::on_configure(const rclcpp_lifecycle::State& previous_state) {
  const auto mode = get_node()->get_parameter("policy.mode").as_string();
  bfmMode_ = (mode == "bfm");
  policy_.reset();
  commandTopic_ = get_node()->get_parameter("command.topic").as_string();

  if (!bfmMode_) {
    const auto policyPath = get_node()->get_parameter("policy.path").as_string();
    const auto startStep = static_cast<size_t>(get_node()->get_parameter("motion.start_step").as_int());

    policy_ = std::make_shared<MotionOnnxPolicy>(policyPath, startStep);
    policy_->init();

    auto policy = std::dynamic_pointer_cast<MotionOnnxPolicy>(policy_);
    cfg_.anchorBody = policy->getAnchorBodyName();
    cfg_.bodyNames = policy->getBodyNames();
    RCLCPP_INFO_STREAM(rclcpp::get_logger("MotionTrackingController"),
                       "[legacy] Load Onnx model from " << policyPath << " successfully !");
  } else {
    bfm::BfmPaths paths;
    paths.residual = get_node()->get_parameter("policy.bfm.residual_path").as_string();
    paths.base = get_node()->get_parameter("policy.bfm.base_path").as_string();
    paths.metadata = get_node()->get_parameter("policy.bfm.metadata_path").as_string();
    paths.obs_norm = get_node()->get_parameter("policy.bfm.obs_norm_path").as_string();
    bfmGraceOverride_ = get_node()->get_parameter("policy.bfm.grace_override").as_int();
    const double default_speed = get_node()->get_parameter("policy.bfm.default_forward_speed").as_double();
    const bool debug_dump = get_node()->get_parameter("policy.bfm.debug_dump").as_bool();

    bfmPolicy_ = std::make_shared<bfm::BehaviorFoundationPolicy>(paths);
    try {
      bfmPolicy_->init();
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_node()->get_logger(), "Failed to initialize BehaviorFoundationPolicy: %s", e.what());
      return controller_interface::CallbackReturn::ERROR;
    }
    bfmPolicy_->setGraceOverride(bfmGraceOverride_);
    bfmPolicy_->setDefaultForwardSpeed(default_speed);
    bfmPolicy_->setDebugDump(debug_dump);
    policy_ = bfmPolicy_;
    RCLCPP_INFO(get_node()->get_logger(),
                "[bfm] residual='%s' base='%s' metadata='%s' obs_norm='%s'",
                paths.residual.c_str(),
                paths.base.c_str(),
                paths.metadata.c_str(),
                paths.obs_norm.c_str());
  }

  return RlController::on_configure(previous_state);
}

controller_interface::CallbackReturn MotionTrackingController::on_activate(const rclcpp_lifecycle::State& previous_state) {
  if (RlController::on_activate(previous_state) != controller_interface::CallbackReturn::SUCCESS) {
    return controller_interface::CallbackReturn::ERROR;
  }

  if (bfmMode_ && bfmPolicy_) {
    bfmPolicy_->setGraceOverride(bfmGraceOverride_);
    bfmPolicy_->setLeggedModel(leggedModel());
    bfmPolicy_->setCommandManager(commandManager_);
    bfmPolicy_->setDefaultForwardSpeed(get_node()->get_parameter("policy.bfm.default_forward_speed").as_double());
    bfmPolicy_->setDebugDump(get_node()->get_parameter("policy.bfm.debug_dump").as_bool());
    bfmPolicy_->reset();
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn MotionTrackingController::on_deactivate(const rclcpp_lifecycle::State& previous_state) {
  if (RlController::on_deactivate(previous_state) != controller_interface::CallbackReturn::SUCCESS) {
    return controller_interface::CallbackReturn::ERROR;
  }

  return controller_interface::CallbackReturn::SUCCESS;
}

bool MotionTrackingController::parserCommand(const std::string& name) {
  if (RlController::parserCommand(name)) {
    return true;
  }
  if (bfmMode_) {
    if ((name == "speed" || name == "velocity_topic") && commandManager_) {
      auto node = get_node();
      if (!node) {
        RCLCPP_ERROR(get_node()->get_logger(), "Lifecycle node unavailable while registering command '%s'", name.c_str());
        return false;
      }
      RCLCPP_INFO(get_node()->get_logger(),
                  "[bfm] Registering VelocityTopicCommandTerm '%s' on topic '%s'",
                  name.c_str(),
                  commandTopic_.c_str());
      try {
        commandManager_->addTerm(std::make_shared<VelocityTopicCommandTerm>(node, commandTopic_));
        commandManager_->reset();
        return true;
      } catch (const std::exception& e) {
        RCLCPP_ERROR(get_node()->get_logger(),
                     "Failed to add VelocityTopicCommandTerm for '%s': %s",
                     name.c_str(),
                     e.what());
        return false;
      }
    }
    return false;
  }
  if (name == "motion") {
    commandTerm_ = std::make_shared<MotionCommandTerm>(cfg_, std::dynamic_pointer_cast<MotionOnnxPolicy>(policy_));
    commandManager_->addTerm(commandTerm_);
    return true;
  }
  return false;
}

bool MotionTrackingController::parserObservation(const std::string& name) {
  if (RlController::parserObservation(name)) {
    return true;
  }
  if (bfmMode_) {
    if (name == "behavior_residual_obs") {
      observationManager_->addTerm(std::make_shared<BehaviorResidualObservationTerm>(bfmPolicy_));
      return true;
    }
    return false;
  }
  if (name == "motion_ref_pos_b" || name == "motion_anchor_pos_b") {
    observationManager_->addTerm(std::make_shared<MotionAnchorPosition>(commandTerm_));
  } else if (name == "motion_ref_ori_b" || name == "motion_anchor_ori_b") {
    observationManager_->addTerm(std::make_shared<MotionAnchorOrientation>(commandTerm_));
  } else if (name == "robot_body_pos") {
    observationManager_->addTerm(std::make_shared<RobotBodyPosition>(commandTerm_));
  } else if (name == "robot_body_ori") {
    observationManager_->addTerm(std::make_shared<RobotBodyOrientation>(commandTerm_));
  } else {
    return false;
  }
  return true;
}

}  // namespace legged

#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(legged::MotionTrackingController, controller_interface::ControllerInterface)

#pragma once

#include <legged_rl_controllers/CommandManager.h>
#include <legged_rl_controllers/RlController.h>

#include "motion_tracking_controller/bfm_support/BehaviorFoundationPolicy.h"
#include "motion_tracking_controller/MotionCommand.h"
#include "motion_tracking_controller/common.h"

namespace legged {
class MotionTrackingController : public RlController {
 public:
  controller_interface::CallbackReturn on_init() override;

  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

 protected:
  bool parserCommand(const std::string& name) override;
  bool parserObservation(const std::string& name) override;

  MotionCommandCfg cfg_;
  MotionCommandTerm::SharedPtr commandTerm_;
  bool bfmMode_{false};
  int bfmGraceOverride_{-1};
  bfm::BehaviorFoundationPolicy::SharedPtr bfmPolicy_;
  std::string commandTopic_{"/cmd_vel"};
};

}  // namespace legged

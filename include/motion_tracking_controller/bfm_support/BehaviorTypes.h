#pragma once

#include <Eigen/Core>

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace legged::bfm {

struct ComponentSlice {
  int start{0};
  int end{0};

  [[nodiscard]] int size() const { return end - start; }
  [[nodiscard]] bool valid() const { return end > start && start >= 0; }
};

struct HistorySpecs {
  int history_length{25};
  int dim_joint_pos{29};
  int dim_joint_vel{29};
  int dim_root_ang_vel{3};
  int dim_gravity{3};
  int dim_last_action{29};
  int dim_base_lin_vel{0};

  [[nodiscard]] int featureDim() const {
    return dim_base_lin_vel + dim_joint_pos + dim_joint_vel + dim_root_ang_vel + dim_gravity;
  }

  [[nodiscard]] int dimSpReal() const { return history_length * featureDim() + dim_last_action; }
};

struct ResidualConfig {
  std::string mode{"blend"};
  double alpha{0.1};
  double clip{1.0};
  int grace_steps{0};
};

struct SpeedCommandConfig {
  bool enable_yaw_command{false};
};

struct ObservationStats {
  Eigen::VectorXf mean;
  Eigen::VectorXf var;
  bool loaded{false};
};

struct BehaviorMetadata {
  int dim_obs{0};
  int dim_sp_real{0};
  int dim_goal{0};
  int action_dim{0};
  int residual_dim{0};
  int latent_dim{0};
  bool host_normalize_obs{false};

  double alpha{0.1};
  double residual_clip{1.0};
  std::string residual_mode{"blend"};
  int grace_steps{0};

  HistorySpecs history;
  SpeedCommandConfig speed_cfg;

  std::vector<std::string> observation_names;
  std::vector<std::string> command_names;
  std::map<std::string, ComponentSlice> component_slices;

  std::vector<float> obs_mean;
  std::vector<float> obs_var;
  std::vector<float> action_scale;
  std::vector<float> action_offset;

  [[nodiscard]] int widthOf(const std::string& name) const {
    auto it = component_slices.find(name);
    return it == component_slices.end() ? 0 : (it->second.end - it->second.start);
  }
};

struct BfmPaths {
  std::string residual;
  std::string base;
  std::string metadata;
  std::string obs_norm;
};

}  // namespace legged::bfm

#pragma once

#include <Eigen/Core>
#include "motion_tracking_controller/bfm_support/BehaviorTypes.h"

namespace legged::bfm {

class HistoryBuffer {
 public:
  HistoryBuffer() = default;
  explicit HistoryBuffer(const HistorySpecs& cfg);

  void configure(const HistorySpecs& cfg);
  void reset(const Eigen::VectorXf& features, const Eigen::VectorXf& last_action);
  void reset(const Eigen::VectorXf& features);
  Eigen::VectorXf push(const Eigen::VectorXf& features, const Eigen::VectorXf& last_action);

  [[nodiscard]] int stride() const { return cfg_.featureDim(); }
  [[nodiscard]] int realStateDim() const { return cfg_.dimSpReal(); }
  [[nodiscard]] bool ready() const { return initialized_; }

 private:
  void ensureStorage();

  HistorySpecs cfg_;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> feature_history_;
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> action_history_;
  bool initialized_{false};
  int cursor_{0};
};

}  // namespace legged::bfm

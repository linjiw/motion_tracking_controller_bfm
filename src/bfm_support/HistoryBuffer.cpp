#include "motion_tracking_controller/bfm_support/HistoryBuffer.h"

#include <stdexcept>

namespace legged::bfm {

HistoryBuffer::HistoryBuffer(const HistorySpecs& cfg) { configure(cfg); }

void HistoryBuffer::configure(const HistorySpecs& cfg) {
  cfg_ = cfg;
  initialized_ = false;
  cursor_ = 0;
  ensureStorage();
}

void HistoryBuffer::reset(const Eigen::VectorXf& features, const Eigen::VectorXf& last_action) {
  if (features.size() != stride()) {
    throw std::runtime_error("HistoryBuffer::reset: feature dimension mismatch.");
  }
  if (cfg_.dim_last_action > 0 && last_action.size() != cfg_.dim_last_action) {
    throw std::runtime_error("HistoryBuffer::reset: last action dimension mismatch.");
  }

  ensureStorage();
  for (int i = 0; i < cfg_.history_length; ++i) {
    feature_history_.row(i) = features;
    if (cfg_.dim_last_action > 0) {
      action_history_.row(i) = last_action;
    }
  }
  initialized_ = true;
  cursor_ = 0;
}

void HistoryBuffer::reset(const Eigen::VectorXf& features) {
  Eigen::VectorXf dummy = Eigen::VectorXf::Zero(cfg_.dim_last_action);
  reset(features, dummy);
}

Eigen::VectorXf HistoryBuffer::push(const Eigen::VectorXf& features, const Eigen::VectorXf& last_action) {
  if (!initialized_) {
    reset(features, last_action);
  } else {
    feature_history_.row(cursor_) = features;
    if (cfg_.dim_last_action > 0) {
      action_history_.row(cursor_) = last_action;
    }
    cursor_ = (cursor_ + 1) % cfg_.history_length;
  }

  Eigen::VectorXf flattened(realStateDim());
  int offset = 0;
  for (int i = 0; i < cfg_.history_length; ++i) {
    const int idx = (cursor_ + i) % cfg_.history_length;
    flattened.segment(offset, stride()) = feature_history_.row(idx);
    offset += stride();
  }
  if (cfg_.dim_last_action > 0) {
    flattened.segment(offset, cfg_.dim_last_action) = action_history_.row((cursor_ + cfg_.history_length - 1) % cfg_.history_length);
  }
  return flattened;
}

void HistoryBuffer::ensureStorage() {
  const int rows = std::max(cfg_.history_length, 1);
  const int cols_features = std::max(stride(), 1);
  feature_history_.resize(rows, cols_features);
  feature_history_.setZero();

  if (cfg_.dim_last_action > 0) {
    action_history_.resize(rows, cfg_.dim_last_action);
    action_history_.setZero();
  } else {
    action_history_.resize(0, 0);
  }
}

}  // namespace legged::bfm

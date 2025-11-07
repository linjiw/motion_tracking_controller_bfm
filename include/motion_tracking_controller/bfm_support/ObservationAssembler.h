#pragma once

#include <Eigen/Core>

#include <map>
#include <string>
#include <vector>

#include "motion_tracking_controller/bfm_support/BehaviorTypes.h"

namespace legged::bfm {

class ObservationAssembler {
 public:
  using ComponentMap = std::map<std::string, Eigen::VectorXf>;

  explicit ObservationAssembler(BehaviorMetadata metadata);

  [[nodiscard]] size_t observationSize() const { return static_cast<size_t>(metadata_.dim_obs); }
  [[nodiscard]] const BehaviorMetadata& metadata() const { return metadata_; }

  Eigen::VectorXf assemble(const ComponentMap& components) const;

 private:
  BehaviorMetadata metadata_;
  std::vector<std::pair<std::string, ComponentSlice>> ordered_slices_;
};

}  // namespace legged::bfm

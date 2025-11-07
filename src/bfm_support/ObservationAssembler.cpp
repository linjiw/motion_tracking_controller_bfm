#include "motion_tracking_controller/bfm_support/ObservationAssembler.h"

#include <algorithm>
#include <stdexcept>

namespace legged::bfm {

ObservationAssembler::ObservationAssembler(BehaviorMetadata metadata) : metadata_(std::move(metadata)) {
  ordered_slices_.reserve(metadata_.component_slices.size());
  for (const auto& entry : metadata_.component_slices) {
    ordered_slices_.emplace_back(entry.first, entry.second);
  }
  std::sort(ordered_slices_.begin(), ordered_slices_.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.second.start < rhs.second.start; });
}

Eigen::VectorXf ObservationAssembler::assemble(const ComponentMap& components) const {
  Eigen::VectorXf result = Eigen::VectorXf::Zero(metadata_.dim_obs);

  for (const auto& [name, slice] : ordered_slices_) {
    auto it = components.find(name);
    if (it == components.end()) {
      throw std::runtime_error("ObservationAssembler: missing component \"" + name + "\"");
    }
    const Eigen::VectorXf& component = it->second;
    if (component.size() != slice.size()) {
      throw std::runtime_error("ObservationAssembler: component \"" + name + "\" expected size " +
                               std::to_string(slice.size()) + " but got " + std::to_string(component.size()));
    }
    result.segment(slice.start, slice.size()) = component;
  }

  return result;
}

}  // namespace legged::bfm

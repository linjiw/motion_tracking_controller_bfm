#pragma once

#include <string>

#include "motion_tracking_controller/bfm_support/BehaviorTypes.h"

namespace legged::bfm {

BehaviorMetadata loadMetadata(const std::string& path);
void loadObservationStats(const std::string& npz_path, BehaviorMetadata& metadata);

}  // namespace legged::bfm

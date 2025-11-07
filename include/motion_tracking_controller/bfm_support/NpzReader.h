#pragma once

#include <string>
#include <utility>
#include <vector>

namespace legged::bfm {

struct NpzArray {
  std::vector<float> data;
  size_t count{0};
};

// Throws std::runtime_error on failure.
NpzArray readArray(const std::string& npz_path, const std::string& array_name);

}  // namespace legged::bfm

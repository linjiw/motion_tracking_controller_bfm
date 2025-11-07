#include "motion_tracking_controller/bfm_support/BehaviorMetadata.h"

#include <jsoncpp/json/json.h>

#include <fstream>
#include <stdexcept>

#include "motion_tracking_controller/bfm_support/NpzReader.h"

namespace legged::bfm {
namespace {

void parseHistorySpecs(const Json::Value& root, HistorySpecs& history) {
  if (!root.isObject()) {
    return;
  }
  history.history_length = root.get("history_length", history.history_length).asInt();
  history.dim_joint_pos = root.get("dim_joint_pos", history.dim_joint_pos).asInt();
  history.dim_joint_vel = root.get("dim_joint_vel", history.dim_joint_vel).asInt();
  history.dim_root_ang_vel = root.get("dim_root_ang_vel", history.dim_root_ang_vel).asInt();
  history.dim_gravity = root.get("dim_gravity", history.dim_gravity).asInt();
  history.dim_last_action = root.get("dim_last_action", history.dim_last_action).asInt();
  history.dim_base_lin_vel = root.get("dim_base_lin_vel", history.dim_base_lin_vel).asInt();
}

void parseCommandObservationLists(const Json::Value& root, BehaviorMetadata& metadata) {
  metadata.command_names.clear();
  if (root.isMember("command_names")) {
    for (const auto& entry : root["command_names"]) {
      metadata.command_names.emplace_back(entry.asString());
    }
  }
  metadata.observation_names.clear();
  if (root.isMember("observation_names")) {
    for (const auto& entry : root["observation_names"]) {
      metadata.observation_names.emplace_back(entry.asString());
    }
  }
}

void parseComponentSlices(const Json::Value& root, BehaviorMetadata& metadata) {
  metadata.component_slices.clear();
  if (!root.isObject()) {
    return;
  }
  for (const auto& name : root.getMemberNames()) {
    const auto& slice = root[name];
    if (!slice.isArray() || slice.size() != 2) {
      continue;
    }
    ComponentSlice comp;
    comp.start = slice[0].asInt();
    comp.end = slice[1].asInt();
    if (!comp.valid()) {
      throw std::runtime_error("Invalid component slice for " + name);
    }
    metadata.component_slices.emplace(name, comp);
  }
}

void parseActionAdapters(const Json::Value& root, BehaviorMetadata& metadata) {
  metadata.action_scale.clear();
  metadata.action_offset.clear();
  if (root.isMember("action_scale")) {
    for (const auto& entry : root["action_scale"]) {
      metadata.action_scale.push_back(entry.asFloat());
    }
  }
  if (root.isMember("action_offset")) {
    for (const auto& entry : root["action_offset"]) {
      metadata.action_offset.push_back(entry.asFloat());
    }
  }
}

}  // namespace

BehaviorMetadata loadMetadata(const std::string& path) {
  if (path.empty()) {
    throw std::invalid_argument("Behavior metadata path is empty.");
  }

  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("Failed to open metadata file: " + path);
  }

  Json::Value root;
  input >> root;

  BehaviorMetadata metadata;
  metadata.dim_obs = root.get("dim_obs", 0).asInt();
  metadata.dim_sp_real = root.get("dim_sp_real", 0).asInt();
  metadata.dim_goal = root.get("dim_goal_unified", 0).asInt();
  metadata.action_dim = root.get("action_dim", 0).asInt();
  metadata.residual_dim = root.get("residual_dim", metadata.action_dim).asInt();
  metadata.latent_dim = root.get("latent_dim", 0).asInt();
  metadata.host_normalize_obs = root.get("host_normalize_obs", false).asBool();

  metadata.alpha = root.get("alpha", metadata.alpha).asDouble();
  metadata.residual_clip = root.get("residual_clip", metadata.residual_clip).asDouble();
  metadata.residual_mode = root.get("residual_mode", metadata.residual_mode).asString();
  metadata.grace_steps = root.get("grace_steps", metadata.grace_steps).asInt();

  parseHistorySpecs(root["history"], metadata.history);
  parseHistorySpecs(root["state_specs"], metadata.history);

  parseCommandObservationLists(root, metadata);
  parseComponentSlices(root["component_slices"], metadata);
  parseActionAdapters(root, metadata);

  if (root.isMember("speed_command")) {
    metadata.speed_cfg.enable_yaw_command =
        root["speed_command"].get("enable_yaw_command", metadata.speed_cfg.enable_yaw_command).asBool();
  }

  if (metadata.dim_sp_real == 0) {
    metadata.dim_sp_real = metadata.history.dimSpReal();
  }

  if (metadata.dim_obs <= 0) {
    throw std::runtime_error("Metadata missing dim_obs.");
  }
  if (metadata.component_slices.empty()) {
    throw std::runtime_error("Metadata missing component_slices.");
  }
  if (metadata.history.history_length <= 0) {
    throw std::runtime_error("Metadata history_length must be positive.");
  }

  return metadata;
}

void loadObservationStats(const std::string& npz_path, BehaviorMetadata& metadata) {
  if (npz_path.empty()) {
    return;
  }
  const auto mean = readArray(npz_path, "mean");
  const auto var = readArray(npz_path, "var");

  if (static_cast<int>(mean.count) != metadata.dim_obs || static_cast<int>(var.count) != metadata.dim_obs) {
    throw std::runtime_error("Observation stats do not match dim_obs.");
  }
  metadata.obs_mean = mean.data;
  metadata.obs_var = var.data;
}

}  // namespace legged::bfm

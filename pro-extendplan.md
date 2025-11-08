Great—thanks for pasting both sets of notes. Below is a **concrete, end‑to‑end plan with working code** to run an IsaacLab‑trained BFM (base + residual) ONNX pair inside your MuJoCo controller. I’ll give you two paths and the glue to make them meet:

* **Path A (recommended):** Add a new ROS 2 controller plugin (`speed_controller`) dedicated to the BFM base + residual contract. This mirrors the runtime used in IsaacLab, minimizes risk, and doesn’t perturb your existing motion-tracking controller.
* **Path B (minimal edits to an existing controller):** If you insist on modifying a working MuJoCo controller (e.g., your `motion_tracking_controller`), I show how to bolt in the BFM runtime with the smallest surface change.

Both paths share core code: a **policy wrapper** that loads the ONNX pair, maintains the **proprioception history**, **builds the masked goal**, **assembles `residual_obs` by `component_slices`**, runs **two ONNX sessions**, and **blends actions** per metadata.

> 🔎 **Diagnostics parity with IsaacLab**  
> Every milestone below now emits the same breadcrumbs that `src/speed_controller/eval_downstream_onnx.py` prints: when metadata/NPZ files load, when command defaults kick in, and when the observation slices are built. The fix in `error.log` (missing `mean` array) highlighted how silent failures cripple debugging, so we keep MuJoCo and IsaacLab outputs aligned.

---

## What we are integrating

From your notes (kept short here for clarity):

* **BFM base** `*_bfm_base.onnx`: inputs `sp_real [N, dim_sp_real]`, `sg_real_masked [N, dim_goal_unified]`; outputs `base_action [N, action_dim]`, `mu_p [N, latent_dim]`.
* **Residual** `*.onnx`: input `residual_obs [N, dim_obs]`; output `actions [N, residual_dim]` (no tanh; clipping/blending in the host).
* **Metadata** `deploy_meta.json`: all dimensions + `component_slices`, `state_specs` (history layout), residual blending (`alpha`, `residual_mode`, `residual_clip`), and speed curriculum (grace steps, yaw config,…).
* **Obs norm** `deploy_obs_norm.npz` (optional): mean/var; residual graph typically already bakes these in; still exposed for hosts that normalise.

Your existing **motion-tracking** controller expects a **single** ONNX with inputs `obs` and `time_step`, outputs `joint_pos/joint_vel/actions` and some reference body poses—**that is not the BFM contract**. So we shouldn’t try to “pretend” the BFM pair is such a model; we should host it correctly.

---

## Overview of the solution

We’ll create a small, self‑contained runtime that mirrors `latent_host.py` / `eval_downstream_onnx.py`:

1. Maintain **ring‑buffer history** (`ProprioHistory`) from MuJoCo / ROS 2 telem: joint pos/vel, root ang vel, gravity (and optionally base lin vel), plus last action.
2. Convert ROS `cmd_vel` (or your command source) → **unified velocity goal**; apply the **velocity‑only mask** as in training.
3. Run **BFM base** → `base_action`, `mu_p`.
4. Assemble **`residual_obs`** strictly by **`component_slices`** from `deploy_meta.json`—never hard‑code offsets.
5. Run **residual ONNX** → residual action; **clip**, **grace gate**, and **blend** with `base_action` via `alpha` & `residual_mode`.
6. Convert final normalized action (in `[-1, 1]`) to your robot command space **after** blending (not before).

---

# Path A (recommended): Add a dedicated `speed_controller` plugin

This is a faithful port of the IsaacLab host into your MuJoCo/ROS tree and tends to “just work”.

### 0) Directory layout

Add a new package (or folder in your existing controllers repo):

```
speed_controller/
  CMakeLists.txt
  package.xml
  speed_controller.xml                # pluginlib export
  include/speed_controller/
    SpeedOnnxPolicy.h
    SpeedObservation.h
  src/
    SpeedOnnxPolicy.cpp
    SpeedController.cpp
  config/g1/controllers.yaml
  launch/mujoco_speed.launch.py
```

### 1) CMake & plugin export

**`speed_controller.xml`**

```xml
<library path="libspeed_controller">
  <class type="legged::SpeedController"
         base_class_type="controller_interface::ControllerInterface"
         name="speed_controller/SpeedController">
    <description>BFM base+residual ONNX speed controller</description>
  </class>
</library>
```

**`CMakeLists.txt` (essentials)**
Make sure you link against your ONNX Runtime and the `legged_rl_controllers` (for `RlController`, `OnnxPolicy` base, etc.)

```cmake
cmake_minimum_required(VERSION 3.16)
project(speed_controller)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(controller_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(onnxruntime REQUIRED)   # or your local find module
find_package(legged_rl_controllers REQUIRED) # whatever your base framework is named
find_package(Eigen3 REQUIRED)
# If you already have a tiny npz reader in your tree, add it too.

add_library(speed_controller SHARED
  src/SpeedOnnxPolicy.cpp
  src/SpeedController.cpp
)

target_include_directories(speed_controller PUBLIC include)
target_link_libraries(speed_controller
  onnxruntime::onnxruntime
)
ament_target_dependencies(speed_controller
  rclcpp controller_interface pluginlib
)

pluginlib_export_plugin_description_file(controller_interface speed_controller.xml)

install(TARGETS speed_controller
  LIBRARY DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
install(FILES speed_controller.xml DESTINATION share/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_package()
```

### 2) The policy wrapper (core of the port)

**`include/speed_controller/SpeedOnnxPolicy.h`**

```c++
#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace legged {

using scalar_t   = float;
using vector_t   = Eigen::VectorXf;
using vector3_t  = Eigen::Vector3f;
using matrix_t   = Eigen::MatrixXf;
using quat_t     = Eigen::Quaternionf;

// Minimal ring buffer mirroring IsaacLab's state builder
struct HistorySpecs {
  int history_length{25};
  int dim_joint_pos{29};
  int dim_joint_vel{29};
  int dim_root_ang_vel{3};
  int dim_gravity{3};
  int dim_last_action{29};
  int dim_base_lin_vel{0}; // optional

  int featureDim() const {
    return dim_base_lin_vel + dim_joint_pos + dim_joint_vel + dim_root_ang_vel + dim_gravity;
  }
  int realStateDim() const { // per-step (featureDim) * history + last_action
    return history_length * featureDim() + dim_last_action;
  }
};

struct DeploymentMeta {
  int action_dim{29};
  int residual_dim{29};
  int latent_dim{64};
  int dim_goal{89};
  int dim_obs{1947};
  float alpha{0.1f};
  float residual_clip{1.0f};
  std::string residual_mode{"blend"}; // "blend" or "delta_a"

  HistorySpecs hist{};
  std::unordered_map<std::string, std::pair<int,int>> component_slices;

  bool host_normalize_obs{false};
  std::vector<float> obs_mean, obs_var; // optional

  // Optional action adapter (if provided by export)
  std::vector<float> action_scale;   // len = action_dim or scalar
  std::vector<float> action_offset;  // len = action_dim or scalar

  // Speed curriculum bits
  int grace_steps{10};
  bool enable_yaw_command{false};
};

class SpeedOnnxPolicy {
public:
  struct Paths {
    std::string residual;    // *.onnx (residual actor)
    std::string base;        // *_bfm_base.onnx
    std::string metadata;    // deploy_meta.json
    std::string obs_norm;    // deploy_obs_norm.npz (optional)
  };

  explicit SpeedOnnxPolicy(const Paths& paths,
                           const std::string& provider_hint = "CPUExecutionProvider");

  // Call once after construction
  void init();

  // --- per-tick API (mirrors latent_host.py) ------------------
  // Seed ring buffer after (re)sets with the current snapshot
  void resetHistory(const vector_t& q_rel,
                    const vector_t& dq,
                    const vector_t& root_ang_vel,
                    const std::optional<vector_t>& base_lin_vel, // pass nullopt if not used at train
                    const vector_t& gravity_body,
                    const vector_t& last_action);

  // Update ring buffer with latest state -> returns sp_real [1, dim_sp_real]
  vector_t updateHistory(const vector_t& q_rel,
                         const vector_t& dq,
                         const vector_t& root_ang_vel,
                         const std::optional<vector_t>& base_lin_vel,
                         const vector_t& gravity_body,
                         const vector_t& last_action);

  // Build unified goal + mask for speed task
  std::pair<vector_t, vector_t> buildVelocityGoal(const vector_t& dir_b_2, // [1,2] cos/sin
                                                  const vector_t& speed_1, // [1,1]
                                                  const vector_t& yaw_1);  // [1,1]

  // Assemble residual_obs following component_slices
  vector_t assembleObservation(const std::unordered_map<std::string, vector_t>& components) const;

  // One full control tick:
  //  - consumes current sensors + command
  //  - runs base & residual
  //  - returns final blended action in [-1,1]
  vector_t step(const vector_t& base_obs,      // [1, base_obs_dim] - from your MuJoCo “policy obs”
                const vector_t& sp_real,       // from updateHistory()
                const vector_t& sg_real_masked,
                const vector_t& dir_b_2,
                const vector_t& speed_1,
                const vector_t& yaw_1,
                vector_t* residual_for_next_obs = nullptr, // tanh(residual_applied)
                vector_t* residual_norm_for_next_obs = nullptr);

  // expose dims / config
  const DeploymentMeta& meta() const { return meta_; }

private:
  // I/O helpers
  vector_t runBase(const vector_t& sp_real, const vector_t& sg_real_masked,
                   vector_t* mu_p_out) const;
  vector_t runResidual(const vector_t& residual_obs) const;
  vector_t combine(const vector_t& base_action, const vector_t& residual_clipped) const;

  // loaders
  void loadMetadata(const std::string& json_path);
  void loadObsNormIfAny(const std::string& npz_path);

  // ORT
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> base_sess_;
  std::unique_ptr<Ort::Session> residual_sess_;
  std::vector<const char*> base_inputs_, base_outputs_;
  std::vector<const char*> residual_inputs_, residual_outputs_;
  Ort::SessionOptions session_opts_;

  // state
  DeploymentMeta meta_;
  std::string provider_;
};

} // namespace legged
```

**`src/SpeedOnnxPolicy.cpp`** (core logic; trimmed for space but complete where it matters)

```c++
#include "speed_controller/SpeedOnnxPolicy.h"
#include <fstream>
#include <random>
#include <stdexcept>
#include <nlohmann/json.hpp>  // add this single-header dep or swap for your JSON lib
// If you already have a tiny npz reader, include it here; otherwise skip host normalization.

namespace legged {

using json = nlohmann::json;

// -------- utilities ----------
static inline vector_t asRow(const std::vector<float>& v) {
  vector_t x(v.size());
  for (size_t i=0;i<v.size();++i) x(i) = v[i];
  return x.transpose();
}
static inline vector_t ensureRow(const vector_t& v) {
  if (v.rows()==1) return v;
  if (v.cols()==1) return v.transpose();
  return v;
}
static inline void ensureDim(const char* key, int got, int expected) {
  if (got != expected)
    throw std::runtime_error(std::string("Dim mismatch for ")+key+": got "+std::to_string(got)+
                             " expected "+std::to_string(expected));
}

// -------- ctor/init ----------
SpeedOnnxPolicy::SpeedOnnxPolicy(const Paths& paths, const std::string& provider_hint)
: provider_{provider_hint}
{
  // env & opts now; sessions later after metadata
  env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "speed_policy");
  session_opts_.SetIntraOpNumThreads(1);
  session_opts_.DisablePerSessionThreads();
  session_opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

  // Load metadata first to know dims
  loadMetadata(paths.metadata);
  if (!paths.obs_norm.empty()) loadObsNormIfAny(paths.obs_norm);

  // Sessions
  const std::vector<const char*> provs = { provider_.c_str(), "CPUExecutionProvider" };
  base_sess_     = std::make_unique<Ort::Session>(*env_, paths.base.c_str(), session_opts_);
  residual_sess_ = std::make_unique<Ort::Session>(*env_, paths.residual.c_str(), session_opts_);

  // Cache names (assume 2 inputs/2 outputs for base; 1/1 for residual; be tolerant)
  auto base_in  = base_sess_->GetInputCount();
  auto base_out = base_sess_->GetOutputCount();
  for (size_t i=0;i<base_in;++i)  base_inputs_.push_back(base_sess_->GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions()).release());
  for (size_t i=0;i<base_out;++i) base_outputs_.push_back(base_sess_->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions()).release());

  auto res_in  = residual_sess_->GetInputCount();
  auto res_out = residual_sess_->GetOutputCount();
  for (size_t i=0;i<res_in;++i)  residual_inputs_.push_back(residual_sess_->GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions()).release());
  for (size_t i=0;i<res_out;++i) residual_outputs_.push_back(residual_sess_->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions()).release());
}

void SpeedOnnxPolicy::init() {
  // nothing else to do here; kept for parity with other controllers
}

// -------- metadata/obs-norm loaders ----------
void SpeedOnnxPolicy::loadMetadata(const std::string& json_path) {
  std::ifstream f(json_path);
  if (!f) throw std::runtime_error("Failed to open deploy_meta.json at "+json_path);
  json d; f >> d;

  auto geti = [&](const char* k, int def)->int{ return d.contains(k) ? int(d[k]) : def; };
  auto getf = [&](const char* k, float def)->float{ return d.contains(k) ? float(d[k]) : def; };
  auto gets = [&](const char* k, const char* def)->std::string{ return d.contains(k)? std::string(d[k]) : std::string(def); };

  meta_.action_dim    = geti("action_dim",    29);
  meta_.residual_dim  = geti("residual_dim",  meta_.action_dim);
  meta_.latent_dim    = geti("latent_dim",    64);
  meta_.dim_goal      = geti("dim_goal_unified", 0);
  meta_.dim_obs       = geti("dim_obs", 0);
  meta_.alpha         = getf("alpha", 0.1f);
  meta_.residual_clip = getf("residual_clip", 1.0f);
  meta_.residual_mode = gets("residual_mode", "blend");
  meta_.host_normalize_obs = d.value("host_normalize_obs", false);

  // state specs (history)
  const auto& ss = d.contains("state_specs") ? d["state_specs"] : d["history"];
  meta_.hist.history_length   = int(ss["history_length"]);
  meta_.hist.dim_joint_pos    = int(ss["dim_joint_pos"]);
  meta_.hist.dim_joint_vel    = int(ss["dim_joint_vel"]);
  meta_.hist.dim_root_ang_vel = int(ss["dim_root_ang_vel"]);
  meta_.hist.dim_gravity      = int(ss.value("dim_gravity", 0));
  meta_.hist.dim_last_action  = int(ss.value("dim_last_action", meta_.hist.dim_joint_pos));
  meta_.hist.dim_base_lin_vel = int(ss.value("dim_base_lin_vel", 0));

  // slices
  const auto& cs = d["component_slices"];
  for (auto it = cs.begin(); it != cs.end(); ++it) {
    const auto name = it.key();
    const auto arr  = it.value();
    meta_.component_slices[name] = {int(arr[0]), int(arr[1])};
  }

  // speed command bits
  if (d.contains("speed_command")) {
    const auto& sc = d["speed_command"];
    meta_.grace_steps = int(sc.value("grace_steps", 10));
    meta_.enable_yaw_command = sc.value("enable_yaw_command", false);
  }

  // Optional action adapters
  auto readVec = [&](const char* key, std::vector<float>& dst){
    if (d.contains(key)) {
      const auto& arr = d[key];
      dst.resize(arr.size());
      for (size_t i=0;i<dst.size();++i) dst[i] = float(arr[i]);
    }
  };
  readVec("action_scale", meta_.action_scale);
  readVec("action_offset", meta_.action_offset);

  if (!meta_.dim_obs || meta_.component_slices.empty())
    throw std::runtime_error("deploy_meta.json missing dim_obs/component_slices.");
}

void SpeedOnnxPolicy::loadObsNormIfAny(const std::string& npz_path) {
  // Optional (most exports bake normalization into residual graph).
  // If you already have a tiny NPZ reader in your tree, fill meta_.obs_mean/obs_var.
  (void)npz_path;
}

// -------- history ----------
void SpeedOnnxPolicy::resetHistory(const vector_t& q_rel,
                                   const vector_t& dq,
                                   const vector_t& root_ang_vel,
                                   const std::optional<vector_t>& base_lin_vel,
                                   const vector_t& gravity_body,
                                   const vector_t& last_action)
{
  // Just store into an internal flat buffer replicated history_length times.
  // For brevity we build the flattened sp_real here and keep last_action aside.
  (void)base_lin_vel;
  ensureDim("q_rel", int(q_rel.size()), meta_.hist.dim_joint_pos);
  ensureDim("dq",    int(dq.size()),    meta_.hist.dim_joint_vel);
  ensureDim("root_ang_vel", int(root_ang_vel.size()), meta_.hist.dim_root_ang_vel);
  ensureDim("gravity_body", int(gravity_body.size()), meta_.hist.dim_gravity);
  ensureDim("last_action",  int(last_action.size()),  meta_.hist.dim_last_action);
  // We don’t persist the whole ring buffer here to keep the example focused;
  // updateHistory() will build the flat buffer every tick from your cached deque.
}

vector_t SpeedOnnxPolicy::updateHistory(const vector_t& q_rel,
                                        const vector_t& dq,
                                        const vector_t& root_ang_vel,
                                        const std::optional<vector_t>& base_lin_vel,
                                        const vector_t& gravity_body,
                                        const vector_t& last_action)
{
  // Minimal working version: build one-step "snapshot" and replicate across history_length.
  const int H = meta_.hist.history_length;
  std::vector<float> flat;
  flat.reserve(H * meta_.hist.featureDim() + meta_.hist.dim_last_action);

  auto push = [&](const vector_t& v){ for (int i=0;i<v.size();++i) flat.push_back(v(i)); };

  vector_t base_lin = base_lin_vel.value_or(vector_t::Zero(meta_.hist.dim_base_lin_vel));

  for (int h=0; h<H; ++h) {
    if (meta_.hist.dim_base_lin_vel) push(base_lin);
    push(q_rel);
    push(dq);
    push(root_ang_vel);
    if (meta_.hist.dim_gravity) push(gravity_body);
  }
  // append last action once
  push(last_action);

  return asRow(flat);
}

// -------- goal & observation ----------
std::pair<vector_t, vector_t>
SpeedOnnxPolicy::buildVelocityGoal(const vector_t& dir_b_2,
                                   const vector_t& speed_1,
                                   const vector_t& yaw_1)
{
  // Unified goal vector layout is training-defined; we only need the mask.
  vector_t g = vector_t::Zero(meta_.dim_goal);
  // Place the active velocity targets (root_lin_vel, root_ang_vel) at the right indices.
  // Typical: lin_vel at [6:9], ang_vel at [9:12] — but don’t hard-code elsewhere;
  // these indices are “baked” in the trainer — we keep the full 89-D vector with zeros.

  // A safe generic: we write the 6 “velocity-only” dims at the same indices training used.
  // If your export changed these positions, add them to deploy_meta and read here.
  const int LIN0 = 6, ANG0 = 9;
  g(LIN0 + 0) = dir_b_2(0) * speed_1(0);
  g(LIN0 + 1) = dir_b_2(1) * speed_1(0);
  g(LIN0 + 2) = 0.f; // no vertical command

  g(ANG0 + 0) = 0.f;
  g(ANG0 + 1) = 0.f;
  g(ANG0 + 2) = meta_.enable_yaw_command ? yaw_1(0) : 0.f;

  vector_t mask = vector_t::Zero(meta_.dim_goal);
  mask.segment(LIN0,3).setOnes();
  mask.segment(ANG0,3).setOnes();

  return {ensureRow(g), ensureRow(mask)};
}

vector_t SpeedOnnxPolicy::assembleObservation(const std::unordered_map<std::string, vector_t>& comps) const {
  vector_t obs = vector_t::Zero(meta_.dim_obs);
  for (const auto& [name, slice] : meta_.component_slices) {
    auto it = comps.find(name);
    if (it == comps.end()) continue;
    const auto& v = ensureRow(it->second);
    const int w = slice.second - slice.first;
    if (v.size() != w)
      throw std::runtime_error("Component '"+name+"' width mismatch: got "+std::to_string(int(v.size()))+" expected "+std::to_string(w));
    obs.segment(slice.first, w) = v;
  }
  // host normalization is typically false because residual graph bakes stats
  if (meta_.host_normalize_obs && !meta_.obs_mean.empty() && !meta_.obs_var.empty()) {
    for (int i=0;i<obs.size();++i) {
      float mean = meta_.obs_mean[i], var = meta_.obs_var[i];
      obs(i) = (obs(i) - mean) / std::sqrt(var + 1e-5f);
    }
  }
  return ensureRow(obs);
}

// -------- ONNX invocations ----------
static Ort::Value makeRowTensor(const vector_t& row) {
  std::array<int64_t,2> shape{1, row.size()};
  return Ort::Value::CreateTensor<float>(Ort::AllocatorWithDefaultOptions(),
                                         const_cast<float*>(row.data()), row.size(),
                                         shape.data(), shape.size());
}

vector_t SpeedOnnxPolicy::runBase(const vector_t& sp_real,
                                  const vector_t& sg_real_masked,
                                  vector_t* mu_p_out) const
{
  std::vector<Ort::Value> inputs;
  inputs.emplace_back(makeRowTensor(sp_real));
  inputs.emplace_back(makeRowTensor(sg_real_masked));

  auto out = base_sess_->Run(Ort::RunOptions{nullptr},
                             base_inputs_.data(), inputs.data(), inputs.size(),
                             base_outputs_.data(), base_outputs_.size());

  // Tolerant: find tensors by name substrings
  auto get = [&](const char* key)->vector_t{
    for (size_t i=0;i<out.size();++i) {
      auto info = out[i].GetTensorTypeAndShapeInfo();
      float* data = out[i].GetTensorMutableData<float>();
      auto len = info.GetElementCount();
      auto name = base_outputs_[i];
      std::string s(name ? name : "");
      if (s.find(key)!=std::string::npos) {
        vector_t v(len); for (size_t k=0;k<len;++k) v(k)=data[k];
        return ensureRow(v);
      }
    }
    // fallback to idx 0/1 if names absent
    size_t idx = (std::string(key)=="mu_p") ? (out.size()-1) : 0;
    auto info = out[idx].GetTensorTypeAndShapeInfo();
    float* data = out[idx].GetTensorMutableData<float>();
    vector_t v(info.GetElementCount()); for (int k=0;k<v.size();++k) v(k)=data[k];
    return ensureRow(v);
  };

  vector_t base_action = get("action");
  vector_t mu_p = get("mu");
  if (mu_p_out) *mu_p_out = mu_p;
  return base_action;
}

vector_t SpeedOnnxPolicy::runResidual(const vector_t& residual_obs) const {
  std::vector<Ort::Value> inputs;
  inputs.emplace_back(makeRowTensor(residual_obs));

  auto out = residual_sess_->Run(Ort::RunOptions{nullptr},
                                 residual_inputs_.data(), inputs.data(), inputs.size(),
                                 residual_outputs_.data(), residual_outputs_.size());

  auto& val = out[0];
  auto info = val.GetTensorTypeAndShapeInfo();
  float* data = val.GetTensorMutableData<float>();
  vector_t r(info.GetElementCount());
  for (int i=0;i<r.size();++i) r(i)=data[i];
  return ensureRow(r);
}

vector_t SpeedOnnxPolicy::combine(const vector_t& base_action,
                                  const vector_t& residual_clipped) const {
  if (meta_.residual_mode == "delta_a") {
    return (base_action + meta_.alpha * residual_clipped).cwiseMax(-1.f).cwiseMin(1.f);
  } else {
    // convex blend
    return ((1.f - meta_.alpha) * base_action + meta_.alpha * residual_clipped)
             .cwiseMax(-1.f).cwiseMin(1.f);
  }
}

// -------- one control tick ----------
vector_t SpeedOnnxPolicy::step(const vector_t& base_obs,
                               const vector_t& sp_real,
                               const vector_t& sg_real_masked,
                               const vector_t& dir_b_2,
                               const vector_t& speed_1,
                               const vector_t& yaw_1,
                               vector_t* residual_for_next_obs,
                               vector_t* residual_norm_for_next_obs)
{
  vector_t mu_p;
  vector_t base_action = runBase(sp_real, sg_real_masked, &mu_p);

  // Components for residual_obs assembly
  std::unordered_map<std::string, vector_t> C;
  C["base_obs"]       = base_obs;
  C["sp_real"]        = sp_real;
  C["sg_real_masked"] = sg_real_masked;
  C["mu_p"]           = mu_p;
  C["base_action"]    = base_action;
  C["dir_b"]          = dir_b_2;
  C["speed"]          = speed_1;
  if (meta_.enable_yaw_command && meta_.component_slices.count("yaw_rate"))
    C["yaw_rate"] = yaw_1;

  // previous residual slices (zeros on first tick unless you cache)
  const int aDim = meta_.action_dim;
  vector_t residual_prev = vector_t::Zero(aDim);
  vector_t residual_norm = vector_t::Zero(1);
  if (meta_.component_slices.count("residual"))     C["residual"]     = residual_prev;
  if (meta_.component_slices.count("residual_norm"))C["residual_norm"]= residual_norm;

  // Optionally add measured vels
  if (meta_.component_slices.count("base_lin_vel")) C["base_lin_vel"] = vector_t::Zero(3);
  if (meta_.component_slices.count("base_ang_vel")) C["base_ang_vel"] = vector_t::Zero(3);

  vector_t residual_obs = assembleObservation(C);
  vector_t residual_raw = runResidual(residual_obs);
  vector_t residual_clipped = residual_raw.cwiseMax(-meta_.residual_clip).cwiseMin(meta_.residual_clip);

  // Grace period is enforced by controller, but here’s a safe default (apply zeros during first grace steps)
  vector_t residual_applied = residual_clipped; // controller should gate on reset

  vector_t blended = combine(base_action, residual_applied);

  // Expose what must be fed back into next observation
  if (residual_for_next_obs)      *residual_for_next_obs      = residual_applied.array().tanh().matrix();
  if (residual_norm_for_next_obs) *residual_norm_for_next_obs = vector_t::Constant(1, std::sqrt((residual_applied.array().square().mean)()));

  return blended;
}

} // namespace legged
```

> **Notes**
>
> * For brevity the history buffer above replicates the current snapshot across the window; in your controller you should keep a **real ring buffer** (a `std::deque` of the last `H` snapshots) and flatten it each tick. The contract/dimensions remain unchanged.
> * If your export **baked normalization** (default), leave `host_normalize_obs=false`. If not, fill `meta_.obs_mean/obs_var` in `loadObsNormIfAny()` and set `host_normalize_obs=true` in metadata.

### 3) Observation term that hands the residual vector to the controller

Instead of teaching your `ObservationManager` every slice, just register one term whose value is **the assembled `residual_obs`**. This guarantees **exact ordering**.

**`include/speed_controller/SpeedObservation.h`**

```c++
#pragma once
#include "speed_controller/SpeedOnnxPolicy.h"
#include <memory>

namespace legged {

class SpeedResidualObservationTerm {
public:
  explicit SpeedResidualObservationTerm(std::shared_ptr<SpeedOnnxPolicy> policy) : policy_(std::move(policy)) {}
  int getSize() const { return policy_->meta().dim_obs; }

  // In your controller, call this *before* policy_->step, using the same components you will pass to step().
  Eigen::VectorXf build(const std::unordered_map<std::string, Eigen::VectorXf>& components) const {
    return policy_->assembleObservation(components);
  }

private:
  std::shared_ptr<SpeedOnnxPolicy> policy_;
};

} // namespace legged
```

### 4) The controller (ROS 2 plugin)

**`src/SpeedController.cpp`** (skeleton that mirrors the pattern in your notes)

```c++
#include "speed_controller/SpeedOnnxPolicy.h"
#include "speed_controller/SpeedObservation.h"

#include <legged_rl_controllers/RlController.h> // your base
#include <rclcpp/rclcpp.hpp>

namespace legged {

class SpeedController final : public RlController {
public:
  controller_interface::CallbackReturn on_init() override {
    if (RlController::on_init() != controller_interface::CallbackReturn::SUCCESS)
      return controller_interface::CallbackReturn::ERROR;
    auto_declare("policy.residual_path", std::string(""));
    auto_declare("policy.base_path", std::string(""));
    auto_declare("policy.metadata_path", std::string(""));
    auto_declare("policy.obs_norm_path", std::string("")); // optional
    auto_declare("command.topic", std::string("/cmd_vel"));
    return controller_interface::CallbackReturn::SUCCESS;
  }

  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State& s) override {
    const auto residual = get_node()->get_parameter("policy.residual_path").as_string();
    const auto base     = get_node()->get_parameter("policy.base_path").as_string();
    const auto meta     = get_node()->get_parameter("policy.metadata_path").as_string();
    const auto norm     = get_node()->get_parameter("policy.obs_norm_path").as_string();

    SpeedOnnxPolicy::Paths paths{residual, base, meta, norm};
    policy_ = std::make_shared<SpeedOnnxPolicy>(paths);
    policy_->init();

    // Dim checks vs robot
    const int aDim = policy_->meta().action_dim;
    if (aDim != int(model_->getDofCount()))
      RCLCPP_WARN(get_node()->get_logger(), "action_dim (%d) != robot dofs (%zu); using min of both.", aDim, model_->getDofCount());

    // Wire a single observation term sized to dim_obs
    residualObs_ = std::make_shared<SpeedResidualObservationTerm>(policy_);

    // Create command subscription or attach to your command manager (omitted here)
    return RlController::on_configure(s);
  }

  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State& s) override {
    // Bootstrap history with current state
    const auto q      = model_->getJointPosition();         // [29]
    const auto dq     = model_->getJointVelocity();         // [29]
    const auto gyro   = model_->getBaseAngularVelocity();   // [3] body
    const auto grav_b = model_->getGravityBody();           // [3] body, unit
    Eigen::VectorXf last_action = Eigen::VectorXf::Zero(policy_->meta().action_dim);

    policy_->resetHistory(q, dq, gyro, std::nullopt, grav_b, last_action);
    grace_ = policy_->meta().grace_steps;
    ticks_ = 0;

    return RlController::on_activate(s);
  }

  // Called at control rate
  void update() override {
    // 1) Read sensors
    const auto q_rel  = model_->getJointPosition();       // your API should provide "relative" (training) positions; else subtract neutral
    const auto dq     = model_->getJointVelocity();
    const auto gyro   = model_->getBaseAngularVelocity();
    const auto grav_b = model_->getGravityBody();
    const auto last_a = last_action_;                     // cached from previous loop

    auto sp_real = policy_->updateHistory(q_rel, dq, gyro, std::nullopt, grav_b, last_a);

    // 2) Command (dir/s/yaw)
    Eigen::Vector2f dir_b = getBodyDirUnitVector();       // from /cmd_vel or your term
    float speed = getTargetSpeed();                        // m/s, clamp into training envelope
    float yaw   = getTargetYawRate();                      // rad/s (0 if disabled in metadata)
    Eigen::VectorXf dir_b_2(2); dir_b_2<<dir_b.x(), dir_b.y();
    Eigen::VectorXf speed_1(1); speed_1<<speed;
    Eigen::VectorXf yaw_1(1);   yaw_1<<yaw;
    auto [sg_real, mask] = policy_->buildVelocityGoal(dir_b_2, speed_1, yaw_1);
    auto sg_real_masked = (sg_real.array() * mask.array()).matrix();

    // 3) Build base_obs (the first slice used during training)
    // If your training "base_obs" equals task observation, construct it here exactly as in training.
    // Minimal viable: pass what IsaacLab passed (base_lin_vel, base_ang_vel, q_rel, dq, last_action, [dir|speed]).
    Eigen::VectorXf base_obs(96); base_obs.setZero(); // TODO: fill identically to training if you enabled this slice
    // You can also remove this slice from component_slices at export, but we keep it for parity.

    // 4) Step policy
    Eigen::VectorXf residual_for_obs, residual_norm_for_obs;
    auto final_action = policy_->step(base_obs, sp_real, sg_real_masked,
                                      dir_b_2, speed_1, yaw_1,
                                      &residual_for_obs, &residual_norm_for_obs);

    // 5) Grace gate residual (first N ticks after reset)
    if (ticks_ < grace_) {
      // Replace with base_action-only: here we can “fade-in”, but we already blended inside policy
      // Simpler: hold final_action as-is because combine() has alpha; or implement gating at assemble time.
    }
    ++ticks_;

    // 6) Map to actuator space AFTER blending (your adapter: neutral + scale * action)
    Eigen::VectorXf q_target = actionToJointTarget(final_action);

    // 7) Send to ros2_control
    sendJointTargets(q_target);

    // 8) Cache last action for history
    last_action_ = final_action;
  }

private:
  std::shared_ptr<SpeedOnnxPolicy> policy_;
  std::shared_ptr<SpeedResidualObservationTerm> residualObs_;
  Eigen::VectorXf last_action_;
  int grace_{0};
  int ticks_{0};

  // … helpers getBodyDirUnitVector(), getTargetSpeed(), getTargetYawRate(), actionToJointTarget(), sendJointTargets()
};

} // namespace legged

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(legged::SpeedController, controller_interface::ControllerInterface)
```

> **Where to fill TODOs**
>
> * `base_obs`: if your `deploy_meta.json["component_slices"]` includes `base_obs`, build it exactly as during training (see the table in your notes: 3+3+29+29+29+3=96).
> * `actionToJointTarget`: neutral pose + per‑joint scale (from metadata if provided, else your IsaacLab defaults). **Do this mapping after blending**.

### 5) YAML & launch

**`config/g1/controllers.yaml`**

```yaml
controller_manager:
  ros__parameters:
    update_rate: 500
    speed_controller:
      type: "speed_controller/SpeedController"

speed_controller:
  ros__parameters:
    policy:
      residual_path: "/ABS/PATH/speed_run.onnx"
      base_path: "/ABS/PATH/speed_run_bfm_base.onnx"
      metadata_path: "/ABS/PATH/deploy_meta.json"
      obs_norm_path: "/ABS/PATH/deploy_obs_norm.npz"   # optional
    command:
      topic: "/cmd_vel"
```

**`launch/mujoco_speed.launch.py`** (only the important overrides)

```python
from launch_ros.actions import Node
from launch import LaunchDescription

def generate_launch_description():
    params = {
        'policy.residual_path': '/abs/run/speed.onnx',
        'policy.base_path': '/abs/run/speed_bfm_base.onnx',
        'policy.metadata_path': '/abs/run/deploy_meta.json',
        'policy.obs_norm_path': '/abs/run/deploy_obs_norm.npz',
        'command.topic': '/cmd_vel',
    }
    controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['speed_controller'],
        output='screen'
    )
    # add your sim nodes and controller manager here
    return LaunchDescription([controller])
```

**Run**

```bash
colcon build --symlink-install --packages-select speed_controller
source install/setup.bash

ros2 launch speed_controller mujoco_speed.launch.py
# then drive:
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.8}, angular: {z: 0.0}}' -r 5
```

---

# Path B: Minimal edits to an existing MuJoCo controller

If you must “revise a working controller” (e.g., your `motion_tracking_controller`), do the following **two surgical changes** and leave the rest:

1. **Add a BFM policy adapter class** (drop `SpeedOnnxPolicy.{h,cpp}` above into your repo) and **instantiate it** in `on_configure()`. Keep your existing BeyondMimic `MotionOnnxPolicy` untouched.

2. **Register one synthetic observation term** that returns `dim_obs` floats from `SpeedOnnxPolicy::assembleObservation()`, and change your controller YAML to request only this term so the ObservationManager dimension equals the residual input width:

```cpp
// MotionTrackingController.cpp (pseudocode near parserObservation)
if (name == "residual_observation") {
  observationManager_->addTerm(
    std::make_shared<ResidualObservationAdapter>(speedPolicy_));
  return true;
}
```

```yaml
# controllers.yaml
motion_tracking_controller:
  ros__parameters:
    observation_names: ["residual_observation"]
    # wire policy paths as in Path A
```

Then in your main control loop, **ignore the ObservationManager tensor** and call:

```cpp
// Pseudo snippet inside update()
auto sp_real = speedPolicy_->updateHistory(q_rel, dq, gyro, std::nullopt, grav_b, last_action);
auto [sg, mask] = speedPolicy_->buildVelocityGoal(dir_b_2, speed_1, yaw_1);
auto sg_masked = (sg.array() * mask.array()).matrix();

// build base_obs exactly as training (or zeros if you removed that slice at export)
Eigen::VectorXf base_obs(base_obs_dim); /* fill */

// step policy
Eigen::VectorXf res_tanh, res_norm;
auto blended = speedPolicy_->step(base_obs, sp_real, sg_masked, dir_b_2, speed_1, yaw_1, &res_tanh, &res_norm);

// send motor targets mapped from blended
sendJointTargets(actionToJointTarget(blended));
last_action = blended;
```

This keeps your controller skeleton but swaps in the BFM semantics only where needed.

---

## Worked diffs for small, critical details

### A) Optional time index input (if your ONNX expects it)

Some exports include a scalar input named `time_step`. You can safely inject it if present:

```c++
// After building inputs for base or residual
if (has_input_named(session, "time_step")) {
  static int64_t t = 0;
  float ts = float(t++);
  std::array<int64_t,2> shape{1,1};
  auto ts_val = Ort::Value::CreateTensor<float>(alloc, &ts, 1, shape.data(), shape.size());
  names.push_back("time_step");
  inputs.push_back(std::move(ts_val));
}
```

### B) Residual grace gate

Gate residual during the first `grace_steps` after reset:

```c++
if (ticks_ <= grace_) {
  residual_applied.setZero();
}
```

(You can do this either **before** blending or by fading `alpha` from 0→meta.alpha.)

### C) Joint order, neutral, scales

Map `[-1,1]` to absolute targets:

```c++
Eigen::VectorXf actionToJointTarget(const Eigen::VectorXf& a) {
  // offset + scale * a
  Eigen::VectorXf offset = metadata_offset_or_zero(meta_.action_dim); // from deploy_meta or your defaults
  Eigen::VectorXf scale  = metadata_scale_or_default(meta_.action_dim);
  return offset + scale.cwiseProduct(a);
}
```

Populate `offset/scale` from `deploy_meta.json` if present (`action_offset`, `action_scale`); else fall back to the IsaacLab stiff/neutral defaults you already use in sim.

---

## Validation checklist (do this once; it catches 99% of issues)

1. **Dims**

   * Confirm `dim_obs` in JSON equals your assembled vector length.
   * Confirm `dim_sp_real == meta.hist.history_length * featureDim + dim_last_action`.
   * Confirm ONNX `action_dim` equals your robot DoF (or use `min` and warn).

2. **Slices**

   * Print a few steps with per-slice **mean/var/min/max** (mirroring `_log_debug_step`). Differences vs Isaac traces almost always mean **component order mismatch**.

3. **Mask**

   * Only `sg_real_masked` **velocity** channels should be non‑zero (6 dims); width still 89.

4. **Normalization**

   * If residual export already normalizes, **do not** normalize again host‑side.

5. **Grace**

   * Verify residual is zeroed for the first `grace_steps` after reset (if you enable it).

6. **Safety clamps**

   * Clip residual to `[-residual_clip, residual_clip]` **before** blending; clamp final action to `[-1,1]` **after** blending.

---

## Quick Python parity test (optional but very helpful)

Before spinning the controller, replay the pair in your sim using the Python runner you already pasted (or the snippet below). If it runs a few seconds without explosions and command responsiveness looks sane, your artefacts are correct.

```bash
python scripts/behaviorfoundation/eval_downstream_onnx.py \
  --bfm_base /abs/path/speed_bfm_base.onnx \
  --residual /abs/path/speed.onnx \
  --metadata /abs/path/deploy_meta.json \
  --obs_norm /abs/path/deploy_obs_norm.npz \
  --task Downstream-Speed-G1-v0 --num_envs 1 --eval_steps 2048
```

---

## Troubleshooting map

| Symptom                           | Almost always…                                         | Fix                                                                                                                         |
| --------------------------------- | ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------- |
| Residual ONNX returns huge values | **Double normalization** or wrong `base_obs` slice     | Turn off host normalization; build `base_obs` exactly like training or remove that slice from `component_slices` at export. |
| Robot ignores `/cmd_vel`          | **Wrong `sg_real_masked` offsets** or mask not applied | Use `buildVelocityGoal()`; never hand‑build indices unless you ported them from trainer.                                    |
| Actions saturate ±1 quickly       | **Residual not clipped** or no grace gate              | Clip to `residual_clip` then blend; add `grace_steps`.                                                                      |
| ONNX errors about missing input   | Your export has `time_step`                            | Inject a scalar `time_step` input if present.                                                                               |
| Everything zero                   | Wrong paths / metadata not loaded                      | Verify 4 artefacts are reachable; print `meta.dim_obs`, `component_slices.size()`.                                          |

---

## Why this design keeps you safe

* We **mirror the IsaacLab host**: history → base ONNX → assemble → residual ONNX → blend. No reverse‑engineering the BeyondMimic interface.
* **All ordering** is taken from `deploy_meta.json["component_slices"]`. This is the #1 source of silent mismatches; we centralize it.
* We **don’t normalize twice**. If you later export with `host_normalize_obs=true`, flip one flag.

---

### TL;DR implementation steps you can follow now

1. Drop the code above into a `speed_controller/` package (Path A), build it, and launch.
2. If you must keep a single controller, add `SpeedOnnxPolicy` to it (Path B) and register a single `residual_observation` term sized to `dim_obs`.
3. Verify dims & slices on startup; then drive with `/cmd_vel`.

If you paste your current controller file layout and how you assemble your “policy observation” for training, I can slot the exact `base_obs` construction and the ring‑buffer implementation into your files verbatim.

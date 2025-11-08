# Speed Controller Deployment Guide (Mujoco & Hardware)

**Last updated:** 2025-10-10  
**Maintainers:** Behavior Foundation / Controls Team

This document explains how to deploy the `speed_controller` package with the
hierarchical base + residual ONNX policies that come out of the Behavior
Foundation downstream pipeline. It covers both simulator validation in MuJoCo
and ROS 2 deployment on the physical Unitree platforms.

---

## 1. Required Artifacts

Every run of `scripts/experiments/train_speed_bfm.sh` emits four files under
`logs/rsl_rl/<run_name>` that must stay together when you deploy:

| File | Purpose |
| --- | --- |
| `model_<iter>.pt` | Raw RSL-RL checkpoint (kept for auditing; not used by C++ runtime). |
| `deploy_meta.json` | Deployment manifest describing network dimensions, observation layout, alpha, etc. |
| `deploy_obs_norm.npz` | Optional observation normalisation statistics (mean/var/count). |
| `deploy/<run_name>/<run_name>.onnx` | Residual actor ONNX exported by `export_speed_residual.py`. |
| `deploy/<run_name>/<run_name>_bfm_base.onnx` | BFM prior + decoder ONNX exported by `export_bfm_base.py`. |

> **Naming convention**  
> For deployment we copy / rename as follows to keep paths short:  
> `speed_bfm_a0.1_1009_1250_iter021000.onnx` → residual policy  
> `speed_bfm_a0.1_1009_1250_bfm_base.onnx` → base policy

Store the four files in a dedicated directory (e.g. `/opt/bfm_speed/<run_id>/`)
and pass those paths to the launch files described later.

---

## 2. ONNX Contracts

### 2.1 Base Policy (`*_bfm_base.onnx`)

| Port | Shape | Type | Notes |
| --- | --- | --- | --- |
| **Inputs** ||||
| `sp_real` | `[N, dim_sp_real]` | `float32` | Flattened history buffer. Default 1629 for G1 (25 steps × 64 features + 29 last-action). |
| `sg_real_masked` | `[N, dim_goal_unified]` | `float32` | Masked unified goal vector (velocity-only in speed task → 89 dims). |
| **Outputs** ||||
| `base_action` | `[N, action_dim]` | `float32` | Primary action in `[-1, 1]` (29 DoF for G1). |
| `mu_p` | `[N, latent_dim]` | `float32` | Prior latent mean (64 by default). Used as residual feature. |

### 2.2 Residual Policy (`*.onnx`)

| Port | Shape | Type | Notes |
| --- | --- | --- | --- |
| **Input** ||||
| `residual_obs` | `[N, dim_obs]` | `float32` | Concatenated observation described by `component_slices`. For the default speed task this is 1,947 elements. |
| **Output** ||||
| `actions` | `[N, residual_dim]` | `float32` | Residual action (29 DoF). This tensor is not tanh-squashed. |

All tensors are batched; we run with `N = 1` per control loop, but the ONNX
graphs are exported with dynamic batch axes so vectorised evaluation is possible.

---

## 3. Metadata & Normalisation Files

### 3.1 `deploy_meta.json`

Example (abridged):

```json
{
  "action_dim": 29,
  "alpha": 0.1,
  "component_slices": {
    "base_obs": [0, 96],
    "sp_real": [96, 1725],
    "sg_real_masked": [1725, 1814],
    "mu_p": [1814, 1878],
    "base_action": [1878, 1907],
    "residual": [1907, 1936],
    "residual_norm": [1936, 1937],
    "dir_b": [1937, 1939],
    "speed": [1939, 1940],
    "yaw_rate": [1940, 1941],
    "base_lin_vel": [1941, 1944],
    "base_ang_vel": [1944, 1947]
  },
  "dim_goal_unified": 89,
  "dim_obs": 1947,
  "goal_source": "speed",
  "history": {
    "history_length": 25,
    "dim_joint_pos": 29,
    "dim_joint_vel": 29,
    "dim_root_ang_vel": 3,
    "dim_gravity": 3,
    "dim_last_action": 29
  },
  "residual_dim": 29,
  "residual_mode": "blend",
  "speed_command": {
    "grace_steps": 10,
    "direction_mode": "fixed_forward",
    "enable_yaw_command": false
  }
}
```

Key fields consumed by `SpeedOnnxPolicy`:

| Key | Description |
| --- | --- |
| `action_dim`, `residual_dim` | Dimensions of both ONNX outputs; must match robot’s joint count. |
| `dim_goal_unified`, `dim_obs` | Expected goal vector and residual observation width. |
| `component_slices` | Offsets used when packing `residual_obs`. Keep in sync with training. |
| `history` | Defines history buffer sizes (`featureDim()` and `realStateDim()`). |
| `alpha`, `residual_mode` | Action blending parameters (additive `delta_a` vs convex `blend`). |
| `speed_command` | Used to keep the Isaac speed curriculum in sync (grace steps, yaw limits, etc.). |

### 3.2 `deploy_obs_norm.npz`

Optional file containing `mean`, `var`, and `count`. C++ normalisation is
currently stubbed (`applyNormalization_ = false`), so either omit the file
or implement NPZ loading before relying on it. The exported residual ONNX
already bakes in normalisation by default; feed **raw** features unless you
explicitly re-enable host-side normalisation.

---

## 4. Runtime Architecture

`SpeedOnnxPolicy` mirrors the Python latent wrapper used during training:

1. **Robot state extraction** – pulls `q`, `v`, orientation, and gravity from the
   `LeggedModel`.
2. **History buffer** – maintains the last `history_length` frames of
   `[joint_pos, joint_vel, base_ang_vel, gravity]` and the previous action.
3. **Command ingestion** – reads a ROS 2 `geometry_msgs/Twist` from `/cmd_vel`
   (via `VelocityTopicCommandTerm`) and converts it into body-frame direction,
   speed magnitude, and yaw rate.
4. **Base inference** – runs `*_bfm_base.onnx` to get `base_action` and the
   latent mean `mu_p`.
5. **Residual observation assembly** – concatenates components following
   `component_slices`.
6. **Residual inference** – runs the residual ONNX to obtain the correction.
7. **Action blending and clipping** – applies the configured mode/alpha and
   clamps to `[-1, 1]`.

`SpeedController` wraps the policy inside the standard `legged_rl_controllers`
stack. It instantiates a `ZeroObservationTerm` so the observation manager still
presents a trace to logging/replay even though the policy bypasses the
framework’s observation vector.

---

## 5. MuJoCo Deployment

### 5.1 Prerequisites

```bash
# 1) Source your ROS 2 & colcon workspace
source /opt/ros/humble/setup.bash
source ~/colcon_ws/install/setup.bash  # after building below

# 2) Build the workspace (if not already)
colcon build --symlink-install --packages-select speed_controller
```

Ensure the ONNX Runtime shared library is available (the package depends on
`onnxruntime` via `ament`). Verify that the four artifacts described in §1 are
accessible.

### 5.2 Launch Command

```bash
ros2 launch speed_controller mujoco_speed.launch.py \
  robot_type:=g1 \
  residual_path:=/opt/bfm_speed/xp20251009/speed_bfm_a0.1_1009_1250_iter021000.onnx \
  base_path:=/opt/bfm_speed/xp20251009/speed_bfm_a0.1_1009_1250_bfm_base.onnx \
  metadata_path:=/opt/bfm_speed/xp20251009/deploy_meta.json \
  obs_norm_path:=/opt/bfm_speed/xp20251009/deploy_obs_norm.npz
```

What the launch file does:

1. Generates a temporary `controllers.yaml` that replaces the walking controller
   with the new `speed_controller`, and wires the policy paths above.
2. Spawns MuJoCo with the Unitree description and the ROS 2 control plugin.
3. Starts the state estimator (after 2 s) and the speed controller (after 3.5 s).
4. Includes the standard teleop launch, allowing you to publish `/cmd_vel`.

### 5.3 Validation Checklist

- `ros2 topic echo /speed_controller/action` shows bounded joint commands.
- `ros2 topic hz /speed_controller/action` ≈ 50 Hz.
- Publishing a twist on `/cmd_vel` moves the simulated robot as expected.
- `ros2 param get /speed_controller alpha` matches the metadata.
- `ros2 run controller_manager spawner speed_controller` returns success when re-run.

If actions are all zeros:
- Confirm `speed_controller` is active via `ros2 control list_controllers` and not still inactive.
- Tail `/tmp/speed_policy_trace.log` to check whether the residual observation is being packed (look for `buildResidualObservation`).
- Review the startup logs: the controller now prints the command/observation names pulled from `deploy_meta.json`. Any unsupported term aborts configuration before activation.

### 5.4 Example Command Using Repository Assets

The repository includes the artefacts from the October export under `src/speed_controller/`.
You can launch MuJoCo directly with them while iterating on code changes:

```bash
ros2 launch speed_controller mujoco_speed.launch.py \
  robot_type:=g1 \
  residual_path:=$HOME/colcon_ws/src/speed_controller/speed_bfm_a0.1_1009_1250_iter038000.onnx \
  base_path:=$HOME/colcon_ws/src/speed_controller/speed_bfm_a0.1_1009_1250_bfm_base.onnx \
  metadata_path:=$HOME/colcon_ws/src/speed_controller/deploy_meta.json \
  obs_norm_path:=$HOME/colcon_ws/src/speed_controller/deploy_obs_norm.npz \
  --ros-args --log-level speed_controller:=info
```

Use `speed_controller:=debug` for even more verbose traces.

## 6. Debugging & Telemetry

- `tail -f /tmp/speed_policy_trace.log` provides a per-cycle trace including history updates, base/residual outputs, and any ONNX errors.
- `ros2 control list_controllers` shows activation state; switch controllers with `ros2 control switch --activate speed_controller --deactivate standby_controller --strict` once MuJoCo is ready.
- `ros2 topic echo /speed_controller/action` confirms joint targets move away from zero when `/cmd_vel` is non-zero.
- Publish manual commands with `ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}, angular: {z: 0.0}}"`.
- Adjust verbosity via `--ros-args --log-level speed_controller:=debug` or set `RCUTILS_LOGGING_USE_STDOUT=1` for cleaner console output.

## 7. Hardware Deployment

The hardware workflow mirrors the simulator launch. Update the Unitree bringup override to point at the residual/base ONNX pair:

```bash
ros2 launch unitree_bringup real.launch.py \
  robot_type:=g1 \
  network_interface:=eth0 \
  controllers_yaml:=$HOME/colcon_ws/src/speed_controller/config/g1/controllers.yaml \
  speed_controller.policy.residual_path:=$HOME/colcon_ws/src/speed_controller/speed_bfm_a0.1_1009_1250_iter038000.onnx \
  speed_controller.policy.base_path:=$HOME/colcon_ws/src/speed_controller/speed_bfm_a0.1_1009_1250_bfm_base.onnx \
  speed_controller.policy.metadata_path:=$HOME/colcon_ws/src/speed_controller/deploy_meta.json \
  speed_controller.policy.obs_norm_path:=$HOME/colcon_ws/src/speed_controller/deploy_obs_norm.npz \
  speed_controller.command.topic:=/cmd_vel \
  --ros-args --log-level speed_controller:=info
```

After the controller manager is running:

```bash
ros2 control switch --activate speed_controller --deactivate standby_controller --strict --start-asap
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.2}, angular: {z: 0.0}}" -r 5
```

Keep the robot in a harness for initial tests and monitor `/tmp/speed_policy_trace.log` plus `/cmd_vel` traffic. The same metadata validation present in MuJoCo protects the hardware launch: if the JSON advertises an unsupported term you will see an error and the controller will refuse to activate.

## 8. Troubleshooting

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| Controller crashes on start with `failed to open metadata` | Wrong path or file missing. | Verify `metadata_path` exists and is readable by the controller process. |
| Actions stay zero in both MuJoCo and hardware | Control mask zeros out goal or `base_action` missing. | Ensure `component_slices` includes `sg_real_masked` and that `controlMask_` is populated (consider loading mask from metadata instead of hardcoding indices). |
| Residual outputs huge values | Observation normalisation mismatch. | Either export the residual ONNX with baked-in stats (default) or implement NPZ parsing and enable `applyNormalization_`. |
| `onnxruntime` complains about missing `CUDAExecutionProvider` | Runtime falls back to CPU only (expected on robot). | The warning is harmless; ensure CPU provider is available. |
| Controller never activates | `controller_manager` timers didn’t fire or prerequisites missing. | Check launch logs, ensure state estimator spawns before the speed controller, verify parameter names (see YAML). |

---

## 9. Quick Reference

### 9.1 ROS Parameters (`speed_controller` namespace)

| Parameter | Default | Description |
| --- | --- | --- |
| `policy.residual_path` | `""` (required) | Path to residual ONNX. |
| `policy.base_path` | `""` | Path to base ONNX (optional if residual-only control is desired). |
| `policy.metadata_path` | `""` | Deployment manifest with component slices and joint order. |
| `policy.obs_norm_path` | `""` | Observation stats NPZ (optional). |
| `command.topic` | `/cmd_vel` | Twist topic consumed by the controller. |

### 9.2 Topics

| Topic | Type | Notes |
| --- | --- | --- |
| `/cmd_vel` | `geometry_msgs/Twist` | Velocity command consumed by controller. |
| `/speed_controller/action` | `unitree_msgs/LowLevelCommand` (via ROS 2 control) | Actuated joint targets after blending. |

---

## 10. Next Steps / Open Items

- Load normalisation statistics from `deploy_obs_norm.npz` so the policy can
  consume un-whitened ONNX exports when necessary.
- Replace the hard-coded control mask with the `mask` info saved in
  `deploy_meta.json` for future tasks that expose more goal channels.
- Wire the observation manager with real terms instead of `ZeroObservationTerm`
  so ROS bags include the actual residual observation vector.
- Provide a hardware-focused launch file (e.g. `hardware_speed.launch.py`)
  mirroring `mujoco_speed.launch.py` but without teleop.

---

By following the steps above, you can validate a new speed residual policy in
simulation and then roll it out on hardware with minimal manual editing. If
anything in this document drifts from the implementation, please update both
the code and this guide in the same change.

---

## Appendix: Detailed ONNX Observation Requirements

This appendix documents the exact observation requirements for the BFM speed residual policy, detailing every feature dimension and its source.

### A. Training Configuration

The default speed training configuration (`scripts/experiments/train_speed_bfm.sh`) uses:

```bash
TASK=Downstream-Speed-G1-v0
ALPHA=0.1
RESIDUAL_MODE=blend
ACTOR_HIDDEN="1024,512,256"
CRITIC_HIDDEN="1024,512,256"
OBS_STRATEGY=augmented
CONTROL_MODE=velocity_only
GOAL_SOURCE=speed
```

This produces a residual policy that expects **1,947-dimensional** observation vectors with the following layout.

### B. Observation Component Breakdown

The residual observation is assembled by `BFMLatentVecEnv` (trained with `--obs_strategy=augmented`) and consists of these components in order:

| Component | Dimensions | Slice | Source | Description |
|-----------|------------|-------|--------|-------------|
| `base_obs` | 96 | `[0:96]` | Isaac Lab task observation | Proprioceptive state from speed task |
| `sp_real` | 1,629 | `[96:1725]` | History buffer | 25-step real-world proprioception history |
| `sg_real_masked` | 89 | `[1725:1814]` | Control interface | Masked unified goal vector (velocity-only mode) |
| `mu_p` | 64 | `[1814:1878]` | BFM prior output | Prior latent mean from BFM base module |
| `base_action` | 29 | `[1878:1907]` | BFM decoder output | Base action from BFM (for residual context) |
| `residual` | 29 | `[1907:1936]` | Previous step | `tanh` of previous residual action |
| `residual_norm` | 1 | `[1936:1937]` | Previous step | RMS of previous residual: `sqrt(mean(residual²))` |
| `dir_b` | 2 | `[1937:1939]` | Speed command | Body-frame direction unit vector `[cos(θ), sin(θ)]` |
| `speed` | 1 | `[1939:1940]` | Speed command | Target speed magnitude (m/s) |
| `yaw_rate` | 1 | `[1940:1941]` | Speed command | Target yaw rate (rad/s) |
| `base_lin_vel` | 3 | `[1941:1944]` | Robot sensor | Current body linear velocity (m/s) |
| `base_ang_vel` | 3 | `[1944:1947]` | Robot sensor | Current body angular velocity (rad/s) |
| **TOTAL** | **1,947** | | | |

### C. Component Details

#### C.1 Base Observation (`base_obs`, 96 dims)

From `DownstreamSpeedEnvCfg.observations.policy` (speed_env_cfg.py:750-766):

```python
base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))     # [3]
base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))     # [3]
joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))     # [29]
joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))       # [29]
actions = ObsTerm(func=mdp.last_action)                                                  # [29]
speed_goal = ObsTerm(func=speed_goal)                                                    # [3] (dir_b[2] + speed[1])
```

**Total:** 3 + 3 + 29 + 29 + 29 + 3 = **96 dimensions**

#### C.2 Real Proprioception History (`sp_real`, 1,629 dims)

Built by `RealStateBuilder` (state_builders.py:206-304) maintaining a 25-step ring buffer:

```
[q_{t-24:t}]       : joint positions          [25 × 29] = 725 dims
[qdot_{t-24:t}]    : joint velocities         [25 × 29] = 725 dims
[ω_root_{t-24:t}]  : root angular velocity    [25 × 3]  = 75 dims
[g_{t-24:t}]       : normalized gravity (body)[25 × 3]  = 75 dims
[a_{t-1}]          : last action              [29]      = 29 dims
```

**Total:** 725 + 725 + 75 + 75 + 29 = **1,629 dimensions**

**Key implementation notes:**
- History buffer is pre-filled on reset with current sensor snapshot (avoids cold-start zeros)
- Gravity vector is transformed to body frame via `quat_apply_inverse(root_quat, gravity_world)` and normalized
- Ring buffer uses FIFO updates (state_builders.py:130-143)

#### C.3 Masked Unified Goal (`sg_real_masked`, 89 dims)

Unified control interface (control_interface.py:44-83) with velocity-only mask:

```
root_pos          : [3]  (position targets, masked out)
root_ori_6d       : [6]  (orientation 6D rotation, masked out)
root_lin_vel      : [3]  (linear velocity targets) ✓ ACTIVE
root_ang_vel      : [3]  (angular velocity targets) ✓ ACTIVE
keypoints         : [45] (15 keypoints × 3, masked out)
joint_targets     : [29] (joint angle targets, masked out)
```

**Total:** 3 + 6 + 3 + 3 + 45 + 29 = **89 dimensions**

**Velocity-only mask** (applied in BFMLatentVecEnv for speed task):
```python
mask = torch.zeros(B, 89, device=device)
mask[:, 6:9] = 1.0   # root_lin_vel [6:9]
mask[:, 9:12] = 1.0  # root_ang_vel [9:12]
sg_real_masked = sg_real_full * mask
```

Only 6 dimensions are non-zero after masking, but the full 89-dimensional vector is passed to maintain interface consistency.

#### C.4 Prior Latent Mean (`mu_p`, 64 dims)

Output from BFM prior network:
```python
mu_p, _ = bfm.prior(sp_real, sg_real_masked)  # [B, 64]
```

This is the **deterministic** mean of the prior distribution P(z | sp_real, sg_real). During deployment, the BFM base ONNX module outputs this alongside `base_action`.

#### C.5 Base Action (`base_action`, 29 dims)

Output from BFM decoder:
```python
base_action = bfm.decoder(sp_real, mu_p)  # [B, 29]
base_action = torch.clamp(base_action, -1.0, 1.0)
```

Normalized joint position targets in `[-1, 1]` before scaling to robot joint ranges.

#### C.6 Previous Residual State (`residual`, `residual_norm`, 30 dims)

Maintained by the latent wrapper to provide temporal context:

```python
residual_prev = torch.tanh(residual_action_clipped)  # [B, 29]
residual_norm = torch.sqrt((residual_prev ** 2).mean(dim=-1, keepdim=True))  # [B, 1]
```

On reset, both are initialized to zeros.

#### C.7 Speed Command (`dir_b`, `speed`, `yaw_rate`, 4 dims)

Built by `SpeedCommand` term (speed_env_cfg.py:209-425):

```python
# Direction vector (normalized)
dir_b = torch.tensor([cos(heading), sin(heading)])  # [2]

# Speed magnitude (curriculum-scheduled)
speed = torch.rand() * (speed_max - speed_min) + speed_min  # [1]

# Yaw rate (optional, zero if not enabled)
yaw_rate = clamp(randn() * yaw_std, yaw_min, yaw_max)  # [1]
```

**Speed curriculum** (train_speed_bfm.sh defaults):
- Initial: `speed_max_start = 1.0 m/s`
- Final: `speed_max_end = 4.5 m/s`
- Warmup: `speed_warmup_iters = 10,000`
- Grace period: `grace_steps = 10` (termination disabled during ramp-up)

#### C.8 Current Body Velocities (`base_lin_vel`, `base_ang_vel`, 6 dims)

Read directly from robot sensors:

```python
base_lin_vel = mdp.base_lin_vel(env)  # [B, 3] in body frame
base_ang_vel = mdp.base_ang_vel(env)  # [B, 3] in body frame
```

These provide the residual policy with immediate feedback on the robot's current motion state.

### D. Observation Normalization

The residual ONNX **embeds normalization statistics** via `ResidualDeployModule` (train-downstream.py:538-596). The runtime host must feed **raw, unnormalized** features.

**Normalization parameters** (saved in `deploy_obs_norm.npz`):
```python
obs_mean: [1947] float32  # Running mean from training
obs_var:  [1947] float32  # Running variance from training
count:    scalar int64    # Sample count (metadata)
```

**Applied inside ONNX**:
```python
x_normalized = (x_raw - obs_mean) / sqrt(obs_var + 1e-5)
```

### E. Runtime Assembly Pseudocode

```cpp
// 1. Update history buffer
history_buffer.push(extract_current_features(robot));

// 2. Build sp_real [1629]
auto sp_real = history_buffer.get_flat();

// 3. Build sg_real_masked [89]
auto goal_vector = pack_control_vector(
    root_pos=zeros(3),
    root_ori=zeros(6),
    root_lin_vel=dir_b * speed,  // [vx, vy, 0]
    root_ang_vel=[0, 0, yaw_rate],
    keypoints=zeros(45),
    joints=zeros(29)
);
auto mask = create_velocity_only_mask();
auto sg_real_masked = goal_vector * mask;

// 4. Run BFM base ONNX
auto [base_action, mu_p] = bfm_base_onnx->run(sp_real, sg_real_masked);

// 5. Assemble residual observation [1947]
auto residual_obs = concat({
    base_obs,          // [96]  from speed task observation
    sp_real,           // [1629]
    sg_real_masked,    // [89]
    mu_p,              // [64]
    base_action,       // [29]
    residual_prev,     // [29]
    residual_norm,     // [1]
    dir_b,             // [2]
    speed,             // [1]
    yaw_rate,          // [1]
    base_lin_vel,      // [3]
    base_ang_vel       // [3]
});

// 6. Run residual ONNX (normalization baked in)
auto residual_action = residual_onnx->run(residual_obs);

// 7. Blend actions
auto action_final = blend(base_action, residual_action, alpha, residual_mode);

// 8. Update residual history
residual_prev = tanh(residual_action);
residual_norm = sqrt(mean(residual_prev^2));
```

### F. Validation Checklist

To verify your runtime implementation matches training:

1. **Dimension check**: Ensure `residual_obs.shape == (1, 1947)`
2. **Component slices**: Verify offsets match `component_slices` in `deploy_meta.json`
3. **History initialization**: Pre-fill buffer on reset (not zeros)
4. **Mask consistency**: Only `sg_real_masked[6:12]` should be non-zero for speed task
5. **Normalization**: Feed **raw** features to ONNX (stats already embedded)
6. **Residual history**: Update `residual_prev`/`residual_norm` after each step
7. **Speed curriculum**: Restore `global_step` from metadata to resume schedule
8. **Grace period**: Disable terminations for first `grace_steps` (default 10)

### G. Common Pitfalls

| Issue | Symptom | Fix |
|-------|---------|-----|
| History buffer not pre-filled | Robot falls immediately after reset | Call `history_buffer.bootstrap_with_current()` on reset |
| Normalization applied twice | Erratic joint commands | Remove host-side normalization; ONNX already normalizes |
| Wrong component order | Actions ignore commands | Verify slice offsets against `component_slices` in metadata |
| Zero `base_obs` | Policy ignores proprioception | Ensure speed task observation is passed correctly |
| Residual history not updated | Policy converges to base-only | Update `residual_prev`/`residual_norm` after blending |

### H. Reference Files

- Training script: `scripts/train-downstream.py` (lines 1-1515)
- Latent wrapper: `scripts/behaviorfoundation/bfm/latent_env.py`
- State builder: `scripts/behaviorfoundation/bfm/state_builders.py` (lines 206-365)
- Control interface: `scripts/behaviorfoundation/bfm/control_interface.py` (lines 86-268)
- Speed task config: `source/whole_body_tracking/whole_body_tracking/tasks/downstream/speed_env_cfg.py` (lines 750-766)
- Training defaults: `scripts/experiments/train_speed_bfm.sh`
- Metadata schema: `scripts/train-downstream.py` (lines 915-1040)

---

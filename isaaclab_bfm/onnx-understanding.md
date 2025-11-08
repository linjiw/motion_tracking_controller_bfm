# ONNX Deployment Reference

This note explains how the downstream Behavior Foundation Model (BFM) exports
their ONNX artefacts, what each graph expects on its inputs, and how the paired
deployment metadata must be consumed when porting the controller to a new host
such as the MuJoCo scripts in `scripts/behaviorfoundation/eval_downstream_onnx.py`.
All specifics come from the latent wrapper (`scripts/behaviorfoundation/bfm/latent_env.py`),
the runtime host (`scripts/behaviorfoundation/bfm/latent_host.py`),
and the exporters embedded in `scripts/train-downstream.py`.

---

## 1. Artefact Bundle

Training with `scripts/train-downstream.py --export_onnx` produces four files
under `logs/rsl_rl/<run>/` (see `deploy-mujoco.md` §1):

- `deploy_meta.json` – manifest of dimensions, observation layout, control
  flags, and curriculum settings captured in `_save_deploy_metadata`.
- `deploy_obs_norm.npz` – optional observation normalisation stats saved from
  `LatentEnv._obs_rnorm`.
- `<run>_bfm_base.onnx` – frozen BFM prior+decoder.
- `<run>.onnx` – residual PPO actor with normalisation folded into the graph.

Keep these files together; the ONNX pair alone is insufficient because the host
needs the metadata to rebuild the observation vector and command mask.

---

## 2. Graph Topology Overview

During deployment the controller runs two graphs per control tick:

1. **BFM base (`*_bfm_base.onnx`)** recreates the motion prior using the
   proprioceptive history (`sp_real`) and masked goal (`sg_real_masked`). It
   outputs the base action and the latent statistics (`mu_p`).
2. **Residual (`*.onnx`)** consumes the downstream observation (assembled from
   components defined at train time) and predicts a residual action. The host
   blends this correction with the base action using the mode/alpha stored in
   the metadata (`ResidualConfig` mirrors `latent_host.py`).

Both graphs are exported with dynamic batch axes so you can evaluate `N=1`
(typical for MuJoCo) or batched rollouts (`eval_downstream_onnx.py` defaults to
`N=2048`).

---

## 3. Input / Output Contracts

### 3.1 BFM base module (`*_bfm_base.onnx`)

| Port | Shape | Description |
| --- | --- | --- |
| `sp_real` | `[N, dim_sp_real]` | Flattened history buffer built by `RealStateBuilder` / `ProprioHistory`. For G1 speed tasks this is `25 × (joint_pos 29 + joint_vel 29 + base_ang_vel 3 + gravity 3)` plus the last action → 1,629 elements. |
| `sg_real_masked` | `[N, dim_goal_unified]` | Unified control vector masked according to the sampled control mode (velocity-only for speed, motion keypoints for push/kick). Typical width is 89 for speed tasks. |
| `base_action` | `[N, action_dim]` | Clamped action in `[-1, 1]`. Action dim comes from the robot (29 DoF for G1). |
| `mu_p` | `[N, latent_dim]` | Prior mean emitted by the BFM student (64 by default). |

Dummy tensors used during export are declared in `_export_bfm_base_module`
(`scripts/train-downstream.py:1173-1235`) and define the symbolic axes.

### 3.2 Residual module (`*.onnx`)

| Port | Shape | Description |
| --- | --- | --- |
| `residual_obs` | `[N, dim_obs]` | Concatenated downstream observation assembled in the exact order recorded under `component_slices`. Default speed layout (inherit mode) is 1,947 floats. |
| `actions` (or `residual`) | `[N, residual_dim]` | Residual action before blending. Not tanh-squashed; clipping happens in the host (`latent_host.py:270-315`). |

The observation layout is not hard-coded: it inherits the canonical component
selection from `LatentEnvCfg.obs_strategy` / `obs_components`. Always generate
the residual observation by following `deploy_meta.json["component_slices"]`.

---

## 4. Metadata Files

### 4.1 `deploy_meta.json`

Key fields emitted in `_save_deploy_metadata` (`scripts/train-downstream.py:997-1124`):

- `component_slices`: map of component name → `[start, end)` offsets used when
  packing `residual_obs` (e.g., `"base_obs": [0, 160]`).
- `obs_components`: ordered tuple of enabled canonical components (base,
  bfm_context, goal, base_action, residual, commands, base_vels, task_extras).
- `dim_sp_real`, `dim_goal_unified`, `dim_obs`, `action_dim`, `residual_dim`,
  `latent_dim`: numerical contracts for both graphs.
- `state_specs`: history buffer sizes (joint dims, gravity flag, history length)
  required to configure `ProprioHistory`.
- `residual_mode`, `alpha`, `residual_clip`, `control_mode`, `goal_source`:
  blending semantics and control mask configuration.
- `speed_command`: current curriculum snapshot (max speeds, yaw config,
  `grace_steps`, `_global_step`). `eval_downstream_onnx.py` shows how to replay
  these settings before running inference (`_apply_speed_metadata`).

### 4.2 `deploy_obs_norm.npz`

Contains `mean`, `var`, and `count` vectors produced by `RunningNorm`. The
residual ONNX already bakes these statistics by default, but the host may still
apply them if `host_normalize_obs` is true in the metadata. `LatentEnvHost`
handles this automatically when instantiated via `from_metadata`.

---

## 5. Observation Assembly

`LatentEnv` exposes two knobs (`obs_strategy`, `obs_components`) that control which
elements appear in `residual_obs` (`bfm-downstream-obs.md`). The canonical
components (see `_OBS_COMPONENT_ORDER` in `latent_env.py`) are:

| Component | Meaning | Typical source |
| --- | --- | --- |
| `base_obs` | Wrapped env `policy` tensor (same obs PPO RL sees). |
| `sp_real` | Proprio history stack from `RealStateBuilder` / `ProprioHistory`. |
| `sg_real_masked` | Masked unified goal vector (speed dir/s magnitude or motion keypoints). |
| `mu_p` | Prior mean from BFM base inference. |
| `base_action` | Decoder output before blending; used by residual to stay close to base. |
| `residual` / `residual_norm` | Previous residual action and its norm for autoregressive features. |
| `dir_b`, `speed`, `yaw_rate` | Command triplet used by the speed curriculum. |
| `base_lin_vel`, `base_ang_vel` | Measured root velocities. |
| `task_extras` | Optional per-task tensors (e.g., push-box object state). |

When `obs_strategy=inherit` (default for speed/tracking) the first slice equals
the env’s native observation so downstream PPO matches baseline RL inputs.
Other strategies (`control`, `augmented`) retain the Stage-2 layout if legacy
experiments need them.

The deployment host must follow these exact slices when re-creating the tensor.
`LatentEnvHost.assemble_observation()` accepts a dict of tensors keyed by the
slice names (`base_obs`, `sp_real`, `sg_real_masked`, …) and enforces width
checks, making it a reliable reference for new runtimes.

---

## 6. Runtime Host Responsibilities

`scripts/behaviorfoundation/bfm/latent_host.py` is the authoritative template
for any non-Isaac runtime. Re-implement the following pieces in the target sim:

1. **History buffer (`ProprioHistory`)** – Maintain a ring buffer for the last
   `history_length` steps of `[base_lin_vel?, joint_pos, joint_vel, root_ang_vel,
   gravity]` plus the last action. On reset, fill the entire buffer with the
   current snapshot to avoid cold starts.
2. **Goal interface (`BFMControlInterface`)** – Convert commands into the unified
   control vector and mask. For the speed task, build the body-frame direction,
   scalar speed, and yaw rate via `build_velocity_targets_body`; multiply by the
   mask returned by `mask_velocity_only`.
3. **Base inference** – Call the BFM ONNX session with the history and masked
   goal to obtain `base_action`, `mu_p`. Use the same device precision for both
   graphs to avoid CPU/GPU mismatches (`eval_downstream_onnx.py` keeps them on
   the env device).
4. **Observation packing** – Create the dict of component tensors and hand it to
   `assemble_observation()`. If `deploy_meta.json["host_normalize_obs"]` is true
   and norm stats are available, the host normalises before feeding the residual.
5. **Residual inference & blending** – Run the residual ONNX, clamp to
   `[-residual_clip, residual_clip]`, optionally gate during warmup (speed term’s
   `grace_steps`), and blend with the base action using
   `ResidualConfig.combine()` (additive `delta_a` or convex `blend`).
6. **State bookkeeping** – Track per-env returns, residual history, and command
   state across resets. `LatentEnvHost.reset_history()` shows how to restore the
   ring buffer for only the envs that terminated.

---

## 7. End-to-End Loop Reference

`scripts/behaviorfoundation/eval_downstream_onnx.py` provides a working example
for Isaac Lab:

1. Load `deploy_meta.json` and `deploy_obs_norm.npz`, hydrate `LatentEnvHost`.
2. Instantiate the ONNX sessions via `onnxruntime.InferenceSession`.
3. Reset the vector env, seed the history buffer with the initial robot state,
   and restore the speed command’s `_global_step` if present.
4. Each simulator tick:
   - Update history with the latest proprio state.
   - Build the masked goal (`build_velocity_goal`).
   - Run the base graph → `base_action`, `mu_p`.
   - Gather component tensors (base observation, state history, commands,
     previous residual, etc.) and assemble `residual_obs`.
  - Run the residual graph, clamp/blend actions, and step the env.
   - When environments reset, clear their history slices and residual state.
5. Periodically log diagnostics (`_log_debug_step`) to verify tensor statistics,
   clipping ratio, and observation slices. This is invaluable when validating a
   new MuJoCo host against Isaac Lab results.

Replicating this flow inside MuJoCo ensures parity with the training/eval path.

---

## 8. MuJoCo Integration Checklist

Use `deploy-mujoco.md` §5 as the baseline and verify the following before
shipping into the new simulator:

1. **Sensor parity** – MuJoCo must expose the same signals used by
   `ProprioHistory`: joint positions/velocities, base angular velocity, base
   linear velocity (if enabled), gravity in world frame, and the last commanded
   action.
2. **Command pipeline** – Match the Isaac speed command curriculum. Port the
   `SpeedCommand` parameters from metadata (speed ranges, yaw enable, grace
   steps) and drive body-frame direction + speed scalars into the control
   interface.
3. **Timing / batching** – Even though ONNX graphs accept batch > 1, MuJoCo
   loops typically run with `N=1`. Keep arrays batched (shape `[1, dim]`) to
   stay compatible with dynamic axis expectations.
4. **Normalisation** – Decide whether to rely on the ONNX-embedded stats or to
   apply host-side normalisation. Be consistent with `host_normalize_obs`.
5. **Action scaling** – Final blended actions live in `[-1, 1]`. Convert to the
   motor command space required by MuJoCo / hardware after blending, not before.
6. **Regression harness** – Run `eval_downstream_onnx.py` with the same artefact
   bundle and seed to record baseline rewards/action norms, then reproduce those
   metrics in MuJoCo (allow a small tolerance for physics differences).

Following this checklist keeps the MuJoCo deployment aligned with the behavior
seen in Isaac Lab and avoids silent mismatches in the observation contract.

---

## 9. Debugging Tips

- Use `--debug_interval N --debug_env_id 0` when running
  `eval_downstream_onnx.py` to print per-component summaries and ensure your
  MuJoCo telemetry matches the Isaac traces.
- If ONNX Runtime complains about missing inputs/outputs, print
  `OnnxPolicyBundle.describe_io()`; it echoes the tensors exported by
  `torch.onnx.export`.
- When observations drift, dump your MuJoCo-packed tensor and compare it
  against the slice stats that Isaac prints (`component:base_obs`, etc.).
  Mismatched offsets almost always trace back to ignoring `component_slices`.
- Remember that residual warmup is enforced by `grace_steps`. If MuJoCo applies
  the residual immediately after reset, expect transient instability.

This document should give enough structure to treat the ONNX exports as a stable
API and re-host them in MuJoCo or any other simulator without reverse
engineering the latent wrapper again.

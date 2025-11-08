# Extending `motion_tracking_controller` for Behavior Foundation ONNX Bundles

This plan outlines how to evolve the existing motion controller so it can optionally host Behavior Foundation Model (BFM) exports (dual ONNX graphs + metadata) while retaining compatibility with current motion-tracking policies. The approach focuses on factoring shared utilities from `speed_controller` and carefully threading them into the motion package without breaking existing APIs.

## 1. Goals & Constraints
- Maintain the current motion-tracking contract (single ONNX policy that emits anchor/body poses) while adding an alternate execution path for BFM residual models.
- Keep all ROS-facing parameters, launch files, and command interfaces backward-compatible by default; new behavior is opt-in.
- Reuse proven infrastructure from `src/speed_controller` (metadata parsing, history buffers, dual ONNX inference, observation assembly) instead of re-implementing it.
- Avoid touching downstream packages (`unitree_bringup`, standalone player) beyond new optional parameters.

## 2. High-Level Architecture

1. **Shared Support Layer** *(✅ completed — see `bfm_support/*` for metadata loader, NPZ reader, history buffer, and observation assembler)*
   - Introduce a `behavior_foundation` helper namespace under `src/motion_tracking_controller` (or a sibling common package) that encapsulates:
     - Metadata loading & validation currently handled in `SpeedOnnxPolicy::loadMetadata()` and `configureControlMask()` (`src/speed_controller/src/SpeedOnnxPolicy.cpp:105-398`).
     - Observation normalization and NPZ parsing (`src/speed_controller/src/SpeedOnnxPolicy.cpp:255-333`, `src/speed_controller/src/internal/npz_reader.cpp`).
     - History buffer management mirroring `LatentEnvHost.update_history()` in `src/speed_controller/eval_downstream_onnx.py:360-386`.
     - Harden the NPZ reader so it transparently resolves both bare keys (`"mean"`, `"var"`) and their `.npy` filenames inside the ZIP; emit a concise log when stats load (and when they do not) so MuJoCo traces immediately reveal the whitening state (see `error.log`).
     - Pull controller-facing calibration straight out of the ONNX metadata (default joint pose, per-joint scale, stiffness/damping) so the normalized residuals map back to physical joint targets exactly like `BFMControlInterface` in `latent_host.py`.
   - Expose clean C++ APIs (e.g., `BehaviorMetadata`, `HistoryBuffer`, `ResidualAssembler`) so both controllers can link without duplicating code.

2. **Dual-Mode Policy Wrapper**
   - Refactor `MotionOnnxPolicy` into a thin façade that can delegate to either the existing motion policy implementation or a new `BehaviorFoundationPolicy`.
   - `BehaviorFoundationPolicy` should:
     - Manage two `onnxruntime::InferenceSession`s (base + residual) similar to `SpeedOnnxPolicy::initialiseBaseSession()` and `OnnxPolicy::forward()` (`src/speed_controller/src/SpeedOnnxPolicy.cpp:405-535`).
     - Store metadata-driven tensors (component slices, residual config, command layout) and expose hooks for the controller to push the latest proprio history and commands.
     - Present a consistent API to the controller (`getAction()`, `getObservationNames()`, `requiresHistory()`, etc.) so `MotionTrackingController` does not need policy-specific branches.

3. **Controller Integration**
   - Extend `MotionTrackingController::on_init()` and `on_configure()` to detect the new bundle parameters (`policy.base_path`, `policy.residual_path`, `policy.metadata_path`, etc.) and instantiate the appropriate policy type (`src/motion_tracking_controller/src/MotionTrackingController.cpp:7-35`).
   - Rework `parserObservation()` so it can register:
     - Existing motion-specific observation terms (anchor/body) when the legacy policy is active.
     - A single metadata-driven observation term (e.g., `BehaviorResidualObservationTerm`) that mirrors the assembly in `eval_downstream_onnx.py` (`src/speed_controller/eval_downstream_onnx.py:360-430`) when the BFM policy is active.
   - Inject the velocity command interface already available in `MessageCommandTerm` or reuse `VelocityTopicCommandTerm` from `legged_rl_controllers` to feed the unified goal vector expected by `BehaviorFoundationPolicy`.

4. **Command & Observation Terms**
   - Keep `MotionCommandTerm` untouched for legacy usage, but gate its creation behind the motion policy mode to avoid requesting body poses that the BFM model never outputs (`src/motion_tracking_controller/src/MotionCommand.cpp:9-88`).
   - Add new terms:
     - `BehaviorHistoryObservation` – pulls the flattened history buffer that the policy wrapper maintains (matching `sp_real` slice).
     - `BehaviorGoalObservation` – exposes masked goals / command components (`sg_real_masked`, `dir_b`, `speed`, optional `yaw_rate`).
     - `BehaviorResidualState` – tracks last residual, norms, and optional warmup gating (cf. `residual_norm` handling in `eval_downstream_onnx.py:401-420`).
   - The controller’s observation manager should assemble these pieces using metadata offsets to ensure the final tensor matches `component_slices`.
   - Thread `RCLCPP_*` debug breadcrumbs (throttled/`*_ONCE`) through these terms so the runtime surfaces the same checkpoints that `src/speed_controller/eval_downstream_onnx.py` prints (metadata summary, NPZ status, command fallbacks). This makes reproducing IsaacLab traces inside MuJoCo straightforward.

5. **Launch & Configuration**
   - Update motion controller YAML (`src/motion_tracking_controller/config/g1/controllers.yaml`) to accept either:
     - `motion.policy.path`/`motion.start_step` (legacy).
     - Or the BFM bundle: `motion.policy.residual_path`, `motion.policy.base_path`, `motion.policy.metadata`, `motion.policy.obs_norm`.
   - Extend launch files to pass through the extra arguments, but keep defaults pointing to the legacy configuration so existing workflows stay unchanged.

## 3. Detailed Work Items

1. **Create Shared Utilities**
   - Move NPZ reader and metadata structs into `motion_tracking_controller/include/motion_tracking_controller/bfm_support/`.
   - Port the JSON parsing and component slice validation logic from `SpeedOnnxPolicy::loadMetadata()` verbatim to avoid skew (`src/speed_controller/src/SpeedOnnxPolicy.cpp:105-225,343-405`).
   - Implement a reusable `HistoryBuffer` class that mirrors the behavior of `LatentEnvHost.update_history()` and `reset_history()` (`src/speed_controller/eval_downstream_onnx.py:360-441`).

2. **Implement `BehaviorFoundationPolicy`** *(mostly implemented — dual sessions, history+goal plumbing, metadata-driven observation assembly, and blending now live in `BehaviorFoundationPolicy`; remaining polish is focused on richer diagnostics and optional unit tests).*
   - Accept the four bundle paths plus optional overrides.
   - During `init()`, load metadata, configure history/control masks, create ONNX sessions, and preload normalization stats.
   - Provide methods:
     - `primeHistory(const RobotState&)`
     - `updateHistory(const RobotState&, const vector_t& last_action)`
     - `runBase(const GoalCommand&)`
     - `runResidual(const ObservationAssembly&)`
     - `blendActions()` to apply `alpha`, `residual_mode`, and `residual_clip` (mirrors `host.combine_actions()` in `eval_downstream_onnx.py:413-430`).

3. **Controller Wiring** *(✅ policy mode/parameter switches landed in `MotionTrackingController`, including the new `behavior_residual_obs` term and grace overrides.)*
   - Modify `MotionTrackingController::on_configure()` to branch on which parameter set is provided, instantiate `MotionOnnxPolicy` (legacy) or `BehaviorFoundationPolicy`, and populate `cfg_` accordingly (`src/motion_tracking_controller/src/MotionTrackingController.cpp:22-35`).
   - Add lifecycle hooks that prime the history buffer at reset/activate (similar to how `LatentEnvHost.reset_history()` is invoked on env resets in `eval_downstream_onnx.py:431-450`).
   - Ensure `commandManager_` feeds the policy with the goal vector each update (existing velocity-topic plumbing from `legged_rl_controllers` can be reused).

4. **Observation Assembly** *(✅ handled inside `BehaviorFoundationPolicy::prepareObservation()`; ObservationManager now sees a single term in BFM mode.)*
   - Implement a `BehaviorObservationAssembler` term that:
     - Pulls all required component tensors from the policy wrapper.
     - Orders them using `componentSlices` metadata (ensuring parity with Python host logic).
     - Applies host-side normalization when enabled.
   - Register this term under a single observation name (e.g., `behavior_residual_obs`) and require users to list only that term in the YAML when running BFM bundles.

5. **Validation & Guardrails** *(🚧 in progress)*
   - [x] Add lightweight slice/stat logging (metadata summary, NPZ load result, command fallback notices) to cross-check MuJoCo traces against `eval_downstream_onnx.py`.
   - [x] Log whether legacy vs. BFM mode is active plus key metadata (dim_obs, action_dim, alpha, residual_mode).
   - [ ] Document a quick-start toggle guide (see README updates below).

## 4. Risk & Mitigation
- **Complexity creep** – Mitigate by isolating BFM logic inside new helper classes so the core controller remains readable.
- **API divergence** – Keep existing public headers untouched; expose new getters via optional interfaces to avoid breaking downstream builds.
- **Testing burden** – Mirror the Python `eval_downstream_onnx.py` loop during bring-up to verify observation parity before integrating with ROS control.

## 5. Next Steps
1. **Diagnostics** – add a temporary `--debug_slices` ROS param (or INFO log burst) to print min/max for each metadata slice during bring-up.
2. **Testing recipe** – script repeatable parity checks (Python host vs. ROS controller) and capture them in README + repo scripts.
3. **Docs/Launch** – finalize README/YAML examples that show how to switch between legacy and BFM modes and point to the validation workflow.
4. **Optional** – gate-time performance profiling or unit tests for `BehaviorFoundationPolicy` once observation parity is confirmed.


How to test (scripts)

  1. Parity check – run the existing Python host to confirm the artefact
     bundle is self-consistent:

     python3 src/speed_controller/eval_downstream_onnx.py \
       --residual_path /abs/policy.onnx \
       --base_path /abs/policy_bfm_base.onnx \
       --metadata_path /abs/deploy_meta.json \
       --obs_norm_path /abs/deploy_obs_norm.npz \
       --num_envs 1 --debug_interval 100
     Keep the slice stats this prints (base_obs, sp_real, etc.) for
     comparison.
     comparison.
  2. ROS/MuJoCo run – launch the motion controller in BFM mode with the
     same artefacts and log the first few seconds for slice checks:

ROS_LOG_LEVEL=INFO ros2 launch motion_tracking_controller mujoco.launch.py \
   policy.mode:=bfm \
   policy.bfm.residual_path:=/home/robotixx/colcon_ws/src/speed_controller/speed_bfm_a0.1_1009_1250_iter038000.onnx \
   policy.bfm.base_path:=/home/robotixx/colcon_ws/src/speed_controller/speed_bfm_a0.1_1009_1250_bfm_base.onnx \
   policy.bfm.metadata_path:=/home/robotixx/colcon_ws/src/speed_controller/deploy_meta.json \
   policy.bfm.obs_norm_path:=/home/robotixx/colcon_ws/src/speed_controller/deploy_obs_norm.npz \
   policy.bfm.grace_override:=-1 \
   > /tmp/bfm_ros.log
s
     Ensure your controller YAML’s observation_names contains only
     behavior_residual_obs. Compare the logged per-slice min/max against
     the Python output; once they match you can proceed to longer MuJoCo
     or hardware sessions.

  Next steps:

  1. Add temporary slice-stat logging flags so observation drift is
     easier to diagnose without scraping /tmp logs.
  2. Update the launch/YAML defaults (or README snippets) to ship a
     ready-made BFM profile, then run a MuJoCo regression to confirm
     velocity commands and grace gating behave as expected.

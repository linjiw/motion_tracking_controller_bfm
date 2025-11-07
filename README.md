# BeyondMimic Motion Tracking Inference

[[Website]](https://beyondmimic.github.io/)
[[Arxiv]](https://arxiv.org/abs/2508.08241)
[[Video]](https://youtu.be/RS_MtKVIAzY)

This repository provides the inference pipeline for motion tracking policies in BeyondMimic. The pipeline is implemented
in C++ using the ONNX CPU inference engine. Model parameters (joint order, impedance, etc.)
are stored in ONNX metadata, and the reference motion is returned via the `forward()` function.
See [this script](https://github.com/HybridRobotics/whole_body_tracking/blob/main/source/whole_body_tracking/whole_body_tracking/utils/exporter.py)
for details on exporting models.

This repo also serves as an example of how to implement a custom controller using the
[legged_control2](https://qiayuanl.github.io/legged_control2_doc/) framework.

## Installation

### Dependencies

This software is built on
the [ROS 2 Humble](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html#ubuntu-deb-packages), which
needs to be installed first. Additionally, this code base depends on `legged_control2`.

### Install `legged_control2`

Pre-built binaries for `legged_control2` are available on ROS 2 Humble. We recommend first reading
the [full documentation](https://qiayuanl.github.io/legged_control2_doc/overview.html).

Specifically, For this repo, follow
the [Debian Source installation](https://qiayuanl.github.io/legged_control2_doc/installation.html#debian-source-recommended).
Additionally, install Unitree-specific packages:

```bash
# Add debian source
echo "deb [trusted=yes] https://github.com/qiayuanl/unitree_buildfarm/raw/jammy-humble-amd64/ ./" | sudo tee /etc/apt/sources.list.d/qiayuanl_unitree_buildfarm.list
echo "yaml https://github.com/qiayuanl/unitree_buildfarm/raw/jammy-humble-amd64/local.yaml humble" | sudo tee /etc/ros/rosdep/sources.list.d/1-qiayuanl_unitree_buildfarm.list
sudo apt-get update
```

```bash
# Install packages
sudo apt-get install ros-humble-unitree-description
sudo apt-get install ros-humble-unitree-systems
```

### Build Package

After installing `legged_control2`, you can build this package. You’ll also need the
`unitree_bringup` repo, which contains utilities not included in the pre-built binaries.

Create a ROS 2 workspace if you don't have one. Below we use `~/colcon_ws` as an example.

```bash
mkdir -p ~/colcon_ws/src
```

Clone two repo into the `src` of workspace.

```bash
cd ~/colcon_ws/src
git clone https://github.com/qiayuanl/unitree_bringup.git
git clone https://github.com/HybridRobotics/motion_tracking_controller.git
cd ../
```

Install dependencies automatically:

```bash
rosdep install --from-paths src --ignore-src -r -y
```

Build the packages:

```bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelwithDebInfo --packages-up-to unitree_bringup
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelwithDebInfo --packages-up-to motion_tracking_controller
source install/setup.bash
```

## Basic Usage

### Sim-to-Sim

We provide a launch file for running the policy in MuJoCo simulation.

```bash
# Load policy from WandB
ros2 launch motion_tracking_controller mujoco.launch.py wandb_path:=<your_wandb_run_path>
```

```bash
# OR load policy from local ONNX file (should be absolute or start with `~`)
ros2 launch motion_tracking_controller mujoco.launch.py policy_path:=<your_onnx_file_path>
```

### Behavior Foundation (BFM) Residual Mode

BFM exports ship as a residual + base pair plus metadata (see `src/speed_controller/onnx-understanding.md`). The motion controller now supports these artefacts behind an opt-in switch:

1. Provide the bundle paths (absolute is safest):

```bash
ros2 launch motion_tracking_controller mujoco.launch.py \
  policy.mode:=bfm \
  policy.bfm.residual_path:=/abs/path/speed_iter.onnx \
  policy.bfm.base_path:=/abs/path/speed_bfm_base.onnx \
  policy.bfm.metadata_path:=/abs/path/deploy_meta.json \
  policy.bfm.obs_norm_path:=/abs/path/deploy_obs_norm.npz \
  command.topic:=/cmd_vel \
  enable_teleop:=false \
  policy.bfm.debug_dump:=true   # optional: log slice stats for first few ticks

# override `command.topic` if your teleop publishes elsewhere. Set `enable_teleop:=true` only when a joystick is present.
```

2. Ensure the controller’s YAML lists the single observation term expected by the residual graph (the launch file will inject this automatically when `policy.mode:=bfm`, but it is useful to know what the controller expects):

```yaml
observation_names:
  - behavior_residual_obs
command_names:
  - speed
```

If you do not have a joystick, either rely on the built-in `policy.bfm.default_forward_speed` (default 1.4 m/s along +X) or publish a constant command to `/cmd_vel`:

```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/TwistStamped "
header:
  stamp: {sec: 0, nanosec: 0}
  frame_id: base
twist:
  linear:  {x: 0.4, y: 0.0, z: 0.0}
  angular: {x: 0.0, y: 0.0, z: 0.0}
" --rate 20
```

3. Optional: override the residual warmup gate (`policy.bfm.grace_override`). Leave it at `-1` to use the metadata value.

Behind the scenes the same velocity-topic command term is reused, so teleop + launch files do not change apart from the parameters above.

### Validation / Test Workflow

To reproduce the training host before running the ROS controller, follow the same two-stage procedure we use internally:

1. **Python parity check (Isaac-style host)**  
   Use the existing evaluation script (can run anywhere inside the workspace):
   ```bash
   python3 src/speed_controller/eval_downstream_onnx.py \
     --residual_path /abs/path/speed_iter.onnx \
     --base_path /abs/path/speed_bfm_base.onnx \
     --metadata_path /abs/path/deploy_meta.json \
     --obs_norm_path /abs/path/deploy_obs_norm.npz \
     --num_envs 1 --debug_interval 100
   ```
   This prints slice-level stats (`base_obs`, `sp_real`, `sg_real_masked`, etc.). Save the console output; it becomes your reference when checking the ROS logs.

2. **ROS controller in MuJoCo**  
   Launch the motion controller with `policy.mode:=bfm` as shown above. Enable debug logs once to compare statistics:
   ```bash
   ROS_LOG_LEVEL=INFO ros2 launch motion_tracking_controller mujoco.launch.py \
     policy.mode:=bfm \
     policy.bfm.residual_path:=... \
     policy.bfm.base_path:=... \
     policy.bfm.metadata_path:=... \
     policy.bfm.obs_norm_path:=... \
     > /tmp/bfm_ros.log
   ```
   The BehaviorFoundationPolicy logs the active mode, action/observation dims, alpha/mode, and per-slice min/max for the first few ticks. Diff these numbers against the Python run—if they agree within small tolerances, you can proceed to longer MuJoCo episodes or real-robot trials.

3. **(Optional) Grace/command verification**  
   Set `policy.bfm.grace_override:=0` to confirm residual actions are live immediately, then revert to the metadata value for deployment. Watching `/motion_tracking_controller/command` telemetry or the `/tmp/bfm_ros.log` residual norms helps confirm the gate is working.
   If your teleop publishes on a custom topic, pass `command.topic:=/my_vel_topic` in the launch invocation so the `VelocityTopicCommandTerm` subscribes correctly.

### Real Experiments

> ⚠️ **Disclaimer**  
> Running these models on real robots is **dangerous** and entirely at your own risk.  
> They are provided **for research only**, and we accept **no responsibility** for any harm, damage, or malfunction.

1. Connect to the robot via ethernet cable.
2. Set the ethernet adapter to static IP: `192.168.123.11`.
3. Use `ifconfig` to find the `<network_interface>`, (e.g.,`eth0` or `enp3s0`).

```bash
# Load policy from WandB
ros2 launch motion_tracking_controller real.launch.py network_interface:=<network_interface> wandb_path:=<your_wandb_run_path>
```

```bash
# OR load policy from local ONNX file (should be absolute or start with `~`)
ros2 launch motion_tracking_controller real.launch.py network_interface:=<network_interface> policy_path:=<your_onnx_file_name>.onnx
```

The robot should enter standby controller in the beginning.
Use the Unitree remote (joystick) to start and stop the policy:

- Standby controller (joint position control): `L1 + A`
- Motion tracking controller (the policy): `R1 + A`
- E-stop (damping): `B`

## Code Structure

This section will be especially helpful if you decide to write your own legged_control2 controller.
For a minimal starting point, check
the [legged_template_controller](https://github.com/qiayuanl/legged_template_controller).

Below is an overview of the code structure for this repository:

- **`include`** or **`src`**
    - **`MotionTrackingController`** Manages observations (like an RL environment) and passes them to the policy.

    - **`MotionOnnxPolicy`** Wraps the neural network, runs inference, and extracts reference motion from the ONNX file.

    - **`MotionCommand`** Defines observation terms aligned with the training code.


- **`launch`**
    - Includes launch files like `mujoco.launch.py` and `real.launch.py` for simulation and real robot execution.
- **`config`**
    - Stores configuration files for standby controller and state estimation params.

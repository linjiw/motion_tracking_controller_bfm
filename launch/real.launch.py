import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    ExecuteProcess,
    DeclareLaunchArgument,
    OpaqueFunction,
    SetLaunchConfiguration,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration, PythonExpression, \
    ThisLaunchFileDir
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


# --------------------------
# Internal: minimal generic override
# --------------------------
def generate_temp_config(config_path, package_name, kv_pairs):
    """
    Load <package_name>/<config_path>, apply overrides from kv_pairs,
    and write to /tmp/<package_name>/temp_controllers.yaml. Returns the path.
    kv_pairs: list of (dotted_key, raw_value_str)
    """
    pkg_dir = get_package_share_directory(package_name)
    src_path = os.path.join(pkg_dir, config_path)
    dst_path = os.path.join('/tmp', package_name, 'temp_controllers.yaml')
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    with open(src_path, 'r') as f:
        cfg = yaml.safe_load(f) or {}

    for dotted_key, raw_val in kv_pairs:
        parts = [p for p in dotted_key.split('.') if p]
        if len(parts) < 2:
            raise ValueError(
                f"Key '{dotted_key}' is incomplete; expected '<ns>.[ros__parameters.]foo.bar'"
            )
        # Auto-insert ros__parameters right after namespace if omitted
        if parts[1] != 'ros__parameters':
            parts.insert(1, 'ros__parameters')

        try:
            val = yaml.safe_load(raw_val)
        except Exception:
            val = raw_val

        cur = cfg
        for k in parts[:-1]:
            if not isinstance(cur.get(k), dict):
                cur[k] = {}
            cur = cur[k]
        cur[parts[-1]] = val

    with open(dst_path, 'w') as f:
        yaml.dump(cfg, f, sort_keys=False)
        print(f"[launch] Temp controllers.yaml written to {dst_path}")

    return dst_path


# --------------------------
# ROS nodes / launch wiring
# --------------------------
def control_spawner(names, inactive=False):
    args = list(names)
    args += ['--param-file', LaunchConfiguration('controllers_yaml')]
    if inactive:
        args.append('--inactive')
    return Node(
        package='controller_manager',
        executable='spawner',
        arguments=args,
        output='screen'
    )


def setup_controllers(context):
    robot_type_value = LaunchConfiguration('robot_type').perform(context)
    policy_path_value = LaunchConfiguration('policy_path').perform(context)
    start_step_value = LaunchConfiguration('start_step').perform(context)
    ext_pos_corr = LaunchConfiguration('ext_pos_corr').perform(context)

    kv_pairs = []
    if policy_path_value:
        abs_path = os.path.abspath(os.path.expanduser(os.path.expandvars(policy_path_value)))
        kv_pairs.append(('walking_controller.policy.path', abs_path))
    if start_step_value:
        kv_pairs.append(('walking_controller.motion.start_step', start_step_value))
    if ext_pos_corr.lower() in ["true", "1", "yes"]:
        kv_pairs.append(('state_estimator.estimation.contact.height_sensor_noise', 1e10))
        kv_pairs.append(('state_estimator.estimation.position.topic', "/glim/odom"))

    controllers_config_path = f'config/{robot_type_value}/controllers.yaml'
    temp_controllers_config_path = generate_temp_config(
        controllers_config_path,
        'motion_tracking_controller',
        kv_pairs
    )

    set_controllers_yaml = SetLaunchConfiguration(
        name='controllers_yaml',
        value=temp_controllers_config_path
    )

    active_list = ["state_estimator", "standby_controller"]
    inactive_list = ["walking_controller"]

    active_spawner = control_spawner(active_list)
    inactive_spawner = control_spawner(inactive_list, inactive=True)

    return [set_controllers_yaml, active_spawner, inactive_spawner]


def generate_launch_description():
    robot_type = LaunchConfiguration('robot_type')
    network_interface = LaunchConfiguration('network_interface')
    urdf_name = PythonExpression(["'g1' if '", robot_type, "' == 'g1' else 'sdk1'"])

    robot_description_command = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]),
        " ",
        PathJoinSubstitution([
            FindPackageShare("unitree_description"),
            "urdf",
            urdf_name,
            "robot.xacro"
        ]),
        " ", "robot_type:=", robot_type,
        " ", "simulation:=", "false",
        " ", "network_interface:=", network_interface
    ])

    robot_description = {"robot_description": robot_description_command}

    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {
            'publish_frequency': 500.0,
            'use_sim_time': True
        }],
    )

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, LaunchConfiguration('controllers_yaml')],
        output="both",
        respawn=True,
    )

    wandb = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([ThisLaunchFileDir(), "/wandb.launch.py"]),
        launch_arguments={
            "wandb_path": LaunchConfiguration("wandb_path")
        }.items(),
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration('policy_path'), "' == ''"])
        )
    )

    controllers_opaque_func = OpaqueFunction(function=setup_controllers)

    # Exclude all Unitree topics... it should start from the same namespace, fuck Unitree!
    exclude_regex = (
        r'(/EstimatorData|/SymState(_back)?|/api/.*'
        r'|/arm/action/state|/arm_sdk'
        r'|/audio_msg|/audiosender|/config_change_status'
        r'|/dex3/(left|right)/(cmd|state)'
        r'|/frontvideostream|/gnss'
        r'|/gpt_(cmd|state)|/gptflowfeedback'
        r'|/lf/(bmsstate|dex3/(left|right)/state|lowstate|mainboardstate|'
        r'odommodestate|secondary_imu|sportmodestate)'
        r'|/low(cmd|state)|/multiplestate|/odommodestate'
        r'|/parameter_events|/public_network_status|/rosout'
        r'|/rtc/(state|status)|/secondary_imu|/selftest'
        r'|/servicestate(activate)?|/slam_info|/sportmodestate'
        r'|/utlidar/range_info|/videohub/inner'
        r'|/webrtc(req|res)|/wirelesscontroller)'
    )

    rosbag2 = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record', '-s', 'mcap', '-a',  # record all topics
            '--exclude', exclude_regex,  # skip those that match the regex
        ],
        output='screen',
    )

    return LaunchDescription([
        DeclareLaunchArgument('robot_type', default_value='g1'),
        DeclareLaunchArgument('network_interface'),
        DeclareLaunchArgument(
            'policy_path',
            default_value='',
            description='Absolute or ~-expanded path for walking_controller.policy.path'
        ),
        DeclareLaunchArgument(
            'start_step',
            default_value='0',
            description='Integer start step for walking_controller.motion.start_step'
        ),
        DeclareLaunchArgument(
            'ext_pos_corr',
            default_value='false',
            description='Enable external position correction'
        ),
        wandb,
        controllers_opaque_func,
        control_node,
        node_robot_state_publisher,
        rosbag2,
        Node(
            package='joy_linux',
            executable='joy_linux_node',
            output='both'
        ),
        Node(
            package='joy_teleop',
            executable='joy_teleop',
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare('motion_tracking_controller'),
                    'config',
                    'teleop',
                    LaunchConfiguration('robot_type'),
                    'joy.yaml'
                ])
            ],
            output='both'
        )
    ])

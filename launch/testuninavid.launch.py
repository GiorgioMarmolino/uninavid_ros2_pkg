import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription

from launch.actions import (
    DeclareLaunchArgument, 
    ExecuteProcess, 
    TimerAction, 
    IncludeLaunchDescription
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration, 
    PythonExpression,
    PathJoinSubstitution
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    uninavid_pkg = 'uni_navid'

    # -------------------------------------------------------------------------------
    # Launch Configurations and Expressions
    safety = LaunchConfiguration("safety")
    task   = LaunchConfiguration("task")

    action_cmdvel   = PythonExpression(["'/cmd_vel_raw' if '", safety, "' == 'true' else '/cmd_vel'"])

    config_filename = "test_uninavid.yaml"
    config_file     = PathJoinSubstitution([FindPackageShare(uninavid_pkg), 'config', config_filename])

    return LaunchDescription([
    # -------------------------------------------------------------------------------
    # Declare Launch Arguments and Include Other Launch Files
        DeclareLaunchArgument(
            name =          "safety",
            default_value = "false",
            choices =       ["true", "false"],
            description=    "If true, start safety_layer_node between action node and twist_mux; if false, action node publishes straight to /cmd_vel.",
        ),
        DeclareLaunchArgument(
            name =          "task",
            default_value = "vln",
            choices =       ["vln", "objectnav", "eqa", "following"],
            description =   "Task type for the model",
        ),
        
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('husky_navigation'),
                    'launch',
                    'pointcloud_to_laserscan.launch.py'
                )
            ),
            launch_arguments={
                'use_sim_time': 'false',
                # 'pcd_input': '/sensors/lidar3d_0/points',
                # 'scan_output': 'scan',
                # 'lidar_frame': 'lidar3d_0_laser',
            }.items()
        ),
    # -------------------------------------------------------------------------------
    # ROS2 Nodes
        # Action node: everything from config, except the dynamic output topic.
        Node(
            package     =   uninavid_pkg,
            executable  =   'action_node',
            name        =   'action_node',
            namespace   =   '',
            output =        'screen',
            emulate_tty =   True,
            parameters =    [
                config_file,
                {"cmd_vel_topic": action_cmdvel},
            ],
        ),

        # Safety layer: fully from config. Started only if safety == true.
        Node(
            package     =   uninavid_pkg,
            executable  =   'safety_layer_node',
            name        =   'safety_layer_node',
            namespace   =   '',
            output      =   'screen',
            emulate_tty =   True,
            condition   =   IfCondition(safety),
            parameters  =   [config_file],
        ),

        ExecuteProcess(
            cmd=[
                'xterm', '-e',
                'bash', '-c',
                'source /opt/ros/humble/setup.bash && '
                'source /home/ros_ws/install/setup.bash && '
                'ros2 run uni_navid instruction_node '
                '--ros-args -p use_sim_time:=true; '
                'echo DONE; read'
            ],
            output='screen',
            emulate_tty=True,
        ),

        # Inference Node
        # Start uninavid after 2 secs (params from config). For deterministic ordering you can replace this TimerAction with a RegisterEventHandler(OnProcessStart=...) keyed on a node that signals the sim/pipeline is actually ready.
        TimerAction(
            period=2.0,
            actions=[
                Node(
                    package     = uninavid_pkg,
                    executable  = 'uninavid_node',
                    name        = 'uninavid_node',
                    output      = 'screen',
                    emulate_tty = True,
                    parameters  = [config_file, {"task": task}],
                ),
            ]
        ),
    ])
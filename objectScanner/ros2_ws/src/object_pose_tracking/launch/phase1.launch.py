from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = Path(get_package_share_directory('object_pose_tracking'))
    project = LaunchConfiguration('project_root')
    params = LaunchConfiguration('params_file')
    devices = LaunchConfiguration('device_params_file')
    camera = Node(package='object_pose_tracking', executable='rgbd_camera', output='screen',
                  condition=IfCondition(LaunchConfiguration('camera')),
                  parameters=[params, {'project_root': project,
                                       'camera_python': LaunchConfiguration('camera_python'),
                                       'camera_backend': LaunchConfiguration('camera_backend'),
                                       'serial': ParameterValue(LaunchConfiguration('camera_serial'), value_type=str)}])
    segmentation = Node(package='object_pose_tracking', executable='yolo_segmentation', output='screen',
                        parameters=[params, devices, {'yolo_model_path': LaunchConfiguration('weights')}])
    cloud = Node(package='object_pose_tracking', executable='object_cloud', output='screen', parameters=[params])
    # Fail as a group instead of leaving a partial pipeline that appears alive.
    exits = [RegisterEventHandler(OnProcessExit(target_action=node,
             on_exit=[EmitEvent(event=Shutdown(reason='Phase 1 node exited'))]))
             for node in (camera, segmentation, cloud)]
    return LaunchDescription([
        DeclareLaunchArgument('project_root', description='Absolute path to objectScanner'),
        DeclareLaunchArgument('camera_backend', default_value='orbbec',
                              choices=['orbbec', 'realsense', 'auto']),
        DeclareLaunchArgument('camera_serial', default_value='',
                              description='Camera serial; empty selects a single matching device'),
        DeclareLaunchArgument('camera_python', default_value=PathJoinSubstitution([project, '.venv', 'bin', 'python'])),
        DeclareLaunchArgument('weights', default_value=PathJoinSubstitution([project, 'weights', 'best.pt'])),
        DeclareLaunchArgument('params_file', default_value=str(share / 'config/phase1.yaml')),
        DeclareLaunchArgument('device_params_file', default_value=str(share / 'config/devices_default.yaml')),
        DeclareLaunchArgument('camera', default_value='true', description='false for external aligned RGB-D topics'),
        DeclareLaunchArgument('rviz', default_value='false'),
        *exits, camera, segmentation, cloud,
        Node(package='rviz2', executable='rviz2', arguments=['-d', str(share / 'rviz/phase1.rviz')],
             condition=IfCondition(LaunchConfiguration('rviz')), output='screen',
             additional_env={'LIBGL_ALWAYS_SOFTWARE': '1', 'QT_QUICK_BACKEND': 'software'}),
    ])

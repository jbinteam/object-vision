"""Opt-in CUDA segmentation and ICP, with the standard Phase 3 ROS/RViz outputs."""
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    share = Path(get_package_share_directory('object_pose_tracking'))
    return LaunchDescription([
        DeclareLaunchArgument('project_root', description='Absolute path to objectScanner'),
        DeclareLaunchArgument('camera_backend', default_value='orbbec',
                              choices=['orbbec', 'realsense', 'auto']),
        DeclareLaunchArgument('camera_serial', default_value='',
                              description='Camera serial; empty selects a single matching device'),
        DeclareLaunchArgument('device_params_file', default_value=str(share / 'config/devices_cuda.yaml')),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(share / 'launch/phase3.launch.py')),
                                 launch_arguments={'project_root': LaunchConfiguration('project_root'),
                                                   'camera_backend': LaunchConfiguration('camera_backend'),
                                                   'camera_serial': LaunchConfiguration('camera_serial'),
                                                   'device_params_file': LaunchConfiguration('device_params_file')}.items()),
    ])

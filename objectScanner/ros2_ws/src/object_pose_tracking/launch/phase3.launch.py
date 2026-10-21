"""Phase 3 reuses the complete Phase 2 pipeline and RViz frame/overlay view."""
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
        DeclareLaunchArgument('icp_params_file', default_value=str(share / 'config/phase3.yaml')),
        DeclareLaunchArgument('device_params_file', default_value=str(share / 'config/devices_default.yaml')),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(share / 'launch/phase2.launch.py')),
                                 launch_arguments={'project_root': LaunchConfiguration('project_root'),
                                                   'camera_backend': LaunchConfiguration('camera_backend'),
                                                   'camera_serial': LaunchConfiguration('camera_serial'),
                                                   'icp_params_file': LaunchConfiguration('icp_params_file'),
                                                   'device_params_file': LaunchConfiguration('device_params_file')}.items()),
    ])

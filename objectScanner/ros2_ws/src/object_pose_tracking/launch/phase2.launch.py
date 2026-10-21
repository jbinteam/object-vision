from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, GroupAction, IncludeLaunchDescription, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory('object_pose_tracking'))
    project = LaunchConfiguration('project_root')
    icp = Node(package='object_pose_tracking', executable='icp_pose', output='screen',
               parameters=[str(share / 'config/phase2.yaml'), LaunchConfiguration('icp_params_file'),
                           LaunchConfiguration('device_params_file'),
                           {'reference_cloud_path': LaunchConfiguration('reference')}],
               additional_env={'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'})
    return LaunchDescription([
        DeclareLaunchArgument('project_root', description='Absolute path to objectScanner'),
        DeclareLaunchArgument('camera_backend', default_value='orbbec',
                              choices=['orbbec', 'realsense', 'auto']),
        DeclareLaunchArgument('camera_serial', default_value='',
                              description='Camera serial; empty selects a single matching device'),
        DeclareLaunchArgument('camera_python', default_value=PathJoinSubstitution([project, '.venv', 'bin', 'python'])),
        DeclareLaunchArgument('weights', default_value=PathJoinSubstitution([project, 'weights', 'best.pt'])),
        DeclareLaunchArgument('reference', default_value=PathJoinSubstitution([project, 'references', 'emergency_stop_ref', 'reference.ply'])),
        DeclareLaunchArgument('params_file', default_value=str(share / 'config/phase1.yaml')),
        DeclareLaunchArgument('icp_params_file', default_value=str(share / 'config/phase2.yaml')),
        DeclareLaunchArgument('device_params_file', default_value=str(share / 'config/devices_default.yaml')),
        DeclareLaunchArgument('camera', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='true'),
        RegisterEventHandler(OnProcessExit(target_action=icp,
            on_exit=[EmitEvent(event=Shutdown(reason='ICP node exited'))])),
        GroupAction(actions=[IncludeLaunchDescription(PythonLaunchDescriptionSource(str(share / 'launch/phase1.launch.py')),
                                 launch_arguments={
                                     'project_root': project,
                                     'camera_python': LaunchConfiguration('camera_python'),
                                     'camera_backend': LaunchConfiguration('camera_backend'),
                                     'camera_serial': LaunchConfiguration('camera_serial'),
                                     'weights': LaunchConfiguration('weights'),
                                     'params_file': LaunchConfiguration('params_file'),
                                     'device_params_file': LaunchConfiguration('device_params_file'),
                                     'camera': LaunchConfiguration('camera'), 'rviz': 'false',
                                 }.items())]),
        icp,
        Node(package='rviz2', executable='rviz2', arguments=['-d', str(share / 'rviz/phase2.rviz')],
             condition=IfCondition(LaunchConfiguration('rviz')), output='screen',
             additional_env={'LIBGL_ALWAYS_SOFTWARE': '1', 'QT_QUICK_BACKEND': 'software'}),
    ])

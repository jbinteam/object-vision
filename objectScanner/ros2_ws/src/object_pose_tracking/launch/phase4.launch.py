"""Complete paired YOLO segmentation/pose -> PnP -> GICP tracking pipeline."""
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
                                       'serial': ParameterValue(LaunchConfiguration('camera_serial'), value_type=str),
                                       'publish_rate': ParameterValue(
                                           LaunchConfiguration('camera_rate'), value_type=float)}])
    observation = Node(
        package='object_pose_tracking', executable='yolo_observation', output='screen',
        parameters=[params, devices,
                    {'segmentation_model_path': LaunchConfiguration('segmentation_weights'),
                     'pose_model_path': LaunchConfiguration('pose_weights')}])
    cloud = Node(package='object_pose_tracking', executable='object_cloud', output='screen',
                 parameters=[params])
    pose = Node(
        package='object_pose_tracking', executable='pnp_gicp_pose', output='screen',
        parameters=[params,
                    {'reference_cloud_path': LaunchConfiguration('reference'),
                     'object_keypoints_path': LaunchConfiguration('keypoints')}],
        additional_env={'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'})
    exits = [RegisterEventHandler(OnProcessExit(target_action=node,
             on_exit=[EmitEvent(event=Shutdown(reason='Phase 4 node exited'))]))
             for node in (camera, observation, cloud, pose)]
    return LaunchDescription([
        DeclareLaunchArgument('project_root', description='Absolute path to objectScanner'),
        DeclareLaunchArgument('camera_backend', default_value='orbbec',
                              choices=['orbbec', 'realsense', 'auto']),
        DeclareLaunchArgument('camera_serial', default_value='',
                              description='Camera serial; empty selects a single matching device'),
        DeclareLaunchArgument('camera_python', default_value=PathJoinSubstitution(
            [project, '.venv', 'bin', 'python'])),
        DeclareLaunchArgument('camera_rate', default_value='20.0',
                              description='Maximum camera publication rate in Hz (camera captures at 30 FPS)'),
        DeclareLaunchArgument('segmentation_weights', default_value=PathJoinSubstitution(
            [project, 'training/yolo/my_object_fresh_onthefly_seg/weights/best.pt'])),
        DeclareLaunchArgument('pose_weights', default_value=PathJoinSubstitution(
            [project, 'training/yolo/my_object_fresh_pose_refined_pose/weights/best.pt'])),
        DeclareLaunchArgument('reference', default_value=PathJoinSubstitution(
            [project, 'datasets/my_object_fresh/references/my_object/reference_aligned.ply'])),
        DeclareLaunchArgument('keypoints', default_value=PathJoinSubstitution(
            [project, 'datasets/my_object_fresh/yolo_seg_pose_onthefly/object_keypoints.yaml'])),
        DeclareLaunchArgument('params_file', default_value=str(share / 'config/phase4.yaml')),
        DeclareLaunchArgument('device_params_file', default_value=str(share / 'config/devices_default.yaml')),
        DeclareLaunchArgument('camera', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='true'),
        *exits, camera, observation, cloud, pose,
        Node(package='rviz2', executable='rviz2',
             arguments=['-d', str(share / 'rviz/phase4.rviz')],
             condition=IfCondition(LaunchConfiguration('rviz')), output='screen',
             additional_env={'LIBGL_ALWAYS_SOFTWARE': '1', 'QT_QUICK_BACKEND': 'software'}),
    ])

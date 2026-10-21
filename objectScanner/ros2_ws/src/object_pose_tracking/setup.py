from glob import glob
from setuptools import find_packages, setup

setup(
    name='object_pose_tracking', version='0.1.0', packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/object_pose_tracking']),
        ('share/object_pose_tracking', ['package.xml']),
        ('share/object_pose_tracking/launch', glob('launch/*.launch.py')),
        ('share/object_pose_tracking/config', glob('config/*.yaml')),
        ('share/object_pose_tracking/rviz', glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'], tests_require=['pytest'], zip_safe=True,
    maintainer='Object Scanner maintainers', maintainer_email='maintainers@example.com',
    description='RGB-D segmentation and fixed-reference object pose estimation',
    license='Apache-2.0',
    entry_points={'console_scripts': [
        'rgbd_camera = object_pose_tracking.camera_node:main',
        'yolo_segmentation = object_pose_tracking.segmentation_node:main',
        'yolo_observation = object_pose_tracking.observation_node:main',
        'object_cloud = object_pose_tracking.cloud_node:main',
        'icp_pose = object_pose_tracking.pose_node:main',
        'pnp_gicp_pose = object_pose_tracking.pnp_gicp_node:main',
    ]},
)

"""Actual ROS pose/TF/marker transport with deterministic synthetic registration."""
import time
import numpy as np
import open3d as o3d
import pytest
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import MarkerArray, Marker
from scipy.spatial.transform import Rotation
from object_pose_tracking.pose_node import PoseNode
from object_pose_tracking.pose_outputs import pose_and_axes
from object_pose_tracking.registration import motion_delta
from object_pose_tracking.ros_support import sensor_qos


@pytest.fixture
def pose_pipeline(tmp_path):
    rng=np.random.default_rng(98)
    points=rng.uniform([-.07,-.045,-.025],[.06,.035,.045],(500,3))
    path=tmp_path/'reference.ply'
    assert o3d.io.write_point_cloud(str(path),o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points)))
    rclpy.init(args=['--ros-args','-p',f'reference_cloud_path:={path}', '-p','icp_type:=point_to_point',
                    '-p','voxel_size:=0.0','-p','icp_time_budget:=2.0','-p','maximum_data_age:=2.0'])
    node=PoseNode();probe=Node('pose_test_probe');executor=SingleThreadedExecutor()
    executor.add_node(node);executor.add_node(probe)
    outputs={key:[] for key in ('pose','tf','axes','overlay')}
    subscriptions=[probe.create_subscription(kind,topic,outputs[key].append,sensor_qos(10)) for kind,topic,key in
                   [(PoseStamped,'/object/pose','pose'),(TFMessage,'/tf','tf'),
                    (MarkerArray,'/object/frame_axes','axes'),(PointCloud2,'/object/reference_cloud_transformed','overlay')]]
    def spin(seconds=.1):
        until=time.monotonic()+seconds
        while time.monotonic()<until:executor.spin_once(timeout_sec=.005)
    deadline=time.monotonic()+3
    while any(s.get_publisher_count()==0 for s in subscriptions) and time.monotonic()<deadline:spin(.02)
    assert all(s.get_publisher_count()>0 for s in subscriptions)
    expected=np.eye(4);expected[:3,:3]=Rotation.from_euler('xyz',[3,-4,6],degrees=True).as_matrix();expected[:3,3]=[.1,-.03,.7]
    def observation(empty=False):
        header=Header(stamp=probe.get_clock().now().to_msg(),frame_id=node.camera_frame)
        xyz=points@expected[:3,:3].T+expected[:3,3]
        return point_cloud2.create_cloud_xyz32(header,xyz.astype(np.float32) if not empty else np.empty((0,3),np.float32))
    yield node,outputs,spin,observation,expected
    executor.shutdown();node.destroy_node();probe.destroy_node();rclpy.shutdown()


def test_pose_tf_axes_overlay_share_transform_and_measurement(pose_pipeline):
    node,out,spin,observation,expected=pose_pipeline
    msg=observation();node.observe(msg);spin()
    assert len(out['pose'])==len(out['tf'])==len(out['axes'])==len(out['overlay'])==1
    pose=out['pose'][0];tf=out['tf'][0].transforms[0]
    assert pose.header==tf.header==msg.header
    assert tf.child_frame_id=='object'
    assert pose.pose.orientation==tf.transform.rotation
    assert pose.pose.position.x==tf.transform.translation.x
    q=pose.pose.orientation;p=pose.pose.position
    actual=np.eye(4);actual[:3,:3]=Rotation.from_quat([q.x,q.y,q.z,q.w]).as_matrix();actual[:3,3]=[p.x,p.y,p.z]
    errors=motion_delta(expected,actual)
    assert errors[0]<1e-5 and errors[1]<1e-4,errors
    for i,marker in enumerate(out['axes'][0].markers[:3]):
        assert marker.header==msg.header and marker.pose==pose.pose
        assert marker.type==Marker.ARROW and marker.action==Marker.ADD
        end=marker.points[1]
        np.testing.assert_allclose([end.x,end.y,end.z],np.eye(3)[i]*node.axis_length)
        assert marker.lifetime.sec>0 or marker.lifetime.nanosec>0
    overlay=point_cloud2.read_points_numpy(out['overlay'][0])
    np.testing.assert_allclose(overlay,node.registration.reference_points@actual[:3,:3].T+actual[:3,3],atol=1e-6)


def test_invalid_observation_clears_visuals_without_pose_or_tf(pose_pipeline):
    node,out,spin,observation,_=pose_pipeline
    node.observe(observation());spin()
    node.observe(observation(empty=True));spin()
    assert len(out['pose'])==len(out['tf'])==1
    assert all(marker.action==Marker.DELETE for marker in out['axes'][-1].markers)
    assert out['overlay'][-1].width==0 and node.accepted_stamp is None


def test_cuda_native_failure_clears_output_and_propagates(pose_pipeline,monkeypatch):
    from dataclasses import replace
    node,out,spin,observation,_=pose_pipeline
    node.observe(observation());spin()
    node.config=replace(node.config,icp_backend='cuda')
    def fail(*args,**kwargs):
        raise RuntimeError('CUDA test failure')
    monkeypatch.setattr(node.registration,'register',fail)
    with pytest.raises(RuntimeError,match='CUDA test failure'):
        node.observe(observation())
    spin()
    assert len(out['pose'])==len(out['tf'])==1
    assert out['axes'][-1].markers[0].action==Marker.DELETE
    assert out['overlay'][-1].width==0


def test_timeout_does_not_refresh_pose_and_next_cloud_reinitializes(pose_pipeline):
    node,out,spin,observation,_=pose_pipeline
    node.observe(observation());spin()
    node.accepted_stamp=node.get_clock().now().nanoseconds-int(3e9)
    node.check_timeout();spin()
    assert len(out['pose'])==len(out['tf'])==1
    assert out['axes'][-1].markers[0].action==Marker.DELETE
    node.observe(observation());spin()
    assert len(out['pose'])==len(out['tf'])==2


@pytest.mark.parametrize('bad', ['frame','stale'])
def test_wrong_frame_or_stale_observation_has_no_pose(pose_pipeline,bad):
    node,out,spin,observation,_=pose_pipeline
    msg=observation()
    if bad=='frame':msg.header.frame_id='unknown_frame'
    else:msg.header.stamp.sec-=5
    node.observe(msg);spin()
    assert not out['pose'] and not out['tf'] and not out['axes']


def test_axes_reject_nonrigid_transform():
    transform=np.eye(4);transform[0,0]=2
    with pytest.raises(ValueError):pose_and_axes(Header(),transform,'object',.06,.5)


def test_tracking_reuses_raw_pose_and_rejected_candidate_does_not_poison_seed(pose_pipeline,monkeypatch):
    from object_pose_tracking.registration import RegistrationResult
    node,out,spin,observation,_=pose_pipeline
    node.tracking_enabled=True
    actual_register=node.registration.register
    seeds=[]
    def register(points,rotation=None,initial_guess=None):
        seeds.append(None if initial_guess is None else initial_guess.copy())
        return actual_register(points,rotation=rotation,initial_guess=initial_guess)
    monkeypatch.setattr(node.registration,'register',register)
    first=observation();node.observe(first);spin()
    assert seeds[0] is None and len(out['pose'])==1
    saved=node.tracking_seed.initial_guess(first.header.stamp.sec*10**9+first.header.stamp.nanosec+1,
                                          node.get_clock().now().nanoseconds)
    node.observe(observation());spin()
    np.testing.assert_array_equal(seeds[1],saved)
    current_stamp=node.tracking_seed.stamp_ns
    current=node.tracking_seed.initial_guess(current_stamp+1,node.get_clock().now().nanoseconds)
    poison=np.eye(4);poison[:3,3]=[9,9,9]
    monkeypatch.setattr(node.registration,'register',lambda *a,**k:RegistrationResult(poison,reason='low_fitness'))
    node.observe(observation());spin()
    assert len(out['pose'])==2 and node.tracking_seed.stamp_ns==current_stamp
    assert out['axes'][-1].markers[0].action==Marker.DELETE
    monkeypatch.setattr(node.registration,'register',register)
    node.observe(observation());spin()
    np.testing.assert_array_equal(seeds[-1],current)
    assert len(out['pose'])==3


def test_tracking_retains_seed_for_empty_input_but_expires_without_refresh(pose_pipeline,monkeypatch):
    node,out,spin,observation,_=pose_pipeline
    node.tracking_enabled=True
    node.observe(observation());spin()
    saved_stamp=node.tracking_seed.stamp_ns
    node.observe(observation(empty=True));spin()
    assert node.tracking_seed.stamp_ns==saved_stamp
    assert len(out['pose'])==1
    node.tracking_seed.stamp_ns=node.get_clock().now().nanoseconds-int(3e9)
    node.check_timeout();spin()
    assert node.tracking_seed.stamp_ns is None and len(out['pose'])==1
    actual_register=node.registration.register;seeds=[]
    def record(points,rotation=None,initial_guess=None):
        seeds.append(initial_guess)
        return actual_register(points,rotation=rotation,initial_guess=initial_guess)
    monkeypatch.setattr(node.registration,'register',record)
    node.observe(observation());spin()
    assert seeds==[None] and len(out['pose'])==2


def test_phase2_continues_independent_initialization(pose_pipeline,monkeypatch):
    node,out,spin,observation,_=pose_pipeline
    assert not node.tracking_enabled
    actual_register=node.registration.register;seeds=[]
    def record(points,rotation=None,initial_guess=None):
        seeds.append(initial_guess)
        return actual_register(points,rotation=rotation,initial_guess=initial_guess)
    monkeypatch.setattr(node.registration,'register',record)
    node.observe(observation());spin()
    node.observe(observation());spin()
    assert seeds==[None,None] and node.tracking_seed.stamp_ns is None


def test_tracking_expired_result_never_replaces_valid_seed(pose_pipeline,monkeypatch):
    node,out,spin,observation,_=pose_pipeline
    node.tracking_enabled=True
    node.observe(observation());spin()
    stamp=node.tracking_seed.stamp_ns
    previous=node.tracking_seed.initial_guess(stamp+1,node.get_clock().now().nanoseconds)
    ages=iter([0.0,3.0])
    monkeypatch.setattr(node,'age',lambda header:next(ages))
    node.observe(observation());spin()
    assert len(out['pose'])==1 and node.tracking_seed.stamp_ns==stamp
    np.testing.assert_array_equal(node.tracking_seed.initial_guess(stamp+1,node.get_clock().now().nanoseconds),previous)

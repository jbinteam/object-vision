import io
import struct
import numpy as np
import pytest
from object_pose_tracking.geometry import CloudConfig, backproject, preprocess, transform_points
from object_pose_tracking.protocol import MAX_PACKET, read_frame, write_frame
from object_pose_tracking.ros_support import MovingAverageMetrics
from object_pose_tracking.segmentation import (Detection, InstanceSelector, PoseDetection,
                                               associate_pose)


def config(**kwargs):
    return CloudConfig(voxel_size=0, outlier_neighbors=0, **kwargs)


def test_backprojection_metres_and_optical_axes():
    depth = np.array([[1000, 1000], [1000, 1000]], dtype=np.uint16)
    k = np.array([[2., 0, 0], [0, 4., 0], [0, 0, 1]])
    points = backproject(depth, np.ones((2, 2)), k, config())
    np.testing.assert_allclose(points, [[0, 0, 1], [.5, 0, 1], [0, .25, 1], [.5, .25, 1]])
    np.testing.assert_allclose(backproject(depth.astype(np.float32) / 1000,
                                          np.ones((2, 2)), k, config(), '32FC1'), points)


def test_depth_invalid_range_and_mask():
    depth = np.array([[0, np.nan, np.inf, -1, .1, 1, 2, 1]], dtype=np.float32)
    mask = np.array([[1, 1, 1, 1, 1, 1, 1, 0]])
    points = backproject(depth, mask, np.eye(3), config(), '32FC1')
    np.testing.assert_allclose(points, [[5, 0, 1]])


@pytest.mark.parametrize('depth,mask,k,encoding', [
    (np.ones((2,2),np.uint16), np.ones((3,2)), np.eye(3), '16UC1'),
    (np.ones((2,2),np.uint16), np.ones((2,2)), np.zeros((3,3)), '16UC1'),
    (np.ones((2,2),np.uint16), np.ones((2,2)), np.eye(3), '32FC1'),
    (np.ones((2,2),np.uint16), np.ones((2,2)), np.eye(3), '8UC1'),
])
def test_reject_bad_geometry_inputs(depth, mask, k, encoding):
    with pytest.raises(ValueError):
        backproject(depth, mask, k, config(), encoding)


@pytest.mark.parametrize('kwargs', [dict(minimum_depth=-1), dict(maximum_depth=.1),
                                  dict(voxel_size=float('nan')), dict(minimum_points=2),
                                  dict(outlier_std_ratio=0), dict(outlier_neighbors=-1)])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        CloudConfig(**kwargs)


def test_voxel_centroids_and_input_immutability():
    points = np.array([[.001,0,1], [.002,0,1], [.02,0,1]],np.float32)
    original = points.copy()
    output = preprocess(points, CloudConfig(voxel_size=.01, outlier_neighbors=0))
    np.testing.assert_allclose(output, [[.0015,0,1], [.02,0,1]])
    np.testing.assert_array_equal(points, original)


def test_outlier_removal_and_empty():
    rng = np.random.default_rng(42)
    points = np.vstack((rng.normal(0,.01,(100,3)), [10,10,10], [np.nan,0,1]))
    output = preprocess(points, CloudConfig(voxel_size=0))
    assert len(output) == 100
    assert preprocess(np.empty((0,3)), CloudConfig()).shape == (0,3)


def test_calibrated_transform_direction():
    # Optical forward -> body forward; optical right -> body right (-Y).
    output = transform_points([[0,0,1], [1,0,0]], [1,2,3], [-.5,.5,-.5,.5])
    np.testing.assert_allclose(output, [[2,2,3], [1,1,3]], atol=1e-6)
    with pytest.raises(ValueError):
        transform_points([[1,2,3]], [0,0,0], [0,0,0,0])


def detection(box, confidence):
    return Detection(np.ones((4,4),bool), np.array(box), confidence)


def test_instance_association_does_not_follow_confidence_switch():
    selector = InstanceSelector(reset_after_misses=2)
    a, b = detection([0,0,10,10], .9), detection([50,50,60,60], .8)
    assert selector.select([a,b])[0] is a
    b.confidence = .99
    assert selector.select([a,b])[0] is a
    assert selector.select([b]) == (None,'no_matching_instance')
    assert selector.select([b]) == (None,'association_lost')
    assert selector.select([b]) == (b,'acquired')
    selector.reset()
    assert selector.box is None


def test_pose_is_paired_by_box_not_independent_confidence():
    segment = detection([0, 0, 10, 10], .9)
    near = PoseDetection(np.array([1, 1, 9, 9]), .5, np.ones((8, 4)))
    far = PoseDetection(np.array([50, 50, 60, 60]), .99, np.ones((8, 4)))
    assert associate_pose(segment, [far, near]) is near
    assert associate_pose(segment, [far]) is None


def test_protocol_roundtrip_and_bounds():
    stream = io.BytesIO()
    rgb = np.ones((3,4,3),np.uint8)
    depth = np.ones((3,4),np.uint16)
    write_frame(stream, rgb, depth, np.eye(3), 123456789)
    stream.seek(0)
    output = read_frame(stream)
    np.testing.assert_array_equal(output[0], rgb)
    np.testing.assert_array_equal(output[1], depth)
    assert output[3] == 123456789
    with pytest.raises(ValueError):
        read_frame(io.BytesIO(struct.pack('!I', MAX_PACKET + 1)))
    with pytest.raises(EOFError):
        read_frame(io.BytesIO(b'\x00'))


def test_moving_average_metrics():
    metrics = MovingAverageMetrics(.5)
    assert metrics.update(stage_ms=10)['stage_ms_average'] == 10
    assert metrics.update(stage_ms=20)['stage_ms_average'] == 15


def test_backend_preserves_pixel_coordinates_and_converts_rgb_to_bgr():
    from types import SimpleNamespace
    from object_pose_tracking.segmentation import YoloSegmenter
    calls = []
    class Tensor:
        def __init__(self, data): self.data = np.asarray(data)
        def cpu(self): return self
        def numpy(self): return self.data
    class Model:
        def predict(self, image, **kwargs):
            calls.append((image, kwargs))
            return [SimpleNamespace(masks=SimpleNamespace(data=Tensor([[[0,1], [0,0]]])),
                                    boxes=SimpleNamespace(xyxy=Tensor([[1,0,2,1]]), conf=Tensor([.9])))]
    backend = YoloSegmenter.__new__(YoloSegmenter)
    backend.model, backend.class_id = Model(), 0
    backend.confidence, backend.device, backend.image_size = .5, 'cpu', 640
    backend.half = False
    rgb = np.full((2,2,3), [10,20,30],np.uint8)
    output = backend.predict(rgb)
    np.testing.assert_array_equal(calls[0][0][0,0], [30,20,10])
    assert calls[0][1]['retina_masks'] is True
    np.testing.assert_array_equal(output[0].mask, [[False,True],[False,False]])


def test_segmentation_cuda_unavailable_and_precision_validation(tmp_path,monkeypatch):
    import torch
    from object_pose_tracking.segmentation import YoloSegmenter
    model=tmp_path/'placeholder.pt';model.touch()
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CUDA:0 is unavailable'):
        YoloSegmenter(model,device='cuda:0')
    with pytest.raises(ValueError,match='yolo_device'):
        YoloSegmenter(model,device='auto')
    with pytest.raises(ValueError,match='requires a CUDA'):
        YoloSegmenter(model,device='cpu',half=True)


def test_backprojection_rgb_uses_only_valid_masked_pixels():
    depth = np.array([[1000,0], [1000,1000]], np.uint16)
    mask = np.array([[1,1], [0,1]], np.uint8)
    rgb = np.array([[[255,0,0],[0,255,0]], [[0,0,255],[12,34,56]]],np.uint8)
    points, colors = backproject(depth, mask, np.eye(3), config(), rgb=rgb)
    np.testing.assert_array_equal(colors, [[255,0,0],[12,34,56]])
    np.testing.assert_allclose(points, [[0,0,1],[1,1,1]])
    with pytest.raises(ValueError):
        backproject(depth, mask, np.eye(3), config(), rgb=rgb[:1])


def test_voxel_colors_are_averaged_and_outliers_keep_correspondence():
    points = np.array([[.001,0,1],[.002,0,1],[.02,0,1],[np.nan,0,1]],np.float32)
    rgb = np.array([[255,0,0],[0,0,255],[20,40,60],[1,2,3]],np.uint8)
    original = rgb.copy()
    output, colors = preprocess(points, CloudConfig(voxel_size=.01, outlier_neighbors=0), rgb)
    np.testing.assert_allclose(output, [[.0015,0,1],[.02,0,1]])
    np.testing.assert_array_equal(colors, [[128,0,128],[20,40,60]])
    np.testing.assert_array_equal(rgb, original)
    points = np.column_stack((np.linspace(0,.1,30),np.zeros(30),np.ones(30)))
    points = np.vstack((points,[10,10,10]))
    colors = np.tile(np.array([12,34,56],np.uint8),(31,1))
    colors[-1] = [255,0,0]
    output, colors = preprocess(points, CloudConfig(voxel_size=0), colors)
    assert len(output) == 30
    np.testing.assert_array_equal(colors, np.tile([12,34,56],(30,1)))


def test_rgb_message_packing_and_empty_schema():
    from std_msgs.msg import Header
    from sensor_msgs.msg import PointField
    from sensor_msgs_py import point_cloud2
    from object_pose_tracking.cloud_messages import create_colored_cloud
    points = np.array([[0,0,1],[1,0,1],[0,1,1]],np.float32)
    colors = np.array([[255,0,0],[0,255,0],[0,0,255]],np.uint8)
    msg = create_colored_cloud(Header(frame_id='optical'), points, colors)
    decoded = point_cloud2.read_points(msg)
    np.testing.assert_array_equal(decoded['rgb'], [0xFF0000,0x00FF00,0x0000FF])
    assert msg.point_step == 16 and msg.fields[-1].datatype == PointField.UINT32
    empty = create_colored_cloud(Header(), np.empty((0,3)), np.empty((0,3),np.uint8))
    assert empty.width == 0 and empty.fields == msg.fields

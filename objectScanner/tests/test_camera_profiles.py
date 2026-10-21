import unittest
from unittest.mock import Mock, patch

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None
from camera import configure_streams, warm_up


def profile(stream, width, height, fmt):
    p = Mock()
    p.fps.return_value = 30
    p.stream_type.return_value = stream
    p.format.return_value = fmt
    p.as_video_stream_profile.return_value = Mock(width=lambda: width, height=lambda: height)
    return p


@unittest.skipIf(rs is None, "optional RealSense SDK is not installed")
class ProfileTests(unittest.TestCase):
    def device(self, depths, colors):
        profiles = [profile(rs.stream.depth, w, h, rs.format.z16) for w, h in depths]
        profiles += [profile(rs.stream.color, w, h, rs.format.rgb8) for w, h in colors]
        return Mock(sensors=[Mock(get_stream_profiles=lambda: profiles)])

    def test_usb2_uses_advertised_qvga_depth(self):
        device = self.device([(320, 240)], [(640, 480)])
        config = Mock()
        config.can_resolve.return_value = True
        with patch("camera.rs.config", return_value=config), patch("camera.rs.pipeline_wrapper"):
            _, depth, color = configure_streams(device, Mock(), "test")
        self.assertEqual(depth, (320, 240))
        self.assertEqual(color, (640, 480))
        config.enable_stream.assert_any_call(rs.stream.depth, 320, 240, rs.format.z16, 30)

    def test_usb3_prefers_vga_then_checks_pair_compatibility(self):
        device = self.device([(320, 240), (640, 480), (1024, 768)], [(960, 540), (1280, 720)])
        first, second = Mock(), Mock()
        first.can_resolve.return_value = False
        second.can_resolve.return_value = True
        with patch("camera.rs.config", side_effect=[first, second]), patch("camera.rs.pipeline_wrapper"):
            result, depth, color = configure_streams(device, Mock(), "test")
        self.assertIs(result, second)
        self.assertEqual(depth, (640, 480))
        self.assertEqual(color, (1280, 720))

    def test_no_joint_configuration_reports_clear_error(self):
        device = self.device([(320, 240)], [(640, 480)])
        config = Mock()
        config.can_resolve.return_value = False
        with patch("camera.rs.config", return_value=config), patch("camera.rs.pipeline_wrapper"):
            with self.assertRaisesRegex(RuntimeError, "No compatible RGB-D stream pair"):
                configure_streams(device, Mock(), "test")


class WarmUpTests(unittest.TestCase):
    def test_delayed_and_incomplete_frames_are_retried(self):
        pipeline = Mock()
        pair = Mock()
        incomplete = Mock()
        incomplete.get_depth_frame.return_value = None
        pipeline.try_wait_for_frames.side_effect = [(False, None), (True, incomplete),
                                                     (True, pair), (True, pair)]
        with patch("camera.time.monotonic", return_value=0):
            warm_up(pipeline, frame_count=2)
        self.assertEqual(pipeline.try_wait_for_frames.call_count, 4)

    def test_absent_frames_have_a_total_deadline(self):
        pipeline = Mock()
        pipeline.try_wait_for_frames.return_value = (False, None)
        with patch("camera.time.monotonic", side_effect=[0, 0, 11]):
            with self.assertRaisesRegex(RuntimeError, "0/15.*within 10s"):
                warm_up(pipeline)
        pipeline.try_wait_for_frames.assert_called_once_with(1000)

    def test_unexpected_sdk_error_is_not_hidden(self):
        pipeline = Mock()
        pipeline.try_wait_for_frames.side_effect = RuntimeError("Device disconnected")
        with self.assertRaisesRegex(RuntimeError, "Device disconnected"):
            warm_up(pipeline)


class ReceiptTimestampTests(unittest.TestCase):
    def test_rgbd_keeps_receipt_timestamp_before_alignment(self):
        import numpy as np
        from camera import L515
        camera = L515.__new__(L515)
        camera.pipeline = Mock()
        frames = Mock()
        frames.get_color_frame.return_value.get_data.return_value = np.zeros((4, 4, 3), np.uint8)
        frames.get_depth_frame.return_value.get_data.return_value = np.full((4, 4), 1000, np.uint16)
        camera.align = Mock()
        camera.align.process.return_value = frames
        camera.maps = None
        camera.intrinsic = np.eye(3)
        camera.scale = .001
        with patch('camera.time.time_ns', return_value=123456789):
            frame = camera.read(.1, 2., 1., 1)
        self.assertEqual(frame.timestamp_ns, 123456789)
        np.testing.assert_array_equal(frame.depth, np.full((4, 4), 1000, np.uint16))

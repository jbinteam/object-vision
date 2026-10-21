"""Gemini 305 geometry and byte conversions without a camera or vendor SDK."""
from types import SimpleNamespace
import importlib
import importlib.util
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from orbbec_camera import (Gemini305, color_rgb, configure_closest_range, depth_raw,
                           stream_candidates)


def fake_sdk():
    return SimpleNamespace(
        OBFormat=SimpleNamespace(**{name: name for name in
            ("RGB", "BGR", "MJPG", "YUYV", "YUY2", "UYVY", "Y16", "Z16", "H264")}),
        OBSensorType=SimpleNamespace(DEPTH_SENSOR="depth", COLOR_SENSOR="color"),
        OBStatus=SimpleNamespace(STATUS_OK="ok"),
        OBPermissionType=SimpleNamespace(PERMISSION_READ_WRITE="read_write"),
        OBPropertyID=SimpleNamespace(OB_PROP_DISP_SEARCH_RANGE_MODE_INT="disparity_range",
                                    OB_PROP_DISP_SEARCH_OFFSET_INT="disparity_offset"),
        OBAlignMode=SimpleNamespace(DISABLE="disabled"),
        OBFrameAggregateOutputMode=SimpleNamespace(FULL_FRAME_REQUIRE="full"),
        OBCameraDistortionModel=SimpleNamespace(BROWN_CONRADY="brown", BROWN_CONRADY_K6="k6"),
        Config=Mock(side_effect=lambda: Mock()))


def close_range_device():
    device = Mock()
    modes = [SimpleNamespace(name="Default"), SimpleNamespace(name="Close Range Default")]
    mode_list = device.get_depth_work_mode_list.return_value
    mode_list.get_count.return_value = len(modes)
    mode_list.get_depth_work_mode_by_index.side_effect = modes.__getitem__
    device.get_depth_work_mode.return_value = SimpleNamespace(name="Close Range Default")
    device.set_depth_work_mode.return_value = "ok"
    device.is_property_supported.return_value = True
    device.get_int_property_range.return_value = SimpleNamespace(min=0, max=2)
    values = dict(disparity_range=2, disparity_offset=0)
    device.get_int_property.side_effect = values.__getitem__
    device.set_int_property.side_effect = values.__setitem__
    return device


class ClosestRangeTests(unittest.TestCase):
    def test_selects_advertised_mode_before_configuring_and_reads_back_properties(self):
        device = close_range_device()
        device.get_depth_work_mode.side_effect = [SimpleNamespace(name="Default"),
                                                  SimpleNamespace(name="Close Range Default")]
        device.get_int_property.side_effect = [1, 2, 3, 0]
        info = configure_closest_range(device, fake_sdk())
        mode = device.get_depth_work_mode_list.return_value.get_depth_work_mode_by_index(1)
        device.set_depth_work_mode.assert_called_once_with(mode)
        self.assertEqual(device.set_int_property.call_args_list,
                         [unittest.mock.call("disparity_range", 2),
                          unittest.mock.call("disparity_offset", 0)])
        self.assertEqual(info["disparity_search_range"], 256)
        self.assertEqual(info["specified_minimum_depth_m"], .04)

    def test_already_correct_configuration_does_not_write(self):
        device = close_range_device()
        configure_closest_range(device, fake_sdk())
        device.set_depth_work_mode.assert_not_called()
        device.set_int_property.assert_not_called()

    def test_missing_advertised_preset_and_failed_switch_are_rejected(self):
        device = close_range_device()
        device.get_depth_work_mode_list.return_value.get_count.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "does not advertise"):
            configure_closest_range(device, fake_sdk())
        device = close_range_device()
        device.get_depth_work_mode.return_value.name = "Default"
        device.set_depth_work_mode.return_value = "SDK error"
        with self.assertRaisesRegex(RuntimeError, "could not select"):
            configure_closest_range(device, fake_sdk())

    def test_mode_and_disparity_readback_mismatch_are_rejected(self):
        device = close_range_device()
        device.get_depth_work_mode.return_value.name = "Default"
        with self.assertRaisesRegex(RuntimeError, "mode readback"):
            configure_closest_range(device, fake_sdk())
        device = close_range_device()
        device.get_int_property.side_effect = lambda prop: 1
        with self.assertRaisesRegex(RuntimeError, "property.*readback"):
            configure_closest_range(device, fake_sdk())

    def test_unsupported_property_and_256_range_are_rejected(self):
        device = close_range_device()
        device.is_property_supported.return_value = False
        with self.assertRaisesRegex(RuntimeError, "cannot configure/read back"):
            configure_closest_range(device, fake_sdk())
        device = close_range_device()
        device.get_int_property_range.return_value.max = 1
        with self.assertRaisesRegex(RuntimeError, "does not support 2"):
            configure_closest_range(device, fake_sdk())


def distortion(model="brown", **overrides):
    values = dict(k1=0., k2=0., k3=0., k4=0., k5=0., k6=0., p1=0., p2=0., model=model)
    values.update(overrides)
    return SimpleNamespace(**values)


def profile(width=4, height=4, fps=30, fmt="RGB", dist=None, k=None):
    if k is None:
        k = np.array([[4., 0., 1.5], [0., 4., 1.5], [0., 0., 1.]])
    result = Mock()
    result.get_width.return_value = width
    result.get_height.return_value = height
    result.get_fps.return_value = fps
    result.get_format.return_value = fmt
    result.get_intrinsic.return_value = SimpleNamespace(
        fx=k[0, 0], fy=k[1, 1], cx=k[0, 2], cy=k[1, 2], width=width, height=height)
    result.get_distortion.return_value = dist or distortion()
    result.get_extrinsic_to.return_value = SimpleNamespace(rot=np.eye(3), transform=np.zeros(3))
    result.as_video_stream_profile.return_value = result
    return result


def profile_list(profiles):
    result = Mock()
    result.get_count.return_value = len(profiles)
    result.get_stream_profile_by_index.side_effect = profiles.__getitem__
    return result


def frame(data, fmt, dist=None, k=None, timestamp_us=1000):
    result = Mock()
    height, width = data.shape[:2]
    result.get_width.return_value = width
    result.get_height.return_value = height
    result.get_format.return_value = fmt
    result.get_data.return_value = np.ascontiguousarray(data).view(np.uint8)
    result.get_stream_profile.return_value = profile(width, height, fmt=fmt, dist=dist, k=k)
    result.get_depth_scale.return_value = 1.0
    result.get_timestamp_us.return_value = timestamp_us
    return result


def frame_set(color=None, depth=None):
    result = Mock()
    result.get_color_frame.return_value = color
    result.get_depth_frame.return_value = depth
    result.as_frame_set.return_value = result
    return result


def camera_for(color=None, depth=None):
    camera = Gemini305.__new__(Gemini305)
    camera.sdk = fake_sdk()
    camera.info = {}
    camera.maps = None
    camera.intrinsic = None
    camera._geometry_key = None
    camera._fps = 30
    camera.started = True
    camera.pipeline = Mock()
    camera.align = Mock()
    frames = frame_set(color, depth)
    camera.pipeline.wait_for_frames.return_value = frames
    camera.align.process.return_value = frames
    return camera


class ColorConversionTests(unittest.TestCase):
    def test_rgb_copy_owns_contiguous_bytes(self):
        data = np.tile(np.array([250, 120, 7], np.uint8), (4, 4, 1))
        image = frame(data, "RGB")
        result = color_rgb(image, fake_sdk())
        np.testing.assert_array_equal(result, data)
        self.assertTrue(result.flags.c_contiguous)
        self.assertFalse(np.shares_memory(result, image.get_data.return_value))

    def test_bgr_is_converted_to_rgb(self):
        data = np.tile(np.array([7, 120, 250], np.uint8), (4, 4, 1))
        result = color_rgb(frame(data, "BGR"), fake_sdk())
        np.testing.assert_array_equal(result[0, 0], [250, 120, 7])

    def test_mjpg_decodes_rgb_and_rejects_bad_payload(self):
        data = np.tile(np.array([5, 20, 230], np.uint8), (8, 8, 1))
        image = frame(data, "MJPG")
        ok, encoded = cv2.imencode(".jpg", data)
        self.assertTrue(ok)
        image.get_data.return_value = encoded
        rgb = color_rgb(image, fake_sdk())
        self.assertGreater(int(rgb[0, 0, 0]), 220)
        self.assertLess(int(rgb[0, 0, 2]), 10)
        image.get_data.return_value = np.zeros(4, np.uint8)
        with self.assertRaisesRegex(RuntimeError, "invalid MJPG"):
            color_rgb(image, fake_sdk())

    def test_yuyv_is_rgb_and_unsupported_compression_is_rejected(self):
        packed = np.tile(np.array([[82, 90], [82, 240]], np.uint8), (4, 2, 1))
        result = color_rgb(frame(packed, "YUYV"), fake_sdk())
        self.assertGreater(int(result[0, 0, 0]), 200)
        self.assertLess(int(result[0, 0, 2]), 10)
        with self.assertRaisesRegex(RuntimeError, "Unsupported.*color format"):
            color_rgb(frame(packed, "H264"), fake_sdk())

    def test_buffer_sizes_and_depth_endianness(self):
        raw = np.array([[1, 256], [1000, 65535]], dtype="<u2")
        np.testing.assert_array_equal(depth_raw(frame(raw, "Y16"), fake_sdk()), raw)
        wrong = frame(raw, "Y16")
        wrong.get_data.return_value = np.zeros(1, np.uint8)
        with self.assertRaisesRegex(RuntimeError, "depth buffer size"):
            depth_raw(wrong, fake_sdk())


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.color = frame(np.zeros((4, 4, 3), np.uint8), "RGB")
        self.depth = frame(np.full((4, 4), 4000, np.uint16), "Y16")

    def test_metric_scale_timestamp_and_half_pixel_resize(self):
        self.depth.get_depth_scale.return_value = 0.1  # millimetres/count, not metres/count.
        k = np.array([[12., 0., 1.2], [0., 14., 2.1], [0., 0., 1.]])
        for image in (self.color, self.depth):
            image.get_stream_profile.return_value = profile(k=k)
        camera = camera_for(self.color, self.depth)
        with patch("orbbec_camera.time.time_ns", return_value=123456):
            result = camera.read(.1, 2., 1., 2)
        np.testing.assert_array_equal(result.depth, np.full((2, 2), 400, np.uint16))
        np.testing.assert_allclose(result.intrinsic, [[6., 0., .35], [0., 7., .8], [0., 0., 1.]])
        self.assertEqual(result.timestamp_ns, 123456)
        self.assertEqual(result.depth.dtype, np.uint16)
        self.assertTrue(result.depth.flags.c_contiguous)
        self.assertEqual(camera.info["last_frame_skew_us"], 0)

    def test_rational_distortion_rectifies_rgb_only(self):
        dist = distortion(model="k6", k1=-.1, k2=.02, p1=.003, p2=.004,
                          k3=.005, k4=.006, k5=.007, k6=.008)
        self.color.get_stream_profile.return_value = profile(dist=dist)
        camera = camera_for(self.color, self.depth)
        with patch("orbbec_camera.cv2.initUndistortRectifyMap", wraps=cv2.initUndistortRectifyMap) as maps, \
                patch("orbbec_camera.cv2.remap", wraps=cv2.remap) as remap:
            result = camera.read(.1, 5., 1., 1)
        np.testing.assert_allclose(maps.call_args.args[1], [-.1, .02, .003, .004, .005, .006, .007, .008])
        remap.assert_called_once()
        self.assertEqual(remap.call_args.args[0].shape, (4, 4, 3))
        np.testing.assert_array_equal(result.depth, np.full((4, 4), 4000, np.uint16))

    def test_nonzero_aligned_distortion_in_any_coefficient_is_rejected(self):
        for name in ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2"):
            with self.subTest(coefficient=name):
                self.depth.get_stream_profile.return_value = profile(dist=distortion(**{name: .001}))
                with self.assertRaisesRegex(RuntimeError, "not rectified"):
                    camera_for(self.color, self.depth).read(.1, 5., 1., 1)

    def test_unsupported_color_distortion_is_rejected(self):
        self.color.get_stream_profile.return_value = profile(dist=distortion(model="fisheye", k1=.01))
        with self.assertRaisesRegex(RuntimeError, "Unsupported.*distortion model"):
            camera_for(self.color, self.depth).read(.1, 5., 1., 1)

    def test_alignment_intrinsics_dimensions_and_frame_are_checked(self):
        k = np.array([[5., 0., 1.5], [0., 4., 1.5], [0., 0., 1.]])
        self.depth.get_stream_profile.return_value = profile(k=k)
        with self.assertRaisesRegex(RuntimeError, "intrinsics differ"):
            camera_for(self.color, self.depth).read(.1, 5., 1., 1)
        self.depth = frame(np.zeros((2, 4), np.uint16), "Y16")
        with self.assertRaisesRegex(RuntimeError, "dimensions differ"):
            camera_for(self.color, self.depth).read(.1, 5., 1., 1)
        self.depth = frame(np.zeros((4, 4), np.uint16), "Y16")
        self.depth.get_stream_profile.return_value.get_extrinsic_to.return_value.transform = [1., 0., 0.]
        with self.assertRaisesRegex(RuntimeError, "color optical frame"):
            camera_for(self.color, self.depth).read(.1, 5., 1., 1)

    def test_invalid_scale_is_rejected(self):
        self.depth.get_depth_scale.return_value = float("nan")
        with self.assertRaisesRegex(RuntimeError, "invalid metric depth scale"):
            camera_for(self.color, self.depth).read(.1, 5., 1., 1)

    def test_stale_pair_is_dropped_and_next_good_pair_recovers(self):
        for skew_us in (20000, 33000):
            with self.subTest(skew_us=skew_us):
                self.depth.get_timestamp_us.return_value = 1000 + skew_us
                camera = camera_for(self.color, self.depth)
                self.assertIsNone(camera.read(.1, 5., 1., 1))
                self.assertEqual(camera.info["rejected_frame_pairs"], 1)
                camera.align.process.assert_not_called()
                self.depth.get_timestamp_us.return_value = 1000
                self.assertIsNotNone(camera.read(.1, 5., 1., 1))

    def test_missing_frames_return_none_without_conversion(self):
        camera = camera_for()
        self.assertIsNone(camera.read(.1, 5., 1., 1))
        camera.align.process.assert_not_called()
        camera.pipeline.wait_for_frames.return_value = None
        self.assertIsNone(camera.read(.1, 5., 1., 1))


class ProfileAndStartupTests(unittest.TestCase):
    def test_closest_range_excludes_profiles_without_verified_four_cm_specification(self):
        pipeline = Mock()
        depths = [profile(1280, 800, 30, "Y16"), profile(640, 480, 30, "Y16"),
                  profile(848, 530, 30, "Y16"), profile(320, 240, 30, "Y16")]
        pipeline.get_stream_profile_list.side_effect = [
            profile_list(depths), profile_list([profile(640, 480, 30, "RGB")])]
        pairs = stream_candidates(pipeline, fake_sdk(), closest_range=True)
        self.assertEqual([pair[0] for pair in pairs], [depths[1], depths[2]])
        pipeline.get_stream_profile_list.side_effect = [
            profile_list([depths[0]]), profile_list([profile(640, 480, 30, "RGB")])]
        with self.assertRaisesRegex(RuntimeError, "Closest-range depth requires"):
            stream_candidates(pipeline, fake_sdk(), closest_range=True)

    def test_advertised_profiles_prefer_vga_30_and_match_fps(self):
        depths = [profile(320, 240, 30, "Y16"), profile(640, 480, 30, "Y16")]
        colors = [profile(1280, 720, 30, "MJPG"), profile(640, 480, 15, "RGB"),
                  profile(640, 480, 30, "RGB")]
        pipeline = Mock()
        pipeline.get_stream_profile_list.side_effect = [profile_list(depths), profile_list(colors)]
        pairs = stream_candidates(pipeline, fake_sdk())
        self.assertIs(pairs[0][0], depths[1])
        self.assertIs(pairs[0][1], colors[2])
        self.assertEqual(len(pairs), 4)
        pipeline.get_d2c_depth_profile_list.assert_not_called()

    def test_no_common_fps_fails_clearly(self):
        pipeline = Mock()
        pipeline.get_stream_profile_list.side_effect = [
            profile_list([profile(640, 480, 15, "Y16")]),
            profile_list([profile(640, 480, 30, "RGB")])]
        with self.assertRaisesRegex(RuntimeError, "no advertised compatible RGB-D"):
            stream_candidates(pipeline, fake_sdk())

    def test_startup_failure_stops_partial_stream_and_tries_next_advertised_pair(self):
        camera = camera_for()
        camera.started = False
        camera._warm_up = Mock()
        camera.pipeline.start.side_effect = [RuntimeError("USB bandwidth"), None]
        pairs = [(profile(640, 480, fmt="Y16"), profile(640, 480)),
                 (profile(320, 240, fmt="Y16"), profile(640, 480, fmt="MJPG"))]
        camera._start(pairs)
        self.assertEqual(camera.pipeline.start.call_count, 2)
        camera.pipeline.stop.assert_called_once()
        self.assertEqual(camera.info["depth_resolution"], [320, 240])
        config = camera.pipeline.start.call_args.args[0]
        config.set_align_mode.assert_called_once_with("disabled")
        config.set_frame_aggregate_output_mode.assert_called_once_with("full")

    def test_warmup_failures_stop_each_started_stream(self):
        camera = camera_for()
        camera.started = False
        camera._warm_up = Mock(side_effect=RuntimeError("not rectified"))
        with self.assertRaisesRegex(RuntimeError, "No Gemini 305 RGB-D stream pair"):
            camera._start([(profile(fmt="Y16"), profile())])
        camera.pipeline.stop.assert_called_once()
        self.assertFalse(camera.started)

    def test_warmup_has_total_deadline(self):
        camera = camera_for()
        with patch("orbbec_camera.time.monotonic", side_effect=[0, 6]):
            with self.assertRaisesRegex(RuntimeError, "0/15.*within 5s"):
                camera._warm_up()
        camera.pipeline.wait_for_frames.assert_not_called()

    def test_close_is_idempotent(self):
        camera = camera_for()
        camera.close()
        camera.close()
        camera.pipeline.stop.assert_called_once()


class ConstructorTests(unittest.TestCase):
    def sdk_with_device(self, name="Gemini305", pid=0x0840):
        sdk = fake_sdk()
        device_list = Mock()
        device_list.get_count.return_value = 1
        device_list.get_device_name_by_index.return_value = name
        device_list.get_device_vid_by_index.return_value = 0x2BC5
        device_list.get_device_pid_by_index.return_value = pid
        device_list.get_device_serial_number_by_index.return_value = "g305"
        device_list.get_device_by_index.return_value = close_range_device()
        info = device_list.get_device_by_index.return_value.get_device_info.return_value
        info.get_name.return_value = name
        info.get_serial_number.return_value = "g305"
        info.get_firmware_version.return_value = "1.0.70"
        info.get_connection_type.return_value = "USB3.0"
        sdk.Context = Mock()
        sdk.Context.return_value.query_devices.return_value = device_list
        sdk.Pipeline = Mock()
        sdk.AlignFilter = Mock()
        sdk.AlignFilter.return_value.get_config_value.return_value = 0.
        sdk.OBStreamType = SimpleNamespace(COLOR_STREAM="color")
        return sdk

    def test_constructor_enables_sync_and_sets_rectified_alignment_explicitly(self):
        sdk = self.sdk_with_device()
        with patch("orbbec_camera._load_sdk", return_value=sdk), \
                patch("orbbec_camera.stream_candidates", return_value=[]), \
                patch.object(Gemini305, "_start"):
            camera = Gemini305("g305")
        sdk.Pipeline.return_value.enable_frame_sync.assert_called_once()
        sdk.AlignFilter.return_value.set_config_value.assert_called_once_with("TargetDistortion", 0.)
        sdk.AlignFilter.return_value.set_match_target_resolution.assert_called_once_with(True)
        self.assertEqual(camera.info["serial"], "g305")
        self.assertEqual(camera.info["depth_work_mode"], "Close Range Default")
        self.assertEqual(camera.info["disparity_search_range"], 256)
        self.assertFalse(camera.started)

    def test_closest_range_failure_prevents_pipeline_creation(self):
        sdk = self.sdk_with_device()
        with patch("orbbec_camera._load_sdk", return_value=sdk), \
                patch("orbbec_camera.configure_closest_range", side_effect=RuntimeError("unsupported")):
            with self.assertRaisesRegex(RuntimeError, "unsupported"):
                Gemini305()
        sdk.Pipeline.assert_not_called()

    def test_unsupported_model_and_wrong_serial_are_rejected_before_pipeline(self):
        for name, pid, serial in (("Gemini 2", 0x0670, None), ("Gemini305", 0x0840, "wrong")):
            with self.subTest(name=name, serial=serial):
                sdk = self.sdk_with_device(name, pid)
                with patch("orbbec_camera._load_sdk", return_value=sdk):
                    with self.assertRaisesRegex(RuntimeError, "No matching Gemini 305"):
                        Gemini305(serial)
                sdk.Pipeline.assert_not_called()

    def test_constructor_stops_on_unexpected_error_after_start(self):
        sdk = self.sdk_with_device()
        def fail(camera, candidates):
            camera.started = True
            raise ValueError("unexpected SDK calibration failure")
        with patch("orbbec_camera._load_sdk", return_value=sdk), \
                patch("orbbec_camera.stream_candidates", return_value=[]), \
                patch.object(Gemini305, "_start", fail):
            with self.assertRaisesRegex(ValueError, "unexpected SDK calibration failure"):
                Gemini305()
        sdk.Pipeline.return_value.stop.assert_called_once()


@unittest.skipIf(importlib.util.find_spec("pyorbbecsdk") is None, "optional Orbbec SDK is not installed")
class InstalledSDKGeometryContractTests(unittest.TestCase):
    """Use actual bound calibration structures; never create a device or pipeline."""
    def test_geometry_reads_real_sdk_intrinsic_distortion_and_extrinsic_fields(self):
        sdk = importlib.import_module("pyorbbecsdk")
        color = frame(np.zeros((4, 4, 3), np.uint8), "RGB")
        depth = frame(np.zeros((4, 4), np.uint16), "Y16")
        for image in (color, depth):
            intr = sdk.OBCameraIntrinsic()
            intr.fx, intr.fy, intr.cx, intr.cy = 4., 4., 1.5, 1.5
            intr.width, intr.height = 4, 4
            image.get_stream_profile.return_value.get_intrinsic.return_value = intr
            image.get_stream_profile.return_value.get_distortion.return_value = sdk.OBCameraDistortion()
        color_dist = color.get_stream_profile.return_value.get_distortion.return_value
        color_dist.model = sdk.OBCameraDistortionModel.BROWN_CONRADY_K6
        color_dist.k1, color_dist.k4 = -.01, -.005
        extr = sdk.OBExtrinsic()
        extr.rot = np.eye(3, dtype=np.float32)
        extr.transform = np.zeros(3, dtype=np.float32)
        depth.get_stream_profile.return_value.get_extrinsic_to.return_value = extr
        camera = camera_for(color, depth)
        camera.sdk = sdk
        camera._geometry(color, depth)
        self.assertIsNotNone(camera.maps)
        np.testing.assert_allclose(camera.intrinsic, [[4., 0., 1.5], [0., 4., 1.5], [0., 0., 1.]])
        extr.transform = np.array([1., 0., 0.], dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, "color optical frame"):
            camera._geometry(color, depth)


if __name__ == "__main__":
    unittest.main()

"""Gemini 305 adapter for pyorbbecsdk2 2.1.2 (embedded Orbbec SDK 2.9.3).

Software D2C with TargetDistortion=0 produces rectified color-optical depth.
The SDK retains the raw color frame, so only color is rectified with OpenCV.
"""
import importlib
import importlib.util
import logging
import time

import cv2
import numpy as np

from camera import RGBD, prepare_depth


CLOSE_RANGE_MODE = "Close Range Default"
# SDK v2.9.3 Property.h: OB_PROP_DISP_SEARCH_RANGE_MODE_INT 2 means 256.
DISPARITY_256 = 2
CLOSE_RANGE_DEPTH_SIZES = ((640, 480), (848, 480), (848, 530))


def configure_closest_range(device, sdk):
    """Require the advertised close-range preset and verify its 256 search range.

    This must run before creating the pipeline or enumerating stream profiles:
    Default mode advertises only disparity ranges up to 128 on firmware 1.0.70.
    """
    modes = device.get_depth_work_mode_list()
    advertised = [modes.get_depth_work_mode_by_index(index)
                  for index in range(modes.get_count())]
    mode = next((mode for mode in advertised if mode.name == CLOSE_RANGE_MODE), None)
    if mode is None:
        raise RuntimeError(
            f"Gemini 305 does not advertise {CLOSE_RANGE_MODE!r}; closest-range capture "
            f"requires compatible firmware. Advertised modes: {[mode.name for mode in advertised]}")
    if device.get_depth_work_mode().name != CLOSE_RANGE_MODE:
        status = device.set_depth_work_mode(mode)
        if status != sdk.OBStatus.STATUS_OK:
            raise RuntimeError(f"Gemini 305 could not select {CLOSE_RANGE_MODE!r}: {status}")
    if device.get_depth_work_mode().name != CLOSE_RANGE_MODE:
        raise RuntimeError("Gemini 305 close-range depth mode readback differs from the requested preset.")
    properties = ((sdk.OBPropertyID.OB_PROP_DISP_SEARCH_RANGE_MODE_INT, DISPARITY_256),
                  (sdk.OBPropertyID.OB_PROP_DISP_SEARCH_OFFSET_INT, 0))
    for prop, requested in properties:
        if not device.is_property_supported(prop, sdk.OBPermissionType.PERMISSION_READ_WRITE):
            raise RuntimeError(f"Gemini 305 cannot configure/read back closest-range property {prop}.")
        bounds = device.get_int_property_range(prop)
        if not bounds.min <= requested <= bounds.max:
            raise RuntimeError(f"Gemini 305 closest-range property {prop} does not support {requested}.")
        if device.get_int_property(prop) != requested:
            device.set_int_property(prop, requested)
        if device.get_int_property(prop) != requested:
            raise RuntimeError(f"Gemini 305 closest-range property {prop} readback differs from {requested}.")
    logging.getLogger("object_scanner").info(
        "Gemini 305 verified depth mode=%s, disparity search=256, offset=0", CLOSE_RANGE_MODE)
    return dict(depth_work_mode=CLOSE_RANGE_MODE, disparity_search_range=256,
                disparity_search_offset=0, specified_minimum_depth_m=0.04)


def sdk_available():
    return importlib.util.find_spec("pyorbbecsdk") is not None


def _load_sdk():
    try:
        return importlib.import_module("pyorbbecsdk")
    except ImportError as error:
        raise RuntimeError(
            "Orbbec backend requires pyorbbecsdk2==2.1.2 in the camera Python environment "
            "(the import module is pyorbbecsdk).") from error


def _supported_device(device_list, index):
    name = device_list.get_device_name_by_index(index).lower().replace(" ", "")
    return name in ("gemini305", "orbbecgemini305") or (
        device_list.get_device_vid_by_index(index) == 0x2BC5
        and device_list.get_device_pid_by_index(index) == 0x0840)


def devices():
    """Enumerate Gemini 305 only; other Orbbec models are not validated here."""
    sdk = _load_sdk()
    context = sdk.Context()
    device_list = context.query_devices()
    result = []
    for index in range(device_list.get_count()):
        if not _supported_device(device_list, index):
            continue
        info = device_list.get_device_by_index(index).get_device_info()
        result.append(dict(name=info.get_name(), serial=info.get_serial_number(),
                           firmware=info.get_firmware_version(), backend="orbbec",
                           usb_type=info.get_connection_type()))
    return result


def _profiles(profile_list):
    return [profile_list.get_stream_profile_by_index(index).as_video_stream_profile()
            for index in range(profile_list.get_count())]


def _profile_key(profile):
    return (profile.get_width(), profile.get_height(), profile.get_fps(), profile.get_format())


def _color_formats(sdk):
    return (sdk.OBFormat.RGB, sdk.OBFormat.BGR, sdk.OBFormat.MJPG,
            sdk.OBFormat.YUYV, sdk.OBFormat.YUY2, sdk.OBFormat.UYVY)


def stream_candidates(pipeline, sdk, closest_range=False):
    """Enumerate advertised same-FPS pairs for the software AlignFilter.

    Pipeline.get_d2c_depth_profile_list supports hardware D2C only in this SDK;
    Gemini 305 supports the advertised RGB/depth resolutions with software D2C.
    The resulting aligned frame geometry is checked before returning any pixels.
    """
    depth_formats = (sdk.OBFormat.Y16, sdk.OBFormat.Z16)
    depths = [p for p in _profiles(
        pipeline.get_stream_profile_list(sdk.OBSensorType.DEPTH_SENSOR))
        if p.get_format() in depth_formats and (not closest_range or
           (p.get_width(), p.get_height()) in CLOSE_RANGE_DEPTH_SIZES)]
    colors = [p for p in _profiles(pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR))
              if p.get_format() in _color_formats(sdk)]
    pairs = []
    for color in colors:
        for depth in depths:
            if depth.get_fps() == color.get_fps():
                pairs.append((depth, color))
    def preference(pair):
        depth, color = pair
        fps = color.get_fps()
        size = (color.get_width(), color.get_height())
        return (fps != 30, abs(fps - 30), size != (640, 480), size[0] * size[1],
                _color_formats(sdk).index(color.get_format()),
                (depth.get_width(), depth.get_height()) != (640, 480),
                depth.get_width() * depth.get_height())
    pairs.sort(key=preference)
    if not pairs:
        raise RuntimeError(
            "Gemini 305 has no advertised compatible RGB-D profiles for software depth-to-color "
            "alignment. Supported depth formats are Y16/Z16; color RGB/BGR/MJPG/YUYV/UYVY. "
            + ("Closest-range depth requires 640x480, 848x480 or 848x530. " if closest_range else "") +
            "Check firmware, USB connection and the pinned Orbbec SDK.")
    return pairs


def color_rgb(frame, sdk):
    """Own the frame buffer and decode its advertised format into RGB uint8."""
    width, height = frame.get_width(), frame.get_height()
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)
    fmt = frame.get_format()
    if fmt == sdk.OBFormat.MJPG:
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape != (height, width, 3):
            raise RuntimeError("Gemini 305 returned an invalid MJPG color frame.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif fmt in (sdk.OBFormat.RGB, sdk.OBFormat.BGR):
        if data.size != width * height * 3:
            raise RuntimeError("Gemini 305 color buffer size differs from its stream profile.")
        rgb = data.reshape(height, width, 3)
        if fmt == sdk.OBFormat.BGR:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    elif fmt in (sdk.OBFormat.YUYV, sdk.OBFormat.YUY2, sdk.OBFormat.UYVY):
        if data.size != width * height * 2:
            raise RuntimeError("Gemini 305 packed YUV buffer size differs from its stream profile.")
        code = cv2.COLOR_YUV2RGB_UYVY if fmt == sdk.OBFormat.UYVY else cv2.COLOR_YUV2RGB_YUYV
        rgb = cv2.cvtColor(data.reshape(height, width, 2), code)
    else:
        raise RuntimeError(f"Unsupported Gemini 305 color format: {fmt}")
    return np.array(rgb, dtype=np.uint8, order="C", copy=True)


def depth_raw(frame, sdk):
    if frame.get_format() not in (sdk.OBFormat.Y16, sdk.OBFormat.Z16):
        raise RuntimeError(f"Unsupported Gemini 305 depth format: {frame.get_format()}")
    data = np.frombuffer(frame.get_data(), dtype=np.uint8)
    width, height = frame.get_width(), frame.get_height()
    if data.size != width * height * 2:
        raise RuntimeError("Gemini 305 depth buffer size differs from its stream profile.")
    return data.view("<u2").reshape(height, width).copy()


def _intrinsic(profile):
    intr = profile.get_intrinsic()
    values = np.array([intr.fx, intr.fy, intr.cx, intr.cy], dtype=np.float64)
    if not np.isfinite(values).all() or intr.fx <= 0 or intr.fy <= 0:
        raise RuntimeError("Gemini 305 returned invalid camera intrinsics.")
    if intr.width != profile.get_width() or intr.height != profile.get_height():
        raise RuntimeError("Gemini 305 intrinsic dimensions differ from its video profile.")
    return np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]], dtype=np.float64)


def _coefficients(distortion):
    # OpenCV rational Brown order differs from the SDK structure field order.
    return np.array([distortion.k1, distortion.k2, distortion.p1, distortion.p2,
                     distortion.k3, distortion.k4, distortion.k5, distortion.k6], dtype=np.float64)


class Gemini305:
    def __init__(self, serial=None):
        self.sdk = _load_sdk()
        self.started = False
        self.context = self.sdk.Context()
        device_list = self.context.query_devices()
        matches = [index for index in range(device_list.get_count())
                   if _supported_device(device_list, index)
                   and (not serial or device_list.get_device_serial_number_by_index(index) == serial)]
        if not matches:
            raise RuntimeError("No matching Gemini 305 found. Check USB, Orbbec udev rules and --serial.")
        if len(matches) != 1:
            raise RuntimeError("Multiple Gemini 305 cameras found; select one with --serial.")
        device = device_list.get_device_by_index(matches[0])
        info = device.get_device_info()
        self.info = dict(name=info.get_name(), serial=info.get_serial_number(),
                         firmware=info.get_firmware_version(), backend="orbbec",
                         usb_type=info.get_connection_type())
        self.info.update(configure_closest_range(device, self.sdk))
        self.pipeline = self.sdk.Pipeline(device)
        self.pipeline.enable_frame_sync()
        self.align = self.sdk.AlignFilter(align_to_stream=self.sdk.OBStreamType.COLOR_STREAM)
        # SDK 2.9.3 creates undistorted target geometry and copies the raw color unchanged.
        self.align.set_config_value("TargetDistortion", 0.0)
        self.align.set_match_target_resolution(True)
        if self.align.get_config_value("TargetDistortion") != 0:
            raise RuntimeError("Orbbec AlignFilter cannot provide rectified target geometry.")
        self.intrinsic = None
        self.maps = None
        self._geometry_key = None
        self._fps = None
        try:
            self._start(stream_candidates(self.pipeline, self.sdk, closest_range=True))
        except Exception:
            self.close()
            raise

    def _start(self, candidates, timeout=20.0):
        deadline = time.monotonic() + timeout
        failures = []
        for depth, color in candidates:
            if time.monotonic() >= deadline:
                break
            config = self.sdk.Config()
            config.enable_stream(depth)
            config.enable_stream(color)
            config.set_align_mode(self.sdk.OBAlignMode.DISABLE)
            config.set_frame_aggregate_output_mode(self.sdk.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
            description = f"depth={_profile_key(depth)}, color={_profile_key(color)}"
            try:
                self.pipeline.start(config)
            except Exception as error:
                failures.append(f"{description}: {error}")
                # A failed SDK start can have partially started one sensor.
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
                continue
            self.started = True
            self._fps = color.get_fps()
            try:
                self._warm_up(timeout=min(5.0, max(0.0, deadline - time.monotonic())))
            except RuntimeError as error:
                failures.append(f"{description}: {error}")
                self.close()
                continue
            self.info.update(depth_resolution=[depth.get_width(), depth.get_height()],
                             color_resolution=[color.get_width(), color.get_height()], fps=color.get_fps(),
                             color_format=str(color.get_format()), alignment="software_d2c_rectified",
                             depth_unit_m=0.001)
            logging.getLogger("object_scanner").info("Gemini 305 %s", description)
            return
        raise RuntimeError(
            "No Gemini 305 RGB-D stream pair could start and deliver valid rectified frames. "
            "Check USB bandwidth, competing camera applications and firmware. " + " | ".join(failures[-5:]))

    def _warm_up(self, frame_count=15, timeout=5.0):
        deadline = time.monotonic() + timeout
        received = 0
        while received < frame_count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"Only {received}/{frame_count} complete Gemini 305 warm-up pairs within {timeout:g}s.")
            frames = self.pipeline.wait_for_frames(max(1, min(1000, int(remaining * 1000))))
            if frames is None or frames.get_color_frame() is None or frames.get_depth_frame() is None:
                continue
            if not self._check_pair_timestamps(frames.get_color_frame(), frames.get_depth_frame()):
                continue
            aligned = self.align.process(frames)
            if aligned is None:
                continue
            aligned = aligned.as_frame_set()
            if aligned.get_color_frame() is None or aligned.get_depth_frame() is None:
                continue
            self._geometry(aligned.get_color_frame(), aligned.get_depth_frame())
            received += 1

    def _check_pair_timestamps(self, color, depth):
        """Host aggregation does not prove simultaneous exposure; bound device skew."""
        skew_us = abs(int(color.get_timestamp_us()) - int(depth.get_timestamp_us()))
        self.info["last_frame_skew_us"] = skew_us
        if self._fps is not None and skew_us > 500_000 / self._fps:
            rejected = self.info.get("rejected_frame_pairs", 0) + 1
            self.info["rejected_frame_pairs"] = rejected
            if rejected == 1 or rejected % 100 == 0:
                logging.getLogger("object_scanner").warning(
                    "Dropping Gemini 305 RGB-D pair with %d us skew (>half-period at %s FPS); rejected=%d",
                    skew_us, self._fps, rejected)
            return False
        return True

    def _geometry(self, color, depth):
        color_profile = color.get_stream_profile().as_video_stream_profile()
        depth_profile = depth.get_stream_profile().as_video_stream_profile()
        color_k, depth_k = _intrinsic(color_profile), _intrinsic(depth_profile)
        size = (color.get_width(), color.get_height())
        if size != (depth.get_width(), depth.get_height()):
            raise RuntimeError("Orbbec aligned depth dimensions differ from color dimensions.")
        if not np.allclose(color_k, depth_k, rtol=1e-6, atol=1e-5):
            raise RuntimeError("Orbbec aligned depth intrinsics differ from rectified color intrinsics.")
        depth_coeffs = _coefficients(depth_profile.get_distortion())
        if not np.isfinite(depth_coeffs).all() or np.any(np.abs(depth_coeffs) > 1e-8):
            raise RuntimeError("Orbbec aligned depth is not rectified; refusing inconsistent RGB-D geometry.")
        extrinsic = depth_profile.get_extrinsic_to(color_profile)
        if not np.allclose(np.asarray(extrinsic.rot).reshape(3, 3), np.eye(3), atol=1e-5) \
                or not np.allclose(extrinsic.transform, 0, atol=1e-4):
            raise RuntimeError("Orbbec aligned depth is not in the color optical frame.")
        distortion = color_profile.get_distortion()
        coeffs = _coefficients(distortion)
        if not np.isfinite(coeffs).all():
            raise RuntimeError("Gemini 305 returned invalid color distortion coefficients.")
        key = (size, tuple(color_k.ravel()), tuple(coeffs), distortion.model)
        if key == self._geometry_key:
            return
        self.maps = None
        if np.any(np.abs(coeffs) > 1e-8):
            if distortion.model not in (self.sdk.OBCameraDistortionModel.BROWN_CONRADY,
                                         self.sdk.OBCameraDistortionModel.BROWN_CONRADY_K6):
                raise RuntimeError(f"Unsupported Gemini 305 color distortion model: {distortion.model}")
            self.maps = cv2.initUndistortRectifyMap(color_k, coeffs, None, color_k, size, cv2.CV_32FC1)
        self.intrinsic = color_k
        self._geometry_key = key

    def read(self, near, far, fov_keep, downscale):
        frames = self.pipeline.wait_for_frames(5000)
        timestamp_ns = time.time_ns()
        if frames is None:
            return None
        if frames.get_color_frame() is None or frames.get_depth_frame() is None:
            return None
        if not self._check_pair_timestamps(frames.get_color_frame(), frames.get_depth_frame()):
            return None
        aligned = self.align.process(frames)
        if aligned is None:
            return None
        aligned = aligned.as_frame_set()
        color, depth = aligned.get_color_frame(), aligned.get_depth_frame()
        if color is None or depth is None:
            return None
        self._geometry(color, depth)
        rgb, raw = color_rgb(color, self.sdk), depth_raw(depth, self.sdk)
        scale = float(depth.get_depth_scale()) * 0.001  # SDK scale is millimetres per count.
        if not np.isfinite(scale) or scale <= 0:
            raise RuntimeError("Gemini 305 returned an invalid metric depth scale.")
        if self.maps is not None:
            rgb = cv2.remap(rgb, *self.maps, cv2.INTER_LINEAR)
        height, width = raw.shape
        intrinsic = self.intrinsic.copy()
        if downscale > 1:
            size = (width // downscale, height // downscale)
            if min(size) < 1:
                raise ValueError("Camera downscale exceeds the frame dimensions.")
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
            raw = cv2.resize(raw, size, interpolation=cv2.INTER_NEAREST)
            sx, sy = size[0] / width, size[1] / height
            intrinsic[0, 0] *= sx
            intrinsic[1, 1] *= sy
            intrinsic[0, 2] = (intrinsic[0, 2] + 0.5) * sx - 0.5
            intrinsic[1, 2] = (intrinsic[1, 2] + 0.5) * sy - 0.5
        return RGBD(np.ascontiguousarray(rgb), prepare_depth(raw, scale, near, far, fov_keep),
                    intrinsic, timestamp_ns)

    def close(self):
        if self.started:
            self.started = False
            self.pipeline.stop()

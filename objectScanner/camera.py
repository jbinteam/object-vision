"""RGB-D acquisition with aligned, rectified RGB and millimetre depth."""
from dataclasses import dataclass
import logging
import time

import cv2
import numpy as np
try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


def _require_realsense():
    if rs is None:
        raise RuntimeError("RealSense backend requires pyrealsense2 in the camera Python environment.")


def _realsense_devices():
    _require_realsense()
    return [
        {"name": d.get_info(rs.camera_info.name),
         "serial": d.get_info(rs.camera_info.serial_number),
         "firmware": d.get_info(rs.camera_info.firmware_version),
         "backend": "realsense"}
        for d in rs.context().query_devices()
    ]


def devices(backend="auto"):
    """List cameras without starting a stream; optional SDKs remain independent."""
    if backend not in ("auto", "orbbec", "realsense"):
        raise ValueError(f"Unknown camera backend: {backend}")
    found = []
    if backend in ("auto", "orbbec"):
        from orbbec_camera import devices as orbbec_devices, sdk_available
        if backend == "orbbec" or sdk_available():
            found.extend(orbbec_devices())
    if backend in ("auto", "realsense") and (backend == "realsense" or rs is not None):
        found.extend(_realsense_devices())
    return found


def open_camera(backend="auto", serial=None):
    """Open one supported Gemini 305 or L515, requiring selection on ambiguity."""
    found = [item for item in devices(backend)
             if item["backend"] == "orbbec" or "L515" in item["name"]]
    if serial:
        found = [item for item in found if item["serial"] == serial]
    if not found:
        raise RuntimeError(
            f"No matching supported RGB-D camera found (backend={backend}, serial={serial!r}). "
            "Check the camera USB connection, SDK installation and udev rules.")
    if len(found) != 1:
        raise RuntimeError("Multiple supported RGB-D cameras found; select --camera-backend and --serial.")
    selected = found[0]
    if selected["backend"] == "orbbec":
        from orbbec_camera import Gemini305
        return Gemini305(selected["serial"])
    return L515(selected["serial"])


def configure_streams(device, pipeline, serial):
    """Select an advertised, jointly resolvable 30 FPS RGB-D configuration."""
    depths, colors = set(), set()
    for sensor in device.sensors:
        for profile in sensor.get_stream_profiles():
            if profile.fps() != 30:
                continue
            if profile.stream_type() == rs.stream.depth and profile.format() == rs.format.z16:
                modes = depths
            elif profile.stream_type() == rs.stream.color and profile.format() == rs.format.rgb8:
                modes = colors
            else:
                continue
            video = profile.as_video_stream_profile()
            modes.add((video.width(), video.height()))
    if not depths or not colors:
        raise RuntimeError("L515 needs both Z16 depth and RGB8 color profiles at 30 FPS.")
    # Prefer the original VGA depth mode, then lower-bandwidth alternatives.
    depths = sorted(depths, key=lambda size: (size != (640, 480), size[0] * size[1]))
    colors = sorted(colors, key=lambda size: size[0] * size[1])
    wrapper = rs.pipeline_wrapper(pipeline)
    for depth in depths:
        for color in colors:
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.depth, *depth, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, *color, rs.format.rgb8, 30)
            if config.can_resolve(wrapper):
                return config, depth, color
    raise RuntimeError(
        f"No compatible RGB-D stream pair. Depth modes: {depths}; color modes: {colors}. "
        "Close other camera applications and check the USB connection.")


def warm_up(pipeline, frame_count=15, timeout=10.0):
    """Wait for complete RGB-D pairs with a bounded total startup budget."""
    logger = logging.getLogger("object_scanner")
    logger.info("Waiting for %d RGB-D warm-up frames (limit %.1fs)", frame_count, timeout)
    deadline = time.monotonic() + timeout
    received = 0
    while received < frame_count:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                f"L515 started but received only {received}/{frame_count} complete RGB-D "
                f"warm-up frames within {timeout:g}s. Close other camera applications, "
                "reconnect the camera, and try a direct USB 3 port instead of a hub.")
        ok, frames = pipeline.try_wait_for_frames(max(1, min(1000, int(remaining * 1000))))
        if ok and frames.get_depth_frame() and frames.get_color_frame():
            received += 1
            logger.info("Warm-up RGB-D pair %d/%d", received, frame_count)
    logger.info("Camera warm-up complete")


@dataclass
class RGBD:
    color: np.ndarray  # RGB uint8
    depth: np.ndarray  # uint16 millimetres
    intrinsic: np.ndarray
    timestamp_ns: int | None = None  # host wall-clock receipt time, before processing


def prepare_depth(raw, scale, near, far, fov_keep):
    """Convert sensor units to mm and apply distance and central-ROI masks."""
    metres = raw.astype(np.float32) * scale
    valid = (metres >= near) & (metres <= far) & (raw > 0)
    h, w = raw.shape
    y, x = int(round(h * (1 - fov_keep) / 2)), int(round(w * (1 - fov_keep) / 2))
    roi = np.zeros_like(valid)
    roi[y:h-y, x:w-x] = True
    return np.ascontiguousarray(
        np.where(valid & roi, np.rint(np.clip(metres * 1000, 0, 65535)), 0).astype(np.uint16))


class L515:
    def __init__(self, serial=None):
        _require_realsense()
        self.pipeline = rs.pipeline()
        self.started = False
        found = [d for d in devices("realsense") if "L515" in d["name"]]
        if serial:
            found = [d for d in found if d["serial"] == serial]
        if not found:
            raise RuntimeError("No matching L515 found. Check USB 3 connection and RealSense udev rules.")
        if len(found) > 1:
            raise RuntimeError("Multiple L515 cameras found; select one with --serial.")
        self.info = found[0]
        selected = next(d for d in rs.context().query_devices()
                        if d.get_info(rs.camera_info.serial_number) == self.info["serial"])
        usb_type = (selected.get_info(rs.camera_info.usb_type_descriptor)
                    if selected.supports(rs.camera_info.usb_type_descriptor) else "unknown")
        config, depth_mode, color_mode = configure_streams(selected, self.pipeline, self.info["serial"])
        self.info.update(usb_type=usb_type, depth_resolution=list(depth_mode),
                         color_resolution=list(color_mode), fps=30)
        message = (f"USB {usb_type}: depth {depth_mode[0]}x{depth_mode[1]}, "
                   f"color {color_mode[0]}x{color_mode[1]} at 30 FPS")
        print(message, flush=True)
        import logging
        logging.getLogger("object_scanner").info(message)
        if usb_type.startswith("2"):
            print("USB 2 connection: using reduced depth resolution. USB 3 provides finer depth sampling.", flush=True)
        try:
            profile = self.pipeline.start(config)
            self.started = True
            self.scale = profile.get_device().first_depth_sensor().get_depth_scale()
            self.align = rs.align(rs.stream.color)
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.intrinsic = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)
            self.maps = None
            if any(abs(c) > 1e-8 for c in intr.coeffs):
                if intr.model not in (rs.distortion.brown_conrady, rs.distortion.modified_brown_conrady):
                    raise RuntimeError(f"Unsupported color distortion model: {intr.model}")
                self.maps = cv2.initUndistortRectifyMap(
                    self.intrinsic, np.array(intr.coeffs), None, self.intrinsic,
                    (intr.width, intr.height), cv2.CV_32FC1)
            # Let exposure settle before initializing the reconstruction.
            warm_up(self.pipeline)
        except Exception:
            self.close()
            raise

    def read(self, near, far, fov_keep, downscale):
        frames = self.pipeline.wait_for_frames(5000)
        timestamp_ns = time.time_ns()
        frames = self.align.process(frames)
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            return None
        rgb = np.asanyarray(color.get_data()).copy()
        raw = np.asanyarray(depth.get_data()).copy()
        if self.maps is not None:
            rgb = cv2.remap(rgb, *self.maps, cv2.INTER_LINEAR)
            raw = cv2.remap(raw, *self.maps, cv2.INTER_NEAREST)
        h, w = raw.shape
        k = self.intrinsic.copy()
        if downscale > 1:
            size = (w // downscale, h // downscale)
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
            raw = cv2.resize(raw, size, interpolation=cv2.INTER_NEAREST)
            sx, sy = size[0] / w, size[1] / h
            k[0, 0] *= sx
            k[1, 1] *= sy
            k[0, 2] = (k[0, 2] + 0.5) * sx - 0.5
            k[1, 2] = (k[1, 2] + 0.5) * sy - 0.5
        return RGBD(np.ascontiguousarray(rgb), prepare_depth(raw, self.scale, near, far, fov_keep), k, timestamp_ns)

    def close(self):
        if self.started:
            self.pipeline.stop()
            self.started = False

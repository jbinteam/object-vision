"""ROS bridge supervising isolated RGB-D capture with the selected camera SDK."""
import os
from pathlib import Path
import subprocess
import threading
import time
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from .protocol import read_frame
from .ros_support import PipelineNode, run


class CameraNode(PipelineNode):
    def __init__(self):
        super().__init__('rgbd_camera')
        if self.get_parameter('use_sim_time').value:
            raise ValueError('Live RGB-D capture uses host time; use_sim_time must be false')
        project = Path(self.param('project_root', '')).resolve()
        python = self.param('camera_python', str(project / '.venv/bin/python'))
        self.frame = self.param('optical_frame', 'camera_color_optical_frame')
        if not self.frame or self.frame.startswith('/'):
            raise ValueError('A nonempty relative optical_frame is required')
        rate = self.positive('publish_rate', 10.0)
        self.max_age = self.positive('maximum_data_age', 0.75)
        self.timeout = self.positive('worker_timeout', 10.0)
        self.startup_timeout = self.positive('worker_startup_timeout', 30.0)
        minimum = self.positive('minimum_depth', 0.15)
        maximum = self.positive('maximum_depth', 1.5)
        downscale = int(self.param('downscale', 1))
        serial = self.param('serial', '')
        backend = self.param('camera_backend', 'orbbec')
        if backend not in ('orbbec', 'realsense', 'auto'):
            raise ValueError('camera_backend must be orbbec, realsense or auto')
        if downscale < 1 or not minimum < maximum <= 65.535:
            raise ValueError('Invalid camera downscale or depth interval')
        if not (project / 'camera.py').is_file() or not Path(python).is_file():
            raise ValueError('Set project_root and camera_python to the scanner and its Python 3.10 venv')
        self.bridge = CvBridge()
        self.color_pub = self.create_publisher(Image, '/camera/color/image_raw', self.image_qos())
        self.depth_pub = self.create_publisher(Image, '/camera/depth/image_raw', self.image_qos(5))
        self.info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', self.image_qos(5))
        self.lock = threading.Lock()
        self.latest, self.error, self.received = None, None, False
        self.last_received = time.monotonic()
        env = os.environ.copy()
        # Do not expose ROS's Python 3.12 modules to the Python 3.10 camera worker.
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        self.worker = subprocess.Popen([
            python, '-u', str(Path(__file__).with_name('camera_worker.py')),
            '--project-root', str(project), '--camera-backend', backend, '--serial', serial,
            '--downscale', str(downscale), '--minimum-depth', str(minimum),
            '--maximum-depth', str(maximum)], stdout=subprocess.PIPE,
            stdin=subprocess.DEVNULL, env=env, bufsize=0)
        self.reader = threading.Thread(target=self.read_frames, daemon=True)
        self.reader.start()
        self.create_timer(1 / rate, self.publish_latest)
        self.status('camera_starting')

    def read_frames(self):
        try:
            while True:
                packet = read_frame(self.worker.stdout)
                with self.lock:
                    self.latest = packet  # replace superseded frames, never queue them
                    self.last_received = time.monotonic()
                    self.received = True
        except Exception as exc:
            self.error = str(exc)

    def publish_latest(self):
        with self.lock:
            packet, self.latest = self.latest, None
            elapsed = time.monotonic() - self.last_received
            received = self.received
        limit = self.timeout if received else self.startup_timeout
        if self.error or elapsed > limit:
            reason = self.error or f'Camera worker timed out after {elapsed:.1f}s'
            self.status(reason, 2)
            self.stop_worker()
            raise RuntimeError(reason)
        if packet is None:
            return
        rgb, depth, intrinsic, timestamp = packet
        age = (self.get_clock().now().nanoseconds - timestamp) / 1e9
        if not 0 <= age <= self.max_age:
            self.status('stale_camera_frame', 1, age_ms=age * 1000)
            return
        start = time.perf_counter()
        header = Header(stamp=Time(sec=timestamp // 10**9, nanosec=timestamp % 10**9), frame_id=self.frame)
        color = self.bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='16UC1')
        color.header = depth_msg.header = header
        h, w = depth.shape
        info = CameraInfo(header=header, height=h, width=w, distortion_model='plumb_bob')
        info.d = [0.] * 5
        info.k = intrinsic.ravel().tolist()
        info.r = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
        info.p = [float(intrinsic[0, 0]), 0., float(intrinsic[0, 2]), 0.,
                  0., float(intrinsic[1, 1]), float(intrinsic[1, 2]), 0., 0., 0., 1., 0.]
        self.color_pub.publish(color)
        self.depth_pub.publish(depth_msg)
        self.info_pub.publish(info)
        self.status('camera_streaming', camera_publish_ms=(time.perf_counter() - start) * 1000,
                    receipt_to_publish_ms=age * 1000, width=w, height=h)

    def stop_worker(self):
        if self.worker.poll() is None:
            self.worker.terminate()
            try:
                self.worker.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.worker.kill()
                self.worker.wait(timeout=2)

    def destroy_node(self):
        self.stop_worker()
        self.reader.join(timeout=1)
        self.worker.stdout.close()
        return super().destroy_node()


def main(args=None):
    run(CameraNode, args)

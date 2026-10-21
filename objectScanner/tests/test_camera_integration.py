"""Exercise the SDK-isolated camera worker and its binary ROS bridge contract."""
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "ros2_ws/src/object_pose_tracking/object_pose_tracking"
sys.path.insert(0, str(PACKAGE))
from protocol import read_frame


class CameraWorkerIntegrationTests(unittest.TestCase):
    def test_backend_selection_and_stdout_isolation_preserve_packet(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            (project / "camera.py").write_text('''
import os
from types import SimpleNamespace
import numpy as np

def open_camera(backend, serial):
    assert backend == "orbbec" and serial == "gemini-serial"
    os.write(1, b"native SDK startup log\\n")
    print("Python camera startup log", flush=True)
    return Camera()

class Camera:
    def __init__(self):
        self.sent = False
    def read(self, near, far, fov_keep, downscale):
        assert (near, far, fov_keep, downscale) == (0.25, 1.5, 1.0, 2)
        if self.sent:
            raise SystemExit(0)
        self.sent = True
        return SimpleNamespace(
            color=np.full((2, 3, 3), [255, 64, 0], np.uint8),
            depth=np.full((2, 3), 725, np.uint16),
            intrinsic=np.array([[100., 0., 1.], [0., 120., .5], [0., 0., 1.]]),
            timestamp_ns=1720000000123456789)
    def close(self):
        print("Camera closed", flush=True)
''')
            worker = subprocess.run([
                sys.executable, str(PACKAGE / "camera_worker.py"),
                "--project-root", str(project), "--camera-backend", "orbbec",
                "--serial", "gemini-serial", "--downscale", "2",
                "--minimum-depth", "0.25", "--maximum-depth", "1.5",
            ], capture_output=True, timeout=15, check=True)
            stream = io.BytesIO(worker.stdout)
            rgb, depth, intrinsic, timestamp = read_frame(stream)
            self.assertEqual(stream.read(), b"")
            np.testing.assert_array_equal(rgb, np.full((2, 3, 3), [255, 64, 0], np.uint8))
            np.testing.assert_array_equal(depth, np.full((2, 3), 725, np.uint16))
            np.testing.assert_allclose(intrinsic, [[100., 0., 1.], [0., 120., .5], [0., 0., 1.]])
            self.assertEqual(timestamp, 1720000000123456789)
            self.assertIn(b"native SDK startup log", worker.stderr)
            self.assertIn(b"Python camera startup log", worker.stderr)
            self.assertIn(b"Camera closed", worker.stderr)


if __name__ == "__main__":
    unittest.main()

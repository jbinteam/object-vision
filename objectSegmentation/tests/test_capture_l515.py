import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from capture_l515 import capture, parse_args


class CaptureTests(unittest.TestCase):
    def fake_sdk(self):
        device = MagicMock()
        device.get_info.side_effect = lambda key: {"name": "Intel RealSense L515", "serial": "test123", "firmware": "test"}[key]
        pipeline = MagicMock()
        color = pipeline.wait_for_frames.return_value.get_color_frame.return_value
        pixels = np.full((480, 640, 3), (10, 80, 230), dtype=np.uint8)
        color.get_data.return_value = pixels
        color.get_timestamp.return_value = 1234.5
        color.get_frame_number.return_value = 42
        rs = SimpleNamespace(context=MagicMock(), pipeline=MagicMock(return_value=pipeline),
                             config=MagicMock(), camera_info=SimpleNamespace(name="name", serial_number="serial", firmware_version="firmware"),
                             stream=SimpleNamespace(color="color"), format=SimpleNamespace(bgr8="bgr8"))
        rs.context.return_value.query_devices.return_value = [device]
        return rs, pipeline, pixels

    def test_timed_capture_count_pixels_and_metadata(self):
        rs, pipeline, pixels = self.fake_sdk()
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(["--output", directory, "--width", "640", "--height", "480",
                               "--auto", "--no-preview", "--count", "2", "--warmup", "0"])
            with patch("capture_l515.color_modes", return_value=[(640, 480, 30)]), \
                 patch("capture_l515.time.monotonic", side_effect=itertools.count()), \
                 patch("builtins.print"):
                capture(args, rs)
            session = next(Path(directory).iterdir())
            rows = [json.loads(line) for line in (session / "frames.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(list(session.glob("*.png"))), 2)
            for row in rows:
                np.testing.assert_array_equal(cv2.imread(str(session / row["file"])), pixels)
                self.assertEqual(row["camera_timestamp_ms"], 1234.5)
            pipeline.stop.assert_called_once()

    def test_disconnect_releases_camera(self):
        rs, pipeline, _ = self.fake_sdk()
        pipeline.wait_for_frames.side_effect = RuntimeError("disconnected")
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(["--output", directory, "--auto", "--no-preview"])
            with patch("capture_l515.color_modes", return_value=[(1920, 1080, 30)]), patch("builtins.print"):
                with self.assertRaisesRegex(RuntimeError, "disconnected"):
                    capture(args, rs)
            pipeline.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()

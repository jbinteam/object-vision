import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from l515_protocol import read_message, write_message
from test_l515 import CameraStream, save_snapshot


class ProtocolTests(unittest.TestCase):
    def test_multiple_messages_preserve_frame_bytes(self):
        reader, writer = os.pipe()
        with os.fdopen(reader, "rb", buffering=0) as source, os.fdopen(writer, "wb", buffering=0) as sink:
            payload = bytes(range(256)) * 3
            write_message(sink, {"type": "ready"})
            write_message(sink, {"type": "frame", "shape": [16, 16, 3]}, payload)
            self.assertEqual(read_message(source)[0]["type"], "ready")
            metadata, received = read_message(source)
            self.assertEqual(metadata["shape"], [16, 16, 3])
            self.assertEqual(received, payload)

    def test_idle_and_closed_pipe_fail_promptly(self):
        reader, writer = os.pipe()
        with os.fdopen(reader, "rb", buffering=0) as source:
            try:
                with self.assertRaises(TimeoutError):
                    read_message(source, timeout=0.01)
            finally:
                os.close(writer)
            with self.assertRaises(EOFError):
                read_message(source)

    def test_oversized_header_rejected(self):
        reader, writer = os.pipe()
        with os.fdopen(reader, "rb", buffering=0) as source, os.fdopen(writer, "wb", buffering=0) as sink:
            sink.write(struct.pack("<I", 65537))
            with self.assertRaisesRegex(ValueError, "header"):
                read_message(source)


class LiveTests(unittest.TestCase):
    def test_startup_error_closes_and_terminates_unresponsive_worker(self):
        args = SimpleNamespace(camera_python=Path("python"), width=1280, height=720, fps=30, serial=None)
        process = MagicMock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("worker", 3), 0]
        with patch("test_l515.subprocess.Popen", return_value=process), \
             patch("test_l515.read_message", side_effect=TimeoutError("startup timeout")):
            with self.assertRaises(TimeoutError):
                with CameraStream(args):
                    self.fail("Startup should fail")
        process.terminate.assert_called_once()
        process.stdin.close.assert_called_once()
        process.stdout.close.assert_called_once()

    def test_snapshots_preserve_pixels_and_instance_masks(self):
        frame = np.full((12, 16, 3), (12, 80, 200), np.uint8)
        mask = np.zeros((1, 12, 16), np.float32)
        mask[:, 3:9, 4:12] = 1
        result = MagicMock()
        result.masks.data.cpu.return_value.numpy.return_value = mask
        result.summary.return_value = [{"class": 0, "confidence": 0.8}]
        with tempfile.TemporaryDirectory() as temporary, patch("builtins.print"):
            folder = Path(temporary)
            save_snapshot(folder, 1, frame, frame, result, {"index": 1})
            saved = folder / "snapshot_000001"
            np.testing.assert_array_equal(cv2.imread(str(saved / "color.png")), frame)
            np.testing.assert_array_equal(cv2.imread(str(saved / "mask_000.png"), 0), mask[0] * 255)
            metadata = json.loads((saved / "predictions.json").read_text())
            self.assertEqual(metadata["instances"][0]["class"], 0)
            result.summary.assert_called_once_with(normalize=True, decimals=6)
            result.masks = None
            result.summary.return_value = []
            save_snapshot(folder, 2, frame, frame, result, {"index": 2})
            self.assertEqual(list((folder / "snapshot_000002").glob("mask_*.png")), [])

    def test_snapshot_rejects_wrong_mask_resolution(self):
        result = MagicMock()
        result.masks.data.cpu.return_value.numpy.return_value = np.ones((1, 2, 2))
        frame = np.zeros((12, 16, 3), np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "resolution"):
                save_snapshot(Path(temporary), 1, frame, frame, result, {})


if __name__ == "__main__":
    unittest.main()

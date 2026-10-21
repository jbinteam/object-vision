import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import logging

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import open3d as o3d

from camera import prepare_depth
from reconstruction import TensorSlam, _isolate_object_cloud, plausible_motion
from scanner import arguments, output_paths, save_scan, _capture


class ScannerTests(unittest.TestCase):
    def test_pose_guard_rejects_large_and_nonrigid_transforms(self):
        self.assertTrue(plausible_motion(np.eye(4)))
        jump = np.eye(4)
        jump[0, 3] = 1
        self.assertFalse(plausible_motion(jump))
        self.assertFalse(plausible_motion(np.full((4, 4), np.nan)))
        scaled = np.eye(4)
        scaled[0, 0] = 2
        self.assertFalse(plausible_motion(scaled))

    def test_safe_mode_disables_cuda_and_windows(self):
        args = arguments(["--safe", "--device", "cuda", "--voxel", "3"])
        self.assertEqual(args.device, "cpu")
        self.assertTrue(args.no_view)
        self.assertGreaterEqual(args.downscale, 2)
        self.assertGreaterEqual(args.voxel, 5)
        self.assertEqual(arguments([]).device, "cpu")

    def test_only_singular_tracking_errors_are_recoverable(self):
        slam = TensorSlam(o3d.core.Tensor([[100., 0., 32.], [0., 100., 24.], [0., 0., 1.]]),
                          64, 48, 5, 1.5, o3d.core.Device("CPU:0"), 1000, 0.3)
        slam.frame_idx = 1
        slam.model = Mock()
        rgb = np.zeros((48, 64, 3), np.uint8)
        depth = np.full((48, 64), 700, np.uint16)
        slam.model.track_frame_to_model.side_effect = RuntimeError("Singular 6x6 linear system detected, tracking failed.")
        self.assertFalse(slam.process(rgb, depth))
        for error in ("CUDA illegal memory access", "out of memory", "unexpected backend error"):
            slam.model.track_frame_to_model.side_effect = RuntimeError(error)
            with self.assertRaisesRegex(RuntimeError, error):
                slam.process(rgb, depth)
        slam.model.integrate.assert_not_called()

    def test_fatal_reconstruction_error_never_exports(self):
        args = arguments(["--safe"])
        frame = Mock(depth=np.full((48, 64), 700, np.uint16),
                     color=np.zeros((48, 64, 3), np.uint8), intrinsic=np.eye(3))
        camera = Mock(info={"name": "test"})
        camera.read.return_value = frame
        slam = Mock(tracked_frames=5, lost_frames=0)
        slam.process.side_effect = RuntimeError("CUDA illegal memory access")
        with patch("camera.open_camera", return_value=camera), patch("reconstruction.TensorSlam", return_value=slam), patch("scanner.save_scan") as save:
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "CUDA illegal"):
                    _capture(args, {"log": "test.log"}, o3d.core.Device("CPU:0"), Mock())
            camera.close.assert_called_once()
            save.assert_not_called()
            slam.extract_point_cloud.assert_not_called()

    def test_new_camera_rejects_old_calibration_before_reconstruction(self):
        with tempfile.TemporaryDirectory() as temporary:
            calibration = Path(temporary) / "calibration.json"
            calibration.write_text(json.dumps({"tag_family": "tagStandard41h12",
                                               "camera_serial": "old-l515"}))
            args = arguments(["--anchor-calibration", str(calibration), "--safe"])
            camera = Mock(info={"name": "Gemini 305", "serial": "new-gemini", "backend": "orbbec"})
            with patch("camera.open_camera", return_value=camera) as factory, \
                    patch("turntable_dataset.make_detector"), \
                    patch("reconstruction.TensorSlam") as slam:
                with self.assertRaisesRegex(RuntimeError, "different camera"):
                    _capture(args, {}, o3d.core.Device("CPU:0"), Mock())
                factory.assert_called_once_with("orbbec", None)
                camera.close.assert_called_once()
                camera.read.assert_not_called()
                slam.assert_not_called()

    def test_l515_depth_scale_and_distance_filter(self):
        raw = np.array([[0, 1000, 4000, 8000, 16000]], dtype=np.uint16)
        result = prepare_depth(raw, 0.00025, 0.25, 1.5, 1)
        np.testing.assert_array_equal(result, [[0, 250, 1000, 0, 0]])
        self.assertEqual(result.dtype, np.uint16)

    def test_roi_masks_all_four_edges(self):
        depth = prepare_depth(np.full((10, 10), 1000, np.uint16), 0.001, 0.25, 1.5, 0.6)
        self.assertEqual(np.count_nonzero(depth), 36)
        self.assertTrue(np.all(depth[2:8, 2:8] == 1000))

    def test_cli_rejects_invalid_limits(self):
        for flags in (["--depth-min", "2", "--depth-max", "1"], ["--voxel", "nan"],
                      ["--seconds", "0"], ["--fov-keep", "0"], ["--output", "scan.obj"]):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    arguments(flags)

    def test_existing_scan_is_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "object.ply"
            path.write_text("existing scan")
            with self.assertRaises(RuntimeError):
                output_paths(arguments(["-o", str(path)]))
            self.assertEqual(path.read_text(), "existing scan")

    def test_empty_frames_do_not_seed_model(self):
        slam = TensorSlam(o3d.core.Tensor([[100., 0., 32.], [0., 100., 24.], [0., 0., 1.]]),
                          64, 48, 5, 1.5, o3d.core.Device("CPU:0"), 1000, 0.3)
        self.assertFalse(slam.process(np.zeros((48, 64, 3), np.uint8), np.zeros((48, 64), np.uint16)))
        self.assertEqual(slam.frame_idx, 0)
        self.assertEqual(slam.lost_frames, 1)

    def test_isolation_keeps_largest_cluster(self):
        rng = np.random.default_rng(4)
        points = np.vstack([rng.normal(0, 0.002, (200, 3)), rng.normal(0.1, 0.002, (110, 3))])
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        result, _ = _isolate_object_cloud(cloud, False, 0.008, 0.015, 10)
        self.assertEqual(len(result.points), 200)

    def test_reconstruction_export_roundtrip(self):
        # A curved surface avoids a singular planar odometry system.
        h, w = 120, 160
        yy, xx = np.mgrid[:h, :w]
        depth = (700 + 70 * np.sin(xx / 17) * np.cos(yy / 23) + xx / 2).astype(np.uint16)
        color = np.full((h, w, 3), [190, 100, 40], np.uint8)
        k = np.array([[160., 0, 80], [0, 160., 60], [0, 0, 1.]])
        slam = TensorSlam(o3d.core.Tensor(k), w, h, 5, 1.5, o3d.core.Device("CPU:0"), 1000, 0.1)
        for _ in range(4):
            self.assertTrue(slam.process(color, depth))
        with tempfile.TemporaryDirectory() as tmp:
            args = arguments(["-o", str(Path(tmp) / "scan.ply"), "--mesh"])
            paths = output_paths(args)
            anchor = {"frame_id": "box_center", "T_scan_object": np.eye(4).tolist()}
            save_scan(slam, k, args, paths, {"name": "synthetic test"}, anchor)
            cloud = o3d.io.read_point_cloud(str(paths["cloud"]))
            self.assertGreater(len(cloud.points), 100)
            self.assertTrue(cloud.has_colors())
            self.assertTrue(np.isfinite(np.asarray(cloud.points)).all())
            self.assertTrue(all(p.is_file() for name, p in paths.items() if name != "log"))
            metadata = json.loads(paths["metadata"].read_text())
            self.assertEqual(metadata["reference_anchor"], anchor)


if __name__ == "__main__":
    unittest.main()

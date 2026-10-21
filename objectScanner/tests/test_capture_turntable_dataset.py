import json
import tempfile
import unittest
from unittest.mock import Mock, patch
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from scripts.capture_turntable_dataset import (
    arguments, canonical_reference, main, reference_scan_command,
    resumable_empty_scaffold,
)
from scripts.capture_nocs_dataset import arguments as nocs_arguments


class CaptureTurntableArgumentsTest(unittest.TestCase):
    def test_orbbec_close_range_reaches_capture_and_diagnostic_scan(self):
        for parse in (arguments, nocs_arguments):
            with self.subTest(entry_point=parse.__module__):
                args = parse(["--name", "widget_gemini305"])
                self.assertEqual(args.camera_backend, "orbbec")
                self.assertEqual((args.depth_min, args.depth_max), (0.04, 1.5))
                command = reference_scan_command(args, Path("/tmp/diagnostic.ply"))
                self.assertEqual(command[command.index("--depth-min") + 1], "0.04")
                self.assertEqual(command[command.index("--camera-backend") + 1], "orbbec")

    def test_legacy_depth_defaults_and_explicit_overrides(self):
        for backend in ("realsense", "auto", "orbbec"):
            with self.subTest(backend=backend):
                base = ["--name", "widget", "--camera-backend", backend]
                args = arguments(base)
                self.assertEqual(args.depth_min, 0.04 if backend == "orbbec" else 0.25)
                args = arguments(base + ["--depth-min", "0.12", "--depth-max", "0.8"])
                self.assertEqual((args.depth_min, args.depth_max), (0.12, 0.8))
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                    arguments(base + ["--depth-min", "0"])

    def test_new_camera_cannot_rewrite_old_session_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "widget"
            output.mkdir()
            calibration = {"setup_ids": [1, 2, 3, 4],
                           "permanent_ids": [5, 6, 7, 8, 9, 10],
                           "setup_tag_size_m": .014, "permanent_tag_size_m": .028,
                           "tag_family": "tagStandard41h12", "camera_serial": "old-l515"}
            (output / "calibration.json").write_text(json.dumps(calibration))
            annotations = output / "centerpose/outf/widget_train/sample.json"
            annotations.parent.mkdir(parents=True)
            annotations.write_text("preserve old annotations")
            before = {str(path.relative_to(output)): path.read_bytes()
                      for path in output.rglob("*") if path.is_file()}
            before_directories = {str(path.relative_to(output))
                                  for path in output.rglob("*") if path.is_dir()}
            camera = Mock(info={"serial": "new-gemini", "backend": "orbbec"})
            with patch("scripts.capture_turntable_dataset.make_detector"), \
                    patch("scripts.capture_turntable_dataset.open_camera", return_value=camera), \
                    patch("scripts.capture_turntable_dataset.export_centerpose_dataset") as exporter, \
                    patch("scripts.capture_turntable_dataset.cv2.namedWindow") as window:
                with self.assertRaisesRegex(SystemExit, "different camera"):
                    main(["--output", str(output)])
                exporter.assert_not_called()
                window.assert_not_called()
                camera.close.assert_called_once()
                camera.read.assert_not_called()
            after = {str(path.relative_to(output)): path.read_bytes()
                     for path in output.rglob("*") if path.is_file()}
            self.assertEqual(after, before)
            self.assertEqual({str(path.relative_to(output))
                              for path in output.rglob("*") if path.is_dir()}, before_directories)

    def test_name_selects_dataset_directory(self):
        args = arguments(["--name", "widget_front"])
        self.assertEqual(args.dataset_name, "widget_front")
        self.assertEqual(args.output, Path("datasets/widget_front"))
        self.assertEqual(args.centerpose_category, "widget_front")
        self.assertEqual(args.centerpose_split, "train")

    def test_dataset_name_alias(self):
        args = arguments(["--dataset-name", "widget_back"])
        self.assertEqual(args.output, Path("datasets/widget_back"))

    def test_explicit_output_remains_supported(self):
        args = arguments(["--output", "/tmp/custom-dataset"])
        self.assertIsNone(args.dataset_name)
        self.assertEqual(args.output, Path("/tmp/custom-dataset"))
        self.assertEqual(args.centerpose_category, "custom-dataset")

    def test_centerpose_category_and_split_can_be_overridden(self):
        args = arguments(["--name", "session_01", "--centerpose-category", "widget",
                          "--centerpose-split", "val"])
        self.assertEqual(args.centerpose_category, "widget")
        self.assertEqual(args.centerpose_split, "val")

    def test_nocs_entry_point_accepts_short_category_and_split_aliases(self):
        args = nocs_arguments([
            "--name", "widget_nocs", "--category", "widget", "--split", "val"])
        self.assertEqual(args.output, Path("datasets/widget_nocs"))
        self.assertEqual(args.centerpose_category, "widget")
        self.assertEqual(args.centerpose_split, "val")

    def test_name_rejects_paths(self):
        for invalid in ("", ".", "..", "nested/name", "nested\\name", " padded"):
            with self.subTest(invalid=invalid), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    arguments(["--name", invalid])

    def test_name_and_output_are_mutually_exclusive(self):
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                arguments(["--name", "widget", "--output", "datasets/other"])

    def test_only_matching_empty_startup_scaffold_is_resumable(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for relative in ("images", "raw_images", "depths", "labels",
                             "centerpose/outf/widget_train"):
                (output / relative).mkdir(parents=True, exist_ok=True)
            metadata = {"category": "widget", "split": "train"}
            (output / "centerpose/export.json").write_text(json.dumps(metadata))
            self.assertTrue(resumable_empty_scaffold(output, "widget", "train"))
            self.assertFalse(resumable_empty_scaffold(output, "widget", "val"))

            (output / "images/partial.png").write_bytes(b"not an image")
            self.assertFalse(resumable_empty_scaffold(output, "widget", "train"))

    def test_unknown_file_is_not_treated_as_empty_scaffold(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "notes.txt").write_text("keep me")
            self.assertFalse(resumable_empty_scaffold(output, "widget", "train"))

    def test_reference_scan_options_and_command(self):
        args = arguments([
            "--name", "widget", "--camera-backend", "realsense", "--serial", "123", "--scan-seconds", "12",
            "--scan-voxel-mm", "2", "--scan-device", "cuda",
        ])
        command = reference_scan_command(args, Path("/tmp/widget_scan.ply"))
        self.assertIn("scanner.py", " ".join(command))
        self.assertIn("/tmp/widget_scan.ply", command)
        self.assertEqual(args.scan_downscale, 1)
        self.assertEqual(args.scan_device, "cuda")
        self.assertEqual(args.scan_stall_timeout, 60)
        self.assertEqual(command[command.index("--camera-backend") + 1], "realsense")
        self.assertIn("--downscale", command)
        self.assertIn("--stall-timeout", command)
        self.assertEqual(command[-2:], ["--serial", "123"])
        command = reference_scan_command(
            args, Path("/tmp/widget_scan.ply"), Path("/tmp/calibration.json"))
        self.assertEqual(command[-2:], ["--anchor-calibration", "/tmp/calibration.json"])

    def test_canonical_reference_prefers_category_and_rejects_ambiguity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "references/first/reference.ply"
            first.parent.mkdir(parents=True)
            first.write_bytes(b"ply")
            self.assertEqual(canonical_reference(root, "widget"), first.resolve())
            second = root / "references/second/reference.ply"
            second.parent.mkdir(parents=True)
            second.write_bytes(b"ply")
            self.assertIsNone(canonical_reference(root, "widget"))
            expected = root / "references/widget/reference.ply"
            expected.parent.mkdir(parents=True)
            expected.write_bytes(b"ply")
            self.assertEqual(canonical_reference(root, "widget"), expected.resolve())


if __name__ == "__main__":
    unittest.main()

"""Check workflow command routing without opening cameras or starting training."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CollectionProcessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name) / "objectScanner"
        self.wrappers = self.project / "scripts/collect_process"
        self.wrappers.mkdir(parents=True)
        for source in (ROOT / "scripts/collect_process").glob("*.sh"):
            shutil.copy2(source, self.wrappers / source.name)
        self.log = self.project / "commands.jsonl"
        # Every command that could start hardware, a server, or training is a recorder.
        recorder = (f"#!{sys.executable}\nimport json, os, sys\n"
                    "with open(os.environ['COLLECT_COMMAND_LOG'], 'a') as stream:\n"
                    "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n")
        for relative in (".venv/bin/python", ".venv-ros-cuda/bin/python",
                         "../objectSegmentation/.venv/bin/python",
                         "scripts/run_phase4_cuda.sh"):
            executable = self.project / relative
            executable.parent.mkdir(parents=True, exist_ok=True)
            executable.write_text(recorder)
            executable.chmod(0o755)
        (self.project.parent / "objectSegmentation/backgrounds/picsum").mkdir(parents=True)
        self.environment = {key: value for key, value in os.environ.items()
                            if not key.startswith("COLLECT_") and key != "POSE_START_WEIGHTS"}
        self.environment["COLLECT_COMMAND_LOG"] = str(self.log)

    def prepare_session(self, session, category):
        dataset = self.project / "datasets" / session
        for relative in ("centerpose_masks/masks", "reference_scans"):
            (dataset / relative).mkdir(parents=True)
        for relative in ("reference_scans/diagnostic.ply",
                         f"references/{category}/reference.ply",
                         f"references/{category}/frame.json",
                         "training_pnp/calibration.json", "training_pnp/object_keypoints.yaml",
                         "training_pnp/manifest.json", "yolo_seg_pose_onthefly/segmentation/data.yaml",
                         "yolo_seg_pose_onthefly/pose/data.yaml"):
            path = dataset / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        for run in (f"{session}_onthefly_seg", f"{session}_pose_refined_pose",
                    "my_object_rgbd_unfrozen_pose"):
            weights = self.project / "training/yolo" / run / "weights/best.pt"
            weights.parent.mkdir(parents=True, exist_ok=True)
            weights.touch()
        return dataset

    def run_wrapper(self, name, *arguments, environment=None):
        self.log.unlink(missing_ok=True)
        subprocess.run(["bash", str(self.wrappers / name), *arguments],
                       env={**self.environment, **(environment or {})},
                       check=True, capture_output=True, text=True, timeout=10)
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_default_capture_uses_new_orbbec_session(self):
        command, = self.run_wrapper("01_capture.sh")
        self.assertEqual(command[command.index("--name") + 1], "my_object_gemini305")
        self.assertEqual(command[command.index("--centerpose-category") + 1], "my_object")
        self.assertEqual(command[command.index("--camera-backend") + 1], "orbbec")

    def test_default_steps_route_to_same_dataset_and_pinned_artifacts(self):
        dataset = self.prepare_session("my_object_gemini305", "my_object")
        for name in ("02_generate_masks.sh", "03_export_yolo.sh", "04_build_reference.sh",
                     "04_align_reference.sh", "05_train_segmentation.sh", "06_train_pose.sh"):
            with self.subTest(name=name):
                commands = self.run_wrapper(name)
                command = commands[-1]
                expected = dataset
                if name.startswith(("05_", "06_")):
                    expected /= "yolo_seg_pose_onthefly"
                    self.assertIn([str(self.project / "scripts/pnp_snapshot.py"),
                                   "save", str(dataset)], commands)
                self.assertEqual(command[command.index("--dataset") + 1], str(expected))
        live = self.run_wrapper("07_run_live.sh")
        self.assertIn([str(self.project / "scripts/pnp_snapshot.py"), "verify", str(dataset)], live)
        self.assertIn("camera_backend:=orbbec", live[-1])
        self.assertIn(f"keypoints:={dataset}/training_pnp/object_keypoints.yaml", live[-1])
        self.assertIn(f"reference:={dataset}/references/my_object/reference.ply", live[-1])
        self.assertIn(f"pose_weights:={self.project}/training/yolo/my_object_gemini305_pose_refined_pose/weights/best.pt",
                      live[-1])
        for name in ("01b_edit_reference.sh", "04b_crop_reference.sh"):
            with self.subTest(name=name):
                command, = self.run_wrapper(name)
                self.assertTrue(any(value.startswith(str(dataset)) for value in command))

    def test_environment_preserves_legacy_session_category_and_backend(self):
        dataset = self.prepare_session("my_object_20260925_r2", "my_object")
        environment = {"COLLECT_SESSION": "my_object_20260925_r2",
                       "COLLECT_CAMERA_BACKEND": "realsense"}
        capture, = self.run_wrapper("01_capture.sh", environment=environment)
        self.assertEqual(capture[capture.index("--name") + 1], dataset.name)
        self.assertEqual(capture[capture.index("--centerpose-category") + 1], "my_object")
        self.assertEqual(capture[capture.index("--camera-backend") + 1], "realsense")
        live = self.run_wrapper("07_run_live.sh", environment=environment)[-1]
        self.assertIn("camera_backend:=realsense", live)
        self.assertIn(f"reference:={dataset}/references/my_object/reference.ply", live)

    def test_explicit_arguments_override_environment_and_backend_default(self):
        dataset = self.prepare_session("other_session", "widget")
        environment = {"COLLECT_SESSION": "ignored", "COLLECT_CATEGORY": "ignored",
                       "COLLECT_CAMERA_BACKEND": "orbbec"}
        capture, = self.run_wrapper("01_capture.sh", dataset.name, "widget",
                                    "--camera-backend", "realsense", "--serial", "legacy",
                                    environment=environment)
        self.assertEqual(capture[capture.index("--name") + 1], dataset.name)
        self.assertEqual(capture[capture.index("--centerpose-category") + 1], "widget")
        self.assertEqual(capture[-4:], ["--camera-backend", "realsense", "--serial", "legacy"])
        live = self.run_wrapper("07_run_live.sh", dataset.name, "widget",
                                "camera_backend:=realsense", environment=environment)[-1]
        self.assertEqual(live[-1], "camera_backend:=realsense")
        self.assertIn(f"reference:={dataset}/references/widget/reference.ply", live)


if __name__ == "__main__":
    unittest.main()

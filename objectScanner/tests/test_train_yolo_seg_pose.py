from pathlib import Path
import tempfile
import unittest

from scripts.train_yolo_seg_pose import paired_counts, parse_args


class TrainYoloSegPoseTest(unittest.TestCase):
    def test_paired_counts_requires_matching_stems(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for task in ("segmentation", "pose"):
                (root / task / "data.yaml").parent.mkdir(parents=True)
                (root / task / "data.yaml").write_text("names:\n  0: widget\n")
                for split in ("train", "val"):
                    (root / task / "images" / split).mkdir(parents=True)
                    (root / task / "labels" / split).mkdir(parents=True)
                    (root / task / "masks" / split).mkdir(parents=True)
                    (root / task / "images" / split / "one.png").write_bytes(b"image")
                    (root / task / "labels" / split / "one.txt").write_text("0 0 0\n")
                    (root / task / "masks" / split / "one.png").write_bytes(b"mask")
            self.assertEqual(paired_counts(root)["pose"], {"train": 1, "val": 1})
            (root / "pose/labels/train/one.txt").rename(
                root / "pose/labels/train/two.txt")
            with self.assertRaisesRegex(ValueError, "Unpaired"):
                paired_counts(root)

    def test_dry_run_arguments(self):
        args = parse_args(["--dry-run", "--task", "pose", "--epochs", "2"])
        self.assertTrue(args.dry_run)
        self.assertEqual(args.task, "pose")
        self.assertEqual(args.epochs, 2)
        self.assertEqual(args.freeze, 11)
        self.assertEqual(args.background_probability, 0.85)
        self.assertEqual(args.epoch_repeats, 12)
        self.assertFalse(args.cache)

    def test_epoch_repeats_must_be_positive(self):
        with self.assertRaises(SystemExit):
            parse_args(["--epoch-repeats", "0"])


if __name__ == "__main__":
    unittest.main()

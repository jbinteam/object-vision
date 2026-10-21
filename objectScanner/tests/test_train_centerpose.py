import json
from pathlib import Path
import tempfile
import unittest

from scripts.train_centerpose import (arguments, prepare_training_split,
                                      validate_prepared_split)


class CenterPoseTrainingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.augmented = self.root / "augmented"
        self.pairs = self.augmented / "outf/widget_train"
        self.pairs.mkdir(parents=True)
        records = []
        for source_index in range(10):
            stem = f"view_{source_index:02d}"
            generated = []
            for variant in range(3):
                suffix = "" if variant == 0 else f"__bg_{variant:03d}"
                name = f"{stem}{suffix}.png"
                (self.pairs / name).write_bytes(b"png")
                (self.pairs / Path(name).with_suffix(".json")).write_text("{}")
                generated.append({"name": name, "background": None if variant == 0 else "bg"})
            records.append({
                "source_image": f"source/{stem}.png",
                "generated": generated,
            })
        (self.augmented / "manifest.json").write_text(json.dumps({
            "category": "widget",
            "split": "train",
            "records": records,
        }))

    def test_source_grouped_split_has_no_augmented_validation_leakage(self):
        output = self.root / "training"
        split = prepare_training_split(
            self.augmented, output, val_fraction=0.2, seed=5)
        self.assertEqual(split["pairs"], {"train": 24, "val": 2})
        self.assertEqual(
            validate_prepared_split(output, "widget"), {"train": 24, "val": 2})
        validation = {path.stem for path in (output / "outf/widget_val").glob("*.png")}
        training = {path.stem.split("__bg_")[0]
                    for path in (output / "outf/widget_train").glob("*.png")}
        self.assertTrue(validation.isdisjoint(training))
        self.assertTrue(all(path.is_symlink()
                            for path in (output / "outf/widget_train").iterdir()))

    def test_preparation_is_idempotent_for_same_manifest(self):
        output = self.root / "training"
        first = prepare_training_split(self.augmented, output, 0.2, 11)
        second = prepare_training_split(self.augmented, output, 0.2, 11)
        self.assertEqual(first, second)

    def test_changed_split_requires_explicit_rebuild(self):
        output = self.root / "training"
        prepare_training_split(self.augmented, output, 0.2, 11)
        with self.assertRaisesRegex(ValueError, "--rebuild-split"):
            prepare_training_split(self.augmented, output, 0.3, 11)
        rebuilt = prepare_training_split(
            self.augmented, output, 0.3, 11, rebuild=True)
        self.assertEqual(rebuilt["pairs"], {"train": 21, "val": 3})

    def test_num_iters_accepts_complete_or_positive_values(self):
        self.assertEqual(arguments([]).num_iters, -1)
        self.assertEqual(arguments(["--num-iters", "3"]).num_iters, 3)
        with self.assertRaises(SystemExit):
            arguments(["--num-iters", "0"])


if __name__ == "__main__":
    unittest.main()

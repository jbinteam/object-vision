from pathlib import Path
import random
import tempfile
import unittest

import cv2
import numpy as np

from background_augmentation import BackgroundReplacer, BackgroundYOLODataset
from ultralytics.cfg import get_cfg


class BackgroundTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for folder in ["images", "masks", "labels", "backgrounds"]:
            (self.root / folder).mkdir()
        self.image = np.full((64, 96, 3), (40, 60, 80), np.uint8)
        self.image[16:48, 24:72] = (0, 0, 255)
        self.mask = np.zeros((64, 96), np.uint8)
        self.mask[16:48, 24:72] = 255
        cv2.imwrite(str(self.root / "images/sample.png"), self.image)
        cv2.imwrite(str(self.root / "masks/sample.png"), self.mask)
        (self.root / "labels/sample.txt").write_text("0 0.25 0.25 0.75 0.25 0.75 0.75 0.25 0.75\n")
        cv2.imwrite(str(self.root / "backgrounds/green.png"), np.full((100, 120, 3), (0, 255, 0), np.uint8))
        cv2.imwrite(str(self.root / "backgrounds/blue.png"), np.full((100, 120, 3), (255, 0, 0), np.uint8))

    def dataset(self, augment):
        return BackgroundYOLODataset(
            img_path=str(self.root / "images"), imgsz=96, batch_size=1, augment=augment,
            hyp=get_cfg(overrides={"task": "segment", "mosaic": 0.0}), task="segment",
            data={"names": {0: "test"}, "nc": 1}, dataset_root=self.root,
            background_dir=self.root / "backgrounds", background_probability=1.0,
        )

    def test_foreground_is_exact_and_source_is_not_mutated(self):
        replacer = BackgroundReplacer(self.root / "backgrounds")
        original = self.image.copy()
        output = replacer.composite(self.image, self.mask, random.Random(1))
        np.testing.assert_array_equal(output[self.mask > 0], original[self.mask > 0])
        self.assertTrue(np.all(np.any(output[self.mask == 0] != original[self.mask == 0], axis=1)))
        np.testing.assert_array_equal(self.image, original)

    def test_each_read_varies_without_moving_labels(self):
        self.addCleanup(random.setstate, random.getstate())
        random.seed(42)
        dataset = self.dataset(True)
        samples = [dataset.get_image_and_label(0) for _ in range(20)]
        self.assertGreater(len({sample["img"].tobytes() for sample in samples}), 1)
        for sample in samples[1:]:
            np.testing.assert_array_equal(sample["instances"].segments, samples[0]["instances"].segments)
            np.testing.assert_array_equal(sample["cls"], samples[0]["cls"])

    def test_validation_is_unchanged_even_with_probability_one(self):
        dataset = self.dataset(False)
        output = dataset.get_image_and_label(0)["img"]
        np.testing.assert_array_equal(output, self.image)

    def test_missing_mask_fails_instead_of_silently_training_wrong_data(self):
        (self.root / "masks/sample.png").unlink()
        with self.assertRaisesRegex(ValueError, "Foreground mask missing"):
            self.dataset(True)


if __name__ == "__main__":
    unittest.main()

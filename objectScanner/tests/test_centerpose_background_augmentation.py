import json
from pathlib import Path
import random
import tempfile
import unittest

import cv2
import numpy as np

from scripts.augment_centerpose_backgrounds import (BackgroundReplacer,
                                                    bright_neutral_disconnect_mask,
                                                    inpaint_exclusion_mask,
                                                    refine_mask,
                                                    mask_metrics,
                                                    paired_sources,
                                                    pose_refine_mask)


class CenterPoseBackgroundAugmentationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "backgrounds").mkdir()
        cv2.imwrite(str(self.root / "backgrounds/a.png"),
                    np.full((100, 140, 3), (0, 255, 0), np.uint8))
        cv2.imwrite(str(self.root / "backgrounds/b.png"),
                    np.full((120, 160, 3), (255, 0, 0), np.uint8))

    def test_composite_keeps_foreground_pixels_and_geometry(self):
        image = np.full((60, 80, 3), (10, 20, 30), np.uint8)
        image[20:45, 25:60] = (0, 0, 255)
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[20:45, 25:60] = 255
        original = image.copy()
        composite, background, transform = BackgroundReplacer(
            self.root / "backgrounds").composite(image, mask, random.Random(4))
        np.testing.assert_array_equal(composite[mask > 0], original[mask > 0])
        self.assertTrue(np.any(composite[mask == 0] != original[mask == 0]))
        np.testing.assert_array_equal(image, original)
        self.assertTrue(background.is_file())
        self.assertIn("crop_xy", transform)

    def test_mask_metrics_compare_segmentation_to_pose_box(self):
        mask = np.zeros((100, 120), np.uint8)
        mask[30:70, 40:80] = 255
        label = {"bbox_xyxy_px_clipped": [35, 25, 85, 75],
                 "center_uv_px": [60, 50]}
        metrics = mask_metrics(mask, label)
        self.assertTrue(metrics["object_center_inside_mask"])
        self.assertEqual(metrics["foreground_inside_projected_bbox"], 1.0)
        self.assertGreater(metrics["bbox_iou"], 0.6)

    def test_pose_refinement_removes_cable_outside_cuboid(self):
        raw = np.zeros((120, 160), np.uint8)
        raw[30:85, 55:105] = 255
        raw[75:115, 78:84] = 255
        label = {"bbox_xyxy_px_clipped": [50, 25, 110, 90],
                 "center_uv_px": [80, 58]}
        refined = pose_refine_mask(raw, label, kernel_size=9, padding_ratio=0)
        self.assertTrue(refined[58, 80])
        self.assertFalse(refined[110, 80])
        self.assertGreater(np.count_nonzero(refined), 2000)

    def test_inpainted_tag_region_can_be_excluded_from_mask(self):
        raw_image = np.zeros((120, 160, 3), np.uint8)
        image = raw_image.copy()
        image[20:45, 78:110] = 180
        exclusion = inpaint_exclusion_mask(image, raw_image)
        raw_mask = np.zeros((120, 160), np.uint8)
        raw_mask[30:85, 55:105] = 255
        label = {"bbox_xyxy_px_clipped": [50, 20, 110, 90],
                 "center_uv_px": [75, 60]}
        refined = pose_refine_mask(
            raw_mask, label, kernel_size=9, padding_ratio=0, exclusion_mask=exclusion)
        self.assertTrue(refined[60, 75])
        self.assertFalse(refined[25, 90])
        self.assertFalse(np.any(refined[exclusion > 0]))

    def test_bright_neutral_tag_bridge_is_disconnected(self):
        image = np.zeros((120, 160, 3), np.uint8)
        image[30:85, 55:105] = (0, 180, 230)
        image[15:35, 65:95] = 220
        raw_mask = np.zeros((120, 160), np.uint8)
        raw_mask[15:85, 55:105] = 255
        label = {"bbox_xyxy_px_clipped": [50, 10, 110, 90],
                 "center_uv_px": [80, 60]}
        refined = pose_refine_mask(
            raw_mask, label, kernel_size=9, padding_ratio=0,
            disconnect_mask=bright_neutral_disconnect_mask(image))
        self.assertTrue(refined[60, 80])
        self.assertFalse(refined[20, 80])

    def test_adaptive_threshold_recovers_low_confidence_object(self):
        probability = np.zeros((100, 120), np.float32)
        probability[30:70, 40:80] = 0.42

        class Predictor:
            def predict(self, _image):
                return probability

        label = {"bbox_xyxy_px_clipped": [35, 25, 85, 75],
                 "center_uv_px": [60, 50]}
        _raw, mask, metrics = refine_mask(
            np.zeros((100, 120, 3), np.uint8), label, Predictor(), 0.5, 0.2)
        self.assertTrue(mask[50, 60])
        self.assertAlmostEqual(metrics["birefnet_threshold"], 0.4)

    def test_pose_guided_crop_recovers_full_frame_miss(self):
        miss = np.zeros((100, 120), np.float32)
        recovered = miss.copy()
        recovered[30:70, 40:80] = 0.9

        class Predictor:
            def predict(self, _image):
                return miss

            def predict_crop(self, _image, _label):
                return recovered

        label = {"bbox_xyxy_px_clipped": [35, 25, 85, 75],
                 "center_uv_px": [60, 50]}
        _raw, mask, metrics = refine_mask(
            np.zeros((100, 120, 3), np.uint8), label, Predictor(), 0.5, 0.2)
        self.assertTrue(mask[50, 60])
        self.assertEqual(metrics["birefnet_inference"], "pose_guided_crop")

    def test_source_discovery_requires_png_json_pairs(self):
        dataset = self.root / "dataset"
        pair_dir = dataset / "centerpose/outf/widget_train"
        pair_dir.mkdir(parents=True)
        (dataset / "labels").mkdir()
        (dataset / "centerpose/export.json").write_text(json.dumps({
            "category": "widget", "split": "train"}))
        cv2.imwrite(str(pair_dir / "one.png"), np.zeros((20, 20, 3), np.uint8))
        (pair_dir / "one.json").write_text("{}")
        (dataset / "labels/one.json").write_text("{}")
        metadata, sources = paired_sources(dataset)
        self.assertEqual(metadata["category"], "widget")
        self.assertEqual(len(sources), 1)


if __name__ == "__main__":
    unittest.main()

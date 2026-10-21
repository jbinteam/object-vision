import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from scripts.generate_yolo_seg_pose_dataset import (
    export_dataset, mask_polygon, mask_to_polygons, normalize_keypoints,
    polygon_mask_iou, pose_label, validation_split,
)


class YoloSegPoseDatasetTest(unittest.TestCase):
    def test_source_validation_split_is_deterministic(self):
        records = [{"source_image": f"/capture/view_{index}.png"}
                   for index in range(8)]
        first = validation_split(records, 0.25, 19)
        second = validation_split(records, 0.25, 19)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)

    def test_mask_polygon_and_keypoint_visibility(self):
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[4:16, 7:23] = 1
        polygon = mask_polygon(mask, epsilon_ratio=0)
        self.assertGreaterEqual(len(polygon), 4)
        self.assertEqual(polygon_mask_iou(mask, polygon), 1.0)

        points = np.asarray([
            [0, 0], [29, 19], [15, 10], [-1, 4],
            [30, 4], [4, 20], [4, -1], [2.5, 3.5],
        ], dtype=np.float64)
        normalized = normalize_keypoints(points, 30, 20)
        np.testing.assert_array_equal(normalized[:, 2], [2, 2, 2, 0, 0, 0, 0, 2])
        np.testing.assert_array_equal(normalized[3:7, :2], 0)
        line = pose_label(0, [0.5, 0.5, 0.4, 0.6], normalized)
        self.assertEqual(len(line.split()), 5 + 8 * 3)

    def test_objectsegmentation_contour_policy_reports_components_and_holes(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[5:70, 5:70] = 1
        mask[25:40, 25:40] = 0
        mask[80:95, 80:95] = 1
        mask[0:2, 0:2] = 1
        polygons, holes = mask_to_polygons(mask, min_area=100, epsilon_ratio=0)
        self.assertEqual(len(polygons), 2)
        self.assertEqual(holes, 1)
        with self.assertRaisesRegex(ValueError, "exactly one object polygon"):
            mask_polygon(mask, epsilon_ratio=0, min_area=100)

    def test_export_writes_synchronized_segmentation_and_pose_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            augmentation = dataset / "centerpose_augmented_clean"
            pair_root = augmentation / "outf/widget_train"
            for directory in (dataset / "labels", augmentation / "masks", pair_root):
                directory.mkdir(parents=True, exist_ok=True)

            image = np.full((48, 64, 3), 120, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(pair_root / "sample.png"), image))
            self.assertTrue(cv2.imwrite(str(pair_root / "sample__bg_001.png"), image))
            mask = np.zeros((48, 64), dtype=np.uint8)
            mask[10:39, 15:50] = 255
            self.assertTrue(cv2.imwrite(str(augmentation / "masks/sample.png"), mask))
            label = {
                "image_size": [64, 48],
                "object_frame": "box_center",
                "box_dimensions_m": [0.2, 0.3, 0.4],
                "box_corners_uv_px": [
                    [16, 12], [48, 12], [16, 36], [48, 36],
                    [20, 16], [44, 16], [20, 32], [44, 32],
                ],
            }
            (dataset / "labels/sample.json").write_text(json.dumps(label))
            manifest = {
                "category": "widget",
                "split": "train",
                "records": [{
                    "source_image": "/old/location/sample.png",
                    "mask": "masks/sample.png",
                    "generated": [
                        {"name": "sample.png"},
                        {"name": "sample__bg_001.png"},
                    ],
                }],
            }
            (augmentation / "manifest.json").write_text(json.dumps(manifest))

            output = root / "export"
            counts, destination = export_dataset(
                dataset, output, augmentation_root=augmentation, make_previews=True)
            self.assertEqual(counts["train"], 1)
            self.assertEqual(destination, output.resolve())
            stem = "sample"
            self.assertTrue((output / f"segmentation/labels/train/{stem}.txt").is_file())
            self.assertTrue((output / f"pose/labels/train/{stem}.txt").is_file())
            self.assertTrue((output / f"annotations/train/{stem}.json").is_file())
            self.assertTrue((output / f"pose/images/train/{stem}.png").is_symlink())
            self.assertTrue((output / f"pose/masks/train/{stem}.png").is_symlink())
            self.assertFalse((output / "pose/images/train/sample__bg_001.png").exists())
            pose_fields = (output / "pose/labels/train/sample.txt").read_text().split()
            self.assertEqual(len(pose_fields), 29)
            segment_fields = (output / "segmentation/labels/train/sample.txt").read_text().split()
            self.assertGreaterEqual(len(segment_fields), 7)
            self.assertTrue((output / "review/sample.jpg").is_file())
            self.assertIn("kpt_shape: [8, 3]", (output / "pose/data.yaml").read_text())
            self.assertIn("object_keypoints:", (output / "object_keypoints.yaml").read_text())
            report = json.loads((output / "manifest.json").read_text())
            self.assertEqual(report["keypoint_count"], 8)
            self.assertEqual(report["counts"], {"train": 1})
            self.assertIn("training-time", report["background_augmentation"])
            self.assertGreaterEqual(report["polygon_mask_iou"]["minimum"], 0.98)

    def test_source_grouped_validation_uses_only_original_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            augmentation = dataset / "centerpose_augmented"
            pair_root = augmentation / "outf/widget_train"
            for directory in (dataset / "labels", augmentation / "masks", pair_root):
                directory.mkdir(parents=True, exist_ok=True)
            image = np.zeros((20, 20, 3), dtype=np.uint8)
            mask = np.zeros((20, 20), dtype=np.uint8)
            mask[3:18, 4:17] = 255
            label = {
                "image_size": [20, 20],
                "box_dimensions_m": [1, 1, 1],
                "box_corners_uv_px": [[5, 5], [15, 5], [5, 15], [15, 15],
                                              [6, 6], [14, 6], [6, 14], [14, 14]],
            }
            records = []
            for stem in ("train_source", "val_source"):
                (dataset / f"labels/{stem}.json").write_text(json.dumps(label))
                self.assertTrue(cv2.imwrite(str(augmentation / f"masks/{stem}.png"), mask))
                generated = [{"name": f"{stem}.png"}, {"name": f"{stem}__bg.png"}]
                for item in generated:
                    self.assertTrue(cv2.imwrite(str(pair_root / item["name"]), image))
                records.append({"source_image": f"/old/{stem}.png",
                                "mask": f"masks/{stem}.png", "generated": generated})
            (augmentation / "manifest.json").write_text(json.dumps({
                "category": "widget", "split": "train", "records": records,
            }))
            split_path = root / "split.json"
            split_path.write_text(json.dumps({
                "category": "widget",
                "validation_policy": "source-grouped; original image only",
                "validation_source_stems": ["val_source"],
            }))

            counts, _ = export_dataset(
                dataset, root / "export", augmentation, split_path, make_previews=False)
            self.assertEqual(counts["train"], 1)
            self.assertEqual(counts["val"], 1)
            self.assertFalse((root / "export/pose/labels/val/val_source__bg.txt").exists())


if __name__ == "__main__":
    unittest.main()

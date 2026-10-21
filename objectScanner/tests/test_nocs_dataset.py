import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import cv2
import numpy as np

from nocs_dataset import (UINT16_MAX, generate_nocs_target,
                          extract_masked_object_points, read_nocs_png,
                          write_nocs_png)
from scripts.generate_nocs_dataset import generate


def label(width, height, stem):
    transform = np.eye(4)
    transform[:3, 3] = [0, 0, 1]
    return {
        "image_size": [width, height],
        "intrinsic": [[100, 0, (width - 1) / 2],
                      [0, 100, (height - 1) / 2],
                      [0, 0, 1]],
        "T_camera_object": transform.tolist(),
        "box_dimensions_m": [0.1, 0.1, 0.1],
        "depth": f"depths/{stem}.png",
    }


class NOCSTargetTest(unittest.TestCase):
    def test_dense_coordinates_roundtrip_and_png_channel_order(self):
        depth = np.full((5, 5), 1000, dtype=np.uint16)
        depth[1, 1] = 0
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[1:4, 1:4] = 255
        instance, rgba, metrics = generate_nocs_target(
            depth, mask, label(5, 5, "sample"))

        self.assertEqual(instance[1, 1], 1)
        self.assertEqual(rgba[1, 1, 3], 0)
        self.assertEqual(rgba[2, 2, 3], UINT16_MAX)
        np.testing.assert_allclose(rgba[2, 2, :3] / UINT16_MAX,
                                   [0.5, 0.5, 0.5], atol=1 / UINT16_MAX)
        self.assertEqual(metrics["mask_pixels"], 9)
        self.assertEqual(metrics["valid_nocs_pixels"], 8)
        self.assertLess(metrics["max_reprojection_error_px"], 1e-10)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "target.png"
            write_nocs_png(path, rgba)
            np.testing.assert_array_equal(read_nocs_png(path), rgba)

    def test_depth_outside_metric_box_is_not_a_valid_coordinate(self):
        depth = np.full((3, 3), 1000, dtype=np.uint16)
        depth[1, 1] = 1500
        mask = np.full((3, 3), 255, dtype=np.uint8)
        _instance, rgba, metrics = generate_nocs_target(
            depth, mask, label(3, 3, "sample"), box_tolerance_ratio=0)
        self.assertEqual(rgba[1, 1, 3], 0)
        self.assertEqual(metrics["valid_nocs_pixels"], 8)

    def test_nontrivial_pose_maps_camera_point_back_to_object_frame(self):
        depth = np.zeros((11, 11), dtype=np.uint16)
        depth[4, 6] = 1000
        mask = np.zeros_like(depth)
        mask[4, 6] = 255
        sample = label(11, 11, "sample")
        rotation = np.array([[0.0, -1.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0]])
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = [0.01, 0.02, 1.0]
        sample["T_camera_object"] = transform.tolist()
        sample["box_dimensions_m"] = [0.2, 0.2, 0.2]
        (_foreground, ys, xs, camera_points,
         object_points, metrics) = extract_masked_object_points(
            depth, mask, sample, box_tolerance_ratio=0)
        self.assertEqual((ys.tolist(), xs.tolist()), ([4], [6]))
        expected_camera = np.array([[0.01, -0.01, 1.0]])
        expected_object = (expected_camera - transform[:3, 3]) @ rotation
        np.testing.assert_allclose(camera_points, expected_camera, atol=1e-12)
        np.testing.assert_allclose(object_points, expected_object, atol=1e-12)
        self.assertLess(metrics["max_reprojection_error_px"], 1e-10)


class NOCSDatasetGeneratorTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        self.augmentation = self.dataset / "centerpose_augmented"
        self.pairs = self.augmentation / "outf/widget_train"
        for directory in (self.dataset / "labels", self.dataset / "depths",
                          self.augmentation / "masks", self.pairs):
            directory.mkdir(parents=True, exist_ok=True)

        records = []
        for stem in ("view_a", "view_b"):
            sample_label = label(8, 8, stem)
            (self.dataset / "labels" / f"{stem}.json").write_text(
                json.dumps(sample_label))
            depth = np.full((8, 8), 1000, dtype=np.uint16)
            cv2.imwrite(str(self.dataset / "depths" / f"{stem}.png"), depth)
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[2:6, 2:6] = 255
            cv2.imwrite(str(self.augmentation / "masks" / f"{stem}.png"), mask)
            generated = []
            for name, background in ((f"{stem}.png", None),
                                     (f"{stem}__bg_001.png", "background.jpg")):
                cv2.imwrite(str(self.pairs / name), np.full((8, 8, 3), 127, np.uint8))
                generated.append({"name": name, "background": background})
            records.append({
                "source_image": str(self.pairs / f"{stem}.png"),
                "mask": f"masks/{stem}.png",
                "generated": generated,
            })
        (self.augmentation / "manifest.json").write_text(json.dumps({
            "category": "widget", "records": records,
        }))
        self.split = self.dataset / "centerpose_training/split.json"
        self.split.parent.mkdir(parents=True)
        self.split.write_text(json.dumps({
            "category": "widget",
            "validation_policy": "source-grouped; original image only",
            "validation_source_stems": ["view_b"],
        }))

    def args(self, output, check_only=False):
        return SimpleNamespace(
            dataset=self.dataset,
            augmentation_root=self.augmentation,
            split_manifest=self.split,
            output=output,
            box_tolerance_ratio=0.08,
            min_valid_pixels=1,
            min_nocs_coverage=0.1,
            check_only=check_only,
        )

    def test_generator_reuses_targets_and_keeps_validation_unaugmented(self):
        output = self.root / "nocs"
        manifest = generate(self.args(output))
        self.assertEqual(manifest["images"], {"train": 2, "val": 1})
        train = json.loads((output / "widget_train_metadata.json").read_text())
        validation = json.loads((output / "widget_val_metadata.json").read_text())
        self.assertEqual(len(train), 2)
        self.assertEqual(len(validation), 1)
        self.assertEqual({row["source_view"] for row in train}, {"view_a"})
        self.assertEqual(validation[0]["source_view"], "view_b")
        self.assertTrue((output / train[0]["image_name"]).is_symlink())
        rgba = read_nocs_png(output / "annotations/view_a_nocs.png")
        self.assertEqual(int(np.count_nonzero(rgba[..., 3])), 16)
        instance = cv2.imread(
            str(output / "annotations/view_a_instances.png"), cv2.IMREAD_UNCHANGED)
        self.assertEqual(instance.dtype, np.uint16)
        self.assertEqual(int(np.count_nonzero(instance == 1)), 16)

    def test_check_only_reports_missing_depth_without_writing_output(self):
        (self.dataset / "depths/view_b.png").unlink()
        output = self.root / "nocs"
        report = generate(self.args(output, check_only=True))
        self.assertFalse(report["ready"])
        self.assertEqual(report["depth_images_missing"], 1)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

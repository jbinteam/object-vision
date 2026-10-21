import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import cv2
import numpy as np
import open3d as o3d

from scripts.generate_centerpose_reference import generate


class CenterPoseReferenceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        self.augmentation = self.dataset / "centerpose_augmented_clean"
        for directory in (self.dataset / "labels", self.dataset / "images",
                          self.dataset / "depths", self.augmentation / "masks"):
            directory.mkdir(parents=True, exist_ok=True)
        records = []
        for index, stem in enumerate(("front", "side")):
            image = np.zeros((12, 12, 3), dtype=np.uint8)
            image[..., index] = 200
            depth = np.full((12, 12), 1000, dtype=np.uint16)
            mask = np.zeros((12, 12), dtype=np.uint8)
            mask[2:10, 2:10] = 255
            cv2.imwrite(str(self.dataset / f"images/{stem}.png"), image)
            cv2.imwrite(str(self.dataset / f"depths/{stem}.png"), depth)
            cv2.imwrite(str(self.augmentation / f"masks/{stem}.png"), mask)
            camera_object = np.eye(4)
            camera_object[:3, 3] = [0, 0, 1]
            label = {
                "image": f"images/{stem}.png",
                "depth": f"depths/{stem}.png",
                "image_size": [12, 12],
                "intrinsic": [[100, 0, 5.5], [0, 100, 5.5], [0, 0, 1]],
                "T_camera_object": camera_object.tolist(),
                "box_dimensions_m": [0.2, 0.2, 0.2],
                "camera_frame": "color_optical_frame",
                "object_frame": "box_center",
                "depth_unit_m": 0.001,
                "depth_aligned_to": "rectified color image",
            }
            (self.dataset / f"labels/{stem}.json").write_text(json.dumps(label))
            records.append({"source_image": f"/old/{stem}.png",
                            "mask": f"masks/{stem}.png"})
        (self.augmentation / "manifest.json").write_text(json.dumps({
            "category": "widget", "records": records,
        }))

    def args(self, output, check_only=False):
        return SimpleNamespace(
            dataset=self.dataset,
            augmentation_root=self.augmentation,
            output=output,
            box_tolerance_ratio=0.08,
            min_valid_points=3,
            observation_voxel_size_mm=0,
            voxel_size_mm=0,
            outlier_neighbors=0,
            outlier_std_ratio=2.0,
            normal_radius_mm=50,
            normal_max_neighbors=10,
            no_observations=False,
            check_only=check_only,
        )

    def test_fuses_cloud_directly_in_centerpose_box_frame(self):
        output = self.root / "reference"
        metadata = generate(self.args(output))
        self.assertTrue((output / "reference.ply").is_file())
        self.assertTrue((output / "reference_bundle.zip").is_file())
        self.assertIn("object_keypoints:",
                      (output / "object_keypoints.yaml").read_text())
        np.testing.assert_array_equal(metadata["T_scan_obj"], np.eye(4))
        np.testing.assert_array_equal(metadata["T_obj_scan"], np.eye(4))
        self.assertEqual(metadata["frame_id"], "box_center")
        self.assertTrue(metadata["reference_already_in_object_frame"])
        cloud = o3d.io.read_point_cloud(str(output / "reference.ply"))
        points = np.asarray(cloud.points)
        self.assertEqual(len(points), 128)
        self.assertLess(np.abs(points[:, 2]).max(), 1e-9)
        camera = o3d.io.read_point_cloud(
            str(output / "observations/camera_frame/front.ply"))
        object_cloud = o3d.io.read_point_cloud(
            str(output / "observations/object_frame/front.ply"))
        np.testing.assert_allclose(
            np.asarray(camera.points)[:, 2] - np.asarray(object_cloud.points)[:, 2],
            1.0, atol=1e-9)

    def test_check_only_reports_missing_depth_without_output(self):
        (self.dataset / "depths/side.png").unlink()
        output = self.root / "reference"
        report = generate(self.args(output, check_only=True))
        self.assertFalse(report["ready"])
        self.assertEqual(report["ready_views"], 1)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

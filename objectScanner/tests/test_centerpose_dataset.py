import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from centerpose_dataset import (CENTERPOSE_UNIT_BOX, OPENGL_FROM_OPENCV,
                                export_centerpose_dataset,
                                make_centerpose_annotation)


class CenterPoseDatasetTest(unittest.TestCase):
    def label(self):
        intrinsic = np.array([[600.0, 0.0, 320.0],
                              [0.0, 610.0, 240.0],
                              [0.0, 0.0, 1.0]])
        camera_object = np.eye(4)
        camera_object[:3, :3] = cv2.Rodrigues(np.array([0.1, -0.2, 0.3]))[0]
        camera_object[:3, 3] = [0.04, -0.02, 1.2]
        return {
            "intrinsic": intrinsic.tolist(),
            "image_size": [640, 480],
            "T_camera_object": camera_object.tolist(),
            "box_dimensions_m": [0.2, 0.3, 0.4],
            "center_uv_px": [340.0, 230.0],
            "image": "images/sample.png",
        }

    def test_annotation_has_objectron_order_and_centerpose_y_up_scale(self):
        annotation = make_centerpose_annotation(self.label(), "widget")
        obj = annotation["objects"][0]
        self.assertEqual(obj["class"], "widget")
        np.testing.assert_allclose(obj["scale"], [0.2, 0.4, 0.3])
        self.assertEqual(np.asarray(obj["projected_cuboid"]).shape, (9, 2))
        self.assertEqual(np.asarray(obj["keypoints_3d"]).shape, (9, 3))
        np.testing.assert_allclose(obj["keypoints_3d"][0], obj["location"], atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(obj["quaternion_xyzw"]), 1.0)

        # Reproduce CenterPose's NDC-to-pixel code and ensure its projection
        # matrix maps all exported OpenGL keypoints to projected_cuboid.
        points = np.c_[obj["keypoints_3d"], np.ones(9)]
        projection = np.asarray(annotation["camera_data"]["camera_projection_matrix"])
        ndc = (projection @ points.T).T
        ndc = ndc[:, :3] / ndc[:, 3:4]
        viewport = (ndc + 1.0) / 2.0 * np.array([480.0, 640.0, 1.0])
        uv = viewport[:, [1, 0]]
        np.testing.assert_allclose(uv, obj["projected_cuboid"], atol=1e-9)

        # CenterPose +Y maps to objectScanner +Z (up), not its base-plane Y.
        centerpose_up = CENTERPOSE_UNIT_BOX[3] - CENTERPOSE_UNIT_BOX[1]
        self.assertGreater(centerpose_up[1], 0)
        self.assertEqual(np.linalg.det(OPENGL_FROM_OPENCV[:3, :3]), 1.0)

    def test_dataset_export_writes_same_basename_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            (dataset / "images").mkdir()
            (dataset / "labels").mkdir()
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(dataset / "images" / "sample.png"), image))
            (dataset / "labels" / "sample.json").write_text(json.dumps(self.label()))

            count, destination = export_centerpose_dataset(dataset, "widget", "train")
            self.assertEqual(count, 1)
            self.assertTrue((destination / "sample.png").is_file())
            self.assertTrue((destination / "sample.json").is_file())
            exported = json.loads((destination / "sample.json").read_text())
            self.assertEqual(exported["objects"][0]["class"], "widget")


if __name__ == "__main__":
    unittest.main()

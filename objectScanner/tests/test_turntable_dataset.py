import unittest
import json
from pathlib import Path

import cv2
import numpy as np

from turntable_dataset import (calibrate, detect_tags, make_detector, make_label, project,
                               reconfigure_box, register_visible_tags,
                               relabel_for_reconfigured_box, remove_tags,
                               tag_corners, transform_points)


K = np.array([[600., 0, 320.], [0, 600., 240.], [0, 0, 1.]])
SIZE = .04


def transform(translation, rotation=None):
    result = np.eye(4)
    result[:3, 3] = translation
    if rotation is not None:
        result[:3, :3] = rotation
    return result


def observations(poses, size=SIZE):
    return {tag_id: project(K, transform_points(pose, tag_corners(size)))
            for tag_id, pose in poses.items()}


class TurntableDatasetTest(unittest.TestCase):
    def setUp(self):
        # Bottom +X points right; +Y points upward, so box +Z points to camera.
        self.setup = {
            0: transform([-.05, .05, .8]),
            1: transform([.05, .05, .8]),
            2: transform([-.05, -.03, .8]),
            3: transform([0, .01, .70]),
        }
        self.board = {
            10: transform([-.18, -.12, .8]),
            11: transform([.18, -.12, .8]),
            12: transform([-.18, .12, .8]),
            13: transform([.18, .12, .8]),
        }
        self.calibration = calibrate(observations({**self.setup, **self.board}),
                                     K, (0, 1, 2, 3), (10, 11, 12, 13), SIZE)

    def test_center_pose_follows_table_and_camera(self):
        dimensions = self.calibration["box_dimensions_m"]
        np.testing.assert_allclose(dimensions, [.1, .08, .1], atol=1e-7)
        first = make_label(self.calibration, observations(self.board), K, (480, 640, 3), 1)
        self.assertEqual(first["camera_frame"], "camera_color_optical_frame")
        np.testing.assert_allclose(first["center_xyz_m"], [0, .01, .75], atol=1e-6)
        self.assertEqual(first["permanent_tags_used"], [10, 11, 12, 13])

        rotation = cv2.Rodrigues(np.array([.08, -.13, .35]))[0]
        movement = transform([.03, -.02, .04], rotation)
        moved_board = {key: movement @ pose for key, pose in self.board.items()}
        # One tag can be occluded; remaining calibrated tags still determine pose.
        del moved_board[13]
        second = make_label(self.calibration, observations(moved_board), K,
                            (480, 640, 3), 2)
        expected = movement @ np.asarray(first["T_camera_object"])
        np.testing.assert_allclose(second["T_camera_object"], expected, atol=1e-6)
        self.assertEqual(second["permanent_tags_used"], [10, 11, 12])

    def test_setup_tag_ids_are_sorted_by_geometry(self):
        # The top is ID 0 here, and the setup ID argument is deliberately shuffled.
        unordered_setup = {
            0: self.setup[3],
            1: self.setup[2],
            2: self.setup[0],
            3: self.setup[1],
        }
        calibration = calibrate(observations({**unordered_setup, **self.board}), K,
                                (3, 0, 2, 1), (10, 11, 12, 13), SIZE)
        np.testing.assert_allclose(calibration["box_dimensions_m"], [.1, .08, .1], atol=1e-7)
        np.testing.assert_allclose(calibration["T_table_object"],
                                   self.calibration["T_table_object"], atol=1e-7)
        self.assertEqual(calibration["resolved_setup_roles"],
                         {"base_origin": 2, "base_x": 3, "base_y": 1, "top": 0})

    def test_setup_and_permanent_tags_can_have_different_sizes(self):
        setup_size, permanent_size = .014, .028
        detections = {**observations(self.setup, setup_size),
                      **observations(self.board, permanent_size)}
        calibration = calibrate(detections, K, (0, 1, 2, 3),
                                (10, 11, 12, 13), setup_size, permanent_size)
        np.testing.assert_allclose(calibration["box_dimensions_m"], [.1, .08, .1], atol=1e-7)
        np.testing.assert_allclose(calibration["T_table_object"],
                                   self.calibration["T_table_object"], atol=1e-7)
        self.assertEqual(calibration["setup_tag_size_m"], setup_size)
        self.assertEqual(calibration["permanent_tag_size_m"], permanent_size)
        label = make_label(calibration, observations(self.board, permanent_size),
                           K, (480, 640, 3), 1)
        np.testing.assert_allclose(label["center_xyz_m"], [0, .01, .75], atol=1e-6)

    def test_setup_tags_must_have_one_above_table(self):
        flat = dict(self.setup)
        flat[3] = transform([.03, -.01, .8])
        with self.assertRaisesRegex(ValueError, "clearly above"):
            calibrate(observations({**flat, **self.board}), K,
                      (0, 1, 2, 3), (10, 11, 12, 13), SIZE)

    def test_reconfigure_box_updates_existing_label_in_same_table_frame(self):
        old_label = make_label(self.calibration, observations(self.board),
                               K, (480, 640, 3), 1)
        changed_setup = {
            0: transform([-.06, .055, .8]),
            1: transform([.06, .055, .8]),
            2: transform([-.06, -.035, .8]),
            3: transform([0, .01, .68]),
        }
        updated = reconfigure_box(
            self.calibration, observations({**changed_setup, **self.board}), K)
        self.assertEqual(updated["revision"], 2)
        self.assertEqual(updated["T_table_tags"], self.calibration["T_table_tags"])
        np.testing.assert_allclose(updated["box_dimensions_m"], [.12, .09, .12], atol=1e-7)

        relabeled = relabel_for_reconfigured_box(old_label, self.calibration, updated)
        old_camera_table = (np.asarray(old_label["T_camera_object"])
                            @ np.linalg.inv(np.asarray(self.calibration["T_table_object"])))
        expected = old_camera_table @ np.asarray(updated["T_table_object"])
        np.testing.assert_allclose(relabeled["T_camera_object"], expected, atol=1e-7)
        self.assertEqual(relabeled["box_dimensions_m"], updated["box_dimensions_m"])
        self.assertEqual(relabeled["calibration_revision"], 2)

    def test_refuse_pose_with_only_one_tag(self):
        with self.assertRaisesRegex(ValueError, "At least two"):
            make_label(self.calibration, observations({10: self.board[10]}),
                       K, (480, 640, 3), 1)

    def test_initial_calibration_needs_only_two_permanent_tags(self):
        tags = {**self.setup, **self.board}
        del tags[13]
        del tags[12]
        calibration = calibrate(observations(tags), K,
                                (0, 1, 2, 3), (10, 11, 12, 13), SIZE)
        self.assertEqual(set(calibration["T_table_tags"]), {"10", "11"})

    def test_progressively_map_tags_then_capture_without_initial_pair(self):
        initial_tags = {**self.setup, 10: self.board[10], 11: self.board[11]}
        calibration = calibrate(observations(initial_tags), K,
                                (0, 1, 2, 3), (10, 11, 12, 13), SIZE)
        first = make_label(calibration, observations({10: self.board[10], 11: self.board[11]}),
                           K, (480, 640, 3), 1)
        move1 = transform([.02, 0, .01], cv2.Rodrigues(np.array([.02, 0, .12]))[0])
        seen1 = {tag_id: move1 @ self.board[tag_id] for tag_id in (10, 11, 12)}
        calibration, added = register_visible_tags(calibration, observations(seen1), K)
        self.assertEqual(added, [12])
        calibration = json.loads(json.dumps(calibration))  # Resume from calibration.json.
        move2 = transform([-.01, .01, .02], cv2.Rodrigues(np.array([0, .02, -.1]))[0])
        seen2 = {tag_id: move2 @ self.board[tag_id] for tag_id in (11, 12, 13)}
        calibration, added = register_visible_tags(calibration, observations(seen2), K)
        self.assertEqual(added, [13])
        move3 = transform([.03, -.02, .04], cv2.Rodrigues(np.array([.08, -.13, .35]))[0])
        seen3 = {tag_id: move3 @ self.board[tag_id] for tag_id in (12, 13)}
        label = make_label(calibration, observations(seen3), K, (480, 640, 3), 2)
        expected = move3 @ np.asarray(first["T_camera_object"])
        np.testing.assert_allclose(label["T_camera_object"], expected, atol=1e-6)
        self.assertEqual(label["permanent_tags_used"], [12, 13])

    def test_refuse_registration_without_two_known_tags(self):
        initial_tags = {**self.setup, 10: self.board[10], 11: self.board[11]}
        calibration = calibrate(observations(initial_tags), K,
                                (0, 1, 2, 3), (10, 11, 12, 13), SIZE)
        with self.assertRaisesRegex(ValueError, "At least two"):
            register_visible_tags(calibration,
                                  observations({11: self.board[11], 12: self.board[12]}), K)

    def test_apriltag_detector_reads_printed_family(self):
        family = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        marker = cv2.aruco.generateImageMarker(family, 10, 160)
        canvas = np.full((300, 300, 3), 255, dtype=np.uint8)
        canvas[70:230, 70:230] = cv2.cvtColor(marker, cv2.COLOR_GRAY2RGB)
        detector = cv2.aruco.ArucoDetector(family, cv2.aruco.DetectorParameters())
        detections = detect_tags(canvas, detector)
        self.assertIn(10, detections)
        cleaned = remove_tags(canvas, detections, (10,))
        self.assertGreater(cleaned[80, 80].mean(), canvas[80, 80].mean())

    def test_standard41h12_detector_reads_generator_output_and_rotation(self):
        path = Path(__file__).parent / "fixtures" / "tagStandard41h12_00010.png"
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(gray)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        detector = make_detector("tagStandard41h12")
        upright = detect_tags(rgb, detector)[10]
        self.assertLess(upright[0, 0], upright[1, 0])
        self.assertLess(upright[0, 1], upright[3, 1])
        rotated = detect_tags(cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE), detector)[10]
        self.assertLess(rotated[0, 1], rotated[1, 1])
        self.assertGreater(rotated[0, 0], rotated[3, 0])
        calibration = calibrate(observations({**self.setup, **self.board}), K,
                                (0, 1, 2, 3), (10, 11, 12, 13), SIZE,
                                tag_family="tagStandard41h12")
        self.assertEqual(calibration["tag_family"], "tagStandard41h12")


if __name__ == "__main__":
    unittest.main()

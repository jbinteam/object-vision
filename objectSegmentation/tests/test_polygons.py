import unittest

import cv2
import numpy as np

from remove_background import mask_to_polygons


class PolygonTests(unittest.TestCase):
    def test_rectangle_coordinates_and_roundtrip(self):
        mask = np.zeros((80, 120), np.uint8)
        mask[10:60, 20:90] = 255
        polygons, holes = mask_to_polygons(mask)
        self.assertEqual((len(polygons), holes), (1, 0))
        normalized = polygons[0] / [120, 80]
        self.assertTrue(np.all((normalized >= 0) & (normalized <= 1)))
        reconstructed = np.zeros_like(mask)
        cv2.fillPoly(reconstructed, polygons, 255)
        np.testing.assert_array_equal(mask, reconstructed)

    def test_hole_is_flagged(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[10:90, 10:90] = 255
        mask[30:70, 30:70] = 0
        polygons, holes = mask_to_polygons(mask)
        self.assertEqual((len(polygons), holes), (1, 1))

    def test_components_and_noise(self):
        mask = np.zeros((100, 100), np.uint8)
        mask[10:40, 10:40] = 255
        mask[60:90, 60:90] = 255
        mask[0:2, 0:2] = 255
        polygons, _ = mask_to_polygons(mask, min_area=100)
        self.assertEqual(len(polygons), 2)

    def test_empty(self):
        self.assertEqual(mask_to_polygons(np.zeros((100, 100), np.uint8)), ([], 0))


if __name__ == "__main__":
    unittest.main()

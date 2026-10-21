import unittest

import numpy as np

from prepare_emergency_stop import clean_mask, yellow_seed, exclude_red_wire


class CleanupTests(unittest.TestCase):
    def test_side_view_button_above_yellow_enclosure(self):
        image = np.full((300, 300, 3), 100, np.uint8)
        image[120:220, 70:230] = (0, 200, 200)
        image[80:120, 120:180] = (0, 0, 200)
        seed, _ = yellow_seed(image)
        self.assertEqual(seed[150, 100], 255)

    def test_wire_exclusion_preserves_red_button(self):
        image = np.full((300, 300, 3), 100, np.uint8)
        image[50:250, 50:250] = (0, 200, 200)
        image[90:160, 90:160] = (0, 0, 200)
        image[248:290, 148:153] = (0, 0, 200)
        mask = np.zeros((300, 300), np.uint8)
        mask[50:250, 50:250] = 255
        mask[248:290, 148:153] = 255
        result = exclude_red_wire(mask, image, (50, 50, 200, 200))
        self.assertEqual(result[120, 120], 255)
        self.assertEqual(result[270, 150], 0)
        self.assertEqual(result[200, 200], 255)

    def test_yellow_selection_requires_red_button(self):
        image = np.full((500, 500, 3), 100, np.uint8)
        image[20:200, 20:400] = (0, 200, 200)
        image[300:450, 100:250] = (0, 200, 200)
        image[340:400, 140:200] = (0, 0, 200)
        seed, _ = yellow_seed(image)
        self.assertEqual(seed[310, 110], 255)
        self.assertEqual(seed[100, 100], 0)

    def test_removes_attached_cable_and_unrelated_region(self):
        mask = np.zeros((400, 600), np.uint8)
        mask[100:300, 50:250] = 255
        mask[195:200, 250:500] = 255
        mask[20:380, 500:590] = 255
        seed = np.zeros_like(mask)
        seed[130:270, 80:220] = 255
        result, coverage = clean_mask(mask, seed)
        self.assertEqual(coverage, 1)
        self.assertEqual(np.count_nonzero(result[:, 270:]), 0)
        self.assertGreater(np.count_nonzero(result[100:300, 50:250]), 39500)

    def test_fills_small_internal_hole(self):
        mask = np.zeros((300, 300), np.uint8)
        mask[30:270, 30:270] = 255
        mask[140:150, 140:150] = 0
        seed = np.zeros_like(mask)
        seed[60:100, 60:100] = 255
        result, _ = clean_mask(mask, seed)
        self.assertEqual(result[145, 145], 255)

    def test_missing_target_has_zero_coverage(self):
        mask = np.zeros((100, 100), np.uint8)
        seed = np.zeros_like(mask)
        seed[20:80, 20:80] = 255
        result, coverage = clean_mask(mask, seed)
        self.assertEqual(coverage, 0)
        self.assertFalse(result.any())


if __name__ == "__main__":
    unittest.main()

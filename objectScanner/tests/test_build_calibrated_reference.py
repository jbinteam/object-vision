import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import open3d as o3d

from scripts.build_calibrated_reference import build


class CalibratedReferenceTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.dataset = Path(temporary.name) / 'dataset'
        self.masks = self.dataset / 'centerpose_masks'
        for path in ('labels', 'images', 'depths', 'centerpose_masks/masks'):
            (self.dataset / path).mkdir(parents=True)
        (self.dataset / 'calibration.json').write_text(json.dumps({
            'object_frame': 'box_center', 'box_dimensions_m': [.2, .2, .2],
            'revision': 1}))
        records = []
        for stem in ('first', 'second'):
            color = np.zeros((64, 64, 3), np.uint8)
            color[..., 1] = 180
            depth = np.full((64, 64), 1000, np.uint16)
            mask = np.zeros((64, 64), np.uint8)
            mask[12:52, 12:52] = 255
            cv2.imwrite(str(self.dataset / f'images/{stem}.png'), color)
            cv2.imwrite(str(self.dataset / f'depths/{stem}.png'), depth)
            cv2.imwrite(str(self.masks / f'masks/{stem}.png'), mask)
            transform = np.eye(4)
            transform[2, 3] = 1
            label = dict(image=f'images/{stem}.png', depth=f'depths/{stem}.png',
                         image_size=[64, 64],
                         intrinsic=[[500, 0, 31.5], [0, 500, 31.5], [0, 0, 1]],
                         T_camera_object=transform.tolist(),
                         box_dimensions_m=[.2, .2, .2],
                         object_frame='box_center', camera_frame='camera_color_optical_frame',
                         calibration_revision=1, depth_unit_m=.001,
                         depth_range_m=[.25, 1.5], depth_aligned_to='rectified color image')
            (self.dataset / f'labels/{stem}.json').write_text(json.dumps(label))
            records.append(dict(source_image=f'/old/{stem}.png', mask=f'masks/{stem}.png'))
        (self.masks / 'manifest.json').write_text(json.dumps({
            'category': 'widget', 'records': records}))

    def test_tsdf_uses_box_center_as_world_frame(self):
        output = self.dataset / 'references/widget'
        report = build(self.dataset, self.masks, output, min_views=2, min_points=30)
        points = np.asarray(o3d.io.read_point_cloud(str(output / 'reference.ply')).points)
        self.assertGreater(len(points), 30)
        self.assertLess(abs(float(np.median(points[:, 2]))), .01)
        self.assertLess(np.max(np.abs(points)), .11)
        self.assertEqual(report['method'], 'calibrated_masked_rgbd_tsdf')
        np.testing.assert_allclose(report['T_scan_obj'], np.eye(4))

    def test_rejects_capture_from_another_calibration_revision(self):
        path = self.dataset / 'labels/second.json'
        label = json.loads(path.read_text())
        label['calibration_revision'] = 2
        path.write_text(json.dumps(label))
        with self.assertRaisesRegex(ValueError, 'Capture frame differs'):
            build(self.dataset, self.masks, self.dataset / 'references/widget',
                  min_views=2, min_points=30)


if __name__ == '__main__':
    unittest.main()

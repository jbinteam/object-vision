import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import open3d as o3d

from reference_geometry import (crop_mask, frame_from_points, frame_transforms,
                                rotation_from_degrees, rotation_to_degrees,
                                transform_points)
from webui import create_app


class ReferenceGeometryTests(unittest.TestCase):
    def test_rotated_crop_around_offset_center_includes_boundaries(self):
        center = np.array([3., -2., 4.])
        half = np.array([2., .3, .5])
        local = np.array([[2., .3, .5], [-2., -.3, -.5], [0., 0., 0.], [2.01, 0., 0.], [0., .31, 0.]])
        angles = [23., -41., 67.]
        points = local @ rotation_from_degrees(angles).T + center
        np.testing.assert_array_equal(crop_mask(points, center-half, center+half, angles), [True, True, True, False, False])
        with self.assertRaises(ValueError):
            crop_mask(points, center-half, center+half, [0, float('nan'), 0])

    def test_inverse_transforms_and_axis_convention(self):
        r = rotation_from_degrees([0, 0, 90])
        scan_obj, obj_scan = frame_transforms([1, 2, 3], r)
        np.testing.assert_allclose(scan_obj @ obj_scan, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(transform_points([[1, 0, 0]], scan_obj), [[1, 3, 3]], atol=1e-12)
        np.testing.assert_allclose(transform_points([[1, 3, 3]], obj_scan), [[1, 0, 0]], atol=1e-12)

    def test_rotation_degree_roundtrip(self):
        for angles in ([12, -23, 47], [0, 90, 31], [-40, -90, 0]):
            rotation = rotation_from_degrees(angles)
            rebuilt = rotation_from_degrees(rotation_to_degrees(rotation))
            np.testing.assert_allclose(rebuilt, rotation, atol=1e-10)

    def test_picked_frame_is_orthonormal_and_right_handed(self):
        r = frame_from_points([1, 2, 3], [1, 3, 3], [0, 2.4, 3])
        np.testing.assert_allclose(r, rotation_from_degrees([0, 0, 90]), atol=1e-12)
        with self.assertRaises(ValueError):
            frame_from_points([0, 0, 0], [1, 0, 0], [2, 0, 0])
        with self.assertRaises(ValueError):
            frame_transforms([0, 0, 0], np.diag([1, 1, -1]))


class ReferenceAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scans, self.references = self.root / "scans", self.root / "references"
        self.scans.mkdir()
        self.points = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.], [2.,2.,2.]])
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.points))
        cloud.colors = o3d.utility.Vector3dVector(np.tile([1., .5, .25], (5, 1)))
        cloud.normals = o3d.utility.Vector3dVector(np.tile([1., 0., 0.], (5, 1)))
        self.path = self.scans / "test.ply"
        o3d.io.write_point_cloud(str(self.path), cloud)
        self.client = create_app(self.scans, self.references).test_client()

    def load(self, units="m"):
        response = self.client.post('/api/load', json={"name":"test.ply", "units": units})
        self.assertEqual(response.status_code, 200, response.json)
        return response.json

    def recipe(self, key):
        return dict(id=key, name="object", crop_min=[-.01,-.01,-.01], crop_max=[1.01,1.01,1.01],
                    origin=[1,0,0], rotation_deg=[0,0,90], voxel_mm=0, estimate_normals=False)

    def test_full_cloud_cropped_transformed_and_bundle_saved(self):
        meta = self.load()
        recipe = self.recipe(meta['id'])
        preview = self.client.get(f"/api/clouds/{meta['id']}/preview")
        self.assertEqual(len(preview.data), 5*6*4)
        self.assertEqual(self.client.post('/api/count', json=recipe).json['points'], 4)
        response = self.client.post('/api/export', json=recipe)
        self.assertEqual(response.status_code, 200, response.json)
        result = o3d.io.read_point_cloud(str(self.references / 'object/reference.ply'))
        np.testing.assert_allclose(np.asarray(result.points), [[0,1,0],[0,0,0],[1,1,0],[0,1,1]], atol=1e-10)
        np.testing.assert_allclose(np.asarray(result.normals), np.tile([0,-1,0], (4,1)), atol=1e-10)
        self.assertTrue(result.has_colors())
        metadata = json.loads((self.references / 'object/frame.json').read_text())
        np.testing.assert_allclose(np.asarray(metadata['T_scan_obj']) @ metadata['T_obj_scan'], np.eye(4), atol=1e-12)
        self.assertEqual(metadata['units'], 'm')
        download = self.client.get(response.json['downloads']['bundle'])
        with zipfile.ZipFile(io.BytesIO(download.data)) as archive:
            self.assertEqual(set(archive.namelist()), {'object/reference.ply','object/frame.json','object/README.txt'})
        download.close()
        self.assertEqual(self.client.post('/api/export', json=recipe).status_code, 409)
        self.assertEqual(len(o3d.io.read_point_cloud(str(self.path)).points), 5)

    def test_upload_and_millimetre_units(self):
        response = self.client.post('/api/upload', data={'file': (io.BytesIO(self.path.read_bytes()), 'uploaded.ply'), 'units': 'mm'})
        self.assertEqual(response.status_code, 200, response.json)
        np.testing.assert_allclose(response.json['extent'], [.002,.002,.002])

    def test_rotated_crop_count_and_export_agree_and_save_recipe(self):
        recipe = self.recipe(self.load()['id'])
        recipe.update(crop_min=[-1.1,-.1,-1.1], crop_max=[1.1,.1,1.1], crop_rotation_deg=[0,0,90])
        self.assertEqual(self.client.post('/api/count', json=recipe).json['points'], 3)
        response = self.client.post('/api/export', json=recipe)
        self.assertEqual(response.status_code, 200, response.json)
        result = o3d.io.read_point_cloud(str(self.references / 'object/reference.ply'))
        # Rotation retains scan Y rather than scan X; obj_frame is independent.
        np.testing.assert_allclose(np.asarray(result.points), [[0,1,0],[1,1,0],[0,1,1]], atol=1e-10)
        metadata = json.loads((self.references / 'object/frame.json').read_text())
        self.assertEqual(metadata['recipe']['crop_rotation_deg'], [0,0,90])
        self.assertEqual(metadata['schema_version'], 2)
        for invalid in ([0,0], [0,'bad',0]):
            recipe['crop_rotation_deg'] = invalid
            self.assertEqual(self.client.post('/api/count', json=recipe).status_code, 400)
            self.assertEqual(self.client.post('/api/export', json=recipe).status_code, 400)

    def test_normals_export(self):
        recipe = self.recipe(self.load()['id'])
        recipe['estimate_normals'] = True
        response = self.client.post('/api/export', json=recipe)
        self.assertEqual(response.status_code, 200, response.json)
        result = o3d.io.read_point_cloud(str(self.references / 'object/reference.ply'))
        self.assertTrue(np.isfinite(np.asarray(result.normals)).all())
        self.assertTrue(result.has_normals())

    def test_capture_workflow_records_box_center_frame(self):
        client = create_app(self.scans, self.references / "centerpose", "box_center").test_client()
        meta = client.post('/api/load', json={"name":"test.ply", "units":"m"}).json
        recipe = self.recipe(meta['id'])
        recipe['name'] = 'widget'
        response = client.post('/api/export', json=recipe)
        self.assertEqual(response.status_code, 200, response.json)
        metadata = json.loads(
            (self.references / "centerpose/widget/frame.json").read_text())
        self.assertEqual(metadata["frame_id"], "box_center")

    def test_fixed_capture_frame_is_loaded_from_scan_sidecar_and_cannot_be_overridden(self):
        transform = np.eye(4)
        transform[:3, :3] = rotation_from_degrees([10, -20, 30])
        transform[:3, 3] = [.2, -.1, .7]
        (self.scans / "test_scan.json").write_text(json.dumps({
            "reference_anchor": {
                "frame_id": "box_center",
                "T_scan_object": transform.tolist(),
                "box_dimensions_m": [.4, .2, .6],
                "tag_reprojection_rms_px": .4,
            }
        }))
        client = create_app(
            self.scans, self.references / "fixed", "box_center", True).test_client()
        response = client.post('/api/load', json={"name":"test.ply", "units":"m"})
        self.assertEqual(response.status_code, 200, response.json)
        self.assertEqual(response.json["suggested_frame"]["frame_id"], "box_center")
        self.assertEqual(response.json["suggested_crop"]["box_dimensions_m"], [.4, .2, .6])
        self.assertEqual(response.json["suggested_crop"]["margin_m"], .01)
        recipe = self.recipe(response.json['id'])
        recipe.update(name="fixed_object", origin=[9, 9, 9], rotation_deg=[0, 0, 0])
        self.assertEqual(client.post('/api/export', json=recipe).status_code, 200)
        metadata = json.loads(
            (self.references / "fixed/fixed_object/frame.json").read_text())
        np.testing.assert_allclose(metadata["T_scan_obj"], transform, atol=1e-10)
        self.assertEqual(metadata["frame_source"],
                         "capture calibration embedded in scan metadata")

    def test_bad_crop_empty_crop_and_path_traversal_rejected(self):
        recipe = self.recipe(self.load()['id'])
        recipe['crop_min'] = [3,3,3]
        self.assertEqual(self.client.post('/api/export', json=recipe).status_code, 400)
        recipe['crop_max'] = [4,4,4]
        self.assertEqual(self.client.post('/api/export', json=recipe).status_code, 400)
        self.assertEqual(self.client.post('/api/load', json={'name':'../outside.ply'}).status_code, 400)
        self.assertEqual(self.client.post('/api/load', json={'name':'test.ply'}, headers={'Origin':'https://example.com'}).status_code, 403)
        recipe=self.recipe(recipe['id']);recipe['name']='../escape'
        self.assertEqual(self.client.post('/api/export', json=recipe).status_code, 400)

    def test_production_crop_keeps_frame_and_backs_up_original(self):
        production = self.references / 'widget'
        production.mkdir(parents=True)
        x = np.linspace(-.04, .04, 1200)
        pts = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        cloud.colors = o3d.utility.Vector3dVector(np.tile([.2, .8, .3], (len(pts), 1)))
        cloud.normals = o3d.utility.Vector3dVector(np.tile([0., 1., 0.], (len(pts), 1)))
        o3d.io.write_point_cloud(str(production / 'reference.ply'), cloud)
        frame = dict(method='calibrated_masked_rgbd_tsdf', frame_id='box_center',
                     points=len(pts), T_scan_obj=np.eye(4).tolist(),
                     T_obj_scan=np.eye(4).tolist())
        (production / 'frame.json').write_text(json.dumps(frame))
        (production / 'object_keypoints.yaml').write_text('object_frame: box_center\n')
        (production / 'README.txt').write_text('calibrated reference\n')
        original = (production / 'reference.ply').read_bytes()
        client = create_app(production_reference_dir=production).test_client()
        self.assertEqual([p['name'] for p in client.get('/api/scans').json], ['reference.ply'])
        self.assertEqual(client.post('/api/upload').status_code, 400)
        metadata = client.post('/api/load', json={'name':'reference.ply','units':'m'}).json
        self.assertEqual(metadata['suggested_frame']['origin'], [0, 0, 0])
        recipe = dict(id=metadata['id'], name='widget', crop_min=[-.041,-.001,-.001],
                      crop_max=[.035,.001,.001], crop_rotation_deg=[0,0,0],
                      origin=[9,9,9], rotation_deg=[90,0,0], voxel_mm=0)
        self.assertEqual(client.post('/api/export', json={**recipe, 'voxel_mm':1}).status_code, 400)
        response = client.post('/api/export', json=recipe)
        self.assertEqual(response.status_code, 200, response.json)
        self.assertGreaterEqual(response.json['points'], 1000)
        self.assertLess(response.json['points'], len(pts))
        backup = Path(response.json['backup'])
        self.assertEqual((backup / 'reference.ply').read_bytes(), original)
        result = o3d.io.read_point_cloud(str(production / 'reference.ply'))
        self.assertEqual(len(result.points), response.json['points'])
        self.assertTrue(result.has_normals())
        np.testing.assert_allclose(np.asarray(result.points)[:,1:], 0)
        updated = json.loads((production / 'frame.json').read_text())
        self.assertEqual(updated['points'], response.json['points'])
        np.testing.assert_allclose(updated['T_scan_obj'], np.eye(4))
        self.assertEqual(updated['manual_crops'][-1]['source_points'], len(pts))
        self.assertEqual(client.post('/api/export', json=recipe).status_code, 400)

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from prepare_emergency_stop import prepare_image
from update_dataset import update, validate_annotation


class FakePredictor:
    def predict(self, image, box):
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[50:250, 50:250] = 255
        mask[150:155, 250:290] = 255
        return mask


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / 'input'
        self.base = self.root / 'base'
        rows = []
        for session, color in [('old_train', 75), ('old_val', 100)]:
            path = self.capture(session, color)
            rows.append(prepare_image(path, path.relative_to(self.input), self.base, FakePredictor()))
        (self.base / 'manifest.json').write_text(json.dumps({'images': rows}))
        for split in ('train', 'val'):
            (self.base / f'{split}.txt').write_text(f'./images/old_{split}/frame.png\n')
        self.args = SimpleNamespace(input=self.input, base=self.base, dataset=self.root / 'dataset',
                                    val_session=[], dry_run=False)
        self.silence = patch('builtins.print')
        self.silence.start()
        self.addCleanup(self.silence.stop)

    def capture(self, session, color, name='frame.png', target=True):
        image = np.full((300, 300, 3), color, np.uint8)
        if target:
            image[50:250, 50:250] = (0, 200, 200)
            image[90:160, 90:160] = (0, 0, 200)
            image[150:155, 250:290] = (0, 0, 200)
        path = self.input / session / name
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)
        return path

    def manifest(self, version):
        return json.loads((version / 'manifest.json').read_text())

    def test_add_preserves_splits_deduplicates_and_rerun_is_noop(self):
        self.capture('new_view', 120)
        self.capture('copy_of_validation', 100)
        version = update(self.args, FakePredictor)
        rows = {r['key']: r for r in self.manifest(version)['images']}
        self.assertEqual(rows['new_view/frame.png']['split'], 'train')
        self.assertEqual(rows['old_val/frame.png']['split'], 'val')
        self.assertEqual(rows['copy_of_validation/frame.png']['status'], 'duplicate')
        self.assertEqual(rows['copy_of_validation/frame.png']['split'], 'excluded')
        self.assertEqual((version / 'val.txt').read_text(), (self.base / 'val.txt').read_text())
        mask = cv2.imread(str(version / 'masks/new_view/frame.png'), 0)
        self.assertEqual(mask[152, 275], 0)  # Attached red cable excluded.
        self.assertEqual(mask[120, 120], 255)  # Button retained.
        for row in rows.values():
            if row['status'] == 'labeled':
                self.assertGreater(validate_annotation(version, row), .98)
        with patch('update_dataset.prepare_image', side_effect=AssertionError('Must not relabel')):
            self.assertEqual(update(self.args, FakePredictor), version)
        self.assertEqual(len(list((self.args.dataset / 'versions').iterdir())), 1)
        self.assertEqual((self.args.dataset / 'latest').resolve(), version)

    def test_explicit_validation_session_and_stable_assignment(self):
        self.capture('new_val', 120)
        self.args.val_session = ['new_val']
        version = update(self.args, FakePredictor)
        self.assertEqual(self.manifest(version)['session_assignments']['new_val'], 'val')
        self.args.val_session = []
        self.capture('new_val', 130, name='second.png')
        second = update(self.args, FakePredictor)
        self.assertIn('./images/new_val/second.png', (second / 'val.txt').read_text())
        self.assertNotIn('second.png', (version / 'val.txt').read_text())
        self.args.val_session = ['old_train']
        with self.assertRaisesRegex(ValueError, 'already belongs'):
            update(self.args, FakePredictor)

    def test_failed_detection_is_quarantined_not_negative_label(self):
        self.capture('empty', 125, target=False)
        version = update(self.args, FakePredictor)
        row = self.manifest(version)['images'][-1]
        self.assertEqual((row['status'], row['split']), ('error', 'excluded'))
        self.assertFalse((version / 'labels/empty/frame.txt').exists())
        self.assertNotIn('empty', (version / 'train.txt').read_text())

    def test_interrupted_publish_reuses_cached_inference(self):
        self.capture('new_view', 120)
        with patch('update_dataset.review_sheets', side_effect=RuntimeError('interrupted')):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                update(self.args, FakePredictor)
        self.assertFalse((self.args.dataset / 'latest').exists())
        with patch('update_dataset.prepare_image', side_effect=AssertionError('Must reuse cache')):
            version = update(self.args, FakePredictor)
        self.assertTrue((version / 'data.yaml').exists())
        self.assertFalse(list((self.args.dataset / 'versions').glob('.building-*')))

    def test_modified_source_rejected_without_changing_latest(self):
        self.capture('new_view', 120)
        version = update(self.args, FakePredictor)
        self.capture('new_view', 140)
        with self.assertRaisesRegex(ValueError, 'source changed'):
            update(self.args, FakePredictor)
        self.assertEqual((self.args.dataset / 'latest').resolve(), version)


if __name__ == '__main__':
    unittest.main()

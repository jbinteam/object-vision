import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from train_backgrounds import FreezeAudit, parse_args


class FreezeTests(unittest.TestCase):
    def make_trainer(self, directory):
        model = nn.Module()
        model.model = nn.Sequential(nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3)), nn.Linear(3, 1))
        model.model[0].requires_grad_(False)
        return SimpleNamespace(model=model, save_dir=Path(directory),
                               args=SimpleNamespace(model='best.pt', freeze=1, optimizer='AdamW',
                                                    lr0=.0001, warmup_bias_lr=0.0))

    def test_trainable_head_can_update_without_changing_frozen_state(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            trainer.model.model[0].eval()
            audit = FreezeAudit(1)
            audit.start(trainer)
            optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=.001)
            head_before = trainer.model.model[1].weight.detach().clone()
            trainer.model.model(torch.ones((2, 3))).sum().backward()
            optimizer.step()
            self.assertFalse(torch.equal(head_before, trainer.model.model[1].weight))
            audit.finish(trainer)
            report = json.loads((Path(directory) / 'freeze_audit.json').read_text())
            self.assertTrue(report['frozen_state_unchanged'])

    def test_batchnorm_drift_is_detected_even_without_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            audit = FreezeAudit(1)
            audit.start(trainer)
            trainer.model.model(torch.ones((2, 3)))  # BN was incorrectly left in training mode.
            with self.assertRaisesRegex(RuntimeError, 'BatchNorm state changed'):
                audit.finish(trainer)

    def test_unfrozen_requested_parameters_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = self.make_trainer(directory)
            trainer.model.model[0].requires_grad_(True)
            with self.assertRaisesRegex(RuntimeError, 'still have trainable'):
                FreezeAudit(1).start(trainer)

    def test_auto_optimizer_cannot_silently_ignore_requested_lr(self):
        with patch('sys.stderr'):
            with self.assertRaises(SystemExit):
                parse_args(['--lr0', '.0001'])
        args = parse_args(['--optimizer', 'AdamW', '--lr0', '.0001', '--freeze', '11'])
        self.assertEqual(args.freeze, 11)
        self.assertEqual(args.lr0, .0001)


if __name__ == '__main__':
    unittest.main()

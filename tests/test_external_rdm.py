"""Unit and synthetic CPU training checks for the author's RDM loss adapter."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project/src"))
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable")
class RDMAdapterTests(unittest.TestCase):
    def test_multiscale_mmd_matches_gaussian_kernel_formula(self):
        import torch

        from elderly_health.rdm_adapter import RDM, RDMSettings

        model = RDM(3, RDMSettings(epochs=3, selection_start_epoch=1))
        x = torch.tensor([[0.1], [0.6], [0.8]])
        y = torch.tensor([[1.5], [1.7], [2.1]])
        expected = (
            model.gaussian_kernel(x, x).mean()
            + model.gaussian_kernel(y, y).mean()
            - 2 * model.gaussian_kernel(x, y).mean()
        )
        torch.testing.assert_close(model.mmd(x, y), expected)
        self.assertGreater(model.mmd(x, y).item(), 0)
        self.assertLess(abs(model.mmd(x, x).item()), 1e-5)

    def test_annealing_and_optimizer_reset(self):
        import torch

        from elderly_health.rdm_adapter import RDM, RDMSettings

        settings = RDMSettings(
            epochs=3,
            selection_start_epoch=1,
            rdm_penalty_anneal_iters=1,
            rdm_lambda=5.0,
        )
        model = RDM(3, settings)
        x1 = torch.randn(8, 3)
        x2 = torch.randn(8, 3) + 1
        y1 = torch.tensor([[0.], [1.]] * 4)
        y2 = 1 - y1
        initial_optimizer = model.optimizer
        out1 = model.update([(x1, y1), (x2, y2)])
        self.assertEqual(out1["matching_weight"], 0.0)
        out2 = model.update([(x1, y1), (x2, y2)])
        self.assertEqual(out2["matching_weight"], 5.0)
        self.assertIsNot(initial_optimizer, model.optimizer)
        self.assertAlmostEqual(model.optimizer.param_groups[0]["lr"], settings.rdm_lr)
        self.assertEqual(model.update_count.item(), 2)

    def test_cpu_source_only_training_and_selection(self):
        from elderly_health.data import SplitData
        from elderly_health.rdm_adapter import RDMSettings, fit_rdm

        rng = np.random.default_rng(16)
        x = rng.normal(size=(140, 19)).astype(np.float32)
        y = np.tile([0, 1], 70).astype(np.float32)
        env = np.repeat(["east", "central"], 70)
        ids = np.arange(140)
        training = SplitData(x[:100], y[:100], env[:100], ids[:100])
        validation = SplitData(x[100:], y[100:], env[100:], ids[100:])
        setting = RDMSettings(
            epochs=3, selection_start_epoch=2, patience=2,
            threads=1, rdm_penalty_anneal_iters=2,
            per_environment_batch_size=8,
        )
        model, info = fit_rdm(training, validation, setting)
        self.assertGreaterEqual(info["best_epoch"], 2)
        self.assertEqual(info["training_domains"], ["central", "east"])
        self.assertFalse(model.training)
        self.assertEqual(model(torch.tensor(x[:2])).shape, (2, 1))

    def test_bad_config_rejected(self):
        from elderly_health.rdm_adapter import RDMSettings

        with self.assertRaises(ValueError):
            RDMSettings(epochs=5, selection_start_epoch=91)
        with self.assertRaises(ValueError):
            RDMSettings(positive_weight=0)


if __name__ == "__main__":
    unittest.main()

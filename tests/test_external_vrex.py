"""Focused tests for the DomainBed VREx tabular adaptation."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project/src"))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not installed")
class VRExAdapterTests(unittest.TestCase):
    def test_risk_variance_matches_domainbed_objective(self):
        import torch
        from torch.nn import functional as F

        from elderly_health.vrex_adapter import vrex_risk_loss

        logits = torch.tensor([[2.0], [2.0], [-2.0], [-2.0]], requires_grad=True)
        labels = torch.ones_like(logits)
        env = torch.tensor([0, 0, 1, 1])
        objective, mean, variance = vrex_risk_loss(
            logits, labels, env, positive_weight=1.0, penalty_weight=10.0
        )
        expected_losses = torch.stack([
            F.binary_cross_entropy_with_logits(logits[:2], labels[:2]),
            F.binary_cross_entropy_with_logits(logits[2:], labels[2:]),
        ])
        expected_mean = expected_losses.mean()
        expected_variance = ((expected_losses - expected_mean) ** 2).mean()
        torch.testing.assert_close(mean, expected_mean)
        torch.testing.assert_close(variance, expected_variance)
        torch.testing.assert_close(objective, expected_mean + 10 * expected_variance)
        objective.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_training_runs_without_data_leakage(self):
        from elderly_health.data import SplitData
        from elderly_health.vrex_adapter import VRExSettings, fit_vrex

        rng = np.random.default_rng(11)
        x = rng.normal(size=(140, 19)).astype(np.float32)
        y = np.tile([0.0, 1.0], 70).astype(np.float32)
        env = np.repeat(["east", "west"], 70)
        ids = np.arange(140)
        training = SplitData(x[:100], y[:100], env[:100], ids[:100])
        validation = SplitData(x[100:], y[100:], env[100:], ids[100:])
        # Both training domains must be present, and validation can use a
        # single seen domain. This test only checks program execution.
        settings = VRExSettings(
            epochs=3, selection_start_epoch=1, patience=2, threads=1,
            penalty_anneal_steps=2, per_environment_batch_size=8,
        )
        model, report = fit_vrex(training, validation, settings)
        self.assertGreaterEqual(report["best_epoch"], 1)
        self.assertLessEqual(report["best_epoch"], 3)
        self.assertEqual(set(report["training_domains"]), {"east", "west"})
        self.assertEqual(model.training, False)

    def test_invalid_config_rejected(self):
        from elderly_health.vrex_adapter import VRExSettings

        with self.assertRaises(ValueError):
            VRExSettings(epochs=20, selection_start_epoch=91)
        with self.assertRaises(ValueError):
            VRExSettings(penalty_anneal_steps=-1)


if __name__ == "__main__":
    unittest.main()

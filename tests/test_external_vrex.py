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
    def test_learning_rate_drop_preserves_adam_state(self):
        """任务四 第3步：学习率干预必须保留Adam动量和更新次数。"""
        import torch
        from elderly_health.vrex_adapter import apply_learning_rate_drop

        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.Adam([parameter], lr=3e-4)
        parameter.square().sum().backward()
        optimizer.step()
        state = optimizer.state[parameter]
        before = {k: v.clone() for k, v in state.items()}
        self.assertIsNone(apply_learning_rate_drop(optimizer, 1499, 1500, 0.05))
        event = apply_learning_rate_drop(optimizer, 1500, 1500, 0.05)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1.5e-5)
        self.assertFalse(event["optimizer_reset"])
        self.assertIs(state, optimizer.state[parameter])
        for key, value in before.items():
            torch.testing.assert_close(state[key], value, rtol=0, atol=0)
        self.assertIsNone(apply_learning_rate_drop(optimizer, 1501, 1500, 0.05))

    def test_learning_rate_control_has_identical_training_prefix(self):
        """任务四 第3步：用合成数据检查两组在干预前相同、只执行一次降速。"""
        from elderly_health.data import SplitData
        from elderly_health.vrex_adapter import VRExSettings, fit_vrex

        rng = np.random.default_rng(41)
        x = rng.normal(size=(160, 19)).astype(np.float32)
        y = np.tile([0.0, 1.0], 80).astype(np.float32)
        env = np.tile(np.repeat(["east", "west"], 40), 2)
        ids = np.arange(160)
        train = SplitData(x[:128], y[:128], env[:128], ids[:128])
        val = SplitData(x[128:], y[128:], env[128:], ids[128:])
        settings = VRExSettings(epochs=3, selection_start_epoch=1, patience=3,
                                threads=1, per_environment_batch_size=8, penalty_anneal_steps=2)
        _, control = fit_vrex(train, val, settings)
        _, treatment = fit_vrex(train, val, settings, lr_drop_step=10, lr_drop_factor=0.05)
        self.assertEqual(control["history"][0], treatment["history"][0])
        self.assertEqual(len(treatment["learning_rate_intervention"]["events"]), 1)
        self.assertEqual(control["learning_rate_intervention"]["events"], [])
        self.assertAlmostEqual(treatment["history"][-1]["learning_rate_end"], 1.5e-5)
        with self.assertRaises(ValueError):
            fit_vrex(train, val, settings, lr_drop_step=2, lr_drop_factor=0.05)

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


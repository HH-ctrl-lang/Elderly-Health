"""Source-derived GroupDRO adversarial group weights and CPU smoke tests."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project/src"))
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable")
class GroupDROAdapterTests(unittest.TestCase):
    def test_higher_loss_group_gains_adversarial_weight(self):
        import torch
        from elderly_health.groupdro_adapter import GroupDRO, GroupDROSettings

        settings = GroupDROSettings(
            epochs=3, selection_start_epoch=1, robust_step_size=0.1
        )
        model = GroupDRO(3, 2, settings)
        # With zero network parameters the logits are 0; weighted positives
        # have higher loss than negatives, so the first group gains weight.
        with torch.no_grad():
            for parameter in model.network.parameters():
                parameter.zero_()
        x = torch.zeros((8, 3))
        first = (x, torch.ones((8, 1)))
        second = (x, torch.zeros((8, 1)))
        result = model.update([first, second])
        self.assertGreater(result["group_losses"][0], result["group_losses"][1])
        self.assertGreater(result["group_weights"][0], 0.5)
        self.assertLess(result["group_weights"][1], 0.5)
        self.assertAlmostEqual(sum(result["group_weights"]), 1.0, places=6)
        expected = (
            result["group_losses"][0] * result["group_weights"][0]
            + result["group_losses"][1] * result["group_weights"][1]
        )
        self.assertAlmostEqual(result["robust_loss"], expected, places=5)

    def test_training_and_checkpoint_on_cpu(self):
        import torch
        from elderly_health.data import SplitData
        from elderly_health.groupdro_adapter import GroupDROSettings, fit_groupdro

        rng = np.random.default_rng(17)
        x = rng.normal(size=(140, 19)).astype(np.float32)
        y = np.tile([0, 1], 70).astype(np.float32)
        env = np.repeat(["east", "central"], 70)
        sample_ids = np.arange(140)
        train = SplitData(x[:100], y[:100], env[:100], sample_ids[:100])
        val = SplitData(x[100:], y[100:], env[100:], sample_ids[100:])
        settings = GroupDROSettings(
            epochs=3, selection_start_epoch=2, patience=2, threads=1,
            per_environment_batch_size=8,
        )
        model, record = fit_groupdro(train, val, settings)
        self.assertGreaterEqual(record["best_epoch"], 2)
        self.assertEqual(record["training_groups"], ["central", "east"])
        self.assertFalse(model.training)
        self.assertEqual(model(torch.tensor(x[:2])).shape, (2, 1))
        self.assertAlmostEqual(sum(record["selected_group_weights"]), 1.0, places=6)

    def test_invalid_hyperparameters(self):
        from elderly_health.groupdro_adapter import GroupDROSettings
        with self.assertRaises(ValueError):
            GroupDROSettings(epochs=3, selection_start_epoch=91)
        with self.assertRaises(ValueError):
            GroupDROSettings(robust_step_size=0)


if __name__ == "__main__":
    unittest.main()

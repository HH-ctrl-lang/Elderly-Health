"""Regression checks for leakage, graph roles, and experiment bookkeeping."""

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project/src"))

from elderly_health.config import TrainConfig  # noqa: E402
from elderly_health.data import (  # noqa: E402
    ENVIRONMENT,
    FEATURES,
    FeaturePreprocessor,
    SplitData,
    held_out_split,
    random_split,
)
from elderly_health.graph import load_graph_prior, prior_jaccard  # noqa: E402


def synthetic_frame():
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(size=(120, len(FEATURES))), columns=FEATURES)
    frame["label"] = np.tile([0, 1], 60)
    frame[ENVIRONMENT] = np.repeat(["A", "B", "C"], 40)
    frame["sample_id"] = np.arange(120)
    return frame


class DataTests(unittest.TestCase):
    def test_preprocessing_never_fits_validation_values(self):
        frame = synthetic_frame()
        train, val = frame.iloc[:80].copy(), frame.iloc[80:].copy()
        train.loc[0, "age"] = np.nan
        preprocessor = FeaturePreprocessor().fit(train)
        medians = preprocessor.imputer.statistics_.copy()
        means = preprocessor.scaler.mean_.copy()
        val.loc[:, list(FEATURES)] = 1e9
        result = preprocessor.transform(val)
        np.testing.assert_array_equal(medians, preprocessor.imputer.statistics_)
        np.testing.assert_array_equal(means, preprocessor.scaler.mean_)
        np.testing.assert_allclose(medians[1], train["age"].median())
        self.assertGreater(result.mean(), 1e6)

    def test_split_identity_and_reproducibility(self):
        first = random_split(synthetic_frame(), seed=0)
        second = random_split(synthetic_frame(), seed=0)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a.sample_id, b.sample_id)
        ids = [set(part.sample_id) for part in first]
        self.assertEqual(len(set.union(*ids)), 120)
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])

    def test_held_out_site_is_excluded_before_fitting(self):
        train, val, test = held_out_split(synthetic_frame(), "C")
        self.assertNotIn("C", set(train[ENVIRONMENT]) | set(val[ENVIRONMENT]))
        self.assertEqual(set(test[ENVIRONMENT]), {"C"})

    def test_all_missing_training_feature_fails(self):
        frame = synthetic_frame()
        frame["age"] = np.nan
        with self.assertRaisesRegex(ValueError, "Entirely missing"):
            FeaturePreprocessor().fit(frame)

    def test_loeo_baseline_does_not_inherit_causal_checkpoint_window(self):
        from elderly_health.cli import loeo_config_for_model

        causal = TrainConfig(model="v5", epochs=100, selection_start_epoch=91)
        self.assertEqual(loeo_config_for_model(causal, "v5").first_selection_epoch(), 91)
        baseline = loeo_config_for_model(causal, "baseline")
        self.assertEqual(baseline.model, "baseline")
        self.assertIsNone(baseline.selection_start_epoch)
        self.assertEqual(baseline.first_selection_epoch(), 1)
        self.assertEqual(causal.selection_start_epoch, 91)
        standalone = TrainConfig(model="baseline", epochs=100, selection_start_epoch=15)
        self.assertEqual(
            loeo_config_for_model(standalone, "baseline").first_selection_epoch(), 15
        )

    def test_help_without_torch_or_data_load(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "project/run.py"), "--help"],
            capture_output=True,
            text=True,
            check=True,
            cwd=tempfile.gettempdir(),
        )
        self.assertIn("preprocess", result.stdout)


class GraphTests(unittest.TestCase):
    def test_roles_have_one_consistent_type(self):
        with tempfile.TemporaryDirectory() as folder:
            graph = Path(folder) / "graph.dot"
            graph.write_text(
                'digraph { "age" -> cognitive_impairment; // comment\n'
                "cognitive_impairment -> depression; }"
            )
            prior = load_graph_prior(graph, "cognitive_impairment", ["age", "depression", "wealth"])
        self.assertEqual(prior, {"age": "parent", "depression": "child", "wealth": "unrelated"})
        self.assertEqual(prior_jaccard([0.9, 0.1, 0.1], list(prior), prior), 1.0)

    def test_invalid_configuration_fails_early(self):
        for params in ({"epochs": 0}, {"sparsity": "bad"}, {"target_sparsity": 2}):
            with self.assertRaises(ValueError):
                TrainConfig(**params)

    def test_selection_policy_validation_and_legacy_defaults(self):
        self.assertEqual(TrainConfig(model="baseline").first_selection_epoch(), 1)
        self.assertEqual(TrainConfig().first_selection_epoch(), 51)
        self.assertEqual(TrainConfig(epochs=3).first_selection_epoch(), 1)
        self.assertEqual(TrainConfig(selection_start_epoch=91).first_selection_epoch(), 91)
        for value in (0, -1, 301, 2.5, True, "91"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                TrainConfig(selection_start_epoch=value)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
class TorchTests(unittest.TestCase):
    def test_late_selection_excludes_early_best_without_changing_training(self):
        from unittest.mock import patch

        import torch
        from elderly_health.training import fit, predict
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(12)
        x = rng.normal(size=(48, 19)).astype(np.float32)
        y = np.tile([0, 1], 24).astype(np.float32)
        env = np.tile(["A", "A", "B", "B"], 12)
        train = SplitData(x[:32], y[:32], env[:32], np.arange(32))
        val = SplitData(x[32:], y[32:], env[32:], np.arange(32, 48))
        for model in ("baseline", "v5"):
            cfg = TrainConfig(
                model=model,
                epochs=4,
                batch_size=16,
                threads=1,
                probe_epochs=0,
                sparse_start=0,
                irm_start=0,
                ramp_epochs=1,
                patience=10,
                selection_start_epoch=3,
            )
            # Force the best reported early score to be outside the eligible window.
            # Epoch 3 must win over epoch 4, and its real predictions must be restored.
            actual_scores = []
            actual_predictions = []

            def score(y_true, probabilities):
                actual_scores.append(roc_auc_score(y_true, probabilities))
                actual_predictions.append(probabilities.copy())
                return [0.99, 0.98, 0.60, 0.55][len(actual_scores) - 1]

            with patch("elderly_health.training.roc_auc_score", side_effect=score):
                trained, report = fit(train, val, cfg)
            self.assertEqual(report["best_epoch"], 3)
            self.assertEqual(report["selection_start_epoch"], 3)
            self.assertEqual(
                [row["eligible_for_selection"] for row in report["history"]],
                [False, False, True, True],
            )
            self.assertAlmostEqual(
                roc_auc_score(val.y, predict(trained, val.x, torch.device("cpu"))),
                actual_scores[2],
            )
            np.testing.assert_array_equal(
                predict(trained, val.x, torch.device("cpu")), actual_predictions[2]
            )
            _, reference = fit(train, val, replace(cfg, selection_start_epoch=None))
            for before, after in zip(report["history"], reference["history"]):
                self.assertEqual(before["train_total"], after["train_total"])
                if model == "v5":
                    self.assertEqual(before["mask"], after["mask"])
            self.assertEqual(report["history"][2]["invariant_ramp"], float(model == "v5"))

    def test_graph_loss_has_nonzero_gradient(self):
        import torch
        from elderly_health.graph import graph_guided_loss

        mask = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
        graph_guided_loss(
            mask, ["a", "b", "c"], {"a": "parent", "b": "child", "c": "unrelated"}
        ).backward()
        self.assertLess(mask.grad[0].item(), 0)
        self.assertEqual(mask.grad[1].item(), 0)
        self.assertGreater(mask.grad[2].item(), 0)
        with self.assertRaises(ValueError):
            graph_guided_loss(mask, ["a", "b", "c"], {"a": 1.0})

    def test_all_objectives_and_best_checkpoint(self):
        import torch
        from elderly_health.training import fit, predict

        rng = np.random.default_rng(42)
        x = rng.normal(size=(48, 19)).astype(np.float32)
        y = np.tile([0, 1], 24).astype(np.float32)
        env = np.tile(["A", "A", "B", "B"], 12)
        train = SplitData(x[:32], y[:32], env[:32], np.arange(32))
        val = SplitData(x[32:], y[32:], env[32:], np.arange(32, 48))
        for model in ("baseline", "v4", "v5"):
            for irm in ("none", "standard", "v11") if model == "v5" else ("none",):
                for independence in ("cosine", "hsic") if model == "v5" else ("none",):
                    with self.subTest(model=model, irm=irm, independence=independence):
                        cfg = TrainConfig(
                            model=model,
                            epochs=3,
                            batch_size=16,
                            sparse_start=0,
                            irm_start=0,
                            irm_mode=irm,
                            independence=independence,
                            probe_epochs=0,
                            threads=1,
                        )
                        prior = {
                            name: "parent" if i == 0 else "unrelated"
                            for i, name in enumerate(FEATURES)
                        }
                        trained, report = fit(train, val, cfg, prior)
                        probabilities = predict(trained, val.x, torch.device("cpu"))
                        from sklearn.metrics import roc_auc_score

                        self.assertAlmostEqual(
                            roc_auc_score(val.y, probabilities), report["best_val_auc"]
                        )
                        self.assertTrue(np.isfinite(probabilities).all())
        cfg = replace(cfg, epochs=1)
        first, first_report = fit(train, val, cfg)
        second, _ = fit(train, val, cfg)
        np.testing.assert_array_equal(
            predict(first, val.x, torch.device("cpu")), predict(second, val.x, torch.device("cpu"))
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pt"
            torch.save(
                {
                    "state_dict": first.state_dict(),
                    "config": cfg.to_dict(),
                    "env_map": first_report["env_map"],
                },
                path,
            )
            checkpoint = torch.load(path, weights_only=True, map_location="cpu")
            self.assertEqual(checkpoint["env_map"], {"A": 0, "B": 1})


if __name__ == "__main__":
    unittest.main()

"""任务五 第3步：来源分组、泄漏防护、等权汇总及合成数据训练检查。"""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "project/src/elderly_health/conditional_mmd_adapter.py"
SPEC = importlib.util.spec_from_file_location("conditional_adapter", PATH)
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


def splits():
    """任务五 第3步：只用人工数据检查代码，不运行研究数据。"""
    rng = np.random.default_rng(5)
    train = SimpleNamespace(x=rng.normal(size=(80, 19)).astype(np.float32),
                            y=np.tile([0, 1], 40).astype(np.float32),
                            env=np.repeat(["east", "west"], 40), sample_ids=np.arange(80))
    val = SimpleNamespace(x=rng.normal(size=(20, 19)).astype(np.float32),
                          y=np.tile([0, 1], 10).astype(np.float32),
                          env=np.repeat(["east", "west"], 10), sample_ids=np.arange(80, 100))
    return train, val


class HealthScreenTests(unittest.TestCase):
    def test_conditional_groups_use_matching_class_and_domain(self):
        train, _ = splits()
        domains, _, groups = ADAPTER.training_groups(train)
        self.assertEqual(domains, ["east", "west"])
        for (domain, label), ids in groups.items():
            self.assertTrue((train.env[ids] == domain).all())
            self.assertTrue((train.y[ids] == label).all())

    def test_missing_source_class_rejected(self):
        train, _ = splits()
        train.y[:40] = 0
        with self.assertRaises(ValueError):
            ADAPTER.training_groups(train)

    def test_overlapping_validation_rejected(self):
        train, val = splits()
        ADAPTER.verify_source_splits(train, val)
        val.sample_ids[0] = train.sample_ids[0]
        with self.assertRaises(ValueError):
            ADAPTER.verify_source_splits(train, val)

    def test_site_macro_does_not_weight_large_sites_more(self):
        """任务五 第3步：独立加载纯汇总函数，不依赖PyTorch。"""
        path = ROOT / "project/src/analysis/task5_step3_health_disentanglement_screen.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "macro_metrics"]
        namespace = {"pd": pd, "REGIONS": ("east", "central", "west")}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        rows = [{"held_out_region": "east", "model": "control", "stage": "before", "n": n,
                 "auc": auc, "auprc": auc, "brier": 0.1, "log_loss": 0.3}
                for n, auc in ((100, 0.5), (1000, 0.9))]
        result = namespace["macro_metrics"](pd.DataFrame(rows))
        self.assertAlmostEqual(float(result.loc[result.scope.eq("ALL"), "auc"].iloc[0]), 0.7)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
    def test_synthetic_fit_has_paired_initialization_and_sampling(self):
        train, val = splits()
        with patch.dict(ADAPTER.SETTINGS, {"epochs": 3, "selection_start_epoch": 1, "patience": 3}):
            model, control = ADAPTER.fit(train, val, 0.0)
            _, aligned = ADAPTER.fit(train, val, 0.1)
        self.assertEqual(control["initial_weights_sha256"], aligned["initial_weights_sha256"])
        self.assertEqual(control["sampling_prefix_sha256"], aligned["sampling_prefix_sha256"])
        self.assertTrue(np.isfinite(ADAPTER.predict(model, val.x)).all())
        self.assertTrue(all(np.isfinite(v) for v in ADAPTER.representation_diagnostics(model, val).values()))


if __name__ == "__main__":
    unittest.main()

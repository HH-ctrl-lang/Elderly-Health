"""任务五 第2步：防止历史协议变化和读出复核失败被忽略。"""

import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project/src/analysis"))
from task5_step2_mmd_weight_control import check_protocol, compare_recovery


class WeightControlTests(unittest.TestCase):
    def test_protocol_rejects_changed_budget(self):
        metadata = {"samples_per_environment": 6000, "test_samples_per_environment": 2000,
                    "epochs": 100, "batch_size_per_environment": 256, "mmd_weight": 1.0,
                    "seed": 42, "health_data_used": False, "latent_truth_used_for_autoencoder_training": False,
                    "reproduction_level": "independent_downscaled_mechanism_reproduction",
                    "mmd_kernel_gammas": [0.5, 1.0, 2.0],
                    "nintervention_means": np.random.default_rng(42).uniform(2, 5, size=3).tolist(),
                    "script_sha256": "294342d9c38dabbd00cd0b860c8f545687eea40bfc3d9d08686ef0ddee413a32"}
        self.assertEqual(len(check_protocol(metadata)), 3)
        with self.assertRaises(ValueError):
            check_protocol({**metadata, "epochs": 200})
        with self.assertRaises(ValueError):
            check_protocol({**metadata, "nintervention_means": [2, 3, 4]})
        with self.assertRaises(ValueError):
            check_protocol({**metadata, "script_sha256": "changed"})

    def test_frozen_reproduction_rejects_difference(self):
        rows = [{"trial": 0, "arm": "old", "representation": f"block_{i}",
                 "latent": f"z{j}", "test_r2": 0.8}
                for i in range(4) for j in range(3)]
        previous = pd.DataFrame(rows)
        self.assertEqual(len(compare_recovery(previous, rows, 0, "old")), 12)
        changed = [dict(row) for row in rows]
        changed[0]["test_r2"] = 0.9
        with self.assertRaises(RuntimeError):
            compare_recovery(previous, changed, 0, "old")
        with self.assertRaises(ValueError):
            compare_recovery(previous.iloc[:11], rows, 0, "old")


if __name__ == "__main__":
    unittest.main()

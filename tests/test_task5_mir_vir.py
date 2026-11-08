"""任务五 第4步：公式、梯度、优化收敛与来源验证选参检查。"""

import sys
from pathlib import Path
import unittest

import numpy as np
from scipy.optimize import check_grad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project/src"))
sys.path.insert(0, str(ROOT / "project/src/analysis"))
from elderly_health.anti_causal_regularization import (
    environment_matrices, environment_weights, objective_gradient, fit_logistic,
)
from task5_step4_mir_vir_health_screen import choose_candidate


class MIRVIRTests(unittest.TestCase):
    def test_exact_mean_covariance_and_matrix_square(self):
        """任务五 第4步：独立手算例子区分矩阵平方与逐元素平方。"""
        x = np.array([[0, 0], [1, 1], [-1, -1], [2, 0], [2, 1], [2, -1]], dtype=float)
        h, _, _, _ = environment_matrices(x, np.repeat(["a", "b"], 3))
        np.testing.assert_allclose(h["MIR"], [[1, 0], [0, 0]], atol=1e-12)
        np.testing.assert_allclose(h["VIR"], [[2/9, 1/9], [1/9, 1/9]], atol=1e-12)

    def test_environment_weight_is_independent_of_group_size(self):
        weights = environment_weights(np.array(["a"]*2+["b"]*8))
        self.assertAlmostEqual(weights[:2].sum(), 0.5)
        self.assertAlmostEqual(weights[2:].sum(), 0.5)

    def test_analytic_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(40, 3))
        y = np.tile([0.0, 1.0], 20)
        weights = environment_weights(np.repeat(["a", "b"], 20))
        theta = rng.normal(size=4)
        h = np.array([[2, 1, 0], [1, 2, 0], [0, 0, 1]], dtype=float)
        args = (x, y, weights, h, 0.3, 1e-5)
        error = check_grad(lambda t: objective_gradient(t, *args)[0],
                           lambda t: objective_gradient(t, *args)[1], theta)
        self.assertLess(error, 1e-6)

    def test_convex_fit_converges_and_zero_penalty_matches_control(self):
        """任务五 第4步：只用人工数据检验求解器，不运行健康研究实验。"""
        rng = np.random.default_rng(3)
        x = rng.normal(size=(120, 4))
        y = (x[:, 0]+rng.normal(size=120) > 0).astype(float)
        env = np.repeat(["a", "b", "c"], 40)
        base, report = fit_logistic(x, y, env, np.zeros((4, 4)), 0)
        same, _ = fit_logistic(x, y, env, np.eye(4), 0)
        penalized, _ = fit_logistic(x, y, env, np.eye(4), 1)
        self.assertTrue(report["success"])
        np.testing.assert_allclose(base, same, atol=1e-10)
        self.assertLess(np.linalg.norm(penalized[:-1]), np.linalg.norm(base[:-1]))

    def test_selection_uses_source_score_and_allows_zero(self):
        candidates = [{"gamma": 0.0, "source_val_auc": 0.7, "target_auc": 0.1},
                      {"gamma": 1.0, "source_val_auc": 0.6, "target_auc": 0.99},
                      {"gamma": 0.1, "source_val_auc": 0.7, "target_auc": 0.99}]
        self.assertEqual(choose_candidate(candidates)["gamma"], 0.0)

    def test_invalid_penalty_rejected(self):
        x = np.ones((6, 2))
        y = np.tile([0.0, 1.0], 3)
        with self.assertRaises(ValueError):
            fit_logistic(x, y, np.repeat(["a", "b"], 3), -np.eye(2), 1)


if __name__ == "__main__":
    unittest.main()

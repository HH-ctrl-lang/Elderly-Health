"""任务五 第1步：验证模拟机制和防止训练/评估数据混用。"""

import importlib.util
from pathlib import Path
import unittest

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "project/src/analysis/task5_step1_invariance_disentanglement_pilot.py"
SPEC = importlib.util.spec_from_file_location("invariance_pilot", PATH)
PILOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PILOT)


class InvariancePilotTests(unittest.TestCase):
    def test_nintervention_keeps_old_descendant(self):
        """任务五 第1步：防止错误地把nintervention写成向下游传播的do干预。"""
        old = PILOT.latent_sample(2000, 7)
        changed = PILOT.latent_sample(2000, 7, 3.0)
        np.testing.assert_array_equal(old[:, [0, 2]], changed[:, [0, 2]])
        self.assertFalse(np.array_equal(old[:, 1], changed[:, 1]))
        self.assertAlmostEqual(float(changed[:, 1].mean()), 3.0, delta=0.002)
        self.assertAlmostEqual(float(changed[:, 1].std()), 0.02, delta=0.002)

    def test_mixing_is_invertible(self):
        """任务五 第1步：生成观测的步骤不能先丢失真实变量。"""
        z = PILOT.latent_sample(100, 8, 2.5)
        matrices = PILOT.make_mixing(9)
        recovered = PILOT.mix(z, matrices)
        for i in (2, 1, 0):
            if i < 2:
                recovered = np.where(recovered >= 0, recovered, recovered/0.2)
            recovered = recovered @ matrices[i].T
        recovered = recovered*np.array([0.8, 1.0, 1.0]) + np.array([10.5, 0.0, 3.0])
        np.testing.assert_allclose(recovered, z, atol=1e-12, rtol=0)

    def test_standardization_fits_training_only(self):
        """任务五 第1步：独立评估样本不能参与预处理。"""
        a, b, za, zb, matrices, scaler = PILOT.make_data(100, 80, 0, 3.2)
        pooled = np.vstack([PILOT.mix(z, matrices) for z in za])
        np.testing.assert_allclose(scaler.mean_, pooled.mean(0))
        self.assertEqual(a[0].shape, (100, 3))
        self.assertEqual(b[0].shape, (80, 3))
        self.assertFalse(np.array_equal(za[0][:80], zb[0]))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch not installed")
    def test_mmd_symmetry_and_finite_gradient(self):
        """任务五 第1步：检查核距离、批次大小不等和反向传播。"""
        import torch
        a = torch.tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
        b = torch.tensor([[3.0, 4.0], [4.0, 3.0], [4.0, 4.0]])
        value = PILOT.gaussian_mmd(a, b)
        torch.testing.assert_close(value, PILOT.gaussian_mmd(b, a))
        self.assertGreater(value.item(), 0)
        self.assertAlmostEqual(PILOT.gaussian_mmd(a, a).item(), 0, places=6)
        value.backward()
        self.assertTrue(torch.isfinite(a.grad).all())


if __name__ == "__main__":
    unittest.main()

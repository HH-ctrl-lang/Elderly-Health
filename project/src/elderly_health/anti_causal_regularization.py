"""任务五 第4步：ICML2026 MIR/VIR公式的独立实现及二分类适配。

依据论文公式(9)(10)与§6.1。无神经表征拆分，不宣称因果识别。
仅使用NumPy/SciPy；真实研究实验由用户运行。
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def environment_matrices(x, environments):
    """任务五 第4步：MIR为环境均值协方差，VIR为协方差偏差的矩阵平方平均。"""
    x = np.asarray(x, dtype=float)
    env = np.asarray(environments, dtype=str)
    if x.ndim != 2 or len(x) != len(env) or not np.isfinite(x).all():
        raise ValueError("训练特征形状或有限性不符合要求。")
    groups = sorted(set(env))
    if len(groups) < 2:
        raise ValueError("至少需要两个来源环境。")
    means, covariances = [], []
    for group in groups:
        values = x[env == group]
        if len(values) < 2:
            raise ValueError("每个来源环境至少两个样本。")
        mean = values.mean(0)
        centered = values-mean
        means.append(mean)
        # 任务五 第4步：原文使用1/n，而不是1/(n-1)。
        covariances.append(centered.T@centered/len(values))
    means, covariances = np.asarray(means), np.asarray(covariances)
    centered_means = means-means.mean(0)
    mir = centered_means.T@centered_means/len(groups)
    differences = covariances-covariances.mean(0)
    # 任务五 第4步：这里必须是矩阵乘法D@D，不能写成逐元素D**2。
    vir = sum(d@d for d in differences)/len(groups)
    matrices = {"MIR": (mir+mir.T)/2, "VIR": (vir+vir.T)/2}
    for matrix in matrices.values():
        if np.linalg.eigvalsh(matrix).min() < -1e-9:
            raise RuntimeError("正则矩阵不是半正定矩阵，停止拟合。")
    return matrices, groups, means, covariances


def environment_weights(environments):
    """任务五 第4步：来源环境等权，环境内部按自然样本比例计算未加权BCE。"""
    env = np.asarray(environments, dtype=str)
    groups = sorted(set(env))
    if not groups:
        raise ValueError("没有来源环境。")
    weights = np.zeros(len(env), dtype=float)
    for group in groups:
        mask = env == group
        weights[mask] = 1/(len(groups)*mask.sum())
    return weights


def objective_gradient(theta, x, y, weights, penalty_matrix, gamma, ridge):
    """任务五 第4步：稳定计算二分类目标与解析梯度，截距不受正则处罚。"""
    beta, intercept = theta[:-1], theta[-1]
    score = x@beta+intercept
    residual = weights*(expit(score)-y)
    value = weights@(np.logaddexp(0, score)-y*score) + gamma*(beta@penalty_matrix@beta) + ridge*(beta@beta)
    gradient = np.r_[x.T@residual+2*gamma*(penalty_matrix@beta)+2*ridge*beta, residual.sum()]
    return float(value), gradient


def fit_logistic(x, y, environments, penalty_matrix, gamma, ridge=1e-5):
    """任务五 第4步：从零初始化解凸优化问题；不接收验证或留出地区数据。"""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    penalty_matrix = np.asarray(penalty_matrix, dtype=float)
    if set(np.unique(y)) != {0.0, 1.0} or x.shape[0] != len(y):
        raise ValueError("训练标签需要是包含两类的0/1数组。")
    if not np.isfinite(x).all() or not np.isfinite(penalty_matrix).all():
        raise ValueError("输入含非有限值。")
    if penalty_matrix.shape != (x.shape[1], x.shape[1]):
        raise ValueError("正则矩阵形状错误。")
    if not np.isfinite(gamma) or gamma < 0 or not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("正则参数无效。")
    if (not np.allclose(penalty_matrix, penalty_matrix.T, atol=1e-10, rtol=0)
            or np.linalg.eigvalsh(penalty_matrix).min() < -1e-9):
        raise ValueError("正则矩阵需要对称半正定。")
    weights = environment_weights(environments)
    if len(weights) != len(y):
        raise ValueError("来源环境长度错误。")
    result = minimize(objective_gradient, np.zeros(x.shape[1]+1),
                      args=(x, y, weights, penalty_matrix, gamma, ridge), jac=True,
                      method="L-BFGS-B", options={"maxiter": 5000, "gtol": 1e-8, "ftol": 1e-13, "maxls": 50})
    gradient_norm = float(np.max(np.abs(result.jac)))
    if not result.success or not np.isfinite(result.x).all() or gradient_norm > 2e-6:
        raise RuntimeError(f"优化未可靠收敛：{result.message}; max_gradient={gradient_norm}")
    return result.x, {"success": True, "iterations": int(result.nit), "objective": float(result.fun),
                      "gradient_max_abs": gradient_norm, "message": str(result.message)}


def probabilities(theta, x):
    """任务五 第4步：冻结线性模型后预测概率。"""
    return expit(np.asarray(x)@theta[:-1]+theta[-1])

"""IRM and independence objectives. v11 is a project gradient-variance heuristic.

The v11 name is kept for compatibility, not claimed as an established IRM variant.
"""

import torch
from torch.nn import functional as F


def irm_penalty(model, z_c, y, env_ids, mode="standard", eps=1e-8):
    """
    可切换的 IRM 惩罚：
    - none: 关闭 IRM
    - standard: 标准 IRM，惩罚每个环境下的梯度与虚拟缩放系数的相关性
    - v11: 项目自定义梯度方差惩罚，不等同于标准 IRMv1
    """
    if mode == "none":
        return torch.tensor(0.0, device=z_c.device)

    unique_envs = env_ids.unique()

    if mode == "standard":
        predictor = model.predictor_zc
        total_penalty = 0.0
        n_env = 0

        for env in unique_envs:
            idx = env_ids == env
            if idx.sum() < 2:
                continue

            z_env = z_c[idx]
            y_env = y[idx]

            scale = torch.tensor(1.0, device=z_env.device, requires_grad=True)
            logits = predictor(z_env)
            loss = F.binary_cross_entropy_with_logits(scale * logits, y_env)
            grad = torch.autograd.grad(loss, scale, create_graph=True)[0]
            total_penalty += grad.pow(2).mean()
            n_env += 1

        if n_env == 0:
            return torch.tensor(0.0, device=z_c.device)
        return total_penalty / n_env

    if mode == "v11":
        grads = []
        for env in unique_envs:
            idx = env_ids == env
            if idx.sum() < 2:
                continue

            logits = model.predictor_zc(z_c[idx])
            loss_e = F.binary_cross_entropy_with_logits(logits, y[idx])
            grad = torch.autograd.grad(
                loss_e, model.predictor_zc.net[-1].weight, create_graph=True
            )[0]
            grads.append(grad.view(-1))

        if len(grads) < 2:
            return torch.tensor(0.0, device=z_c.device)

        grads = torch.stack(grads)
        g_mean = grads.mean(0)
        penalty = ((grads - g_mean) ** 2).mean()
        grad_norm = grads.norm(dim=1).mean().clamp_min(eps)
        return penalty / grad_norm

    raise ValueError(f"Unknown IRM mode: {mode}")


def hsic_loss(x, y, sigma=1.0):
    """
    HSIC (RBF kernel version)
    """

    def rbf_kernel(a, b, sigma):
        dist = torch.cdist(a, b).pow(2)
        return torch.exp(-dist / (2 * sigma**2))

    n = x.size(0)
    if n <= 1:
        return torch.tensor(0.0, device=x.device)

    K = rbf_kernel(x, x, sigma)
    L = rbf_kernel(y, y, sigma)
    H = torch.eye(n, device=x.device) - (1.0 / n)

    Kc = H @ K @ H
    Lc = H @ L @ H
    return torch.trace(Kc @ Lc) / ((n - 1) ** 2)


def independence_loss(z_c, z_s, mode="cosine", sigma=1.0):
    """
    mode:
        - cosine
        - hsic
    """
    if mode == "none":
        return z_c.sum() * 0.0
    if mode == "cosine":
        z_c_centered = z_c - z_c.mean(dim=0, keepdim=True)
        z_s_centered = z_s - z_s.mean(dim=0, keepdim=True)
        cross = torch.mm(
            F.normalize(z_c_centered, p=2, dim=0).T, F.normalize(z_s_centered, p=2, dim=0)
        )
        return (cross**2).mean()
    elif mode == "hsic":
        return hsic_loss(z_c, z_s, sigma=sigma)
    else:
        raise ValueError(f"Unknown mode: {mode}")

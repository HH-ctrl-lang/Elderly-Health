"""任务五 第3步：预测块＋残余块、重建与来源类别条件MMD的探索性改编。

不是ICLR2025论文原样复现，不将任一块命名为已识别的因果变量。
fit只接收来源训练集和来源验证集，不能接触留出地区数据。
"""

import copy
import hashlib
import math
import random

import numpy as np
from sklearn.metrics import roc_auc_score

SETTINGS = {"seed": 42, "epochs": 300, "selection_start_epoch": 91, "patience": 80,
            "learning_rate": 3e-4, "weight_decay": 1e-5, "positive_weight": 3.0,
            "per_environment_batch_size": 32, "mmd_per_class_environment": 16,
            "prediction_block_dim": 8, "residual_block_dim": 8, "reconstruction_weight": 1.0,
            "threads": 2, "device": "cpu"}


def training_groups(train):
    """任务五 第3步：只用来源训练样本构造地区与类别分组，拒绝缺类情形。"""
    domains = sorted(set(map(str, train.env)))
    if len(domains) != 2 or set(np.unique(train.y)) != {0, 1}:
        raise ValueError("本步骤预定使用两个来源地区和二分类标签。")
    groups = {domain: np.flatnonzero(train.env == domain) for domain in domains}
    conditional = {(domain, label): np.flatnonzero((train.env == domain) & (train.y == label))
                   for label in (0, 1) for domain in domains}
    if any(len(indices) < 2 for indices in conditional.values()):
        raise ValueError("每个来源地区的每一类别至少需要两个训练样本。")
    return domains, groups, conditional


def verify_source_splits(train, val):
    """任务五 第3步：训练与验证样本不交叉，验证环境必须在来源训练中出现。"""
    if set(train.sample_ids) & set(val.sample_ids):
        raise ValueError("训练与验证样本重叠。")
    if set(val.env) - set(train.env) or set(np.unique(val.y)) != {0, 1}:
        raise ValueError("来源验证集环境或类别不符合方案。")
    if not np.isfinite(train.x).all() or not np.isfinite(val.x).all():
        raise ValueError("预处理后仍有非有限输入。")


def build_model(input_dim):
    """任务五 第3步：两块共同重建输入，分类仅使用预测块；两实验组结构相同。"""
    from torch import nn
    def encoder(size):
        return nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, size))
    model = nn.ModuleDict({
        "prediction_encoder": nn.Sequential(encoder(8), nn.LayerNorm(8, elementwise_affine=False)),
        "residual_encoder": encoder(8),
        "classifier": nn.Linear(8, 1),
        "decoder": nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, input_dim)),
    })
    return model


def mmd_squared(left, right):
    """任务五 第3步：固定维度缩放的多尺度高斯核，避免按留出数据选择带宽。"""
    import torch
    def kernel(a, b):
        distance = torch.cdist(a, b).square()/a.shape[1]
        return sum(torch.exp(-gamma*distance) for gamma in (0.5, 1.0, 2.0))/3
    return kernel(left, left).mean() + kernel(right, right).mean() - 2*kernel(left, right).mean()


def predict(model, x):
    """任务五 第3步：固定CPU预测，不导出逐人预测。"""
    import torch
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(x), 256):
            block = model["prediction_encoder"](torch.as_tensor(x[start:start+256]))
            probabilities.append(torch.sigmoid(model["classifier"](block)).numpy().ravel())
    return np.concatenate(probabilities)


def representation_diagnostics(model, split):
    """任务五 第3步：仅输出重建误差、块方差和有效秩等聚合诊断。"""
    import torch
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(split.x)
        a = model["prediction_encoder"](x)
        b = model["residual_encoder"](x)
        reconstruction = model["decoder"](torch.cat([a, b], dim=1))
        result = {"reconstruction_mse": float((reconstruction-x).square().mean())}
    for name, block in (("prediction", a.numpy()), ("residual", b.numpy())):
        std = block.std(0)
        result.update({f"{name}_mean_std": float(std.mean()),
                       f"{name}_min_std": float(std.min()),
                       f"{name}_numeric_rank": int(np.linalg.matrix_rank(block-block.mean(0)))})
    return result


def fit(train, val, mmd_weight):
    """任务五 第3步：只允许预定0/0.1两组；选模仅看来源验证AUC。"""
    if mmd_weight not in (0.0, 0.1):
        raise ValueError("本次筛选只比较固定MMD权重0与0.1。")
    verify_source_splits(train, val)
    domains, groups, conditional = training_groups(train)
    import torch
    random.seed(SETTINGS["seed"])
    np.random.seed(SETTINGS["seed"])
    torch.manual_seed(SETTINGS["seed"])
    torch.set_num_threads(SETTINGS["threads"])
    torch.use_deterministic_algorithms(True)
    model = build_model(train.x.shape[1])
    initial_hash = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS["learning_rate"], weight_decay=SETTINGS["weight_decay"])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([SETTINGS["positive_weight"]]))
    # 任务五 第3步：预测批次保留来源类别比例；MMD另采同类批次，不改变预测损失采样。
    batch_rng = torch.Generator().manual_seed(SETTINGS["seed"])
    mmd_rng = torch.Generator().manual_seed(SETTINGS["seed"]+1)
    sampling_hash = hashlib.sha256()
    x_all = torch.as_tensor(train.x)
    y_all = torch.as_tensor(train.y[:, None])
    steps = math.ceil(len(train.x)/(len(domains)*SETTINGS["per_environment_batch_size"]))
    history, best_state, best_auc, best_epoch, stale = [], None, -float("inf"), 0, 0
    prefix_hashes = {}
    for epoch in range(1, SETTINGS["epochs"]+1):
        model.train()
        totals = np.zeros(3)
        for _ in range(steps):
            ids = np.concatenate([indices[torch.randint(len(indices), (32,), generator=batch_rng).numpy()]
                                  for indices in (groups[d] for d in domains)])
            sampling_hash.update(ids.astype(np.int64).tobytes())
            x, y = x_all[ids], y_all[ids]
            prediction_block = model["prediction_encoder"](x)
            residual_block = model["residual_encoder"](x)
            prediction_loss = criterion(model["classifier"](prediction_block), y)
            reconstruction = model["decoder"](torch.cat([prediction_block, residual_block], dim=1))
            reconstruction_loss = (reconstruction-x).square().mean()
            penalties = []
            for label in (0, 1):
                class_blocks = []
                for domain in domains:
                    positions = conditional[(domain, label)]
                    selected = positions[torch.randint(len(positions), (16,), generator=mmd_rng).numpy()]
                    sampling_hash.update(selected.astype(np.int64).tobytes())
                    class_blocks.append(model["prediction_encoder"](x_all[selected]))
                penalties.append(mmd_squared(*class_blocks))
            penalty = torch.stack(penalties).mean()
            objective = prediction_loss + reconstruction_loss + mmd_weight*penalty
            if not torch.isfinite(objective):
                raise FloatingPointError(f"非有限训练损失，epoch={epoch}")
            optimizer.zero_grad()
            objective.backward()
            optimizer.step()
            totals += [prediction_loss.item(), reconstruction_loss.item(), penalty.item()]
        val_auc = float(roc_auc_score(val.y, predict(model, val.x)))
        prefix_hashes[str(epoch)] = sampling_hash.hexdigest()
        selectable = epoch >= SETTINGS["selection_start_epoch"]
        history.append({"epoch": epoch, "val_auc": val_auc, "eligible_for_selection": selectable,
                        **dict(zip(("prediction_loss", "reconstruction_loss", "conditional_mmd"), (totals/steps).tolist()))})
        if selectable:
            if val_auc > best_auc:
                best_auc, best_epoch, stale = val_auc, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
        if epoch == 1 or epoch % 30 == 0:
            print(f"MMD={mmd_weight}, epoch={epoch}, source_val_auc={val_auc:.4f}", flush=True)
        if selectable and stale >= SETTINGS["patience"]:
            break
    if best_state is None:
        raise RuntimeError("没有可用来源验证检查点。")
    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch, "best_val_auc": best_auc, "mmd_weight": mmd_weight,
                   "initial_weights_sha256": initial_hash, "sampling_prefix_sha256": prefix_hashes,
                   "source_domains": domains, "steps_per_epoch": steps, "history": history}

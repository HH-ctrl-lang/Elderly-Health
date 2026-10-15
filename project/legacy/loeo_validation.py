"""
LOEO: Leave-One-Environment-Out 外部验证
读 best_params.json → 用最优超参跑 12 轮 LOEO → 对比 V4 vs MLP
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# 将 train_v4 的路径加入 sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src", "train"))
from train_v4 import CausalDisentangledModel, irm_penalty, cosine_independence_loss, LATENT_DIM

# ============================================================
# 0. 加载配置
# ============================================================
DATA_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "preprocessed_data.npz")
PARAMS_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "best_params.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "preprocessed")

with open(PARAMS_PATH) as f:
    cfg = json.load(f)

# 使用 best_params 的超参
LAMBDA_SPARSE = cfg["lambda_sparse"]
LAMBDA_ENTROPY = cfg["lambda_entropy"]
LAMBDA_IRM = cfg["lambda_irm"]
LAMBDA_ENV = cfg["lambda_env"]
LAMBDA_COSINE = cfg["lambda_cosine"]
GATE_INIT_STD = cfg["gate_init_std"]
LR = cfg["lr"]
POS_WEIGHT = cfg["pos_weight"]
TARGET_SPARSITY = cfg["target_sparsity"]
BATCH_SIZE = cfg["batch_size"]
NUM_EPOCHS = cfg["num_epochs"]
WARMUP = cfg["warmup_epochs"]
SPARSE_START = cfg["sparse_start_epoch"]
IRM_START = cfg["irm_start_epoch"]
GATE_TEMP = cfg["gate_temperature"]

EARLY_STOP_PATIENCE = 80
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"设备: {device}")
print(f"λ_sparse={LAMBDA_SPARSE}, λ_entropy={LAMBDA_ENTROPY}, λ_IRM={LAMBDA_IRM}, λ_ENV={LAMBDA_ENV}")
print(f"lr={LR}, pos_weight={POS_WEIGHT}, target_sparsity={TARGET_SPARSITY:.3f}")

# ============================================================
# 1. 加载数据
# ============================================================
data = np.load(DATA_PATH)
X_all = np.concatenate([data["X_train"], data["X_val"], data["X_test"]], axis=0)
y_all = np.concatenate([data["y_train"], data["y_val"], data["y_test"]], axis=0)
env_all = np.concatenate([data["env_train"], data["env_val"], data["env_test"]], axis=0)
feature_names = data["feature_names"]
y_all = (y_all == 2.0).astype(np.float32)

# 将环境码映射到 0-11
env_codes = sorted(np.unique(env_all))
env_map_global = {code: i for i, code in enumerate(env_codes)}
env_all_mapped = np.array([env_map_global[e] for e in env_all])

num_envs = len(env_codes)
input_dim = X_all.shape[1]
print(f"数据: {len(X_all)} 样本, {input_dim} 特征, {num_envs} 环境")

for e in range(num_envs):
    mask = env_all_mapped == e
    print(f"  环境 {e}: {mask.sum()} 样本, 正例率={y_all[mask].mean():.3f}")

# 图先验加载
sys.path.insert(0, os.path.join(BASE_DIR, "src", "train"))
from graph_prior import load_graph_prior, graph_guided_loss
GRAPH_DOT_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "causal_graph.dot")
try:
    GRAPH_PRIOR = load_graph_prior(GRAPH_DOT_PATH, "cognitive_impairment", list(feature_names))
    USE_GRAPH_PRIOR = True
    LAMBDA_GRAPH = 0.1
    print(f"图先验已加载, LAMBDA_GRAPH={LAMBDA_GRAPH}")
except FileNotFoundError:
    print("[WARN] 因果图未找到，跳过图先验")
    USE_GRAPH_PRIOR = False
    LAMBDA_GRAPH = 0.0

# ============================================================
# 2. Baseline MLP
# ============================================================
class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x)


# ============================================================
# 3. 训练函数
# ============================================================
def train_causal_loeo(X_tr, y_tr, env_tr, X_vl, y_vl, env_vl, num_envs_fold):
    """分阶段训练因果解耦模型 (复用 train_v4 的逻辑)"""
    tr_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
        torch.tensor(env_tr, dtype=torch.long),
    )
    vl_ds = TensorDataset(
        torch.tensor(X_vl, dtype=torch.float32),
        torch.tensor(y_vl, dtype=torch.float32).unsqueeze(1),
        torch.tensor(env_vl, dtype=torch.long),
    )
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    vl_loader = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = CausalDisentangledModel(input_dim, num_envs_fold, LATENT_DIM, GATE_INIT_STD).to(device)
    # 为 gate 设置温度
    model.gate.temperature = GATE_TEMP

    pw = torch.tensor([POS_WEIGHT]).to(device)
    crit_pred = nn.BCEWithLogitsLoss(pos_weight=pw)
    crit_env = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=LR)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=20)

    best_auc, patience = 0.0, 0
    irm_reset_done = False
    for epoch in range(NUM_EPOCHS):
        is_warmup = epoch < WARMUP
        is_sparse = epoch >= SPARSE_START
        is_irm = epoch >= IRM_START
        sp_ramp = min(1.0, (epoch - SPARSE_START) / 40.0) if is_sparse else 0.0
        irm_ramp = min(1.0, (epoch - IRM_START) / 40.0) if is_irm else 0.0

        if is_irm and not irm_reset_done:
            best_auc = 0.0; patience = 0; irm_reset_done = True

        model.train()
        for X_b, y_b, env_b in tr_loader:
            X_b, y_b, env_b = X_b.to(device), y_b.to(device), env_b.to(device)
            logits, env_logits, z_c, z_s, mask = model(X_b)

            loss = crit_pred(logits, y_b)
            if is_sparse and sp_ramp > 0:
                loss += sp_ramp * LAMBDA_SPARSE * torch.abs(mask.mean() - TARGET_SPARSITY)
                loss += sp_ramp * LAMBDA_ENTROPY * (mask * (1.0 - mask)).mean()
            if is_irm and irm_ramp > 0:
                loss += irm_ramp * LAMBDA_IRM * irm_penalty(z_c, y_b, env_b)
                loss += irm_ramp * LAMBDA_ENV * crit_env(env_logits, env_b)
                loss += irm_ramp * LAMBDA_COSINE * cosine_independence_loss(z_c, z_s)
            if USE_GRAPH_PRIOR and is_sparse and sp_ramp > 0:
                loss += sp_ramp * LAMBDA_GRAPH * graph_guided_loss(mask, feature_names, GRAPH_PRIOR)

            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        preds, targets, vloss = [], [], 0.0
        with torch.no_grad():
            for X_b, y_b, _ in vl_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits = model.predict(X_b)
                vloss += crit_pred(logits, y_b).item() * X_b.size(0)
                preds.extend(torch.sigmoid(logits).cpu().numpy())
                targets.extend(y_b.cpu().numpy())
        val_auc = roc_auc_score(targets, np.array(preds).squeeze())
        sch.step(vloss / len(vl_ds))

        if val_auc > best_auc:
            best_auc = val_auc; patience = 0
        else:
            patience += 1
            min_irm_epochs = IRM_START + 50
            if not is_warmup and epoch > min_irm_epochs and patience >= EARLY_STOP_PATIENCE:
                break

    model.eval()
    return model, best_auc


def train_mlp_loeo(X_tr, y_tr, X_vl, y_vl):
    """训练简单 MLP"""
    tr_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
    )
    vl_ds = TensorDataset(
        torch.tensor(X_vl, dtype=torch.float32),
        torch.tensor(y_vl, dtype=torch.float32).unsqueeze(1),
    )
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    vl_loader = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleMLP(input_dim).to(device)
    pw = torch.tensor([POS_WEIGHT]).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = optim.Adam(model.parameters(), lr=LR)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=20)

    best_auc, patience = 0.0, 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        for X_b, y_b in tr_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            loss = crit(model(X_b), y_b)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for X_b, y_b in vl_loader:
                X_b = X_b.to(device)
                logits = model(X_b)
                preds.extend(torch.sigmoid(logits).cpu().numpy())
                targets.extend(y_b.cpu().numpy())
        val_auc = roc_auc_score(targets, np.array(preds).squeeze())

        if val_auc > best_auc:
            best_auc = val_auc; patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                break

    model.eval()
    return model


# ============================================================
# 4. LOEO 主循环
# ============================================================
causal_aucs, mlp_aucs = [], []
causal_masks = []

for held_out in range(num_envs):
    print(f"\n{'='*50}")
    print(f"Round {held_out+1}/{num_envs}: hold-out 环境 {held_out}")
    print(f"{'='*50}")

    test_mask = env_all_mapped == held_out
    train_val_mask = ~test_mask
    X_te, y_te = X_all[test_mask], y_all[test_mask]
    X_tv, y_tv, env_tv = X_all[train_val_mask], y_all[train_val_mask], env_all_mapped[train_val_mask]

    # 内部 85/15 划分
    n_total = len(X_tv)
    idx = np.random.permutation(n_total)
    n_tr = int(n_total * 0.85)
    X_tr, y_tr, env_tr = X_tv[idx[:n_tr]], y_tv[idx[:n_tr]], env_tv[idx[:n_tr]]
    X_vl, y_vl, env_vl = X_tv[idx[n_tr:]], y_tv[idx[n_tr:]], env_tv[idx[n_tr:]]

    # 环境 ID 重映射
    unique_e = sorted(set(env_tr) | set(env_vl))
    emap = {old: new for new, old in enumerate(unique_e)}
    env_tr_m = np.array([emap[e] for e in env_tr])
    env_vl_m = np.array([emap[e] for e in env_vl])

    print(f"  train={len(X_tr)}, val={len(X_vl)}, test={len(X_te)} (envs={len(unique_e)})")

    # 训练 Causal
    print("  训练 Causal Model...")
    cmodel, _ = train_causal_loeo(X_tr, y_tr, env_tr_m, X_vl, y_vl, env_vl_m, len(unique_e))
    cmodel.eval()
    with torch.no_grad():
        probs_c = torch.sigmoid(cmodel.predict(
            torch.tensor(X_te, dtype=torch.float32).to(device)
        )).cpu().numpy().squeeze()
    auc_c = roc_auc_score(y_te, probs_c)
    causal_aucs.append(auc_c)
    causal_masks.append(cmodel.gate.get_scores().cpu().numpy())
    print(f"  Causal AUC on env {held_out}: {auc_c:.4f}")

    # 训练 MLP
    print("  训练 Baseline MLP...")
    mlp_model = train_mlp_loeo(X_tr, y_tr, X_vl, y_vl)
    mlp_model.eval()
    with torch.no_grad():
        probs_m = torch.sigmoid(mlp_model(
            torch.tensor(X_te, dtype=torch.float32).to(device)
        )).cpu().numpy().squeeze()
    auc_m = roc_auc_score(y_te, probs_m)
    mlp_aucs.append(auc_m)
    print(f"  MLP AUC on env {held_out}: {auc_m:.4f}")

# ============================================================
# 5. 结果汇总
# ============================================================
causal_aucs = np.array(causal_aucs)
mlp_aucs = np.array(mlp_aucs)
delta = causal_aucs - mlp_aucs

print(f"\n{'='*60}")
print("LOEO 结果")
print(f"{'='*60}")
print(f"{'Model':<20} {'Mean±Std':<18} {'Min':<8} {'Max':<8}")
print(f"{'-'*54}")
print(f"{'Causal V4':<20} {causal_aucs.mean():.4f}±{causal_aucs.std():.4f}   {causal_aucs.min():.4f}   {causal_aucs.max():.4f}")
print(f"{'Baseline MLP':<20} {mlp_aucs.mean():.4f}±{mlp_aucs.std():.4f}   {mlp_aucs.min():.4f}   {mlp_aucs.max():.4f}")
print(f"\nΔAUC (Causal - MLP): {delta.mean():+.4f}±{delta.std():.4f}")
print(f"Causal 胜: {(delta>0).sum()}/{num_envs}")

for e in range(num_envs):
    print(f"  env {e:2d}: Causal={causal_aucs[e]:.4f}  MLP={mlp_aucs[e]:.4f}  Δ={delta[e]:+.4f}")

# 跨轮次 mask 稳定性
mask_stack = np.stack(causal_masks, axis=0)  # [12, 19]
mask_mean = mask_stack.mean(axis=0)
mask_std = mask_stack.std(axis=0)
top_k = 8
top_sets = []
for m in causal_masks:
    top_sets.append(set(np.argsort(m)[::-1][:top_k]))

pairwise_jaccard = []
for i in range(num_envs):
    for j in range(i+1, num_envs):
        inter = len(top_sets[i] & top_sets[j])
        union = len(top_sets[i] | top_sets[j])
        pairwise_jaccard.append(inter / union if union > 0 else 0)
mean_jaccard = np.mean(pairwise_jaccard)

print(f"\n跨轮次 Top-{top_k} 特征 Jaccard 均值: {mean_jaccard:.4f}")
print(f"\n特征稳定性 (跨12轮 mask 均值±std):")
sorted_idx = np.argsort(mask_mean)[::-1]
for i, idx in enumerate(sorted_idx):
    print(f"  {i+1:2d}. {feature_names[idx]:<25s} μ={mask_mean[idx]:.4f} σ={mask_std[idx]:.4f}")

# ============================================================
# 6. 可视化
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

x = range(num_envs)
ax1.plot(x, causal_aucs, 'o-', label=f'Causal V4 ({causal_aucs.mean():.3f}±{causal_aucs.std():.3f})', lw=2, ms=8)
ax1.plot(x, mlp_aucs, 's--', label=f'Baseline MLP ({mlp_aucs.mean():.3f}±{mlp_aucs.std():.3f})', lw=2, ms=8)
ax1.axhline(y=causal_aucs.mean(), color='C0', ls=':', alpha=0.5)
ax1.axhline(y=mlp_aucs.mean(), color='C1', ls=':', alpha=0.5)
ax1.set_xlabel('Held-out Environment')
ax1.set_ylabel('AUC')
ax1.set_title(f'LOEO: Causal vs MLP\nMean ΔAUC={delta.mean():+.3f}')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.barh(range(input_dim), mask_mean, xerr=mask_std, color=['#2196F3' if v > 0.5 else '#9E9E9E' for v in mask_mean])
ax2.set_yticks(range(input_dim))
ax2.set_yticklabels([feature_names[i] for i in sorted_idx])
ax2.invert_yaxis()
ax2.axvline(x=0.5, color='r', ls='--', alpha=0.5)
ax2.set_xlabel('Causal Score (mean over 12 folds)')
ax2.set_title(f'Feature Stability (Jaccard={mean_jaccard:.3f})')

plt.tight_layout()
loeo_path = os.path.join(OUTPUT_DIR, "loeo_results.png")
plt.savefig(loeo_path, dpi=150)
print(f"\n图已保存: {loeo_path}")

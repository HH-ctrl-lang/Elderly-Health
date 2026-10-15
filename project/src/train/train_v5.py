import os
import sys
import json
from networkx import expected_degree_graph
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             precision_score, recall_score, roc_curve,
                             average_precision_score)
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 0. 路径 & 配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = Path(BASE_DIR) / "data" / "preprocessed" / "preprocessed_data.npz"
OUTPUT_DIR = Path(BASE_DIR) / "data" / "preprocessed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = OUTPUT_DIR / "best_model_v5.pth"
ROC_SAVE_PATH = OUTPUT_DIR / "roc_curve_v5.png"
experiment_id = "16"
print("ExpID:", experiment_id)

# --- 训练配置 ---
NUM_EPOCHS = 300
BATCH_SIZE = 64
LR = 3e-4                     # 较低 LR，给 mask 分化留出空间
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 80      # 只在 sparse 阶段开始后计数
LATENT_DIM = 32
INDEP_MODE = "hsic"   # "cosine" or "hsic"
IRM_MODE = "v11"      # "none" | "standard" | "v11"

# --- 分阶段训练 (固定阶段长度，warmup 内不早停) ---
WARMUP_EPOCHS = 20        # Phase 1: 纯预测 (固定跑完，不早停)
SPARSE_START_EPOCH = 20   # Phase 2: +L_sparse +L_entropy
IRM_START_EPOCH = 50      # Phase 3: +L_IRM +L_env +L_indep

# --- 损失权重 ---
LAMBDA_IRM = 0.01
LAMBDA_ENV = 1.0
LAMBDA_SPARSE_BASE = 0.3
LAMBDA_INDEP = 0.02
LAMBDA_ENTROPY_BASE = 30.0
TARGET_SPARSITY = 5.0 / 19.0
GATE_INIT_STD = 0.02
GATE_TEMPERATURE = 5.0
POS_WEIGHT = 3.0

# --- 图结构先验 ---
GRAPH_DOT_PATH = Path(BASE_DIR) / "data" / "preprocessed" / "causal_graph.dot"
LAMBDA_GRAPH = 0.1               # 图先验强度
TARGET_VAR = "cognitive_impairment"

# ============================================================
# 1. 设备
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ============================================================
# 2. 加载数据
# ============================================================
data = np.load(DATA_PATH)
X_train_np = data["X_train"]
X_val_np = data["X_val"]
X_test_np = data["X_test"]
y_train_raw = data["y_train"]
y_val_raw = data["y_val"]
y_test_raw = data["y_test"]
env_train_np = data["env_train"]
env_val_np = data["env_val"]
env_test_np = data["env_test"]

y_train_np = (y_train_raw == 2.0).astype(np.float32)
y_val_np = (y_val_raw == 2.0).astype(np.float32)
y_test_np = (y_test_raw == 2.0).astype(np.float32)

input_dim = X_train_np.shape[1]
print(f"特征: {input_dim} | 训练: {len(X_train_np)} | 验证: {len(X_val_np)} | 测试: {len(X_test_np)}")
print(f"正样本率: train={y_train_np.mean():.3f}, val={y_val_np.mean():.3f}, test={y_test_np.mean():.3f}")

feature_names = [str(n) for n in data["feature_names"]]

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# 加载因果图先验（如果存在）
try:
    from graph_prior import load_graph_prior, graph_guided_loss
    GRAPH_PRIOR = load_graph_prior(str(GRAPH_DOT_PATH), TARGET_VAR, feature_names)
    USE_GRAPH_PRIOR = True
except FileNotFoundError:
    print(f"[WARN] 因果图文件未找到: {GRAPH_DOT_PATH}，跳过图先验")
    GRAPH_PRIOR = None
    USE_GRAPH_PRIOR = False
    def graph_guided_loss(mask, feature_names, prior):
        return torch.tensor(0.0, device=mask.device)
except Exception as exc:
    print(f"[WARN] 因果图先验加载失败: {exc}")
    GRAPH_PRIOR = None
    USE_GRAPH_PRIOR = False
    def graph_guided_loss(mask, feature_names, prior):
        return torch.tensor(0.0, device=mask.device)

# 环境映射
all_envs = np.concatenate([env_train_np, env_val_np])
unique_envs = sorted(set(all_envs))
env_map = {old: new for new, old in enumerate(unique_envs)}
num_envs = len(unique_envs)
print(f"环境数: {num_envs}")

# ============================================================
# 3. 模型 (同 V4)
# ============================================================

class CausalGate(nn.Module):
    def __init__(self, input_dim, init_std=0.1, temperature=5.0):
        super().__init__()
        self.raw_mask = nn.Parameter(torch.randn(input_dim) * init_std)
        self.temperature = temperature

    def forward(self):
        return torch.sigmoid(self.temperature * self.raw_mask)

    def get_scores(self):
        return torch.sigmoid(self.temperature * self.raw_mask).detach()


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, z):
        return self.net(z)


class EnvClassifier(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, num_envs=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_envs),
        )

    def forward(self, z):
        return self.net(z)


class CausalDisentangledModel(nn.Module):
    def __init__(self, input_dim, num_envs, latent_dim=LATENT_DIM, gate_init_std=0.1):
        super().__init__()
        self.gate = CausalGate(input_dim, init_std=gate_init_std, temperature=GATE_TEMPERATURE)
        self.encoder_c = Encoder(input_dim, latent_dim)
        self.encoder_s = Encoder(input_dim, latent_dim)
        self.predictor_zc = Predictor(latent_dim)
        self.predictor_joint = Predictor(latent_dim * 2)
        self.env_cls = EnvClassifier(latent_dim, num_envs)

    def forward(self, x):
        mask = self.gate()
        x_c = x * mask
        x_s = x * (1.0 - mask)
        z_c = self.encoder_c(x_c)
        z_s = self.encoder_s(x_s)

        joint_z = torch.cat([z_c, z_s], dim=1)
        logits = self.predictor_joint(joint_z)
        env_logits = self.env_cls(z_s)
        return logits, env_logits, z_c, z_s, mask

    def predict(self, x):
        mask = self.gate()

        z_c = self.encoder_c(x * mask)
        z_s = self.encoder_s(x * (1.0 - mask))
        joint_z = torch.cat([z_c, z_s], dim=1)

        return self.predictor_joint(joint_z)


# ============================================================
# 4. 损失函数
# ============================================================

def irm_penalty(model, z_c, y, env_ids, mode="standard", eps=1e-8):
    """
    可切换的 IRM 惩罚：
    - none: 关闭 IRM
    - standard: 标准 IRM，惩罚每个环境下的梯度与虚拟缩放系数的相关性
    - v11: 改进版 IRM，惩罚不同环境下分类器梯度的方差，稳定性更高
    """
    if mode == "none":
        return torch.tensor(0.0, device=z_c.device)

    unique_envs = env_ids.unique()

    if mode == "standard":
        predictor = model.predictor_zc
        total_penalty = 0.0
        n_env = 0

        for env in unique_envs:
            idx = (env_ids == env)
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
            idx = (env_ids == env)
            if idx.sum() < 2:
                continue

            logits = model.predictor_zc(z_c[idx])
            loss_e = F.binary_cross_entropy_with_logits(logits, y[idx])
            grad = torch.autograd.grad(
                loss_e,
                model.predictor_zc.net[-1].weight,
                create_graph=True
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
        return torch.exp(-dist / (2 * sigma ** 2))

    n = x.size(0)
    if n <= 1:
        return torch.tensor(0.0, device=x.device)

    K = rbf_kernel(x, x, sigma)
    L = rbf_kernel(y, y, sigma)
    H = torch.eye(n, device=x.device) - (1.0 / n)

    Kc = H @ K @ H
    Lc = H @ L @ H
    return torch.trace(Kc @ Lc) / ((n - 1) ** 2)


def independence_loss(z_c, z_s, mode="cosine"):
    """
    mode:
        - cosine
        - hsic
    """
    if mode == "cosine":
        z_c_centered = z_c - z_c.mean(dim=0, keepdim=True)
        z_s_centered = z_s - z_s.mean(dim=0, keepdim=True)
        cross = torch.mm(
            F.normalize(z_c_centered, p=2, dim=0).T,
            F.normalize(z_s_centered, p=2, dim=0)
        )
        return (cross ** 2).mean()
    elif mode == "hsic":
        return hsic_loss(z_c, z_s)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def phase_name(epoch, warmup, sparse_start, irm_start):
    if epoch < warmup:
        return "WARM"
    if epoch < irm_start:
        return "SPARSE"
    return "IRM"


def ramp_weight(epoch, start, span=40):
    if epoch < start:
        return 0.0
    return min(1.0, (epoch - start) / span)

# ============================================================
# 5. 模块化训练 (可被 LOEO 复用)
# ============================================================

def build_dataloaders(X_tr, y_tr, env_tr, X_vl, y_vl, env_vl, env_map):
    """构建 DataLoader，自动处理环境映射"""
    def map_env(arr):
        return torch.tensor([env_map[e] for e in arr], dtype=torch.long)

    tr_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1),
        map_env(env_tr),
    )
    vl_ds = TensorDataset(
        torch.tensor(X_vl, dtype=torch.float32),
        torch.tensor(y_vl, dtype=torch.float32).unsqueeze(1),
        map_env(env_vl),
    )
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    vl_loader = DataLoader(vl_ds, batch_size=BATCH_SIZE, shuffle=False)
    return tr_loader, vl_loader


def train_model(X_tr, y_tr, env_tr, X_vl, y_vl, env_vl,
                lambda_sparse=LAMBDA_SPARSE_BASE,
                lambda_entropy=LAMBDA_ENTROPY_BASE,
                gate_init_std=GATE_INIT_STD, lr=LR, pos_weight_val=POS_WEIGHT,
                verbose=True, save_path=MODEL_SAVE_PATH,
                warmup=WARMUP_EPOCHS, sparse_start=SPARSE_START_EPOCH,
                irm_start=IRM_START_EPOCH, irm_mode=IRM_MODE):
    """
    分阶段 V4 训练。
    Phase 1 [0, warmup):       纯预测 (mask 冻结 0.5)
    Phase 2 [warmup, irm_start): +sparse +entropy (mask 分化)
    Phase 3 [irm_start, end):   +IRM +env +independence (不变性)
    """
    tr_loader, vl_loader = build_dataloaders(X_tr, y_tr, env_tr, X_vl, y_vl, env_vl, env_map)
    actual_num_envs = len(set(env_map[e] for e in env_tr) | set(env_map[e] for e in env_vl))

    model = CausalDisentangledModel(input_dim, actual_num_envs, LATENT_DIM, gate_init_std).to(device)

    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion_pred = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_env = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    best_val_auc = 0.0
    mask_history = []
    patience_counter = 0
    irm_reset_done = False

    for epoch in range(NUM_EPOCHS):
        phase = phase_name(epoch, warmup, sparse_start, irm_start)
        sp_ramp = ramp_weight(epoch, sparse_start)
        irm_ramp = ramp_weight(epoch, irm_start)

        if phase == "IRM" and not irm_reset_done:
            best_val_auc = 0.0
            patience_counter = 0
            irm_reset_done = True

        model.train()
        train_loss_sum = 0.0
        train_examples = 0
        for X_b, y_b, env_b in tr_loader:
            X_b, y_b, env_b = X_b.to(device), y_b.to(device), env_b.to(device)

            logits, env_logits, z_c, z_s, mask = model(X_b)

            loss_pred = criterion_pred(logits, y_b)
            logits_zc = model.predictor_zc(z_c)
            loss_zc = criterion_pred(logits_zc, y_b)

            loss = loss_pred + loss_zc

            if sp_ramp > 0:
                loss_sparse = (mask.mean() - TARGET_SPARSITY).abs()
                loss_entropy = (mask * (1.0 - mask)).mean()
                loss = loss + sp_ramp * lambda_sparse * loss_sparse
                loss = loss + sp_ramp * lambda_entropy * loss_entropy

            if irm_ramp > 0:
                loss_irm = irm_penalty(model, z_c, y_b, env_b, mode=irm_mode)
                loss_env = criterion_env(env_logits, env_b)
                loss_indep = independence_loss(z_c, z_s, mode=INDEP_MODE)
                loss = loss + irm_ramp * LAMBDA_IRM * loss_irm
                loss = loss + irm_ramp * LAMBDA_ENV * loss_env
                loss = loss + irm_ramp * LAMBDA_INDEP * loss_indep

            if USE_GRAPH_PRIOR and sp_ramp > 0:
                loss_graph = graph_guided_loss(mask, feature_names, GRAPH_PRIOR)
                loss = loss + sp_ramp * LAMBDA_GRAPH * loss_graph

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * X_b.size(0)
            train_examples += X_b.size(0)

        model.eval()
        val_preds, val_targets, val_loss_sum = [], [], 0.0
        with torch.no_grad():
            for X_b, y_b, _ in vl_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits = model.predict(X_b)
                val_loss_sum += criterion_pred(logits, y_b).item() * X_b.size(0)
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_targets.extend(y_b.cpu().numpy())

        val_loss = val_loss_sum / len(vl_loader.dataset)
        val_auc = roc_auc_score(val_targets, np.array(val_preds).squeeze())
        scheduler.step(val_loss)

        mask_np = model.gate.get_scores().cpu().numpy()
        mask_history.append(mask_np.copy())
        n_causal = (mask_np > 0.5).sum()
        is_warmup = phase == "WARM"

        if verbose and (epoch % 30 == 0 or epoch == NUM_EPOCHS - 1):
            print(f"E{epoch:3d} [{phase}] | Train={train_loss_sum / train_examples:.3f} "
                  f"| Val AUC={val_auc:.4f} | Mask μ={mask_np.mean():.3f} σ={mask_np.std():.3f} "
                  f"n>0.5={n_causal}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), str(save_path))
        else:
            patience_counter += 1
            # warmup 期间不早停，sparse/IRM 阶段启用早停
            if not is_warmup and patience_counter >= EARLY_STOP_PATIENCE:
                if verbose:
                    print(f"早停 @ epoch {epoch+1}")
                break

    mask_history = np.array(mask_history)
    mask_stability = (mask_history.std(axis=0).mean())

    if save_path:
        model.load_state_dict(torch.load(str(save_path), weights_only=True))
    return model, best_val_auc, mask_stability


# ============================================================
# 6. 评估
# ============================================================
class EnvProbe(nn.Module):
    def __init__(self, latent_dim, num_envs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_envs),
        )

    def forward(self, z):
        return self.net(z)


def extract_representations(model, X):
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        mask = model.gate()
        z_c = model.encoder_c(X_tensor * mask)
        z_s = model.encoder_s(X_tensor * (1.0 - mask))
    return z_c, z_s


def map_env_tensor(env_labels):
    mapped = []
    for e in env_labels:
        mapped.append(env_map.get(e, -1))
    return torch.tensor(mapped, dtype=torch.long).to(device)


def train_env_probe(z_train, env_train, z_test, env_test, num_envs,
                    lr=1e-3, weight_decay=1e-5, epochs=30):
    probe = EnvProbe(z_train.size(1), num_envs).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)

    train_ds = TensorDataset(z_train, env_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    for _ in range(epochs):
        probe.train()
        for z_b, e_b in train_loader:
            logits = probe(z_b)
            loss = criterion(logits, e_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        logits = probe(z_test)
        pred = logits.argmax(dim=1)
        return (pred == env_test).float().mean().item()


def compute_env_probe_metrics(model, X_tr, env_tr, X_vl, env_vl, X_t, env_t, num_envs):
    zc_tr, zs_tr = extract_representations(model, X_tr)
    zc_vl, zs_vl = extract_representations(model, X_vl)
    zc_te, zs_te = extract_representations(model, X_t)

    env_tr_t = map_env_tensor(env_tr)
    env_vl_t = map_env_tensor(env_vl)
    env_te_t = map_env_tensor(env_t)

    valid_idx = env_te_t >= 0
    zc_te = zc_te[valid_idx]
    zs_te = zs_te[valid_idx]
    env_te_t = env_te_t[valid_idx]

    zc_train_all = torch.cat([zc_tr, zc_vl], dim=0)
    zs_train_all = torch.cat([zs_tr, zs_vl], dim=0)
    env_train_all = torch.cat([env_tr_t, env_vl_t], dim=0)

    zc_env_probe_acc = train_env_probe(zc_train_all, env_train_all, zc_te, env_te_t, num_envs)
    zs_env_probe_acc = train_env_probe(zs_train_all, env_train_all, zs_te, env_te_t, num_envs)

    return {
        "zc_env_probe_acc": zc_env_probe_acc,
        "zs_env_probe_acc": zs_env_probe_acc,
    }


def compute_zc_label_auc(model, X_t, y_t):
    model.eval()
    X_tensor = torch.tensor(X_t, dtype=torch.float32).to(device)
    with torch.no_grad():
        mask = model.gate()
        z_c = model.encoder_c(X_tensor * mask)
        logits = model.predictor_zc(z_c)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()

    return roc_auc_score(y_t, probs)


def compute_jaccard_with_graph_prior(mask, feature_names, graph_prior):
    if graph_prior is None:
        return np.nan

    selected = {str(feature_names[i]) for i, score in enumerate(mask) if score > 0.5}
    parent_set = {name for name, weight in graph_prior.items() if weight == 1.0}
    union = selected | parent_set

    if not union:
        return 0.0

    return len(selected & parent_set) / len(union)

def export_representations(model, X, y, env, split, experiment_id, seed, save_dir=OUTPUT_DIR):
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        mask = model.gate()
        z_c = model.encoder_c(X_tensor * mask).cpu().numpy()
        z_s = model.encoder_s(X_tensor * (1-mask)).cpu().numpy()
        probs = torch.sigmoid(model.predict(X_tensor)).cpu().numpy().ravel()

    rows = []
    for i in range(len(X)):
        row = {
            "sample_id": i,
            "environment_code": int(env[i]),
            "cognitive_impairment": int(y[i]),
            "split": split,
            "seed": seed,
            "y_prob": probs[i],
        }

        for j in range(z_c.shape[1]):
            row[f"zc_{j}"] = z_c[i,j]
        for j in range(z_s.shape[1]):
            row[f"zs_{j}"] = z_s[i,j]
        rows.append(row)
    df = pd.DataFrame(rows)

    save_path = os.path.join(save_dir, f"representations_exp{experiment_id}_{split}.csv")
    df.to_csv(save_path, index=False)

    print(f"[SAVE] {save_path}")
    return df

def evaluate(model, X_t, y_t, env_t):
    """测试集评估"""
    model.eval()
    X_tensor = torch.tensor(X_t, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model.predict(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().squeeze()

    y_true = y_t.squeeze()
    y_pred_labels = (probs > 0.5).astype(int)

    auc = roc_auc_score(y_true, probs)
    auprc = average_precision_score(y_true, probs)
    acc = accuracy_score(y_true, y_pred_labels)
    f1 = f1_score(y_true, y_pred_labels)
    prec = precision_score(y_true, y_pred_labels)
    rec = recall_score(y_true, y_pred_labels)

    mask = model.gate.get_scores().cpu().numpy()
    n_causal = (mask > 0.5).sum()
    jaccard = compute_jaccard_with_graph_prior(mask, feature_names, GRAPH_PRIOR)
    probe_metrics = compute_env_probe_metrics(
        model,
        X_train_np, env_train_np,
        X_val_np, env_val_np,
        X_t, env_t,
        num_envs,
    )
    zc_label_auc = compute_zc_label_auc(model, X_t, y_t)

    sorted_idx = np.argsort(mask)[::-1]
    features_ranked = [(feature_names[i], mask[i]) for i in sorted_idx]

    return {
        "auc": auc,
        "auprc": auprc,
        "acc": acc,
        "f1": f1,
        "prec": prec,
        "rec": rec,
        "jaccard": jaccard,
        "zc_env_acc": probe_metrics["zc_env_probe_acc"],
        "zs_env_acc": probe_metrics["zs_env_probe_acc"],
        "zc_label_auc": zc_label_auc,
        "mask": mask,
        "n_causal": n_causal,
        "features_ranked": features_ranked,
        "probs": probs,
        "y_true": y_true,
    }


def print_results(result, title="结果"):
    print(f"\n{'='*50}")
    print(title)
    print(f"{'='*50}")
    print(f"AUC: {result['auc']:.4f} | AUPRC: {result['auprc']:.4f} | Acc: {result['acc']:.4f} | F1: {result['f1']:.4f} "
          f"| Prec: {result['prec']:.4f} | Recall: {result['rec']:.4f} | Jaccard: {result['jaccard']:.4f} "
          f"| ZC Env Acc: {result['zc_env_acc']:.4f} | ZS Env Acc: {result['zs_env_acc']:.4f} | ZC Label AUC: {result['zc_label_auc']:.4f}")
    print(f"Mask μ={result['mask'].mean():.3f} σ={result['mask'].std():.3f} "
          f"因果特征({result['n_causal']}/{len(result['mask'])})")
    print("\n特征排序:")
    for i, (name, score) in enumerate(result["features_ranked"]):
        mark = " ★" if score > 0.5 else ""
        print(f"  {i+1:2d}. {name:<25s} {score:.4f}{mark}")


# ============================================================
# 7. 简单 MLP Baseline (用于对比)
# ============================================================

class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_baseline_mlp(X_tr, y_tr, X_vl, y_vl, lr=LR, pos_weight_val=POS_WEIGHT, verbose=True):
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
    pos_weight = torch.tensor([pos_weight_val]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    best_val_auc, patience = 0.0, 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        for X_b, y_b in tr_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            loss = criterion(model(X_b), y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for X_b, y_b in vl_loader:
                X_b = X_b.to(device)
                logits = model(X_b)
                preds.extend(torch.sigmoid(logits).cpu().numpy())
                targets.extend(y_b.cpu().numpy())
        val_preds = torch.tensor(np.vstack(preds), dtype=torch.float32)
        val_targets = torch.tensor(np.vstack(targets), dtype=torch.float32)
        val_auc = roc_auc_score(val_targets.numpy().squeeze(), val_preds.numpy().squeeze())
        val_loss = criterion(val_preds, val_targets)
        scheduler.step(val_loss)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                break

    return model, best_val_auc


# ============================================================
# 8. 主入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bayes", action="store_true")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--irm-mode", choices=["none", "standard", "v11"], default=IRM_MODE)
    args = parser.parse_args()

    if args.bayes:
        import optuna

        def objective(trial):
            ls = trial.suggest_float("lambda_sparse", 0.05, 1.0, step=0.05)
            le = trial.suggest_float("lambda_entropy", 1.0, 50.0, log=True)
            gis = trial.suggest_float("gate_init_std", 0.02, 0.2, step=0.02)
            lr_ = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
            pw = trial.suggest_float("pos_weight", 1.0, 5.0, step=0.5)
            _, auc, _ = train_model(X_train_np, y_train_np, env_train_np,
                                  X_val_np, y_val_np, env_val_np,
                                  lambda_sparse=ls, lambda_entropy=le,
                                  gate_init_std=gis, lr=lr_, pos_weight_val=pw,
                                  verbose=False, irm_mode=args.irm_mode)
            return auc

        print("\n========== Bayesian Optimization (Optuna TPE) ==========")
        study = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=42),
                                     pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=15))
        study.optimize(objective, n_trials=args.trials, timeout=args.timeout)
        print(f"\nBest Val AUC: {study.best_value:.4f}")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        # 保存最优超参
        import json
        best_config = {**study.best_params,
                       "irm_mode": args.irm_mode,
                       "target_sparsity": TARGET_SPARSITY,
                       "lambda_irm": LAMBDA_IRM, "lambda_env": LAMBDA_ENV,
                       "lambda_indep": LAMBDA_INDEP,
                       "warmup_epochs": WARMUP_EPOCHS,
                       "sparse_start_epoch": SPARSE_START_EPOCH,
                       "irm_start_epoch": IRM_START_EPOCH,
                       "num_epochs": NUM_EPOCHS, "batch_size": BATCH_SIZE,
                       "latent_dim": LATENT_DIM, "gate_temperature": GATE_TEMPERATURE}
        params_path = os.path.join(OUTPUT_DIR, "best_params.json")
        with open(params_path, "w") as f:
            json.dump(best_config, f, indent=2)
        print(f"参数已保存: {params_path}")

        # 用最优超参训练最终模型
        print("\n=== Final Training ===")
        bp = study.best_params
        model, _, _ = train_model(X_train_np, y_train_np, env_train_np,
                                X_val_np, y_val_np, env_val_np,
                                lambda_sparse=bp["lambda_sparse"],
                                lambda_entropy=bp["lambda_entropy"],
                                gate_init_std=bp["gate_init_std"],
                                lr=bp["lr"], pos_weight_val=bp["pos_weight"],
                                save_path=None,
                                verbose=True, irm_mode=args.irm_mode)
    else:
        print(f"\n=== V5 Training ===")
        print(f"Target sparsity: {TARGET_SPARSITY:.3f} (~{int(TARGET_SPARSITY*19)}/19)")
        print(f"Phases: warmup={WARMUP_EPOCHS}, sparse={SPARSE_START_EPOCH}, IRM={IRM_START_EPOCH}")
        model, val_auc, mask_stability = train_model(X_train_np, y_train_np, env_train_np,
                                      X_val_np, y_val_np, env_val_np, save_path=MODEL_SAVE_PATH, verbose=True,
                                      irm_mode=args.irm_mode)
        print(f"\nBest Val AUC: {val_auc:.4f}")
        print(f"Mask Stability: {mask_stability:.4f}")

    # 测试评估
    result = evaluate(model, X_test_np, y_test_np, env_test_np)
    print_results(result, "Test Results")
    export_representations(model, X_train_np, y_train_np, env_train_np, split="train", experiment_id=experiment_id, seed=42)
    export_representations(model, X_val_np, y_val_np, env_val_np, split="val", experiment_id=experiment_id, seed=42)
    export_representations(model, X_test_np, y_test_np, env_test_np, split="test", experiment_id=experiment_id, seed=42)

    # ROC
    fpr, tpr, _ = roc_curve(y_test_np, result["probs"])
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"V5 Causal Model (AUC={result['auc']:.4f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curve — V5 Causal Disentanglement")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(ROC_SAVE_PATH, dpi=150)
    print(f"\nROC saved: {ROC_SAVE_PATH}")
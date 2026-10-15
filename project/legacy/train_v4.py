"""
V4 因果解耦训练 — 19 特征版本
分阶段训练: 先学会预测 → 再解耦特征 → 最后不变性约束

"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             precision_score, recall_score, roc_curve,
                             average_precision_score)
import matplotlib.pyplot as plt

# ============================================================
# 0. 路径 & 配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "preprocessed_data.npz")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "preprocessed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model_v4.pth")
ROC_SAVE_PATH = os.path.join(OUTPUT_DIR, "roc_curve_v4.png")

# --- 训练配置 ---
NUM_EPOCHS = 300
BATCH_SIZE = 64
LR = 3e-4                     # 较低 LR，给 mask 分化留出空间
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 80      # 只在 sparse 阶段开始后计数
LATENT_DIM = 32

# --- 分阶段训练 (固定阶段长度，warmup 内不早停) ---
WARMUP_EPOCHS = 20        # Phase 1: 纯预测 (固定跑完，不早停)
SPARSE_START_EPOCH = 20   # Phase 2: +L_sparse +L_entropy
IRM_START_EPOCH = 50      # Phase 3: +L_IRM +L_env +L_cosine

# --- 损失权重 ---
LAMBDA_IRM = 0.01
LAMBDA_ENV = 1.0
LAMBDA_SPARSE_BASE = 0.3
LAMBDA_COSINE = 0.02
LAMBDA_ENTROPY_BASE = 30.0
TARGET_SPARSITY = 5.0 / 19.0
GATE_INIT_STD = 0.02
GATE_TEMPERATURE = 5.0
POS_WEIGHT = 3.0

# --- 图结构先验 ---
GRAPH_DOT_PATH = os.path.join(OUTPUT_DIR, "causal_graph.dot")
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

feature_names = data["feature_names"]

# 加载因果图先验（如果存在）
try:
    from graph_prior import load_graph_prior, graph_guided_loss
    GRAPH_PRIOR = load_graph_prior(GRAPH_DOT_PATH, TARGET_VAR, list(feature_names))
    USE_GRAPH_PRIOR = True
except FileNotFoundError:
    print(f"[WARN] 因果图文件未找到: {GRAPH_DOT_PATH}，跳过图先验")
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
        self.predictor = Predictor(latent_dim)
        self.env_cls = EnvClassifier(latent_dim, num_envs)

    def forward(self, x):
        mask = self.gate()
        x_c = x * mask
        x_s = x * (1.0 - mask)
        z_c = self.encoder_c(x_c)
        z_s = self.encoder_s(x_s)
        logits = self.predictor(z_c)
        env_logits = self.env_cls(z_s)
        return logits, env_logits, z_c, z_s, mask

    def predict(self, x):
        mask = self.gate()
        return self.predictor(self.encoder_c(x * mask))


# ============================================================
# 4. 损失函数
# ============================================================

def irm_penalty(z_c, y, env_ids):
    unique_e = env_ids.unique()
    if len(unique_e) < 2:
        return torch.tensor(0.0, device=z_c.device)
    total, n = 0.0, 0
    for env in unique_e:
        idx = (env_ids == env)
        z_env, y_env = z_c[idx], y[idx]
        if len(y_env) < 2:
            continue
        scale = torch.ones(1, 1, device=z_env.device, requires_grad=True)
        logits = scale * z_env.sum(dim=1, keepdim=True)
        loss = F.binary_cross_entropy_with_logits(logits, y_env)
        grad = torch.autograd.grad(loss, scale, create_graph=True)[0]
        total += (grad ** 2).mean()
        n += 1
    return torch.tensor(0.0, device=z_c.device) if n == 0 else total / n


def cosine_independence_loss(z_c, z_s):
    z_c_c = z_c - z_c.mean(dim=0, keepdim=True)
    z_s_c = z_s - z_s.mean(dim=0, keepdim=True)
    cross = torch.mm(F.normalize(z_c_c, p=2, dim=0).T,
                     F.normalize(z_s_c, p=2, dim=0))
    return (cross ** 2).mean()


# ============================================================
# 5. 模块化训练 (可被 LOEO 复用)
# ============================================================

def build_dataloaders(X_tr, y_tr, env_tr, X_vl, y_vl, env_vl):
    """构建 DataLoader，自动处理环境映射"""
    def map_env(arr):
        return torch.tensor([env_map.get(e, e) for e in arr], dtype=torch.long)

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
                irm_start=IRM_START_EPOCH):
    """
    分阶段 V4 训练。
    Phase 1 [0, warmup):       纯预测 (mask 冻结 0.5)
    Phase 2 [warmup, irm_start): +sparse +entropy (mask 分化)
    Phase 3 [irm_start, end):   +IRM +env +cosine (不变性)
    """
    tr_loader, vl_loader = build_dataloaders(X_tr, y_tr, env_tr, X_vl, y_vl, env_vl)
    actual_num_envs = len(set(env_tr) | set(env_vl))

    model = CausalDisentangledModel(input_dim, actual_num_envs, LATENT_DIM, gate_init_std).to(device)

    pos_weight = torch.tensor([pos_weight_val]).to(device)
    criterion_pred = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_env = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)

    best_val_auc = 0.0
    patience_counter = 0
    irm_reset_done = False

    for epoch in range(NUM_EPOCHS):
        # --- 判断当前阶段 ---
        is_warmup = epoch < warmup
        is_sparse = epoch >= sparse_start
        is_irm = epoch >= irm_start

        # IRM 阶段首次入场时，重置 best_val_auc
        # 让模型在完整约束下重新寻找最优，而非永远守着 warmup 时期的纯预测 AUC
        if is_irm and not irm_reset_done:
            best_val_auc = 0.0
            patience_counter = 0
            irm_reset_done = True

        # 计算当前阶段权重 (gradual ramp-up)
        if is_sparse:
            sp_ramp = min(1.0, (epoch - sparse_start) / 40.0)
        else:
            sp_ramp = 0.0
        if is_irm:
            irm_ramp = min(1.0, (epoch - irm_start) / 40.0)
        else:
            irm_ramp = 0.0

        # --- train ---
        model.train()
        for X_b, y_b, env_b in tr_loader:
            X_b, y_b, env_b = X_b.to(device), y_b.to(device), env_b.to(device)

            logits, env_logits, z_c, z_s, mask = model(X_b)

            loss_pred = criterion_pred(logits, y_b)
            loss = loss_pred.clone()

            if is_sparse and sp_ramp > 0:
                loss_sparse = torch.abs(mask.mean() - TARGET_SPARSITY)
                loss_entropy = (mask * (1.0 - mask)).mean()
                loss = loss + sp_ramp * lambda_sparse * loss_sparse
                loss = loss + sp_ramp * lambda_entropy * loss_entropy

            if is_irm and irm_ramp > 0:
                loss_irm = irm_penalty(z_c, y_b, env_b)
                loss_env = criterion_env(env_logits, env_b)
                loss_cosine = cosine_independence_loss(z_c, z_s)
                loss = loss + irm_ramp * LAMBDA_IRM * loss_irm
                loss = loss + irm_ramp * LAMBDA_ENV * loss_env
                loss = loss + irm_ramp * LAMBDA_COSINE * loss_cosine

            # 图结构先验：从 Phase 2 (sparse) 开始施加
            if USE_GRAPH_PRIOR and is_sparse and sp_ramp > 0:
                loss_graph = graph_guided_loss(mask, feature_names, GRAPH_PRIOR)
                loss = loss + sp_ramp * LAMBDA_GRAPH * loss_graph

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- val ---
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
        n_causal = (mask_np > 0.5).sum()

        if verbose and (epoch % 30 == 0 or epoch == NUM_EPOCHS - 1):
            phase = "WARM" if is_warmup else ("SPARSE" if not is_irm else "IRM")
            print(f"E{epoch:3d} [{phase}] | Pred={loss_pred.item():.3f} "
                  f"| Val AUC={val_auc:.4f} | Mask μ={mask_np.mean():.3f} σ={mask_np.std():.3f} "
                  f"n>0.5={n_causal}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            # warmup 期间不早停，sparse/IRM 阶段启用早停
            if not is_warmup and patience_counter >= EARLY_STOP_PATIENCE:
                if verbose:
                    print(f"早停 @ epoch {epoch+1}")
                break

    if save_path:
        model.load_state_dict(torch.load(save_path, weights_only=True))
    return model, best_val_auc


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
    return torch.tensor([env_map[e] for e in env_labels], dtype=torch.long).to(device)



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
        logits = model.predictor(z_c)
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
        val_auc = roc_auc_score(targets, np.array(preds).squeeze())
        # 计算验证损失用于 scheduler
        val_loss = criterion(torch.tensor(preds), torch.tensor(targets))
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
    args = parser.parse_args()

    if args.bayes:
        import optuna

        def objective(trial):
            ls = trial.suggest_float("lambda_sparse", 0.05, 1.0, step=0.05)
            le = trial.suggest_float("lambda_entropy", 1.0, 50.0, log=True)
            gis = trial.suggest_float("gate_init_std", 0.02, 0.2, step=0.02)
            lr_ = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
            pw = trial.suggest_float("pos_weight", 1.0, 5.0, step=0.5)
            _, auc = train_model(X_train_np, y_train_np, env_train_np,
                                  X_val_np, y_val_np, env_val_np,
                                  lambda_sparse=ls, lambda_entropy=le,
                                  gate_init_std=gis, lr=lr_, pos_weight_val=pw,
                                  verbose=False)
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
                       "target_sparsity": TARGET_SPARSITY,
                       "lambda_irm": LAMBDA_IRM, "lambda_env": LAMBDA_ENV,
                       "lambda_cosine": LAMBDA_COSINE,
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
        model, _ = train_model(X_train_np, y_train_np, env_train_np,
                                X_val_np, y_val_np, env_val_np,
                                lambda_sparse=bp["lambda_sparse"],
                                lambda_entropy=bp["lambda_entropy"],
                                gate_init_std=bp["gate_init_std"],
                                lr=bp["lr"], pos_weight_val=bp["pos_weight"],
                                verbose=True)
    else:
        print(f"\n=== V4 Training ===")
        print(f"Target sparsity: {TARGET_SPARSITY:.3f} (~{int(TARGET_SPARSITY*19)}/19)")
        print(f"Phases: warmup={WARMUP_EPOCHS}, sparse={SPARSE_START_EPOCH}, IRM={IRM_START_EPOCH}")
        model, val_auc = train_model(X_train_np, y_train_np, env_train_np,
                                      X_val_np, y_val_np, env_val_np, verbose=True)
        print(f"\nBest Val AUC: {val_auc:.4f}")

    # 测试评估 (与 IRM 版本完全一致)
    result = evaluate(model, X_test_np, y_test_np, env_test_np)
    print_results(result, "Test Results")

    # ROC 曲线
    fpr, tpr, _ = roc_curve(result["y_true"], result["probs"])
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"V4 Causal Model (AUC={result['auc']:.4f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve — V4 Causal Disentanglement")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_SAVE_PATH, dpi=150)
    print(f"\nROC saved: {ROC_SAVE_PATH}")
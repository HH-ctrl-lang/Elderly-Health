import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             precision_score, recall_score, roc_curve)
import matplotlib.pyplot as plt

# ============================================================
# 0. 配置
# ============================================================
TARGET_NAME = "1=无，2=有"
DATA_PATH = f"pre_data/{TARGET_NAME}/{TARGET_NAME}_data.npz"
MODEL_SAVE_PATH = "best_model_causal_device.pth"
ROC_SAVE_PATH = "roc_curve_causal_device.png"
MASK_PLOT_PATH = "causal_mask.png"

# 损失权重
LAMBDA_IRM = 0.01
LAMBDA_ENV = 2.0
LAMBDA_SPARSE = 1.0
LAMBDA_COSINE = 0.05
LAMBDA_ENTROPY = 70.0

NUM_EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 20
LATENT_DIM = 64
TARGET_SPARSITY = 50/151

# ============================================================
# 1. 设备初始化
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
print(f"标签：{TARGET_NAME}")

# ============================================================
# 2. 加载数据（含环境标签）
# ============================================================
data = np.load(DATA_PATH)

X_train = torch.tensor(data['X_train'], dtype=torch.float32)
X_val = torch.tensor(data['X_val'], dtype=torch.float32)
X_test = torch.tensor(data['X_test'], dtype=torch.float32)

y_train_raw = data['y_train']
y_val_raw = data['y_val']
y_test_raw = data['y_test']

y_train = torch.tensor((y_train_raw == 2).astype(np.float32)).unsqueeze(1)
y_val = torch.tensor((y_val_raw == 2).astype(np.float32)).unsqueeze(1)
y_test = torch.tensor((y_test_raw == 2).astype(np.float32)).unsqueeze(1)

if 'env_train' in data:
    env_train = torch.tensor(data['env_train'], dtype=torch.long)
    env_val = torch.tensor(data['env_val'], dtype=torch.long)
    env_test = torch.tensor(data['env_test'], dtype=torch.long)
    num_envs = int(max(env_train.max().item(), env_val.max().item(), env_test.max().item())) + 1
    print(f"环境标签已加载，共 {num_envs} 个环境")
else:
    raise RuntimeError("数据中未找到环境标签")

input_dim = X_train.shape[1]
print(f"输入特征维度: {input_dim}")
print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

# ============================================================
# 3. 模型定义
# ============================================================

class CausalGate(nn.Module):
    """实例无关的因果门控：学习全局特征掩码，直接可解释"""
    def __init__(self, input_dim):
        super().__init__()
        self.raw_mask = nn.Parameter(torch.randn(input_dim) * 0.1)

    def forward(self):
        return torch.sigmoid(self.raw_mask)

    def get_causal_scores(self):
        return torch.sigmoid(self.raw_mask).detach()


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, z):
        return self.net(z)


class EnvClassifier(nn.Module):
    """环境分类器：从混杂特征预测环境 ID"""
    def __init__(self, latent_dim=64, num_envs=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_envs)
        )

    def forward(self, z):
        return self.net(z)


class CausalDisentangledModel(nn.Module):
    """因果解耦模型"""
    def __init__(self, input_dim, latent_dim=64, num_envs=12):
        super().__init__()
        self.gate = CausalGate(input_dim)
        self.encoder_c = Encoder(input_dim, latent_dim)
        self.encoder_s = Encoder(input_dim, latent_dim)
        self.predictor_c = Predictor(latent_dim)
        self.env_classifier = EnvClassifier(latent_dim, num_envs)

    def forward(self, x):
        mask = self.gate()
        x_c = x * mask
        x_s = x * (1 - mask)
        z_c = self.encoder_c(x_c)
        z_s = self.encoder_s(x_s)
        logits = self.predictor_c(z_c)
        env_logits = self.env_classifier(z_s)
        return logits, env_logits, z_c, z_s, mask

    def forward_inference(self, x):
        """推理模式：仅用因果分支"""
        mask = self.gate()
        x_c = x * mask
        z_c = self.encoder_c(x_c)
        logits = self.predictor_c(z_c)
        return logits


def irm_penalty(z_c, y, env_ids):
    """IRMv1 惩罚：跨环境梯度范数，强制 z_c 对环境不变。"""
    unique_envs = env_ids.unique()
    if len(unique_envs) < 2:
        return torch.tensor(0.0, device=z_c.device)

    total_penalty = 0.0
    n_valid = 0
    for env in unique_envs:
        mask = (env_ids == env)
        z_env = z_c[mask]
        y_env = y[mask]
        if len(y_env) < 2:
            continue
        scale = torch.ones(1, 1, device=z_env.device, requires_grad=True)
        logits = scale * z_env.sum(dim=1, keepdim=True)
        loss = F.binary_cross_entropy_with_logits(logits, y_env)
        grad = torch.autograd.grad(loss, scale, create_graph=True)[0]
        total_penalty += (grad ** 2).mean()
        n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=z_c.device)
    return total_penalty / n_valid


def cosine_independence_loss(z_c, z_s):
    """余弦独立性损失：最小化 z_c 与 z_s 的列空间相关性"""
    z_c_c = z_c - z_c.mean(dim=0, keepdim=True)
    z_s_c = z_s - z_s.mean(dim=0, keepdim=True)
    z_c_n = F.normalize(z_c_c, p=2, dim=0)
    z_s_n = F.normalize(z_s_c, p=2, dim=0)
    cross = torch.mm(z_c_n.T, z_s_n)
    return (cross ** 2).mean()


# ============================================================
# 4. 实例化模型
# ============================================================
model = CausalDisentangledModel(input_dim, LATENT_DIM, num_envs)
model.to(device)

criterion_pred = nn.BCEWithLogitsLoss()
criterion_env = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# ============================================================
# 5. 创建 DataLoader（含环境标签）
# ============================================================
train_dataset = TensorDataset(X_train, y_train, env_train)
val_dataset = TensorDataset(X_val, y_val, env_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# 6. 训练循环
# ============================================================
best_val_auc = 0.0
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    # --- 训练阶段 ---
    model.train()
    total_loss_sum = 0.0
    pred_loss_sum = 0.0
    irm_loss_sum = 0.0
    env_loss_sum = 0.0
    sparse_loss_sum = 0.0
    cosine_loss_sum = 0.0
    entropy_loss_sum = 0.0

    for X_batch, y_batch, env_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        env_batch = env_batch.to(device)

        logits, env_logits, z_c, z_s, mask = model(X_batch)

        # L_pred: 预测损失
        loss_pred = criterion_pred(logits, y_batch)

        # L_irm: 不变性惩罚
        loss_irm = irm_penalty(z_c, y_batch, env_batch)

        # L_env: 环境分类损失（仅从混杂特征）
        loss_env = criterion_env(env_logits, env_batch)

        # L_sparse: 因果门控稀疏性（保持总体选取率接近目标值）
        loss_sparse = torch.abs(mask.mean() - TARGET_SPARSITY)

        # L_cosine: 因果-混杂独立性
        loss_cosine = cosine_independence_loss(z_c, z_s)

        # L_entropy: 因果门控熵正则化
        loss_entropy = (mask * (1 - mask)).mean()

        # 总损失
        total_loss = (loss_pred +
                      LAMBDA_IRM * loss_irm +
                      LAMBDA_ENV * loss_env +
                      LAMBDA_SPARSE * loss_sparse +
                      LAMBDA_COSINE * loss_cosine +
                      LAMBDA_ENTROPY * loss_entropy)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        n = X_batch.size(0)
        total_loss_sum += total_loss.item() * n
        pred_loss_sum += loss_pred.item() * n
        irm_loss_sum += loss_irm.item() * n
        env_loss_sum += loss_env.item() * n
        sparse_loss_sum += loss_sparse.item() * n
        cosine_loss_sum += loss_cosine.item() * n
        entropy_loss_sum += loss_entropy.item() * n

    n_train = len(train_loader.dataset)
    avg_total = total_loss_sum / n_train
    avg_pred = pred_loss_sum / n_train
    avg_irm = irm_loss_sum / n_train
    avg_env = env_loss_sum / n_train
    avg_sparse = sparse_loss_sum / n_train
    avg_cosine = cosine_loss_sum / n_train
    avg_entropy = entropy_loss_sum / n_train

    # --- 验证阶段 ---
    model.eval()
    val_preds = []
    val_targets = []
    val_loss_sum = 0.0
    with torch.no_grad():
        for X_batch, y_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model.forward_inference(X_batch)
            val_loss_sum += criterion_pred(logits, y_batch).item() * X_batch.size(0)
            val_preds.extend(torch.sigmoid(logits).cpu().numpy())
            val_targets.extend(y_batch.cpu().numpy())

    val_loss = val_loss_sum / len(val_loader.dataset)
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    val_auc = roc_auc_score(val_targets, val_preds)

    scheduler.step(val_loss)

    print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
          f"Total: {avg_total:.4f} | Pred: {avg_pred:.4f} | "
          f"IRM: {avg_irm:.4f} | Env: {avg_env:.4f} | "
          f"Sparse: {avg_sparse:.4f} | Cosine: {avg_cosine:.4f} | "
          f"Entropy: {avg_entropy:.4f} | "
          f"Val AUC: {val_auc:.4f}")
    mask_values = model.gate.get_causal_scores().cpu().numpy()
    print(f"Mask mean: {mask_values.mean():.3f}, max: {mask_values.max():.3f}, min: {mask_values.min():.3f}")

    # 早停与模型保存
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  --> 保存最佳模型, Val AUC: {val_auc:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"早停触发，训练结束于 epoch {epoch+1}")
            break

# ============================================================
# 7. 测试评估
# ============================================================
model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
model.to(device)
model.eval()

test_dataset = TensorDataset(X_test, y_test, env_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_preds = []
test_targets = []
all_env_logits = []
all_env_true = []

with torch.no_grad():
    for X_batch, y_batch, env_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        env_batch = env_batch.to(device)

        logits, env_logits, z_c, z_s, _ = model(X_batch)

        test_preds.extend(torch.sigmoid(logits).cpu().numpy())
        test_targets.extend(y_batch.cpu().numpy())
        all_env_logits.append(env_logits.cpu())
        all_env_true.append(env_batch.cpu())

# ---- 预测指标 ----
test_preds = np.array(test_preds)
test_targets = np.array(test_targets)
test_pred_labels = (test_preds > 0.5).astype(int)

auc = roc_auc_score(test_targets, test_preds)
accuracy = accuracy_score(test_targets, test_pred_labels)
f1 = f1_score(test_targets, test_pred_labels)
precision = precision_score(test_targets, test_pred_labels)
recall = recall_score(test_targets, test_pred_labels)

print("\n" + "=" * 50)
print("测试集评估结果 (Causal Disentangled Model)")
print("=" * 50)
print(f"AUC:       {auc:.4f}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

# ---- 环境分类准确率 ----
env_logits = torch.cat(all_env_logits, dim=0)
env_true   = torch.cat(all_env_true, dim=0)
env_acc    = (env_logits.argmax(dim=1) == env_true).float().mean().item()

# ---- 因果门控掩码 ----
mask = model.gate.get_causal_scores().cpu().numpy()
mask_mean = mask.mean()
mask_std  = mask.std()

print(f"\nEnvClassifier 准确率: {env_acc:.4f}")
print(f"Mask 均值: {mask_mean:.4f}")
print(f"Mask 标准差: {mask_std:.4f}")
print(f"\n总特征数: {len(mask)}")
print(f"因果特征 (score > 0.5): {(mask > 0.5).sum()}")
print(f"混杂特征 (score <= 0.5): {(mask <= 0.5).sum()}")
print("=" * 50)

# ============================================================
# 8. ROC 曲线
# ============================================================
fpr, tpr, _ = roc_curve(test_targets, test_preds)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'Causal Model (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve - Causal Disentanglement')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(ROC_SAVE_PATH, dpi=150)
print(f"ROC 曲线已保存至 {ROC_SAVE_PATH}")

# ============================================================
# 9. 因果特征可解释性可视化
# ============================================================
feature_names = data['feature_names']
causal_scores = model.gate.get_causal_scores().cpu().numpy()

sorted_idx = np.argsort(causal_scores)[::-1]
top_k = 30

# plt.figure(figsize=(10, 8))
# colors = ['#2196F3' if s > 0.5 else '#9E9E9E' for s in causal_scores[sorted_idx[:top_k]]]
# plt.barh(range(top_k), causal_scores[sorted_idx[:top_k]][::-1], color=colors[::-1])
# plt.yticks(range(top_k), [feature_names[i] for i in sorted_idx[:top_k]][::-1],
#            fontsize=9)
# plt.xlabel('Causal Importance Score', fontsize=12)
# plt.title(f'Top {top_k} Causal Features ({TARGET_NAME})', fontsize=14)
# plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='threshold=0.5')
# plt.legend()
# plt.tight_layout()
# plt.savefig(MASK_PLOT_PATH, dpi=150)
# print(f"因果特征图已保存至 {MASK_PLOT_PATH}")

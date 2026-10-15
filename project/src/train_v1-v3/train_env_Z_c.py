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
MODEL_SAVE_PATH = "best_model_zc_only.pth"
ROC_SAVE_PATH = "roc_curve_zc_only.png"

NUM_EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 20
LATENT_DIM = 64

# ============================================================
# 1. 设备初始化
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
print(f"标签：{TARGET_NAME}")

# ============================================================
# 2. 加载数据
# ============================================================
data = np.load(DATA_PATH)

X_train = torch.tensor(data['X_train'], dtype=torch.float32)
X_val   = torch.tensor(data['X_val'], dtype=torch.float32)
X_test  = torch.tensor(data['X_test'], dtype=torch.float32)

y_train_raw = data['y_train']
y_val_raw   = data['y_val']
y_test_raw  = data['y_test']

y_train = torch.tensor((y_train_raw == 2).astype(np.float32)).unsqueeze(1)
y_val   = torch.tensor((y_val_raw == 2).astype(np.float32)).unsqueeze(1)
y_test  = torch.tensor((y_test_raw == 2).astype(np.float32)).unsqueeze(1)

input_dim = X_train.shape[1]
print(f"输入特征维度: {input_dim}")
print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

# ============================================================
# 3. 模型定义（仅保留因果分支）
# ============================================================

class CausalGate(nn.Module):
    """实例无关的因果门控"""
    def __init__(self, input_dim):
        super().__init__()
        self.raw_mask = nn.Parameter(torch.full((input_dim,), 0.0))

    def forward(self):
        return torch.sigmoid(self.raw_mask)

    def get_causal_scores(self):
        return torch.sigmoid(self.raw_mask).detach()


class Encoder(nn.Module):
    """特征编码器"""
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
    """预测头"""
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


class ZCOnlyModel(nn.Module):
    """仅使用因果表示 zc 的预测模型"""
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.gate = CausalGate(input_dim)
        self.encoder_c = Encoder(input_dim, latent_dim)
        self.predictor_c = Predictor(latent_dim)

    def forward(self, x):
        mask = self.gate()
        x_c = x * mask
        z_c = self.encoder_c(x_c)
        logits = self.predictor_c(z_c)
        return logits

# ============================================================
# 4. 实例化模型
# ============================================================
model = ZCOnlyModel(input_dim, LATENT_DIM)
model.to(device)

criterion_pred = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# ============================================================
# 5. 创建 DataLoader（仅特征和标签）
# ============================================================
train_dataset = TensorDataset(X_train, y_train)
val_dataset   = TensorDataset(X_val, y_val)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader    = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# 6. 训练循环（只有预测损失）
# ============================================================
best_val_auc = 0.0
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    # --- 训练阶段 ---
    model.train()
    total_loss_sum = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion_pred(logits, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_sum += loss.item() * X_batch.size(0)

    avg_loss = total_loss_sum / len(train_loader.dataset)

    # --- 验证阶段 ---
    model.eval()
    val_preds, val_targets = [], []
    val_loss_sum = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            val_loss_sum += criterion_pred(logits, y_batch).item() * X_batch.size(0)
            val_preds.extend(torch.sigmoid(logits).cpu().numpy())
            val_targets.extend(y_batch.cpu().numpy())

    val_loss = val_loss_sum / len(val_loader.dataset)
    val_preds = np.array(val_preds).squeeze()
    val_targets = np.array(val_targets).squeeze()
    val_auc = roc_auc_score(val_targets, val_preds)

    scheduler.step(val_loss)

    mask = model.gate.get_causal_scores().cpu().numpy()
    print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
          f"Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f} | "
          f"Mask mean: {mask.mean():.3f}, max: {mask.max():.3f}, min: {mask.min():.3f}")

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

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_preds, test_targets = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        logits = model(X_batch)
        test_preds.extend(torch.sigmoid(logits).cpu().numpy())
        test_targets.extend(y_batch.cpu().numpy())

test_preds = np.array(test_preds).squeeze()
test_targets = np.array(test_targets).squeeze()
test_pred_labels = (test_preds > 0.5).astype(int)

auc = roc_auc_score(test_targets, test_preds)
accuracy = accuracy_score(test_targets, test_pred_labels)
f1 = f1_score(test_targets, test_pred_labels)
precision = precision_score(test_targets, test_pred_labels)
recall = recall_score(test_targets, test_pred_labels)

mask = model.gate.get_causal_scores().cpu().numpy()
print("\n" + "=" * 50)
print("测试集评估结果 (Only ZC Model)")
print("=" * 50)
print(f"AUC:       {auc:.4f}")
print(f"Accuracy:  {accuracy:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"\nMask 均值: {mask.mean():.4f}  标准差: {mask.std():.4f}")
print(f"因果特征 (score > 0.5): {(mask > 0.5).sum()} / {len(mask)}")
print("=" * 50)

# ============================================================
# 8. ROC 曲线
# ============================================================
fpr, tpr, _ = roc_curve(test_targets, test_preds)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'Only ZC Model (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Only ZC Model')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(ROC_SAVE_PATH, dpi=150)
print(f"ROC 曲线已保存至 {ROC_SAVE_PATH}")
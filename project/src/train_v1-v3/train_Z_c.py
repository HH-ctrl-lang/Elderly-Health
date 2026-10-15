import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             precision_score, recall_score, roc_curve)
import matplotlib.pyplot as plt
import torch_directml

# --- 1. 设备初始化 ---
dml = torch_directml.device()
print(f"使用设备: {dml}")

# --- 2. 加载预处理好的数据 ---
label="1=无，2=有"
DATA_PATH = f"pre_data/{label}/{label}_data.npz"
data = np.load(DATA_PATH)

X_train = torch.tensor(data['X_train'], dtype=torch.float32)
y_train = torch.tensor(data['y_train'], dtype=torch.float32).unsqueeze(1)
X_val = torch.tensor(data['X_val'], dtype=torch.float32)
y_val = torch.tensor(data['y_val'], dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(data['X_test'], dtype=torch.float32)
y_test = torch.tensor(data['y_test'], dtype=torch.float32).unsqueeze(1)

# 将标签转换为二分类格式（0 和 1）
y_train = torch.tensor((data['y_train'] == 2).astype(np.float32)).unsqueeze(1)
y_val = torch.tensor((data['y_val'] == 2).astype(np.float32)).unsqueeze(1)
y_test = torch.tensor((data['y_test'] == 2).astype(np.float32)).unsqueeze(1)

# 创建 DataLoader
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

input_dim = X_train.shape[1]
print(f"输入特征维度: {input_dim}")

# --- 3. 因果解耦模型定义 ---
class CausalEncoder(nn.Module):
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

class CausalDisentanglePredictor(nn.Module):
    def __init__(self, input_dim, latent_dim=64, causal_dim=32, spurious_dim=32):
        super().__init__()
        self.encoder = CausalEncoder(input_dim, latent_dim)
        # 因果特征提取器
        self.causal_proj = nn.Sequential(
            nn.Linear(latent_dim, causal_dim),
            nn.BatchNorm1d(causal_dim),
            nn.ReLU()
        )
        # 非因果特征提取器
        self.spurious_proj = nn.Sequential(
            nn.Linear(latent_dim, spurious_dim),
            nn.BatchNorm1d(spurious_dim),
            nn.ReLU()
        )
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(causal_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        z = self.encoder(x)                 # 共享潜在特征
        z_c = self.causal_proj(z)           # 因果特征
        z_s = self.spurious_proj(z)         # 非因果特征
        y_pred = self.classifier(z_c)       # 仅用因果特征预测
        return y_pred, z_c, z_s

# 实例化模型
model = CausalDisentanglePredictor(input_dim, latent_dim=64, causal_dim=32, spurious_dim=32)
model.to(dml)

# --- 4. 损失函数定义 ---
criterion_cls = nn.BCELoss()


def orthogonality_loss(z_c, z_s):  #正交性损失
    z_c_norm = nn.functional.normalize(z_c, dim=1)
    z_s_norm = nn.functional.normalize(z_s, dim=1)
    corr = torch.mm(z_c_norm.T, z_s_norm)   # shape: [causal_dim, spurious_dim]
    loss_orth = torch.sum(corr ** 2)
    return loss_orth

def independence_loss(z_s, y):  #独立性损失
    y_onehot = torch.cat([1 - y, y], dim=1)   # [batch, 2]
    z_s_centered = z_s - z_s.mean(dim=0, keepdim=True)
    y_centered = y_onehot - y_onehot.mean(dim=0, keepdim=True)
    cross_cov = torch.mm(z_s_centered.T, y_centered) / (z_s.size(0) - 1)   # [dim_s, 2]
    loss_indep = torch.sum(cross_cov ** 2)
    return loss_indep

# --- 5. 优化器 ---
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

#损失权重
lambda_orth = 1e-4
lambda_indep = 1e-4

# --- 6. 训练循环 ---
num_epochs = 100
best_val_auc = 0.0
patience_counter = 0
early_stop_patience = 15

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss_cls = 0.0  #分类损失
    train_loss_orth = 0.0  #正交性损失
    train_loss_indep = 0.0  #独立性损失
    train_loss_total = 0.0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(dml), y_batch.to(dml)
        optimizer.zero_grad()
        
        y_pred, z_c, z_s = model(X_batch)
        
        loss_cls = criterion_cls(y_pred, y_batch)
        loss_orth = orthogonality_loss(z_c, z_s)
        loss_indep = independence_loss(z_s, y_batch)
        
        loss_total = loss_cls + lambda_orth * loss_orth + lambda_indep * loss_indep
        
        loss_total.backward()
        optimizer.step()
        
        train_loss_cls += loss_cls.item() * X_batch.size(0)
        train_loss_orth += loss_orth.item() * X_batch.size(0)
        train_loss_indep += loss_indep.item() * X_batch.size(0)
        train_loss_total += loss_total.item() * X_batch.size(0)
    
    train_loss_cls /= len(train_loader.dataset)
    train_loss_orth /= len(train_loader.dataset)
    train_loss_indep /= len(train_loader.dataset)
    train_loss_total /= len(train_loader.dataset)
    
    # 验证阶段
    model.eval()
    val_loss_cls = 0.0
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(dml), y_batch.to(dml)
            y_pred, z_c, z_s = model(X_batch)
            loss_cls = criterion_cls(y_pred, y_batch)
            val_loss_cls += loss_cls.item() * X_batch.size(0)
            val_preds.extend(y_pred.cpu().numpy())
            val_targets.extend(y_batch.cpu().numpy())
    
    val_loss_cls /= len(val_loader.dataset)
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    val_auc = roc_auc_score(val_targets, val_preds)
    
    scheduler.step(val_loss_cls)
    print(f"Epoch {epoch+1:3d}/{num_epochs} | Cls Loss: {train_loss_cls:.4f} | Orth: {train_loss_orth:.4f} | Indep: {train_loss_indep:.4f} | Val AUC: {val_auc:.4f}")
    
    # 早停与模型保存
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0
        torch.save(model.state_dict(), "best_model_causal_dml.pth")
        print(f"  --> 保存最佳模型, Val AUC: {val_auc:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"早停触发，训练结束于 epoch {epoch+1}")
            break

# --- 7. 测试与评估 ---
model.load_state_dict(torch.load("best_model_causal_dml.pth"))
model.to(dml)
model.eval()

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

test_preds = []
test_targets = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(dml), y_batch.to(dml)
        y_pred, _, _ = model(X_batch)
        test_preds.extend(y_pred.cpu().numpy())
        test_targets.extend(y_batch.cpu().numpy())

test_preds = np.array(test_preds)
test_targets = np.array(test_targets)
test_pred_labels = (test_preds > 0.5).astype(int)

auc = roc_auc_score(test_targets, test_preds)
accuracy = accuracy_score(test_targets, test_pred_labels)
f1 = f1_score(test_targets, test_pred_labels)
precision = precision_score(test_targets, test_pred_labels)
recall = recall_score(test_targets, test_pred_labels)

print("\n" + "="*40)
print(f"测试集评估结果 (Causal Disentanglement):")
print(f"AUC: {auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("="*40)

# --- 8. 绘制 ROC 曲线 ---
fpr, tpr, _ = roc_curve(test_targets, test_preds)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig(f"{label}_roc.png")
print(f"ROC 曲线已保存至 {label}_roc.png")
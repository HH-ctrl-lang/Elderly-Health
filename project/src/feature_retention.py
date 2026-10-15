import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


DATA_PATH = "pre_data/1=无，2=有/1=无，2=有_data.npz"
MODEL_PATH = "best_model_causal_device.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 性能保留曲线参数
MAX_FEATURES_TO_TEST = 80   # 最多测试的特征数
STEP = 1                    # 步长

# ============================================================
# 1. 加载数据
# ============================================================
data = np.load(DATA_PATH)
X_val = data['X_val']
y_val_raw = data['y_val']
y_val = (y_val_raw == 2).astype(np.float32)

# ============================================================
# 2. 加载模型并获取掩码
# ============================================================
from train_env import CausalDisentangledModel

# 初始化模型结构并加载权重
input_dim = X_val.shape[1]
num_envs = len(np.unique(data['env_val']))
model = CausalDisentangledModel(input_dim, latent_dim=64, num_envs=num_envs)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

mask = model.gate.get_causal_scores().cpu().numpy()
print(f"当前掩码均值: {mask.mean():.3f}, 标准差: {mask.std():.3f}")

# ============================================================
# 3. 按掩码降序排列特征索引
# ============================================================
sorted_idx = np.argsort(mask)[::-1]  # 从大到小

# ============================================================
# 4. 逐步增加特征，训练简单分类器并评估 AUC
# ============================================================
n_features_list = []
auc_scores = []

scaler = StandardScaler()
X_val_scaled = scaler.fit_transform(X_val)

for k in range(1, min(MAX_FEATURES_TO_TEST, input_dim) + 1, STEP):
    selected_idx = sorted_idx[:k]
    X_selected = X_val_scaled[:, selected_idx]

    # 使用逻辑回归
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_selected, y_val)
    
    y_pred_proba = clf.predict_proba(X_selected)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    
    n_features_list.append(k)
    auc_scores.append(auc)


# ============================================================
# 6. 绘制性能保留曲线
# ============================================================
plt.figure(figsize=(10, 5))
plt.plot(n_features_list, auc_scores, marker='o', markersize=3, linewidth=1.5, label='Validation AUC')

plt.xlabel('Number of selected causal features (top-k by mask)')
plt.ylabel('Validation AUC')
plt.title('Causal Feature Retention Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('feature_retention_curve.png', dpi=150)
plt.show()

print(f"\n性能保留曲线已保存至 feature_retention_curve.png")
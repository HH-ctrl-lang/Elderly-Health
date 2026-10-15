"""
预处理 data_mission2.csv → V4 因果解耦训练数据
- 填充 using_phone_year / hours_phone_per_day NaN → 0
- 去除 ID / environment_name / 数据泄露列
- 缺失值填充 + 标准化 + 70/15/15 分层划分
- 保留 environment_code 用于环境解耦
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle
import os
import torch

# ============================================================
# 0. 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_CSV = os.path.join(DATA_DIR, "data_mission2.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "preprocessed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "cognitive_impairment"
ENV_COL = "environment_code"
RANDOM_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 50)
print("加载数据...")
df = pd.read_csv(INPUT_CSV, encoding="gbk")
print(f"原始数据: {df.shape}")

# ============================================================
# 2. 丢弃不需要的列
# ============================================================
DROP_COLS = ["ID", "environment_name"]  # ID 是标识，environment_name 与 environment_code 重复
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
print(f"丢弃 {DROP_COLS} 后: {df.shape}")

# ============================================================
# 3. 丢弃目标变量缺失的行
# ============================================================
df = df.dropna(subset=[TARGET]).copy()
print(f"丢弃目标缺失后: {len(df)} 样本")

# ============================================================
# 4. 特殊填充：using_phone_year / hours_phone_per_day NaN → 0
#    (未使用手机的人，使用年数和每日时长为 0)
# ============================================================
for col in ["using_phone_year", "hours_phone_per_day"]:
    if col in df.columns:
        before = df[col].isna().sum()
        df[col] = df[col].fillna(0.0)
        print(f"{col}: NaN {before} → 0")

# ============================================================
# 5. 其余缺失值填充
# ============================================================
# 分离特征和目标
feature_cols = [c for c in df.columns if c not in [TARGET, ENV_COL]]
env_data = df[ENV_COL].values.astype(np.int64)
y_data = df[TARGET].values.astype(np.float32)
X_data = df[feature_cols].copy()

print(f"\n特征数: {len(feature_cols)}")
print(f"特征列: {feature_cols}")

# 检查各列缺失
print("\n缺失值统计:")
for c in feature_cols:
    n_miss = X_data[c].isna().sum()
    if n_miss > 0:
        print(f"  {c}: {n_miss} NaN")

# 中位数填充数值型，众数填充分类型
num_cols = X_data.select_dtypes(include=[np.number]).columns.tolist()
if num_cols:
    num_imputer = SimpleImputer(strategy="median")
    X_data[num_cols] = num_imputer.fit_transform(X_data[num_cols])
    print(f"\n数值列 ({len(num_cols)}个): 中位数填充")

# ============================================================
# 6. 确认所有分类变量为整数编码
# ============================================================
for c in feature_cols:
    if X_data[c].dtype == 'float64' and c not in ["using_phone_year", "hours_phone_per_day"]:
        # 将原本应是整数的浮点列转回整数
        if X_data[c].apply(lambda x: x == int(x) if not pd.isna(x) else True).all():
            X_data[c] = X_data[c].astype(np.int64)

# ============================================================
# 7. 标准化
# ============================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_data)
print(f"标准化完成, X shape: {X_scaled.shape}")

# ============================================================
# 8. 数据集划分 (70/15/15, 分层)
# ============================================================
# 测试集
X_temp, X_test, y_temp, y_test, env_temp, env_test = train_test_split(
    X_scaled, y_data, env_data,
    test_size=TEST_RATIO,
    random_state=RANDOM_SEED,
    stratify=y_data
)

# 验证集 (从剩余中取 val_ratio/(train_ratio+val_ratio))
val_ratio_adj = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
X_train, X_val, y_train, y_val, env_train, env_val = train_test_split(
    X_temp, y_temp, env_temp,
    test_size=val_ratio_adj,
    random_state=RANDOM_SEED,
    stratify=y_temp
)

print(f"\n数据划分:")
print(f"  训练集: {len(X_train)} 样本")
print(f"  验证集: {len(X_val)} 样本")
print(f"  测试集: {len(X_test)} 样本")
print(f"  标签分布 (train): 1={sum(y_train==1)}, 2={sum(y_train==2)}")

# ============================================================
# 9. 保存
# ============================================================
# npz (用于 V4 训练)
np.savez(
    os.path.join(OUTPUT_DIR, "preprocessed_data.npz"),
    X_train=X_train.astype(np.float32),
    X_val=X_val.astype(np.float32),
    X_test=X_test.astype(np.float32),
    y_train=y_train.astype(np.float32),
    y_val=y_val.astype(np.float32),
    y_test=y_test.astype(np.float32),
    env_train=env_train.astype(np.int64),
    env_val=env_val.astype(np.int64),
    env_test=env_test.astype(np.int64),
    feature_names=np.array(feature_cols),
)

# pt (可选, 方便 torch.load)
torch.save({
    "X_train": torch.tensor(X_train, dtype=torch.float32),
    "X_val": torch.tensor(X_val, dtype=torch.float32),
    "X_test": torch.tensor(X_test, dtype=torch.float32),
    "y_train": torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    "y_val": torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
    "y_test": torch.tensor(y_test, dtype=torch.float32).unsqueeze(1),
    "env_train": torch.tensor(env_train, dtype=torch.long),
    "env_val": torch.tensor(env_val, dtype=torch.long),
    "env_test": torch.tensor(env_test, dtype=torch.long),
    "feature_names": feature_cols,
}, os.path.join(OUTPUT_DIR, "preprocessed_data.pt"))

# scaler
with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# 特征列表 (可读文本)
with open(os.path.join(OUTPUT_DIR, "feature_names.txt"), "w", encoding="utf-8") as f:
    for feat in feature_cols:
        f.write(feat + "\n")

# 环境列表
with open(os.path.join(OUTPUT_DIR, "environment_codes.txt"), "w", encoding="utf-8") as f:
    env_names = df["environment_name"] if "environment_name" in df.columns else None
    if env_names is not None:
        for code, name in sorted(zip(df[ENV_COL], df["environment_name"])):
            f.write(f"{code}\t{name}\n")
    else:
        for code in sorted(df[ENV_COL].unique()):
            f.write(f"{code}\n")

print(f"\n预处理数据已保存至: {OUTPUT_DIR}")
print("=" * 50)

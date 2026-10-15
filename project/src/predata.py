import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
import os
import torch

DATA_PATH = "project/data/数据.xlsx"   # 原始数据路径
OUTPUT_DIR = "pre_data"          # 输出目录

# 目标列名称
TARGET = "是否抑郁"

#需要丢弃的列
DROP_COLS = [
    # 标识与隐私
    "个人编码", "调查对象姓名", "监测点名称", "监测点代码", "乡镇名称", "村居名称",
    "联系电话", "家庭住址", "身份证号", "调查员姓名", "日期", "生日", "出生年份", "出生月份",
    "本人联系电话", "本人座机", "配偶联系电话", "配偶座机", "子女联系电话", "子女座机",
    "A1", "a2_a", "a2_b", "a2_c", "a2_d", "a2_e", "a2_f",
    # 调查管理
    "是否为新增调查对象", "是否能完成调查", "无法完成调查原因", "调查开始时间", "调查结束时间",
    "SelfSn", "省代码", "省名称", "是否死亡", "是否有死亡医学证明书",
    "DD1_y", "DD1_m", "DD1_d", "DD2", "DD3I_a1", "DD3I_a2", "DD3I_b1", "DD3I_b2",
    "DD3I_c1", "DD3I_c2", "DD3I_d1", "DD3I_d2", "DD3II1", "DD3II2", "DD4", "DD5",
    "DD6a", "DD6b", "DD6c", "DD6c_other", "DD6d", "DD7a", "DD7b",
    # 认知量表细项（泄露目标）
    "定向力得分", "F1_1", "F1_2", "F1_3", "F1_4", "F1_5", "F1_6", "F1_7", "F1_8", "F1_9", "F1_10",
    "F2_11", "F2_12", "F2_13", "F2_13_OTHER", "F3_14", "F3_15", "F3_16", "F3_17", "F3_18",
    "F4_19", "F4_20", "F4_21", "F5_22", "F5_23", "F5_24", "F5_25", "F5_26", "F5_27", "F5_28",
    "F5_29", "F5_29_OTHER", "F5_30", "MMSE得分", "F5_31", "F5_32", "F5_32_OTHER",
    # 抑郁量表细项（泄露目标）
    "抑郁总分", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10",
    "H11", "H12", "H13", "H14", "H15", "H16", "H17",
    # ADL/IADL细项（泄露功能状态）
    "ADL总分", "日常活动受损", "Ea1", "Ea2", "Ea3", "Ea4", "Ea5", "Ea6",
    "IALD总分", "工具日常活动受损", "Eb1", "Eb2", "Eb3", "Eb4", "Eb5", "Eb6", "Eb7", "Eb8",
    # 主观睡眠问题细项
    "C10a", "C10b", "C10c", "C10d", "C10e", "C10f", "C10g", "C10h", "C10i", "C10_other", "C10j",
    "C11", "C12", "C13", "C14",
    # 自由文本及其他
    "A8_other", "A9_other", "B7b", "城乡", "年龄分组", "慢病数量分组", "体重情况"
]

# 分类特征（需要进行 Label Encoding）
CATEGORICAL_COLS = [
    "城乡", "性别", "民族", "文化程度", "婚姻状态", "独居", "父母同住", "配偶同住", "子女同住",
    "孙辈同住", "保姆同住", "职业", "生活来源", "财富状况",
    "血压", "B1c", "B1d", "血糖", "B2c_a", "B2c_b", "B2c_c", "B2c_d",
    "血脂异常", "B3b1_1", "B3b1_2", "B3b1_3", "B3b1_4", "B3d",
    "心绞痛", "心肌梗死", "房颤", "心脏搭桥/支架手术", "缺血性脑卒中", "出血性脑卒中",
    "COPD", "哮喘", "骨关节疾病", "颈、腰部疾病", "慢性消化系统", "慢性泌尿系统", "恶性肿瘤",
    "吸烟", "饮酒", "C2b", "饮茶", "家务", "户外活动", "种花养宠物", "阅读书报",
    "饲养家禽", "打牌打麻将", "看电视听广播", "社会活动", "高强度活动", "中等强度体力活动", "锻炼",
    "智能手机", "G3_1", "G3_2", "G3_3", "G3_4", "G3_5", "G3_6", "G3_7", "G3_8", "G3_9",
    "G3_10", "G3_11", "G5_1", "G5_2", "G5_3", "G5_4", "G5_5", "G5_6", "G5_7", "G5_8", "G5_9", "G5_10",
    "G6_1", "G6_2", "G6_3", "G6_4", "G6_5", "G6_6", "G6_7", "G6_8", "G6_9", "G6_10"
]

# 缺失值填充策略
NUMERIC_IMPUTE_STRATEGY = "median"
CATEGORICAL_IMPUTE_STRATEGY = "most_frequent"

# 特征筛选阈值
MISSING_THRESHOLD = 0.5       #数值缺失比例超过该值则丢弃
VARIANCE_THRESHOLD = 0.01     #标准化后数值方差小于该值则丢弃
CORRELATION_THRESHOLD = 0.95  #数值相关性大于该值只保留一列

# 数据集划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

#辅助函数
def load_data(file_path):
    df = pd.read_excel(file_path)
    print(f"原始数据形状: {df.shape}")
    return df

def basic_info(df):
    print("\n【数据信息】")
    print(df.info())
    print("\n【前5行数据】")
    print(df.head())
    print("\n【描述性统计】")
    print(df.describe(include='all'))

def drop_useless_columns(df, drop_list):
    existing_drop = [col for col in drop_list if col in df.columns]
    if existing_drop:
        df = df.drop(columns=existing_drop)
        print(f"已丢弃列: {existing_drop}")
    return df

def handle_missing_values(df, categorical_cols, num_strategy, cat_strategy):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [col for col in categorical_cols if col in df.columns]
    
    if num_cols:
        num_imputer = SimpleImputer(strategy=num_strategy)
        df[num_cols] = num_imputer.fit_transform(df[num_cols])
        print(f"数值列({len(num_cols)}个)缺失值已用 {num_strategy} 填充")
    if cat_cols:
        cat_imputer = SimpleImputer(strategy=cat_strategy)
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
        print(f"分类列({len(cat_cols)}个)缺失值已用 {cat_strategy} 填充")
    return df

def drop_high_missing_columns(df, threshold):
    missing_ratio = df.isnull().mean()
    high_missing_cols = missing_ratio[missing_ratio > threshold].index.tolist()
    if high_missing_cols:
        df = df.drop(columns=high_missing_cols)
        print(f"已丢弃缺失比例 > {threshold} 的列: {high_missing_cols}")
    return df

def drop_low_variance_columns(df, threshold, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    low_var_cols = []
    for col in num_cols:
        if col in exclude_cols:
            continue
        if df[col].var() < threshold:
            low_var_cols.append(col)
    if low_var_cols:
        df = df.drop(columns=low_var_cols)
        print(f"已丢弃低方差( < {threshold})的数值列: {low_var_cols}")
    return df

def drop_high_correlation_columns(df, threshold):
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return df
    corr_matrix = num_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    if to_drop:
        df = df.drop(columns=to_drop)
        print(f"已丢弃高相关( > {threshold})的列: {to_drop}")
    return df

def standardize_features(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scale_cols = [col for col in num_cols if col not in exclude_cols]
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    print(f"已完成 {scale_cols} 列数值特征的标准化")
    return df, scaler

def split_data(df, target_col, train_ratio, val_ratio, test_ratio, random_seed):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_seed, stratify=y
    )
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adjusted, random_state=random_seed, stratify=y_temp
    )
    print(f"\n数据集划分: 训练集 {len(X_train)}, 验证集 {len(X_val)}, 测试集 {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test,
                        scaler, feature_list, output_dir, target_name):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存为 NumPy 压缩格式（供 DirectML 训练脚本直接使用）
    np.savez(os.path.join(output_dir, f"{target_name}_data.npz"),
             X_train=X_train.values.astype(np.float32),
             X_val=X_val.values.astype(np.float32),
             X_test=X_test.values.astype(np.float32),
             y_train=y_train.values.astype(np.float32),
             y_val=y_val.values.astype(np.float32),
             y_test=y_test.values.astype(np.float32),
             feature_names=X_train.columns.tolist())
    
    # 2. (可选) 同时保存为 PyTorch .pt 格式，方便直接 torch.load
    torch.save({
        'X_train': torch.tensor(X_train.values, dtype=torch.float32),
        'X_val': torch.tensor(X_val.values, dtype=torch.float32),
        'X_test': torch.tensor(X_test.values, dtype=torch.float32),
        'y_train': torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1),
        'y_val': torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1),
        'y_test': torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1),
        'feature_names': X_train.columns.tolist()
    }, os.path.join(output_dir, f"{target_name}_data.pt"))
    
    # 3. 保存预处理器
    with open(os.path.join(output_dir, f"{target_name}_scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)
    
    # 4. 保存特征列表（文本格式，便于查看）
    with open(os.path.join(output_dir, f"{target_name}_features.txt"), 'w', encoding='utf-8') as f:
        for feat in X_train.columns:
            f.write(feat + '\n')
    
    print(f"处理后的数据已保存至: {output_dir}")

def preprocess_pipeline(df, target_col, categorical_cols, drop_cols, output_dir):
    print(f"\n{'='*50}")
    print(f"处理目标变量: {target_col}")
    print(f"{'='*50}")
    
    if target_col not in df.columns:
        raise ValueError(f"目标列 '{target_col}' 不存在")
    df = df.dropna(subset=[target_col]).copy()
    print(f"删除缺失目标值后样本数: {len(df)}")
    
    df = drop_useless_columns(df, drop_cols)
    df = drop_high_missing_columns(df, MISSING_THRESHOLD)
    df = handle_missing_values(df, categorical_cols,
                               NUMERIC_IMPUTE_STRATEGY, CATEGORICAL_IMPUTE_STRATEGY)
    
    # 临时标准化用于低方差筛选
    temp_df = df.copy()
    exclude_cols = [target_col]
    num_cols = temp_df.select_dtypes(include=[np.number]).columns.tolist()
    scale_cols = [col for col in num_cols if col not in exclude_cols]
    if scale_cols:
        temp_scaler = StandardScaler()
        temp_df[scale_cols] = temp_scaler.fit_transform(temp_df[scale_cols])
    temp_df = drop_low_variance_columns(temp_df, VARIANCE_THRESHOLD, exclude_cols=exclude_cols)
    df = df[temp_df.columns]
    
    df = drop_high_correlation_columns(df, CORRELATION_THRESHOLD)
    df, scaler = standardize_features(df, exclude_cols=[target_col])
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, target_col, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )
    
    save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test,
                        scaler, X_train.columns.tolist(), output_dir, target_col)
    
    print(f"目标 '{target_col}' 预处理完成，最终特征数: {X_train.shape[1]}")
    return X_train, X_val, X_test, y_train, y_val, y_test

# ==================== 主程序 ====================

if __name__ == "__main__":
    print("=== 数据预处理 ===")
    df_raw = load_data(DATA_PATH)
    basic_info(df_raw)
    
    # 处理目标变量
    target=TARGET
    if target not in df_raw.columns:
        print(f"列 '{target}' 不存在")
    preprocess_pipeline(
        df=df_raw.copy(),
        target_col=target,
        categorical_cols=CATEGORICAL_COLS,
        drop_cols=DROP_COLS,
        output_dir=os.path.join(OUTPUT_DIR, target.replace(" ", "_"))
    )
    
    print("\n所有预处理完成！")
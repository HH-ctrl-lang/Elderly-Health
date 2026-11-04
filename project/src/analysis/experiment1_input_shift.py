"""任务二，实验一：东部与中西部输入分布偏移诊断。

本脚本只读取已有的 data_mission2.csv，不重新执行数据预处理。
分析分为两部分：
1. 以监测点为单位，比较东部与中西部19个输入特征的类别分布和缺失情况；
2. 使用19个输入特征识别东部与中西部，按监测点分组进行交叉验证。

地区识别模型的预测目标是地区标签，不是认知功能受损标签。
认知功能受损标签不会被读取，也不会参与模型训练。
"""

from datetime import datetime
from itertools import combinations
from pathlib import Path
import hashlib
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ==================== 一、基础配置 ====================

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = ROOT / "project" / "data" / "data_mission2.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "analysis_2611" / "experiment1_input_shift"

RANDOM_SEED = 42
N_SPLITS = 4
CLASSIFIER_C = 1.0

# 任务二，实验一：沿用项目中已经确定的监测点地区映射。
PROVINCE_TO_REGION = {
    "11": "east",
    "37": "east",
    "22": "central",
    "36": "central",
    "45": "west",
    "64": "west",
}

# 任务二，实验一：认知功能受损任务当前使用的19个输入特征。
FEATURES = [
    "residence",
    "age",
    "gender",
    "education",
    "marital_status",
    "living_alone",
    "wealth",
    "chronic_disease",
    "social_activity",
    "IADL",
    "depression",
    "sleep_quality",
    "sleep_duration",
    "physical_activity",
    "if_use_mobile_phone",
    "using_phone_year",
    "hours_phone_per_day",
    "smoke",
    "alcohol",
]

# 任务二，实验一：记录变量类型；本实验对所有变量使用完整类别分布，
# 不把有序编码之间的数值间距解释为真实距离。
FEATURE_TYPES = {
    "residence": "nominal",
    "age": "ordinal",
    "gender": "nominal",
    "education": "ordinal",
    "marital_status": "nominal",
    "living_alone": "nominal",
    "wealth": "ordinal",
    "chronic_disease": "ordinal",
    "social_activity": "ordinal",
    "IADL": "ordinal",
    "depression": "nominal",
    "sleep_quality": "ordinal",
    "sleep_duration": "ordinal",
    "physical_activity": "nominal",
    "if_use_mobile_phone": "nominal",
    "using_phone_year": "ordinal",
    "hours_phone_per_day": "ordinal",
    "smoke": "nominal",
    "alcohol": "nominal",
}


# ==================== 二、通用统计函数 ====================

def normalize_category(value):
    """任务二，实验一：将数值编码转换为稳定的类别标签。"""
    if pd.isna(value):
        return None

    number = float(value)
    if number.is_integer():
        return str(int(number))

    return format(number, ".8g")


def js_divergence(count_a, count_b):
    """任务二，实验一：计算两个类别分布的Jensen-Shannon散度，单位为bits。"""
    count_a = np.asarray(count_a, dtype=float)
    count_b = np.asarray(count_b, dtype=float)

    if count_a.sum() <= 0 or count_b.sum() <= 0:
        return np.nan

    p = count_a / count_a.sum()
    q = count_b / count_b.sum()
    midpoint = (p + q) / 2

    mask_p = p > 0
    mask_q = q > 0

    kl_p = np.sum(p[mask_p] * np.log2(p[mask_p] / midpoint[mask_p]))
    kl_q = np.sum(q[mask_q] * np.log2(q[mask_q] / midpoint[mask_q]))

    return float((kl_p + kl_q) / 2)


def exact_site_permutation(count_matrix, east_indices):
    """任务二，实验一：按监测点进行精确置换检验。

    保持东部监测点数量不变，在12个监测点中枚举所有等规模分组。
    该检验以监测点为置换单位，不把个人样本误当作独立地区单位。
    由于地区标签并非随机分配，检验结果只作为探索性证据解释。
    """
    count_matrix = np.asarray(count_matrix, dtype=float)
    n_sites = count_matrix.shape[0]
    east_indices = tuple(sorted(east_indices))
    east_size = len(east_indices)

    observed_other = tuple(
        index for index in range(n_sites) if index not in set(east_indices)
    )
    observed = js_divergence(
        count_matrix[list(east_indices)].sum(axis=0),
        count_matrix[list(observed_other)].sum(axis=0),
    )

    total = 0
    extreme = 0

    for selected in combinations(range(n_sites), east_size):
        selected_set = set(selected)
        remaining = tuple(
            index for index in range(n_sites) if index not in selected_set
        )
        statistic = js_divergence(
            count_matrix[list(selected)].sum(axis=0),
            count_matrix[list(remaining)].sum(axis=0),
        )
        total += 1
        if statistic >= observed - 1e-12:
            extreme += 1

    return observed, extreme / total, total


def bh_adjust(p_values):
    """任务二，实验一：对同一检验族执行Benjamini-Hochberg校正。"""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan)

    valid_indices = np.flatnonzero(np.isfinite(p_values))
    if len(valid_indices) == 0:
        return adjusted

    order = valid_indices[np.argsort(p_values[valid_indices])]
    total = len(order)
    running_minimum = 1.0

    for rank_index in range(total - 1, -1, -1):
        rank = rank_index + 1
        current = p_values[order[rank_index]] * total / rank
        running_minimum = min(running_minimum, current)
        adjusted[order[rank_index]] = min(running_minimum, 1.0)

    return adjusted


# ==================== 三、读取现有数据 ====================

def load_data(data_path):
    """任务二，实验一：只读取地区识别所需字段，不读取健康结局和个人ID。"""
    data_path = Path(data_path)

    if not data_path.is_file():
        raise FileNotFoundError(f"找不到已有模型输入：{data_path}")

    required_columns = FEATURES + ["environment_code"]

    frame = pd.read_csv(
        data_path,
        encoding="utf-8-sig",
        usecols=required_columns,
        dtype={"environment_code": "string"},
    )

    if frame.empty:
        raise ValueError("数据文件没有有效记录。")

    frame["environment_code"] = (
        frame["environment_code"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    if frame["environment_code"].isna().any():
        raise ValueError("监测点代码存在缺失。")

    valid_code = frame["environment_code"].str.fullmatch(r"[0-9]{6}")
    if not valid_code.all():
        raise ValueError("监测点代码必须为六位数字。")

    frame["region"] = frame["environment_code"].str[:2].map(PROVINCE_TO_REGION)

    if frame["region"].isna().any():
        invalid_codes = sorted(
            frame.loc[frame["region"].isna(), "environment_code"].unique().tolist()
        )
        raise ValueError(f"以下监测点没有地区映射：{invalid_codes}")

    for feature in FEATURES:
        original = frame[feature]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()

        if invalid.any():
            raise ValueError(f"特征 {feature} 存在无法转换的编码。")

        if np.isinf(converted.dropna().to_numpy(dtype=float)).any():
            raise ValueError(f"特征 {feature} 存在无穷值。")

        frame[feature] = converted

    # 任务二，实验一：核对监测点数量和三个地区的监测点构成。
    site_region = frame[["environment_code", "region"]].drop_duplicates()

    if site_region["environment_code"].duplicated().any():
        raise ValueError("同一监测点被映射到多个地区。")

    site_counts = site_region.groupby("region")["environment_code"].nunique().to_dict()
    expected_regions = {"east": 4, "central": 4, "west": 4}

    if site_counts != expected_regions:
        raise ValueError(
            f"监测点构成与当前12站点研究设计不一致：{site_counts}"
        )

    return frame


# ==================== 四、特征分布与缺失分析 ====================

def analyze_feature_distribution(frame):
    """任务二，实验一：计算类别分布、JS距离和监测点层面的置换检验。"""
    site_codes = sorted(frame["environment_code"].unique().tolist())
    east_indices = [
        index
        for index, site in enumerate(site_codes)
        if PROVINCE_TO_REGION[site[:2]] == "east"
    ]

    east_mask = frame["region"] == "east"
    source_mask = frame["region"].isin(["central", "west"])

    distribution_rows = []
    summary_rows = []

    for feature in FEATURES:
        labels = frame[feature].map(normalize_category)
        categories = sorted(
            labels.dropna().unique().tolist(),
            key=lambda value: float(value),
        )

        # 任务二，实验一：输出各地区各类别的样本数和有效样本比例。
        for region in ["east", "central", "west"]:
            region_values = labels[frame["region"] == region]
            valid_n = int(region_values.notna().sum())
            missing_n = int(region_values.isna().sum())
            total_n = len(region_values)

            for category in categories:
                count = int((region_values == category).sum())
                distribution_rows.append(
                    {
                        "feature": feature,
                        "feature_type": FEATURE_TYPES[feature],
                        "region": region,
                        "category_code": category,
                        "count": count,
                        "valid_n": valid_n,
                        "percentage": count / valid_n if valid_n else np.nan,
                        "missing_n": missing_n,
                        "total_n": total_n,
                        "missing_rate": missing_n / total_n if total_n else np.nan,
                    }
                )

        east_values = labels[east_mask]
        source_values = labels[source_mask]

        east_counts = np.asarray(
            [(east_values == category).sum() for category in categories],
            dtype=int,
        )
        source_counts = np.asarray(
            [(source_values == category).sum() for category in categories],
            dtype=int,
        )

        east_distribution = east_counts / east_counts.sum()
        source_distribution = source_counts / source_counts.sum()
        max_category_difference = float(
            np.max(np.abs(east_distribution - source_distribution))
        )

        # 任务二，实验一：构建监测点×类别频数矩阵。
        site_category_matrix = np.zeros(
            (len(site_codes), len(categories)),
            dtype=int,
        )
        site_missing_matrix = np.zeros((len(site_codes), 2), dtype=int)

        for site_index, site in enumerate(site_codes):
            site_values = labels[frame["environment_code"] == site]

            for category_index, category in enumerate(categories):
                site_category_matrix[site_index, category_index] = int(
                    (site_values == category).sum()
                )

            site_missing_matrix[site_index, 0] = int(site_values.isna().sum())
            site_missing_matrix[site_index, 1] = int(site_values.notna().sum())

        js_value, js_p, permutation_count = exact_site_permutation(
            site_category_matrix,
            east_indices,
        )
        missing_js, missing_p, _ = exact_site_permutation(
            site_missing_matrix,
            east_indices,
        )

        east_missing = int(east_values.isna().sum())
        source_missing = int(source_values.isna().sum())
        east_missing_rate = east_missing / len(east_values)
        source_missing_rate = source_missing / len(source_values)

        summary_rows.append(
            {
                "feature": feature,
                "feature_type": FEATURE_TYPES[feature],
                "east_valid_n": int(east_values.notna().sum()),
                "source_valid_n": int(source_values.notna().sum()),
                "js_divergence_bits": js_value,
                "max_category_difference": max_category_difference,
                "east_missing_n": east_missing,
                "source_missing_n": source_missing,
                "east_missing_rate": east_missing_rate,
                "source_missing_rate": source_missing_rate,
                "missing_rate_difference": east_missing_rate - source_missing_rate,
                "site_permutation_p": js_p,
                "missing_js_divergence_bits": missing_js,
                "missing_permutation_p": missing_p,
                "permutation_count": permutation_count,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary["js_q_BH"] = bh_adjust(summary["site_permutation_p"])
    summary["missing_q_BH"] = bh_adjust(summary["missing_permutation_p"])

    distribution = pd.DataFrame(distribution_rows)
    return summary, distribution


def build_site_summary(frame):
    """任务二，实验一：记录各监测点样本量，不使用健康结局。"""
    return (
        frame.groupby(["region", "environment_code"])
        .size()
        .reset_index(name="sample_count")
        .sort_values(["region", "environment_code"])
        .reset_index(drop=True)
    )


def build_region_summary(frame):
    """任务二，实验一：记录地区样本量和监测点数量，不计算健康结局比例。"""
    return (
        frame.groupby("region")
        .agg(
            sample_count=("environment_code", "size"),
            site_count=("environment_code", "nunique"),
        )
        .reset_index()
        .sort_values("region")
        .reset_index(drop=True)
    )


# ==================== 五、地区识别模型 ====================

def build_region_classifier():
    """任务二，实验一：建立不使用健康结局的地区识别逻辑回归模型。"""
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value=-999.0,
                    keep_empty_features=True,
                ),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categories", categorical_pipeline, FEATURES),
            (
                "missing_flags",
                MissingIndicator(features="all", error_on_new=False),
                FEATURES,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    C=CLASSIFIER_C,
                    class_weight="balanced",
                    max_iter=3000,
                    solver="liblinear",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def binary_metrics(y_true, probability, threshold=0.5):
    """任务二，实验一：计算地区识别模型的分类指标。"""
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    ).ravel()

    return {
        "auc": roc_auc_score(y_true, probability),
        "average_precision": average_precision_score(y_true, probability),
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "sensitivity_east": tp / (tp + fn) if tp + fn else np.nan,
        "specificity_non_east": tn / (tn + fp) if tn + fp else np.nan,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def build_site_grouped_folds(frame):
    """任务二，实验一：按地区分层分配监测点，构造可复现的4折分组交叉验证。

    当前设计为东、中、西各4个监测点。每折从每个地区留出1个监测点，
    因而测试集必定同时包含东部和非东部样本；其余9个监测点用于训练。
    同一监测点的所有样本只会进入一个测试折，不会跨训练集和测试集。
    """
    site_table = (
        frame[["environment_code", "region"]]
        .drop_duplicates()
        .sort_values(["region", "environment_code"])
    )

    if site_table["environment_code"].duplicated().any():
        raise ValueError("同一监测点对应多个地区，无法构造分组交叉验证。")

    rng = np.random.RandomState(RANDOM_SEED)
    sites_by_region = {}

    for region in ("east", "central", "west"):
        sites = site_table.loc[
            site_table["region"] == region, "environment_code"
        ].astype(str).tolist()

        if len(sites) != N_SPLITS:
            raise ValueError(
                f"{region}地区有{len(sites)}个监测点，"
                f"当前{N_SPLITS}折设计要求每个地区恰好有{N_SPLITS}个监测点。"
            )

        # 任务二，实验一：仅打乱监测点分配，不拆分监测点内的个人样本。
        rng.shuffle(sites)
        sites_by_region[region] = sites

    folds = []
    all_test_sites = []

    for fold_index in range(N_SPLITS):
        # 任务二，实验一：每折从东、中、西各留出1个完整监测点。
        test_sites_by_region = {
            region: sites_by_region[region][fold_index]
            for region in ("east", "central", "west")
        }
        test_sites = set(test_sites_by_region.values())

        test_mask = frame["environment_code"].astype(str).isin(test_sites).to_numpy()
        test_indices = np.flatnonzero(test_mask)
        train_indices = np.flatnonzero(~test_mask)

        train_sites = set(frame.iloc[train_indices]["environment_code"].astype(str))
        actual_test_sites = set(
            frame.iloc[test_indices]["environment_code"].astype(str)
        )

        if actual_test_sites != test_sites:
            raise RuntimeError(
                f"第{fold_index + 1}折监测点分配与样本索引不一致。"
            )

        if train_sites.intersection(actual_test_sites):
            raise RuntimeError(f"第{fold_index + 1}折出现监测点交叉泄漏。")

        test_regions = set(frame.iloc[test_indices]["region"])
        train_regions = set(frame.iloc[train_indices]["region"])

        if test_regions != {"east", "central", "west"}:
            raise RuntimeError(
                f"第{fold_index + 1}折测试集未覆盖三个地区。"
            )

        if train_regions != {"east", "central", "west"}:
            raise RuntimeError(
                f"第{fold_index + 1}折训练集未覆盖三个地区。"
            )

        folds.append(
            {
                "train_indices": train_indices,
                "test_indices": test_indices,
                "test_sites_by_region": test_sites_by_region,
            }
        )
        all_test_sites.extend(actual_test_sites)

    expected_sites = set(site_table["environment_code"].astype(str))
    if len(all_test_sites) != len(expected_sites) or set(all_test_sites) != expected_sites:
        raise RuntimeError("交叉验证分配未能保证每个监测点恰好作为测试组一次。")

    return folds


def evaluate_region_classifier(frame):
    """任务二，实验一：按地区分层的监测点分组4折交叉验证。"""
    features = frame[FEATURES].copy()
    target = (frame["region"] == "east").astype(int).to_numpy()
    groups = frame["environment_code"].astype(str).to_numpy()

    base_model = build_region_classifier()
    oof_probability = np.full(len(frame), np.nan, dtype=float)
    fold_assignment = np.full(len(frame), -1, dtype=int)
    fold_rows = []

    # 任务二，实验一：固定地区内监测点分配，保证每折都有三类地区的留出站点。
    folds = build_site_grouped_folds(frame)

    for fold, split in enumerate(folds, start=1):
        train_indices = split["train_indices"]
        test_indices = split["test_indices"]
        test_sites_by_region = split["test_sites_by_region"]

        train_groups = set(groups[train_indices])
        test_groups = set(groups[test_indices])

        if train_groups.intersection(test_groups):
            raise RuntimeError(f"第{fold}折出现监测点交叉泄漏。")

        if set(np.unique(target[test_indices])) != {0, 1}:
            raise RuntimeError(f"第{fold}折测试集缺少地区类别，无法计算AUC。")

        if set(np.unique(target[train_indices])) != {0, 1}:
            raise RuntimeError(f"第{fold}折训练集缺少地区类别。")

        # 任务二，实验一：每折独立克隆模型，预处理器只在该折训练数据上拟合。
        model = clone(base_model)
        model.fit(
            features.iloc[train_indices],
            target[train_indices],
        )

        probability = model.predict_proba(
            features.iloc[test_indices]
        )[:, 1]

        oof_probability[test_indices] = probability
        fold_assignment[test_indices] = fold

        fold_result = binary_metrics(
            target[test_indices],
            probability,
        )

        fold_rows.append(
            {
                "fold": fold,
                "train_n": len(train_indices),
                "test_n": len(test_indices),
                "train_site_count": len(train_groups),
                "test_site_count": len(test_groups),
                "train_sites": ";".join(sorted(train_groups)),
                "test_sites": ";".join(sorted(test_groups)),
                "test_east_site": test_sites_by_region["east"],
                "test_central_site": test_sites_by_region["central"],
                "test_west_site": test_sites_by_region["west"],
                **fold_result,
            }
        )

    if np.isnan(oof_probability).any() or (fold_assignment < 1).any():
        raise RuntimeError("存在未获得折外预测值的样本。")

    overall_metrics = binary_metrics(target, oof_probability)

    metrics = pd.DataFrame(
        [
            {
                "sample_count": len(frame),
                "east_count": int(target.sum()),
                "non_east_count": int((target == 0).sum()),
                "site_count": len(np.unique(groups)),
                "fold_count": N_SPLITS,
                "classifier": "L2 Logistic Regression",
                "features": len(FEATURES),
                "feature_target": "east=1; central_or_west=0",
                "cross_validation": "region-stratified site-grouped 4-fold",
                "threshold": 0.5,
                **overall_metrics,
            }
        ]
    )

    fold_metrics = pd.DataFrame(fold_rows)

    predictions = pd.DataFrame(
        {
            "row_index": np.arange(len(frame)),
            "environment_code": groups,
            "region": frame["region"].to_numpy(),
            "fold": fold_assignment,
            "true_east": target,
            "predicted_probability_east": oof_probability,
            "predicted_east": (oof_probability >= 0.5).astype(int),
        }
    )

    return metrics, fold_metrics, predictions


# ==================== 六、可视化 ====================

def create_figures(summary, metrics, predictions, output_dir):
    """任务二，实验一：生成特征分布偏移、缺失差异和ROC图。"""
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["axes.unicode_minus"] = False

    # 任务二，实验一：绘制JS距离最大的10个特征。
    top_features = summary.nlargest(10, "js_divergence_bits").sort_values(
        "js_divergence_bits"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        top_features["feature"],
        top_features["js_divergence_bits"],
    )
    ax.set_xlabel("Jensen-Shannon Divergence (bits)")
    ax.set_title("East vs Central/West: Feature Distribution Shift")
    fig.tight_layout()
    fig.savefig(
        figure_dir / "feature_js_divergence.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 任务二，实验一：绘制东部与中西部缺失率差异。
    missing_features = summary.reindex(
        summary["missing_rate_difference"].abs().sort_values(ascending=False).index
    ).head(10).sort_values("missing_rate_difference")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        missing_features["feature"],
        missing_features["missing_rate_difference"],
    )
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("East Missing Rate - Central/West Missing Rate")
    ax.set_title("Feature Missingness Difference")
    fig.tight_layout()
    fig.savefig(
        figure_dir / "feature_missing_difference.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 任务二，实验一：绘制全部折外预测值对应的ROC曲线。
    y_true = predictions["true_east"].to_numpy()
    y_score = predictions["predicted_probability_east"].to_numpy()
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_score)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"AUC = {metrics.iloc[0]['auc']:.4f}",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Grouped Cross-Validation: East vs Central/West")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(
        figure_dir / "region_classifier_roc.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# ==================== 七、实验记录与主程序 ====================

def run_experiment(data_path, output_root):
    """任务二，实验一：执行分析并保存可复核的实验记录。"""
    data_path = Path(data_path).resolve()
    output_root = Path(output_root).resolve()

    frame = load_data(data_path)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    print("=" * 65)
    print("任务二，实验一：东部与中西部输入分布偏移诊断")
    print("=" * 65)
    print(f"输入文件：{data_path}")
    print(f"有效样本数：{len(frame)}")
    print(f"监测点数量：{frame['environment_code'].nunique()}")
    print(f"输入特征数：{len(FEATURES)}")
    print("注意：本实验不读取认知功能受损标签。")

    # 任务二，实验一：计算类别分布和监测点层面的置换检验。
    feature_summary, feature_distribution = analyze_feature_distribution(frame)

    # 任务二，实验一：记录地区和监测点样本构成。
    region_summary = build_region_summary(frame)
    site_summary = build_site_summary(frame)

    # 任务二，实验一：运行地区识别模型，保留折外预测证据。
    classifier_metrics, fold_metrics, predictions = (
        evaluate_region_classifier(frame)
    )

    # 任务二，实验一：保存表格结果。
    feature_summary.to_csv(
        output_dir / "feature_shift_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feature_distribution.to_csv(
        output_dir / "region_feature_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )
    region_summary.to_csv(
        output_dir / "region_sample_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    site_summary.to_csv(
        output_dir / "site_sample_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    classifier_metrics.to_csv(
        output_dir / "region_classifier_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fold_metrics.to_csv(
        output_dir / "region_classifier_fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        output_dir / "region_classifier_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 任务二，实验一：记录数据版本、模型配置和软件环境。
    metadata = {
        "experiment": "Task 2 Experiment 1 - Input Distribution Shift",
        "run_id": run_id,
        "input_file": str(data_path),
        "input_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "sample_count": int(len(frame)),
        "site_count": int(frame["environment_code"].nunique()),
        "region_sample_counts": {
            str(key): int(value)
            for key, value in frame["region"].value_counts().to_dict().items()
        },
        "features": FEATURES,
        "feature_types": FEATURE_TYPES,
        "region_mapping": PROVINCE_TO_REGION,
        "classifier_target": "east=1; central_or_west=0",
        "health_outcome_read": False,
        "cross_validation": "4-fold region-stratified site-grouped split; one east, one central, one west site per test fold",
        "group_column": "environment_code",
        "classifier": "L2 Logistic Regression",
        "class_weight": "balanced",
        "one_hot_encoding": True,
        "missing_indicators": True,
        "random_seed": RANDOM_SEED,
        "n_splits": N_SPLITS,
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }

    with (output_dir / "experiment_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)

    # 任务二，实验一：保存图表。
    create_figures(
        feature_summary,
        classifier_metrics,
        predictions,
        output_dir,
    )

    print("\n地区样本构成：")
    print(region_summary.to_string(index=False))

    print("\n特征分布偏移最大的10项：")
    print(
        feature_summary.nlargest(10, "js_divergence_bits")[
            [
                "feature",
                "js_divergence_bits",
                "js_q_BH",
                "missing_rate_difference",
                "missing_q_BH",
            ]
        ].to_string(index=False)
    )

    print("\n地区识别模型折外预测结果：")
    print(classifier_metrics.to_string(index=False))

    print("\n各折结果：")
    print(
        fold_metrics[
            [
                "fold",
                "test_site_count",
                "test_sites",
                "auc",
                "average_precision",
                "balanced_accuracy",
            ]
        ].to_string(index=False)
    )

    print(f"\n实验完成，结果保存位置：{output_dir}")
    return output_dir


def main():
    """任务二，实验一：解析运行参数并启动实验。"""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="已有模型输入CSV路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="实验结果根目录",
    )
    args = parser.parse_args()

    run_experiment(args.data, args.output)


if __name__ == "__main__":
    main()

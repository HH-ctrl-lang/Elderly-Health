"""任务二，实验一补充：复用历史LORO基线，检验协变量偏移加权的收益。

控制变量原则：
1. 未加权MLP直接复用历史LORO结果，不重复训练；
2. 加权MLP沿用历史LORO的地区划分、样本划分、预处理、模型结构和训练参数；
3. 仅改变训练损失中的源域样本权重；
4. 验证集仍使用未加权AUC和未加权损失进行checkpoint选择，匹配历史baseline；
5. 目标地区无标签特征可用于密度比估计，目标地区健康标签仅用于最终评价。

重要方法边界：
本实验使用目标地区无标签特征估计密度比，属于transductive unsupervised domain adaptation，
不是完全不接触目标域特征的纯inductive LORO。它检验的是协变量偏移修正，不证明因果关系。
"""

import hashlib
import json
import pickle
import platform
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

plt.switch_backend("Agg")

# 任务二，实验一补充：加载项目现有的数据、模型和评估流程。
PROJECT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

from elderly_health.config import TrainConfig  # noqa: E402
from elderly_health.data import (  # noqa: E402
    ENVIRONMENT,
    FEATURES,
    PROVINCE_TO_REGION,
    REGION,
    attach_regions,
    held_out_split,
    prepare_splits,
    read_raw,
)
from elderly_health.evaluation import metrics  # noqa: E402
from elderly_health.training import fit, get_device, predict  # noqa: E402


# ==================== 一、实验配置 ====================

DEFAULT_DATA_PATH = PROJECT / "data" / "data_mission2.csv"
DEFAULT_CONFIG_PATH = PROJECT / "configs" / "v2_region_v1_selection91.json"

# 任务二，实验一补充：这是已经完成并提交的未加权LORO基线目录。
DEFAULT_BASELINE_OUTPUT = (
    REPOSITORY_ROOT / "outputs" / "v2-region-loro-selection91-42"
)

DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT
    / "outputs"
    / "analysis_2611"
    / "experiment1b_covariate_reweighting"
)

RANDOM_SEED = 42
N_SPLITS = 4
DOMAIN_CLASSIFIER_C = 0.1
WEIGHT_LOWER_QUANTILE = 0.01
WEIGHT_UPPER_QUANTILE = 0.99

METRIC_KEYS = [
    "auc",
    "auprc",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "precision",
    "recall",
    "log_loss",
    "brier",
    "positive_rate",
]
PREDICTION_METRIC_KEYS = [
    "auc",
    "auprc",
    "log_loss",
    "brier",
    "positive_rate",
]


# ==================== 二、通用函数 ====================

def sha256_file(path):
    """任务二，实验一补充：计算文件SHA256，核对历史实验数据版本。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    """任务二，实验一补充：保存可复核的JSON实验记录。"""
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def normalize_category(value):
    """任务二，实验一补充：将数值编码转换为类别标签，显式保留缺失类别。"""
    if pd.isna(value):
        return "<MISSING>"

    number = float(value)
    if number.is_integer():
        return str(int(number))

    return format(number, ".8g")


def js_divergence(count_a, count_b):
    """任务二，实验一补充：计算类别分布的Jensen-Shannon散度，单位为bits。"""
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


def effective_sample_size(weights):
    """任务二，实验一补充：计算加权样本的有效样本量。"""
    weights = np.asarray(weights, dtype=float).reshape(-1)

    if len(weights) == 0 or not np.isfinite(weights).all():
        return np.nan

    denominator = np.square(weights).sum()
    if denominator <= 0:
        return np.nan

    return float(weights.sum() ** 2 / denominator)


def assert_metric_match(
    reference,
    actual,
    context,
    keys=None,
    atol=2e-6,
):
    """任务二，实验一补充：检查历史指标与保存预测能否对应。"""
    keys = METRIC_KEYS if keys is None else keys"
    if int(reference["n"]) != int(actual["n"]):
        raise RuntimeError(
            f"{context}样本数不一致：metrics.json={reference['n']}，"
            f"逐样本预测重算={actual['n']}。"
        )

    for key in keys:
        left = reference.get(key)
        right = actual.get(key)

        if left is None or right is None:
            if left != right:
                raise RuntimeError(
                    f"{context}指标{key}不一致：{left} vs {right}"
                )
            continue

        if not np.isclose(float(left), float(right), atol=atol, rtol=atol):
            raise RuntimeError(
                f"{context}指标{key}不一致：{left} vs {right}。"
                "请勿将该历史基线用于配对比较。"
            )


# ==================== 三、历史未加权LORO基线核验 ====================

def validate_runtime_versions(config_record):
    """任务二，实验一补充：核对当前运行环境与历史baseline记录。"""
    current = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }

    mismatches = {
        key: {
            "historical": config_record.get(key),
            "current": value,
        }
        for key, value in current.items()
        if config_record.get(key) != value
    }

    if mismatches:
        raise RuntimeError(
            "当前软件版本与历史未加权LORO基线不一致，无法保证严格控制变量："
            + json.dumps(mismatches, ensure_ascii=False)
        )

    return current


def validate_baseline_config(config_record, expected_config, data_hash, held_out_region):
    """任务二，实验一补充：核对历史baseline的模型参数、数据和环境设置。"""
    if config_record.get("input_sha256") != data_hash:
        raise RuntimeError(
            f"{held_out_region}历史baseline的数据SHA256与当前数据不一致。"
            "不能直接复用该基线。"
        )

    if config_record.get("config") != expected_config.to_dict():
        raise RuntimeError(
            f"{held_out_region}历史baseline的训练配置与当前加权实验不一致。"
            "请检查模型参数和训练设置。"
        )

    if config_record.get("holdout_level") != "region":
        raise RuntimeError(f"{held_out_region}历史baseline不是留一地区实验。")

    if config_record.get("environment_granularity") != "region":
        raise RuntimeError(f"{held_out_region}历史baseline的环境粒度不是region。")

    if config_record.get("region_mapping") != PROVINCE_TO_REGION:
        raise RuntimeError(f"{held_out_region}历史baseline的地区映射与当前设置不一致。")

    if config_record.get("feature_names") != list(FEATURES):
        raise RuntimeError(f"{held_out_region}历史baseline的输入特征与当前设置不一致。")

    if config_record.get("label_encoding") != "0=no impairment;1=impairment":
        raise RuntimeError(f"{held_out_region}历史baseline的标签编码不一致。")

    return validate_runtime_versions(config_record)


def load_historical_baseline(
    baseline_root,
    held_out_region,
    target_split,
    expected_config,
    data_hash,
    summary_table,
):
    """任务二，实验一补充：读取并核验历史未加权MLP，不重新训练baseline。"""
    fold_dir = Path(baseline_root) / held_out_region / "baseline"

    required_files = [
        fold_dir / "config.json",
        fold_dir / "metrics.json",
        fold_dir / "training.json",
        fold_dir / "representations_test.csv",
    ]

    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "历史未加权LORO结果不完整，不能复用："
            + "; ".join(missing)
        )

    config_record = json.loads((fold_dir / "config.json").read_text(encoding="utf-8"))
    runtime_versions = validate_baseline_config(
        config_record,
        expected_config,
        data_hash,
        held_out_region,
    )

    training_record = json.loads(
        (fold_dir / "training.json").read_text(encoding="utf-8")
    )

    if training_record.get("selection_start_epoch") != 1:
        raise RuntimeError(
            f"{held_out_region}历史baseline的checkpoint选择起始轮次不是第1轮。"
        )

    if training_record.get("device") != expected_config.device:
        raise RuntimeError(f"{held_out_region}历史baseline训练设备不一致。")

    metrics_record = json.loads(
        (fold_dir / "metrics.json").read_text(encoding="utf-8")
    )["main"]

    predictions = pd.read_csv(
        fold_dir / "representations_test.csv",
        encoding="utf-8-sig",
    )

    required_columns = {
        "sample_id",
        "environment_code",
        "environment_group",
        "cognitive_impairment",
        "split",
        "seed",
        "y_prob",
    }

    if not required_columns.issubset(predictions.columns):
        raise ValueError(
            f"{held_out_region}历史逐样本预测缺少字段："
            f"{sorted(required_columns - set(predictions.columns))}"
        )

    predictions["sample_id"] = pd.to_numeric(
        predictions["sample_id"], errors="raise"
    ).astype(np.int64)

    predictions["environment_code"] = (
        predictions["environment_code"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )

    if predictions["sample_id"].duplicated().any():
        raise RuntimeError(f"{held_out_region}历史baseline存在重复sample_id。")

    if not (predictions["split"] == "test").all():
        raise RuntimeError(f"{held_out_region}历史预测文件包含非测试集样本。")

    if not (predictions["seed"].astype(int) == expected_config.seed).all():
        raise RuntimeError(f"{held_out_region}历史预测文件的随机种子不一致。")

    # 任务二，实验一补充：测试样本必须与当前held_out_split得到的目标样本完全一致。
    expected_ids = set(target_split.sample_ids.astype(np.int64).tolist())
    actual_ids = set(predictions["sample_id"].tolist())

    if expected_ids != actual_ids:
        raise RuntimeError(
            f"{held_out_region}历史baseline测试样本与当前数据划分不一致，"
            "不能进行配对比较。"
        )

    predictions = predictions.set_index("sample_id").loc[
        target_split.sample_ids.astype(np.int64)
    ].reset_index()

    expected_sites = target_split.site_env.astype(str)
    actual_sites = predictions["environment_code"].astype(str).to_numpy()

    if not np.array_equal(expected_sites, actual_sites):
        raise RuntimeError(f"{held_out_region}历史baseline监测点顺序或映射不一致。")

    if not (predictions["environment_group"].astype(str) == held_out_region).all():
        raise RuntimeError(f"{held_out_region}历史baseline包含其他地区样本。")

    actual_y = predictions["cognitive_impairment"].astype(int).to_numpy()
    expected_y = target_split.y.astype(int)

    if not np.array_equal(actual_y, expected_y):
        raise RuntimeError(f"{held_out_region}历史baseline测试标签与当前数据不一致。")

    baseline_probability = predictions["y_prob"].to_numpy(dtype=float)

    if not np.isfinite(baseline_probability).all():
        raise RuntimeError(f"{held_out_region}历史baseline预测概率存在非有限值。")

    # 任务二，实验一补充：从逐样本预测重算不受阈值影响的指标。
    # CSV预测概率保存为有限小数位，故不使用它重算阈值型指标作严格核验。
    prediction_metrics = metrics(expected_y, baseline_probability)
    assert_metric_match(
        metrics_record,
        prediction_metrics,
        f"{held_out_region}历史baseline逐样本预测",
        keys=PREDICTION_METRIC_KEYS,
        atol=1e-5,
    )

    summary_rows = summary_table.loc[
        (summary_table["held_out"].astype(str) == held_out_region)
        & (summary_table["model"].astype(str) == "baseline")
    ]

    if len(summary_rows) != 1:
        raise RuntimeError(
            f"历史summary.csv中{held_out_region}的baseline记录不是唯一一条。"
        )

    summary_record = summary_rows.iloc[0].to_dict()
    assert_metric_match(
        summary_record,
        metrics_record,
        f"{held_out_region}历史summary.csv",
    )

    # 任务二，实验一补充：核对历史per-site汇总与逐样本预测的样本量和阳性率。
    per_site_metrics = metrics_record.get("per_site", {})
    observed_sites = set(predictions["environment_code"].astype(str))
    if observed_sites != set(per_site_metrics):
        raise RuntimeError(
            f"{held_out_region}历史baseline的per-site指标与预测文件站点不一致。"
        )

    for site, site_metrics in per_site_metrics.items():
        site_rows = predictions.loc[
            predictions["environment_code"].astype(str) == str(site)
        ]
        site_positive_rate = float(
            site_rows["cognitive_impairment"].astype(int).mean()
        )
        if int(site_metrics["n"]) != len(site_rows):
            raise RuntimeError(
                f"{held_out_region}/{site}历史per-site样本数不一致。"
            )
        if not np.isclose(
            float(site_metrics["positive_rate"]),
            site_positive_rate,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"{held_out_region}/{site}历史per-site阳性率不一致。"
            )

    artifact_hashes = {
        path.name: sha256_file(path)
        for path in required_files
    }

    manifest = {
        "held_out_region": held_out_region,
        "baseline_directory": str(fold_dir),
        "baseline_retrained": False,
        "baseline_config": config_record["config"],
        "input_sha256": data_hash,
        "runtime_versions": runtime_versions,
        "best_epoch": int(training_record["best_epoch"]),
        "best_validation_auc": float(training_record["best_val_auc"]),
        "selection_start_epoch": int(training_record["selection_start_epoch"]),
        "test_sample_count": int(len(predictions)),
        "test_prediction_file": str(fold_dir / "representations_test.csv"),
        "artifact_sha256": artifact_hashes,
        "metrics_from_historical_metrics_json": metrics_record,
        "prediction_metrics_recomputed": prediction_metrics,
        "historical_metrics_verified": True,
    }

    return (
        baseline_probability,
        metrics_record,
        per_site_metrics,
        predictions,
        manifest,
    )


# ==================== 四、域分类器与密度比估计 ====================

def build_domain_classifier():
    """任务二，实验一补充：建立源域/目标域识别器，不使用健康结局标签。"""
    feature_pipeline = Pipeline(
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
            ("features", feature_pipeline, FEATURES),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    C=DOMAIN_CLASSIFIER_C,
                    max_iter=3000,
                    solver="liblinear",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def estimate_density_ratio(source_frame, target_frame, seed):
    """任务二，实验一补充：按监测点交叉拟合目标域/源域密度比。

    source_frame为源地区train+validation的无标签特征；
    target_frame为目标地区无标签特征。
    目标地区健康标签不会进入本函数。
    """
    selected_columns = FEATURES + [ENVIRONMENT, REGION, "sample_id"]

    source = source_frame[selected_columns].copy().reset_index(drop=True)
    target = target_frame[selected_columns].copy().reset_index(drop=True)

    if source["sample_id"].duplicated().any():
        raise ValueError("源域样本ID重复，无法追踪密度比权重。")

    if target["sample_id"].duplicated().any():
        raise ValueError("目标域样本ID重复，无法追踪目标域预测。")

    if set(source["sample_id"]).intersection(set(target["sample_id"])):
        raise ValueError("源域和目标域出现相同sample_id。")

    source["domain_label"] = 0
    target["domain_label"] = 1

    # 任务二，实验一补充：只合并19个特征、监测点和地区信息。
    domain_frame = pd.concat([source, target], ignore_index=True)

    domain_x = domain_frame[FEATURES].copy()
    domain_y = domain_frame["domain_label"].to_numpy(dtype=int)
    domain_sites = domain_frame[ENVIRONMENT].astype(str).to_numpy()
    domain_regions = domain_frame[REGION].astype(str).to_numpy()

    site_table = (
        domain_frame[[ENVIRONMENT, REGION]]
        .drop_duplicates()
        .sort_values([REGION, ENVIRONMENT])
    )

    if site_table[ENVIRONMENT].duplicated().any():
        raise ValueError("同一监测点映射到多个地区。")

    rng = np.random.RandomState(seed)
    sites_by_region = {}

    for region in ("east", "central", "west"):
        sites = (
            site_table.loc[
                site_table[REGION] == region,
                ENVIRONMENT,
            ]
            .astype(str)
            .tolist()
        )

        if len(sites) != N_SPLITS:
            raise ValueError(
                f"{region}地区监测点数量为{len(sites)}，"
                f"当前设计要求每个地区有{N_SPLITS}个监测点。"
            )

        rng.shuffle(sites)
        sites_by_region[region] = sites

    oof_probability = np.full(len(domain_frame), np.nan, dtype=float)
    oof_density_ratio = np.full(len(domain_frame), np.nan, dtype=float)
    fold_assignment = np.full(len(domain_frame), -1, dtype=int)
    fold_rows = []

    for fold_index in range(N_SPLITS):
        fold = fold_index + 1

        # 任务二，实验一补充：每折留出东、中、西各一个完整监测点。
        held_out_sites = {
            region: sites_by_region[region][fold_index]
            for region in ("east", "central", "west")
        }
        test_sites = set(held_out_sites.values())

        test_mask = np.isin(domain_sites, list(test_sites))
        train_indices = np.flatnonzero(~test_mask)
        test_indices = np.flatnonzero(test_mask)

        train_site_set = set(domain_sites[train_indices])
        test_site_set = set(domain_sites[test_indices])

        if train_site_set.intersection(test_site_set):
            raise RuntimeError(f"密度比估计第{fold}折出现监测点交叉泄漏。")

        if set(np.unique(domain_y[train_indices])) != {0, 1}:
            raise RuntimeError(f"密度比估计第{fold}折训练集缺少域类别。")

        if set(np.unique(domain_y[test_indices])) != {0, 1}:
            raise RuntimeError(f"密度比估计第{fold}折测试集缺少域类别。")

        model = build_domain_classifier()
        model.fit(domain_x.iloc[train_indices], domain_y[train_indices])

        probability = model.predict_proba(domain_x.iloc[test_indices])[:, 1]
        probability = np.clip(probability, 1e-6, 1 - 1e-6)

        n_source_fit = int((domain_y[train_indices] == 0).sum())
        n_target_fit = int((domain_y[train_indices] == 1).sum())

        # 任务二，实验一补充：校正域分类训练样本比例，得到目标域/源域密度比。
        density_ratio = (
            probability
            / (1 - probability)
            * (n_source_fit / n_target_fit)
        )

        oof_probability[test_indices] = probability
        oof_density_ratio[test_indices] = density_ratio
        fold_assignment[test_indices] = fold

        fold_auc = roc_auc_score(
            domain_y[test_indices],
            probability,
        )

        fold_rows.append(
            {
                "fold": fold,
                "train_site_count": len(train_site_set),
                "test_site_count": len(test_site_set),
                "train_sites": ";".join(sorted(train_site_set)),
                "test_sites": ";".join(sorted(test_site_set)),
                "test_east_site": held_out_sites["east"],
                "test_central_site": held_out_sites["central"],
                "test_west_site": held_out_sites["west"],
                "domain_train_source_n": n_source_fit,
                "domain_train_target_n": n_target_fit,
                "domain_test_auc": float(fold_auc),
            }
        )

    if np.isnan(oof_probability).any() or np.isnan(oof_density_ratio).any():
        raise RuntimeError("部分样本未获得折外密度比估计。")

    if (fold_assignment < 1).any():
        raise RuntimeError("部分样本未被分配到密度比交叉验证折。")

    domain_auc = roc_auc_score(domain_y, oof_probability)

    # 任务二，实验一补充：只为源域train+validation样本保留密度比。
    source_n = len(source)
    raw_weights = oof_density_ratio[:source_n]

    lower_bound = float(np.quantile(raw_weights, WEIGHT_LOWER_QUANTILE))
    upper_bound = float(np.quantile(raw_weights, WEIGHT_UPPER_QUANTILE))

    clipped_weights = np.clip(raw_weights, lower_bound, upper_bound)
    normalized_weights = clipped_weights / clipped_weights.mean()

    weight_table = pd.DataFrame(
        {
            "sample_id": source["sample_id"].to_numpy(),
            "environment_code": source[ENVIRONMENT].astype(str).to_numpy(),
            "region": source[REGION].astype(str).to_numpy(),
            "domain_probability_target_oof": oof_probability[:source_n],
            "density_ratio_raw": raw_weights,
            "weight_clipped": clipped_weights,
            "weight_normalized": normalized_weights,
        }
    )

    domain_predictions = pd.DataFrame(
        {
            "sample_id": domain_frame["sample_id"].to_numpy(),
            "environment_code": domain_sites,
            "region": domain_regions,
            "domain_label": domain_y,
            "fold": fold_assignment,
            "probability_target_oof": oof_probability,
            "density_ratio_oof": oof_density_ratio,
        }
    )

    diagnostics = {
        "domain_classifier_oof_auc": float(domain_auc),
        "source_sample_count": int(source_n),
        "target_sample_count": int(len(target)),
        "weight_clip_lower_quantile": WEIGHT_LOWER_QUANTILE,
        "weight_clip_upper_quantile": WEIGHT_UPPER_QUANTILE,
        "weight_clip_lower_bound": lower_bound,
        "weight_clip_upper_bound": upper_bound,
        "weight_min": float(normalized_weights.min()),
        "weight_q01": float(np.quantile(normalized_weights, 0.01)),
        "weight_median": float(np.median(normalized_weights)),
        "weight_q99": float(np.quantile(normalized_weights, 0.99)),
        "weight_max": float(normalized_weights.max()),
        "weight_mean": float(normalized_weights.mean()),
        "effective_sample_size_source_train_validation": effective_sample_size(
            normalized_weights
        ),
        "effective_sample_size_ratio_source_train_validation": (
            effective_sample_size(normalized_weights) / source_n
        ),
        "target_outcome_used_for_weight_estimation": False,
    }

    return (
        weight_table,
        domain_predictions,
        pd.DataFrame(fold_rows),
        diagnostics,
    )


# ==================== 五、分布平衡评价 ====================

def evaluate_distribution_balance(source_frame, target_frame, source_weights):
    """任务二，实验一补充：比较加权前后源域与目标域的19个特征分布。

    该函数只读取输入特征，不读取任何健康结局标签。
    缺失值作为独立类别参与分布比较。
    """
    source_weights = np.asarray(source_weights, dtype=float).reshape(-1)

    if len(source_weights) != len(source_frame):
        raise ValueError("分布平衡权重数量与源域样本数不一致。")

    rows = []

    for feature in FEATURES:
        source_values = source_frame[feature].map(normalize_category).to_numpy()
        target_values = target_frame[feature].map(normalize_category).to_numpy()

        categories = sorted(
            set(source_values).union(set(target_values)),
            key=lambda value: (value == "<MISSING>", value),
        )

        source_mask = {
            category: source_values == category for category in categories
        }
        target_mask = {
            category: target_values == category for category in categories
        }

        source_counts = np.asarray(
            [source_mask[category].sum() for category in categories],
            dtype=float,
        )
        weighted_source_counts = np.asarray(
            [
                source_weights[source_mask[category]].sum()
                for category in categories
            ],
            dtype=float,
        )
        target_counts = np.asarray(
            [target_mask[category].sum() for category in categories],
            dtype=float,
        )

        source_distribution = source_counts / source_counts.sum()
        weighted_distribution = (
            weighted_source_counts / weighted_source_counts.sum()
        )
        target_distribution = target_counts / target_counts.sum()

        js_before = js_divergence(source_counts, target_counts)
        js_after = js_divergence(weighted_source_counts, target_counts)

        max_difference_before = float(
            np.max(np.abs(source_distribution - target_distribution))
        )
        max_difference_after = float(
            np.max(np.abs(weighted_distribution - target_distribution))
        )

        rows.append(
            {
                "feature": feature,
                "js_before": js_before,
                "js_after": js_after,
                "js_reduction": js_before - js_after,
                "max_category_difference_before": max_difference_before,
                "max_category_difference_after": max_difference_after,
                "source_missing_rate": float(
                    (source_values == "<MISSING>").mean()
                ),
                "target_missing_rate": float(
                    (target_values == "<MISSING>").mean()
                ),
            }
        )

    return pd.DataFrame(rows)


# ==================== 六、预测评价与结果保存 ====================

def evaluate_by_site(test_split, probability, model_name, held_out_region):
    """任务二，实验一补充：计算目标地区各监测点的预测表现。"""
    rows = []

    for site in sorted(set(test_split.site_env.astype(str))):
        mask = test_split.site_env.astype(str) == site
        result = metrics(
            test_split.y[mask].astype(int),
            np.asarray(probability)[mask],
        )

        rows.append(
            {
                "held_out_region": held_out_region,
                "site_code": site,
                "model": model_name,
                **result,
            }
        )

    return rows


def run_region_fold(
    frame,
    held_out_region,
    config,
    data_hash,
    baseline_root,
    baseline_summary,
    output_dir,
):
    """任务二，实验一补充：复用历史baseline，只训练新的加权MLP。"""
    output_dir.mkdir(parents=True, exist_ok=False)

    # 任务二，实验一补充：严格复用历史LORO的held_out_split和seed。
    train_frame, validation_frame, target_frame = held_out_split(
        frame,
        held_out_region,
        config.seed,
        column=REGION,
    )

    # 任务二，实验一补充：所有预处理参数仅在源域训练集拟合。
    splits, preprocessor = prepare_splits(
        [train_frame, validation_frame, target_frame],
        environment_column=REGION,
    )
    train_split, validation_split, target_split = splits

    # 任务二，实验一补充：读取历史未加权预测，并逐样本核验测试集一致。
    (
        probability_unweighted,
        result_unweighted,
        baseline_per_site_metrics,
        baseline_predictions,
        baseline_manifest,
    ) = load_historical_baseline(
        baseline_root,
        held_out_region,
        target_split,
        config,
        data_hash,
        baseline_summary,
    )

    # 任务二，实验一补充：使用源域train+validation和目标域无标签X估计密度比。
    source_unlabeled = pd.concat(
        [train_frame, validation_frame],
        axis=0,
    ).sort_values("sample_id")

    (
        weight_table,
        domain_predictions,
        domain_fold_metrics,
        weight_diagnostics,
    ) = estimate_density_ratio(
        source_unlabeled,
        target_frame,
        config.seed,
    )

    weight_map = weight_table.set_index("sample_id")["weight_normalized"]

    # 任务二，实验一补充：只将训练样本权重传入fit，不对验证集加权。
    train_weights = weight_map.loc[train_split.sample_ids].to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(train_weights).all() or (train_weights <= 0).any():
        raise RuntimeError("训练样本权重存在非有限值或非正值。")

    # 任务二，实验一补充：只调整训练损失；验证AUC和验证loss保持未加权。
    # 这样checkpoint选择规则与历史未加权baseline一致。
    model_weighted, training_weighted = fit(
        train_split,
        validation_split,
        config,
        sample_weights=train_weights,
    )

    if any(
        record.get("validation_weighted", False)
        for record in training_weighted["history"]
    ):
        raise RuntimeError("验证集意外使用了样本权重，控制变量条件被破坏。")

    device = get_device(config.device)
    probability_weighted = predict(
        model_weighted,
        target_split.x,
        device,
    )

    result_weighted = metrics(
        target_split.y.astype(int),
        probability_weighted,
    )

    # 任务二，实验一补充：核对历史baseline与当前目标测试样本的顺序。
    if len(probability_unweighted) != len(target_split.y):
        raise RuntimeError("历史baseline预测数量与当前目标测试集不一致。")

    # 任务二，实验一补充：保存同一批目标样本的加权前后预测概率。
    predictions = pd.DataFrame(
        {
            "sample_id": target_split.sample_ids,
            "site_code": target_split.site_env,
            "held_out_region": held_out_region,
            "cognitive_impairment": target_split.y.astype(int),
            "probability_unweighted_reused": probability_unweighted,
            "probability_weighted": probability_weighted,
        }
    )
    predictions.to_csv(
        output_dir / "target_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 任务二，实验一补充：保存样本权重、域识别折外预测和分布平衡结果。
    weight_table.to_csv(
        output_dir / "source_sample_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )
    domain_predictions.to_csv(
        output_dir / "domain_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    domain_fold_metrics.to_csv(
        output_dir / "domain_cv_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )

    balance_table = evaluate_distribution_balance(
        source_unlabeled,
        target_frame,
        weight_table["weight_normalized"].to_numpy(dtype=float),
    )
    balance_table.to_csv(
        output_dir / "feature_balance_before_after.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 任务二，实验一补充：记录实际用于模型训练的权重质量。
    weight_diagnostics["effective_sample_size_train"] = effective_sample_size(
        train_weights
    )
    weight_diagnostics["effective_sample_size_ratio_train"] = (
        weight_diagnostics["effective_sample_size_train"] / len(train_weights)
    )
    weight_diagnostics["train_weight_mean"] = float(train_weights.mean())
    weight_diagnostics["train_weight_std"] = float(train_weights.std())
    weight_diagnostics["mean_feature_js_before"] = float(
        balance_table["js_before"].mean()
    )
    weight_diagnostics["mean_feature_js_after"] = float(
        balance_table["js_after"].mean()
    )
    weight_diagnostics["mean_feature_js_reduction"] = float(
        balance_table["js_reduction"].mean()
    )
    weight_diagnostics["features_with_js_reduction"] = int(
        (balance_table["js_reduction"] > 0).sum()
    )

    # 任务二，实验一补充：保存加权模型和训练记录，不重复保存未加权模型。
    torch.save(
        {
            "state_dict": model_weighted.state_dict(),
            "config": config.to_dict(),
            "best_epoch": training_weighted["best_epoch"],
            "model": "baseline_mlp_importance_weighted",
            "weight_method": "cross_fitted_domain_classifier_density_ratio",
        },
        output_dir / "model_weighted.pt",
    )

    with (output_dir / "preprocessor.pkl").open("wb") as stream:
        pickle.dump(preprocessor, stream)

    write_json(
        output_dir / "training_weighted.json",
        training_weighted,
    )
    write_json(
        output_dir / "weight_diagnostics.json",
        weight_diagnostics,
    )
    write_json(
        output_dir / "baseline_reuse_manifest.json",
        baseline_manifest,
    )
    write_json(
        output_dir / "metrics_comparison.json",
        {
            "unweighted_baseline_reused": result_unweighted,
            "importance_weighted": result_weighted,
            "baseline_retrained": False,
            "validation_checkpoint_selection_weighted": False,
        },
    )

    # 任务二，实验一补充：总体指标比较。
    metric_rows = [
        {
            "held_out_region": held_out_region,
            "model": "MLP_unweighted_historical_reused",
            "baseline_retrained": False,
            "best_epoch": baseline_manifest["best_epoch"],
            **result_unweighted,
        },
        {
            "held_out_region": held_out_region,
            "model": "MLP_importance_weighted",
            "baseline_retrained": False,
            "best_epoch": training_weighted["best_epoch"],
            **result_weighted,
        },
    ]

    # 任务二，实验一补充：监测点层面使用同一测试样本做配对比较。
    site_metric_rows = []
    for site, site_metrics in baseline_per_site_metrics.items():
        site_metric_rows.append(
            {
                "held_out_region": held_out_region,
                "site_code": str(site),
                "model": "MLP_unweighted_historical_reused",
                **site_metrics,
            }
        )
    site_metric_rows.extend(
        evaluate_by_site(
            target_split,
            probability_weighted,
            "MLP_importance_weighted",
            held_out_region,
        )
    )

    # 任务二，实验一补充：绘制源域权重分布。
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        weight_table["weight_normalized"],
        bins=35,
        edgecolor="black",
        alpha=0.8,
    )
    ax.set_xlabel("Normalized source-sample weight")
    ax.set_ylabel("Count")
    ax.set_title(f"Importance weights for held-out {held_out_region}")
    fig.tight_layout()
    fig.savefig(output_dir / "weight_distribution.png", dpi=300)
    plt.close(fig)

    # 任务二，实验一补充：绘制加权前后特征JS距离。
    balance_plot = balance_table.nlargest(12, "js_before").sort_values("js_before")
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(balance_plot))
    width = 0.38
    ax.barh(
        positions - width / 2,
        balance_plot["js_before"],
        height=width,
        label="Before weighting",
    )
    ax.barh(
        positions + width / 2,
        balance_plot["js_after"],
        height=width,
        label="After weighting",
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(balance_plot["feature"])
    ax.set_xlabel("Jensen-Shannon Divergence (bits)")
    ax.set_title(f"Feature distribution balance: {held_out_region}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "feature_balance_before_after.png", dpi=300)
    plt.close(fig)

    # 任务二，实验一补充：绘制目标地区预测概率对照。
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        probability_unweighted,
        bins=30,
        alpha=0.55,
        label="Historical unweighted MLP",
    )
    ax.hist(
        probability_weighted,
        bins=30,
        alpha=0.55,
        label="Importance-weighted MLP",
    )
    ax.set_xlabel("Predicted probability of cognitive impairment")
    ax.set_ylabel("Count")
    ax.set_title(f"Target prediction distribution: {held_out_region}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "target_prediction_distribution.png", dpi=300)
    plt.close(fig)

    write_json(
        output_dir / "fold_metadata.json",
        {
            "held_out_region": held_out_region,
            "source_regions": sorted(set(train_frame[REGION].astype(str))),
            "source_train_n": len(train_frame),
            "source_validation_n": len(validation_frame),
            "target_n": len(target_frame),
            "target_site_count": int(target_frame[ENVIRONMENT].nunique()),
            "seed": config.seed,
            "model_config": config.to_dict(),
            "preprocessor_fit_on": "source training split only",
            "target_covariates_used_for_weight_estimation": True,
            "target_outcomes_used_for_weight_estimation": False,
            "target_outcomes_used_for_training": False,
            "target_outcomes_used_only_for_final_evaluation": True,
            "baseline_retrained": False,
            "baseline_best_epoch": baseline_manifest["best_epoch"],
            "weighted_best_epoch": training_weighted["best_epoch"],
            "validation_sample_weights_used": False,
            "checkpoint_selection": "unweighted source validation AUC, matching historical baseline",
            "weight_diagnostics": weight_diagnostics,
        },
    )

    return metric_rows, site_metric_rows, weight_diagnostics, balance_table


# ==================== 七、主程序 ====================

def main():
    """任务二，实验一补充：复用历史未加权LORO，只训练加权MLP。"""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--baseline-output",
        type=Path,
        default=DEFAULT_BASELINE_OUTPUT,
        help="历史未加权LORO结果目录，不会重新训练该baseline。",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--encoding", default="gbk")
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(f"找不到输入数据：{args.data}")
    if not args.config.is_file():
        raise FileNotFoundError(f"找不到模型配置：{args.config}")
    if not args.baseline_output.is_dir():
        raise FileNotFoundError(
            f"找不到历史未加权LORO结果：{args.baseline_output}"
        )
    if not (args.baseline_output / "summary.csv").is_file():
        raise FileNotFoundError("历史LORO结果缺少summary.csv，无法复用baseline。")

    # 任务二，实验一补充：读取历史使用的原始数据并记录SHA256。
    data_hash = sha256_file(args.data)

    frame = read_raw(args.data, encoding=args.encoding)
    frame = attach_regions(frame)

    base_config_values = json.loads(
        args.config.read_text(encoding="utf-8")
    )
    base_config = TrainConfig(**base_config_values)

    if base_config.model != "v5":
        raise ValueError("历史LORO配置应以v5为主模型，再通过原入口切换baseline。")

    if base_config.environment_granularity != "region":
        raise ValueError("当前实验必须使用region环境粒度。")

    # 任务二，实验一补充：严格复现历史loeo_config_for_model的baseline配置。
    config = replace(
        base_config,
        model="baseline",
        selection_start_epoch=None,
    )

    if config.seed != RANDOM_SEED:
        raise ValueError("历史baseline使用seed=42，当前配置不一致。")

    if config.device != "cpu" or config.threads != 2:
        raise ValueError("历史baseline使用CPU和2个线程，当前配置不一致。")

    baseline_summary = pd.read_csv(
        args.baseline_output / "summary.csv",
        encoding="utf-8-sig",
    )

    # 任务二，实验一补充：确认历史summary中三个地区均有baseline结果。
    for region in ("east", "central", "west"):
        count = len(
            baseline_summary.loc[
                (baseline_summary["held_out"].astype(str) == region)
                & (baseline_summary["model"].astype(str) == "baseline")
            ]
        )
        if count != 1:
            raise RuntimeError(
                f"历史summary.csv中{region}的baseline记录数量为{count}，应为1。"
            )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output / run_id
    output_root.mkdir(parents=True, exist_ok=False)

    all_metrics = []
    all_site_metrics = []
    all_weight_diagnostics = []
    all_balance_rows = []
    baseline_manifests = []

    print("=" * 72)
    print("任务二，实验一补充：历史LORO基线复用 + 协变量偏移加权")
    print("=" * 72)
    print(f"输入数据：{args.data}")
    print(f"数据SHA256：{data_hash}")
    print(f"历史未加权基线：{args.baseline_output}")
    print(f"样本数：{len(frame)}")
    print(f"输入特征数：{len(FEATURES)}")
    print(f"模型配置：{config.to_dict()}")
    print("未加权MLP：直接复用历史预测，不重新训练。")
    print("加权MLP：只新增训练。")
    print("目标地区X可用于密度比估计；目标地区Y仅用于最终评价。")

    for held_out_region in ("east", "central", "west"):
        print(f"\n开始留出地区：{held_out_region}")

        fold_output = output_root / held_out_region

        (
            metric_rows,
            site_rows,
            weight_diagnostics,
            balance_table,
        ) = run_region_fold(
            frame,
            held_out_region,
            config,
            data_hash,
            args.baseline_output,
            baseline_summary,
            fold_output,
        )

        all_metrics.extend(metric_rows)
        all_site_metrics.extend(site_rows)

        balance_table.insert(0, "held_out_region", held_out_region)
        all_balance_rows.extend(balance_table.to_dict(orient="records"))

        all_weight_diagnostics.append(
            {
                "held_out_region": held_out_region,
                **weight_diagnostics,
            }
        )

        manifest = json.loads(
            (fold_output / "baseline_reuse_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        baseline_manifests.append(manifest)

        for row in metric_rows:
            print(
                f"{row['model']}: "
                f"AUC={row['auc']:.4f}, "
                f"AUPRC={row['auprc']:.4f}, "
                f"Brier={row['brier']:.4f}"
            )

        print(
            "训练集权重ESS比例："
            f"{weight_diagnostics['effective_sample_size_ratio_train']:.3f}"
        )
        print(
            "平均特征JS："
            f"{weight_diagnostics['mean_feature_js_before']:.4f}"
            " -> "
            f"{weight_diagnostics['mean_feature_js_after']:.4f}"
        )

    metrics_table = pd.DataFrame(all_metrics)
    site_table = pd.DataFrame(all_site_metrics)
    weight_table = pd.DataFrame(all_weight_diagnostics)
    balance_summary_table = pd.DataFrame(all_balance_rows)

    metrics_table.to_csv(
        output_root / "region_model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    site_table.to_csv(
        output_root / "site_model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    weight_table.to_csv(
        output_root / "region_weight_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    balance_summary_table.to_csv(
        output_root / "region_feature_balance_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    write_json(
        output_root / "baseline_reuse_manifest.json",
        {
            "baseline_retrained": False,
            "baseline_output_root": str(args.baseline_output.resolve()),
            "baseline_folds": baseline_manifests,
        },
    )

    # 任务二，实验一补充：绘制三个留出地区的历史baseline与加权MLP比较。
    auc_pivot = metrics_table.pivot(
        index="held_out_region",
        columns="model",
        values="auc",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    auc_pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("AUC")
    ax.set_xlabel("Held-out region")
    ax.set_title("Historical Unweighted vs Importance-weighted MLP")
    ax.set_ylim(0, 1)
    ax.legend(title="Model")
    fig.tight_layout()
    fig.savefig(output_root / "region_auc_comparison.png", dpi=300)
    plt.close(fig)

    # 任务二，实验一补充：记录实验版本、历史基线和无监督适配边界。
    metadata = {
        "experiment": "Task 2 Experiment 1B - Covariate Shift Reweighting",
        "run_id": run_id,
        "input_file": str(args.data.resolve()),
        "input_sha256": data_hash,
        "sample_count": int(len(frame)),
        "features": list(FEATURES),
        "region_mapping": PROVINCE_TO_REGION,
        "model_config": config.to_dict(),
        "baseline_output": str(args.baseline_output.resolve()),
        "baseline_retrained": False,
        "baseline_predictions_reused": True,
        "baseline_validation_selection": "unweighted validation AUC",
        "weighted_model_validation_selection": "unweighted validation AUC",
        "validation_sample_weights_used": False,
        "weight_method": "cross-fitted domain-classifier density ratio",
        "domain_classifier": "L2 Logistic Regression",
        "domain_classifier_C": DOMAIN_CLASSIFIER_C,
        "domain_cross_validation": (
            "4 folds; each fold holds out one complete site from east, central and west"
        ),
        "weight_clipping_quantiles": [
            WEIGHT_LOWER_QUANTILE,
            WEIGHT_UPPER_QUANTILE,
        ],
        "target_covariates_used_for_weight_estimation": True,
        "target_outcome_used_for_weight_estimation": False,
        "target_outcome_used_for_training": False,
        "target_outcome_used_only_for_final_evaluation": True,
        "adaptation_setting": "transductive unsupervised domain adaptation",
        "assumption": (
            "Importance weighting primarily addresses covariate shift and requires "
            "adequate source-target support and approximately stable P(Y|X)."
        ),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
    }

    write_json(output_root / "experiment_metadata.json", metadata)

    print("\n实验完成。")
    print(f"结果保存位置：{output_root}")
    print("\n历史未加权baseline已复用，未重新训练。")
    print("加权模型总体比较：")
    print(
        metrics_table[
            [
                "held_out_region",
                "model",
                "auc",
                "auprc",
                "brier",
                "balanced_accuracy",
                "best_epoch",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

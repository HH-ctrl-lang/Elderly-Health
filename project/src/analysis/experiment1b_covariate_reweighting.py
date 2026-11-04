"""任务二，实验一补充：基于密度比加权改善跨地区训练分布可比性。

研究目的：
1. 在留一地区设置下，使用目标地区无标签特征估计密度比；
2. 仅对源地区训练样本的预测损失加权，使训练分布更接近目标地区；
3. 比较未加权MLP与加权MLP在三个留出地区的预测表现；
4. 检查权重重叠、极端权重和有效样本量，避免把不稳定加权误判为改善。

重要边界：
- 目标地区的认知功能受损标签不参与密度比估计或模型训练；
- 目标地区标签仅在训练结束后用于最终评价；
- 本实验检验的是协变量偏移修正，不证明因果关系；
- 只有在特征-结局条件关系近似稳定且具有足够分布重叠时，密度比加权才有合理依据。
"""

import hashlib
import json
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


# ==================== 二、地区识别与密度比估计 ====================

def build_domain_classifier():
    """任务二，实验一补充：建立源域/目标域识别器。

    只使用19个输入特征，不使用认知功能受损标签。
    类别编码按类别处理，避免假定等级编码之间等距。
    """
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


def normalize_category(value):
    """任务二，实验一补充：将数值编码转换为类别标签，显式保留缺失类别。"""
    if pd.isna(value):
        return "<MISSING>"

    number = float(value)
    if number.is_integer():
        return str(int(number))

    return format(number, ".8g")


def js_divergence(count_a, count_b):
    """任务二，实验一补充：计算类别分布的Jensen-Shannon散度。"""
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


def evaluate_distribution_balance(source_frame, target_frame, source_weights):
    """任务二，实验一补充：比较加权前后源域与目标域的特征分布。

    该函数只比较输入特征分布，不读取任何健康结局标签。
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


def estimate_density_ratio(source_frame, target_frame, seed):
    """任务二，实验一补充：按监测点交叉拟合目标域/源域密度比。

    source_frame由源地区的训练集和验证集组成；
    target_frame只提供目标地区19个无标签输入特征。

    每折从东、中、西各留出一个完整监测点，使用其他9个监测点拟合
    域分类器，再对留出监测点生成折外目标域概率，降低个体级伪重复。
    """
    source_columns = FEATURES + [ENVIRONMENT, REGION, "sample_id"]
    target_columns = FEATURES + [ENVIRONMENT, REGION, "sample_id"]

    source = source_frame[source_columns].copy().reset_index(drop=True)
    target = target_frame[target_columns].copy().reset_index(drop=True)

    if source["sample_id"].duplicated().any():
        raise ValueError("源域样本ID重复，无法追踪密度比权重。")
    if target["sample_id"].duplicated().any():
        raise ValueError("目标域样本ID重复，无法追踪目标域预测。")

    source["domain_label"] = 0
    target["domain_label"] = 1

    # 任务二，实验一补充：只合并特征和监测点信息，不读取目标域健康标签。
    domain_frame = pd.concat(
        [source, target],
        ignore_index=True,
    )

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

    sites_by_region = {}
    rng = np.random.RandomState(seed)

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

    base_model = build_domain_classifier()

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
        model.fit(
            domain_x.iloc[train_indices],
            domain_y[train_indices],
        )

        probability = model.predict_proba(
            domain_x.iloc[test_indices]
        )[:, 1]

        probability = np.clip(probability, 1e-6, 1 - 1e-6)

        n_source_fit = int((domain_y[train_indices] == 0).sum())
        n_target_fit = int((domain_y[train_indices] == 1).sum())

        # 任务二，实验一补充：由域分类概率换算密度比。
        # p(target|x)/p(source|x) 需要乘以训练域样本量比例进行先验校正。
        density_ratio = (
            probability
            / (1 - probability)
            * (n_source_fit / n_target_fit)
        )

        oof_probability[test_indices] = probability
        oof_density_ratio[test_indices] = density_ratio
        fold_assignment[test_indices] = fold

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
            }
        )

    if np.isnan(oof_probability).any() or np.isnan(oof_density_ratio).any():
        raise RuntimeError("部分样本未获得折外密度比估计。")

    if (fold_assignment < 1).any():
        raise RuntimeError("部分样本未被分配到密度比交叉验证折。")

    domain_auc = roc_auc_score(domain_y, oof_probability)

    # 任务二，实验一补充：仅提取源域样本的密度比用于预测模型训练。
    source_n = len(source)
    raw_weights = oof_density_ratio[:source_n]

    lower_bound = float(np.quantile(raw_weights, WEIGHT_LOWER_QUANTILE))
    upper_bound = float(np.quantile(raw_weights, WEIGHT_UPPER_QUANTILE))

    clipped_weights = np.clip(
        raw_weights,
        lower_bound,
        upper_bound,
    )

    normalized_weights = clipped_weights / clipped_weights.mean()

    effective_sample_size = (
        normalized_weights.sum() ** 2
        / np.square(normalized_weights).sum()
    )

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
        "effective_sample_size": float(effective_sample_size),
        "effective_sample_size_ratio": float(effective_sample_size / source_n),
        "target_outcome_used_for_weight_estimation": False,
    }

    return (
        weight_table,
        domain_predictions,
        pd.DataFrame(fold_rows),
        diagnostics,
    )


# ==================== 三、训练与评价 ====================

def write_json(path, value):
    """任务二，实验一补充：保存可复核的JSON实验记录。"""
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def evaluate_predictions(test_split, probability):
    """任务二，实验一补充：计算目标地区总体预测指标。"""
    return metrics(
        test_split.y.astype(int),
        np.asarray(probability, dtype=float),
    )


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


def run_region_fold(frame, held_out_region, config, output_dir, seed):
    """任务二，实验一补充：运行一个留一地区的配对加权实验。"""
    output_dir.mkdir(parents=True, exist_ok=False)

    train_frame, validation_frame, target_frame = held_out_split(
        frame,
        held_out_region,
        seed,
        column=REGION,
    )

    # 任务二，实验一补充：权重估计可使用源域训练/验证特征和目标域无标签特征。
    # 目标域认知受损标签不会传入密度比估计函数。
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
        seed,
    )

    # 任务二，实验一补充：建立样本ID到密度比权重的映射。
    weight_map = weight_table.set_index("sample_id")["weight_normalized"]

    # 任务二，实验一补充：先验证加权是否真正缩小了特征分布距离。
    source_balance_weights = weight_map.loc[
        source_unlabeled["sample_id"]
    ].to_numpy(dtype=float)
    balance_table = evaluate_distribution_balance(
        source_unlabeled,
        target_frame,
        source_balance_weights,
    )
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

    # 任务二，实验一补充：所有预处理参数仅在源域训练集拟合。
    splits, preprocessor = prepare_splits(
        [train_frame, validation_frame, target_frame],
        environment_column=REGION,
    )
    train_split, validation_split, target_split = splits

    train_weights = weight_map.loc[train_split.sample_ids].to_numpy(
        dtype=np.float32
    )
    validation_weights = weight_map.loc[validation_split.sample_ids].to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(train_weights).all():
        raise RuntimeError("训练样本权重存在缺失或非有限值。")
    if not np.isfinite(validation_weights).all():
        raise RuntimeError("验证样本权重存在缺失或非有限值。")

    # 任务二，实验一补充：先运行原始未加权MLP作为配对对照。
    model_unweighted, training_unweighted = fit(
        train_split,
        validation_split,
        config,
    )

    # 任务二，实验一补充：再运行密度比加权MLP。
    model_weighted, training_weighted = fit(
        train_split,
        validation_split,
        config,
        sample_weights=train_weights,
        validation_sample_weights=validation_weights,
    )

    device = get_device(config.device)

    probability_unweighted = predict(
        model_unweighted,
        target_split.x,
        device,
    )
    probability_weighted = predict(
        model_weighted,
        target_split.x,
        device,
    )

    result_unweighted = evaluate_predictions(
        target_split,
        probability_unweighted,
    )
    result_weighted = evaluate_predictions(
        target_split,
        probability_weighted,
    )

    # 任务二，实验一补充：保存每个目标样本的配对预测概率。
    predictions = pd.DataFrame(
        {
            "sample_id": target_split.sample_ids,
            "site_code": target_split.site_env,
            "held_out_region": held_out_region,
            "cognitive_impairment": target_split.y.astype(int),
            "probability_unweighted": probability_unweighted,
            "probability_weighted": probability_weighted,
        }
    )
    predictions.to_csv(
        output_dir / "target_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 任务二，实验一补充：保存源域权重、域分类折外预测及折分配。
    weight_table.to_csv(
        output_dir / "source_sample_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )
    balance_table.to_csv(
        output_dir / "feature_balance_before_after.csv",
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

    # 任务二，实验一补充：保存模型训练记录。
    torch.save(
        {
            "state_dict": model_unweighted.state_dict(),
            "config": config.to_dict(),
            "best_epoch": training_unweighted["best_epoch"],
            "model": "baseline_mlp_unweighted",
        },
        output_dir / "model_unweighted.pt",
    )
    torch.save(
        {
            "state_dict": model_weighted.state_dict(),
            "config": config.to_dict(),
            "best_epoch": training_weighted["best_epoch"],
            "model": "baseline_mlp_importance_weighted",
        },
        output_dir / "model_weighted.pt",
    )

    with (output_dir / "preprocessor.pkl").open("wb") as stream:
        import pickle

        pickle.dump(preprocessor, stream)

    write_json(
        output_dir / "training_unweighted.json",
        training_unweighted,
    )
    write_json(
        output_dir / "training_weighted.json",
        training_weighted,
    )
    write_json(
        output_dir / "weight_diagnostics.json",
        weight_diagnostics,
    )

    # 任务二，实验一补充：保存总体及监测点层面的指标。
    metric_rows = [
        {
            "held_out_region": held_out_region,
            "model": "MLP_unweighted",
            **result_unweighted,
        },
        {
            "held_out_region": held_out_region,
            "model": "MLP_importance_weighted",
            **result_weighted,
        },
    ]

    site_metric_rows = []
    site_metric_rows.extend(
        evaluate_by_site(
            target_split,
            probability_unweighted,
            "MLP_unweighted",
            held_out_region,
        )
    )
    site_metric_rows.extend(
        evaluate_by_site(
            target_split,
            probability_weighted,
            "MLP_importance_weighted",
            held_out_region,
        )
    )

    # 任务二，实验一补充：绘制样本权重分布。
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

    # 任务二，实验一补充：绘制加权前后特征JS距离对照。
    balance_plot = balance_table.sort_values("js_before", ascending=False).head(12)
    balance_plot = balance_plot.sort_values("js_before")

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
        label="Unweighted MLP",
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
            "source_regions": sorted(
                set(train_frame[REGION].astype(str))
            ),
            "source_train_n": len(train_frame),
            "source_validation_n": len(validation_frame),
            "target_n": len(target_frame),
            "target_site_count": int(target_frame[ENVIRONMENT].nunique()),
            "target_covariates_used_for_weight_estimation": True,
            "target_outcomes_used_for_weight_estimation": False,
            "target_outcomes_used_for_training": False,
            "target_outcomes_used_only_for_final_evaluation": True,
            "train_best_epoch_unweighted": training_unweighted["best_epoch"],
            "train_best_epoch_weighted": training_weighted["best_epoch"],
            "weight_diagnostics": weight_diagnostics,
        },
    )

    return metric_rows, site_metric_rows, weight_diagnostics, balance_table


# ==================== 四、主程序与实验记录 ====================

def main():
    """任务二，实验一补充：执行三个地区的配对密度比加权实验。"""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--held-out",
        choices=["east", "central", "west"],
        default=None,
        help="默认依次运行东、中、西三个留出地区。",
    )
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(f"找不到输入数据：{args.data}")
    if not args.config.is_file():
        raise FileNotFoundError(f"找不到模型配置：{args.config}")
    if args.output.exists() and not args.output.is_dir():
        raise NotADirectoryError(f"输出路径不是目录：{args.output}")

    frame = read_raw(args.data, encoding=args.encoding)
    frame = attach_regions(frame)

    config_values = json.loads(args.config.read_text(encoding="utf-8"))
    base_config = TrainConfig(**config_values)

    # 任务二，实验一补充：使用原项目MLP baseline配置，不修改原模型文件。
    config = replace(
        base_config,
        model="baseline",
        selection_start_epoch=None,
        seed=args.seed,
        epochs=args.epochs,
        device=args.device,
        threads=args.threads,
        environment_granularity="region",
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output / run_id
    output_root.mkdir(parents=True, exist_ok=False)

    held_out_regions = (
        [args.held_out]
        if args.held_out
        else ["east", "central", "west"]
    )

    all_metrics = []
    all_site_metrics = []
    all_weight_diagnostics = []
    all_balance_rows = []

    print("=" * 70)
    print("任务二，实验一补充：跨地区协变量偏移加权实验")
    print("=" * 70)
    print(f"输入数据：{args.data}")
    print(f"样本数：{len(frame)}")
    print(f"特征数：{len(FEATURES)}")
    print(f"模型：{config.model}")
    print(f"训练轮数：{config.epochs}")
    print("目标地区健康结局仅用于最终评价，不参与权重估计。")

    for held_out_region in held_out_regions:
        print(f"\n开始留出地区：{held_out_region}")

        fold_output = output_root / held_out_region

        metric_rows, site_rows, weight_diagnostics, balance_table = run_region_fold(
            frame,
            held_out_region,
            config,
            fold_output,
            args.seed,
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

        for row in metric_rows:
            print(
                f"{row['model']}: "
                f"AUC={row['auc']:.4f}, "
                f"AUPRC={row['auprc']:.4f}, "
                f"Brier={row['brier']:.4f}"
            )

        print(
            "权重ESS比例："
            f"{weight_diagnostics['effective_sample_size_ratio']:.3f}"
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

    # 任务二，实验一补充：绘制三个留出地区的AUC配对比较。
    auc_pivot = metrics_table.pivot(
        index="held_out_region",
        columns="model",
        values="auc",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    auc_pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("AUC")
    ax.set_xlabel("Held-out region")
    ax.set_title("Unweighted vs Importance-weighted MLP")
    ax.set_ylim(0, 1)
    ax.legend(title="Model")
    fig.tight_layout()
    fig.savefig(output_root / "region_auc_comparison.png", dpi=300)
    plt.close(fig)

    # 任务二，实验一补充：记录输入数据版本和方法假设。
    metadata = {
        "experiment": "Task 2 Experiment 1B - Covariate Shift Reweighting",
        "run_id": run_id,
        "input_file": str(args.data.resolve()),
        "input_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "sample_count": int(len(frame)),
        "features": list(FEATURES),
        "region_mapping": PROVINCE_TO_REGION,
        "model": config.to_dict(),
        "comparison": [
            "MLP_unweighted",
            "MLP_importance_weighted",
        ],
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
        "target_covariates_used": True,
        "target_outcome_used_for_weight_estimation": False,
        "target_outcome_used_for_training": False,
        "target_outcome_used_only_for_final_evaluation": True,
        "assumption": (
            "Importance weighting primarily addresses covariate shift and requires "
            "adequate source-target support and approximately stable P(Y|X)."
        ),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    }

    write_json(output_root / "experiment_metadata.json", metadata)

    print("\n实验完成。")
    print(f"结果保存位置：{output_root}")
    print("\n总体比较：")
    print(
        metrics_table[
            [
                "held_out_region",
                "model",
                "auc",
                "auprc",
                "brier",
                "balanced_accuracy",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

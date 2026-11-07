"""任务三 第2步：V5因果/非稳定表征空间诊断（复用历史结果，不训练）。

核心问题：
1. zc / zs 表征是否出现低有效秩、ReLU零激活或真正的整体退化迹象？
2. 目标监测点相对同一LORO折源域的表征分布是否偏移？
3. V5最终联合预测头是否比仅使用zc的预测头获得额外信息？
4. 220202等失败站点的zc/zs表征与预测信息分配有何特点？

严格边界：
- 每个LORO折的潜在坐标系由不同模型产生，只在同一折内比较source与target；
  不直接比较不同折的zc_0或zs_0等坐标。
- 有效秩较低不等于表征崩溃；同时检查活动维度、零激活率、方差谱、
  源域/目标域状态和预测头表现。
- 阳性/阴性表征分离是观察性统计描述，不是因果效应。
- 不导出sample_id、逐人预测或逐人潜在表示。
- 不训练或微调任何模型，只读取历史representations CSV。
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parent
MODEL_ROOT = ROOT / "outputs" / "v2-region-loro-selection91-42"
OUTPUT_DIR = (
    ROOT / "outputs" / "analysis_2611"
    / "task3_step2_v5_representation_diagnostics"
)
REGIONS = ("east", "central", "west")
BRANCHES = ("zc", "zs")
LATENT_DIM = 32
EXPECTED_TOTAL_SAMPLES = 4769
EXPECTED_SITE_COUNT = 12
EXPECTED_DATA_SHA256 = (
    "969f0c1c82faafcfa402f37572e4a8ceb324492062cf1e590465f82dc6b5b6de"
)
VARIANCE_EPSILON = 1e-4
ZERO_EPSILON = 1e-8
RIDGE = 1e-5


# ==================== 一、文件与输入核验 ====================

def sha256_file(path):
    """任务三 第2步：计算历史输入文件哈希。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """任务三 第2步：保存分析元数据。"""
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def latent_columns(branch):
    """任务三 第2步：获取固定32维潜在表征列。"""
    return [f"{branch}_{index}" for index in range(LATENT_DIM)]


def read_representation(path, expected_split, held_out_region):
    """任务三 第2步：读取并核验一个历史表征文件。"""
    if not path.is_file():
        raise FileNotFoundError(f"缺少历史表征文件：{path}")

    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={
            "environment_code": "string",
            "environment_group": "string",
            "split": "string",
        },
    )

    required = {
        "sample_id",
        "environment_code",
        "environment_group",
        "cognitive_impairment",
        "split",
        "seed",
        "y_prob",
        "zc_prob",
        *latent_columns("zc"),
        *latent_columns("zs"),
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path}缺少字段：{sorted(missing)}")

    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError(f"{path}sample_id为空或重复。")
    if not frame["split"].eq(expected_split).all():
        raise ValueError(f"{path}split字段与文件类型不一致。")
    if not frame["seed"].astype(int).eq(42).all():
        raise ValueError(f"{path}不是seed=42的历史模型结果。")

    frame["environment_code"] = (
        frame["environment_code"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    if not frame["environment_code"].str.fullmatch(r"[0-9]{6}").all():
        raise ValueError(f"{path}监测点编码格式错误。")

    if expected_split == "test":
        if not frame["environment_group"].eq(held_out_region).all():
            raise ValueError(f"{path}目标测试样本不属于留出地区{held_out_region}。")
    else:
        if frame["environment_group"].eq(held_out_region).any():
            raise ValueError(f"{path}源域文件中混入留出地区{held_out_region}。")

    label = frame["cognitive_impairment"].to_numpy(dtype=float)
    if not np.isin(label, [0, 1]).all():
        raise ValueError(f"{path}结局标签不是0/1编码。")

    numeric = ["y_prob", "zc_prob"] + latent_columns("zc") + latent_columns("zs")
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError(f"{path}包含非有限预测概率或表征值。")
    if not frame[["y_prob", "zc_prob"]].apply(
        lambda col: col.between(0, 1).all()
    ).all():
        raise ValueError(f"{path}预测概率超出[0,1]。")

    return frame


def load_fold(region, model_root):
    """任务三 第2步：加载一个LORO折的train/val/test表征及模型元数据。"""
    fold_dir = model_root / region / "v5"
    config_path = fold_dir / "config.json"
    metrics_path = fold_dir / "metrics.json"
    training_path = fold_dir / "training.json"

    for path in (config_path, metrics_path, training_path):
        if not path.is_file():
            raise FileNotFoundError(f"缺少V5元数据：{path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("input_sha256") != EXPECTED_DATA_SHA256:
        raise RuntimeError(f"{region} V5模型数据哈希与研究数据不一致。")
    if config.get("holdout_level") != "region":
        raise RuntimeError(f"{region}不是留一地区模型。")
    if config.get("feature_names") is None or len(config["feature_names"]) != 19:
        raise RuntimeError(f"{region}模型输入特征配置异常。")
    if config.get("config", {}).get("latent_dim") != LATENT_DIM:
        raise RuntimeError(f"{region}V5潜在维度不是{LATENT_DIM}。")

    train = read_representation(
        fold_dir / "representations_train.csv", "train", region
    )
    validation = read_representation(
        fold_dir / "representations_val.csv", "val", region
    )
    test = read_representation(
        fold_dir / "representations_test.csv", "test", region
    )

    ids = [
        set(train["sample_id"].astype(int)),
        set(validation["sample_id"].astype(int)),
        set(test["sample_id"].astype(int)),
    ]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError(f"{region}的train/val/test sample_id存在重叠。")

    test_sites = sorted(test["environment_code"].unique().tolist())
    source_sites = sorted(
        set(train["environment_code"]).union(set(validation["environment_code"]))
    )
    if len(test_sites) != 4 or len(source_sites) != 8:
        raise ValueError(
            f"{region}折应有4个留出站点和8个源站点，"
            f"实际为{len(test_sites)}和{len(source_sites)}。"
        )

    metrics_record = json.loads(metrics_path.read_text(encoding="utf-8"))
    training_record = json.loads(training_path.read_text(encoding="utf-8"))

    if int(training_record["best_epoch"]) != int(
        metrics_record.get("best_epoch", training_record["best_epoch"])
    ):
        raise ValueError(f"{region}训练记录最佳轮次不一致。")

    # 任务三 第2步：用历史metrics.json复核表征CSV中的两个预测头。
    test_y = test["cognitive_impairment"].to_numpy(dtype=int)
    prediction_checks = []
    for head_name, column, reference_name in (
        ("joint_zc_plus_zs", "y_prob", "main"),
        ("zc_only", "zc_prob", "zc"),
    ):
        computed = safe_prediction_metrics(
            test_y, test[column].to_numpy(dtype=float)
        )
        reference = metrics_record[reference_name]
        for metric_name in ("auc", "auprc", "brier", "log_loss"):
            expected = float(reference[metric_name])
            observed = float(computed[metric_name])
            difference = observed - expected
            if not np.isclose(observed, expected, atol=5e-5, rtol=0):
                raise RuntimeError(
                    f"{region}/{head_name}/{metric_name}与历史metrics.json不一致："
                    f"CSV={observed:.8f}, JSON={expected:.8f}"
                )
            prediction_checks.append({
                "held_out_region": region,
                "prediction_head": head_name,
                "metric": metric_name,
                "metrics_json_value": expected,
                "representation_csv_value": observed,
                "absolute_difference": abs(difference),
                "passed": True,
            })

    artifacts = {}
    for name in (
        "config.json",
        "metrics.json",
        "training.json",
        "representations_train.csv",
        "representations_val.csv",
        "representations_test.csv",
        "best_model.pt",
    ):
        path = fold_dir / name
        if path.is_file():
            artifacts[name] = sha256_file(path)

    return {
        "region": region,
        "fold_dir": fold_dir,
        "train": train,
        "validation": validation,
        "test": test,
        "source": pd.concat([train, validation], ignore_index=True),
        "source_sites": source_sites,
        "test_sites": test_sites,
        "config": config,
        "metrics": metrics_record,
        "prediction_checks": prediction_checks,
        "training": training_record,
        "artifacts": artifacts,
    }


# ==================== 二、表征谱与退化诊断 ====================

def effective_rank(eigenvalues):
    """任务三 第2步：计算participation ratio和entropy effective rank。"""
    values = np.clip(np.asarray(eigenvalues, dtype=float), 0, None)
    total = float(values.sum())
    if total <= 0:
        return 0.0, 0.0

    probabilities = values / total
    participation_ratio = total**2 / float(np.square(values).sum())
    positive = probabilities[probabilities > 0]
    entropy_rank = float(np.exp(-np.sum(positive * np.log(positive))))
    return float(participation_ratio), entropy_rank


def representation_spectrum(values):
    """任务三 第2步：计算潜在维度方差、协方差谱和ReLU激活情况。"""
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[1] != LATENT_DIM:
        raise ValueError("潜在表征矩阵形状不符合32维要求。")
    if len(x) < 3:
        raise ValueError("样本数不足，无法稳定计算协方差谱。")

    variance = np.var(x, axis=0, ddof=1)
    covariance = np.cov(x, rowvar=False, ddof=1)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    eigenvalues = np.clip(eigenvalues, 0, None)
    total = float(eigenvalues.sum())
    explained = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
    pr, entropy = effective_rank(eigenvalues)

    return {
        "n": int(len(x)),
        "latent_dim": int(x.shape[1]),
        "active_dimensions_var_gt_1e4": int(np.count_nonzero(variance > VARIANCE_EPSILON)),
        "zero_activation_rate": float(np.mean(np.abs(x) <= ZERO_EPSILON)),
        "mean_dimension_variance": float(np.mean(variance)),
        "median_dimension_variance": float(np.median(variance)),
        "minimum_dimension_variance": float(np.min(variance)),
        "maximum_dimension_variance": float(np.max(variance)),
        "covariance_trace": total,
        "effective_rank_participation_ratio": pr,
        "effective_rank_entropy": entropy,
        "top1_variance_explained": float(explained[:1].sum()),
        "top3_variance_explained": float(explained[:3].sum()),
        "top5_variance_explained": float(explained[:5].sum()),
        "variance_spectrum": ";".join(f"{v:.10g}" for v in explained),
    }


def covariance_matrix(values):
    x = np.asarray(values, dtype=float)
    cov = np.cov(x, rowvar=False, ddof=1)
    return cov + np.eye(cov.shape[0]) * RIDGE


def representation_shift(source_values, target_values):
    """任务三 第2步：同一LORO模型折内的源域—目标站点表征偏移。"""
    source = np.asarray(source_values, dtype=float)
    target = np.asarray(target_values, dtype=float)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_cov = covariance_matrix(source)
    target_cov = covariance_matrix(target)

    difference = target_mean - source_mean
    diagonal = np.diag(source_cov)
    standardized_rms = float(
        np.sqrt(np.mean(np.square(difference) / np.maximum(diagonal, RIDGE)))
    )
    mahalanobis = float(
        np.sqrt(max(0.0, difference @ np.linalg.pinv(source_cov) @ difference))
    )
    covariance_relative_frobenius = float(
        np.linalg.norm(target_cov - source_cov, ord="fro")
        / max(np.linalg.norm(source_cov, ord="fro"), RIDGE)
    )

    source_spectrum = representation_spectrum(source)
    target_spectrum = representation_spectrum(target)

    return {
        "source_n": int(len(source)),
        "target_n": int(len(target)),
        "standardized_mean_shift_rms": standardized_rms,
        "mahalanobis_mean_shift": mahalanobis,
        "relative_covariance_frobenius_shift": covariance_relative_frobenius,
        "source_effective_rank_pr": source_spectrum["effective_rank_participation_ratio"],
        "target_effective_rank_pr": target_spectrum["effective_rank_participation_ratio"],
        "effective_rank_delta_target_minus_source": (
            target_spectrum["effective_rank_participation_ratio"]
            - source_spectrum["effective_rank_participation_ratio"]
        ),
        "source_zero_activation_rate": source_spectrum["zero_activation_rate"],
        "target_zero_activation_rate": target_spectrum["zero_activation_rate"],
        "zero_activation_rate_delta": (
            target_spectrum["zero_activation_rate"]
            - source_spectrum["zero_activation_rate"]
        ),
    }


# ==================== 三、预测头与表征分离 ====================

def safe_prediction_metrics(y, probability):
    """任务三 第2步：计算固定预测头的站点级指标。"""
    y = np.asarray(y, dtype=int)
    p = np.asarray(probability, dtype=float)
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)),
        "positive_count": int(y.sum()),
        "positive_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, p)) if both else np.nan,
        "auprc": float(average_precision_score(y, p)) if y.sum() else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "mean_probability": float(np.mean(p)),
        "predicted_positive_at_0_5": int(np.count_nonzero(p >= 0.5)),
    }


def standardized_centroid_separation(values, labels):
    """任务三 第2步：阳性/阴性表征中心的标准化距离（非因果效应）。"""
    x = np.asarray(values, dtype=float)
    y = np.asarray(labels, dtype=int)
    positive = x[y == 1]
    negative = x[y == 0]

    if len(positive) < 2 or len(negative) < 2:
        return {
            "positive_n": int(len(positive)),
            "negative_n": int(len(negative)),
            "standardized_centroid_distance_rms": np.nan,
        }

    mean_difference = positive.mean(axis=0) - negative.mean(axis=0)
    pooled_variance = (
        np.var(positive, axis=0, ddof=1)
        + np.var(negative, axis=0, ddof=1)
    ) / 2
    distance = float(
        np.sqrt(
            np.mean(
                np.square(mean_difference)
                / np.maximum(pooled_variance, RIDGE)
            )
        )
    )
    return {
        "positive_n": int(len(positive)),
        "negative_n": int(len(negative)),
        "standardized_centroid_distance_rms": distance,
    }


def linear_cka(left, right):
    """任务三 第2步：计算同一批样本上zc与zs的线性CKA相似度。"""
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    if len(x) != len(y) or len(x) < 3:
        return np.nan

    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)

    cross = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    denominator = np.linalg.norm(xx, ord="fro") * np.linalg.norm(yy, ord="fro")
    if denominator <= 0:
        return np.nan
    return float(np.square(np.linalg.norm(cross, ord="fro")) / denominator)


# ==================== 四、主分析 ====================

def analyze_fold(fold):
    """任务三 第2步：在单个LORO折内计算表征谱、偏移和预测差异。"""
    region = fold["region"]
    source = fold["source"]
    test = fold["test"]

    spectrum_rows = []
    shift_rows = []
    metric_rows = []
    separation_rows = []
    similarity_rows = []

    # 源域分别保留train/validation，再合并形成源域参考分布。
    partitions = [
        ("source_train", fold["train"]),
        ("source_validation", fold["validation"]),
        ("source_pool_train_val_site", source),
    ]

    for partition_name, partition in partitions:
        for site, group in partition.groupby("environment_code"):
            for branch in BRANCHES:
                columns = latent_columns(branch)
                spectrum = representation_spectrum(group[columns].to_numpy())
                spectrum_rows.append({
                    "held_out_region": region,
                    "partition": partition_name,
                    "site_code": str(site),
                    "branch": branch,
                    **spectrum,
                })

    # 任务三 第2步：单独保留源域合并分布，供目标站点偏移比较。
    source_values = {
        branch: source[latent_columns(branch)].to_numpy(dtype=float)
        for branch in BRANCHES
    }
    for branch in BRANCHES:
        spectrum_rows.append({
            "held_out_region": region,
            "partition": "source_pool_train_val",
            "site_code": "SOURCE_POOL",
            "branch": branch,
            **representation_spectrum(source_values[branch]),
        })
        source_labels = source["cognitive_impairment"].to_numpy(dtype=int)
        separation_rows.append({
            "held_out_region": region,
            "site_code": "SOURCE_POOL",
            "partition": "source_pool_train_val",
            "branch": branch,
            **standardized_centroid_separation(
                source_values[branch],
                source_labels,
            ),
        })

    similarity_rows.append({
        "held_out_region": region,
        "site_code": "SOURCE_POOL",
        "partition": "source_pool_train_val",
        "n": int(len(source)),
        "linear_cka_zc_zs": linear_cka(
            source_values["zc"],
            source_values["zs"],
        ),
    })

    # 目标站点：计算表征谱、预测头表现、类别分离和分支相似度。

    for site, group in test.groupby("environment_code"):
        site = str(site)
        labels = group["cognitive_impairment"].to_numpy(dtype=int)

        for branch in BRANCHES:
            columns = latent_columns(branch)
            target_values = group[columns].to_numpy(dtype=float)
            spectrum = representation_spectrum(target_values)
            spectrum_rows.append({
                "held_out_region": region,
                "partition": "target_test",
                "site_code": site,
                "branch": branch,
                **spectrum,
            })

            shift = representation_shift(
                source_values[branch],
                target_values,
            )
            shift_rows.append({
                "held_out_region": region,
                "site_code": site,
                "branch": branch,
                **shift,
            })

            separation = standardized_centroid_separation(
                target_values,
                labels,
            )
            separation_rows.append({
                "held_out_region": region,
                "site_code": site,
                "partition": "target_test",
                "branch": branch,
                **separation,
            })

        similarity_rows.append({
            "held_out_region": region,
            "site_code": site,
            "partition": "target_test",
            "n": int(len(group)),
            "linear_cka_zc_zs": linear_cka(
                group[latent_columns("zc")].to_numpy(dtype=float),
                group[latent_columns("zs")].to_numpy(dtype=float),
            ),
        })

        for head, column in (
            ("joint_zc_plus_zs", "y_prob"),
            ("zc_only", "zc_prob"),
        ):
            metric_rows.append({
                "held_out_region": region,
                "site_code": site,
                "prediction_head": head,
                **safe_prediction_metrics(labels, group[column].to_numpy(dtype=float)),
            })

    return {
        "spectrum": spectrum_rows,
        "shift": shift_rows,
        "metrics": metric_rows,
        "separation": separation_rows,
        "similarity": similarity_rows,
    }


def summarize_prediction_heads(metrics_table):
    """任务三 第2步：计算联合头相对zc-only的性能差异。"""
    rows = []
    for (region, site), group in metrics_table.groupby(
        ["held_out_region", "site_code"]
    ):
        joint = group.loc[group["prediction_head"] == "joint_zc_plus_zs"].iloc[0]
        causal = group.loc[group["prediction_head"] == "zc_only"].iloc[0]
        rows.append({
            "held_out_region": region,
            "site_code": site,
            "n": int(joint["n"]),
            "positive_count": int(joint["positive_count"]),
            "positive_rate": float(joint["positive_rate"]),
            "joint_auc": float(joint["auc"]),
            "zc_only_auc": float(causal["auc"]),
            "delta_auc_joint_minus_zc": float(joint["auc"] - causal["auc"]),
            "joint_auprc": float(joint["auprc"]),
            "zc_only_auprc": float(causal["auprc"]),
            "delta_auprc_joint_minus_zc": float(joint["auprc"] - causal["auprc"]),
            "joint_brier": float(joint["brier"]),
            "zc_only_brier": float(causal["brier"]),
            "delta_brier_joint_minus_zc": float(joint["brier"] - causal["brier"]),
        })
    return pd.DataFrame(rows)


# ==================== 五、绘图 ====================

def draw_figures(metrics_table, spectrum_table, shift_table, separation_table, output):
    """任务三 第2步：统一将图表保存至figures目录。"""
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    # 图1：各目标监测点联合头与zc-only AUC。
    pivot = metrics_table.pivot(
        index="site_code",
        columns="prediction_head",
        values="auc",
    )
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("ROC-AUC")
    ax.set_xlabel("Held-out monitoring site")
    ax.set_title("V5 joint prediction head vs zc-only head")
    ax.legend(title="Prediction head")
    fig.tight_layout()
    fig.savefig(figure_dir / "joint_vs_zc_prediction_auc.png", dpi=250)
    plt.close(fig)

    # 图2：目标站点zc/zs有效秩。
    target_spectrum = spectrum_table.loc[
        spectrum_table["partition"] == "target_test"
    ]
    rank_pivot = target_spectrum.pivot(
        index="site_code",
        columns="branch",
        values="effective_rank_participation_ratio",
    ).sort_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    rank_pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("Effective rank (participation ratio)")
    ax.set_xlabel("Held-out monitoring site")
    ax.set_title("V5 latent effective rank by monitoring site")
    ax.set_ylim(0, LATENT_DIM)
    fig.tight_layout()
    fig.savefig(figure_dir / "target_latent_effective_rank.png", dpi=250)
    plt.close(fig)

    # 图3：源域到目标站点的表征均值偏移。
    shift_pivot = shift_table.pivot(
        index="site_code",
        columns="branch",
        values="standardized_mean_shift_rms",
    ).sort_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    shift_pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("Standardized mean shift (RMS)")
    ax.set_xlabel("Held-out monitoring site")
    ax.set_title("Source-to-target latent mean shift")
    fig.tight_layout()
    fig.savefig(figure_dir / "source_target_latent_mean_shift.png", dpi=250)
    plt.close(fig)

    # 图4：阳性/阴性表征中心距离。
    sep_pivot = separation_table.pivot(
        index="site_code",
        columns="branch",
        values="standardized_centroid_distance_rms",
    ).sort_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    sep_pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("Standardized centroid distance (RMS)")
    ax.set_xlabel("Held-out monitoring site")
    ax.set_title("Positive-negative separation in V5 latent spaces")
    fig.tight_layout()
    fig.savefig(figure_dir / "latent_class_centroid_separation.png", dpi=250)
    plt.close(fig)

    # 图5：ReLU零激活率。
    zero_pivot = target_spectrum.pivot(
        index="site_code",
        columns="branch",
        values="zero_activation_rate",
    ).sort_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    zero_pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("Fraction of latent entries equal to zero")
    ax.set_xlabel("Held-out monitoring site")
    ax.set_title("ReLU zero-activation rate by latent branch")
    fig.tight_layout()
    fig.savefig(figure_dir / "latent_zero_activation_rate.png", dpi=250)
    plt.close(fig)


# ==================== 六、主程序 ====================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(
            f"固定实验目录已存在，为避免覆盖：{args.output}"
        )

    # 所有折先完整核验，再计算结果。
    folds = {
        region: load_fold(region, args.model_root)
        for region in REGIONS
    }

    # 检查每个样本只在其所属留出折中作为target出现一次。
    target_ids = []
    target_sites = set()
    for fold in folds.values():
        target_ids.extend(fold["test"]["sample_id"].astype(int).tolist())
        target_sites.update(fold["test"]["environment_code"].astype(str).unique())
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("三个留出地区测试集sample_id存在重复。")
    if len(target_ids) != EXPECTED_TOTAL_SAMPLES:
        raise ValueError(
            f"三个留出折测试样本合计应为{EXPECTED_TOTAL_SAMPLES}，"
            f"实际为{len(target_ids)}。"
        )
    if len(target_sites) != EXPECTED_SITE_COUNT:
        raise ValueError(
            f"目标监测点应为{EXPECTED_SITE_COUNT}个，实际为{len(target_sites)}。"
        )

    results = {
        "spectrum": [],
        "shift": [],
        "metrics": [],
        "separation": [],
        "similarity": [],
        "prediction_checks": [],
    }
    for region in REGIONS:
        print(f"分析V5表征：留出地区 {region}", flush=True)
        fold_result = analyze_fold(folds[region])
        for key in ("spectrum", "shift", "metrics", "separation", "similarity"):
            results[key].extend(fold_result[key])
        results["prediction_checks"].extend(folds[region]["prediction_checks"])

    spectrum_table = pd.DataFrame(results["spectrum"])
    shift_table = pd.DataFrame(results["shift"])
    metrics_table = pd.DataFrame(results["metrics"])
    separation_table = pd.DataFrame(results["separation"])
    similarity_table = pd.DataFrame(results["similarity"])
    prediction_check_table = pd.DataFrame(results["prediction_checks"])
    head_delta_table = summarize_prediction_heads(metrics_table)

    # 只聚合不依赖坐标轴方向的谱指标；不合并不同fold的latent坐标。
    spectrum_summary = (
        spectrum_table.groupby(
            ["partition", "branch"],
            as_index=False,
        )
        .agg(
            record_count=("site_code", "size"),
            mean_effective_rank_pr=("effective_rank_participation_ratio", "mean"),
            median_effective_rank_pr=("effective_rank_participation_ratio", "median"),
            mean_entropy_rank=("effective_rank_entropy", "mean"),
            mean_active_dimensions=("active_dimensions_var_gt_1e4", "mean"),
            mean_zero_activation_rate=("zero_activation_rate", "mean"),
            mean_top3_variance_explained=("top3_variance_explained", "mean"),
        )
    )

    args.output.mkdir(parents=True, exist_ok=False)
    outputs = (
        ("representation_spectrum_by_site.csv", spectrum_table),
        ("representation_spectrum_summary.csv", spectrum_summary),
        ("source_target_representation_shift.csv", shift_table),
        ("site_prediction_head_metrics.csv", metrics_table),
        ("joint_vs_zc_head_comparison.csv", head_delta_table),
        ("site_latent_class_separation.csv", separation_table),
        ("site_zc_zs_similarity.csv", similarity_table),
        ("fold_prediction_head_validation.csv", prediction_check_table),
    )
    for filename, table in outputs:
        table.to_csv(
            args.output / filename,
            index=False,
            encoding="utf-8-sig",
        )

    draw_figures(
        metrics_table,
        spectrum_table,
        shift_table,
        separation_table,
        args.output,
    )

    metadata = {
        "task": "Task 3 Step 2 - V5 Representation Diagnostics",
        "model": "V5 CausalDisentangledModel",
        "model_root": str(args.model_root.resolve()),
        "regions": list(REGIONS),
        "latent_dim_per_branch": LATENT_DIM,
        "branches": list(BRANCHES),
        "expected_data_sha256": EXPECTED_DATA_SHA256,
        "models_retrained": False,
        "representations_reused": True,
        "target_labels_used_only_for_prediction_metrics_and_class_separation": True,
        "individual_records_exported": False,
        "cross_fold_coordinate_comparison": False,
        "source_reference": (
            "Within each LORO fold, concatenate that fold's saved source train and "
            "validation representations; compare each held-out target site to this pool."
        ),
        "effective_rank": {
            "participation_ratio": "(sum eigenvalues)^2 / sum(eigenvalues^2)",
            "entropy_rank": "exp(-sum p_i log p_i)",
            "active_dimension_threshold": VARIANCE_EPSILON,
            "zero_activation_threshold": ZERO_EPSILON,
        },
        "shift_metrics": [
            "standardized_mean_shift_rms",
            "mahalanobis_mean_shift",
            "relative_covariance_frobenius_shift",
        ],
        "class_separation": (
            "Standardized positive-negative latent centroid distance; descriptive, not causal."
        ),
        "branch_similarity": "Linear CKA between zc and zs for the same site samples.",
        "limitations": [
            "Low effective rank alone is not proof of representation collapse.",
            "ReLU zero activations are expected and are not sufficient evidence of collapse.",
            "Joint head versus zc-only head is not a direct zs-only ablation; predictor_joint can use interactions between zc and zs.",
            "No zs-only prediction head is stored in the historical model.",
            "Latent coordinates from different LORO folds are not directly comparable.",
            "Source-target distribution distances are descriptive and do not identify causal mechanisms.",
            "Positive-negative latent separation uses held-out labels and is an exploratory diagnostic.",
        ],
        "fold_artifacts": {
            region: folds[region]["artifacts"]
            for region in REGIONS
        },
    }
    write_json(args.output / "experiment_metadata.json", metadata)

    print("=" * 72)
    print("任务三 第2步：V5表征诊断完成（未训练模型）")
    print(f"结果目录：{args.output}")
    print("\n联合预测头与zc-only对照：")
    print(head_delta_table.to_string(index=False))
    print("\n目标站点表征有效秩：")
    print(
        spectrum_table.loc[
            spectrum_table["partition"] == "target_test",
            [
                "held_out_region",
                "site_code",
                "branch",
                "effective_rank_participation_ratio",
                "active_dimensions_var_gt_1e4",
                "zero_activation_rate",
            ],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

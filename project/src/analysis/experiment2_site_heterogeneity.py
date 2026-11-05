"""任务二，实验二：监测点预测性能与输入特征偏移关联分析。

研究目的：
1. 复用已有LORO原始/加权MLP的监测点预测指标；
2. 对每个监测点，比较该站点特征分布与其所属LORO实验的源地区训练数据分布；
3. 探索特征偏移程度与监测点预测表现/加权收益之间的关联。

本脚本不训练模型，不读取认知受损标签来计算特征偏移。
预测指标来自已提交的site_model_comparison.csv。
仅12个监测点，所有相关分析均为探索性描述，不作因果解释。
"""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

from elderly_health.data import (  # noqa: E402
    ENVIRONMENT,
    FEATURES,
    PROVINCE_TO_REGION,
)

# ==================== 一、固定实验路径 ====================

DATA_PATH = PROJECT / "data" / "data_mission2.csv"
SITE_METRICS_PATH = (
    ROOT / "outputs" / "analysis_2611"
    / "experiment1b_covariate_reweighting"
    / "site_model_comparison.csv"
)
OUTPUT_DIR = (
    ROOT / "outputs" / "analysis_2611"
    / "experiment2_site_heterogeneity"
)

REGIONS = ("east", "central", "west")
BASELINE_MODEL = "MLP_unweighted_historical_reused"
WEIGHTED_MODEL = "MLP_importance_weighted"

PERFORMANCE_COLUMNS = [
    "auc",
    "auprc",
    "brier",
    "log_loss",
    "balanced_accuracy",
    "f1",
    "precision",
    "recall",
]


# ==================== 二、基础函数 ====================

def sha256_file(path):
    """任务二，实验二：计算输入文件SHA256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_category(value):
    """任务二，实验二：类别化数值编码，并将缺失作为独立类别。"""
    if pd.isna(value):
        return "<MISSING>"
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return format(number, ".8g")


def js_divergence(count_a, count_b):
    """任务二，实验二：计算两个类别分布的Jensen-Shannon散度。"""
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


def bh_adjust(p_values):
    """任务二，实验二：对探索性特征相关检验执行BH校正。"""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan)
    valid = np.flatnonzero(np.isfinite(p_values))

    if len(valid) == 0:
        return adjusted

    order = valid[np.argsort(p_values[valid])]
    running = 1.0
    total = len(order)

    for i in range(total - 1, -1, -1):
        rank = i + 1
        running = min(running, p_values[order[i]] * total / rank)
        adjusted[order[i]] = min(running, 1.0)

    return adjusted


# ==================== 三、读取和核验数据 ====================

def load_feature_data(path):
    """任务二，实验二：只读取19个特征和监测点代码，不读取健康结局。"""
    required = list(FEATURES) + [ENVIRONMENT]
    frame = pd.read_csv(
        path,
        encoding="gbk",
        usecols=required,
        dtype={ENVIRONMENT: "string"},
    )

    frame[ENVIRONMENT] = (
        frame[ENVIRONMENT]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    if frame[ENVIRONMENT].isna().any():
        raise ValueError("监测点代码存在缺失。")

    valid = frame[ENVIRONMENT].str.fullmatch(r"[0-9]{6}")
    if not valid.all():
        raise ValueError("监测点代码必须为六位数字。")

    province = frame[ENVIRONMENT].str[:2]
    frame["region"] = province.map(PROVINCE_TO_REGION)

    if frame["region"].isna().any():
        bad = sorted(frame.loc[frame["region"].isna(), ENVIRONMENT].unique())
        raise ValueError(f"存在未映射的监测点：{bad}")

    for feature in FEATURES:
        original = frame[feature]
        converted = pd.to_numeric(original, errors="coerce")
        if (original.notna() & converted.isna()).any():
            raise ValueError(f"特征{feature}存在无法转换的编码。")
        if np.isinf(converted.dropna().to_numpy(dtype=float)).any():
            raise ValueError(f"特征{feature}存在无穷值。")
        frame[feature] = converted

    site_table = frame[[ENVIRONMENT, "region"]].drop_duplicates()
    if site_table[ENVIRONMENT].duplicated().any():
        raise ValueError("同一监测点被映射到多个地区。")

    counts = site_table["region"].value_counts().to_dict()
    expected = {"east": 4, "central": 4, "west": 4}
    if counts != expected:
        raise ValueError(f"监测点构成与12站点研究设计不一致：{counts}")

    return frame


def load_site_performance(path):
    """任务二，实验二：读取既有站点预测指标，不重新训练模型。"""
    table = pd.read_csv(path, encoding="utf-8-sig")

    required = {
        "held_out_region",
        "site_code",
        "model",
        "n",
        "positive_rate",
        *PERFORMANCE_COLUMNS,
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"监测点指标文件缺少字段：{sorted(missing)}")

    table["site_code"] = (
        table["site_code"].astype(str).str.replace(r"\.0$", "", regex=True)
    )

    table = table.loc[
        table["model"].isin([BASELINE_MODEL, WEIGHTED_MODEL])
    ].copy()

    if table.duplicated(["site_code", "model"]).any():
        raise ValueError("同一监测点/模型存在重复指标。")

    baseline = table.loc[table["model"] == BASELINE_MODEL].set_index("site_code")
    weighted = table.loc[table["model"] == WEIGHTED_MODEL].set_index("site_code")

    if set(baseline.index) != set(weighted.index) or len(baseline) != 12:
        raise ValueError("原始与加权模型的监测点集合不一致，或不是12个站点。")

    rows = []
    for site in sorted(baseline.index):
        old = baseline.loc[site]
        new = weighted.loc[site]

        if old["held_out_region"] != new["held_out_region"]:
            raise ValueError(f"{site}两个模型的留出地区不一致。")
        if int(old["n"]) != int(new["n"]):
            raise ValueError(f"{site}两个模型的测试样本数不一致。")
        if not np.isclose(
            float(old["positive_rate"]),
            float(new["positive_rate"]),
            atol=1e-6,
        ):
            raise ValueError(f"{site}两个模型的阳性率不一致。")

        row = {
            "site_code": site,
            "held_out_region": str(old["held_out_region"]),
            "sample_count": int(old["n"]),
            "positive_rate": float(old["positive_rate"]),
        }

        for metric in PERFORMANCE_COLUMNS:
            row[f"{metric}_baseline"] = float(old[metric])
            row[f"{metric}_weighted"] = float(new[metric])
            row[f"delta_{metric}"] = float(new[metric]) - float(old[metric])

        rows.append(row)

    return pd.DataFrame(rows)


# ==================== 四、监测点特征偏移计算 ====================

def calculate_site_feature_shift(frame):
    """任务二，实验二：每个站点与其LORO源地区数据比较19个特征。

    对于某站点所属的留出地区，源域定义为另外两个地区的全部样本，
    与实验1b中源域train+validation的合并样本一致。
    目标站点只提供无标签特征分布，不使用该站点健康结局。
    """
    detail_rows = []
    summary_rows = []

    sites = (
        frame[[ENVIRONMENT, "region"]]
        .drop_duplicates()
        .sort_values(["region", ENVIRONMENT])
    )

    for site_row in sites.itertuples(index=False):
        site = str(getattr(site_row, ENVIRONMENT))
        held_out_region = str(site_row.region)

        source = frame.loc[frame["region"] != held_out_region]
        target_site = frame.loc[frame[ENVIRONMENT].astype(str) == site]

        if source.empty or target_site.empty:
            raise ValueError(f"{site}源域或目标站点数据为空。")

        site_js_values = []
        missing_differences = []

        for feature in FEATURES:
            source_values = source[feature].map(normalize_category)
            site_values = target_site[feature].map(normalize_category)

            categories = sorted(
                set(source_values).union(set(site_values)),
                key=lambda value: (value == "<MISSING>", value),
            )

            source_counts = np.asarray(
                [(source_values == category).sum() for category in categories],
                dtype=float,
            )
            site_counts = np.asarray(
                [(site_values == category).sum() for category in categories],
                dtype=float,
            )

            js_value = js_divergence(source_counts, site_counts)

            source_missing_rate = float(source_values.eq("<MISSING>").mean())
            site_missing_rate = float(site_values.eq("<MISSING>").mean())
            missing_difference = site_missing_rate - source_missing_rate

            source_distribution = source_counts / source_counts.sum()
            site_distribution = site_counts / site_counts.sum()
            max_category_difference = float(
                np.max(np.abs(site_distribution - source_distribution))
            )

            site_js_values.append(js_value)
            missing_differences.append(abs(missing_difference))

            detail_rows.append(
                {
                    "site_code": site,
                    "held_out_region": held_out_region,
                    "feature": feature,
                    "source_sample_count": len(source),
                    "site_sample_count": len(target_site),
                    "js_divergence_bits": js_value,
                    "max_category_proportion_difference": max_category_difference,
                    "source_missing_rate": source_missing_rate,
                    "site_missing_rate": site_missing_rate,
                    "missing_rate_difference": missing_difference,
                }
            )

        detail = pd.DataFrame(
            [row for row in detail_rows if row["site_code"] == site]
        )
        top = detail.sort_values(
            "js_divergence_bits", ascending=False
        ).iloc[0]

        summary_rows.append(
            {
                "site_code": site,
                "held_out_region": held_out_region,
                "source_sample_count": len(source),
                "site_sample_count": len(target_site),
                "mean_feature_js": float(np.nanmean(site_js_values)),
                "median_feature_js": float(np.nanmedian(site_js_values)),
                "max_feature_js": float(np.nanmax(site_js_values)),
                "top_shift_feature": str(top["feature"]),
                "top_shift_js": float(top["js_divergence_bits"]),
                "mean_absolute_missing_rate_difference": float(
                    np.mean(missing_differences)
                ),
            }
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


# ==================== 五、偏移与性能的探索性关联 ====================

def calculate_correlations(performance, feature_detail):
    """任务二，实验二：计算12站点特征偏移与性能的Spearman相关。

    相关分析只用于生成后续假设。站点数为12，且站点嵌套在3个留出地区，
    不将p值解释为因果证据或稳健总体推断。
    """
    summary_metrics = [
        "auc_baseline",
        "auc_weighted",
        "delta_auc",
        "delta_auprc",
        "delta_brier",
        "delta_log_loss",
    ]

    rows = []
    for metric in summary_metrics:
        result = spearmanr(
            performance["mean_feature_js"],
            performance[metric],
            nan_policy="omit",
        )
        rows.append(
            {
                "shift_measure": "mean_feature_js",
                "performance_measure": metric,
                "n_sites": int(performance[["mean_feature_js", metric]].dropna().shape[0]),
                "spearman_rho": float(result.statistic),
                "nominal_p_value": float(result.pvalue),
                "interpretation": "exploratory_only_not_causal",
            }
        )

    summary_correlations = pd.DataFrame(rows)

    detail = feature_detail.merge(
        performance[
            [
                "site_code",
                "auc_baseline",
                "auc_weighted",
                "delta_auc",
                "delta_auprc",
                "delta_brier",
            ]
        ],
        on="site_code",
        how="inner",
        validate="many_to_one",
    )

    feature_rows = []
    outcomes = ["auc_baseline", "auc_weighted", "delta_auc", "delta_auprc", "delta_brier"]

    for feature in FEATURES:
        subset = detail.loc[detail["feature"] == feature]

        for outcome in outcomes:
            valid = subset[["js_divergence_bits", outcome]].dropna()
            if len(valid) < 4 or valid["js_divergence_bits"].nunique() < 2:
                rho, p_value = np.nan, np.nan
            else:
                result = spearmanr(
                    valid["js_divergence_bits"],
                    valid[outcome],
                )
                rho, p_value = float(result.statistic), float(result.pvalue)

            feature_rows.append(
                {
                    "feature": feature,
                    "performance_measure": outcome,
                    "n_sites": len(valid),
                    "spearman_rho": rho,
                    "nominal_p_value": p_value,
                    "interpretation": "exploratory_only_not_causal",
                }
            )

    feature_correlations = pd.DataFrame(feature_rows)
    feature_correlations["bh_q_within_performance"] = np.nan

    for outcome in outcomes:
        mask = feature_correlations["performance_measure"] == outcome
        feature_correlations.loc[mask, "bh_q_within_performance"] = bh_adjust(
            feature_correlations.loc[mask, "nominal_p_value"].to_numpy()
        )

    return summary_correlations, feature_correlations


# ==================== 六、可视化 ====================

def create_figures(performance, feature_detail, output_dir):
    """任务二，实验二：绘制站点性能与输入偏移关系图。"""
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        performance["mean_feature_js"],
        performance["delta_auc"],
        c=pd.Categorical(performance["held_out_region"]).codes,
        s=70,
    )

    for row in performance.itertuples(index=False):
        ax.annotate(
            row.site_code,
            (row.mean_feature_js, row.delta_auc),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Mean feature JS divergence (site vs LORO source)")
    ax.set_ylabel("AUC change after importance weighting")
    ax.set_title("Site-level feature shift vs prediction improvement")
    fig.tight_layout()
    fig.savefig(
        figure_dir / "site_shift_vs_auc_delta.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)

    ordered = performance.sort_values(
        ["held_out_region", "site_code"]
    )
    x = np.arange(len(ordered))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x - width / 2,
        ordered["auc_baseline"],
        width,
        label="Historical unweighted MLP",
    )
    ax.bar(
        x + width / 2,
        ordered["auc_weighted"],
        width,
        label="Importance-weighted MLP",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["site_code"], rotation=45, ha="right")
    ax.set_ylabel("Site-level AUC")
    ax.set_ylim(0, 1)
    ax.set_title("Unweighted vs weighted MLP by monitoring site")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        figure_dir / "site_auc_before_after.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)

    # 任务二，实验二：绘制特征JS与AUC变化的Spearman相关热图。
    matrix = feature_detail.pivot(
        index="site_code",
        columns="feature",
        values="js_divergence_bits",
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title("Feature distribution shift by monitoring site")
    fig.colorbar(image, ax=ax, label="Jensen-Shannon divergence (bits)")
    fig.tight_layout()
    fig.savefig(
        figure_dir / "site_feature_shift_heatmap.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


# ==================== 七、主程序 ====================

def main():
    """任务二，实验二：复用已有预测指标，计算监测点输入偏移。"""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--site-metrics", type=Path, default=SITE_METRICS_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(f"找不到数据文件：{args.data}")
    if not args.site_metrics.is_file():
        raise FileNotFoundError(f"找不到已有监测点指标：{args.site_metrics}")
    if args.output.exists():
        raise FileExistsError(
            f"固定输出目录已存在，为避免覆盖已有实验结果：{args.output}"
        )

    # 任务二，实验二：读取不含健康结局的输入特征。
    frame = load_feature_data(args.data)

    # 任务二，实验二：复用实验1b已经生成的站点级预测指标。
    performance = load_site_performance(args.site_metrics)

    site_regions = frame[[ENVIRONMENT, "region"]].drop_duplicates()
    site_regions[ENVIRONMENT] = site_regions[ENVIRONMENT].astype(str)

    if set(performance["site_code"]) != set(site_regions[ENVIRONMENT]):
        raise ValueError("预测指标中的监测点与原始特征数据不一致。")

    performance = performance.merge(
        site_regions.rename(columns={ENVIRONMENT: "site_code"}),
        on="site_code",
        how="left",
        validate="one_to_one",
    )

    if not (
        performance["held_out_region"].astype(str)
        == performance["region"].astype(str)
    ).all():
        raise ValueError("站点所属地区与LORO留出地区不一致。")

    site_shift, feature_shift = calculate_site_feature_shift(frame)

    performance = performance.merge(
        site_shift,
        on=["site_code", "held_out_region"],
        how="inner",
        validate="one_to_one",
    )

    if len(performance) != 12 or len(feature_shift) != 12 * len(FEATURES):
        raise RuntimeError("监测点特征偏移结果数量不完整。")

    summary_correlations, feature_correlations = calculate_correlations(
        performance,
        feature_shift,
    )

    # 任务二，实验二：记录数据版本和方法边界。
    data_hash = sha256_file(args.data)
    metrics_hash = sha256_file(args.site_metrics)

    args.output.mkdir(parents=True, exist_ok=False)

    performance.to_csv(
        args.output / "site_performance_and_shift.csv",
        index=False,
        encoding="utf-8-sig",
    )
    site_shift.to_csv(
        args.output / "site_feature_shift_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feature_shift.to_csv(
        args.output / "site_feature_shift_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_correlations.to_csv(
        args.output / "site_shift_performance_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    feature_correlations.to_csv(
        args.output / "feature_shift_performance_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    create_figures(performance, feature_shift, args.output)

    metadata = {
        "experiment": "Task 2 Experiment 2 - Site Heterogeneity",
        "data_file": str(args.data.resolve()),
        "data_sha256": data_hash,
        "site_metrics_file": str(args.site_metrics.resolve()),
        "site_metrics_sha256": metrics_hash,
        "sample_count": int(len(frame)),
        "site_count": int(frame[ENVIRONMENT].nunique()),
        "feature_count": len(FEATURES),
        "features": list(FEATURES),
        "region_mapping": PROVINCE_TO_REGION,
        "source_definition": (
            "For each held-out region, source is all samples from the other two regions; "
            "target is one monitoring site within the held-out region."
        ),
        "feature_shift": (
            "Categorical Jensen-Shannon divergence; missing values are retained "
            "as an explicit category."
        ),
        "outcome_used_for_feature_shift": False,
        "prediction_metrics_retrained": False,
        "prediction_metrics_source": "experiment1b site_model_comparison.csv",
        "correlation_method": "Spearman, exploratory only",
        "site_count_for_correlation": 12,
        "causal_interpretation": False,
        "limitations": [
            "Only 12 monitoring sites are available.",
            "Site-level AUC/AUPRC/Brier have sampling uncertainty.",
            "Sites are nested within three held-out-region experiments.",
            "Feature-shift/performance associations are not causal evidence.",
        ],
        "python_note": "This script reads existing results and does not train or infer any model.",
    }

    with (args.output / "experiment_metadata.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)

    print("=" * 68)
    print("任务二，实验二：监测点预测性能与特征偏移分析")
    print("=" * 68)
    print(f"监测点数量：{len(performance)}")
    print(f"特征偏移记录：{len(feature_shift)}")
    print("\n站点性能与平均特征偏移：")
    print(
        performance[
            [
                "site_code",
                "held_out_region",
                "mean_feature_js",
                "auc_baseline",
                "auc_weighted",
                "delta_auc",
                "delta_auprc",
                "delta_brier",
            ]
        ].to_string(index=False)
    )
    print("\n平均特征偏移与性能的探索性相关：")
    print(summary_correlations.to_string(index=False))
    print(f"\n结果保存位置：{args.output}")


if __name__ == "__main__":
    main()

"""任务三 第6步：复用已有结果，从12站点和3地区检查预测不稳定。

不训练模型、不计算新的个体预测、不修改原实验。
汇总5种模型表现、19特征关联、V5表征分离和偏移。
地区均值为站点指标的等权平均，不冒充地区合并样本指标。
220202保留为可返回的案例，不将其规律推广为全部站点的规律。
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
from elderly_health.data import FEATURES  # noqa: E402

ANALYSIS = ROOT / "outputs" / "analysis_2611"
OUTPUT = ANALYSIS / "task3_step6_global_diagnostics"
REGIONS = ("east", "central", "west")
MODELS = ("MLP", "V5", "RDM", "VREx", "GroupDRO")
KEYS = ["held_out_region", "site_code"]
EXPECTED_HASH = "969f0c1c82faafcfa402f37572e4a8ceb324492062cf1e590465f82dc6b5b6de"
INPUTS = {
    "metrics": "task3_step1_site_failure_diagnostics/site_model_score_diagnostics.csv",
    "association": "task3_step1_site_failure_diagnostics/site_feature_label_association.csv",
    "spectrum": "task3_step2_v5_representation_diagnostics/representation_spectrum_by_site.csv",
    "shift": "task3_step2_v5_representation_diagnostics/source_target_representation_shift.csv",
    "separation": "task3_step2_v5_representation_diagnostics/site_latent_class_separation.csv",
}


def sha256_file(path):
    """任务三 第6步：记录输入结果版本，不读取研究个体数据。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_grid(table, dimensions, expected, name):
    """任务三 第6步：检查每个站点、模型、特征或分支组合恰好出现一次。"""
    observed = set(map(tuple, table[dimensions].itertuples(index=False, name=None)))
    if table.duplicated(dimensions).any() or observed != set(expected):
        raise ValueError(f"{name}存在缺失、重复或额外记录。")


def load_results(root):
    """任务三 第6步：核对两步历史实验的数据版本、样本数及站点对应关系。"""
    manifest, tables = {}, {}
    for key, relative in INPUTS.items():
        path = root / relative
        table = pd.read_csv(path, encoding="utf-8-sig", dtype={"site_code": str})
        tables[key] = table
        manifest[relative] = sha256_file(path)
    for step, field in (("task3_step1_site_failure_diagnostics", "data_sha256"),
                        ("task3_step2_v5_representation_diagnostics", "expected_data_sha256")):
        path = root / step / "experiment_metadata.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        if record[field] != EXPECTED_HASH:
            raise ValueError(f"{step}的研究数据版本不一致。")
        manifest[str(path.relative_to(root))] = sha256_file(path)
    metrics = tables["metrics"]
    sites = metrics[KEYS].drop_duplicates()
    if len(sites) != 12 or sites["site_code"].nunique() != 12:
        raise ValueError("预期12个互不重复的目标站点。")
    if sites.groupby("held_out_region").size().to_dict() != {r: 4 for r in REGIONS}:
        raise ValueError("每个地区应有4个目标站点。")
    pairs = list(sites.itertuples(index=False, name=None))
    require_grid(metrics, KEYS + ["model"], [(*p, m) for p in pairs for m in MODELS], "模型指标")
    core = metrics.groupby(KEYS)[["n", "positive_count", "positive_rate"]].nunique()
    if not core.eq(1).all().all():
        raise ValueError("不同模型的样本数或阳性率不一致。")
    base = metrics.drop_duplicates(KEYS).set_index(KEYS)[["n", "positive_count", "positive_rate"]]
    if int(base["n"].sum()) != 4769 or not np.allclose(base["positive_count"] / base["n"], base["positive_rate"]):
        raise ValueError("总样本数或阳性率核对失败。")
    if not np.isfinite(metrics[["auc", "auprc", "brier"]].to_numpy(dtype=float)).all():
        raise ValueError("模型指标含非有限值。")
    if not metrics[["auc", "auprc", "brier"]].apply(lambda s: s.between(0, 1).all()).all():
        raise ValueError("模型指标超出[0,1]。")
    assoc = tables["association"]
    require_grid(assoc, KEYS + ["feature"], [(*p, f) for p in pairs for f in FEATURES], "特征关联")
    for row in assoc.itertuples(index=False):
        b = base.loc[(row.held_out_region, row.site_code)]
        if row.sample_count != b["n"] or row.positive_count != b["positive_count"]:
            raise ValueError("特征关联表与模型指标样本数量不一致。")
    for key in ("spectrum", "separation"):
        tables[key] = tables[key].loc[tables[key]["partition"].eq("target_test")].copy()
    grid = [(*p, branch) for p in pairs for branch in ("zc", "zs")]
    for key in ("spectrum", "shift", "separation"):
        require_grid(tables[key], KEYS + ["branch"], grid, key)
        for row in tables[key].itertuples(index=False):
            b = base.loc[(row.held_out_region, row.site_code)]
            count = row.n if key == "spectrum" else (row.target_n if key == "shift" else row.positive_n + row.negative_n)
            if count != b["n"] or (key == "separation" and row.positive_n != b["positive_count"]):
                raise ValueError(f"{key}与模型指标样本数量不一致。")
    return tables, base.reset_index(), manifest


def build_overview(tables, base):
    """任务三 第6步：将已有聚合结果按站点对齐，不合成新的综合评分。"""
    overview = base.copy()
    metrics = tables["metrics"]
    for metric in ("auc", "auprc", "brier"):
        wide = metrics.pivot(index=KEYS, columns="model", values=metric)
        wide.columns = [f"{model}_{metric}" for model in wide.columns]
        overview = overview.merge(wide.reset_index(), on=KEYS, validate="one_to_one")
    fields = {
        "spectrum": ["effective_rank_participation_ratio", "active_dimensions_var_gt_1e4", "zero_activation_rate"],
        "shift": ["standardized_mean_shift_rms", "relative_covariance_frobenius_shift"],
        "separation": ["standardized_centroid_distance_rms"],
    }
    for key, columns in fields.items():
        wide = tables[key].pivot(index=KEYS, columns="branch", values=columns)
        wide.columns = [f"{branch}_{metric}" for metric, branch in wide.columns]
        overview = overview.merge(wide.reset_index(), on=KEYS, validate="one_to_one")
    auc_columns = [f"{m}_auc" for m in MODELS]
    overview["mean_auc_across_five_models"] = overview[auc_columns].mean(axis=1)
    overview["minimum_auc_across_five_models"] = overview[auc_columns].min(axis=1)
    overview["maximum_auc_across_five_models"] = overview[auc_columns].max(axis=1)
    overview["model_auc_range"] = overview["maximum_auc_across_five_models"] - overview["minimum_auc_across_five_models"]
    overview["held_out_region"] = pd.Categorical(overview["held_out_region"], categories=REGIONS, ordered=True)
    return overview.sort_values(KEYS).reset_index(drop=True)


def model_summaries(metrics):
    """任务三 第6步：明确输出站点等权均值、站点间标准差和最差值。"""
    records = []
    for region in (*REGIONS, "ALL"):
        subset = metrics if region == "ALL" else metrics.loc[metrics["held_out_region"].eq(region)]
        for model in MODELS:
            part = subset.loc[subset["model"].eq(model)]
            record = {"region_scope": region, "model": model, "site_count": len(part),
                      "samples": int(part["n"].sum()), "positive_count": int(part["positive_count"].sum())}
            for metric in ("auc", "auprc", "brier"):
                record.update({f"macro_mean_site_{metric}": float(part[metric].mean()),
                               f"minimum_site_{metric}": float(part[metric].min()),
                               f"maximum_site_{metric}": float(part[metric].max()),
                               f"std_across_site_{metric}": float(part[metric].std(ddof=1))})
            records.append(record)
    return pd.DataFrame(records)


def paired_model_differences(metrics):
    """任务三 第6步：同一站点各方法与MLP对照，不把点估计高低当显著差异。"""
    base = metrics.loc[metrics["model"].eq("MLP"), KEYS + ["auc", "auprc", "brier"]]
    paired = metrics.loc[~metrics["model"].eq("MLP")].merge(base, on=KEYS, suffixes=("", "_MLP"), validate="many_to_one")
    for metric in ("auc", "auprc", "brier"):
        paired[f"delta_{metric}_minus_MLP"] = paired[metric] - paired[f"{metric}_MLP"]
    return paired[KEYS + ["model", "delta_auc_minus_MLP", "delta_auprc_minus_MLP", "delta_brier_minus_MLP"]]


def heatmap(table, path, title, colorbar, diverging=False, annotate=False, figsize=(10, 8)):
    """任务三 第6步：绘制统一站点顺序的结果热力图。"""
    fig, ax = plt.subplots(figsize=figsize)
    values = table.to_numpy(dtype=float)
    options = {}
    if diverging:
        limit = max(0.001, float(np.nanmax(np.abs(values))))
        options = {"vmin": -limit, "vmax": limit}
    else:
        options = {"vmin": 0, "vmax": 1}
    plot = ax.imshow(values, aspect="auto", cmap="RdBu_r" if diverging else "viridis", **options)
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    if annotate:
        for i in range(len(table.index)):
            for j in range(len(table.columns)):
                ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=8,
                        color="black" if diverging else ("white" if values[i,j] < 0.6 else "black"))
    ax.set_title(title)
    fig.colorbar(plot, ax=ax, label=colorbar)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def draw_figures(tables, overview, paired, output):
    """任务三 第6步：四张全局图，不为单个站点新增专属图。"""
    figures = output / "figures"
    figures.mkdir()
    order = overview["site_code"].tolist()
    labels = [f"{row.held_out_region}: {row.site_code}" for row in overview.itertuples(index=False)]
    auc = tables["metrics"].pivot(index="site_code", columns="model", values="auc").reindex(index=order, columns=MODELS)
    auc.index = labels
    heatmap(auc, figures / "all_sites_model_auc.png", "Five models across all 12 held-out sites", "ROC-AUC", annotate=True)
    delta = paired.pivot(index="site_code", columns="model", values="delta_auc_minus_MLP").reindex(index=order, columns=MODELS[1:])
    delta.index = labels
    heatmap(delta, figures / "all_sites_auc_difference_from_mlp.png", "ROC-AUC difference from MLP (descriptive)", "Method ROC-AUC minus MLP ROC-AUC", diverging=True, annotate=True)
    assoc = tables["association"].pivot(index="feature", columns="site_code", values="excess_js_over_null_bits").reindex(index=FEATURES, columns=order)
    assoc.columns = labels
    heatmap(assoc, figures / "all_sites_feature_label_association.png", "Feature-label separation above permutation reference", "Excess JS (bits); association strength, not direction", diverging=True, figsize=(13, 10))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, region in zip(axes, REGIONS):
        part = overview.loc[overview["held_out_region"].eq(region)]
        ax.scatter(part["zc_standardized_centroid_distance_rms"], part["V5_auc"], label="zc", marker="o")
        ax.scatter(part["zs_standardized_centroid_distance_rms"], part["V5_auc"], label="zs", marker="^")
        for row in part.itertuples(index=False):
            ax.annotate(row.site_code, (row.zc_standardized_centroid_distance_rms, row.V5_auc), fontsize=8, xytext=(3, 5), textcoords="offset points")
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
        ax.set_ylim(0.4, 1)
        ax.set_xlabel("Positive-negative latent centroid distance")
        ax.set_title(f"{region}: comparisons within the same fold")
        ax.legend()
    axes[0].set_ylabel("V5 site ROC-AUC")
    fig.tight_layout()
    fig.savefig(figures / "within_region_latent_separation_vs_auc.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=ANALYSIS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"固定输出目录已存在，禁止覆盖：{args.output}")
    print("任务三 第6步：核对并整理已有全局结果，不训练模型。", flush=True)
    tables, base, manifest = load_results(args.analysis_root)
    overview = build_overview(tables, base)
    summaries = model_summaries(tables["metrics"])
    paired = paired_model_differences(tables["metrics"])
    args.output.mkdir(parents=True, exist_ok=False)
    for name, table in (("all_site_diagnostic_overview.csv", overview), ("model_region_site_macro_summary.csv", summaries),
                        ("paired_site_model_differences.csv", paired),
                        ("all_site_model_metrics.csv", tables["metrics"]),
                        ("all_site_feature_label_association.csv", tables["association"])):
        table.to_csv(args.output / name, index=False, encoding="utf-8-sig")
    draw_figures(tables, overview, paired, args.output)
    metadata = {"task": "Task 3 Step 6 - Global diagnostics", "data_sha256": EXPECTED_HASH,
                "models": list(MODELS), "regions": list(REGIONS), "site_count": 12, "models_trained": False,
                "individual_records_read": False, "individual_records_exported": False,
                "source_result_hashes": manifest,
                "case_analysis_retained": {"focus_sites": ["220202", "360102"],
                    "completed_steps": ["task3_step3_fixed_representation_probes", "task3_step4_feature_association_transfer", "task3_step5_linear_score_contributions"],
                    "followup_status": "retained_as_optional_return_path_after_global_review"},
                "limitations": [
                    "Only reuses previous aggregated outputs; no new training, predictions or confirmatory experiment.",
                    "Region macro metrics are equal-weight means of four site metrics; ALL means 12 site metrics, not pooled-person metrics.",
                    "Cross-site average precision depends on prevalence; model comparisons at the same site are easier to interpret.",
                    "Model differences are descriptive point estimates without uncertainty or significance claims.",
                    "JS above the permutation null is an exploratory association strength measure, not a direction, conditional effect or causal effect.",
                    "Different features have different category counts and missingness; JS heatmap is not a causal-feature ranking.",
                    "Latent coordinates belong to separate fold models; plots separate the three regions and do not compare coordinate axes across folds.",
                    "Four target sites per region are insufficient for reliable regression or causal explanation; no correlation tests are added.",
                    "Low rank alone does not prove collapse; latent class separation does not measure all discriminative information.",
                    "Case-specific phone-feature findings have not been established for other sites or neural models.",
                    "Failure of one central-region site cannot explain the eastern-region results.",
                ]}
    (args.output / "experiment_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"任务三 第6步完成；结果目录：{args.output}")


if __name__ == "__main__":
    main()

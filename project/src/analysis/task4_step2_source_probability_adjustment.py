"""任务四 第2步：源域验证集单参数概率调整，检查排序与概率误差的区别。

保持RDM/VREx/GroupDRO原模型冻结，恢复历史划分和训练集预处理。
每折每模型仅在源域验证集拟合一个截距，固定斜率为1。
留出地区标签只用于最后评估，不参与拟合或选择调整方法。
只做本步骤预先指定的调整，不尝试多个方法再挑目标域成绩最好者。
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
from elderly_health.data import (  # noqa: E402
    FEATURES, ENVIRONMENT, REGION, read_raw, attach_regions, held_out_split, prepare_splits,
)
from elderly_health.models import SimpleMLP  # noqa: E402
from elderly_health.training import predict, get_device  # noqa: E402
from task3_step1_site_failure_diagnostics import EXPECTED_DATA_SHA256, file_sha256  # noqa: E402

REGIONS = ("east", "central", "west")
ALGORITHMS = {"RDM": "rdm", "VREx": "vrex", "GroupDRO": "groupdro"}
OUTPUT = ROOT / "outputs" / "analysis_2611" / "task4_step2_source_probability_adjustment"
HISTORY = ROOT / "outputs" / "analysis_2611" / "task3_step1_site_failure_diagnostics"


def check_probability(probability):
    """任务四 第2步：禁止静默截断概率，以免人为改变分数并列关系。"""
    p = np.asarray(probability, dtype=float)
    if not np.isfinite(p).all() or not ((p > 0) & (p < 1)).all():
        raise ValueError("需要严格位于(0,1)的有限概率；停止而不静默截断。")
    return p


def adjust_probability(probability, intercept):
    """任务四 第2步：只整体平移原始分数，保持风险排序和并列分数不变。"""
    p = check_probability(probability)
    adjusted = check_probability(expit(logit(p) + intercept))
    unique = np.unique(p)
    if len(unique) > 1 and not (np.diff(expit(logit(unique) + intercept)) > 0).all():
        raise RuntimeError("数值精度导致调整后的排序发生变化，停止分析。")
    return adjusted


def fit_intercept(validation_probability, validation_labels):
    """任务四 第2步：仅用源域验证集最小化未加权log-loss，斜率固定为1。"""
    p = check_probability(validation_probability)
    y = np.asarray(validation_labels, dtype=int)
    if len(y) != len(p) or set(np.unique(y)) != {0, 1}:
        raise ValueError("源域验证集需要包含两类标签且与概率对齐。")
    z = logit(p)
    shift = float(brentq(lambda b: expit(z + b).mean() - y.mean(), -100, 100, xtol=1e-12))
    q = adjust_probability(p, shift)
    if log_loss(y, q, labels=[0, 1]) > log_loss(y, p, labels=[0, 1]) + 1e-10:
        raise RuntimeError("源域验证集目标函数检查失败。")
    return shift


def metrics(y, p):
    """任务四 第2步：计算排序指标、概率误差和平均风险。"""
    y = np.asarray(y, dtype=int)
    return {"n": len(y), "positive_count": int(y.sum()), "positive_rate": float(y.mean()),
            "auc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "mean_probability": float(np.mean(p))}


def evaluate_pair(y, before, after, info):
    """任务四 第2步：验证排序指标不变，概率误差允许改善或恶化。"""
    old, new = metrics(y, before), metrics(y, after)
    if not all(np.isclose(old[m], new[m], atol=1e-12, rtol=0) for m in ("auc", "auprc")):
        raise RuntimeError("调整前后的AUC/AP不一致，停止输出。")
    return [{**info, "stage": stage, **result} for stage, result in (("before", old), ("after", new))]


def reliability_bins(y, p, info):
    """任务四 第2步：固定十分箱保存人数、平均预测及实际比例，不用它选择方法。"""
    idx = np.minimum((np.asarray(p)*10).astype(int), 9)
    rows = []
    for k in range(10):
        mask = idx == k
        if mask.any():
            rows.append({**info, "bin": k, "n": int(mask.sum()),
                         "mean_probability": float(np.asarray(p)[mask].mean()),
                         "positive_rate": float(np.asarray(y)[mask].mean())})
    return rows


def load_frozen_predictions(frame, model_name, region, output_root):
    """任务四 第2步：恢复原训练集预处理，读取冻结权重并与保存的测试概率核对。"""
    slug = ALGORITHMS[model_name]
    directory = output_root / f"external-{slug}-region-loro-42" / region
    paths = {name: directory / name for name in ("config.json", "training.json", "best_model.pt", "predictions_test.csv")}
    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError(f"缺少历史文件：{p}；请先检查本地文件及git lfs pull。")
    with paths["best_model.pt"].open("rb") as stream:
        if stream.read(100).startswith(b"version https://git-lfs.github.com/spec"):
            raise RuntimeError("best_model.pt是LFS指针；请先执行git lfs pull。")
    config = json.loads(paths["config.json"].read_text(encoding="utf-8"))
    training = json.loads(paths["training.json"].read_text(encoding="utf-8"))
    if config["input_sha256"] != EXPECTED_DATA_SHA256 or config["holdout_mode"] != "loro" or config["environment_granularity"] != "region":
        raise ValueError("历史模型的数据或地区划分不符合本实验。")
    settings = config["config"]
    frames = held_out_split(frame, region, seed=settings["seed"], column=REGION)
    (train, val, test), _ = prepare_splits(frames, environment_column=REGION)
    checkpoint = torch.load(paths["best_model.pt"], map_location="cpu", weights_only=True)
    if (checkpoint["algorithm"] != f"tabular_{slug}" or tuple(checkpoint["features"]) != tuple(FEATURES)
            or checkpoint["settings"] != settings or checkpoint["best_epoch"] != training["best_epoch"]):
        raise ValueError("模型权重、配置或选模轮次不一致。")
    model = SimpleMLP(len(FEATURES))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    val_p = check_probability(predict(model, val.x, get_device("cpu")))
    test_reconstructed = check_probability(predict(model, test.x, get_device("cpu")))
    saved = pd.read_csv(paths["predictions_test.csv"], dtype={"site_code": str})
    if saved["sample_id"].duplicated().any() or set(saved["sample_id"]) != set(test.sample_ids):
        raise ValueError("保存的测试预测样本与恢复的划分不一致。")
    saved = saved.set_index("sample_id").loc[test.sample_ids]
    if (not np.array_equal(saved["cognitive_impairment"].to_numpy(dtype=int), test.y.astype(int))
            or not np.array_equal(saved["site_code"].to_numpy(dtype=str), test.site_env.astype(str))):
        raise ValueError("保存预测的标签或站点不一致。")
    test_p = check_probability(saved["y_prob"].to_numpy(dtype=float))
    error = float(np.max(np.abs(test_reconstructed-test_p)))
    if not np.allclose(test_reconstructed, test_p, atol=2e-6, rtol=0):
        raise RuntimeError(f"{model_name}/{region}历史测试预测复核失败，最大误差={error}。")
    validation_auc = roc_auc_score(val.y, val_p)
    if not np.isclose(validation_auc, training["best_val_auc"], atol=5e-5, rtol=0):
        raise RuntimeError("恢复的验证集AUC与选模记录不一致。")
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    return val, test, val_p, test_p, error, hashes, tuple(train.sample_ids), tuple(val.sample_ids)


def draw_figures(pooled, site_metrics, comparisons, output):
    """任务四 第2步：展示整体概率、站点误差变化以及算法差距变化。"""
    figures = output / "figures"
    figures.mkdir()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, region in zip(axes, REGIONS):
        part = pooled.loc[pooled["held_out_region"].eq(region) & pooled["partition"].eq("target_test")]
        pivot = part.pivot(index="model", columns="stage", values="mean_probability").reindex(ALGORITHMS)
        pivot[["before", "after"]].plot.bar(ax=ax, rot=0)
        ax.axhline(part["positive_rate"].iloc[0], color="black", linestyle="--", label="Observed positive rate")
        ax.set_ylim(0, 1)
        ax.set_title(region)
        ax.set_xlabel("")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Mean predicted probability")
    fig.tight_layout()
    fig.savefig(figures / "target_mean_probability_before_after.png", dpi=220)
    plt.close(fig)

    before = site_metrics.loc[site_metrics["stage"].eq("before")].set_index(["held_out_region", "site_code", "model"])
    after = site_metrics.loc[site_metrics["stage"].eq("after")].set_index(["held_out_region", "site_code", "model"])
    delta = (after["brier"]-before["brier"]).unstack("model").reindex(columns=ALGORITHMS)
    order = [(r, s) for r in REGIONS for s in sorted(site_metrics.loc[site_metrics["held_out_region"].eq(r), "site_code"].unique())]
    delta = delta.reindex(pd.MultiIndex.from_tuples(order, names=["held_out_region", "site_code"]))
    limit = max(0.001, float(np.abs(delta.to_numpy()).max()))
    fig, ax = plt.subplots(figsize=(8, 8))
    plot = ax.imshow(delta.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(ALGORITHMS)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{r}: {s}" for r, s in order])
    for i in range(len(order)):
        for j in range(3):
            ax.text(j, i, f"{delta.iloc[i,j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("Target Brier change: after minus before (negative is better)")
    fig.colorbar(plot, ax=ax)
    fig.tight_layout()
    fig.savefig(figures / "site_brier_change.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    part = comparisons.loc[comparisons["model"].eq("RDM") & comparisons["reference"].eq("VREx")]
    pivot = part.pivot(index="region_scope", columns="stage", values="brier_difference").reindex([*REGIONS, "ALL"])
    pivot[["before", "after"]].plot.bar(ax=ax, rot=0)
    ax.axhline(0, color="grey")
    ax.set_ylabel("RDM minus VREx: macro Brier difference")
    ax.set_title("Does the RDM-VREx probability-error gap remain after adjustment?")
    fig.tight_layout()
    fig.savefig(figures / "rdm_vrex_brier_gap_before_after.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data" / "data_mission2.csv")
    parser.add_argument("--model-output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"输出目录已存在，禁止覆盖：{args.output}")
    if file_sha256(args.data) != EXPECTED_DATA_SHA256:
        raise ValueError("研究数据版本与历史实验不一致。")
    torch.set_num_threads(2)
    frame = attach_regions(read_raw(args.data, encoding="gbk"))
    if len(frame) != 4769 or frame[ENVIRONMENT].nunique() != 12:
        raise ValueError("研究样本数或站点数异常。")
    historical_path = HISTORY / "site_model_score_diagnostics.csv"
    historical = pd.read_csv(historical_path, encoding="utf-8-sig", dtype={"site_code": str})
    if historical.duplicated(["held_out_region", "site_code", "model"]).any():
        raise ValueError("历史指标记录重复。")
    sites, pooled, adjustments, checks, bins, manifest = [], [], [], [], [], {}
    split_ids = {}
    for region in REGIONS:
        for model_name in ALGORITHMS:
            print(f"任务四 第2步：恢复 {region}/{model_name} 原模型并核对历史结果。", flush=True)
            val, test, val_p, test_p, error, hashes, train_ids, val_ids = load_frozen_predictions(frame, model_name, region, args.model_output_root)
            current_split = (train_ids, val_ids, tuple(test.sample_ids))
            if region in split_ids and split_ids[region] != current_split:
                raise ValueError("三种算法的历史划分不一致。")
            split_ids[region] = current_split
            manifest[f"{region}/{model_name}"] = hashes
            # 任务四 第2步：拟合函数仅接收源域验证数据，不接收目标标签。
            intercept = fit_intercept(val_p, val.y)
            adjusted_val = adjust_probability(val_p, intercept)
            adjusted_test = adjust_probability(test_p, intercept)
            adjustments.append({"held_out_region": region, "model": model_name, "intercept": intercept,
                                "validation_n": len(val.y), "validation_positive_rate": float(val.y.mean()),
                                "validation_mean_probability_before": float(val_p.mean()),
                                "validation_mean_probability_after": float(adjusted_val.mean()),
                                "max_test_prediction_reproduction_error": error})
            for partition, y, before, after in (("source_validation", val.y, val_p, adjusted_val), ("target_test", test.y, test_p, adjusted_test)):
                info = {"held_out_region": region, "model": model_name, "partition": partition}
                pooled.extend(evaluate_pair(y, before, after, info))
                for stage, p in (("before", before), ("after", after)):
                    bins.extend(reliability_bins(y, p, {**info, "stage": stage}))
            for site in sorted(set(test.site_env.astype(str))):
                mask = test.site_env.astype(str) == site
                records = evaluate_pair(test.y[mask], test_p[mask], adjusted_test[mask], {"held_out_region": region, "site_code": site, "model": model_name})
                reference = historical.loc[historical["held_out_region"].eq(region) & historical["site_code"].eq(site) & historical["model"].eq(model_name)]
                if len(reference) != 1:
                    raise ValueError("缺少历史站点指标。")
                for metric in ("auc", "auprc", "brier"):
                    previous = float(reference[metric].iloc[0]); current = records[0][metric]
                    passed = bool(np.isclose(previous, current, atol=1e-10, rtol=0))
                    checks.append({"region": region, "site": site, "model": model_name, "metric": metric, "previous_value": previous, "current_value": current, "passed": passed})
                    if not passed:
                        raise RuntimeError("原始站点指标与历史结果不一致。")
                sites.extend(records)
    sites, pooled = pd.DataFrame(sites), pd.DataFrame(pooled)
    if len(sites) != 72 or len(pooled) != 36 or len(adjustments) != 9 or len(checks) != 108:
        raise RuntimeError("实验输出记录数异常。")
    macro_rows = []
    for region in (*REGIONS, "ALL"):
        subset = sites if region == "ALL" else sites.loc[sites["held_out_region"].eq(region)]
        for (model, stage), part in subset.groupby(["model", "stage"]):
            macro_rows.append({"region_scope": region, "model": model, "stage": stage, "site_count": len(part),
                               **{metric: float(part[metric].mean()) for metric in ("auc", "auprc", "brier", "log_loss")}})
    macros = pd.DataFrame(macro_rows)
    comparisons = []
    for (region, stage), part in macros.groupby(["region_scope", "stage"]):
        indexed = part.set_index("model")
        for left, right in (("RDM", "VREx"), ("RDM", "GroupDRO"), ("VREx", "GroupDRO")):
            comparisons.append({"region_scope": region, "stage": stage, "model": left, "reference": right,
                                **{f"{m}_difference": float(indexed.loc[left, m]-indexed.loc[right, m]) for m in ("auc", "auprc", "brier", "log_loss")}})
    comparisons = pd.DataFrame(comparisons)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, table in (("source_validation_adjustment.csv", pd.DataFrame(adjustments)), ("site_metrics_before_after.csv", sites),
                        ("pooled_region_metrics_before_after.csv", pooled), ("site_macro_metrics_before_after.csv", macros),
                        ("algorithm_macro_gaps_before_after.csv", comparisons), ("reliability_bins.csv", pd.DataFrame(bins)),
                        ("historical_reproduction_checks.csv", pd.DataFrame(checks))):
        table.to_csv(args.output / name, index=False, encoding="utf-8-sig")
    draw_figures(pooled, sites, comparisons, args.output)
    metadata = {"task": "Task 4 Step 2 - Source-validation intercept-only probability adjustment",
                "data_sha256": EXPECTED_DATA_SHA256, "models_retrained": False, "model_parameters_frozen": True,
                "adjustment_parameters_fitted": "One intercept per model and fold, source validation only; slope fixed at 1",
                "adjustment_objective": "Unweighted source-validation binary log-loss", "target_labels_used_for_fitting": False,
                "individual_records_exported": False, "input_artifact_hashes": manifest,
                "historical_site_metrics_sha256": file_sha256(historical_path),
                "limitations": [
                    "Source-validation calibration may fail to transfer to held-out regions; target error can increase.",
                    "Only overall log-odds level is adjusted, not calibration slope, feature relationships or the original learning rate.",
                    "AUC and average precision must stay unchanged in every evaluated group; checks stop on violations.",
                    "The same historical source validation was used for checkpoint selection; calibration is exploratory reuse, not an independent calibration cohort.",
                    "This is not a controlled algorithm-objective ablation and cannot attribute a difference to RDM or VREx mechanisms.",
                    "Before/after differences are descriptive; no new confidence intervals or general superiority claims are added.",
                    "Site macros are equal-weight means of site metrics; pooled-region metrics are separately named.",
                    "No target prevalence or target outcome is used to select the adjustment or its size.",
                    "The 220202/360102 case direction remains available for later research.",
                ]}
    (args.output / "experiment_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"任务四 第2步完成：原模型未训练；结果目录：{args.output}")


if __name__ == "__main__":
    main()

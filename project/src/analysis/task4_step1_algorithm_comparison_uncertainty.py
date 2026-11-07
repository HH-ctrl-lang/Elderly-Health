"""任务四 第1步：比较固定历史模型的成绩差异及测试样本不确定性。

仅重抽样已有测试预测；不训练、调参或选择特征。
同一站点各模型每次使用相同的阳性和阴性样本，进行公平配对。
地区与全局结果是固定站点的等权平均，不是合并个体AUC。
区间不包括重新训练、随机种子变化或新站点总体的不确定性。
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from task3_step1_site_failure_diagnostics import (
    ROOT, DATA_PATH, EXPECTED_DATA_SHA256, ENVIRONMENT, REGION, MODELS, REGIONS,
    attach_regions, read_raw, file_sha256, load_and_verify_predictions, compute_site_metrics,
)

PREVIOUS = ROOT / "outputs" / "analysis_2611" / "task3_step1_site_failure_diagnostics"
OUTPUT = ROOT / "outputs" / "analysis_2611" / "task4_step1_algorithm_comparison_uncertainty"
PAIRS = (("V5", "MLP"), ("RDM", "MLP"), ("VREx", "MLP"), ("GroupDRO", "MLP"),
         ("RDM", "VREx"), ("RDM", "GroupDRO"), ("VREx", "GroupDRO"))
METRICS = ("auc", "auprc", "brier")
SEED = 26110401


def calculate_metrics(labels, scores):
    """任务四 第1步：对同一批样本计算五个模型的三个指标。"""
    return np.asarray([[roc_auc_score(labels, scores[:, j]),
                        average_precision_score(labels, scores[:, j]),
                        brier_score_loss(labels, scores[:, j])]
                       for j in range(scores.shape[1])], dtype=float)


def verify_reference(observed, reference):
    """任务四 第1步：先验证60组历史成绩，防止输入结果混用。"""
    keys = ["held_out_region", "site_code", "model"]
    if reference.duplicated(keys).any() or observed.duplicated(keys).any():
        raise ValueError("历史指标包含重复记录。")
    merged = observed.merge(reference, on=keys, suffixes=("_now", "_previous"), how="outer", indicator=True, validate="one_to_one")
    if len(merged) != 60 or not merged["_merge"].eq("both").all():
        raise ValueError("模型、地区或站点记录不完整。")
    checks = []
    for row in merged.to_dict("records"):
        for metric in ("n", "positive_count", "positive_rate", *METRICS):
            previous, current = float(row[metric + "_previous"]), float(row[metric + "_now"])
            passed = bool(np.isclose(current, previous, atol=1e-10, rtol=0))
            checks.append({**{k: row[k] for k in keys}, "metric": metric,
                           "previous_value": previous, "current_value": current, "passed": passed})
            if not passed:
                raise RuntimeError(f"历史指标核对失败：{row['site_code']}/{row['model']}/{metric}。")
    return pd.DataFrame(checks)


def bootstrap_site(labels, scores, site, repeats, progress=False):
    """任务四 第1步：固定阳性/阴性人数，同一抽样索引用于所有模型。"""
    positive, negative = [np.flatnonzero(labels == k) for k in (1, 0)]
    if min(len(positive), len(negative)) < 2:
        raise ValueError(f"{site}的阳性或阴性人数不足。")
    seed = int.from_bytes(hashlib.sha256(f"{SEED}|{site}".encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    observed = calculate_metrics(labels, scores)
    draws = np.empty((repeats, len(MODELS), len(METRICS)), dtype=float)
    for i in range(repeats):
        sample = np.concatenate((rng.choice(positive, len(positive), replace=True),
                                 rng.choice(negative, len(negative), replace=True)))
        draws[i] = calculate_metrics(labels[sample], scores[sample])
        if progress and (i + 1) % 200 == 0:
            print(f"监测点 {site}：完成 {i+1}/{repeats} 次配对抽样", flush=True)
    return observed, draws


def difference_rows(observed, draws, scope, site, region, n, positives):
    """任务四 第1步：保存预先规定的7组模型差值，不计算后验胜出概率。"""
    records = []
    for model, reference in PAIRS:
        left, right = MODELS.index(model), MODELS.index(reference)
        for k, metric in enumerate(METRICS):
            delta = draws[:, left, k] - draws[:, right, k]
            low, high = np.quantile(delta, [0.025, 0.975])
            records.append({"scope": scope, "held_out_region": region, "site_code": site,
                            "n": n, "positive_count": positives, "model": model, "reference": reference,
                            "metric": metric, "model_value": float(observed[left, k]),
                            "reference_value": float(observed[right, k]),
                            "difference_model_minus_reference": float(observed[left, k] - observed[right, k]),
                            "ci_lower": float(low), "ci_upper": float(high),
                            "bootstrap_repeats": len(draws), "higher_is_better": metric != "brier"})
    return records


def draw_panel(ax, table, labels, title):
    """任务四 第1步：以差值及区间表示模型变化，零线表示成绩相同。"""
    y = np.arange(len(table))
    ax.hlines(y, table["ci_lower"], table["ci_upper"], color="tab:blue")
    ax.scatter(table["difference_model_minus_reference"], y, color="tab:blue", s=20)
    ax.axvline(0, color="grey", linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("ROC-AUC difference; unadjusted 95% bootstrap CI")


def draw_figures(site_table, macro_table, site_order, output):
    """任务四 第1步：绘制全站点和地区宏平均AUC对照，不新增单站点专属图。"""
    figures = output / "figures"
    figures.mkdir()
    auc = site_table.loc[site_table["metric"].eq("auc")]
    for pairs, name, shape in ((PAIRS[:4], "all_sites_methods_vs_mlp.png", (2, 2)),
                                (PAIRS[4:], "all_sites_external_method_comparisons.png", (1, 3))):
        fig, axes = plt.subplots(*shape, figsize=(15, 10 if shape[0] == 2 else 7), squeeze=False)
        for ax, (model, reference) in zip(axes.flat, pairs):
            table = auc.loc[auc["model"].eq(model) & auc["reference"].eq(reference)].set_index("site_code").loc[site_order]
            labels = [f"{row.held_out_region}: {site}" for site, row in table.iterrows()]
            draw_panel(ax, table, labels, f"{model} minus {reference}")
        fig.tight_layout()
        fig.savefig(figures / name, dpi=220)
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, region in zip(axes.flat, (*REGIONS, "ALL")):
        table = macro_table.loc[macro_table["metric"].eq("auc") & macro_table["held_out_region"].eq(region)].copy()
        labels = [f"{row.model} - {row.reference}" for row in table.itertuples(index=False)]
        draw_panel(ax, table, labels, f"{region}: fixed-site equal-weight mean")
    fig.tight_layout()
    fig.savefig(figures / "region_and_global_macro_auc_differences.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--previous-output", type=Path, default=PREVIOUS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bootstraps", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"输出目录已存在，禁止覆盖：{args.output}")
    if args.bootstraps < 200:
        raise ValueError("正式比较至少使用200次bootstrap。")
    if file_sha256(args.data) != EXPECTED_DATA_SHA256:
        raise ValueError("研究数据版本与历史模型不一致。")
    previous_meta_path = args.previous_output / "experiment_metadata.json"
    previous_metrics_path = args.previous_output / "site_model_score_diagnostics.csv"
    previous = json.loads(previous_meta_path.read_text(encoding="utf-8"))
    if previous["data_sha256"] != EXPECTED_DATA_SHA256:
        raise ValueError("历史诊断使用了不同版本的数据。")
    frame = attach_regions(read_raw(args.data, encoding="gbk"))
    if len(frame) != 4769 or frame[ENVIRONMENT].nunique() != 12:
        raise ValueError("研究样本数或站点数异常。")
    predictions, manifest = load_and_verify_predictions(frame)
    tokens = lambda sources: {(x["model"], x["held_out_region"], x["sha256"]) for x in sources}
    if len(manifest) != 15 or tokens(manifest) != tokens(previous["model_prediction_sources"]):
        raise ValueError("第3任务第1步以来历史预测文件发生变化。")
    observed_metrics = compute_site_metrics(frame, predictions)
    reference = pd.read_csv(previous_metrics_path, encoding="utf-8-sig", dtype={"site_code": str})
    checks = verify_reference(observed_metrics, reference)
    records, results, site_order = [], {}, []
    for region in REGIONS:
        subset = frame.loc[frame[REGION].eq(region)]
        for site, part in subset.groupby(ENVIRONMENT, sort=True):
            site = str(site)
            print(f"任务四 第1步：开始比较 {region}/{site}", flush=True)
            labels = part["label"].to_numpy(dtype=int)
            ids = part["sample_id"].to_numpy()
            scores = np.column_stack([predictions[m].loc[ids].to_numpy(dtype=float) for m in MODELS])
            observed, draws = bootstrap_site(labels, scores, site, args.bootstraps, progress=True)
            results[site] = {"region": region, "n": len(part), "positives": int(labels.sum()),
                             "observed": observed, "draws": draws}
            site_order.append(site)
            records.extend(difference_rows(observed, draws, "site", site, region, len(part), int(labels.sum())))
    macro_rows = []
    for region in (*REGIONS, "ALL"):
        selected = [r for r in results.values() if region == "ALL" or r["region"] == region]
        # 任务四 第1步：各站点独立抽样后等权平均，固定站点集合，不重抽站点。
        observed = np.mean([r["observed"] for r in selected], axis=0)
        draws = np.mean([r["draws"] for r in selected], axis=0)
        rows = difference_rows(observed, draws, "fixed_site_macro", "MACRO", region,
                               sum(r["n"] for r in selected), sum(r["positives"] for r in selected))
        macro_rows.extend({"site_count": len(selected), **row} for row in rows)
    sites, macros = pd.DataFrame(records), pd.DataFrame(macro_rows)
    if len(sites) != 12*len(PAIRS)*3 or len(macros) != 4*len(PAIRS)*3:
        raise RuntimeError("比较记录数量异常。")
    args.output.mkdir(parents=True, exist_ok=False)
    for name, table in (("paired_site_model_metric_intervals.csv", sites),
                        ("fixed_site_macro_model_metric_intervals.csv", macros),
                        ("historical_metric_reproduction_checks.csv", checks)):
        table.to_csv(args.output / name, index=False, encoding="utf-8-sig")
    draw_figures(sites, macros, site_order, args.output)
    metadata = {"task": "Task 4 Step 1 - Fixed-model paired comparison uncertainty",
                "data_sha256": EXPECTED_DATA_SHA256, "models": list(MODELS), "regions": list(REGIONS),
                "comparisons": [list(p) for p in PAIRS], "primary_metric": "auc", "secondary_metrics": ["auprc", "brier"],
                "seed": SEED, "bootstraps": args.bootstraps, "models_trained": False,
                "individual_records_exported": False, "input_predictions": manifest,
                "previous_metadata_sha256": file_sha256(previous_meta_path),
                "previous_metrics_sha256": file_sha256(previous_metrics_path),
                "causal_hypotheses_retained": {
                    "candidate_explanations": ["Model relies on noncausal or confounded associations", "Learned prediction function transfers poorly", "Population composition, measurement or outcome mechanisms differ"],
                    "status": "Unranked hypotheses; this bootstrap comparison does not identify direct causality."
                },
                "bootstrap": "Within-site outcome-stratified pairs, same sampled subjects for all models; site-specific independent random streams.",
                "macro": "Equal-weight average of fixed-site metrics in each bootstrap replicate, not pooled-person AUC or new-site inference.",
                "limitations": [
                    "Intervals measure test-subject sampling uncertainty conditional on these fitted models and the observed sites.",
                    "No training-seed, training-sample, site-population, label-error or prevalence uncertainty is included.",
                    "All intervals are exploratory, marginal and unadjusted for multiple comparisons; crossing zero is not evidence of equivalence.",
                    "Bootstrap frequency is not a posterior probability or a formal superiority test; no such claims are computed.",
                    "Small numbers of positives limit tail precision; 1000 replicates do not remove finite-sample uncertainty.",
                    "Average precision depends on prevalence; different sites are not exchangeable tasks and are not pooled.",
                    "Brier difference below zero favors the first model; AUC/AP difference above zero favors the first model.",
                    "A reliable fixed-model score difference still does not identify which algorithm mechanism or feature caused it.",
                    "Cross-environment marginal associations and prediction metrics do not identify direct causality.",
                    "The 220202/360102 case direction remains available after global comparisons.",
                ]}
    (args.output / "experiment_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"任务四 第1步完成：未训练模型；结果目录：{args.output}")


if __name__ == "__main__":
    main()

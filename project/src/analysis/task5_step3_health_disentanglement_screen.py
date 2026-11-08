"""任务五 第3步：健康数据上的一次有限解耦候选筛选，由用户运行。

固定两块自编码模型，只比较类别条件MMD权重0和0.1。
来源同类人群之间对齐，留出地区的特征和标签均不参与训练或选模。
这是受不变性思想启发的改编，不是论文原样复现或因果识别保证。
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
from elderly_health.data import FEATURES, ENVIRONMENT, REGION, attach_regions, read_raw, held_out_split, prepare_splits
from elderly_health.conditional_mmd_adapter import SETTINGS, fit, predict, representation_diagnostics
from task4_step2_source_probability_adjustment import fit_intercept, adjust_probability, evaluate_pair

REGIONS = ("east", "central", "west")
ARMS = {"two_block_control": 0.0, "two_block_conditional_mmd": 0.1}
OUTPUT = ROOT / "outputs/analysis_2611/task5_step3_health_disentanglement_screen"
EXPECTED_DATA_SHA256 = "969f0c1c82faafcfa402f37572e4a8ceb324492062cf1e590465f82dc6b5b6de"


def digest(path):
    """任务五 第3步：记录数据与代码版本，不导出逐人记录。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    """任务五 第3步：保存配置、训练记录及检查结果。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def macro_metrics(site):
    """任务五 第3步：逐站点评分后等权平均，地区合并指标另存。"""
    rows = []
    for scope in (*REGIONS, "ALL"):
        selected = site if scope == "ALL" else site.loc[site.held_out_region.eq(scope)]
        for (model, stage), group in selected.groupby(["model", "stage"]):
            rows.append({"scope": scope, "model": model, "stage": stage, "site_count": len(group),
                         **group[["auc", "auprc", "brier", "log_loss"]].mean().to_dict()})
    return pd.DataFrame(rows)


def compare_macros(macro):
    """任务五 第3步：同时报告相同结构对照与历史MLP参考，不自动挑赢家。"""
    rows = []
    treatment = macro.loc[macro.model.eq("two_block_conditional_mmd")]
    for row in treatment.to_dict("records"):
        for reference in ("two_block_control", "MLP_historical_reference"):
            old = macro.loc[macro.scope.eq(row["scope"]) & macro.stage.eq(row["stage"]) & macro.model.eq(reference)]
            if old.empty:
                continue
            old = old.iloc[0]
            rows.append({"scope": row["scope"], "stage": row["stage"], "reference": reference,
                         **{f"delta_{metric}": float(row[metric]-old[metric])
                            for metric in ("auc", "auprc", "brier")}})
    return pd.DataFrame(rows)


def draw_figures(site, macro, output):
    """任务五 第3步：突出真实任务的站点变化和地区宏指标。"""
    figures = output / "figures"
    figures.mkdir()
    raw = site.loc[site.stage.eq("before") & site.model.isin(ARMS)]
    pivot = raw.pivot(index=["held_out_region", "site_code"], columns="model", values="auc")
    order = [(region, code) for region in REGIONS
             for code in sorted(raw.loc[raw.held_out_region.eq(region), "site_code"].unique())]
    pivot = pivot.reindex(pd.MultiIndex.from_tuples(order, names=["held_out_region", "site_code"]))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(pivot)), pivot["two_block_conditional_mmd"]-pivot["two_block_control"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(pivot)))
    ax.set_xticklabels([f"{r}: {s}" for r, s in order], rotation=55, ha="right")
    ax.set_ylabel("Site AUC: conditional MMD minus identical control")
    ax.set_title("Health-data screening; seed 42; not a causal identification test")
    fig.tight_layout()
    fig.savefig(figures / "site_auc_change.png", dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ("auc", "auprc")):
        table = macro.loc[macro.stage.eq("before")].pivot(index="scope", columns="model", values=metric)
        table = table.reindex([*REGIONS, "ALL"])
        table.plot.bar(ax=ax, rot=0)
        ax.set_ylabel(f"Equal-site macro {metric.upper()}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "region_and_global_macro_performance.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data/data_mission2.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"禁止覆盖已有结果：{args.output}")
    if digest(args.data) != EXPECTED_DATA_SHA256:
        raise ValueError("数据版本与历史结果不同。")
    frame = attach_regions(read_raw(args.data, encoding="gbk"))
    if len(frame) != 4769 or frame[ENVIRONMENT].nunique() != 12:
        raise ValueError("样本数量或站点数量发生变化。")
    reference_path = ROOT / "outputs/analysis_2611/task3_step1_site_failure_diagnostics/site_model_score_diagnostics.csv"
    reference = pd.read_csv(reference_path, dtype={"site_code": str})
    reference = reference.loc[reference.model.eq("MLP")].copy()
    if len(reference) != 12 or reference.site_code.duplicated().any():
        raise ValueError("历史MLP参考站点不完整。")
    for row in reference.to_dict("records"):
        part = frame.loc[frame[ENVIRONMENT].eq(row["site_code"])]
        if (len(part) != row["n"] or int(part.label.sum()) != row["positive_count"]
                or set(part[REGION]) != {row["held_out_region"]}):
            raise ValueError("历史MLP参考与当前数据不一致。")
    import torch
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "experiment_design.json", {
        "task": "Task 5 Step 3: Limited health-data candidate screening",
        "data_sha256": EXPECTED_DATA_SHA256, "settings": SETTINGS, "arms": ARMS,
        "historical_mlp_reference_sha256": digest(reference_path),
        "code_sha256": {"runner": digest(Path(__file__)),
                        "adapter": digest(PROJECT / "src/elderly_health/conditional_mmd_adapter.py")},
        "torch_version": torch.__version__, "individual_records_exported": False,
        "primary_evaluation": "Equal-site macro AUC and AUPRC on all 12 held-out sites; report all regions and sites",
        "practical_attention_threshold": "Macro AUC increase about 0.01 is a screening signal only; also inspect AUPRC, regional losses and historical MLP. No automatic pass decision.",
        "limitations": ["Inspired adaptation, not a faithful paper replication or causal identification.",
                        "Class-conditional source alignment is an extra hypothesis and can still remove useful information.",
                        "Residual block is not guaranteed to be environment-only or independent of prediction block.",
                        "Prediction-block LayerNorm limits simple overall rescaling freedom but does not guarantee freedom from collapse.",
                        "Numeric coding and standardized MSE reconstruction are retained for this limited paired screen; categorical-code distances have no clinical interpretation.",
                        "One seed and one nonzero weight; no target data used for fitting or hyperparameter selection.",
                        "Historical MLP differs in architecture and training; it is a practical reference, not the mechanism control.",
                        "Source validation reused for intercept adjustment; probability differences remain exploratory."]})
    site_rows, pooled_rows, diag_rows, shifts, controls = [], [], [], [], []
    for region in REGIONS:
        frames = held_out_split(frame, region, seed=42, column=REGION)
        (train, val, test), preprocessor = prepare_splits(frames, environment_column=REGION)
        if set(test.sample_ids) & (set(train.sample_ids) | set(val.sample_ids)):
            raise RuntimeError("留出地区样本与来源数据交叉。")
        fold = args.output / region
        fold.mkdir()
        np.savez(fold / "preprocessing.npz", median=preprocessor.imputer.statistics_,
                 mean=preprocessor.scaler.mean_, scale=preprocessor.scaler.scale_)
        write_json(fold / "split_manifest.json", {
            name: {"n": len(split.y), "positive_count": int(split.y.sum()),
                   "sample_ids_sha256": hashlib.sha256(np.asarray(split.sample_ids, dtype=np.int64).tobytes()).hexdigest()}
            for name, split in (("train", train), ("validation", val), ("test", test))})
        reports = {}
        for arm, weight in ARMS.items():
            print(f"任务五 第3步：留出{region}, {arm}", flush=True)
            model, report = fit(train, val, weight)
            reports[arm] = report
            write_json(fold / f"{arm}_training.json", report)
            torch.save({"state_dict": model.state_dict(), "settings": SETTINGS,
                        "mmd_weight": weight, "features": list(FEATURES), "best_epoch": report["best_epoch"]},
                       fold / f"{arm}.pt")
            # 任务五 第3步：两组均先由来源验证AUC选模，之后才接触留出数据进行评价。
            val_p = predict(model, val.x)
            test_p = predict(model, test.x)
            intercept = fit_intercept(val_p, val.y)
            adjusted = adjust_probability(test_p, intercept)
            info = {"held_out_region": region, "model": arm}
            shifts.append({**info, "intercept": intercept, "best_epoch": report["best_epoch"], "source_val_auc": report["best_val_auc"]})
            pooled_rows.extend(evaluate_pair(test.y, test_p, adjusted, info))
            for code in sorted(set(test.site_env)):
                mask = test.site_env == code
                site_rows.extend(evaluate_pair(test.y[mask], test_p[mask], adjusted[mask], {**info, "site_code": str(code)}))
            for name, split in (("source_validation", val), ("target_test", test)):
                diag_rows.append({**info, "partition": name, **representation_diagnostics(model, split)})
        left, right = reports.values()
        common = set(left["sampling_prefix_sha256"]) & set(right["sampling_prefix_sha256"])
        if (left["initial_weights_sha256"] != right["initial_weights_sha256"] or not common
                or any(left["sampling_prefix_sha256"][key] != right["sampling_prefix_sha256"][key] for key in common)):
            raise RuntimeError("两组初始化或共同训练轮次的抽样不同。")
        controls.append({"region": region, "same_initial_weights": True,
                         "same_sampling_common_epochs": len(common), "passed": True})
        pd.DataFrame(controls).to_csv(args.output / "control_checks.csv", index=False, encoding="utf-8-sig")
    site = pd.DataFrame(site_rows)
    if len(site) != 48:
        raise RuntimeError("站点指标数量不完整。")
    macro = macro_metrics(site)
    old_macro = macro_metrics(reference.assign(model="MLP_historical_reference", stage="before", log_loss=np.nan))
    macro = pd.concat([macro, old_macro], ignore_index=True)
    comparisons = compare_macros(macro)
    for name, table in (("site_metrics_before_after.csv", site),
                        ("historical_mlp_site_reference.csv", reference),
                        ("site_macro_metrics_before_after.csv", macro),
                        ("candidate_macro_differences.csv", comparisons),
                        ("pooled_region_metrics_before_after.csv", pd.DataFrame(pooled_rows)),
                        ("representation_diagnostics.csv", pd.DataFrame(diag_rows)),
                        ("source_validation_adjustment.csv", pd.DataFrame(shifts))):
        table.to_csv(args.output / name, index=False, encoding="utf-8-sig")
    draw_figures(site, macro, args.output)
    write_json(args.output / "completion.json", {"completed": True, "training_runs": 6,
                                                 "test_sites": 12, "all_control_checks_passed": True})
    print(f"任务五 第3步完成，请提交整个目录：{args.output}")


if __name__ == "__main__":
    main()

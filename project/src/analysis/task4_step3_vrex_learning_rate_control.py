"""任务四 第3步：VREx单因素学习率对照，由用户运行训练。

两组均使用历史划分、种子42、原VREx损失和第500次更新前的Adam重置。
唯一干预：处理组在已完成1500次更新后，将学习率乘0.05，不重置Adam。
这只检验VREx对该学习率设置的敏感性，不是RDM与VREx损失的直接对照。
运行：python project/src/analysis/task4_step3_vrex_learning_rate_control.py
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

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
from elderly_health.data import (  # noqa: E402
    FEATURES, ENVIRONMENT, REGION, attach_regions, read_raw, held_out_split, prepare_splits,
)
from elderly_health.training import get_device, predict  # noqa: E402
from elderly_health.vrex_adapter import VRExSettings, fit_vrex  # noqa: E402
from task3_step1_site_failure_diagnostics import EXPECTED_DATA_SHA256, file_sha256  # noqa: E402
from task4_step2_source_probability_adjustment import (  # noqa: E402
    load_frozen_predictions, fit_intercept, adjust_probability, evaluate_pair,
)

OUTPUT = ROOT / "outputs/analysis_2611/task4_step3_vrex_learning_rate_control"
REGIONS = ("east", "central", "west")
ARMS = ("original_lr", "drop_lr_20x")
DROP_STEP, DROP_FACTOR = 1500, 0.05


def write_json(path, value):
    """任务四 第3步：保存可复核配置和记录，不生成阶段结论。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def verify_settings(settings, rdm_config):
    """任务四 第3步：锁定实验设置，拒绝在读取结果后调参。"""
    expected = VRExSettings().to_dict()
    if settings.to_dict() != expected:
        raise ValueError("历史VREx设置与预定对照不同，停止而不自动适配。")
    rdm = rdm_config["config"]
    if (rdm["rdm_penalty_anneal_iters"] != DROP_STEP
            or not np.isclose(rdm["rdm_lr"] / rdm["learning_rate"], DROP_FACTOR)):
        raise ValueError("历史RDM的学习率时点或比例已变化。")


def verify_prefix(control, treatment):
    """任务四 第3步：两组在干预发生前的每轮训练记录必须完全相同。"""
    prefix = [row for row in control["history"] if row["updates"] <= DROP_STEP]
    if not prefix or len(treatment["history"]) < len(prefix):
        raise RuntimeError("缺少干预前记录。")
    for old, new in zip(prefix, treatment["history"]):
        for key in ("epoch", "updates", "val_auc", "mean_risk", "risk_variance"):
            if old[key] != new[key]:
                raise RuntimeError(f"干预前训练轨迹不同：{key}，epoch={old['epoch']}")
    events = treatment["learning_rate_intervention"]["events"]
    if len(events) != 1 or events[0]["step_zero_based"] != DROP_STEP or events[0]["optimizer_reset"]:
        raise RuntimeError("学习率干预未按预定方案执行。")
    if control["learning_rate_intervention"]["events"]:
        raise RuntimeError("对照组不应发生额外学习率变化。")
    return len(prefix)


def macro_tables(site_metrics):
    """任务四 第3步：先逐站点评分，再等权平均，地区合并指标另存。"""
    columns = ["auc", "auprc", "brier", "log_loss"]
    records = []
    for scope in (*REGIONS, "ALL"):
        part = site_metrics if scope == "ALL" else site_metrics.loc[site_metrics.held_out_region.eq(scope)]
        for (arm, stage), group in part.groupby(["arm", "stage"]):
            records.append({"scope": scope, "arm": arm, "stage": stage,
                            "site_count": len(group), **group[columns].mean().to_dict()})
    macro = pd.DataFrame(records)
    differences = []
    for (scope, stage), group in macro.groupby(["scope", "stage"]):
        indexed = group.set_index("arm")
        differences.append({"scope": scope, "stage": stage,
                            **{c: float(indexed.loc[ARMS[1], c]-indexed.loc[ARMS[0], c]) for c in columns}})
    return macro, pd.DataFrame(differences)


def draw_figures(site, differences, output):
    """任务四 第3步：展示排序变化和原始/调整后的概率误差变化。"""
    figures = output / "figures"
    figures.mkdir()
    after = site.loc[site.stage.eq("after")]
    order = after[["held_out_region", "site_code"]].drop_duplicates().sort_values(["held_out_region", "site_code"])
    values = after.pivot(index="site_code", columns="arm", values="auc").loc[order.site_code]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(values)), values[ARMS[1]]-values[ARMS[0]])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([f"{r.held_out_region}: {r.site_code}" for r in order.itertuples()], rotation=55, ha="right")
    ax.set_ylabel("Site AUC: lower LR minus original LR")
    ax.set_title("VREx learning-rate control (seed 42; descriptive)")
    fig.tight_layout()
    fig.savefig(figures / "site_auc_change.png", dpi=220)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    values = differences.pivot(index="scope", columns="stage", values="brier").loc[[*REGIONS, "ALL"]]
    values[["before", "after"]].plot.bar(ax=ax, rot=0)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Macro Brier: lower LR minus original LR (negative is better)")
    ax.set_title("Before / after source-validation probability adjustment")
    fig.tight_layout()
    fig.savefig(figures / "macro_brier_change_before_after_adjustment.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data/data_mission2.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"禁止覆盖现有结果：{args.output}")
    if file_sha256(args.data) != EXPECTED_DATA_SHA256:
        raise RuntimeError("数据版本不同，停止对照。")
    frame = attach_regions(read_raw(args.data, encoding="gbk"))
    if len(frame) != 4769 or frame[ENVIRONMENT].nunique() != 12:
        raise ValueError("样本或站点数不同。")

    # 任务四 第3步：先复核全部历史输入，再创建新输出目录。
    prepared, manifest = {}, {}
    for region in REGIONS:
        base = ROOT / "outputs/external-vrex-region-loro-42" / region
        config_path = base / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        rdm_path = ROOT / "outputs/external-rdm-region-loro-42" / region / "config.json"
        settings = VRExSettings(**config["config"])
        verify_settings(settings, json.loads(rdm_path.read_text(encoding="utf-8")))
        val_old, test_old, _, historical_p, error, hashes, train_ids, val_ids = load_frozen_predictions(
            frame, "VREx", region, ROOT / "outputs")
        splits = held_out_split(frame, region, seed=settings.seed, column=REGION)
        (train, val, test), _ = prepare_splits(splits, environment_column=REGION)
        if (tuple(train.sample_ids) != train_ids or tuple(val.sample_ids) != val_ids
                or not np.array_equal(test.sample_ids, test_old.sample_ids)
                or not np.array_equal(val.y, val_old.y)):
            raise RuntimeError("划分与历史模型不一致。")
        if DROP_STEP >= (settings.selection_start_epoch-1) * int(np.ceil(len(train.x)/64)):
            raise RuntimeError("干预必须发生在允许选模之前。")
        prepared[region] = (train, val, test, settings, historical_p)
        manifest[region] = {"historical_files": hashes, "rdm_config_sha256": file_sha256(rdm_path),
                            "historical_reconstruction_max_error": error}

    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "experiment_design.json", {
        "data_sha256": EXPECTED_DATA_SHA256, "features": list(FEATURES),
        "arms": list(ARMS), "drop_step_zero_based": DROP_STEP, "drop_factor": DROP_FACTOR,
        "extra_optimizer_reset": False, "source_artifacts": manifest,
        "code_sha256": {"runner": file_sha256(Path(__file__)),
                        "vrex_adapter": file_sha256(PROJECT / "src/elderly_health/vrex_adapter.py")},
        "runtime": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__},
        "selection": "source validation AUC from epoch 91, patience 80; identical between arms",
        "individual_records_exported": False,
        "limitations": ["One seed only; exploratory sensitivity analysis, not algorithm superiority.",
                        "Matches RDM drop timing and ratio, but deliberately does not copy its Adam reset.",
                        "Separate validation-selected checkpoints allow early-stopping effects to be part of the intervention.",
                        "Source validation reused for probability adjustment; no target labels used for fitting.",
                        "No new confidence intervals; cannot attribute results to risk-distribution versus variance constraints."]})
    site_rows, pooled_rows, adjustments, audits = [], [], [], []
    for region, (train, val, test, settings, historical_p) in prepared.items():
        print(f"\n任务四 第3步：留出{region}，先复现原版VREx，再运行降低学习率组。", flush=True)
        reports = {}
        for arm in ARMS:
            directory = args.output / region / arm
            directory.mkdir(parents=True)
            kwargs = {} if arm == ARMS[0] else {"lr_drop_step": DROP_STEP, "lr_drop_factor": DROP_FACTOR}
            model, report = fit_vrex(train, val, settings, **kwargs)
            reports[arm] = report
            probability = predict(model, test.x, get_device("cpu"))
            if arm == ARMS[0]:
                error = float(np.max(np.abs(probability-historical_p)))
                old = json.loads((ROOT / "outputs/external-vrex-region-loro-42" / region / "training.json").read_text(encoding="utf-8"))
                if (not np.allclose(probability, historical_p, atol=2e-6, rtol=0)
                        or report["best_epoch"] != old["best_epoch"]
                        or not np.isclose(report["best_val_auc"], old["best_val_auc"], atol=1e-12, rtol=0)):
                    raise RuntimeError(f"{region}原版VREx未能复现历史结果；停止干预比较，最大概率误差={error}")
                audits.append({"region": region, "check": "baseline_historical_max_probability_error", "value": error, "passed": True})
            else:
                count = verify_prefix(reports[ARMS[0]], report)
                audits.append({"region": region, "check": "identical_preintervention_epochs", "value": count, "passed": True})
            write_json(directory / "training.json", report)
            write_json(directory / "config.json", {"settings": settings.to_dict(), "intervention": kwargs})
            torch.save({"state_dict": model.state_dict(), "features": list(FEATURES),
                        "settings": settings.to_dict(), "intervention": kwargs,
                        "best_epoch": report["best_epoch"]}, directory / "best_model.pt")
            val_p = predict(model, val.x, get_device("cpu"))
            shift = fit_intercept(val_p, val.y)
            adjusted = adjust_probability(probability, shift)
            adjustments.append({"held_out_region": region, "arm": arm, "intercept": shift,
                                "best_epoch": report["best_epoch"], "best_val_auc": report["best_val_auc"],
                                "source_validation_n": len(val.y),
                                "source_validation_positive_rate": float(val.y.mean()),
                                "source_validation_mean_probability_before": float(val_p.mean()),
                                "source_validation_mean_probability_after": float(adjust_probability(val_p, shift).mean())})
            pooled_rows.extend(evaluate_pair(test.y, probability, adjusted, {"held_out_region": region, "arm": arm}))
            for site in sorted(set(test.site_env)):
                selected = test.site_env == site
                site_rows.extend(evaluate_pair(test.y[selected], probability[selected], adjusted[selected],
                                                {"held_out_region": region, "site_code": str(site), "arm": arm}))
        pd.DataFrame(audits).to_csv(args.output / "control_checks.csv", index=False, encoding="utf-8-sig")

    site = pd.DataFrame(site_rows)
    if len(site) != 12 * 2 * 2:
        raise RuntimeError("站点输出数量错误。")
    macro, differences = macro_tables(site)
    for name, table in (("site_metrics_before_after.csv", site),
                        ("pooled_region_metrics_before_after.csv", pd.DataFrame(pooled_rows)),
                        ("site_macro_metrics_before_after.csv", macro),
                        ("learning_rate_macro_differences.csv", differences),
                        ("source_validation_adjustment.csv", pd.DataFrame(adjustments))):
        table.to_csv(args.output / name, index=False, encoding="utf-8-sig")
    draw_figures(site, differences, args.output)
    write_json(args.output / "completion.json", {"completed": True, "folds": 3, "training_runs": 6,
                                                "all_control_checks_passed": True})
    print(f"任务四 第3步完成。请提交整个目录：{args.output}", flush=True)


if __name__ == "__main__":
    main()

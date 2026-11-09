"""任务2 第4步：来源选模与东部表现的逐轮诊断。

沿用第3步未加权完整MLP，固定划分42，仅重复训练种子42/43/44。
训练中只复制模型快照；训练结束后才计算东部指标。仅来源指标可选模型。
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch

PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
from elderly_health.config import TrainConfig
from elderly_health.data import REGION, attach_regions, held_out_split, prepare_splits, read_raw
from elderly_health.evaluation import metrics
from elderly_health.training import fit, get_device, predict

EXPECTED_HASH = "969f0c1c82faafcfa402f37572e4a8ceb324492062cf1e590465f82dc6b5b6de"


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def grouped_metrics(split, probability, grouping):
    """任务2 第4步：返回各组指标；单类别组AUC保留空值，不伪造为0.5。"""
    return {str(group): metrics(split.y[grouping == group], probability[grouping == group]) for group in sorted(set(grouping))}


def safe_macro(groups, metric):
    values = [row[metric] for row in groups.values()]
    return float(np.mean(values)) if values and all(v is not None for v in values) else None


def pick_epochs(curve):
    """任务2 第4步：四种预先声明的来源选模规则，只比较同一训练轨迹。"""
    policies = {"source_pooled_auc": ("source_val_auc", False),
                "source_site_macro_auc": ("source_val_site_macro_auc", False),
                "source_region_macro_auc": ("source_val_region_macro_auc", False),
                "source_validation_loss": ("source_val_weighted_bce", True)}
    result = {}
    for name, (column, minimum) in policies.items():
        valid = curve.loc[curve[column].notna()]
        if valid.empty:
            result[name] = None
        else:
            # 同分选择最早轮次，与原fit严格大于才更新的规则一致。
            index = valid[column].idxmin() if minimum else valid[column].idxmax()
            result[name] = int(valid.loc[index, "epoch"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data/data_mission2.csv")
    parser.add_argument("--previous", type=Path, default=ROOT / "outputs/analysis_2611/task2_step3_east_controlled_followup")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/analysis_2611/task2_step4_east_checkpoint_diagnostics")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("训练种子重复")
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != EXPECTED_HASH:
        raise ValueError("固定CSV版本不一致")
    meta = json.loads((args.previous / "experiment_metadata.json").read_text(encoding="utf-8"))
    versions = {"python": platform.python_version(), "torch": torch.__version__, "sklearn": sklearn.__version__}
    for key, value in versions.items():
        if meta[key] != value:
            raise ValueError(f"{key}与第3步环境不一致：{value} vs {meta[key]}，请先恢复原实验环境")
    prior_metrics = pd.read_csv(args.previous / "metrics.csv")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {"completed": False, "input_sha256": EXPECTED_HASH, "split_seed": 42,
                "training_seeds": args.seeds, "versions": versions,
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "target_evaluated_after_training": True, "target_used_for_selection": False,
                "scope": "未加权完整MLP；纯域泛化；来源划分不变；同一轨迹内比较选模规则"}
    dump(args.output / "experiment_metadata.json", metadata)
    frame = attach_regions(read_raw(args.data))
    (train, val, test), _ = prepare_splits(held_out_split(frame, "east", seed=42, column=REGION), environment_column=REGION)
    curves, sites, source_sites, selected, audits = [], [], [], [], []
    for seed in args.seeds:
        cfg = TrainConfig(model="baseline", environment_granularity="region", seed=seed, selection_start_epoch=1, device="cpu")
        folder = args.output / str(seed)
        folder.mkdir()
        snapshots = []

        def capture(model, record):
            # 任务2 第4步：仅复制张量，不执行推理，不消耗随机数，不读取东部结局。
            snapshots.append((record, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}))

        print(f"任务2 第4步：重现seed={seed}完整MLP训练", flush=True)
        model, training = fit(train, val, cfg, epoch_callback=capture)
        dump(folder / "training.json", training)
        # 任务2 第4步：先核对第3步模型是否重现，再进行轨迹诊断。
        original = json.loads((args.previous / "east" / str(seed) / "full/training.json").read_text(encoding="utf-8"))
        actual = metrics(test.y, predict(model, test.x, get_device("cpu")))
        old = prior_metrics.loc[(prior_metrics.region == "east") & (prior_metrics.seed == seed) & (prior_metrics.arm == "full") & (prior_metrics.scope == "region_pooled")]
        if len(old) != 1:
            raise ValueError(f"seed={seed}第3步完整模型记录不唯一")
        ok = training["best_epoch"] == original["best_epoch"] and all(np.isclose(actual[k], old.iloc[0][k], atol=2e-6, rtol=0) for k in ("auc", "auprc", "brier"))
        audit = {"seed": seed, "reproduced": bool(ok), "best_epoch": training["best_epoch"], "previous_best_epoch": original["best_epoch"], **{f"difference_{k}": float(actual[k]-old.iloc[0][k]) for k in ("auc", "auprc", "brier")}}
        audits.append(audit)
        dump(folder / "reproduction_audit.json", audit)
        if not ok:
            raise RuntimeError("原实验未准确重现，停止机制解释；请提交reproduction_audit.json核对")
        for record, state in snapshots:
            epoch = int(record["epoch"])
            model.load_state_dict(state)
            pv = predict(model, val.x, get_device("cpu"))
            pt = predict(model, test.x, get_device("cpu"))
            ps = predict(model, train.x, get_device("cpu"))
            vg = grouped_metrics(val, pv, val.site_env)
            vr = grouped_metrics(val, pv, val.env)
            tg = grouped_metrics(test, pt, test.site_env)
            vm, tm, sm = metrics(val.y, pv), metrics(test.y, pt), metrics(train.y, ps)
            row = {"seed": seed, "epoch": epoch, "source_train_auc": sm["auc"], "source_val_auc": record["val_auc"],
                   "source_val_site_macro_auc": safe_macro(vg, "auc"), "source_val_region_macro_auc": safe_macro(vr, "auc"),
                   "source_val_weighted_bce": record["val_loss"], "source_val_brier": vm["brier"],
                   "valid_source_sites": sum(v["auc"] is not None for v in vg.values()),
                   "source_site_count": len(vg), "east_site_macro_auc": safe_macro(tg, "auc"),
                   **{f"east_{key}": tm[key] for key in ("auc", "auprc", "brier")}}
            curves.append(row)
            for site, values in tg.items():
                sites.append({"seed": seed, "epoch": epoch, "site_code": site, **values})
            for site, values in vg.items():
                source_sites.append({"seed": seed, "epoch": epoch, "site_code": site, **values})
        curve = pd.DataFrame([r for r in curves if r["seed"] == seed])
        policies = pick_epochs(curve)
        if policies["source_pooled_auc"] != training["best_epoch"]:
            raise RuntimeError("来源AUC选模轮次重算不一致")
        for policy, epoch in policies.items():
            if epoch is None:
                selected.append({"seed": seed, "policy": policy, "available": False})
                continue
            row = curve.loc[curve.epoch == epoch].iloc[0].to_dict()
            selected.append({"policy": policy, "available": True, **row})
        dump(folder / "selection_epochs.json", policies)
        pd.DataFrame(curves).to_csv(args.output / "epoch_curves.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(sites).to_csv(args.output / "east_site_epoch_metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(source_sites).to_csv(args.output / "source_validation_site_epoch_metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(selected).to_csv(args.output / "source_policy_comparison.csv", index=False, encoding="utf-8-sig")
        del snapshots
    # 任务2 第4步：逐种子对照，仅描述这条固定划分上的选模敏感性。
    comparison = pd.DataFrame(selected)
    delta_rows = []
    for seed, group in comparison.groupby("seed"):
        base = group.loc[group.policy == "source_pooled_auc"].iloc[0]
        for _, row in group.loc[group.available].iterrows():
            delta_rows.append({"seed": seed, "policy": row.policy, "epoch": row.epoch,
                               **{f"delta_{k}": row[k]-base[k] for k in ("east_auc", "east_auprc", "east_brier", "east_site_macro_auc")}})
    pd.DataFrame(delta_rows).to_csv(args.output / "paired_policy_deltas.csv", index=False, encoding="utf-8-sig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = args.output / "figures"
    figures.mkdir()
    curve_all = pd.DataFrame(curves)
    site_all = pd.DataFrame(sites)
    for seed in args.seeds:
        part = curve_all.loc[curve_all.seed == seed]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for key in ("source_train_auc", "source_val_auc", "source_val_site_macro_auc", "east_auc"):
            axes[0].plot(part.epoch, part[key], label=key)
        for site, group in site_all.loc[site_all.seed == seed].groupby("site_code"):
            axes[1].plot(group.epoch, group.auc, label=str(site))
        chosen = int(comparison.loc[(comparison.seed == seed) & (comparison.policy == "source_pooled_auc"), "epoch"].iloc[0])
        for ax in axes:
            ax.axvline(chosen, color="black", linestyle="--", label="Source-selected epoch")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("AUC")
            ax.legend(fontsize=7)
        fig.suptitle(f"Seed {seed}: target curves are retrospective diagnostics only")
        fig.tight_layout()
        fig.savefig(figures / f"seed_{seed}_selection_and_site_curves.png", dpi=180)
        plt.close(fig)
    metadata.update(completed=True, reproduction_audits=audits, limitations="同一来源验证集和东部数据反复用于探索；不能把规则改进视为新外部验证。各规则共用原来源AUC早停轨迹，不代表各自独立早停的算法。")
    dump(args.output / "experiment_metadata.json", metadata)
    print(f"任务2 第4步完成：{args.output}")


if __name__ == "__main__":
    main()

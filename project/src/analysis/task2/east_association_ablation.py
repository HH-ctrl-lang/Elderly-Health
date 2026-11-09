"""任务2 第6步：验证婚姻状态、年龄的来源关联是否实际妨碍东部预测。

固定19维结构，训练/验证/测试同时屏蔽指定列；三种子、有限随机双列对照。
同时比较原模型固定轮次与各处理自己的来源选模，区分信息变化和选模变化。
本步是模型信息使用干预，不是对人的年龄、婚姻状态实施因果干预。
"""
import argparse
import hashlib
import json
import platform
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch

PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
from elderly_health.config import TrainConfig
from elderly_health.data import FEATURES, REGION, attach_regions, held_out_split, prepare_splits, read_raw
from elderly_health.evaluation import metrics
from elderly_health.training import fit, get_device, predict

EXPECTED_HASH = "969f0c1c82faafcfa402f37572e4a8ceb324492062cf1e590465f82dc6b5b6de"
SEEDS = (42, 43, 44)
POLICIES = ("matched_source_auc_epoch", "matched_source_loss_epoch", "own_source_auc", "own_source_loss")


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def mask(split, removed):
    """任务2 第6步：屏蔽标准化后的信息，保持模型参数量及随机初始化一致。"""
    x = split.x.copy()
    x[:, [FEATURES.index(f) for f in removed]] = 0
    return replace(split, x=x)


def make_arms():
    """任务2 第6步：在查看本轮结果前固定候选和三个互不重叠的随机双列组。"""
    pool = [f for f in FEATURES if f not in ("marital_status", "age")]
    chosen = np.random.RandomState(26110906).choice(pool, size=6, replace=False).tolist()
    return {"full": (), "mask_marital": ("marital_status",), "mask_age": ("age",),
            "mask_marital_age": ("marital_status", "age"),
            **{f"random_pair_{i+1}": tuple(chosen[2*i:2*i+2]) for i in range(3)}}


def evaluate(model, splits, seed, arm, policy, epoch):
    """任务2 第6步：来源与东部同时报告；宏指标与合并指标分开。"""
    rows = []
    for scope, split in (("source_validation", splits[1]), ("east", splits[2])):
        p = predict(model, split.x, get_device("cpu"))
        base = {"seed": seed, "arm": arm, "policy": policy, "epoch": epoch, "scope": scope}
        rows.append({**base, "group": "POOL", **metrics(split.y, p)})
        grouped = []
        for site in sorted(set(split.site_env)):
            ix = split.site_env == site
            m = metrics(split.y[ix], p[ix])
            grouped.append(m)
            rows.append({**base, "group": str(site), **m})
        # 所有站点均有可估计值时才报告宏值，单类别站点不悄悄排除。
        macro = {k: float(np.mean([v[k] for v in grouped])) if all(v[k] is not None for v in grouped) else None for k in ("auc", "auprc", "brier")}
        rows.append({**base, "group": "MACRO", "site_count": len(grouped), **macro})
    return rows


def contrasts(table):
    """任务2 第6步：同一种子、规则和人群内配对，避免比较不一致的人群。"""
    keys = ["seed", "policy", "scope", "group"]
    measures = ["auc", "auprc", "brier"]
    def against(left, right, reference):
        joined = left.merge(right[keys+measures], on=keys, suffixes=("", "_reference"), validate="many_to_one")
        for key in measures:
            joined[f"delta_{key}"] = joined[key] - joined[f"{key}_reference"]
        joined["reference"] = reference
        return joined[keys+["arm", "reference"]+[f"delta_{k}" for k in measures]]
    full = table.loc[table.arm == "full"]
    effects = against(table.loc[table.arm != "full"], full, "full")
    candidate = table.loc[table.arm == "mask_marital_age"]
    random = pd.concat([against(candidate, table.loc[table.arm == name], name) for name in ("random_pair_1", "random_pair_2", "random_pair_3")], ignore_index=True)
    return effects, random


def draw(effects, output):
    """任务2 第6步：画每个种子的真实差值，保留负向结果，不按最好效果挑图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "figures"
    folder.mkdir()
    arms = effects.arm.drop_duplicates().tolist()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for ax, policy in zip(axes.flat, POLICIES):
        part = effects.loc[(effects.policy == policy) & (effects.scope == "east") & (effects.group == "POOL")]
        for offset, seed in zip((-.15, 0, .15), SEEDS):
            v = part.loc[part.seed == seed].set_index("arm").reindex(arms)
            ax.scatter(np.arange(len(arms))+offset, v.delta_auc, label=f"seed {seed}")
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_xticks(range(len(arms))); ax.set_xticklabels(arms, rotation=35, ha="right", fontsize=8)
        ax.set_title(policy); ax.set_ylabel("East pooled AUC change vs full"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(folder / "01_east_ablation_across_seeds.png", dpi=190); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, policy in zip(axes, ("matched_source_auc_epoch", "own_source_auc")):
        part = effects.loc[(effects.policy == policy) & (effects.scope == "east") & effects.group.isin(["110113", "110114", "371082", "371121"])]
        for arm in ("mask_marital", "mask_age", "mask_marital_age"):
            mean = part.loc[part.arm == arm].groupby("group").delta_auc.mean()
            ax.plot(mean.index, mean, marker="o", label=arm)
        ax.axhline(0, color="black", linewidth=.8); ax.set_title(policy)
        ax.set_ylabel("Site AUC change; mean of 3 seeds"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(folder / "02_county_effects.png", dpi=190); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, policy in zip(axes, ("matched_source_auc_epoch", "own_source_auc")):
        part = effects.loc[(effects.policy == policy) & (effects.group == "POOL")]
        for arm in ("mask_marital", "mask_age", "mask_marital_age"):
            g = part.loc[part.arm == arm].pivot(index="seed", columns="scope", values="delta_auc")
            ax.scatter(g.source_validation, g.east, label=arm)
            for seed, row in g.iterrows():
                ax.annotate(str(seed), (row.source_validation, row.east), fontsize=7)
        ax.axhline(0, color="black", linewidth=.8); ax.axvline(0, color="black", linewidth=.8)
        ax.set_xlabel("Source-validation AUC change"); ax.set_ylabel("East AUC change")
        ax.set_title(policy); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(folder / "03_source_vs_east_effect.png", dpi=190); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data/data_mission2.csv")
    parser.add_argument("--previous", type=Path, default=ROOT / "outputs/analysis_2611/task2_step4_east_checkpoint_diagnostics")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/analysis_2611/task2_step6_east_association_ablation")
    args = parser.parse_args()
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != EXPECTED_HASH:
        raise ValueError("研究CSV版本不一致")
    old = json.loads((args.previous / "experiment_metadata.json").read_text(encoding="utf-8"))
    versions = {"python": platform.python_version(), "torch": torch.__version__, "sklearn": sklearn.__version__}
    if not old["completed"] or any(old["versions"][k] != v for k, v in versions.items()):
        raise ValueError("请使用第4步的原运行环境和完整结果")
    reference = pd.read_csv(args.previous / "source_policy_comparison.csv")
    arms = make_arms()
    args.output.mkdir(parents=True, exist_ok=False)
    meta = {"completed": False, "input_sha256": EXPECTED_HASH, "versions": versions, "split_seed": 42,
            "training_seeds": SEEDS, "arms": arms, "target_used_for_training_or_selection": False,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "primary_contrast": "mask_marital_age vs full and random pairs, matched_source_auc_epoch; inspect all seeds and four counties",
            "limitation": "围绕已观察东部结果提出假设，属于探索性验证。屏蔽效应不是临床因果效应；微小不稳定变化不构成根因证据。"}
    dump(args.output / "experiment_metadata.json", meta)
    frame = attach_regions(read_raw(args.data))
    splits, _ = prepare_splits(held_out_split(frame, "east", seed=42, column=REGION), environment_column=REGION)
    rows, audits = [], []
    for seed in SEEDS:
        ref = reference.loc[reference.seed == seed].set_index("policy")
        fixed = {"matched_source_auc_epoch": int(ref.loc["source_pooled_auc", "epoch"]),
                 "matched_source_loss_epoch": int(ref.loc["source_validation_loss", "epoch"])}
        for arm, removed in arms.items():
            print(f"任务2 第6步：seed={seed} / {arm} / 屏蔽{removed}", flush=True)
            altered = [mask(s, removed) for s in splits]
            snapshots, lowest, loss_epoch, loss_state = {}, float("inf"), None, None

            def capture(model, record):
                nonlocal lowest, loss_epoch, loss_state
                epoch = record["epoch"]
                need_fixed = [p for p, e in fixed.items() if e == epoch]
                better_loss = record["val_loss"] < lowest
                if need_fixed or better_loss:
                    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    for p in need_fixed:
                        snapshots[p] = (state, epoch)
                    if better_loss:
                        lowest, loss_epoch, loss_state = record["val_loss"], epoch, state

            cfg = TrainConfig(model="baseline", environment_granularity="region", seed=seed, selection_start_epoch=1, device="cpu")
            model, training = fit(altered[0], altered[1], cfg, epoch_callback=capture)
            if any(p not in snapshots for p in fixed):
                raise RuntimeError("该处理早停早于预定固定轮次，无法公平对照；停止后请提交错误，不擅自延长训练")
            snapshots["own_source_auc"] = ({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, training["best_epoch"])
            snapshots["own_source_loss"] = (loss_state, loss_epoch)
            folder = args.output / str(seed) / arm
            folder.mkdir(parents=True)
            dump(folder / "training.json", training)
            dump(folder / "config.json", {"config": cfg.to_dict(), "removed_features": removed, "selected_epochs": {p: e for p, (_, e) in snapshots.items()}})
            for policy in POLICIES:
                state, epoch = snapshots[policy]
                model.load_state_dict(state)
                observed = evaluate(model, altered, seed, arm, policy, epoch)
                if arm == "full":
                    name = "source_pooled_auc" if policy in ("matched_source_auc_epoch", "own_source_auc") else "source_validation_loss"
                    actual = next(r for r in observed if r["scope"] == "east" and r["group"] == "POOL")
                    previous = ref.loc[name]
                    passed = epoch == int(previous.epoch) and all(np.isclose(actual[k], previous[f"east_{k}"], atol=2e-6, rtol=0) for k in ("auc", "auprc", "brier"))
                    audits.append({"seed": seed, "policy": policy, "reproduced": bool(passed)})
                    dump(args.output / "reproduction_audits.json", audits)
                    if not passed:
                        raise RuntimeError("完整模型未重现第4步，停止机制解释")
                rows.extend(observed)
            pd.DataFrame(rows).to_csv(args.output / "metrics.csv", index=False, encoding="utf-8-sig")
    table = pd.DataFrame(rows)
    effects, random = contrasts(table)
    effects.to_csv(args.output / "paired_ablation_effects.csv", index=False, encoding="utf-8-sig")
    random.to_csv(args.output / "candidate_vs_random_pairs.csv", index=False, encoding="utf-8-sig")
    draw(effects, args.output)
    meta.update(completed=True, reproduction_audits=audits)
    dump(args.output / "experiment_metadata.json", meta)
    print(f"任务2 第6步完成：{args.output}")


if __name__ == "__main__":
    main()

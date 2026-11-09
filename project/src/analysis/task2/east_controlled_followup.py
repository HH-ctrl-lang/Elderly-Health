"""任务2 第3步：东部输入偏移与特征依赖的配对重复实验。

固定split_seed=42；训练种子42/43/44。屏蔽标准化后的列而不改变网络参数量。
含目标无标签特征的加权分支属于目标域适配，其余分支属于来源域训练。
所有条件预先指定，不用东部标签选择条件或训练轮次。
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
import torch
import sklearn

PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT.parent
sys.path[:0] = [str(PROJECT / "src"), str(PROJECT / "src/analysis")]
from elderly_health.config import TrainConfig
from elderly_health.data import FEATURES, REGION, ENVIRONMENT, read_raw, attach_regions, held_out_split, prepare_splits
from elderly_health.training import fit, predict, get_device
from elderly_health.evaluation import metrics
from experiment1b_covariate_reweighting import estimate_density_ratio, effective_sample_size, evaluate_distribution_balance

EXPECTED_HASH = "969f0c1c82faafcfa402f37572e4a8ceb324492062cf1e590465f82dc6b5b6de"
SOCIO = ("residence", "education", "wealth")


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def masked(split, names):
    """任务2 第3步：标准化后置零，只改变可用信息，保持网络宽度一致。"""
    x = split.x.copy()
    x[:, [FEATURES.index(n) for n in names]] = 0
    return replace(split, x=x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data/data_mission2.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/analysis_2611/task2_step3_east_controlled_followup")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--regions", nargs="+", choices=["east", "central", "west"], default=["east"])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("训练种子不能重复")
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != EXPECTED_HASH:
        raise ValueError("输入CSV版本与固定研究基准不一致")
    args.output.mkdir(parents=True, exist_ok=False)
    frame = attach_regions(read_raw(args.data))
    # 任务2 第3步：随机删组在读取结局结果之前固定，三个组不含社会经济变量。
    rng = np.random.RandomState(20261109)
    pool = [f for f in FEATURES if f not in SOCIO]
    random_groups = {f"random_mask_{i+1}": tuple(rng.choice(pool, 3, replace=False)) for i in range(3)}
    masks = {"full": (), "socio_mask": SOCIO, **random_groups}
    write_json(args.output / "protocol.json", {"split_seed": 42, "training_seeds": args.seeds, "regions": args.regions, "mask_groups": masks, "input_sha256": EXPECTED_HASH, "completed": False})
    rows, deltas, diagnostics = [], [], {}
    for region in args.regions:
        frames = held_out_split(frame, region, seed=42, column=REGION)
        (train, val, test), _ = prepare_splits(frames, environment_column=REGION)
        # 任务2 第3步：复用原始交叉拟合密度比；本轮固定权重估计种子，仅重复模型训练。
        weights_table, _, _, diag = estimate_density_ratio(pd.concat(frames[:2]), frames[2], seed=42)
        w = weights_table.set_index("sample_id").loc[train.sample_ids, "weight_normalized"].to_numpy(dtype=np.float32)
        w /= w.mean()
        shuffled = np.random.RandomState(20261109).permutation(w)
        diagnostics[region] = {**diag, "train_weight_ess": effective_sample_size(w)}
        source = pd.concat(frames[:2])
        sw = weights_table.set_index("sample_id").loc[source.sample_id, "weight_normalized"].to_numpy()
        evaluate_distribution_balance(source, frames[2], sw).to_csv(args.output / f"{region}_distribution_balance.csv", index=False, encoding="utf-8-sig")
        arms = [(name, removed, None) for name, removed in masks.items()]
        arms += [("weighted_full", (), w), ("weighted_socio_mask", SOCIO, w), ("shuffled_weight_full", (), shuffled)]
        for seed in args.seeds:
            baseline = {}
            for arm, removed, weight in arms:
                print(f"任务2 第3步：{region} / seed={seed} / {arm}", flush=True)
                cfg = TrainConfig(model="baseline", environment_granularity="region", seed=seed, selection_start_epoch=1, device="cpu")
                a, b, c = [masked(s, removed) for s in (train, val, test)]
                model, record = fit(a, b, cfg, sample_weights=weight)
                p = predict(model, c.x, get_device("cpu"))
                folder = args.output / region / str(seed) / arm
                folder.mkdir(parents=True)
                write_json(folder / "training.json", record)
                # 任务2 第3步：仅保存聚合指标，不导出逐人标签或概率。
                scores = metrics(c.y, p)
                item = {"region": region, "seed": seed, "arm": arm, "scope": "region_pooled", **scores}
                rows.append(item)
                baseline[arm] = scores
                for site in sorted(set(c.site_env)):
                    ix = c.site_env == site
                    rows.append({"region": region, "seed": seed, "arm": arm, "scope": "site", "site_code": site, **metrics(c.y[ix], p[ix])})
                write_json(folder / "config.json", {"config": cfg.to_dict(), "removed_features": removed, "split_seed": 42})
            for left, right in [(a, "full") for a, _, _ in arms if a != "full"] + [("weighted_socio_mask", "weighted_full"), ("weighted_full", "shuffled_weight_full")] + [("socio_mask", name) for name in random_groups]:
                deltas.append({"region": region, "seed": seed, "arm": left, "reference": right,
                               **{f"delta_{k}": baseline[left][k]-baseline[right][k] for k in ("auc", "auprc", "brier")}})
            pd.DataFrame(rows).to_csv(args.output / "metrics.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(deltas).to_csv(args.output / "paired_seed_deltas.csv", index=False, encoding="utf-8-sig")
    table = pd.DataFrame(deltas)
    table.groupby(["region", "arm", "reference"])[["delta_auc", "delta_auprc", "delta_brier"]].agg(["mean", "std", "min", "max"]).to_csv(args.output / "repeat_summary.csv", encoding="utf-8-sig")
    # 任务2 第3步：展示每个训练种子的配对差异，避免只画均值掩盖反向变化。
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = args.output / "figures"
    figures.mkdir()
    for region in args.regions:
        part = table.loc[(table.region == region) & (table.reference == "full")]
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, (arm, g) in enumerate(part.groupby("arm", sort=False)):
            ax.scatter(np.full(len(g), i), g.delta_auc, label=arm)
        ax.set_xticks(range(part.arm.nunique()))
        ax.set_xticklabels(part.arm.drop_duplicates(), rotation=30, ha="right")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Paired AUC change from full model")
        fig.tight_layout()
        fig.savefig(figures / f"{region}_paired_seed_auc_changes.png", dpi=180)
        plt.close(fig)
    write_json(args.output / "experiment_metadata.json", {"input_sha256": EXPECTED_HASH, "split_seed": 42, "training_seeds": args.seeds, "regions": args.regions, "random_groups": random_groups, "weight_diagnostics": diagnostics, "python": platform.python_version(), "torch": torch.__version__, "sklearn": sklearn.__version__, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "interpretation": "探索性重复；3种子不是总体置信区间；固定权重只检验模型训练变化；不能自动宣布因果根因"})
    print(f"任务2 第3步完成：{args.output}")


if __name__ == "__main__":
    main()

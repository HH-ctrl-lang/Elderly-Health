"""任务2 第5步：县区特征关联迁移与MLP依赖对照。

来源训练集学习单特征类别风险，来源验证集决定展示顺序；东部仅事后诊断。
复现第4步两种来源选模模型，检查特征打乱后的站内AUC变化。
"""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
import sklearn

PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
from elderly_health.config import TrainConfig
from elderly_health.data import FEATURES, REGION, ENVIRONMENT, attach_regions, held_out_split, prepare_splits, read_raw
from elderly_health.evaluation import metrics
from elderly_health.training import fit, get_device, predict

EXPECTED_HASH = "969f0c1c82faafcfa402f37572e4a8ceb324492062cf1e590465f82dc6b5b6de"
SITES = ("110113", "110114", "371082", "371121")
POLICIES = ("source_pooled_auc", "source_validation_loss")
BLOCKS = {**{f: (f,) for f in FEATURES}, "phone_group": ("if_use_mobile_phone", "using_phone_year", "hours_phone_per_day"),
          "socioeconomic_group": ("residence", "education", "wealth")}


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def categories(values):
    """任务2 第5步：按编码类别建模，缺失单列，不假设数值间距或临床方向。"""
    return values.map(lambda v: "<MISSING>" if pd.isna(v) else format(float(v), ".12g"))


def rng_for(*tokens):
    code = hashlib.sha256("|".join(map(str, (26110905,) + tokens)).encode()).digest()
    return np.random.default_rng(int.from_bytes(code[:8], "little"))


def bootstrap_auc(y, scores, draws, token):
    """任务2 第5步：县区内分别重采样阳性与阴性；区间只反映固定评分下的抽样变化。"""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = rng_for(token)
    result = np.empty(draws)
    for i in range(draws):
        ix = np.r_[rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        result[i] = roc_auc_score(y[ix], scores[ix])
    return result


def association_diagnostics(frames, output, draws):
    """任务2 第5步：冻结来源类别风险，检查同一种评分是否能迁移到各县区。"""
    train, val, target = frames
    prior = float(train.label.mean())
    groups = {"SOURCE_VALIDATION": val,
              **{f"SOURCE_VAL_{s}": val.loc[val[ENVIRONMENT] == s] for s in sorted(val[ENVIRONMENT].unique())},
              **{s: target.loc[target[ENVIRONMENT] == s] for s in SITES}}
    rows, category_rows, contrasts = [], [], []
    for feature in FEATURES:
        keys = categories(train[feature])
        stats = train.assign(category=keys).groupby("category").label.agg(["sum", "count"])
        # 固定20个总体先验伪样本，减弱稀有类别的偶然极端阳性率；不看东部结果调强度。
        mapping = ((stats["sum"] + 20 * prior) / (stats["count"] + 20)).to_dict()
        distributions = {}
        for name, part in groups.items():
            cats = categories(part[feature])
            p = cats.map(mapping).fillna(prior).to_numpy(dtype=float)
            y = part.label.to_numpy(dtype=int)
            valid = len(np.unique(y)) == 2
            auc = float(roc_auc_score(y, p)) if valid else np.nan
            row = {"group": name, "feature": feature, "n": len(part), "positive_count": int(y.sum()),
                   "auc": auc, "unknown_category_rate": float((~cats.isin(mapping)).mean()),
                   "observed_category_count": int(cats.nunique()), "score_constant": bool(np.ptp(p) == 0),
                   "ci_lower": np.nan, "ci_upper": np.nan}
            if name in SITES and valid:
                d = bootstrap_auc(y, p, draws, (name, feature))
                distributions[name] = d
                row.update(ci_lower=float(np.quantile(d, .025)), ci_upper=float(np.quantile(d, .975)))
            rows.append(row)
            stats_part = part.assign(category=cats).groupby("category").label.agg(["sum", "count"])
            for cat, rec in stats_part.iterrows():
                category_rows.append({"group": name, "feature": feature, "category": cat, "n": int(rec["count"]),
                                      "positive_count": int(rec["sum"]), "positive_rate": float(rec["sum"]/rec["count"]),
                                      "source_learned_risk": float(mapping.get(cat, prior)), "unknown_to_source": cat not in mapping})
        delta = distributions["371082"] - distributions["110113"]
        point = {r["group"]: r["auc"] for r in rows if r["feature"] == feature}
        contrasts.append({"feature": feature, "auc_371082_minus_110113": point["371082"]-point["110113"],
                          "ci_lower": float(np.quantile(delta, .025)), "ci_upper": float(np.quantile(delta, .975))})
    table = pd.DataFrame(rows)
    table.to_csv(output / "feature_transfer_auc.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(category_rows).to_csv(output / "category_risk_profiles.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(contrasts).to_csv(output / "county_feature_auc_differences.csv", index=False, encoding="utf-8-sig")
    # 任务2 第5步：图中前八特征只根据来源验证AUC排序，不按东部差异挑图。
    order = table.loc[table.group == "SOURCE_VALIDATION"].sort_values("auc", ascending=False, kind="stable").feature.tolist()
    return table, order


def within_site_permutation(groups, rng):
    """任务2 第5步：来源验证集在各自站点内打乱，防止人为制造跨站点混合。"""
    index = np.arange(len(groups))
    for group in sorted(set(groups)):
        ids = np.flatnonzero(groups == group)
        index[ids] = rng.permutation(ids)
    return index


def dependence_diagnostics(splits, output, previous, repeats):
    """任务2 第5步：重现固定模型后置换输入；全程不重训或按东部筛选特征。"""
    train, val, test = splits
    reference = pd.read_csv(previous / "source_policy_comparison.csv")
    rows, baseline_rows, audits = [], [], []
    for seed in (42, 43, 44):
        cfg = TrainConfig(model="baseline", environment_granularity="region", seed=seed, selection_start_epoch=1, device="cpu")
        minimum_loss = float("inf")
        loss_state, loss_epoch = None, None

        def capture(model, record):
            nonlocal minimum_loss, loss_state, loss_epoch
            if record["val_loss"] < minimum_loss:
                minimum_loss = record["val_loss"]
                loss_epoch = record["epoch"]
                loss_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"任务2 第5步：复现seed={seed}，随后进行站内特征置换", flush=True)
        model, training = fit(train, val, cfg, epoch_callback=capture)
        states = {"source_pooled_auc": ({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, training["best_epoch"]),
                  "source_validation_loss": (loss_state, loss_epoch)}
        for policy, (state, epoch) in states.items():
            model.load_state_dict(state)
            actual = metrics(test.y, predict(model, test.x, get_device("cpu")))
            ref = reference.loc[(reference.seed == seed) & (reference.policy == policy)]
            if len(ref) != 1:
                raise ValueError("第4步选模结果记录缺失或重复")
            rec = ref.iloc[0]
            ok = epoch == int(rec.epoch) and all(np.isclose(actual[k], rec[f"east_{k}"], atol=2e-6, rtol=0) for k in ("auc", "auprc", "brier"))
            audits.append({"seed": seed, "policy": policy, "epoch": epoch, "reproduced": bool(ok)})
            dump(output / "reproduction_audits.json", audits)
            if not ok:
                raise RuntimeError("第4步模型未准确重现，停止特征依赖解释")
            groups = {"SOURCE_VALIDATION": val,
                      **{site: type(test)(test.x[test.site_env == site], test.y[test.site_env == site], test.env[test.site_env == site], test.sample_ids[test.site_env == site], test.site_env[test.site_env == site]) for site in SITES}}
            for name, split in groups.items():
                baseline = metrics(split.y, predict(model, split.x, get_device("cpu")))
                baseline_rows.append({"seed": seed, "policy": policy, "epoch": epoch, "group": name, **baseline})
                for block, features in BLOCKS.items():
                    changes = []
                    columns = [FEATURES.index(f) for f in features]
                    for repeat in range(repeats):
                        # 相同模型和站点的各特征共用同一批行置换；成组特征联合置换保留组内关系。
                        perm = within_site_permutation(split.site_env, rng_for("perm", seed, name, repeat))
                        x = split.x.copy()
                        x[:, columns] = split.x[perm][:, columns]
                        p = predict(model, x, get_device("cpu"))
                        changes.append(baseline["auc"]-roc_auc_score(split.y, p))
                    rows.append({"seed": seed, "policy": policy, "epoch": epoch, "group": name, "feature_block": block,
                                 "baseline_auc": baseline["auc"], "mean_auc_drop": float(np.mean(changes)),
                                 "shuffle_sd": float(np.std(changes, ddof=1)), "repeats": repeats})
            pd.DataFrame(rows).to_csv(output / "permutation_dependence.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(baseline_rows).to_csv(output / "model_group_metrics.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(rows)


def figures(association, dependence, order, output):
    """任务2 第5步：生成三张可直接引用的结果图，固定来源排序展示全部县区。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "figures"
    folder.mkdir()
    top = order[:8]
    groups = ["SOURCE_VALIDATION", *SITES]
    heat = association.pivot(index="feature", columns="group", values="auc").reindex(index=top, columns=groups)
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(heat, vmin=0.35, vmax=0.8, cmap="RdYlBu")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, rotation=25, ha="right")
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top)
    for i in range(len(top)):
        for j in range(len(groups)):
            ax.text(j, i, f"{heat.iloc[i,j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("Source-learned single-feature risk: transfer AUC")
    fig.colorbar(im, ax=ax, label="AUC; 0.5 means no ranking information")
    fig.tight_layout(); fig.savefig(folder / "01_feature_transfer_auc.png", dpi=200); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, policy in zip(axes, POLICIES):
        part = dependence.loc[dependence.policy == policy]
        for name, offset in [("SOURCE_VALIDATION", -.2), ("110113", 0), ("371082", .2)]:
            v = part.loc[(part.group == name) & part.feature_block.isin(top)].groupby("feature_block").mean_auc_drop.mean().reindex(top)
            ax.barh(np.arange(len(top))+offset, v, height=.2, label=name)
        ax.set_yticks(range(len(top))); ax.set_yticklabels(top)
        ax.axvline(0, color="black", linewidth=.8)
        ax.set_title(policy); ax.set_xlabel("AUC drop after shuffling; mean of 3 seeds"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(folder / "02_model_feature_dependence.png", dpi=200); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, policy in zip(axes, POLICIES):
        part = dependence.loc[(dependence.policy == policy) & dependence.group.isin(SITES) & dependence.feature_block.isin(["phone_group", "socioeconomic_group"])]
        part.groupby(["group", "feature_block"]).mean_auc_drop.mean().unstack().reindex(SITES).plot.bar(ax=ax, rot=0)
        ax.axhline(0, color="black", linewidth=.8); ax.set_title(policy)
        ax.set_ylabel("Group-shuffle AUC drop; mean of 3 seeds")
    fig.tight_layout(); fig.savefig(folder / "03_group_dependence_all_sites.png", dpi=200); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data/data_mission2.csv")
    parser.add_argument("--previous", type=Path, default=ROOT / "outputs/analysis_2611/task2_step4_east_checkpoint_diagnostics")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/analysis_2611/task2_step5_east_site_feature_transfer")
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--permutations", type=int, default=10)
    args = parser.parse_args()
    if args.bootstrap < 100 or args.permutations < 2:
        raise ValueError("bootstrap至少100，置换次数至少2")
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != EXPECTED_HASH:
        raise ValueError("固定CSV版本不一致")
    old = json.loads((args.previous / "experiment_metadata.json").read_text(encoding="utf-8"))
    versions = {"python": platform.python_version(), "torch": torch.__version__, "sklearn": sklearn.__version__}
    if not old["completed"] or any(old["versions"][k] != v for k, v in versions.items()):
        raise ValueError("请使用与已完成第4步相同的运行环境")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {"completed": False, "input_sha256": EXPECTED_HASH, "versions": versions, "split_seed": 42,
                "seeds": [42,43,44], "bootstrap": args.bootstrap, "permutations": args.permutations,
                "smoothing_prior_n": 20, "display_order": "source_validation_auc_only",
                "target_used_for_training_or_selection": False,
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dump(args.output / "experiment_metadata.json", metadata)
    frame = attach_regions(read_raw(args.data))
    frames = held_out_split(frame, "east", seed=42, column=REGION)
    splits, _ = prepare_splits(frames, environment_column=REGION)
    association, order = association_diagnostics(frames, args.output, args.bootstrap)
    dependence = dependence_diagnostics(splits, args.output, args.previous, args.permutations)
    figures(association, dependence, order, args.output)
    metadata.update(completed=True, source_order=order,
                    limitations="关联与置换不是因果效应；相关特征置换可产生不常见组合。Bootstrap区间固定来源评分，不含重拟合误差，未作多重比较校正；置换标准差不是置信区间。东部已被反复探索，本步不构成新外部验证。")
    dump(args.output / "experiment_metadata.json", metadata)
    print(f"任务2 第5步完成：{args.output}")


if __name__ == "__main__":
    main()

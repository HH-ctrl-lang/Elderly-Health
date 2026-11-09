"""任务2 第9步：东部持续排序失败人群、特征组合与来源阳性覆盖。

仅用来源训练/验证选模；东部标签仅用于训练完成后的描述性诊断。
逐人排序损失只保留内存，输出站点和预设特征组的聚合统计。
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
from sklearn.metrics import roc_auc_score

PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
from elderly_health.config import TrainConfig
from elderly_health.data import FEATURES, REGION, ENVIRONMENT, attach_regions, held_out_split, prepare_splits, read_raw
from elderly_health.evaluation import metrics
from elderly_health.training import fit, get_device, predict
from east_local_learnability import train_model, probability

EXPECTED_HASH = "969f0c1c82faafcfa402f37572e4a8ceb324492062cf1e590465f82dc6b5b6de"
SEEDS = (42, 43, 44)
CONDITIONS = ("mlp_source_auc", "mlp_source_loss", "shallow_tree")
PAIRS = (("age", "marital_status"), ("IADL", "hours_phone_per_day"), ("education", "if_use_mobile_phone"))
MIN_POSITIVE = 10


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def positive_rank_loss(y, p):
    """任务2 第9步：每名阳性被本站阴性超过的比例；同分计半次错误。"""
    y, p = np.asarray(y), np.asarray(p)
    negative = np.sort(p[y == 0])
    if not len(negative) or not (y == 1).any() or not np.isfinite(p).all():
        raise ValueError("排序诊断要求两类样本和有限评分")
    scores = p[y == 1]
    left = np.searchsorted(negative, scores, side="left")
    right = np.searchsorted(negative, scores, side="right")
    losses = (len(negative)-right + .5*(right-left))/len(negative)
    if not np.isclose(losses.mean(), 1-roc_auc_score(y, p), atol=1e-10):
        raise RuntimeError("阳性排序损失与1-AUC不一致")
    result = np.full(len(y), np.nan)
    result[y == 1] = losses
    return result


def make_rules(source):
    """任务2 第9步：少类别保留原编码，其余按来源三分位切分；不推测编码医学含义。"""
    rules = {}
    for feature in FEATURES:
        values = source[feature].dropna().to_numpy(dtype=float)
        unique = np.unique(values)
        if len(unique) <= 8:
            rules[feature] = {"kind": "category", "known": [format(x, ".12g") for x in unique]}
        else:
            cuts = np.unique(np.quantile(values, [1/3, 2/3])).tolist()
            rules[feature] = {"kind": "source_tertile", "cuts": cuts}
    return rules


def assign_groups(frame, rules):
    groups = pd.DataFrame(index=frame.index)
    for feature, rule in rules.items():
        values = frame[feature]
        if rule["kind"] == "category":
            groups[feature] = values.map(lambda x: "MISSING" if pd.isna(x) else format(float(x), ".12g"))
        else:
            groups[feature] = ["MISSING" if pd.isna(x) else f"bin{np.searchsorted(rule['cuts'], x, side='left')+1}" for x in values]
    for first, second in PAIRS:
        groups[first+"__"+second] = groups[first]+" | "+groups[second]
    return groups


def distance_spec(source):
    """任务2 第9步：来源决定混合特征距离，低基数按类别，其他按稳健尺度。"""
    spec = {}
    for f in FEATURES:
        values = source[f].dropna().to_numpy(dtype=float)
        categorical = len(np.unique(values)) <= 8
        scale = float(np.quantile(values, .95)-np.quantile(values, .05))
        spec[f] = {"categorical": categorical, "scale": scale if scale > 0 else 1.}
    return spec


def nearest_distance(query, reference, spec, k=5):
    """任务2 第9步：等权混合距离，取最近5人的均值；分块避免大矩阵占用。"""
    if len(reference) < k:
        raise ValueError("来源参照不足5人")
    output = []
    for start in range(0, len(query), 64):
        block = query.iloc[start:start+64]
        distance = np.zeros((len(block), len(reference)), dtype=float)
        for feature, rule in spec.items():
            a = block[feature].to_numpy(dtype=float)[:, None]
            b = reference[feature].to_numpy(dtype=float)[None, :]
            missing_a, missing_b = np.isnan(a), np.isnan(b)
            delta = (a != b).astype(float) if rule["categorical"] else np.minimum(np.abs(a-b)/rule["scale"], 1.)
            delta = np.where(missing_a | missing_b, 1., delta)
            delta = np.where(missing_a & missing_b, 0., delta)
            distance += delta
        distance /= len(spec)
        output.extend(np.partition(distance, k-1, axis=1)[:, :k].mean(axis=1))
    return np.asarray(output)


def coverage_diagnostic(source, validation, target):
    """任务2 第9步：用未参与拟合的来源验证阳性确定距离80%分位阈值。"""
    spec = distance_spec(source)
    val_positive = validation.loc[validation.label == 1]
    if len(val_positive) < 20:
        raise ValueError("来源验证阳性不足以构造参照")
    coverage = {}
    definitions = {"distance": spec, "neighbors": 5, "threshold_quantile": .8, "thresholds": {}}
    for name, reference in (("positive", source.loc[source.label == 1]), ("all", source)):
        calibration = nearest_distance(val_positive, reference, spec)
        threshold = float(np.quantile(calibration, .8))
        distance = nearest_distance(target, reference, spec)
        coverage[name+"_distance"] = distance
        coverage[name+"_far"] = distance > threshold
        definitions["thresholds"][name] = threshold
    return coverage, definitions


def profile_tables(source, source_groups, part, groups, losses, coverage, stable):
    """任务2 第9步：枚举预设分组，不导出个人记录；小阳性组仅保留计数。"""
    positive = part.label.to_numpy() == 1
    rows = []
    for feature in groups.columns:
        for level in sorted(groups[feature].unique()):
            source_mask = source_groups[feature].to_numpy() == level
            mask = groups[feature].to_numpy() == level
            pos = mask & positive
            n, npos = int(mask.sum()), int(pos.sum())
            eligible = npos >= MIN_POSITIVE
            base = {"feature_group": feature, "level": level, "n": n, "positive_count": npos,
                    "negative_count": n-npos, "eligible": eligible,
                    "source_n": int(source_mask.sum()), "source_positive": int(source.loc[source_mask, "label"].sum()),
                    "positive_share": npos/int(positive.sum()),
                    "positive_far_fraction": float(coverage["positive_far"][pos].mean()) if eligible else np.nan,
                    "all_far_fraction": float(coverage["all_far"][pos].mean()) if eligible else np.nan,
                    "stable_across_all_fraction": float(stable["all"][pos].mean()) if eligible else np.nan}
            for condition in CONDITIONS:
                per_seed = losses[condition][:, pos].mean(axis=1) if eligible else np.full(3, np.nan)
                rows.append({**base, "condition": condition, "mean_rank_loss": float(per_seed.mean()),
                             "min_seed_rank_loss": float(per_seed.min()), "max_seed_rank_loss": float(per_seed.max()),
                             "auc_error_contribution": float(per_seed.mean())*base["positive_share"],
                             "stable_poor_fraction": float(stable[condition][pos].mean()) if eligible else np.nan})
    return rows


def draw(site_table, stable_table, coverage_table, profiles, output):
    """任务2 第9步：各站点并列，预设画像不按东部错误大小筛选。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "figures"
    folder.mkdir()
    sites = sorted(site_table.site_code.unique())
    colors = ("tab:orange", "tab:blue", "tab:green")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for offset, condition, color in zip((-.2, 0, .2), CONDITIONS, colors):
        for i, site in enumerate(sites):
            values = site_table.loc[(site_table.site_code == site) & (site_table.condition == condition)]
            axes[0].scatter(np.full(len(values), i+offset), values.auc, color=color, label=condition if i == 0 else None)
        values = stable_table.loc[stable_table.condition == condition].set_index("site_code")
        axes[1].bar(np.arange(len(sites))+offset, values.loc[sites, "stable_poor_fraction"], width=.2, color=color, label=condition)
    axes[0].axhline(.5, color="gray", linestyle="--"); axes[0].set_ylabel("Within-site AUC; three seeds")
    axes[1].set_ylabel("Fraction of positives below negative median in all seeds")
    for ax in axes:
        ax.set_xticks(range(len(sites))); ax.set_xticklabels(sites); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(folder / "01_site_auc_and_persistent_failures.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    selected = coverage_table.loc[coverage_table.condition == "mlp_source_loss"]
    for offset, category, color in zip((-.15, .15), ("near_source_positive", "far_source_positive"), ("tab:blue", "tab:red")):
        values = selected.loc[selected.coverage_group == category].set_index("site_code")
        for i, site in enumerate(sites):
            row = values.loc[site]
            if row.eligible:
                axes[0].bar(i+offset, row.positive_fraction, width=.3, color=color, label=category if i == 0 else None)
                axes[1].scatter(i+offset, row.mean_rank_loss, color=color, label=category if i == 0 else None)
    for ax in axes:
        ax.set_xticks(range(len(sites))); ax.set_xticklabels(sites); ax.legend(fontsize=7)
    axes[0].set_ylabel("Share of site positives; groups with at least 10")
    axes[1].set_ylabel("MLP source-loss: mean positive rank loss")
    fig.tight_layout(); fig.savefig(folder / "02_source_positive_coverage_and_errors.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), squeeze=False)
    fixed_features = ("age", "marital_status", "if_use_mobile_phone", "IADL")
    for ax, site in zip(axes.ravel(), sites):
        values = profiles.loc[(profiles.site_code == site) & (profiles.condition == "mlp_source_loss") & profiles.feature_group.isin(fixed_features) & profiles.eligible].copy()
        values = values.sort_values(["feature_group", "level"], kind="stable")
        if values.empty:
            ax.text(.5, .5, "No eligible profiles", ha="center"); continue
        im = ax.imshow(values[["mean_rank_loss", "positive_far_fraction", "stable_across_all_fraction"]], vmin=0, vmax=1, aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels([f"{r.feature_group}={r.level} (pos {r.positive_count}; src pos {r.source_positive})" for r in values.itertuples()], fontsize=7)
        ax.set_xticks(range(3)); ax.set_xticklabels(["Rank loss", "Far source +", "Persistent all"], fontsize=8)
        ax.set_title(site); fig.colorbar(im, ax=ax, fraction=.03)
    fig.tight_layout(); fig.savefig(folder / "03_predefined_positive_profiles.png", dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data/data_mission2.csv")
    parser.add_argument("--previous", type=Path, default=ROOT / "outputs/analysis_2611/task2_step8_east_positive_sample_diagnostics")
    parser.add_argument("--reference", type=Path, default=ROOT / "outputs/analysis_2611/task2_step7_east_local_learnability")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/analysis_2611/task2_step9_east_persistent_ranking_failures")
    args = parser.parse_args()
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != EXPECTED_HASH:
        raise ValueError("固定CSV版本不一致")
    old = json.loads((args.previous / "experiment_metadata.json").read_text(encoding="utf-8"))
    versions = {"python": platform.python_version(), "torch": torch.__version__, "sklearn": sklearn.__version__}
    if not old["completed"] or versions != old["versions"]:
        raise ValueError("请使用第8步已完成实验的相同环境")
    frame = attach_regions(read_raw(args.data))
    source, validation, target = held_out_split(frame, "east", seed=42, column=REGION)
    if ((set(source.sample_id) & set(target.sample_id)) or
            (set(validation.sample_id) & set(target.sample_id)) or
            (set(source.sample_id) & set(validation.sample_id))):
        raise RuntimeError("来源与东部测试重叠")
    args.output.mkdir(parents=True, exist_ok=False)
    meta = {"completed": False, "input_sha256": EXPECTED_HASH, "versions": versions, "seeds": SEEDS,
            "source_split_seed": 42, "conditions": CONDITIONS, "predeclared_pairs": PAIRS,
            "minimum_positive_for_profile": MIN_POSITIVE, "poor_ranking_definition": "阳性本站排序损失严格大于0.5，三种子均满足；不是概率0.5阈值",
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "dependency_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
                                  (("training", PROJECT / "src/elderly_health/training.py"),
                                   ("step7", Path(__file__).with_name("east_local_learnability.py")))},
            "target_used_for_training_or_selection": False, "individual_records_exported": False,
            "limitations": "事后探索，存在重叠分组与历史反复检验，不能当因果证明或独立验证。来源低覆盖仅是距离启发式，不是缺样本的证明。来源全样本覆盖用于区分整体陌生和阳性陌生。树的三种子可能产生相同模型。不同选模条件复用同一MLP轨迹。排序损失使用同站阴性，站间差异包含阴性构成变化。"}
    dump(args.output / "experiment_metadata.json", meta)
    rules = make_rules(source)
    source_groups, target_groups = assign_groups(source, rules), assign_groups(target, rules)
    coverage, definition = coverage_diagnostic(source, validation, target)
    dump(args.output / "source_defined_groups_and_distance.json", {"groups": rules, "coverage": definition})
    (train_split, val_split, test_split), _ = prepare_splits((source, validation, target), environment_column=REGION)
    predictions = {key: [] for key in CONDITIONS}
    for seed in SEEDS:
        print(f"任务2 第9步：完整来源训练 seed={seed}", flush=True)
        lowest, loss_state, loss_epoch = float("inf"), None, None

        def capture(model, record):
            nonlocal lowest, loss_state, loss_epoch
            if record["val_loss"] < lowest:
                lowest, loss_epoch = float(record["val_loss"]), int(record["epoch"])
                loss_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        cfg = TrainConfig(model="baseline", environment_granularity="region", seed=seed, selection_start_epoch=1, device="cpu")
        model, history = fit(train_split, val_split, cfg, epoch_callback=capture)
        predictions["mlp_source_auc"].append(predict(model, test_split.x, get_device("cpu")))
        model.load_state_dict(loss_state)
        predictions["mlp_source_loss"].append(predict(model, test_split.x, get_device("cpu")))
        tree = train_model(source, validation, "shallow_tree", seed)
        predictions["shallow_tree"].append(probability(tree, "shallow_tree", target))
        folder = args.output / str(seed)
        folder.mkdir()
        dump(folder / "training.json", {"mlp": history, "mlp_config": cfg.to_dict(), "loss_epoch": loss_epoch, "tree": tree[2]})
    predictions = {key: np.stack(values) for key, values in predictions.items()}
    # 任务2 第9步：复核第7步完整来源模型，避免把训练实现变化解释成失败机制。
    previous = pd.read_csv(args.reference / "repeat_summary.csv", dtype={"site_code": str})
    reproduction = []
    for site in ("110113", "371082"):
        mask = target[ENVIRONMENT].to_numpy() == site
        for condition, family in (("mlp_source_loss", "mlp"), ("shallow_tree", "shallow_tree")):
            for j, seed in enumerate(SEEDS):
                match = previous.loc[(previous.site_code == site) & (previous.seed == seed) &
                                     (previous.model == family) & (previous["mode"] == "source_full")]
                if len(match) != 1:
                    raise ValueError("第7步完整来源参照缺失或重复")
                actual = metrics(target.loc[mask, "label"].to_numpy(dtype=int), predictions[condition][j, mask])
                differences = {key: float(actual[key]-match.iloc[0]["oof_"+key]) for key in ("auc", "auprc", "brier")}
                reproduction.append({"site_code": site, "condition": condition, "seed": seed,
                                     **{"difference_"+key: value for key, value in differences.items()},
                                     "passed": all(abs(value) <= 2e-6 for value in differences.values())})
    pd.DataFrame(reproduction).to_csv(args.output / "source_model_reproduction_audits.csv", index=False, encoding="utf-8-sig")
    if not all(row["passed"] for row in reproduction):
        raise RuntimeError("完整来源模型未复现，请先核对复现检查表，暂停机制解释")
    site_rows, stable_rows, coverage_rows, profiles, audits = [], [], [], [], []
    sites = sorted(target[ENVIRONMENT].unique())
    if len(sites) != 4:
        raise ValueError("预设东部四站点不一致")
    for site in sites:
        ix = np.flatnonzero(target[ENVIRONMENT].to_numpy() == site)
        part, groups = target.iloc[ix], target_groups.iloc[ix]
        positive = part.label.to_numpy() == 1
        local_coverage = {key: value[ix] for key, value in coverage.items()}
        losses = {condition: np.stack([positive_rank_loss(part.label, p[ix]) for p in predictions[condition]]) for condition in CONDITIONS}
        stable = {condition: (losses[condition] > .5).all(axis=0) for condition in CONDITIONS}
        stable["all"] = np.logical_and.reduce(list(stable.values()))
        for condition in CONDITIONS:
            for j, seed in enumerate(SEEDS):
                score = metrics(part.label.to_numpy(dtype=int), predictions[condition][j, ix])
                site_rows.append({"site_code": site, "condition": condition, "seed": seed, **score})
                difference = float(losses[condition][j, positive].mean()-(1-score["auc"]))
                audits.append({"site_code": site, "condition": condition, "seed": seed, "rank_loss_identity_error": difference})
            stable_rows.append({"site_code": site, "condition": condition, "positive_count": int(positive.sum()),
                                "stable_poor_count": int(stable[condition][positive].sum()), "stable_poor_fraction": float(stable[condition][positive].mean())})
            for far in (False, True):
                mask = positive & (local_coverage["positive_far"] == far)
                n = int(mask.sum()); eligible = n >= MIN_POSITIVE
                coverage_rows.append({"site_code": site, "condition": condition, "coverage_group": "far_source_positive" if far else "near_source_positive",
                                      "positive_count": n, "positive_fraction": n/int(positive.sum()), "eligible": eligible,
                                      "mean_rank_loss": float(losses[condition][:, mask].mean()) if eligible else np.nan,
                                      "stable_poor_fraction": float(stable[condition][mask].mean()) if eligible else np.nan,
                                      "also_far_from_all_source_fraction": float(local_coverage["all_far"][mask].mean()) if eligible else np.nan})
        stable_rows.append({"site_code": site, "condition": "all_conditions", "positive_count": int(positive.sum()),
                            "stable_poor_count": int(stable["all"][positive].sum()), "stable_poor_fraction": float(stable["all"][positive].mean())})
        profiles.extend({"site_code": site, **row} for row in profile_tables(source, source_groups, part, groups, losses, local_coverage, stable))
    tables = {"site_metrics": site_rows, "persistent_positive_failures": stable_rows, "source_coverage_rank_loss": coverage_rows,
              "predefined_feature_profiles": profiles, "auc_identity_audits": audits}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(args.output / (name+".csv"), index=False, encoding="utf-8-sig")
    draw(pd.DataFrame(site_rows), pd.DataFrame(stable_rows), pd.DataFrame(coverage_rows), pd.DataFrame(profiles), args.output)
    meta.update(completed=True, all_auc_identities_passed=True, source_models_reproduced=True,
                source_train_n=len(source), source_positive=int(source.label.sum()), sites=sites)
    dump(args.output / "experiment_metadata.json", meta)
    print(f"任务2 第9步完成：{args.output}")


if __name__ == "__main__":
    main()

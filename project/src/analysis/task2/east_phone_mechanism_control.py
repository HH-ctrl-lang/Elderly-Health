"""任务2 第10步：手机特征关联迁移的分层比较与信息屏蔽对照。

固定年龄×教育×IADL分层、来源训练划分、三种子和屏蔽组。
来源决定选模；东部标签仅用于事后诊断，不用于训练或调参。
本步屏蔽模型可见信息，不是手机使用的临床因果干预。
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
from elderly_health.data import FEATURES, REGION, ENVIRONMENT, attach_regions, held_out_split, prepare_splits, read_raw
from elderly_health.evaluation import metrics
from elderly_health.training import fit, predict, get_device
from east_local_learnability import train_model, probability
from east_persistent_ranking_failures import positive_rank_loss

EXPECTED_HASH = "969f0c1c82faafcfa402f37572e4a8ceb324492062cf1e590465f82dc6b5b6de"
SEEDS = (42, 43, 44)
PHONE = ("if_use_mobile_phone", "using_phone_year", "hours_phone_per_day")
STRATA = ("age", "education", "IADL")
POLICIES = ("matched_source_auc_epoch", "matched_source_loss_epoch", "own_source_auc", "own_source_loss")
MIN_CELL = 5
MIN_POSITIVE = 10


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def make_arms():
    """任务2 第10步：两个固定随机三列对照，排除只是丢信息导致的变化。"""
    pool = [f for f in FEATURES if f not in PHONE]
    chosen = np.random.RandomState(26110910).choice(pool, size=6, replace=False).tolist()
    return {"full": (), "mask_phone_group": PHONE,
            "random_triple_1": tuple(chosen[:3]), "random_triple_2": tuple(chosen[3:])}


def masked(split, removed):
    """任务2 第10步：训练、验证、测试标准化后的相同列置零，结构与初始化不变。"""
    x = split.x.copy()
    x[:, [FEATURES.index(f) for f in removed]] = 0
    return replace(split, x=x)


def cell_keys(frame):
    return frame[list(STRATA)].apply(lambda col: col.map(lambda x: "MISSING" if pd.isna(x) else format(float(x), ".12g"))).agg(" | ".join, axis=1)


def stratified_association(source, target):
    """任务2 第10步：共同分层内四组人数均≥5，再用相同来源权重标准化风险。"""
    rows, cells = [], []
    source_keys = cell_keys(source)
    for site in sorted(target[ENVIRONMENT].unique()):
        part = target.loc[target[ENVIRONMENT] == site]
        target_keys = cell_keys(part)
        local = []
        for key in sorted(set(source_keys) | set(target_keys)):
            record = {"site_code": str(site), "stratum": key}
            for name, frame, keys in (("source", source, source_keys), ("target", part, target_keys)):
                for phone in (1, 2):
                    group = frame.loc[(keys == key) & (frame[PHONE[0]] == phone)]
                    record[f"{name}_phone{phone}_n"] = len(group)
                    record[f"{name}_phone{phone}_positive"] = int(group.label.sum())
            record["eligible"] = all(record[f"{scope}_phone{phone}_n"] >= MIN_CELL for scope in ("source", "target") for phone in (1, 2))
            local.append(record)
        valid = [r for r in local if r["eligible"]]
        source_kept = sum(r["source_phone1_n"]+r["source_phone2_n"] for r in valid)
        summary = {"site_code": str(site), "common_strata": len(valid), "source_retained_n": source_kept,
                   "source_retained_fraction": source_kept/len(source), "target_n": len(part)}
        target_kept = sum(r["target_phone1_n"]+r["target_phone2_n"] for r in valid)
        summary.update(target_retained_n=target_kept, target_retained_fraction=target_kept/len(part))
        for scope, frame in (("source", source), ("target", part)):
            for phone in (1, 2):
                group = frame.loc[frame[PHONE[0]] == phone]
                kept = sum(r[f"{scope}_phone{phone}_n"] for r in valid)
                summary[f"{scope}_phone{phone}_retained_n"] = kept
                summary[f"{scope}_phone{phone}_retained_fraction"] = kept/len(group) if len(group) else np.nan
                summary[f"{scope}_phone{phone}_crude_risk"] = float(group.label.mean()) if len(group) else np.nan
                summary[f"{scope}_phone{phone}_standardized_risk"] = sum(
                    (r["source_phone1_n"]+r["source_phone2_n"])/source_kept * r[f"{scope}_phone{phone}_positive"]/r[f"{scope}_phone{phone}_n"] for r in valid) if valid else np.nan
            summary[scope+"_risk_difference"] = summary[f"{scope}_phone1_standardized_risk"]-summary[f"{scope}_phone2_standardized_risk"]
            # 同层2×2表合并优势比，补充风险差；避免只凭患病率变化解释关联迁移。
            numerator, denominator = 0., 0.
            for record in valid:
                a = record[f"{scope}_phone1_positive"]
                b = record[f"{scope}_phone1_n"]-a
                c = record[f"{scope}_phone2_positive"]
                d = record[f"{scope}_phone2_n"]-c
                total = a+b+c+d
                numerator += a*d/total
                denominator += b*c/total
            summary[scope+"_mh_odds_ratio"] = numerator/denominator if denominator > 0 else np.nan
        summary["target_minus_source_risk_difference"] = summary["target_risk_difference"]-summary["source_risk_difference"]
        source_or, target_or = summary["source_mh_odds_ratio"], summary["target_mh_odds_ratio"]
        summary["target_minus_source_log_mh_odds_ratio"] = float(np.log(target_or)-np.log(source_or)) if source_or > 0 and target_or > 0 else np.nan
        for record in local:
            record["source_standardization_weight"] = (record["source_phone1_n"]+record["source_phone2_n"])/source_kept if record["eligible"] else 0.
        cells.extend(local); rows.append(summary)
    return pd.DataFrame(rows), pd.DataFrame(cells)


def take_predictions(model, splits):
    return tuple(predict(model, split.x, get_device("cpu")) for split in splits[1:])


def fit_mlp(splits, seed, fixed):
    """任务2 第10步：同轨迹保留来源最优点及原模型固定轮次，目标不参与选模。"""
    snapshots, lowest, loss_state, loss_epoch = {}, float("inf"), None, None

    def capture(model, record):
        nonlocal lowest, loss_state, loss_epoch
        epoch = int(record["epoch"])
        needed = [name for name, number in fixed.items() if number == epoch]
        better = record["val_loss"] < lowest
        if needed or better:
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            for name in needed:
                snapshots[name] = (state, epoch)
            if better:
                lowest, loss_epoch, loss_state = float(record["val_loss"]), epoch, state

    cfg = TrainConfig(model="baseline", environment_granularity="region", seed=seed, selection_start_epoch=1, device="cpu")
    model, history = fit(splits[0], splits[1], cfg, epoch_callback=capture)
    if any(name not in snapshots for name in fixed):
        raise RuntimeError("处理在固定轮次前早停，不能作同轮次对照；请上传当前输出核对")
    snapshots["own_source_auc"] = ({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, int(history["best_epoch"]))
    snapshots["own_source_loss"] = (loss_state, loss_epoch)
    results = {}
    for policy in POLICIES:
        state, epoch = snapshots[policy]
        model.load_state_dict(state)
        pv, pt = take_predictions(model, splits)
        results[policy] = {"epoch": epoch, "validation": pv, "target": pt}
    return results, {"config": cfg.to_dict(), "training": history, "epochs": {k: v["epoch"] for k,v in results.items()}}


def check_reproduction(results, previous, target, seed, family):
    """任务2 第10步：只接受第9步完整模型的原县区成绩复现。"""
    rows = []
    for policy, prediction in results.items():
        condition = "shallow_tree" if family == "shallow_tree" else ("mlp_source_auc" if "auc" in policy else "mlp_source_loss")
        for site in sorted(target[ENVIRONMENT].unique()):
            ix = target[ENVIRONMENT].to_numpy() == site
            old = previous.loc[(previous.site_code == site) & (previous.seed == seed) & (previous.condition == condition)]
            if len(old) != 1:
                raise ValueError("第9步模型成绩缺失或重复")
            actual = metrics(target.loc[ix, "label"].to_numpy(dtype=int), prediction["target"][ix])
            differences = {key: float(actual[key]-old.iloc[0][key]) for key in ("auc", "auprc", "brier")}
            rows.append({"seed": seed, "model": family, "policy": policy, "site_code": str(site),
                         **{"difference_"+k: v for k,v in differences.items()}, "passed": all(abs(v) <= 2e-6 for v in differences.values())})
    return rows


def freeze_failures(predictions, target):
    """任务2 第10步：由完整模型冻结困难阳性名单；屏蔽组不能重新定义困难人群。"""
    result = np.zeros(len(target), dtype=bool)
    for site in sorted(target[ENVIRONMENT].unique()):
        ix = np.flatnonzero(target[ENVIRONMENT].to_numpy() == site)
        failures = []
        for family, policy in (("mlp", "own_source_auc"), ("mlp", "own_source_loss"), ("shallow_tree", "fixed_tree")):
            for seed in SEEDS:
                p = predictions[("full", family, seed)][policy]["target"][ix]
                failures.append(positive_rank_loss(target.iloc[ix].label, p) > .5)
        result[ix] = np.logical_and.reduce(failures)
    return result


def evaluate_all(predictions, validation, target, frozen):
    """任务2 第10步：同时看困难组、手机两组和全站，避免局部改善掩盖其他人受损。"""
    metric_rows, rank_rows = [], []
    for (arm, family, seed), policies in predictions.items():
        for policy, pred in policies.items():
            base = {"arm": arm, "model": family, "seed": seed, "policy": policy, "epoch": pred["epoch"]}
            for scope, frame, p in (("source_validation", validation, pred["validation"]), ("east", target, pred["target"])):
                cohorts = {"POOL": np.ones(len(frame), dtype=bool)}
                if scope == "east":
                    cohorts.update({str(site): frame[ENVIRONMENT].to_numpy() == site for site in sorted(frame[ENVIRONMENT].unique())})
                for group, ix in cohorts.items():
                    metric_rows.append({**base, "scope": scope, "group": group, **metrics(frame.loc[ix, "label"].to_numpy(dtype=int), p[ix])})
            for site in sorted(target[ENVIRONMENT].unique()):
                ix = np.flatnonzero(target[ENVIRONMENT].to_numpy() == site)
                part = target.iloc[ix]
                loss = positive_rank_loss(part.label, pred["target"][ix])
                positive = part.label.to_numpy() == 1
                groups = {"all_positive": positive, "phone1_positive": positive & (part[PHONE[0]].to_numpy() == 1),
                          "phone2_positive": positive & (part[PHONE[0]].to_numpy() == 2),
                          "frozen_persistent_positive": frozen[ix], "other_positive": positive & ~frozen[ix]}
                for group, keep in groups.items():
                    n = int(keep.sum()); eligible = n >= MIN_POSITIVE
                    rank_rows.append({**base, "site_code": str(site), "subgroup": group, "positive_count": n, "eligible": eligible,
                                      "mean_rank_loss": float(loss[keep].mean()) if eligible else np.nan,
                                      "poor_rank_fraction": float((loss[keep] > .5).mean()) if eligible else np.nan})
    return pd.DataFrame(metric_rows), pd.DataFrame(rank_rows)


def paired_effects(table, keys, measures):
    """任务2 第10步：同种子、同规则、同人群内比较；方向不一致的结果完整保留。"""
    baseline = table.loc[table.arm == "full", keys+list(measures)]
    result = table.loc[table.arm != "full"].merge(baseline, on=keys, suffixes=("", "_full"), validate="many_to_one")
    for key in measures:
        result["delta_"+key] = result[key]-result[key+"_full"]
    return result


def draw(association, effects, ranks, output):
    """任务2 第10步：图号固定，后续聊天用图名、表格和GitHub页面解释。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "figures"; folder.mkdir()
    sites = sorted(association.site_code.unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, measure in zip(axes, ("risk_difference", "mh_odds_ratio")):
        for scope, offset in (("source", -.1), ("target", .1)):
            values = association.set_index("site_code").loc[sites, scope+"_"+measure]
            if measure == "mh_odds_ratio":
                values = np.log(values.where(values > 0))
            ax.scatter(np.arange(len(sites))+offset, values, label=scope)
        ax.axhline(0, color="gray", linestyle="--"); ax.set_xticks(range(len(sites))); ax.set_xticklabels(sites)
        ax.set_ylabel("Standardized risk difference" if measure == "risk_difference" else "Log stratified odds ratio")
        ax.set_title("Phone code 1 vs 2; same age / education / IADL strata"); ax.legend()
    fig.tight_layout(); fig.savefig(folder / "01_stratified_phone_association.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, policy in zip(axes, ("matched_source_loss_epoch", "own_source_loss")):
        values = effects.loc[(effects.model == "mlp") & (effects.policy == policy) & (effects.scope == "east") & effects.group.isin(sites)]
        for offset, arm, color in zip((-.2, 0, .2), ("mask_phone_group", "random_triple_1", "random_triple_2"), ("tab:blue", "tab:orange", "tab:green")):
            for i, site in enumerate(sites):
                v = values.loc[(values.arm == arm) & (values.group == site)]
                ax.scatter(np.full(len(v), i+offset), v.delta_auc, color=color, label=arm if i == 0 else None)
        ax.axhline(0, color="black", linewidth=.8); ax.set_xticks(range(len(sites))); ax.set_xticklabels(sites)
        ax.set_title(policy); ax.set_ylabel("Site AUC change vs full"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(folder / "02_phone_ablation_vs_random_controls.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    subgroups = ("phone1_positive", "phone2_positive", "frozen_persistent_positive", "other_positive")
    for ax, policy in zip(axes, ("matched_source_loss_epoch", "own_source_loss")):
        values = ranks.loc[(ranks.model == "mlp") & (ranks.policy == policy) & (ranks.arm == "mask_phone_group") & (ranks.site_code == "371082")]
        for offset, seed in zip((-.12, 0, .12), SEEDS):
            v = values.loc[values.seed == seed].set_index("subgroup").reindex(subgroups)
            ax.scatter(np.arange(len(subgroups))+offset, v.delta_mean_rank_loss, label=f"seed {seed}")
        ax.axhline(0, color="black", linewidth=.8); ax.set_xticks(range(len(subgroups)))
        ax.set_xticklabels(subgroups, rotation=25, ha="right", fontsize=8); ax.set_title(policy)
        ax.set_ylabel("371082 positive rank-loss change; lower is better"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(folder / "03_frozen_failure_group_and_tradeoffs.png", dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data/data_mission2.csv")
    parser.add_argument("--previous", type=Path, default=ROOT / "outputs/analysis_2611/task2_step9_east_persistent_ranking_failures")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/analysis_2611/task2_step10_east_phone_mechanism_control")
    args = parser.parse_args()
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != EXPECTED_HASH:
        raise ValueError("固定CSV版本不一致")
    old = json.loads((args.previous / "experiment_metadata.json").read_text(encoding="utf-8"))
    versions = {"python": platform.python_version(), "torch": torch.__version__, "sklearn": sklearn.__version__}
    if not old["completed"] or versions != old["versions"] or old["input_sha256"] != EXPECTED_HASH:
        raise ValueError("请使用第9步相同运行环境及完整结果")
    frame = attach_regions(read_raw(args.data))
    if not frame[PHONE[0]].isin([1, 2]).all():
        raise ValueError("手机编码不符合预设1/2，不能静默排除未知编码")
    source, validation, target = held_out_split(frame, "east", seed=42, column=REGION)
    if set(source.sample_id) & set(validation.sample_id) or (set(source.sample_id) | set(validation.sample_id)) & set(target.sample_id):
        raise RuntimeError("来源训练、验证、目标样本重叠")
    previous = pd.read_csv(args.previous / "site_metrics.csv", dtype={"site_code": str})
    arms = make_arms()
    args.output.mkdir(parents=True, exist_ok=False)
    meta = {"completed": False, "input_sha256": EXPECTED_HASH, "versions": versions, "seeds": SEEDS,
            "arms": arms, "policies": POLICIES, "strata": STRATA, "minimum_each_population_phone_cell": MIN_CELL,
            "minimum_positive_for_rank_summary": MIN_POSITIVE, "source_split_seed": 42,
            "target_used_for_training_or_selection": False, "individual_records_exported": False,
            "primary_comparison": "371082 手机三列屏蔽 vs 完整，匹配来源损失轮次；看冻结困难组与其他阳性的代价",
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "association_outputs": "来源共同人群权重标准化风险差，以及共同分层Mantel-Haenszel优势比；零分母不估计，不自行加伪计数",
            "limitations": "既有东部结果提出的事后探索，不是独立确认。分层只控制年龄、教育、IADL，保留人群有限时不能外推到全站。风险差受患病率影响，不能单独证明机制或因果效应。屏蔽改变模型信息，不是临床因果干预。冻结困难组按原模型差结果挑选，改善含回归均值可能，必须对照随机删组和其他人群。树三种子可能相同。固定轮次控制选模变化，但不保证优化过程相同。"}
    dump(args.output / "experiment_metadata.json", meta)
    association, cells = stratified_association(source, target)
    association.to_csv(args.output / "stratified_phone_association.csv", index=False, encoding="utf-8-sig")
    cells.to_csv(args.output / "common_stratum_counts.csv", index=False, encoding="utf-8-sig")
    for row in association.itertuples():
        print(f"任务2 第10步：站点{row.site_code}共同分层{row.common_strata}个，保留本站{row.target_retained_fraction:.1%}样本", flush=True)
    splits, _ = prepare_splits((source, validation, target), environment_column=REGION)
    predictions, audits = {}, []
    for arm, removed in arms.items():
        for seed in SEEDS:
            original = json.loads((args.previous / str(seed) / "training.json").read_text(encoding="utf-8"))
            fixed = {"matched_source_auc_epoch": int(original["mlp"]["best_epoch"]), "matched_source_loss_epoch": int(original["loss_epoch"])}
            print(f"任务2 第10步：{arm} seed={seed} 屏蔽{removed}", flush=True)
            altered = [masked(split, removed) for split in splits]
            result, record = fit_mlp(altered, seed, fixed)
            predictions[(arm, "mlp", seed)] = result
            folder = args.output / arm / str(seed); folder.mkdir(parents=True)
            dump(folder / "training.json", {"removed": removed, **record})
            if arm == "full":
                audits.extend(check_reproduction(result, previous, target, seed, "mlp"))
            if arm in ("full", "mask_phone_group"):
                # 树同步屏蔽三列；原始常量经各自来源预处理后同样为零。
                raw = [f.copy() for f in (source, validation, target)]
                for item in raw:
                    for feature in removed:
                        item[feature] = 0.
                fitted = train_model(raw[0], raw[1], "shallow_tree", seed)
                result = {"fixed_tree": {"epoch": 100, "validation": probability(fitted, "shallow_tree", raw[1]),
                                         "target": probability(fitted, "shallow_tree", raw[2])}}
                predictions[(arm, "shallow_tree", seed)] = result
                dump(folder / "tree_training.json", fitted[2])
                if arm == "full":
                    audits.extend(check_reproduction(result, previous, target, seed, "shallow_tree"))
            pd.DataFrame(audits).to_csv(args.output / "reproduction_audits.csv", index=False, encoding="utf-8-sig")
            if not all(row["passed"] for row in audits):
                raise RuntimeError("完整模型未复现第9步，请先核对复现检查表")
    frozen = freeze_failures(predictions, target)
    expected = pd.read_csv(args.previous / "persistent_positive_failures.csv", dtype={"site_code": str})
    frozen_rows = []
    for site in sorted(target[ENVIRONMENT].unique()):
        count = int(frozen[target[ENVIRONMENT].to_numpy() == site].sum())
        oldrow = expected.loc[(expected.site_code == site) & (expected.condition == "all_conditions")]
        if len(oldrow) != 1 or count != int(oldrow.iloc[0].stable_poor_count):
            raise RuntimeError("冻结困难组未重现第9步")
        frozen_rows.append({"site_code": str(site), "frozen_positive_count": count, "reproduced": True})
    pd.DataFrame(frozen_rows).to_csv(args.output / "frozen_group_audits.csv", index=False, encoding="utf-8-sig")
    scores, ranks = evaluate_all(predictions, validation, target, frozen)
    effects = paired_effects(scores, ["model", "seed", "policy", "scope", "group"], ("auc", "auprc", "brier"))
    rank_effects = paired_effects(ranks, ["model", "seed", "policy", "site_code", "subgroup"], ("mean_rank_loss", "poor_rank_fraction"))
    for name, table in (("metrics", scores), ("positive_rank_profiles", ranks), ("paired_model_effects", effects), ("paired_rank_effects", rank_effects)):
        table.to_csv(args.output / (name+".csv"), index=False, encoding="utf-8-sig")
    # 分别保存随机对照，不能只从三个屏蔽方案中挑最好者。
    candidate = rank_effects.loc[(rank_effects.arm == "mask_phone_group") & (rank_effects.model == "mlp")]
    comparisons = []
    keys = ["model", "seed", "policy", "site_code", "subgroup"]
    for random_arm in ("random_triple_1", "random_triple_2"):
        random = rank_effects.loc[rank_effects.arm == random_arm]
        joined = candidate.merge(random[keys+["delta_mean_rank_loss"]], on=keys, suffixes=("", "_random"), validate="one_to_one")
        joined["random_reference"] = random_arm
        joined["phone_minus_random_rank_loss_change"] = joined.delta_mean_rank_loss-joined.delta_mean_rank_loss_random
        comparisons.append(joined)
    pd.concat(comparisons, ignore_index=True).to_csv(args.output / "phone_vs_random_rank_effects.csv", index=False, encoding="utf-8-sig")
    draw(association, effects, rank_effects, args.output)
    meta.update(completed=True, source_models_reproduced=True, frozen_groups_reproduced=True)
    dump(args.output / "experiment_metadata.json", meta)
    print(f"任务2 第10步完成：{args.output}")


if __name__ == "__main__":
    main()

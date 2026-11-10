"""任务2 第11步：手机组内/组间AUC分解与固定特征组合诊断。

复现第10步完整模型与手机三列屏蔽模型，不增加新算法或目标调参。
AUC的代数分解只定位错误构成，不提供临床因果归因。
逐人评分和标签只留内存，输出聚合表与英文图表。
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
from elderly_health.data import REGION, ENVIRONMENT, attach_regions, held_out_split, prepare_splits, read_raw
from elderly_health.evaluation import metrics
from east_local_learnability import train_model, probability
import east_phone_mechanism_control as prior_code
from east_phone_mechanism_control import fit_mlp, masked, cell_keys, PHONE, POLICIES, SEEDS

EXPECTED_HASH = "969f0c1c82faafcfa402f37572e4a8ceb324492062cf1e590465f82dc6b5b6de"
PRIMARY = "matched_source_loss_epoch"
BLOCKS = ((1, 1), (1, 2), (2, 1), (2, 2))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def concordance(positive, negative):
    """任务2 第11步：阳性评分更高记1，同分记0.5；不使用概率分类阈值。"""
    return (positive[:, None] > negative[None, :]).astype(float) + .5*(positive[:, None] == negative[None, :])


def block_metrics(frame, scores):
    """任务2 第11步：四块AUC×类别内组比例，严格还原总体AUC。"""
    y = frame.label.to_numpy(dtype=int)
    phone = frame[PHONE[0]].to_numpy()
    scores = np.asarray(scores)
    if not np.isin(phone, [1, 2]).all() or not np.isfinite(scores).all() or len(np.unique(y)) != 2:
        raise ValueError("分解要求两类结局、手机编码1/2和有限评分")
    npos, nneg = int(y.sum()), int((y == 0).sum())
    rows = []
    for a, b in BLOCKS:
        pos = (y == 1) & (phone == a)
        neg = (y == 0) & (phone == b)
        np_, nn = int(pos.sum()), int(neg.sum())
        weight = np_/npos * nn/nneg
        auc = float(concordance(scores[pos], scores[neg]).mean()) if np_ and nn else np.nan
        rows.append({"positive_phone": a, "negative_phone": b, "block": f"P{a}_N{b}",
                     "comparison": "WITHIN" if a == b else "BETWEEN", "positive_n": np_, "negative_n": nn,
                     "pair_count": np_*nn, "positive_share": np_/npos, "negative_share": nn/nneg,
                     "weight": weight, "block_auc": auc, "small_cell": np_ < 10 or nn < 20,
                     "auc_contribution": weight*auc if weight else 0.,
                     "error_contribution": weight*(1-auc) if weight else 0.})
    actual = float(roc_auc_score(y, scores))
    error = sum(r["auc_contribution"] for r in rows)-actual
    if not np.isclose(error, 0, atol=1e-10) or not np.isclose(sum(r["weight"] for r in rows), 1):
        raise RuntimeError("四块未还原总AUC")
    return rows, {"auc": actual, "identity_error": float(error)}


def gap_decomposition(source_rows, target_rows):
    """任务2 第11步：对称分解构成权重变化与块内AUC变化；两项严格相加为总差。"""
    complete = all(np.isfinite(r["block_auc"]) for r in source_rows+target_rows)
    rows = []
    for s, t in zip(source_rows, target_rows):
        if s["block"] != t["block"]:
            raise ValueError("来源与目标块顺序不一致")
        composition = .5*(t["weight"]-s["weight"])*(t["block_auc"]+s["block_auc"]) if complete else np.nan
        separation = .5*(t["block_auc"]-s["block_auc"])*(t["weight"]+s["weight"]) if complete else np.nan
        rows.append({"block": t["block"], "source_weight": s["weight"], "target_weight": t["weight"],
                     "source_block_auc": s["block_auc"], "target_block_auc": t["block_auc"],
                     "composition_component": composition, "separation_component": separation,
                     "complete_support": complete})
    source_auc = sum(s["auc_contribution"] for s in source_rows)
    target_auc = sum(t["auc_contribution"] for t in target_rows)
    composition = sum(r["composition_component"] for r in rows) if complete else np.nan
    separation = sum(r["separation_component"] for r in rows) if complete else np.nan
    error = composition+separation-(target_auc-source_auc) if complete else np.nan
    if complete and not np.isclose(error, 0, atol=1e-10):
        raise RuntimeError("来源目标差分解未还原")
    summary = {"source_auc": source_auc, "target_auc": target_auc, "target_minus_source_auc": target_auc-source_auc,
               "composition_component": composition, "separation_component": separation,
               "target_auc_at_source_weights": sum(s["weight"]*t["block_auc"] for s,t in zip(source_rows,target_rows)) if complete else np.nan,
               "complete_support": complete, "identity_error": error}
    return rows, summary


def bootstrap_change(frame, full, altered, draws=500, seed=26111011):
    """任务2 第11步：固定评分，类别×手机组内配对重采样；保留四块共享患者的相关性。"""
    y = frame.label.to_numpy(dtype=int); phone = frame[PHONE[0]].to_numpy()
    rng = np.random.default_rng(seed)
    positions, counts = {}, {}
    for label in (0, 1):
        for group in (1, 2):
            ids = np.flatnonzero((y == label) & (phone == group))
            positions[label, group] = ids
            if len(ids):
                counts[label, group] = rng.multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=draws)
    npos, nneg = int(y.sum()), int((y == 0).sum())
    total = np.zeros(draws); within = np.zeros(draws); between = np.zeros(draws)
    rows = []
    for a, b in BLOCKS:
        ip, inn = positions[1, a], positions[0, b]
        weight = len(ip)/npos * len(inn)/nneg
        if len(ip) and len(inn):
            difference = np.mean([concordance(pa[ip], pa[inn])-concordance(pf[ip], pf[inn]) for pf,pa in zip(full, altered)], axis=0)
            point = float(difference.mean())
            samples = np.einsum("bi,ij,bj->b", counts[1,a], difference, counts[0,b], optimize=True)/(len(ip)*len(inn))
            lo, hi = np.quantile(samples, [.025, .975])
            contribution = weight*samples
        else:
            point, lo, hi = np.nan, np.nan, np.nan
            contribution = np.zeros(draws)
        total += contribution
        if a == b:
            within += contribution
        else:
            between += contribution
        rows.append({"block": f"P{a}_N{b}", "positive_n": len(ip), "negative_n": len(inn), "weight": weight,
                     "delta_block_auc": point, "delta_block_auc_lower": lo, "delta_block_auc_upper": hi,
                     "delta_auc_contribution": weight*point if weight else 0.,
                     "contribution_lower": weight*lo if weight else 0., "contribution_upper": weight*hi if weight else 0.,
                     "small_cell": len(ip) < 10 or len(inn) < 20})
    for name, samples in (("TOTAL", total), ("WITHIN", within), ("BETWEEN", between)):
        point = sum(r["delta_auc_contribution"] for r in rows[:4] if name == "TOTAL" or ((r["block"] in ("P1_N1", "P2_N2")) == (name == "WITHIN")))
        lo, hi = np.quantile(samples, [.025, .975])
        rows.append({"block": name, "delta_auc_contribution": point, "contribution_lower": lo, "contribution_upper": hi})
    direct = np.mean([roc_auc_score(y, pa)-roc_auc_score(y, pf) for pf,pa in zip(full,altered)])
    if not np.isclose(rows[4]["delta_auc_contribution"], direct, atol=1e-10):
        raise RuntimeError("配对块变化未还原总AUC变化")
    return rows


def combination_profiles(source, part, scores):
    """任务2 第11步：固定年龄×教育×IADL画像；小于10名阳性的组仅保留计数。"""
    source_key, target_key = cell_keys(source).to_numpy(), cell_keys(part).to_numpy()
    y = part.label.to_numpy(dtype=int); phone = part[PHONE[0]].to_numpy()
    rows = []
    for a in (1, 2):
        for key in sorted(set(target_key[(y == 1) & (phone == a)])):
            positive = (y == 1) & (phone == a) & (target_key == key)
            source_mask = (source[PHONE[0]].to_numpy() == a) & (source_key == key)
            n = int(positive.sum())
            for b in (1, 2):
                negative = (y == 0) & (phone == b)
                nn = int(negative.sum())
                eligible = n >= 10 and nn >= 20
                loss = float(1-concordance(scores[positive], scores[negative]).mean()) if eligible else np.nan
                weight = n/int(y.sum()) * nn/int((y == 0).sum())
                rows.append({"positive_phone": a, "negative_phone": b, "stratum": key, "positive_n": n,
                             "negative_n": nn, "source_n": int(source_mask.sum()),
                             "source_positive_n": int(source.loc[source_mask, "label"].sum()), "eligible": eligible,
                             "rank_loss": loss, "error_contribution": weight*loss if eligible else np.nan})
    return rows


def draw(blocks, gaps, uncertainty, output):
    """任务2 第11步：固定图号，不按东部效果挑选展示人群。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    primary = blocks.loc[(blocks.model == "mlp") & (blocks.policy == PRIMARY) & (blocks.arm == "full")]
    sites = sorted(primary.loc[primary.group != "SOURCE_VALIDATION", "group"].unique())
    labels = ["P1/N1", "P1/N2", "P2/N1", "P2/N2"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for ax, site in zip(axes.ravel(), sites):
        for group, offset, color in (("SOURCE_VALIDATION", -.1, "gray"), (site, .1, "tab:blue")):
            for i, block in enumerate([f"P{a}_N{b}" for a,b in BLOCKS]):
                v = primary.loc[(primary.group == group) & (primary.block == block)]
                ax.scatter(np.full(len(v), i+offset), v.block_auc, color=color, label=group if i == 0 else None)
        ax.axhline(.5, color="black", linestyle="--", linewidth=.8)
        ax.set_xticks(range(4)); ax.set_xticklabels(labels); ax.set_ylim(0, 1)
        ax.set_title(site); ax.set_ylabel("Block AUC; inspect positive/negative counts"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output / "figures/01_within_between_phone_block_auc.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    g = gaps.loc[(gaps.model == "mlp") & (gaps.policy == PRIMARY) & (gaps.arm == "full")]
    for ax, component in zip(axes, ("composition_component", "separation_component")):
        for i, site in enumerate(sites):
            v = g.loc[g.group == site]
            ax.scatter(np.full(len(v), i), v[component], color="tab:blue")
        ax.axhline(0, color="black", linewidth=.8); ax.set_xticks(range(len(sites))); ax.set_xticklabels(sites)
        ax.set_ylabel("Contribution to target minus source-validation AUC"); ax.set_title(component)
    fig.tight_layout(); fig.savefig(output / "figures/02_composition_vs_block_separation.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for ax, site in zip(axes.ravel(), sites):
        names = ("WITHIN", "BETWEEN", "TOTAL")
        v = uncertainty.loc[(uncertainty.group == site) & (uncertainty.model == "mlp") & (uncertainty.policy == PRIMARY)].set_index("block").loc[list(names)]
        ax.vlines(range(3), v.contribution_lower, v.contribution_upper, color="tab:blue")
        ax.scatter(range(3), v.delta_auc_contribution, color="tab:blue")
        ax.axhline(0, color="black", linewidth=.8); ax.set_xticks(range(3)); ax.set_xticklabels(names)
        ax.set_title(site); ax.set_ylabel("Phone masking: weighted AUC change contribution")
    fig.tight_layout(); fig.savefig(output / "figures/03_phone_masking_tradeoff_decomposition.png", dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data/data_mission2.csv")
    parser.add_argument("--previous", type=Path, default=ROOT / "outputs/analysis_2611/task2_step10_east_phone_mechanism_control")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/analysis_2611/task2_step11_east_phone_pairwise_decomposition")
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    if args.bootstrap < 200:
        raise ValueError("配对重采样次数至少200")
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != EXPECTED_HASH:
        raise ValueError("固定CSV版本不一致")
    old = json.loads((args.previous / "experiment_metadata.json").read_text(encoding="utf-8"))
    versions = {"python": platform.python_version(), "torch": torch.__version__, "sklearn": sklearn.__version__}
    if not old["completed"] or versions != old["versions"] or old["input_sha256"] != EXPECTED_HASH:
        raise ValueError("请使用第10步相同环境及完整结果")
    # 允许Windows提交时的换行转换，但不忽略实际代码变化。
    prior_text = Path(prior_code.__file__).read_text(encoding="utf-8")
    prior_hashes = {hashlib.sha256(text.encode("utf-8")).hexdigest() for text in
                    (prior_text, prior_text.replace("\n", "\r\n"))}
    if old["code_sha256"] not in prior_hashes:
        raise ValueError("第10步代码与已完成结果不一致，请先恢复对应版本")
    frame = attach_regions(read_raw(args.data))
    source, validation, target = held_out_split(frame, "east", seed=42, column=REGION)
    if not frame[PHONE[0]].isin([1,2]).all():
        raise ValueError("手机编码不是完整的1/2，不能静默丢弃")
    if set(source.sample_id) & set(validation.sample_id) or (set(source.sample_id) | set(validation.sample_id)) & set(target.sample_id):
        raise RuntimeError("训练、验证、东部人群重叠")
    previous = pd.read_csv(args.previous / "metrics.csv", dtype={"group": str})
    args.output.mkdir(parents=True, exist_ok=False); (args.output / "figures").mkdir()
    meta = {"completed": False, "input_sha256": EXPECTED_HASH, "versions": versions, "seeds": SEEDS,
            "arms": ["full", "mask_phone_group"], "primary_policy": PRIMARY, "blocks": BLOCKS,
            "target_used_for_training_or_selection": False, "individual_records_exported": False,
            "source_split_seed": 42, "bootstrap_draws": args.bootstrap, "combination_features": ["age", "education", "IADL"],
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "bootstrap_scope": "固定完整与屏蔽评分，先平均三种子比较矩阵；在标签×手机组内共享配对重采样。区间只反映固定组人数下的抽样变化，不包含训练与构成权重不确定性。",
            "limitations": "反复查看东部的探索性诊断，不是独立确认或因果归因。AUC差分解是数学恒等式；来源验证混合站点与目标县区的块内差异仍包含其他特征构成变化。来源验证参与选模，其成绩可能乐观。手机组内AUC不受统一组间评分平移影响，但不能单独判断原因。MLP本身可学非线性交互，不能由分解宣称其缺乏交互能力。树三种子可能相同，不算三次独立支持。小类别块报告精确点值但标记不稳；组合阳性不足10人只报告计数。"}
    dump(args.output / "experiment_metadata.json", meta)
    splits, _ = prepare_splits((source, validation, target), environment_column=REGION)
    predictions, audits = {}, []
    for arm, removed in (("full", ()), ("mask_phone_group", PHONE)):
        for seed in SEEDS:
            print(f"任务2 第11步：复现{arm} seed={seed}", flush=True)
            old_record = json.loads((args.previous / "full" / str(seed) / "training.json").read_text(encoding="utf-8"))
            fixed = {key: int(old_record["epochs"][key]) for key in ("matched_source_auc_epoch", PRIMARY)}
            result, record = fit_mlp([masked(s, removed) for s in splits], seed, fixed)
            predictions[arm, "mlp", seed] = result
            folder = args.output / arm / str(seed); folder.mkdir(parents=True)
            dump(folder / "training.json", {"removed": removed, **record})
            raw = [f.copy() for f in (source, validation, target)]
            for part in raw:
                for feature in removed:
                    part[feature] = 0.
            fitted = train_model(raw[0], raw[1], "shallow_tree", seed)
            predictions[arm, "shallow_tree", seed] = {"fixed_tree": {"epoch": 100,
                "validation": probability(fitted, "shallow_tree", raw[1]), "target": probability(fitted, "shallow_tree", raw[2])}}
            dump(folder / "tree_training.json", fitted[2])
            for family in ("mlp", "shallow_tree"):
                for policy, pred in predictions[arm, family, seed].items():
                    for group, part, p in [("POOL", validation, pred["validation"])] + [(str(site), target.loc[target[ENVIRONMENT] == site], pred["target"][target[ENVIRONMENT].to_numpy() == site]) for site in sorted(target[ENVIRONMENT].unique())]:
                        scope = "source_validation" if group == "POOL" else "east"
                        match = previous.loc[(previous.arm == arm) & (previous.model == family) & (previous.seed == seed) & (previous.policy == policy) & (previous.scope == scope) & (previous.group == group)]
                        if len(match) != 1:
                            raise ValueError("第10步成绩参照缺失或重复")
                        actual = metrics(part.label.to_numpy(dtype=int), p)
                        delta = {key: float(actual[key]-match.iloc[0][key]) for key in ("auc", "auprc", "brier")}
                        audits.append({"arm": arm, "model": family, "seed": seed, "policy": policy, "group": group,
                                       **{"difference_"+k: v for k,v in delta.items()}, "passed": all(abs(v) <= 2e-6 for v in delta.values())})
            pd.DataFrame(audits).to_csv(args.output / "model_reproduction_audits.csv", index=False, encoding="utf-8-sig")
            if not all(r["passed"] for r in audits):
                raise RuntimeError("第10步模型未复现，暂停机制分解，请核对复现检查表")
    blocks, identities, profiles = [], [], []
    for (arm, family, seed), policies in predictions.items():
        for policy, pred in policies.items():
            base = {"arm": arm, "model": family, "seed": seed, "policy": policy}
            cohorts = [("SOURCE_VALIDATION", validation, pred["validation"])]
            cohorts += [(str(site), target.loc[target[ENVIRONMENT] == site], pred["target"][target[ENVIRONMENT].to_numpy() == site]) for site in sorted(target[ENVIRONMENT].unique())]
            for group, part, p in cohorts:
                rows, audit = block_metrics(part, p)
                blocks.extend({**base, "group": group, **row} for row in rows)
                identities.append({**base, "group": group, **audit})
                if group != "SOURCE_VALIDATION":
                    profiles.extend({**base, "group": group, **row} for row in combination_profiles(source, part, p))
    table = pd.DataFrame(blocks)
    gap_rows, gap_summary = [], []
    for (arm, family, seed, policy), part in table.groupby(["arm", "model", "seed", "policy"], sort=False):
        base = {"arm": arm, "model": family, "seed": seed, "policy": policy}
        src = part.loc[part.group == "SOURCE_VALIDATION"].to_dict("records")
        for group in sorted(set(part.group)-{"SOURCE_VALIDATION"}):
            rows, summary = gap_decomposition(src, part.loc[part.group == group].to_dict("records"))
            gap_rows.extend({**base, "group": group, **r} for r in rows)
            gap_summary.append({**base, "group": group, **summary})
    keys = ["model", "seed", "policy", "group", "block"]
    baseline = table.loc[table.arm == "full", keys+["block_auc", "auc_contribution", "error_contribution"]]
    paired = table.loc[table.arm == "mask_phone_group"].merge(baseline, on=keys, suffixes=("", "_full"), validate="one_to_one")
    for key in ("block_auc", "auc_contribution", "error_contribution"):
        paired["delta_"+key] = paired[key]-paired[key+"_full"]
    uncertainty = []
    for family, policies in (("mlp", (PRIMARY, "own_source_loss")), ("shallow_tree", ("fixed_tree",))):
        for policy in policies:
            for site in sorted(target[ENVIRONMENT].unique()):
                ix = target[ENVIRONMENT].to_numpy() == site
                full = np.stack([predictions["full", family, seed][policy]["target"][ix] for seed in SEEDS])
                altered = np.stack([predictions["mask_phone_group", family, seed][policy]["target"][ix] for seed in SEEDS])
                print(f"任务2 第11步：{site} {family}/{policy} 配对区间", flush=True)
                rows = bootstrap_change(target.loc[ix], full, altered, args.bootstrap)
                uncertainty.extend({"model": family, "policy": policy, "group": str(site), **r} for r in rows)
    tables = {"auc_pair_blocks": table, "auc_identity_audits": pd.DataFrame(identities),
              "source_target_gap_blocks": pd.DataFrame(gap_rows), "source_target_gap_summary": pd.DataFrame(gap_summary),
              "paired_block_changes": paired, "paired_bootstrap_block_changes": pd.DataFrame(uncertainty),
              "positive_combination_profiles": pd.DataFrame(profiles)}
    for name, data in tables.items():
        data.to_csv(args.output / (name+".csv"), index=False, encoding="utf-8-sig")
    draw(table, pd.DataFrame(gap_summary), pd.DataFrame(uncertainty), args.output)
    meta.update(completed=True, models_reproduced=True, all_auc_identities_passed=True)
    dump(args.output / "experiment_metadata.json", meta)
    print(f"任务2 第11步完成：{args.output}")


if __name__ == "__main__":
    main()

"""任务2 第12步：检查类别输入表达是否限制MLP在手机组内的区分能力。

来源训练集拟合共同缺失填补、独热类别和标准化；不读取东部标签选编码。
二分类仅保留一个独热列；数值等宽对照按特征块放置，不改变信息。
同一来源划分、网络隐藏层、训练种子与选模规则；只导出聚合结果。
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
from elderly_health.data import FEATURES, REGION, ENVIRONMENT, attach_regions, held_out_split, prepare_splits, read_raw
from elderly_health.evaluation import metrics
from elderly_health.models import SimpleMLP
from east_phone_mechanism_control import fit_mlp, PHONE, POLICIES, SEEDS
import east_phone_pairwise_decomposition as previous_code
from east_phone_pairwise_decomposition import block_metrics, bootstrap_change

EXPECTED_HASH = "969f0c1c82faafcfa402f37572e4a8ceb324492062cf1e590465f82dc6b5b6de"
PRIMARY = "matched_source_loss_epoch"
ARMS = ("numeric_original", "numeric_padded", "onehot_scaled", "onehot_raw")
CONTRASTS = (("numeric_padded", "numeric_original"), ("onehot_scaled", "numeric_padded"),
             ("onehot_raw", "onehot_scaled"), ("onehot_scaled", "numeric_original"))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def make_encodings(frames, splits, processor):
    """任务2 第12步：共同来源中位数填补，排除新增缺失指示信息的影响。"""
    imputed = [processor.imputer.transform(f[list(FEATURES)]) for f in frames]
    encoder = OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    onehot = [encoder.fit_transform(imputed[0])]
    onehot.extend(encoder.transform(values) for values in imputed[1:])
    scaler = StandardScaler().fit(onehot[0])
    scaled = [scaler.transform(values).astype(np.float32) for values in onehot]
    width = onehot[0].shape[1]
    if width < len(FEATURES):
        raise ValueError("独热宽度小于原特征数，无法构造预设等宽对照")
    padded = [np.zeros((len(s.x), width), dtype=np.float32) for s in splits]
    columns, offset = [], 0
    for j, (feature, categories) in enumerate(zip(FEATURES, encoder.categories_)):
        count = 1 if len(categories) == 2 else len(categories)
        if len(categories) > 8:
            raise ValueError("来源特征类别超过预期，先检查是否仍为同一套分组编码")
        for out, split in zip(padded, splits):
            out[:, offset] = split.x[:, j]
        columns.append({"feature": feature, "source_categories": [float(v) for v in categories],
                        "start": offset, "width": count, "numeric_slot": offset,
                        "binary_drop_first": len(categories) == 2})
        offset += count
    if offset != width:
        raise RuntimeError("独热特征块宽度与等宽输入不一致")
    arrays = {"numeric_original": [s.x for s in splits], "numeric_padded": padded,
              "onehot_scaled": scaled, "onehot_raw": onehot}
    altered, audits = {}, []
    for arm, values in arrays.items():
        altered[arm] = []
        for name, split, x in zip(("train", "validation", "target"), splits, values):
            x = np.ascontiguousarray(x, dtype=np.float32)
            if not np.isfinite(x).all() or x.shape[0] != len(split.y):
                raise RuntimeError("输入含非有限值或样本顺序长度不一致")
            altered[arm].append(replace(split, x=x))
            audits.append({"arm": arm, "split": name, "rows": len(x), "input_dim": x.shape[1],
                           "nonzero_columns": int(np.any(x != 0, axis=0).sum()), "finite": True})
    if len({altered[arm][0].x.shape[1] for arm in ARMS[1:]}) != 1:
        raise RuntimeError("数值等宽、独热标准化、独热原值网络输入宽度不相同")
    unknown = []
    for i in (1, 2):
        frame, values = frames[i], imputed[i]
        cohorts = {"SOURCE_VALIDATION": np.ones(len(frame), dtype=bool)} if i == 1 else {
            str(site): frame[ENVIRONMENT].to_numpy() == site for site in sorted(frame[ENVIRONMENT].unique())}
        for group, keep in cohorts.items():
            for j, feature in enumerate(FEATURES):
                unseen = ~np.isin(values[keep, j], encoder.categories_[j])
                unknown.append({"group": group, "feature": feature, "n": int(keep.sum()),
                                "unknown_n": int(unseen.sum()), "unknown_fraction": float(unseen.mean())})
    protocol = {"shared_imputation": "共同来源训练中位数；不新增缺失指示列", "onehot_width": int(width),
                "drop": "if_binary", "unknown_handling": "ignore，单独审计未知类别；二分类未知与基准类同为零",
                "blocks": columns, "imputation_medians": [float(v) for v in processor.imputer.statistics_]}
    return altered, audits, unknown, protocol


def compare_effects(table, keys, measures):
    """任务2 第12步：显式区分维度/位置、编码、标准化三个对照。"""
    results = []
    for treatment, reference in CONTRASTS:
        left = table.loc[table.arm == treatment]
        right = table.loc[table.arm == reference, keys+list(measures)]
        joined = left.merge(right, on=keys, suffixes=("", "_reference"), validate="one_to_one")
        joined["reference"] = reference
        for key in measures:
            joined["delta_"+key] = joined[key]-joined[key+"_reference"]
        results.append(joined)
    return pd.concat(results, ignore_index=True)


def draw(blocks, scores, changes, tree_reference, output):
    """任务2 第12步：固定四站点展示，主图聚焦手机编码1组内。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sites = sorted(blocks.loc[blocks.group != "SOURCE_VALIDATION", "group"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for ax, site in zip(axes.ravel(), sites):
        part = blocks.loc[(blocks.group == site) & (blocks.policy == PRIMARY) & (blocks.block == "P1_N1")]
        for i, arm in enumerate(ARMS):
            v = part.loc[part.arm == arm].sort_values("seed")
            ax.scatter(i+np.array([-.08, 0, .08]), v.block_auc, color=f"C{i}")
        ref = tree_reference.loc[(tree_reference.group == site) & (tree_reference.block == "P1_N1"), "block_auc"].mean()
        ax.axhline(ref, color="gray", linestyle="--", label="Previous tree reference")
        ax.axhline(.5, color="black", linewidth=.7)
        ax.set_xticks(range(len(ARMS))); ax.set_xticklabels(ARMS, rotation=25, ha="right", fontsize=8)
        npos, nneg = int(part.iloc[0].positive_n), int(part.iloc[0].negative_n)
        ax.set_title(f"{site}: phone 1, positives={npos}, negatives={nneg}")
        ax.set_ylabel("Within-phone-1 AUC; three seeds"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output / "figures/01_encoding_within_phone1_auc.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, policy in zip(axes, (PRIMARY, "own_source_loss")):
        for i, site in enumerate(sites):
            v = changes.loc[(changes.group == site) & (changes.policy == policy) & (changes.arm == "onehot_scaled") & (changes.reference == "numeric_padded") & (changes.block == "P1_N1")]
            ax.vlines(i, v.delta_block_auc_lower, v.delta_block_auc_upper, color="tab:blue")
            ax.scatter(np.full(len(v), i), v.delta_block_auc, color="tab:blue")
        ax.axhline(0, color="black", linewidth=.8); ax.set_xticks(range(len(sites))); ax.set_xticklabels(sites)
        ax.set_title(policy); ax.set_ylabel("One-hot scaled minus numeric padded: phone-1 AUC")
    fig.tight_layout(); fig.savefig(output / "figures/02_encoding_effect_paired_intervals.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, policy in zip(axes, (PRIMARY, "own_source_loss")):
        for offset, arm, color in zip((-.2, 0, .2), ARMS[1:], ("tab:orange", "tab:blue", "tab:green")):
            for i, site in enumerate(sites):
                v = scores.loc[(scores.policy == policy) & (scores.arm == arm) & (scores.group == site)]
                base = scores.loc[(scores.policy == policy) & (scores.arm == "numeric_original") & (scores.group == site), ["seed", "auc"]]
                joined = v.merge(base, on="seed", suffixes=("", "_base"), validate="one_to_one")
                ax.scatter(np.full(len(joined), i+offset), joined.auc-joined.auc_base, color=color, label=arm if i == 0 else None)
        ax.axhline(0, color="black", linewidth=.8); ax.set_xticks(range(len(sites))); ax.set_xticklabels(sites)
        ax.set_title(policy); ax.set_ylabel("Whole-site AUC change vs original numeric"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output / "figures/03_encoding_whole_site_tradeoffs.png", dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data/data_mission2.csv")
    parser.add_argument("--previous", type=Path, default=ROOT / "outputs/analysis_2611/task2_step11_east_phone_pairwise_decomposition")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/analysis_2611/task2_step12_east_mlp_encoding_control")
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    if args.bootstrap < 200:
        raise ValueError("重采样次数至少200")
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != EXPECTED_HASH:
        raise ValueError("固定CSV版本不一致")
    old = json.loads((args.previous / "experiment_metadata.json").read_text(encoding="utf-8"))
    versions = {"python": platform.python_version(), "torch": torch.__version__, "sklearn": sklearn.__version__}
    if not old["completed"] or old["versions"] != versions or old["input_sha256"] != EXPECTED_HASH:
        raise ValueError("请使用第11步相同环境及完整结果")
    previous_text = Path(previous_code.__file__).read_text(encoding="utf-8")
    hashes = {hashlib.sha256(text.encode("utf-8")).hexdigest() for text in (previous_text, previous_text.replace("\n", "\r\n"))}
    if old["code_sha256"] not in hashes:
        raise ValueError("第11步代码不对应已完成结果，请恢复对应版本")
    frame = attach_regions(read_raw(args.data))
    frames = held_out_split(frame, "east", seed=42, column=REGION)
    source, validation, target = frames
    if set(source.sample_id) & set(validation.sample_id) or (set(source.sample_id) | set(validation.sample_id)) & set(target.sample_id):
        raise RuntimeError("来源训练、验证、东部重叠")
    splits, processor = prepare_splits(frames, environment_column=REGION)
    encodings, input_audits, unknown, protocol = make_encodings(frames, splits, processor)
    old_blocks = pd.read_csv(args.previous / "auc_pair_blocks.csv", dtype={"group": str})
    tree_reference = old_blocks.loc[(old_blocks.arm == "full") & (old_blocks.model == "shallow_tree")]
    args.output.mkdir(parents=True, exist_ok=False); (args.output / "figures").mkdir()
    parameter_counts = {}
    # 只用于记录结构；共享训练函数每次重新设种子，不影响实际初始化。
    with torch.random.fork_rng(devices=[]):
        for arm, altered in encodings.items():
            parameter_counts[arm] = sum(p.numel() for p in SimpleMLP(altered[0].x.shape[1]).parameters())
    if len({parameter_counts[arm] for arm in ARMS[1:]}) != 1:
        raise RuntimeError("等宽对照与两种独热的网络总参数量不一致")
    meta = {"completed": False, "input_sha256": EXPECTED_HASH, "versions": versions, "seeds": SEEDS,
            "source_split_seed": 42, "arms": ARMS, "policies": POLICIES, "parameter_counts": parameter_counts,
            "primary_contrast": "onehot_scaled vs numeric_padded；371082手机编码1组内AUC；原来源损失固定轮次并与各自来源损失选模交叉检查",
            "contrasts": CONTRASTS, "bootstrap_draws": args.bootstrap,
            "target_used_for_preprocessing_training_or_selection": False, "individual_records_exported": False,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "bootstrap_scope": "固定三种子评分后，手机组×标签内配对重采样；不含训练与组构成不确定性",
            "limitations": "独热仅是诊断，不预设全部变量都应无序。等宽对照控制总参数量并对齐二分类位置，不能完全控制有效连接数、优化几何及初始化与信息的对应关系。数值填补相同，不新增缺失指示；独热未知类别另行审计。相同轮次不代表相同收敛程度。反复查看东部，结果属于探索，不能用微小单次改善宣布根因。树参照复用第11步，三个种子可能是同一模型。"}
    dump(args.output / "experiment_metadata.json", meta)
    dump(args.output / "encoding_protocol.json", protocol)
    pd.DataFrame(input_audits).to_csv(args.output / "input_dimension_audits.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(unknown).to_csv(args.output / "unknown_category_audits.csv", index=False, encoding="utf-8-sig")
    predictions, blocks, scores, audits, score_profiles = {}, [], [], [], []
    for arm in ARMS:
        print(f"任务2 第12步：{arm} 输入{encodings[arm][0].x.shape[1]}维，参数{parameter_counts[arm]}个", flush=True)
        for seed in SEEDS:
            original = json.loads((args.previous / "full" / str(seed) / "training.json").read_text(encoding="utf-8"))
            fixed = {key: int(original["epochs"][key]) for key in ("matched_source_auc_epoch", PRIMARY)}
            result, record = fit_mlp(encodings[arm], seed, fixed)
            if arm == "numeric_original" and record["epochs"] != original["epochs"]:
                raise RuntimeError("原MLP选模轮次未复现第11步，暂停编码解释")
            predictions[arm, seed] = result
            folder = args.output / arm / str(seed); folder.mkdir(parents=True)
            dump(folder / "training.json", record)
            for policy, pred in result.items():
                base = {"arm": arm, "seed": seed, "policy": policy, "epoch": pred["epoch"]}
                cohorts = [("SOURCE_VALIDATION", validation, pred["validation"]), ("EAST_POOL", target, pred["target"])]
                cohorts += [(str(site), target.loc[target[ENVIRONMENT] == site], pred["target"][target[ENVIRONMENT].to_numpy() == site]) for site in sorted(target[ENVIRONMENT].unique())]
                for group, part, p in cohorts:
                    scores.append({**base, "group": group, **metrics(part.label.to_numpy(dtype=int), p)})
                    if group == "EAST_POOL":
                        continue
                    observed, identity = block_metrics(part, p)
                    blocks.extend({**base, "group": group, **r} for r in observed)
                    for phone in (1, 2):
                        positive = (part.label.to_numpy() == 1) & (part[PHONE[0]].to_numpy() == phone)
                        negative = (part.label.to_numpy() == 0) & (part[PHONE[0]].to_numpy() == phone)
                        score_profiles.append({**base, "group": group, "phone": phone, "positive_n": int(positive.sum()),
                            "negative_n": int(negative.sum()), "positive_mean_score": float(p[positive].mean()) if positive.any() else np.nan,
                            "negative_mean_score": float(p[negative].mean()) if negative.any() else np.nan,
                            "score_sd": float(p[positive | negative].std())})
                    if arm == "numeric_original":
                        match = old_blocks.loc[(old_blocks.arm == "full") & (old_blocks.model == "mlp") & (old_blocks.seed == seed) & (old_blocks.policy == policy) & (old_blocks.group == group)].set_index("block")
                        if len(match) != 4:
                            raise ValueError("第11步完整MLP的四块参照缺失")
                        differences = {r["block"]: float(r["block_auc"]-match.loc[r["block"], "block_auc"]) for r in observed}
                        audits.append({"seed": seed, "policy": policy, "group": group,
                                       "max_block_auc_difference": max(abs(v) for v in differences.values()),
                                       "identity_error": identity["identity_error"], "passed": all(abs(v) <= 2e-6 for v in differences.values())})
            if arm == "numeric_original":
                pd.DataFrame(audits).to_csv(args.output / "baseline_reproduction_audits.csv", index=False, encoding="utf-8-sig")
                if not all(r["passed"] for r in audits):
                    raise RuntimeError("原19维MLP未复现第11步，暂停编码解释")
    blocks, scores = pd.DataFrame(blocks), pd.DataFrame(scores)
    score_effects = compare_effects(scores, ["seed", "policy", "group"], ("auc", "auprc", "brier"))
    block_effects = compare_effects(blocks, ["seed", "policy", "group", "block"], ("block_auc", "auc_contribution", "error_contribution"))
    uncertainty = []
    for arm, reference in CONTRASTS:
        for policy in (PRIMARY, "own_source_loss"):
            for site in sorted(target[ENVIRONMENT].unique()):
                ix = target[ENVIRONMENT].to_numpy() == site
                full = np.stack([predictions[reference, seed][policy]["target"][ix] for seed in SEEDS])
                changed = np.stack([predictions[arm, seed][policy]["target"][ix] for seed in SEEDS])
                print(f"任务2 第12步：{site} {arm} 对照 {reference}，{policy}配对区间", flush=True)
                rows = bootstrap_change(target.loc[ix], full, changed, args.bootstrap, seed=26111012)
                uncertainty.extend({"arm": arm, "reference": reference, "policy": policy, "group": str(site), **r} for r in rows)
    tables = {"metrics": scores, "auc_pair_blocks": blocks, "paired_metric_effects": score_effects,
              "paired_block_effects": block_effects, "paired_bootstrap_effects": pd.DataFrame(uncertainty),
              "score_distribution_profiles": pd.DataFrame(score_profiles), "previous_tree_reference": tree_reference}
    for name, table in tables.items():
        table.to_csv(args.output / (name+".csv"), index=False, encoding="utf-8-sig")
    draw(blocks, scores, pd.DataFrame(uncertainty), tree_reference, args.output)
    meta.update(completed=True, baseline_reproduced=True, equal_width_parameter_counts_passed=True, all_auc_identities_passed=True)
    dump(args.output / "experiment_metadata.json", meta)
    print(f"任务2 第12步完成：{args.output}")


if __name__ == "__main__":
    main()

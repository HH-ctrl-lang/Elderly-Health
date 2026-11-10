"""任务2 第13步：来源类别内部手机组平衡权重，检查少数阳性的学习不足。

只改变来源训练样本权重，保持阳性/阴性各自的总权重与原19维输入。
不增加患者、不改目标标签、不加权验证集；完整、阳性平衡、阴性平衡、
两类平衡及阳性权重打乱五个预设条件，三种子与四种来源选模规则。
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
from elderly_health.data import REGION, ENVIRONMENT, attach_regions, held_out_split, prepare_splits, read_raw
from elderly_health.evaluation import metrics
from elderly_health.training import fit, predict, get_device
from east_phone_mechanism_control import PHONE, POLICIES, SEEDS
from east_phone_pairwise_decomposition import block_metrics, bootstrap_change
import east_mlp_encoding_control as previous_code

EXPECTED_HASH = "969f0c1c82faafcfa402f37572e4a8ceb324492062cf1e590465f82dc6b5b6de"
PRIMARY = "matched_source_loss_epoch"
ARMS = ("full", "positive_phone_balance", "negative_phone_balance", "both_phone_balance", "shuffled_positive_weights")
CONTRASTS = tuple((arm, "full") for arm in ARMS[1:]) + (("positive_phone_balance", "shuffled_positive_weights"),)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def effective_n(weights):
    weights = np.asarray(weights, dtype=float)
    return float(weights.sum()**2/np.square(weights).sum())


def make_weights(source):
    """任务2 第13步：每类总权重保持原人数；被平衡类内部手机1/2各占一半。"""
    y, phone = source.label.to_numpy(dtype=int), source[PHONE[0]].to_numpy()
    if not np.isin(y, [0,1]).all() or not np.isin(phone, [1,2]).all():
        raise ValueError("来源结局或手机编码不符合固定预设")
    group_weights = {}
    for label in (0,1):
        ix = y == label
        for group in (1,2):
            count = int((ix & (phone == group)).sum())
            if not count:
                raise ValueError("来源类别×手机组为空，不能平衡")
            group_weights[label,group] = int(ix.sum())/(2*count)
    arms = {name: np.ones(len(source), dtype=float) for name in ARMS}
    for arm, labels in (("positive_phone_balance", (1,)), ("negative_phone_balance", (0,)), ("both_phone_balance", (0,1))):
        for label in labels:
            for group in (1,2):
                arms[arm][(y == label) & (phone == group)] = group_weights[label,group]
    # 固定一次类内打乱，三次训练复用同一权重，避免混入重新抽权重的变化。
    pos = np.flatnonzero(y == 1)
    arms["shuffled_positive_weights"][pos] = np.random.default_rng(26111013).permutation(arms["positive_phone_balance"][pos])
    groups, classes = [], []
    for arm, weights in arms.items():
        # 与共享fit完全相同的float32归一化方式，审计实际进入训练的权重。
        weights = weights.astype(np.float32)
        weights = weights/weights.mean()
        arms[arm] = weights
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise RuntimeError("训练权重无效")
        for label in (0,1):
            keep = y == label
            mass = float(weights[keep].sum(dtype=np.float64))
            if not np.isclose(mass, keep.sum(), atol=2e-4, rtol=1e-6):
                raise RuntimeError("阳性或阴性总权重发生改变")
            classes.append({"arm": arm, "label": label, "n": int(keep.sum()), "weight_sum": mass,
                            "effective_n": effective_n(weights[keep]), "total_preserved": True})
            for group in (1,2):
                subset = weights[keep & (phone == group)]
                groups.append({"arm": arm, "label": label, "phone": group, "n": len(subset),
                               "weight_sum": float(subset.sum(dtype=np.float64)), "weight_mean": float(subset.mean()),
                               "weight_min": float(subset.min()), "weight_max": float(subset.max()),
                               "fraction_within_label": float(subset.sum(dtype=np.float64)/mass), "effective_n": effective_n(subset)})
    same_distribution = bool(np.allclose(np.sort(arms["positive_phone_balance"][y == 1]),
                                        np.sort(arms["shuffled_positive_weights"][y == 1]), atol=1e-6, rtol=1e-6))
    if not same_distribution or not np.allclose(arms["shuffled_positive_weights"][y == 0], 1):
        raise RuntimeError("打乱对照未保留阳性权重分布或改变阴性权重")
    protocol = {"shuffle_seed": 26111013, "shuffled_within_positive_only": True,
                "positive_weight_distribution_preserved": same_distribution,
                "global_effective_n": {arm: effective_n(w) for arm,w in arms.items()},
                "rule": "被平衡类中，权重=该类总人数/(2×该类该手机组人数)；每类总权重不变"}
    return arms, pd.DataFrame(groups), pd.DataFrame(classes), protocol


def group_probe(model, splits, frames, epoch, policy):
    """任务2 第13步：只读来源训练/验证组内学习情况，不读取东部结局或改变随机数。"""
    rows = []
    for scope, split, frame in (("source_train", splits[0], frames[0]), ("source_validation", splits[1], frames[1])):
        p = predict(model, split.x, get_device("cpu"))
        for group in (1,2):
            keep = frame[PHONE[0]].to_numpy() == group
            y = split.y[keep].astype(int)
            rows.append({"scope": scope, "phone": group, "epoch": epoch, "policy": policy,
                         "positive_n": int(y.sum()), "negative_n": int((y == 0).sum()), **metrics(y, p[keep])})
    return rows


def fit_weighted(splits, frames, weights, seed, fixed):
    """任务2 第13步：保持原训练轨迹配置，复制所需状态，来源验证仍不加权。"""
    snapshots, lowest, loss_state, loss_epoch = {}, float("inf"), None, None
    curve = []

    def capture(model, record):
        nonlocal lowest, loss_state, loss_epoch
        epoch = int(record["epoch"])
        needed = [name for name, value in fixed.items() if value == epoch]
        better = record["val_loss"] < lowest
        if needed or better:
            state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            for name in needed:
                snapshots[name] = (state, epoch)
            if better:
                lowest, loss_epoch, loss_state = float(record["val_loss"]), epoch, state
        if epoch <= 5 or epoch % 10 == 0 or needed:
            curve.extend(group_probe(model, splits, frames, epoch, "learning_curve"))

    cfg = TrainConfig(model="baseline", environment_granularity="region", seed=seed, selection_start_epoch=1, device="cpu")
    model, history = fit(splits[0], splits[1], cfg, sample_weights=weights, epoch_callback=capture)
    if any(name not in snapshots for name in fixed):
        raise RuntimeError("加权轨迹早于原固定轮次停止，不能直接比较；请上传当前结果核对")
    snapshots["own_source_auc"] = ({k: v.detach().cpu().clone() for k,v in model.state_dict().items()}, int(history["best_epoch"]))
    snapshots["own_source_loss"] = (loss_state, loss_epoch)
    result, selected = {}, []
    for policy in POLICIES:
        state, epoch = snapshots[policy]
        model.load_state_dict(state)
        pv = predict(model, splits[1].x, get_device("cpu"))
        pt = predict(model, splits[2].x, get_device("cpu"))
        result[policy] = {"epoch": epoch, "validation": pv, "target": pt}
        selected.extend(group_probe(model, splits, frames, epoch, policy))
    return result, {"config": cfg.to_dict(), "training": history, "epochs": {k:v["epoch"] for k,v in result.items()}}, curve, selected


def paired_effects(table, keys, measures):
    """任务2 第13步：保持种子、选模规则和人群一致，输出所有预设对照。"""
    rows = []
    for arm, reference in CONTRASTS:
        joined = table.loc[table.arm == arm].merge(table.loc[table.arm == reference, keys+list(measures)],
                  on=keys, suffixes=("", "_reference"), validate="one_to_one")
        joined["reference"] = reference
        for key in measures:
            joined["delta_"+key] = joined[key]-joined[key+"_reference"]
        rows.append(joined)
    return pd.concat(rows, ignore_index=True)


def draw(blocks, effects, intervals, source_groups, output):
    """任务2 第13步：图号固定，展示四站点、打乱对照、全站代价和来源学习。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sites = sorted(blocks.loc[blocks.group != "SOURCE_VALIDATION", "group"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    for ax, site in zip(axes.ravel(), sites):
        part = blocks.loc[(blocks.group == site) & (blocks.policy == PRIMARY) & (blocks.block == "P1_N1")]
        for i, arm in enumerate(ARMS):
            v = part.loc[part.arm == arm].sort_values("seed")
            ax.scatter(i+np.array([-.08, 0, .08]), v.block_auc, color=f"C{i}")
        ax.axhline(.5, color="gray", linestyle="--"); ax.set_xticks(range(len(ARMS)))
        ax.set_xticklabels(ARMS, rotation=25, ha="right", fontsize=7)
        ax.set_title(f"{site}: phone-1 positives={int(part.iloc[0].positive_n)}")
        ax.set_ylabel("Within-phone-1 AUC; three seeds")
    fig.tight_layout(); fig.savefig(output / "figures/01_weighting_within_phone1_auc.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, policy in zip(axes, (PRIMARY, "own_source_loss")):
        for offset, reference, color in ((-.1, "full", "tab:blue"), (.1, "shuffled_positive_weights", "tab:orange")):
            for i, site in enumerate(sites):
                v = intervals.loc[(intervals.arm == "positive_phone_balance") & (intervals.reference == reference) & (intervals.policy == policy) & (intervals.group == site) & (intervals.block == "P1_N1")]
                ax.vlines(i+offset, v.delta_block_auc_lower, v.delta_block_auc_upper, color=color)
                ax.scatter(np.full(len(v), i+offset), v.delta_block_auc, color=color, label="vs "+reference if i == 0 else None)
        ax.axhline(0, color="black", linewidth=.8); ax.set_xticks(range(len(sites))); ax.set_xticklabels(sites)
        ax.set_title(policy); ax.set_ylabel("Positive balancing: phone-1 AUC change"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output / "figures/02_positive_weighting_vs_shuffle.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, policy in zip(axes, (PRIMARY, "own_source_loss")):
        for offset, arm, color in zip((-.24, -.08, .08, .24), ARMS[1:], ("tab:blue", "tab:green", "tab:red", "tab:orange")):
            for i, site in enumerate(sites):
                v = effects.loc[(effects.policy == policy) & (effects.arm == arm) & (effects.reference == "full") & (effects.group == site)]
                ax.scatter(np.full(len(v), i+offset), v.delta_auc, color=color, label=arm if i == 0 else None)
        ax.axhline(0, color="black", linewidth=.8); ax.set_xticks(range(len(sites))); ax.set_xticklabels(sites)
        ax.set_title(policy); ax.set_ylabel("Whole-site AUC change vs full"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(output / "figures/03_weighting_whole_site_tradeoffs.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    for offset, scope, color in ((-.15, "source_train", "tab:green"), (0, "source_validation", "tab:orange"), (.15, "371082", "tab:blue")):
        for i, arm in enumerate(ARMS):
            if scope == "371082":
                values = blocks.loc[(blocks.group == scope) & (blocks.policy == PRIMARY) & (blocks.arm == arm) & (blocks.block == "P1_N1"), "block_auc"]
            else:
                values = source_groups.loc[(source_groups.scope == scope) & (source_groups.policy == PRIMARY) & (source_groups.arm == arm) & (source_groups.phone == 1), "auc"]
            ax.scatter(np.full(len(values), i+offset), values, color=color, label=scope if i == 0 else None)
    ax.set_xticks(range(len(ARMS))); ax.set_xticklabels(ARMS, rotation=20, ha="right", fontsize=8)
    ax.axhline(.5, color="gray", linestyle="--"); ax.set_ylabel("Phone-1 AUC at matched source-loss epoch")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(output / "figures/04_source_learning_vs_east_generalization.png", dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data/data_mission2.csv")
    parser.add_argument("--previous", type=Path, default=ROOT / "outputs/analysis_2611/task2_step12_east_mlp_encoding_control")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/analysis_2611/task2_step13_east_phone_group_weight_control")
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    if args.bootstrap < 200:
        raise ValueError("配对重采样次数至少200")
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != EXPECTED_HASH:
        raise ValueError("固定CSV版本不一致")
    old = json.loads((args.previous / "experiment_metadata.json").read_text(encoding="utf-8"))
    versions = {"python": platform.python_version(), "torch": torch.__version__, "sklearn": sklearn.__version__}
    if not old["completed"] or versions != old["versions"] or old["input_sha256"] != EXPECTED_HASH:
        raise ValueError("请使用第12步相同环境及完整结果")
    previous_text = Path(previous_code.__file__).read_text(encoding="utf-8")
    hashes = {hashlib.sha256(t.encode("utf-8")).hexdigest() for t in (previous_text, previous_text.replace("\n", "\r\n"))}
    if old["code_sha256"] not in hashes:
        raise ValueError("第12步代码不对应已完成结果，请恢复对应版本")
    frame = attach_regions(read_raw(args.data))
    frames = held_out_split(frame, "east", seed=42, column=REGION)
    source, validation, target = frames
    if set(source.sample_id) & set(validation.sample_id) or (set(source.sample_id) | set(validation.sample_id)) & set(target.sample_id):
        raise RuntimeError("来源训练、验证、东部样本重叠")
    if not frame[PHONE[0]].isin([1,2]).all():
        raise ValueError("手机编码不是完整的1/2")
    weights, weight_groups, weight_classes, weight_protocol = make_weights(source)
    splits, _ = prepare_splits(frames, environment_column=REGION)
    previous = pd.read_csv(args.previous / "metrics.csv", dtype={"group": str})
    args.output.mkdir(parents=True, exist_ok=False); (args.output / "figures").mkdir()
    meta = {"completed": False, "input_sha256": EXPECTED_HASH, "versions": versions, "seeds": SEEDS,
            "source_split_seed": 42, "arms": ARMS, "policies": POLICIES, "contrasts": CONTRASTS,
            "primary_contrast": "positive_phone_balance vs full及shuffled_positive_weights；371082手机编码1组内AUC；匹配原来源损失轮次",
            "target_used_for_preprocessing_training_or_selection": False, "individual_records_exported": False,
            "bootstrap_draws": args.bootstrap, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "training_dependency_sha256": hashlib.sha256((PROJECT / "src/elderly_health/training.py").read_bytes()).hexdigest(),
            "validation_weighting": "不加样本权重，原pos_weight=3保持不变",
            "source_probes": "每轮1至5、每10轮及固定参照轮次只读来源组内指标；不消费随机数或修改模型；完整模型复现另行核对",
            "limitations": "权重不产生新患者，阳性内部平衡会降低有效样本量；无改善不能证明阳性覆盖充分。只有一次固定类内打乱，不代表所有随机权重情况。组内AUC改善才支持组内学习改善，只有组间或概率抬升不够。相同轮次与各自来源选模并列报告，不能事后挑最佳规则。固定评分配对区间不含训练与构成不确定性。长期反复查看东部，属于探索而非独立确认或临床因果证据。"}
    dump(args.output / "experiment_metadata.json", meta)
    dump(args.output / "weight_protocol.json", weight_protocol)
    weight_groups.to_csv(args.output / "weight_group_audits.csv", index=False, encoding="utf-8-sig")
    weight_classes.to_csv(args.output / "weight_class_audits.csv", index=False, encoding="utf-8-sig")
    predictions, scores, blocks, audits, curves, source_groups = {}, [], [], [], [], []
    for arm in ARMS:
        for seed in SEEDS:
            print(f"任务2 第13步：{arm} seed={seed}", flush=True)
            original = json.loads((args.previous / "numeric_original" / str(seed) / "training.json").read_text(encoding="utf-8"))
            fixed = {k: int(original["epochs"][k]) for k in ("matched_source_auc_epoch", PRIMARY)}
            result, record, curve, selected = fit_weighted(splits, frames, None if arm == "full" else weights[arm], seed, fixed)
            if arm == "full" and record["epochs"] != original["epochs"]:
                raise RuntimeError("原模型选模轮次未复现第12步")
            predictions[arm,seed] = result
            folder = args.output / arm / str(seed); folder.mkdir(parents=True)
            dump(folder / "training.json", record)
            pd.DataFrame(curve).to_csv(folder / "source_group_learning_curve.csv", index=False, encoding="utf-8-sig")
            curves.extend({"arm": arm, "seed": seed, **r} for r in curve)
            source_groups.extend({"arm": arm, "seed": seed, **r} for r in selected)
            for policy, pred in result.items():
                base = {"arm": arm, "seed": seed, "policy": policy, "epoch": pred["epoch"]}
                cohorts = [("SOURCE_VALIDATION", validation, pred["validation"]), ("EAST_POOL", target, pred["target"])]
                cohorts += [(str(site), target.loc[target[ENVIRONMENT] == site], pred["target"][target[ENVIRONMENT].to_numpy() == site]) for site in sorted(target[ENVIRONMENT].unique())]
                for group, part, p in cohorts:
                    observed = metrics(part.label.to_numpy(dtype=int), p)
                    scores.append({**base, "group": group, **observed})
                    if group != "EAST_POOL":
                        block_rows, _ = block_metrics(part, p)
                        blocks.extend({**base, "group": group, **r} for r in block_rows)
                    if arm == "full":
                        oldrow = previous.loc[(previous.arm == "numeric_original") & (previous.seed == seed) & (previous.policy == policy) & (previous.group == group)]
                        if len(oldrow) != 1:
                            raise ValueError("第12步原编码参照缺失或重复")
                        delta = {k: float(observed[k]-oldrow.iloc[0][k]) for k in ("auc", "auprc", "brier")}
                        audits.append({"seed": seed, "policy": policy, "group": group, **{"difference_"+k:v for k,v in delta.items()},
                                       "passed": all(abs(v) <= 2e-6 for v in delta.values())})
            if arm == "full":
                pd.DataFrame(audits).to_csv(args.output / "baseline_reproduction_audits.csv", index=False, encoding="utf-8-sig")
                if not all(r["passed"] for r in audits):
                    raise RuntimeError("只读来源探针或环境导致原模型未复现，请先核对复现检查表")
    scores, blocks = pd.DataFrame(scores), pd.DataFrame(blocks)
    metric_effects = paired_effects(scores, ["seed", "policy", "group"], ("auc", "auprc", "brier"))
    block_effects = paired_effects(blocks, ["seed", "policy", "group", "block"], ("block_auc", "auc_contribution", "error_contribution"))
    intervals = []
    for arm, reference in CONTRASTS:
        for policy in (PRIMARY, "own_source_loss"):
            for site in sorted(target[ENVIRONMENT].unique()):
                ix = target[ENVIRONMENT].to_numpy() == site
                full = np.stack([predictions[reference,seed][policy]["target"][ix] for seed in SEEDS])
                changed = np.stack([predictions[arm,seed][policy]["target"][ix] for seed in SEEDS])
                print(f"任务2 第13步：{site} {arm} 对照 {reference}，{policy}配对区间", flush=True)
                rows = bootstrap_change(target.loc[ix], full, changed, args.bootstrap, seed=26111013)
                intervals.extend({"arm": arm, "reference": reference, "policy": policy, "group": str(site), **r} for r in rows)
    tables = {"metrics": scores, "auc_pair_blocks": blocks, "paired_metric_effects": metric_effects,
              "paired_block_effects": block_effects, "paired_bootstrap_effects": pd.DataFrame(intervals),
              "source_group_learning_curves": pd.DataFrame(curves), "selected_source_group_profiles": pd.DataFrame(source_groups)}
    for name, table in tables.items():
        table.to_csv(args.output / (name+".csv"), index=False, encoding="utf-8-sig")
    draw(blocks, metric_effects, pd.DataFrame(intervals), pd.DataFrame(source_groups), args.output)
    meta.update(completed=True, baseline_reproduced=True, class_weight_totals_preserved=True, all_auc_identities_passed=True)
    dump(args.output / "experiment_metadata.json", meta)
    print(f"任务2 第13步完成：{args.output}")


if __name__ == "__main__":
    main()

"""任务2 第15步：固定MLP，用来源教育残差修正检查组内信息利用。

保持原19维MLP、三种子和两种来源选模规则；不重新训练主模型。
修正参数仅由来源训练集拟合，固定正则强度，不按东部选择条件。
只输出聚合结果；本实验为预测机制干预，不能解释教育的临床因果效应。
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
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
from elderly_health.config import TrainConfig
from elderly_health.data import FEATURES, REGION, ENVIRONMENT, attach_regions, held_out_split, prepare_splits, read_raw
from elderly_health.training import fit, predict, get_device
from elderly_health.evaluation import metrics
from east_phone_mechanism_control import PHONE, SEEDS
from east_phone_pairwise_decomposition import block_metrics, bootstrap_change

EXPECTED_HASH = "969f0c1c82faafcfa402f37572e4a8ceb324492062cf1e590465f82dc6b5b6de"
ARMS = ("full", "phone_intercept", "education_shared", "education_by_phone", "education_shuffled")
POLICIES = ("own_source_auc", "own_source_loss")
RIDGE = 20.0
CONTRASTS = (("phone_intercept", "full"), ("education_shared", "phone_intercept"),
             ("education_by_phone", "phone_intercept"), ("education_by_phone", "education_shared"),
             ("education_by_phone", "education_shuffled"))


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def design_spec(source):
    """任务2 第15步：类别及中心化常数只读来源训练特征，不读取标签。"""
    categories = sorted(source.education.unique().tolist())
    onehot = (source.education.to_numpy()[:, None] == np.array(categories)[None, :]).astype(float)
    phone = source[PHONE[0]].to_numpy()
    if not np.isin(phone, [1, 2]).all():
        raise ValueError("手机编码超出固定的1/2")
    return {"categories": categories, "global_mean": onehot.mean(axis=0),
            "group_mean": {g: onehot[phone == g].mean(axis=0) for g in (1, 2)}}


def correction_design(frame, spec, arm):
    """任务2 第15步：手机组截距单列；教育列按来源均值中心化，未知类别不给教育修正。"""
    phone = frame[PHONE[0]].to_numpy()
    intercepts = np.column_stack([phone == g for g in (1, 2)]).astype(float)
    cats = frame.education.to_numpy()
    onehot = (cats[:, None] == np.array(spec["categories"])[None, :]).astype(float)
    known = onehot.sum(axis=1) > 0
    if arm == "phone_intercept":
        return intercepts
    if arm == "education_shared":
        education = onehot-spec["global_mean"]
        education[~known] = 0
    elif arm in ("education_by_phone", "education_shuffled"):
        parts = []
        for group in (1, 2):
            centered = onehot-spec["group_mean"][group]
            centered[~known] = 0
            parts.append(centered*(phone == group)[:, None])
        education = np.column_stack(parts)
    else:
        raise ValueError("未知修正条件")
    return np.column_stack([intercepts, education])


def shuffled_education(source):
    """任务2 第15步：固定一次同来源站点、同手机组内教育置换，不按标签分组。"""
    result = source.copy()
    rng = np.random.default_rng(26111015)
    for _, index in source.groupby([ENVIRONMENT, PHONE[0]], sort=True).groups.items():
        result.loc[index, "education"] = rng.permutation(source.loc[index, "education"].to_numpy())
    return result


def fit_correction(design, offset, labels):
    """任务2 第15步：固定主模型logit，最小化来源训练加权BCE加教育系数L2惩罚。"""
    labels = np.asarray(labels, dtype=float)
    weight = np.where(labels == 1, 3.0, 1.0)
    penalty = np.full(design.shape[1], RIDGE)
    penalty[:2] = 1e-6

    def objective(beta):
        score = offset+design@beta
        loss = np.sum(weight*(np.logaddexp(0, score)-labels*score))+.5*np.sum(penalty*beta**2)
        gradient = design.T@(weight*(expit(score)-labels))+penalty*beta
        return loss, gradient

    result = minimize(objective, np.zeros(design.shape[1]), method="L-BFGS-B", jac=True,
                      options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-7})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"来源修正优化失败：{result.message}")
    return result.x, {"converged": bool(result.success), "iterations": int(result.nit),
                      "objective": float(result.fun), "maximum_gradient": float(np.abs(result.jac).max())}


def apply_correction(probability, design, beta):
    return expit(logit(np.clip(np.asarray(probability, dtype=float), 1e-12, 1-1e-12))+design@beta)


def draw(blocks, intervals, metric_table, output):
    """任务2 第15步：固定图号，显示四站点、预设对照及其他人群代价。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sites = sorted(set(blocks.group)-{"SOURCE_TRAIN", "SOURCE_VALIDATION", "EAST_ALL"})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    for ax, site in zip(axes.ravel(), sites):
        part = blocks.loc[(blocks.group == site) & (blocks.policy == "own_source_loss") & (blocks.block == "P1_N1")]
        for i, arm in enumerate(ARMS):
            rows = part.loc[part.arm == arm].sort_values("seed")
            ax.scatter(i+np.linspace(-.07, .07, len(rows)), rows.block_auc, color=f"C{i}")
        ax.axhline(.5, color="gray", linestyle="--")
        ax.set_xticks(range(len(ARMS))); ax.set_xticklabels(ARMS, rotation=25, ha="right", fontsize=7)
        ax.set_title(site); ax.set_ylabel("Within-phone-1 AUC; three seeds")
    fig.tight_layout(); fig.savefig(output/"figures/01_education_correction_within_phone1.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    refs = ("phone_intercept", "education_shared", "education_shuffled")
    for ax, policy in zip(axes, POLICIES):
        for j, ref in enumerate(refs):
            for i, site in enumerate(sites):
                row = intervals.loc[(intervals.group == site) & (intervals.policy == policy) &
                      (intervals.arm == "education_by_phone") & (intervals.reference == ref) & (intervals.block == "P1_N1")].iloc[0]
                x = i+(j-1)*.18
                ax.vlines(x, row.delta_block_auc_lower, row.delta_block_auc_upper, color=f"C{j}")
                ax.scatter(x, row.delta_block_auc, color=f"C{j}", label="vs "+ref if i == 0 else None)
        ax.axhline(0, color="black", linewidth=.8); ax.set_xticks(range(len(sites))); ax.set_xticklabels(sites)
        ax.set_title(policy); ax.set_ylabel("Paired within-phone-1 AUC change"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output/"figures/02_education_specificity_controls.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    part = blocks.loc[(blocks.policy == "own_source_loss") & (blocks.arm == "education_by_phone") & (blocks.block == "P1_N1")]
    groups = ["SOURCE_TRAIN", "SOURCE_VALIDATION"]+sites
    for i, group in enumerate(groups):
        rows = part.loc[part.group == group]
        axes[0].scatter(np.full(len(rows), i), rows.block_auc, color="tab:blue")
    axes[0].set_xticks(range(len(groups))); axes[0].set_xticklabels(groups, rotation=30, ha="right", fontsize=7)
    axes[0].set_ylabel("Conditional education correction: phone-1 AUC")
    base = metric_table.loc[(metric_table.arm == "full") & (metric_table.policy == "own_source_loss")]
    altered = metric_table.loc[(metric_table.arm == "education_by_phone") & (metric_table.policy == "own_source_loss")]
    merged = altered.merge(base, on=["seed", "policy", "group"], suffixes=("", "_base"))
    for i, group in enumerate(sites+["EAST_ALL"]):
        rows = merged.loc[merged.group == group]
        axes[1].scatter(np.full(len(rows), i), rows.auc-rows.auc_base, color="tab:orange")
    axes[1].axhline(0, color="black", linewidth=.8); axes[1].set_xticks(range(len(sites)+1))
    axes[1].set_xticklabels(sites+["EAST_ALL"]); axes[1].set_ylabel("Whole-population AUC change vs full")
    fig.tight_layout(); fig.savefig(output/"figures/03_source_learning_and_whole_site_costs.png", dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT/"data/data_mission2.csv")
    parser.add_argument("--previous", type=Path, default=ROOT/"outputs/analysis_2611/task2_step14_east_tree_feature_advantage")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/analysis_2611/task2_step15_east_education_residual_control")
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args()
    if args.bootstrap < 200:
        raise ValueError("配对重采样至少200次")
    versions = {"python": platform.python_version(), "torch": torch.__version__, "sklearn": sklearn.__version__}
    old = json.loads((args.previous/"experiment_metadata.json").read_text(encoding="utf-8"))
    if not old["completed"] or not old["models_reproduced"] or old["versions"] != versions:
        raise ValueError("请先完成第14步，并保持相同运行环境")
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != EXPECTED_HASH or old["input_sha256"] != EXPECTED_HASH:
        raise ValueError("固定CSV版本不一致")
    previous_code = Path(__file__).with_name("east_tree_feature_advantage.py").read_text(encoding="utf-8")
    if old["code_sha256"] not in {hashlib.sha256(s.encode()).hexdigest() for s in (previous_code, previous_code.replace("\n", "\r\n"))}:
        raise ValueError("第14步代码与已完成结果不一致")
    frames = held_out_split(attach_regions(read_raw(args.data)), "east", seed=42, column=REGION)
    if any(set(frames[a].sample_id)&set(frames[b].sample_id) for a,b in ((0,1),(0,2),(1,2))):
        raise RuntimeError("来源训练、验证、东部样本重叠")
    splits, processor = prepare_splits(frames, environment_column=REGION)
    imputed = []
    for frame in frames:
        part = frame.copy()
        part.loc[:, list(FEATURES)] = processor.imputer.transform(part[list(FEATURES)])
        imputed.append(part)
    source, validation, target = imputed
    spec = design_spec(source); shuffled = shuffled_education(source)
    designs = {arm: [correction_design(f, spec, arm) for f in imputed] for arm in ARMS[1:]}
    designs["education_shuffled"][0] = correction_design(shuffled, spec, "education_shuffled")
    reference = pd.read_csv(args.previous/"phone1_baseline_metrics.csv", dtype={"group": str})
    args.output.mkdir(parents=True, exist_ok=False); (args.output/"figures").mkdir()
    meta = {"completed": False, "input_sha256": EXPECTED_HASH, "versions": versions,
            "scipy_version": __import__("scipy").__version__, "seeds": SEEDS, "arms": ARMS,
            "policies": POLICIES, "ridge": RIDGE, "positive_weight": 3.0,
            "source_split_seed": 42, "shuffle_seed": 26111015, "bootstrap_draws": args.bootstrap,
            "target_used_for_training_or_selection": False, "individual_records_exported": False,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "primary_comparison": "来源损失选模，371082手机1：分组教育修正vs手机截距、共享教育表和教育打乱；110114及其余站点并列检查",
            "limits": "固定MLP之后的来源训练残差修正，不证明原网络训练时被手机带偏。残差拟合使用主模型已见过的来源训练人群，存在训练内过拟合，必须同时看来源验证。来源验证仍参与原MLP选模；所有条件和正则预先固定，不按验证或东部挑最好条件。打乱对照仅打乱修正表的来源拟合教育输入，主MLP输入不变，评估使用真实教育。教育编码可能是代理因素，不能作临床因果解释。区间固定三次模型评分，只反映当前患者抽样，未包含重新训练及多次探索的不确定性。新类别教育修正为零，仍保留对应手机截距。"}
    dump(args.output/"experiment_metadata.json", meta)
    metric_rows, blocks, audits, coefficients, optimizer_rows = [], [], [], [], []
    scores = {}
    cohorts = [("SOURCE_TRAIN", 0, np.ones(len(source), dtype=bool)),
               ("SOURCE_VALIDATION", 1, np.ones(len(validation), dtype=bool)),
               ("EAST_ALL", 2, np.ones(len(target), dtype=bool))]
    cohorts += [(str(site), 2, target[ENVIRONMENT].to_numpy() == site) for site in sorted(target[ENVIRONMENT].unique())]
    for seed in SEEDS:
        print(f"任务2 第15步：复现MLP，随机种子={seed}", flush=True)
        lowest, loss_state, loss_epoch = float("inf"), None, None
        def capture(model, record):
            nonlocal lowest, loss_state, loss_epoch
            if record["val_loss"] < lowest:
                lowest, loss_epoch = float(record["val_loss"]), int(record["epoch"])
                loss_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        cfg = TrainConfig(model="baseline", environment_granularity="region", seed=seed, selection_start_epoch=1, device="cpu")
        model, history = fit(splits[0], splits[1], cfg, epoch_callback=capture)
        auc_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        folder = args.output/str(seed); folder.mkdir()
        dump(folder/"training.json", {"config": cfg.to_dict(), "training": history,
                                      "epochs": {"own_source_auc": int(history["best_epoch"]), "own_source_loss": loss_epoch}})
        for policy, state in (("own_source_auc", auc_state), ("own_source_loss", loss_state)):
            model.load_state_dict(state)
            original = [predict(model, split.x, get_device("cpu")).astype(float) for split in splits]
            # 在拟合任何修正表前，逐站点及来源验证手机1指标复现第14步。
            for name, index, keep in cohorts:
                if name in ("SOURCE_TRAIN", "EAST_ALL"):
                    continue
                mask = keep & (imputed[index][PHONE[0]].to_numpy() == 1)
                actual = metrics(imputed[index].loc[mask, "label"].to_numpy(dtype=int), original[index][mask])
                row = reference.loc[(reference.group == name) & (reference.model == "mlp") &
                                    (reference.policy == policy) & (reference.seed == seed)]
                if len(row) != 1:
                    raise ValueError("第14步参照缺失或重复")
                differences = {key: float(actual[key]-row.iloc[0][key]) for key in ("auc", "auprc", "brier")}
                passed = all(abs(v) <= 2e-6 for v in differences.values())
                audits.append({"seed": seed, "policy": policy, "group": name, "passed": passed,
                               **{"difference_"+k:v for k,v in differences.items()}})
                if not passed:
                    raise RuntimeError("MLP未复现，停止修正实验")
            for arm in ARMS:
                print(f"任务2 第15步：{policy}，条件={arm}", flush=True)
                if arm == "full":
                    predictions = original
                else:
                    offset = logit(np.clip(original[0], 1e-12, 1-1e-12))
                    beta, optimizer = fit_correction(designs[arm][0], offset, source.label.to_numpy(dtype=int))
                    optimizer_rows.append({"seed": seed, "policy": policy, "arm": arm, **optimizer})
                    names = ["phone1_intercept", "phone2_intercept"]
                    if arm == "education_shared":
                        names += [f"education_{cat}" for cat in spec["categories"]]
                    elif arm != "phone_intercept":
                        names += [f"phone{g}_education_{cat}" for g in (1,2) for cat in spec["categories"]]
                    coefficients.extend({"seed": seed, "policy": policy, "arm": arm, "term": name, "coefficient": float(v)}
                                        for name,v in zip(names,beta))
                    predictions = [apply_correction(p, design, beta) for p,design in zip(original, designs[arm])]
                for name, index, keep in cohorts:
                    part = imputed[index].loc[keep]; p = predictions[index][keep]
                    scores[seed, policy, arm, name] = p
                    metric_rows.append({"seed": seed, "policy": policy, "arm": arm, "group": name,
                                        **metrics(part.label.to_numpy(dtype=int), p)})
                    rows, identity = block_metrics(part, p)
                    blocks.extend({"seed": seed, "policy": policy, "arm": arm, "group": name, **row} for row in rows)
                    if arm == "phone_intercept":
                        before, _ = block_metrics(part, original[index][keep])
                        for a,b in zip(before,rows):
                            if a["block"] in ("P1_N1", "P2_N2") and np.isfinite(a["block_auc"]):
                                if abs(a["block_auc"]-b["block_auc"]) > 1e-10:
                                    raise RuntimeError("仅手机截距不应改变组内排序，出现数值饱和或实现错误")
    metric_table, block_table = pd.DataFrame(metric_rows), pd.DataFrame(blocks)
    intervals = []
    for name,index,keep in cohorts:
        if name == "SOURCE_TRAIN":
            continue
        for policy in POLICIES:
            for arm,reference_arm in CONTRASTS:
                base = np.stack([scores[seed,policy,reference_arm,name] for seed in SEEDS])
                altered = np.stack([scores[seed,policy,arm,name] for seed in SEEDS])
                rows = bootstrap_change(imputed[index].loc[keep], base, altered, args.bootstrap, seed=26111015)
                intervals.extend({"group": name, "policy": policy, "arm": arm, "reference": reference_arm, **row} for row in rows)
    interval_table = pd.DataFrame(intervals)
    support = []
    for name,index,keep in cohorts:
        part = imputed[index].loc[keep]
        support.append({"group": name, "n": len(part), "unknown_education_n": int((~part.education.isin(spec["categories"])).sum())})
    tables = {"metrics": metric_table, "auc_pair_blocks": block_table, "paired_intervals": interval_table,
              "model_reproduction_audits": pd.DataFrame(audits), "correction_coefficients": pd.DataFrame(coefficients),
              "optimizer_audits": pd.DataFrame(optimizer_rows), "category_support": pd.DataFrame(support)}
    for name,table in tables.items():
        table.to_csv(args.output/(name+".csv"), index=False, encoding="utf-8-sig")
    dump(args.output/"design_protocol.json", {"categories": [float(v) for v in spec["categories"]],
         "global_center": spec["global_mean"].tolist(), "group_centers": {str(k):v.tolist() for k,v in spec["group_mean"].items()},
         "ridge": RIDGE, "intercepts_penalty": 1e-6, "fitting_scope": "来源训练集；验证与东部只评估"})
    draw(block_table, interval_table, metric_table, args.output)
    meta.update(completed=True, models_reproduced=True, phone_intercept_preserves_within_group_auc=True)
    dump(args.output/"experiment_metadata.json", meta)
    print(f"任务2 第15步完成，输出目录：{args.output}")


if __name__ == "__main__":
    main()

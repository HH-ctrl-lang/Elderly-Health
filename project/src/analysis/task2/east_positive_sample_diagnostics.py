"""任务2 第8步：371082本地阳性信息量与重复加权效应的对照。

保持第7步外层测试划分；固定各折验证集和190名训练阴性。
比较20名不同阳性、全部约50名不同阳性、20名阳性重复到相同条数。
本地训练使用目标标签，仅作监督诊断，不作为域泛化成绩。
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
import east_local_learnability as previous_code
from east_local_learnability import make_plan, train_model, probability, counts, sample_counts
from elderly_health.data import REGION, ENVIRONMENT, attach_regions, held_out_split, read_raw
from elderly_health.evaluation import metrics

EXPECTED_HASH = "969f0c1c82faafcfa402f37572e4a8ceb324492062cf1e590465f82dc6b5b6de"
SITE = "371082"
SEEDS = (42, 43, 44)
MODELS = ("shallow_tree", "mlp")
ARMS = ("local_20", "local_all_positive", "local_repeat20", "source_20", "source_many_positive")


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def ordered_class(frame, label, seed):
    part = frame.loc[frame.label == label]
    return part.iloc[np.random.default_rng(seed).permutation(len(part))].copy()


def build_arms(local_train, source_train, seed):
    """任务2 第8步：嵌套阳性子集、固定阴性，并设同条数重复阳性对照。"""
    lp = ordered_class(local_train, 1, seed)
    ln = ordered_class(local_train, 0, seed+1)
    high_n = len(lp)
    if high_n <= 20 or len(ln) < 190:
        raise ValueError("本折可用阳性或阴性人数不满足预设设计")
    # 本数据各折约50名阳性、190名阴性；全部本地阳性均用于高信息条件。
    ln = ln.iloc[:190]
    sp = ordered_class(source_train, 1, seed+2)
    sn = ordered_class(source_train, 0, seed+3).iloc[:190]
    if len(sp) < high_n or len(sn) != 190:
        raise ValueError("来源人数不足以匹配本地条件")
    low = lp.iloc[:20]
    extra = low.iloc[np.random.default_rng(seed+4).integers(0, 20, size=high_n-20)]
    arms = {"local_20": pd.concat([ln, low]), "local_all_positive": pd.concat([ln, lp]),
            "local_repeat20": pd.concat([ln, low, extra]), "source_20": pd.concat([sn, sp.iloc[:20]]),
            "source_many_positive": pd.concat([sn, sp.iloc[:high_n]])}
    # 排序随机流固定；重复条件特意包含重复训练行，任何验证/测试行都不能进入。
    for name, part in arms.items():
        arms[name] = part.iloc[np.random.default_rng(seed+5).permutation(len(part))].copy()
    return arms, high_n


def summarize(fold_rows):
    """任务2 第8步：先在每个重复中平均四折，不把重叠训练折当独立重复。"""
    keys = ["seed", "model", "arm"]
    table = pd.DataFrame(fold_rows)
    result = table.groupby(keys, sort=False).agg(mean_fold_auc=("auc", "mean"), mean_fold_auprc=("auprc", "mean"),
                                                mean_fold_brier=("brier", "mean"), fold_auc_sd=("auc", "std"),
                                                fold_count=("fold", "nunique")).reset_index()
    return result


def paired_gains(summary):
    """任务2 第8步：预先声明四个对照，检查新阳性信息是否超过重复加权收益。"""
    comparisons = [("local_all_positive", "local_20"), ("local_all_positive", "local_repeat20"),
                   ("source_many_positive", "source_20"), ("local_all_positive", "source_many_positive")]
    rows = []
    for seed in SEEDS:
        for model in MODELS:
            group = summary.loc[(summary.seed == seed) & (summary.model == model)].set_index("arm")
            for left, right in comparisons:
                rows.append({"seed": seed, "model": model, "treatment": left, "reference": right,
                             **{f"delta_{key}": float(group.loc[left, f"mean_fold_{key}"]-group.loc[right, f"mean_fold_{key}"]) for key in ("auc", "auprc", "brier")}})
    return pd.DataFrame(rows)


def draw(summary, gains, source_reference, output):
    """任务2 第8步：展示三个重复，不凭单个点或均值判定根因。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "figures"
    folder.mkdir()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    labels = {"local_20": "Local 20 unique", "local_all_positive": "Local all unique", "local_repeat20": "Local 20 repeated",
              "source_20": "Source 20 unique", "source_many_positive": "Source matched unique"}
    for ax, model in zip(axes, MODELS):
        part = summary.loc[summary.model == model]
        for i, arm in enumerate(ARMS):
            values = part.loc[part.arm == arm].sort_values("seed")
            ax.scatter(i+np.array([-.1, 0, .1]), values.mean_fold_auc, label=labels[arm])
        ref = source_reference.loc[source_reference.model == model, "mean_fold_auc"].mean()
        ax.axhline(ref, color="gray", linestyle="--", label="Previous full-source reference")
        ax.set_xticks(range(len(ARMS))); ax.set_xticklabels([labels[x] for x in ARMS], rotation=30, ha="right", fontsize=8)
        ax.set_title(model); ax.set_ylabel("Mean outer-fold AUC; three repeats")
        ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(folder / "01_positive_information_vs_repetition.png", dpi=190); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    contrasts = [("local_all_positive", "local_20"), ("local_all_positive", "local_repeat20"), ("source_many_positive", "source_20")]
    names = ["Local: more unique vs 20", "Local: unique vs repeated", "Source: more unique vs 20"]
    for ax, model in zip(axes, MODELS):
        for offset, seed in zip((-.15, 0, .15), SEEDS):
            values = [float(gains.loc[(gains.model == model) & (gains.seed == seed) & (gains.treatment == a) & (gains.reference == b), "delta_auc"].iloc[0]) for a,b in contrasts]
            ax.scatter(np.arange(3)+offset, values, label=f"seed {seed}")
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_xticks(range(3)); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_title(model); ax.set_ylabel("Paired AUC change"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(folder / "02_paired_positive_sample_gains.png", dpi=190); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data/data_mission2.csv")
    parser.add_argument("--previous", type=Path, default=ROOT / "outputs/analysis_2611/task2_step7_east_local_learnability")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/analysis_2611/task2_step8_east_positive_sample_diagnostics")
    args = parser.parse_args()
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != EXPECTED_HASH:
        raise ValueError("固定研究CSV不一致")
    old = json.loads((args.previous / "experiment_metadata.json").read_text(encoding="utf-8"))
    versions = {"python": platform.python_version(), "torch": torch.__version__, "sklearn": sklearn.__version__}
    if not old["completed"] or any(old[k] != v for k,v in versions.items()):
        raise ValueError("请使用已完成第7步的相同环境")
    previous_hash = hashlib.sha256(Path(previous_code.__file__).read_bytes()).hexdigest()
    # Windows提交可能转换换行；只容许LF/CRLF差异，不忽略代码内容变化。
    previous_text = Path(previous_code.__file__).read_text(encoding="utf-8")
    equivalent_hashes = {hashlib.sha256(previous_text.encode("utf-8")).hexdigest(),
                         hashlib.sha256(previous_text.replace("\n", "\r\n").encode("utf-8")).hexdigest()}
    if old["code_sha256"] not in equivalent_hashes:
        raise ValueError("第7步代码与运行记录不同，不能直接复用其外折设计和来源参考")
    frame = attach_regions(read_raw(args.data))
    plan, _ = make_plan(frame)
    plan = [p for p in plan if p["site"] == SITE]
    county_ids = set(frame.loc[frame[ENVIRONMENT] == SITE, "sample_id"])
    for seed in SEEDS:
        seed_plan = [p for p in plan if p["seed"] == seed]
        testing_ids = [i for p in seed_plan for i in p["test"].sample_id]
        if len(seed_plan) != 4 or len(testing_ids) != len(set(testing_ids)) or set(testing_ids) != county_ids:
            raise RuntimeError("每次重复必须让县区每人恰好进入一次外层测试")
    source_train, source_val, _ = held_out_split(frame, "east", seed=42, column=REGION)
    if (source_train[REGION] == "east").any() or (source_val[REGION] == "east").any():
        raise RuntimeError("来源数据混入东部")
    prior = pd.read_csv(args.previous / "outer_fold_metrics.csv", dtype={"site_code": str})
    reference = pd.read_csv(args.previous / "repeat_summary.csv", dtype={"site_code": str})
    reference = reference.loc[(reference.site_code == SITE) & (reference["mode"] == "source_full") & reference.model.isin(MODELS)]
    if len(reference) != 6:
        raise ValueError("第7步完整来源参照缺失或重复")
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {"completed": False, "input_sha256": EXPECTED_HASH, "versions": versions, "seeds": SEEDS,
                "site": SITE, "models": MODELS, "arms": ARMS, "previous_code_sha256": previous_hash,
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "local_labels_used_for_supervised_diagnostic": True, "outer_test_used_for_selection": False,
                "validation": "本地使用固定的全部内层验证集；来源验证匹配相同类别人数，所有条件固定验证样本",
                "limitations": "仅371082，不能代表东部全部。增加阳性同时改变训练类别比例，因此设置重复阳性对照；新增信息与重复加权仍影响优化路径及预处理。无明显收益不能证明特征无信息。重叠外折和重复不作为独立显著性检验样本。本步验证集比第7步匹配条件更大，仅在本步内部比较20人和全部阳性；旧完整来源成绩仅供背景参考。"}
    dump(args.output / "experiment_metadata.json", metadata)
    fold_rows, audit_rows = [], []
    for item in plan:
        seed, fold = item["seed"], item["fold"]
        test, local_val = item["test"], item["val"]
        source_validation = sample_counts(source_val, counts(local_val), seed*100+fold+40000)
        arms, high_n = build_arms(item["train"], source_train, seed*100+fold)
        # 第7步和本步必须评价同一外折；数据与生成代码哈希提供样本身份复核。
        for family in MODELS:
            oldrow = prior.loc[(prior.seed == seed) & (prior.site_code == SITE) & (prior.fold == fold) & (prior.model == family) & (prior["mode"] == "source_full")]
            if len(oldrow) != 1 or int(oldrow.iloc[0].n) != len(test) or not np.isclose(float(oldrow.iloc[0].positive_rate), test.label.mean(), atol=1e-7):
                raise ValueError("外折测试人群与第7步不一致")
        for arm, train in arms.items():
            val = local_val if arm.startswith("local") else source_validation
            training_ids = set(train.sample_id)
            if training_ids & set(val.sample_id) or training_ids & set(test.sample_id) or set(val.sample_id) & set(test.sample_id):
                raise RuntimeError("训练、验证、外层测试未隔离")
            pos = train.loc[train.label == 1]
            expected_unique = 20 if arm in ("local_20", "local_repeat20", "source_20") else high_n
            if pos.sample_id.nunique() != expected_unique or (train.label == 0).sum() != 190:
                raise RuntimeError("阳性唯一人数或阴性人数错误")
            audit_rows.append({"seed": seed, "fold": fold, "arm": arm, "train_rows": len(train),
                               "positive_rows": len(pos), "unique_positive": pos.sample_id.nunique(),
                               "negative_rows": 190, "validation_n": len(val), "validation_positive": int(val.label.sum()),
                               "outer_test_n": len(test), "outer_test_positive": int(test.label.sum()), "disjoint": True})
            for family in MODELS:
                print(f"任务2 第8步：seed={seed} fold={fold} {arm}/{family}", flush=True)
                fitted = train_model(train, val, family, seed+fold)
                p = probability(fitted, family, test)
                values = metrics(test.label.to_numpy(dtype=int), p)
                fold_rows.append({"seed": seed, "fold": fold, "model": family, "arm": arm, "unique_positive": expected_unique,
                                  "positive_rows": len(pos), **values})
                folder = args.output / str(seed) / str(fold) / arm / family
                folder.mkdir(parents=True); dump(folder / "training.json", fitted[2])
            pd.DataFrame(fold_rows).to_csv(args.output / "outer_fold_metrics.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(audit_rows).to_csv(args.output / "sample_information_audits.csv", index=False, encoding="utf-8-sig")
    summary = summarize(fold_rows)
    summary.to_csv(args.output / "repeat_summary.csv", index=False, encoding="utf-8-sig")
    gains = paired_gains(summary)
    gains.to_csv(args.output / "paired_sample_information_gains.csv", index=False, encoding="utf-8-sig")
    reference.to_csv(args.output / "previous_full_source_reference.csv", index=False, encoding="utf-8-sig")
    draw(summary, gains, reference, args.output)
    metadata.update(completed=True, all_test_folds_matched=True)
    dump(args.output / "experiment_metadata.json", metadata)
    print(f"任务2 第8步完成：{args.output}")


if __name__ == "__main__":
    main()

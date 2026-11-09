"""任务2 第7步：县区内部可预测性与来源迁移的匹配样本对照。

本地训练是使用目标县区标签的监督诊断，不是域泛化方案或性能上界。
同一外折测试人群比较本地训练、等规模来源训练和完整来源训练。
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
from elderly_health.config import TrainConfig
from elderly_health.data import ENVIRONMENT, FEATURES, REGION, attach_regions, held_out_split, prepare_splits, read_raw
from elderly_health.training import fit, get_device, predict
from elderly_health.evaluation import metrics

EXPECTED_HASH = "969f0c1c82faafcfa402f37572e4a8ceb324492062cf1e590465f82dc6b5b6de"
SITES = ("110113", "371082")
SEEDS = (42, 43, 44)
MODELS = ("logistic", "shallow_tree", "mlp")
MODES = ("source_full", "source_matched", "local_matched")


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def counts(frame):
    return {label: int((frame.label == label).sum()) for label in (0, 1)}


def make_plan(frame):
    """任务2 第7步：四折外层测试与内层验证完全隔离，预先计算统一类别人数预算。"""
    plan = []
    for seed in SEEDS:
        for site in SITES:
            part = frame.loc[frame[ENVIRONMENT] == site].copy()
            outer = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
            for fold, (development, testing) in enumerate(outer.split(part, part.label), start=1):
                development = part.iloc[development]
                train, val = train_test_split(development, test_size=.2, stratify=development.label, random_state=seed+fold)
                test = part.iloc[testing]
                if set(train.sample_id) & set(val.sample_id) or (set(train.sample_id) | set(val.sample_id)) & set(test.sample_id):
                    raise RuntimeError("本地外层测试样本泄漏")
                plan.append({"seed": seed, "site": site, "fold": fold, "train": train, "val": val, "test": test})
    budget = {name: {label: min(counts(p[name])[label] for p in plan) for label in (0, 1)} for name in ("train", "val")}
    if min(budget["train"].values()) < 5 or min(budget["val"].values()) < 2:
        raise ValueError("本地类别样本过少，不能按预设设计比较")
    return plan, budget


def sample_counts(frame, budget, seed):
    """任务2 第7步：按指定阳性/阴性人数无放回抽样，不补采、不改标签。"""
    rng = np.random.default_rng(seed)
    positions = []
    for label in (0, 1):
        ids = np.flatnonzero(frame.label.to_numpy() == label)
        if len(ids) < budget[label]:
            raise ValueError("可用类别人数不足")
        positions.extend(rng.choice(ids, size=budget[label], replace=False).tolist())
    return frame.iloc[rng.permutation(positions)].copy()


def train_model(train_frame, val_frame, family, seed):
    """任务2 第7步：预处理只拟合当前训练集；超参数固定，不用外层测试选模。"""
    (train, val, _), processor = prepare_splits((train_frame, val_frame, val_frame), environment_column=REGION)
    if family == "logistic":
        model = LogisticRegression(C=1., solver="liblinear", max_iter=5000, random_state=seed)
        model.fit(train.x, train.y)
        record = {"model": family, "C": 1., "validation_selection": "none_fixed_parameters"}
    elif family == "shallow_tree":
        model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, l2_regularization=1., early_stopping=False, random_state=seed)
        model.fit(train.x, train.y)
        record = {"model": family, "max_iter": 100, "max_leaf_nodes": 7, "validation_selection": "none_fixed_parameters"}
    else:
        lowest, loss_state, loss_epoch = float("inf"), None, None

        def capture(model, rec):
            nonlocal lowest, loss_state, loss_epoch
            if rec["val_loss"] < lowest:
                lowest = rec["val_loss"]
                loss_epoch = rec["epoch"]
                loss_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        cfg = TrainConfig(model="baseline", environment_granularity="region", seed=seed, selection_start_epoch=1, device="cpu")
        model, history = fit(train, val, cfg, epoch_callback=capture)
        model.load_state_dict(loss_state)
        record = {"model": family, "selected_loss_epoch": loss_epoch, "config": cfg.to_dict(), "training": history,
                  "validation_selection": "minimum_weighted_bce_in_original_source_auc_early_stop_trajectory"}
    return model, processor, record


def probability(fitted, family, test_frame):
    model, processor, _ = fitted
    x = processor.transform(test_frame)
    return predict(model, x, get_device("cpu")) if family == "mlp" else model.predict_proba(x)[:, list(model.classes_).index(1)]


def delta_table(summary):
    """任务2 第7步：同县区、同重复、同模型配对；不把重叠训练折当独立实验。"""
    keys = ["seed", "site_code", "model"]
    measures = ["mean_fold_auc", "mean_fold_auprc", "oof_auc", "oof_auprc"]
    local = summary.loc[summary["mode"] == "local_matched"]
    result = []
    for reference in ("source_matched", "source_full"):
        old = summary.loc[summary["mode"] == reference]
        joined = local.merge(old[keys+measures], on=keys, suffixes=("", "_reference"), validate="one_to_one")
        for row in joined.to_dict("records"):
            result.append({**{k: row[k] for k in keys}, "contrast": f"local_matched_minus_{reference}",
                           **{f"delta_{k}": row[k]-row[k+"_reference"] for k in measures}})
    return pd.DataFrame(result)


def draw(summary, deltas, output):
    """任务2 第7步：分别展示县区内表现和本地相对来源的差距，保留各重复。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = output / "figures"
    folder.mkdir()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    colors = {"source_full": "tab:gray", "source_matched": "tab:orange", "local_matched": "tab:blue"}
    for ax, site in zip(axes, SITES):
        for offset, mode in zip((-.2, 0, .2), MODES):
            for i, family in enumerate(MODELS):
                g = summary.loc[(summary.site_code == site) & (summary.model == family) & (summary["mode"] == mode)]
                ax.scatter(np.full(len(g), i+offset), g.mean_fold_auc, color=colors[mode], label=mode if i == 0 else None)
        ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS)
        ax.axhline(.5, color="black", linestyle="--", linewidth=.8)
        ax.set_title(f"County {site}"); ax.set_ylabel("Mean outer-fold AUC; each dot is one repeat")
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(folder / "01_local_vs_source_auc.png", dpi=190); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, site in zip(axes, SITES):
        for offset, contrast in zip((-.1, .1), ("local_matched_minus_source_matched", "local_matched_minus_source_full")):
            for i, family in enumerate(MODELS):
                g = deltas.loc[(deltas.site_code == site) & (deltas.model == family) & (deltas.contrast == contrast)]
                color = "tab:orange" if contrast.endswith("source_matched") else "tab:gray"
                ax.scatter(np.full(len(g), i+offset), g.delta_mean_fold_auc, color=color, label=contrast if i == 0 else None)
        ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS)
        ax.axhline(0, color="black", linewidth=.8); ax.set_title(f"County {site}")
        ax.set_ylabel("Local minus source: paired mean-fold AUC"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(folder / "02_local_training_advantage.png", dpi=190); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data/data_mission2.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/analysis_2611/task2_step7_east_local_learnability")
    args = parser.parse_args()
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != EXPECTED_HASH:
        raise ValueError("固定CSV版本不一致")
    frame = attach_regions(read_raw(args.data))
    plan, budget = make_plan(frame)
    source_train, source_val, _ = held_out_split(frame, "east", seed=42, column=REGION)
    if set(source_train[ENVIRONMENT]) & set(SITES) or set(source_val[ENVIRONMENT]) & set(SITES):
        raise RuntimeError("来源训练/验证包含东部县区")
    args.output.mkdir(parents=True, exist_ok=False)
    meta = {"completed": False, "input_sha256": EXPECTED_HASH, "seeds": SEEDS, "sites": SITES,
            "outer_folds": 4, "inner_validation_fraction": .2, "matched_class_counts": budget,
            "source_split_seed": 42, "models": MODELS,
            "source_full_class_counts": {"train": counts(source_train), "val": counts(source_val)},
            "target_labels_used_for_local_supervised_diagnostic": True,
            "outer_test_used_for_fitting_or_selection": False,
            "python": platform.python_version(), "torch": torch.__version__, "sklearn": sklearn.__version__,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "limitations": "本地有标签诊断不等于域泛化或可达性能上界；本地未明显胜出不能证明不存在预测信号。重复外折训练重叠，折和种子不能当独立样本计算显著性。匹配改变训练阳性率，Brier仅辅助。"}
    dump(args.output / "experiment_metadata.json", meta)
    fold_rows, summary_rows, audits = [], [], []
    for seed in SEEDS:
        # 完整来源模型每种子/模型只训练一次，所有外折复用，测试集始终不参与训练。
        full_models, matched_models = {}, {}
        for family in MODELS:
            print(f"任务2 第7步：seed={seed} 完整来源 {family}", flush=True)
            fitted = train_model(source_train, source_val, family, seed)
            full_models[family] = fitted
            folder = args.output / str(seed) / "source_full" / family
            folder.mkdir(parents=True); dump(folder / "training.json", fitted[2])
        for site in SITES:
            county = frame.loc[frame[ENVIRONMENT] == site]
            # 仅内存保存折外预测，最终不导出逐人记录。
            predictions = {(family, mode): {} for family in MODELS for mode in MODES}
            for item in [p for p in plan if p["seed"] == seed and p["site"] == site]:
                fold, test = item["fold"], item["test"]
                local_train = sample_counts(item["train"], budget["train"], seed*100+fold)
                local_val = sample_counts(item["val"], budget["val"], seed*100+fold+10000)
                matched_train = sample_counts(source_train, budget["train"], seed*100+fold+20000)
                matched_val = sample_counts(source_val, budget["val"], seed*100+fold+30000)
                # 相同seed/fold两县区共用同一批来源抽样和训练种子。
                modes = {"source_matched": (matched_train, matched_val), "local_matched": (local_train, local_val)}
                for mode, (tr, va) in modes.items():
                    passed = not (set(tr.sample_id) & set(va.sample_id) or (set(tr.sample_id) | set(va.sample_id)) & set(test.sample_id))
                    if not passed or counts(tr) != budget["train"] or counts(va) != budget["val"]:
                        raise RuntimeError("样本隔离或匹配人数检查失败")
                    audits.append({"seed": seed, "site_code": site, "fold": fold, "mode": mode,
                                   "train_n": len(tr), "train_positive": counts(tr)[1], "val_n": len(va), "val_positive": counts(va)[1],
                                   "test_n": len(test), "test_positive": counts(test)[1], "disjoint": passed})
                for family in MODELS:
                    for mode in MODES:
                        print(f"任务2 第7步：seed={seed} site={site} fold={fold} {family}/{mode}", flush=True)
                        if mode == "source_full":
                            fitted = full_models[family]
                        elif mode == "source_matched":
                            if (fold, family) not in matched_models:
                                fitted = train_model(*modes[mode], family, seed+fold)
                                matched_models[fold, family] = fitted
                                folder = args.output / str(seed) / mode / str(fold) / family
                                folder.mkdir(parents=True); dump(folder / "training.json", fitted[2])
                            fitted = matched_models[fold, family]
                        else:
                            fitted = train_model(*modes[mode], family, seed+fold)
                            folder = args.output / str(seed) / site / str(fold) / family / mode
                            folder.mkdir(parents=True); dump(folder / "training.json", fitted[2])
                        p = probability(fitted, family, test)
                        score = metrics(test.label.to_numpy(dtype=int), p)
                        fold_rows.append({"seed": seed, "site_code": site, "fold": fold, "model": family, "mode": mode, **score})
                        dest = predictions[family, mode]
                        for sample, prob in zip(test.sample_id, p):
                            if int(sample) in dest:
                                raise RuntimeError("同一县区重复测试样本")
                            dest[int(sample)] = float(prob)
                pd.DataFrame(fold_rows).to_csv(args.output / "outer_fold_metrics.csv", index=False, encoding="utf-8-sig")
                pd.DataFrame(audits).to_csv(args.output / "sample_matching_audits.csv", index=False, encoding="utf-8-sig")
            for (family, mode), values in predictions.items():
                if set(values) != set(county.sample_id):
                    raise RuntimeError("折外预测未完整覆盖县区，每名样本须且仅测试一次")
                p = np.array([values[int(i)] for i in county.sample_id])
                oof = metrics(county.label.to_numpy(dtype=int), p)
                folds = pd.DataFrame([r for r in fold_rows if r["seed"] == seed and r["site_code"] == site and r["model"] == family and r["mode"] == mode])
                summary_rows.append({"seed": seed, "site_code": site, "model": family, "mode": mode,
                                     "mean_fold_auc": float(folds.auc.mean()), "mean_fold_auprc": float(folds.auprc.mean()),
                                     "fold_auc_sd": float(folds.auc.std(ddof=1)),
                                     **{f"oof_{k}": oof[k] for k in ("auc", "auprc", "brier", "n", "positive_rate")}})
            pd.DataFrame(summary_rows).to_csv(args.output / "repeat_summary.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(summary_rows)
    deltas = delta_table(summary)
    deltas.to_csv(args.output / "paired_local_advantage.csv", index=False, encoding="utf-8-sig")
    draw(summary, deltas, args.output)
    meta.update(completed=True, oof_complete=True)
    dump(args.output / "experiment_metadata.json", meta)
    print(f"任务2 第7步完成：{args.output}")


if __name__ == "__main__":
    main()

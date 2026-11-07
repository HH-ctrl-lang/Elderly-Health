"""任务三 第3步：原始特征与固定V5表征的源域线性探针对照。

仅分析留出中部地区折；历史编码器冻结，不训练或微调V5。
四种输入采用相同训练/验证/测试样本，使用同类逻辑回归。
目标标签不参与拟合、预处理或参数选择；仅输出聚合指标。
探针衡量可迁移的线性可读性，不是信息量或因果效应检验。
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

from elderly_health.data import (  # noqa: E402
    ENVIRONMENT, REGION, FEATURES, FeaturePreprocessor, attach_regions, read_raw,
)
from task3_step2_v5_representation_diagnostics import (  # noqa: E402
    MODEL_ROOT, EXPECTED_DATA_SHA256, latent_columns, load_fold,
    safe_prediction_metrics, sha256_file,
)

OUTPUT = ROOT / "outputs" / "analysis_2611" / "task3_step3_fixed_representation_probes"
SPACES = ("raw19", "zc", "zs", "zc_zs")
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
SEED = 26110303
FOCUS = ("220202", "360102")


def align_raw(frame, representation):
    """任务三 第3步：按历史sample_id对齐，逐项验证标签、地区与监测点。"""
    ids = representation["sample_id"].to_numpy()
    indexed = frame.set_index("sample_id", drop=False)
    if not np.isin(ids, indexed.index).all():
        raise ValueError("历史表征含输入数据中不存在的样本。")
    aligned = indexed.loc[ids].reset_index(drop=True)
    for left, right in (
        (aligned["label"].to_numpy(dtype=int), representation["cognitive_impairment"].to_numpy(dtype=int)),
        (aligned[ENVIRONMENT].to_numpy(dtype=str), representation["environment_code"].to_numpy(dtype=str)),
        (aligned[REGION].to_numpy(dtype=str), representation["environment_group"].to_numpy(dtype=str)),
    ):
        if not np.array_equal(left, right):
            raise ValueError("历史表征与模型输入的标签、地区或监测点不一致。")
    return aligned


def build_inputs(frame, fold):
    """任务三 第3步：原始输入仅在训练集拟合缺失填补，沿用历史数值编码。"""
    parts = (fold["train"], fold["validation"], fold["test"])
    raw = [align_raw(frame, part) for part in parts]
    observed_ids = set().union(*(set(part["sample_id"]) for part in parts))
    if observed_ids != set(frame["sample_id"]):
        raise ValueError("本折未完整覆盖研究样本。")
    prep = FeaturePreprocessor().fit(raw[0])
    inputs = {"raw19": [prep.transform(part) for part in raw]}
    for branch in ("zc", "zs"):
        inputs[branch] = [part[latent_columns(branch)].to_numpy(dtype=float) for part in parts]
    inputs["zc_zs"] = [np.concatenate([a, b], axis=1) for a, b in zip(inputs["zc"], inputs["zs"])]
    return inputs, parts


def fit_probe(inputs, labels, space):
    """任务三 第3步：仅用源域验证AUC选C；同分时保留更强正则化。"""
    rows, best_model, best_score, best_c = [], None, -np.inf, None
    for c in C_GRID:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, solver="lbfgs", max_iter=5000,
                               class_weight=None, random_state=SEED),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(inputs[0], labels[0])
        val_p = model.predict_proba(inputs[1])[:, 1]
        score = roc_auc_score(labels[1], val_p)
        rows.append({"space": space, "C": c, "selection_metric": "source_validation_auc",
                     **safe_prediction_metrics(labels[1], val_p)})
        if score > best_score + 1e-12:
            best_model, best_score, best_c = model, score, c
    for row in rows:
        row["selected"] = row["C"] == best_c
    # 任务三 第3步：选择后不合并验证集重拟合，保持四种输入训练样本一致。
    predictions = [best_model.predict_proba(x)[:, 1] for x in inputs]
    return predictions, rows, best_c


def bootstrap_deltas(labels, probabilities, repeats):
    """任务三 第3步：同一站点各探针共用分层重抽样，估计相对raw19的差值区间。"""
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    if min(map(len, groups)) < 2:
        raise ValueError("配对bootstrap至少需要两个阳性和两个阴性。")
    metrics = {"auc": roc_auc_score, "auprc": average_precision_score}
    draws = {(space, metric): [] for space in SPACES[1:] for metric in metrics}
    for _ in range(repeats):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        for metric, func in metrics.items():
            raw_score = func(labels[idx], probabilities["raw19"][idx])
            for space in SPACES[1:]:
                draws[(space, metric)].append(func(labels[idx], probabilities[space][idx]) - raw_score)
    rows = []
    for (space, metric), values in draws.items():
        func = metrics[metric]
        rows.append({"space": space, "reference": "raw19", "metric": metric,
                     "difference": float(func(labels, probabilities[space]) - func(labels, probabilities["raw19"])),
                     "ci_lower": float(np.quantile(values, 0.025)),
                     "ci_upper": float(np.quantile(values, 0.975)), "bootstrap_repeats": repeats})
    return rows


def draw_figures(metrics, deltas, output):
    """任务三 第3步：绘制目标站点指标与配对差值区间，图中文字统一英文。"""
    figures = output / "figures"
    figures.mkdir()
    target = metrics.loc[metrics["partition"] == "target_test_site"]
    for metric in ("auc", "auprc"):
        pivot = target.pivot(index="site_code", columns="space", values=metric).reindex(columns=SPACES)
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot.plot.bar(ax=ax, width=0.8)
        if metric == "auc":
            ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
        else:
            rates = target.drop_duplicates("site_code").set_index("site_code")["positive_rate"]
            ax.scatter(np.arange(len(pivot)), rates.reindex(pivot.index), color="black", marker="_", s=150, label="Site prevalence")
        ax.set_ylim(0, 1)
        ax.set_ylabel("ROC-AUC" if metric == "auc" else "Average precision")
        ax.set_xlabel("Held-out central-region site")
        ax.set_title("Source-trained logistic probes on fixed inputs")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / f"target_probe_{metric}.png", dpi=220)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, site in zip(axes, FOCUS):
        part = deltas.loc[(deltas["site_code"] == site) & (deltas["metric"] == "auc")].set_index("space").reindex(SPACES[1:])
        positions = np.arange(len(part))
        ax.vlines(positions, part["ci_lower"], part["ci_upper"], color="tab:blue")
        ax.scatter(positions, part["difference"], color="tab:blue")
        ax.axhline(0, color="grey", linestyle="--")
        ax.set_xticks(positions)
        ax.set_xticklabels(part.index)
        ax.set_title(f"Site {site}: latent minus raw19")
        ax.set_ylabel("ROC-AUC difference (paired 95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(figures / "matched_site_probe_auc_differences.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data" / "data_mission2.csv")
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bootstraps", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"输出目录已存在，禁止覆盖：{args.output}")
    if args.bootstraps < 200:
        raise ValueError("至少使用200次配对bootstrap。")
    if sha256_file(args.data) != EXPECTED_DATA_SHA256:
        raise ValueError("数据哈希与历史V5模型不一致。")
    frame = attach_regions(read_raw(args.data, encoding="gbk"))
    if len(frame) != 4769 or frame[ENVIRONMENT].nunique() != 12:
        raise ValueError("研究样本数或监测点数异常。")
    fold = load_fold("central", args.model_root)
    if tuple(fold["config"]["feature_names"]) != tuple(FEATURES):
        raise ValueError("历史模型特征顺序与当前读取器不一致。")
    inputs, parts = build_inputs(frame, fold)
    labels = [part["cognitive_impairment"].to_numpy(dtype=int) for part in parts]
    for site in FOCUS:
        y = labels[2][parts[2]["environment_code"].eq(site).to_numpy()]
        if len(y) != 400 or y.sum() != 17:
            raise ValueError(f"配对案例{site}的样本数或阳性数异常。")
    metrics, selection, predictions, chosen = [], [], {}, {}
    for space in SPACES:
        print(f"任务三 第3步：拟合源域线性探针 {space}", flush=True)
        p, rows, c = fit_probe(inputs[space], labels, space)
        predictions[space], chosen[space] = p, c
        selection.extend(rows)
        for index, partition in enumerate(("source_train_pool", "source_validation_pool", "target_test_pool")):
            metrics.append({"space": space, "partition": partition, "site_code": "POOL", "selected_C": c,
                            **safe_prediction_metrics(labels[index], p[index])})
            if index == 0:
                continue
            for site in sorted(parts[index]["environment_code"].unique()):
                mask = parts[index]["environment_code"].eq(site).to_numpy()
                metrics.append({"space": space,
                                "partition": "source_validation_site" if index == 1 else "target_test_site",
                                "site_code": str(site), "selected_C": c,
                                **safe_prediction_metrics(labels[index][mask], p[index][mask])})
    differences = []
    for site in FOCUS:
        mask = parts[2]["environment_code"].eq(site).to_numpy()
        rows = bootstrap_deltas(labels[2][mask], {s: predictions[s][2][mask] for s in SPACES}, args.bootstraps)
        differences.extend({"site_code": site, **row} for row in rows)
    metrics, differences = pd.DataFrame(metrics), pd.DataFrame(differences)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, table in (("probe_metrics.csv", metrics), ("source_validation_parameter_selection.csv", pd.DataFrame(selection)),
                        ("matched_site_probe_differences.csv", differences)):
        table.to_csv(args.output / name, index=False, encoding="utf-8-sig")
    draw_figures(metrics, differences, args.output)
    metadata = {
        "task": "Task 3 Step 3 - Fixed representation linear probes", "held_out_region": "central",
        "data_sha256": sha256_file(args.data), "encoder_retrained": False, "new_logistic_probes_fitted": True,
        "spaces": {space: int(inputs[space][0].shape[1]) for space in SPACES},
        "feature_names": list(FEATURES), "C_grid": list(C_GRID), "selected_C": chosen,
        "seed": SEED, "bootstrap_repeats": args.bootstraps,
        "split_counts": dict(zip(("train", "validation", "test"), map(len, parts))),
        "preprocessing": "Train-only median imputation for raw numeric codes; train-only standardization for all probe inputs.",
        "selection": "Source validation ROC-AUC only; ties prefer smaller C; no refit on validation.",
        "class_weight": None, "individual_records_exported": False,
        "historical_artifact_hashes": fold["artifacts"],
        "limitations": [
            "Linear readability and source-to-target transfer are tested, not total information or causality.",
            "Poor target performance may reflect conditional shift, finite samples, regularization or nonlinear structure.",
            "Historical encoder already used source validation for checkpoint selection; this is exploratory reuse.",
            "Numeric categorical codes are retained to match historical inputs; no new encoding comparison is introduced.",
            "Spaces differ in dimension and representation; equal model family and selection rule do not make capacities identical.",
            "Fixed-fit paired stratified bootstrap excludes training uncertainty; intervals are exploratory and unadjusted for multiplicity.",
            "Only 17 positives in each focus site; focus sites were chosen after earlier results were inspected.",
            "New unweighted logistic probes are not controlled replacements for the historical weighted neural prediction heads.",
            "Target labels are never used to fit probes or choose hyperparameters.",
        ],
    }
    (args.output / "experiment_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("任务三 第3步完成：V5保持冻结，仅拟合源域逻辑回归探针。")
    print(f"结果目录：{args.output}")
    print(metrics.loc[metrics["partition"] == "target_test_site", ["site_code", "space", "auc", "auprc"]].to_string(index=False))


if __name__ == "__main__":
    main()

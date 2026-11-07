"""任务三 第5步：拆分第3步raw19逻辑回归的打分依据。

用原训练样本和原参数重建raw19探针，核对第3步指标后冻结。
每个特征贡献=标准化输入乘系数；组内阳性减阴性的贡献可精确相加。
贡献之和解释平均原始分数差，不等于ROC-AUC的分解或因果效应。
不修改V5，不调参，不删除特征，不导出逐人分数。
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
from elderly_health.data import ENVIRONMENT, FEATURES, attach_regions, read_raw  # noqa: E402
from task3_step2_v5_representation_diagnostics import (  # noqa: E402
    MODEL_ROOT, EXPECTED_DATA_SHA256, load_fold, safe_prediction_metrics, sha256_file,
)
from task3_step3_fixed_representation_probes import build_inputs, align_raw, SEED as PROBE_SEED  # noqa: E402

PREVIOUS = ROOT / "outputs" / "analysis_2611" / "task3_step3_fixed_representation_probes"
OUTPUT = ROOT / "outputs" / "analysis_2611" / "task3_step5_linear_score_contributions"
FOCUS = ("220202", "360102")
GROUP_ORDER = ("SOURCE_TRAIN", "SOURCE_VALIDATION", "220202", "360102")
MOBILE = ("if_use_mobile_phone", "using_phone_year", "hours_phone_per_day")
SEED = 26110305


def rebuild_probe(inputs, labels, c):
    """任务三 第5步：只在原训练集重建已选定参数的raw19探针，不重新选择参数。"""
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=c, solver="lbfgs", max_iter=5000,
                                             class_weight=None, random_state=PROBE_SEED),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(inputs[0], labels[0])
    return model


def verify_previous(model, inputs, parts, reference, selected_c):
    """任务三 第5步：核对上一实验的所有raw19组指标与C；不一致时停止输出。"""
    checks = []
    raw_reference = reference.loc[reference["space"].eq("raw19")]
    expected_keys = {(p, "POOL") for p in ("source_train_pool", "source_validation_pool", "target_test_pool")}
    expected_keys.update(("source_validation_site", str(s)) for s in parts[1]["environment_code"].unique())
    expected_keys.update(("target_test_site", str(s)) for s in parts[2]["environment_code"].unique())
    observed_keys = set(zip(raw_reference["partition"], raw_reference["site_code"].astype(str)))
    if observed_keys != expected_keys or len(raw_reference) != len(expected_keys):
        raise ValueError("第3步raw19指标记录缺失或重复。")
    for row in raw_reference.to_dict("records"):
        if not np.isclose(float(row["selected_C"]), selected_c, atol=0, rtol=0):
            raise ValueError("第3步参数表与元数据中的C不一致。")
        partition = row["partition"]
        index = 0 if partition.startswith("source_train") else (1 if partition.startswith("source_validation") else 2)
        mask = np.ones(len(parts[index]), dtype=bool)
        if str(row["site_code"]) != "POOL":
            mask = parts[index]["environment_code"].eq(str(row["site_code"])).to_numpy()
        if int(mask.sum()) != int(row["n"]):
            raise ValueError("第3步样本数与重建划分不一致。")
        y = parts[index].loc[mask, "cognitive_impairment"].to_numpy(dtype=int)
        observed = safe_prediction_metrics(y, model.predict_proba(inputs[index][mask])[:, 1])
        for metric in ("positive_count", "auc", "auprc", "brier", "log_loss", "mean_probability"):
            old, new = float(row[metric]), float(observed[metric])
            passed = bool(np.isclose(old, new, atol=1e-7, rtol=0, equal_nan=True))
            checks.append({"partition": partition, "site_code": str(row["site_code"]), "metric": metric,
                           "previous_value": old, "reconstructed_value": new, "passed": passed})
            if not passed:
                raise RuntimeError(f"raw19复现未通过：{partition}/{row['site_code']}/{metric}。")
    if len(checks) != len(expected_keys) * 6:
        raise ValueError("第3步raw19指标记录数量不完整。")
    return pd.DataFrame(checks)


def decompose_score(standardized, coefficients, intercept, expected_score):
    """任务三 第5步：验证每个人的特征贡献加截距等于模型原始分数。"""
    contributions = standardized * coefficients
    reconstructed = contributions.sum(axis=1) + intercept
    error = float(np.max(np.abs(reconstructed - expected_score)))
    if not np.allclose(reconstructed, expected_score, atol=1e-10, rtol=1e-10):
        raise RuntimeError("逐人分数加法核对失败。")
    return contributions, error


def group_summary(contributions, labels, missing, name, repeats):
    """任务三 第5步：分层抽样时保留整行19个特征，使贡献间的相关关系保持。"""
    labels = np.asarray(labels, dtype=int)
    positive, negative = contributions[labels == 1], contributions[labels == 0]
    if min(len(positive), len(negative)) < 2:
        raise ValueError(f"{name}的阳性或阴性人数不足。")
    gap = positive.mean(axis=0) - negative.mean(axis=0)
    rng = np.random.default_rng(SEED + GROUP_ORDER.index(name))
    draws = np.empty((repeats, len(FEATURES)))
    for i in range(repeats):
        draws[i] = (positive[rng.integers(len(positive), size=len(positive))].mean(axis=0)
                    - negative[rng.integers(len(negative), size=len(negative))].mean(axis=0))
    rows = []
    for j, feature in enumerate(FEATURES):
        rows.append({"group": name, "feature": feature, "n": len(labels), "positive_count": len(positive),
                     "positive_mean_contribution": float(positive[:, j].mean()),
                     "negative_mean_contribution": float(negative[:, j].mean()),
                     "contribution_gap": float(gap[j]),
                     "ci_lower": float(np.quantile(draws[:, j], 0.025)),
                     "ci_upper": float(np.quantile(draws[:, j], 0.975)),
                     "positive_imputed_count": int(missing[labels == 1, j].sum()),
                     "negative_imputed_count": int(missing[labels == 0, j].sum())})
    checks, blocks = [], []
    blocks_def = {"phone_three_features": [FEATURES.index(f) for f in MOBILE],
                  "other_sixteen_features": [i for i, f in enumerate(FEATURES) if f not in MOBILE],
                  "all_nineteen_features": list(range(len(FEATURES)))}
    for block, indices in blocks_def.items():
        values = draws[:, indices].sum(axis=1)
        blocks.append({"group": name, "block": block, "contribution_gap": float(gap[indices].sum()),
                       "ci_lower": float(np.quantile(values, 0.025)), "ci_upper": float(np.quantile(values, 0.975))})
    score_gap = float(positive.sum(axis=1).mean() - negative.sum(axis=1).mean())
    checks.append({"group": name, "sum_feature_gaps": float(gap.sum()), "mean_raw_score_gap": score_gap,
                   "absolute_difference": abs(float(gap.sum()) - score_gap)})
    if not np.isclose(gap.sum(), score_gap, atol=1e-10, rtol=1e-10):
        raise RuntimeError("组内平均分差加法核对失败。")
    return rows, blocks, checks


def draw_figures(weights, contributions, blocks, output):
    """任务三 第5步：绘制权重、分数贡献和手机变量整体贡献。"""
    figures = output / "figures"
    figures.mkdir()
    colors = ["tab:red" if v > 0 else "tab:blue" for v in weights["coefficient"]]
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.barh(weights["feature"], weights["coefficient"], color=colors)
    ax.axvline(0, color="grey", linewidth=1)
    ax.invert_yaxis()
    ax.set_xlabel("Logistic coefficient per standardized input unit")
    ax.set_title("raw19 probe: fitted weights (not causal importance)")
    fig.tight_layout()
    fig.savefig(figures / "raw19_feature_weights.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 10), sharey=True)
    for ax, site in zip(axes, FOCUS):
        table = contributions.loc[contributions["group"] == site].set_index("feature").reindex(FEATURES)
        y = np.arange(len(FEATURES))
        ax.hlines(y, table["ci_lower"], table["ci_upper"], color="grey")
        ax.scatter(table["contribution_gap"], y, c=["tab:red" if v < 0 else "tab:blue" for v in table["contribution_gap"]])
        ax.axvline(0, color="grey", linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels(FEATURES)
        ax.set_xlabel("Contribution to mean raw-score gap: positive minus negative")
        ax.set_title(f"{site}: right helps mean separation; left opposes it")
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(figures / "matched_site_feature_score_gaps.png", dpi=220)
    plt.close(fig)

    names = ("phone_three_features", "other_sixteen_features", "all_nineteen_features")
    fig, ax = plt.subplots(figsize=(11, 5))
    for j, block in enumerate(names):
        table = blocks.loc[blocks["block"] == block].set_index("group").reindex(GROUP_ORDER)
        x = np.arange(len(GROUP_ORDER)) + (j-1)*0.23
        ax.bar(x, table["contribution_gap"], width=0.23, label=block, alpha=0.8)
        ax.vlines(x, table["ci_lower"], table["ci_upper"], color="black", linewidth=0.8)
    ax.axhline(0, color="grey")
    ax.set_xticks(np.arange(len(GROUP_ORDER)))
    ax.set_xticklabels(GROUP_ORDER)
    ax.set_ylabel("Mean raw-score gap (positive minus negative)")
    ax.set_title("How phone-related and other features combine in the fixed probe")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "phone_block_and_total_score_gaps.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data" / "data_mission2.csv")
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--previous-output", type=Path, default=PREVIOUS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bootstraps", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"输出目录已存在，禁止覆盖：{args.output}")
    if args.bootstraps < 200:
        raise ValueError("至少使用200次bootstrap。")
    if sha256_file(args.data) != EXPECTED_DATA_SHA256:
        raise ValueError("输入数据哈希与历史模型不一致。")
    previous_meta_path = args.previous_output / "experiment_metadata.json"
    previous_metrics_path = args.previous_output / "probe_metrics.csv"
    previous = json.loads(previous_meta_path.read_text(encoding="utf-8"))
    if previous["data_sha256"] != EXPECTED_DATA_SHA256 or previous["held_out_region"] != "central":
        raise ValueError("第3步元数据不对应当前研究。")
    reference = pd.read_csv(previous_metrics_path, dtype={"site_code": str}, encoding="utf-8-sig")
    fold = load_fold("central", args.model_root)
    if previous["historical_artifact_hashes"] != fold["artifacts"]:
        raise ValueError("第3步以来历史模型或表征文件发生变化。")
    if tuple(previous["feature_names"]) != tuple(FEATURES) or tuple(fold["config"]["feature_names"]) != tuple(FEATURES):
        raise ValueError("特征顺序与历史实验不一致。")
    frame = attach_regions(read_raw(args.data, encoding="gbk"))
    if len(frame) != 4769 or frame[ENVIRONMENT].nunique() != 12:
        raise ValueError("研究样本数或站点数异常。")
    inputs, parts = build_inputs(frame, fold)
    raw_inputs = inputs["raw19"]
    labels = [part["cognitive_impairment"].to_numpy(dtype=int) for part in parts]
    print("任务三 第5步：按第3步参数重建raw19简单模型，并核对原结果。", flush=True)
    selected_c = float(previous["selected_C"]["raw19"])
    model = rebuild_probe(raw_inputs, labels, selected_c)
    reproduction = verify_previous(model, raw_inputs, parts, reference, selected_c)
    scaler, classifier = model.steps[0][1], model.steps[1][1]
    coefficients = classifier.coef_[0]
    intercept = float(classifier.intercept_[0])
    weights = pd.DataFrame({"feature": FEATURES, "coefficient": coefficients,
                            "absolute_coefficient": np.abs(coefficients)})
    rows, block_rows, score_checks, decomposition_errors = [], [], [], []
    for i, part in enumerate(parts):
        raw = align_raw(frame, part)
        missing = raw[list(FEATURES)].isna().to_numpy()
        values, error = decompose_score(scaler.transform(raw_inputs[i]), coefficients, intercept, model.decision_function(raw_inputs[i]))
        decomposition_errors.append(error)
        groups = [("SOURCE_TRAIN" if i == 0 else "SOURCE_VALIDATION", np.ones(len(part), dtype=bool))] if i < 2 else [(s, part["environment_code"].eq(s).to_numpy()) for s in FOCUS]
        for name, mask in groups:
            if name in FOCUS and (int(mask.sum()) != 400 or int(labels[i][mask].sum()) != 17):
                raise ValueError("配对案例样本数或阳性数异常。")
            print(f"任务三 第5步：拆分 {name} 的平均打分差。", flush=True)
            r, b, c = group_summary(values[mask], labels[i][mask], missing[mask], name, args.bootstraps)
            rows.extend(r); block_rows.extend(b); score_checks.extend(c)
    contributions, blocks = pd.DataFrame(rows), pd.DataFrame(block_rows)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, table in (("raw19_feature_weights.csv", weights), ("feature_score_gap_contributions.csv", contributions),
                        ("phone_block_score_gap_contributions.csv", blocks), ("score_additivity_checks.csv", pd.DataFrame(score_checks)),
                        ("step3_reproduction_checks.csv", reproduction)):
        table.to_csv(args.output / name, index=False, encoding="utf-8-sig")
    draw_figures(weights, contributions, blocks, args.output)
    metadata = {
        "task": "Task 3 Step 5 - raw19 linear score contributions", "held_out_region": "central",
        "data_sha256": sha256_file(args.data), "C": selected_c, "seed": SEED, "probe_seed": PROBE_SEED,
        "raw19_probe_reconstructed_once": True, "hyperparameters_reselected": False, "V5_retrained": False,
        "source_train_only_fit": True, "individual_records_exported": False, "bootstraps": args.bootstraps,
        "score_scale": "logistic decision_function (log odds), not probability or ROC-AUC",
        "formula": "score = intercept + sum_j standardized_input_j * coefficient_j",
        "contribution_gap": "mean contribution in positive group minus mean contribution in negative group",
        "maximum_individual_additivity_error": max(decomposition_errors), "intercept": intercept,
        "phone_block_features": list(MOBILE), "reference_artifacts": fold["artifacts"],
        "previous_metadata_sha256": sha256_file(previous_meta_path), "previous_metrics_sha256": sha256_file(previous_metrics_path),
        "limitations": [
            "Explains only the reconstructed step3 raw19 logistic probe, not V5's neural prediction heads.",
            "Positive mean contribution gap helps average score separation but need not improve ROC-AUC; ROC-AUC is not additively decomposed.",
            "Coefficients and contribution gaps are model descriptions, not causal effects or validated feature importance.",
            "Correlated features share fitted weights; phone block is summed jointly and is not three independent findings.",
            "Missing features use historical train-median imputation; this differs from step4 complete-case association summaries.",
            "Bootstrap resamples positive and negative rows, preserving dependence across all features; coefficients and preprocessing stay fixed.",
            "Intervals are exploratory and unadjusted for multiple comparisons; training, site-population and label uncertainty are excluded.",
            "Numeric categories retain historical coding; clinical meanings are not inferred from coefficient signs.",
            "No feature-removal experiment, counterfactual intervention, new tuning or target-label-based fitting is performed.",
        ],
    }
    (args.output / "experiment_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"任务三 第5步完成；结果目录：{args.output}")


if __name__ == "__main__":
    main()

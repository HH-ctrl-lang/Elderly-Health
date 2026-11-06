"""任务三 第一步：异常监测点的预测失败诊断（不训练模型）。

研究问题：
1. 相同低阳性率的监测点，为什么不同算法的风险排序差异巨大？
2. RDM、VREx、GroupDRO、MLP、V5 是否在相同监测点一致失败？
3. 各监测点的结局条件特征分布差异，是否提供了进一步检查的线索？

所有结论仅为探索性诊断；类别分布与标签之间的联系不是因果效应。
仅保存聚合统计，不导出个人编号、逐人特征或逐人预测。
"""

import argparse
import hashlib
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

from elderly_health.data import (  # noqa: E402
    ENVIRONMENT,
    FEATURES,
    REGION,
    attach_regions,
    read_raw,
)

DATA_PATH = PROJECT / "data" / "data_mission2.csv"
OUTPUT_DIR = ROOT / "outputs" / "analysis_2611" / "task3_step1_site_failure_diagnostics"
EXPECTED_DATA_SHA256 = (
    "969f0c1c82faafcfa402f37572e4a8ceb324492062cf1e590465f82dc6b5b6de"
)
REGIONS = ("east", "central", "west")
MODEL_ROOTS = {
    "MLP": ROOT / "outputs" / "v2-region-loro-selection91-42",
    "V5": ROOT / "outputs" / "v2-region-loro-selection91-42",
    "RDM": ROOT / "outputs" / "external-rdm-region-loro-42",
    "VREx": ROOT / "outputs" / "external-vrex-region-loro-42",
    "GroupDRO": ROOT / "outputs" / "external-groupdro-region-loro-42",
}
MODELS = tuple(MODEL_ROOTS)
MATCHED_PAIR = ("220202", "360102")
OTHER_LOW_PREVALENCE_SITES = ("220421", "640106")
SEED = 26110301
PERMUTATIONS = 300
BOOTSTRAPS = 1000


# ==================== 一、基础校验与指标 ====================

def file_sha256(path):
    """任务三 第一步：计算文件版本哈希。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def source_predictions_path(region, model):
    """任务三 第一步：按模型原目录读取测试集预测，不用重建模型。"""
    root = MODEL_ROOTS[model]
    if model == "MLP":
        return root / region / "baseline" / "representations_test.csv"
    if model == "V5":
        return root / region / "v5" / "representations_test.csv"
    return root / region / "predictions_test.csv"


def load_and_verify_predictions(frame):
    """任务三 第一步：逐sample_id检查五种算法预测的标签和站点完全一致。"""
    expected = frame.set_index("sample_id")
    all_data = {}
    manifest = []

    for model in MODELS:
        parts = []
        for region in REGIONS:
            path = source_predictions_path(region, model)
            if not path.is_file():
                raise FileNotFoundError(f"预测文件不存在：{path}")

            rename = {"environment_code": "site_code"}
            site_column = "environment_code" if model in ("MLP", "V5") else "site_code"

            table = pd.read_csv(
                path,
                usecols=["sample_id", site_column, "cognitive_impairment", "y_prob"],
            ).rename(columns=rename)

            if table["sample_id"].duplicated().any():
                raise ValueError(f"{model}/{region} sample_id有重复。")

            if not np.isfinite(table["y_prob"].to_numpy(dtype=float)).all():
                raise ValueError(f"{model}/{region}预测概率包含非有限值。")

            if not table["y_prob"].between(0, 1).all():
                raise ValueError(f"{model}/{region}预测概率不在[0,1]内。")

            if not table["sample_id"].isin(expected.index).all():
                raise ValueError(f"{model}/{region}存在原数据中不存在的sample_id。")

            original = expected.loc[table["sample_id"].to_numpy()]
            sites = table["site_code"].astype(str).str.replace(r"\.0$", "", regex=True)
            if not np.array_equal(
                sites.to_numpy(), original[ENVIRONMENT].astype(str).to_numpy()
            ):
                raise ValueError(f"{model}/{region}监测点与原始数据不一致。")
            if not np.array_equal(
                table["cognitive_impairment"].astype(int).to_numpy(),
                original["label"].astype(int).to_numpy(),
            ):
                raise ValueError(f"{model}/{region}结局标签与原始数据不一致。")
            if set(original[REGION].unique()) != {region}:
                raise ValueError(f"{model}/{region}包含不属于留出地区的数据。")

            parts.append(table[["sample_id", "y_prob"]])
            manifest.append({
                "model": model,
                "held_out_region": region,
                "file": str(path.relative_to(ROOT)),
                "sha256": file_sha256(path),
                "test_rows": len(table),
            })

        all_predictions = pd.concat(parts, ignore_index=True)
        if all_predictions["sample_id"].duplicated().any():
            raise ValueError(f"{model}跨地区测试预测出现重复sample_id。")
        if set(all_predictions["sample_id"]) != set(frame["sample_id"]):
            raise ValueError(f"{model}未完整覆盖4769名研究对象。")
        all_data[model] = all_predictions.set_index("sample_id")["y_prob"]

    return all_data, manifest


def compute_site_metrics(frame, prediction_map):
    """任务三 第一步：计算风险排序、概率误差和高风险队列捕获率。"""
    rows = []
    for (region, site), part in frame.groupby([REGION, ENVIRONMENT]):
        site = str(site)
        ids = part["sample_id"].to_numpy(dtype=np.int64)
        labels = part["label"].to_numpy(dtype=int)

        positives = int(labels.sum())
        if positives == 0 or positives == len(labels):
            raise ValueError(f"{site}缺少计算AUC需要的阳性或阴性样本。")

        for model, predictions in prediction_map.items():
            p = predictions.loc[ids].to_numpy(dtype=float)
            pos = p[labels == 1]
            neg = p[labels == 0]

            k = max(1, int(math.ceil(len(labels) * 0.10)))
            top = np.argsort(-p, kind="mergesort")[:k]
            top_n_positive = int(labels[top].sum())
            site_rate = positives / len(labels)

            rows.append({
                "held_out_region": region,
                "site_code": site,
                "model": model,
                "n": len(labels),
                "positive_count": positives,
                "positive_rate": site_rate,
                "auc": float(roc_auc_score(labels, p)),
                "auprc": float(average_precision_score(labels, p)),
                "brier": float(brier_score_loss(labels, p)),
                "mean_predicted_probability": float(np.mean(p)),
                "mean_score_positive": float(np.mean(pos)),
                "mean_score_negative": float(np.mean(neg)),
                "mean_score_gap_positive_minus_negative": float(np.mean(pos)-np.mean(neg)),
                "top_decile_count": k,
                "top_decile_positive_count": top_n_positive,
                "top_decile_positive_rate": top_n_positive / k,
                "top_decile_lift": (top_n_positive / k) / site_rate,
                "predicted_positive_at_0_5": int(np.count_nonzero(p >= 0.5)),
            })

    result = pd.DataFrame(rows)
    if len(result) != 12 * len(MODELS):
        raise RuntimeError(f"期望60条站点模型记录，实际{len(result)}。")
    return result.sort_values(["held_out_region", "site_code", "model"])


def calculate_model_rank_agreement(frame, prediction_map):
    """任务三 第一步：比较同站点不同算法给出的风险排序。"""
    records = []
    for site, part in frame.groupby(ENVIRONMENT):
        ids = part["sample_id"].to_numpy(dtype=np.int64)
        region = str(part[REGION].iloc[0])
        for left, right in combinations(MODELS, 2):
            rank_left = prediction_map[left].loc[ids].to_numpy(dtype=float)
            rank_right = prediction_map[right].loc[ids].to_numpy(dtype=float)
            value = spearmanr(rank_left, rank_right).statistic
            records.append({
                "site_code": str(site),
                "held_out_region": region,
                "model_a": left,
                "model_b": right,
                "n": len(ids),
                "spearman_rho": float(value),
            })
    return pd.DataFrame(records)


# ==================== 二、条件特征分布分析 ====================

def js_divergence(counts_a, counts_b):
    """任务三 第一步：计算两个经验类别分布的JS散度，单位bits。"""
    a = np.asarray(counts_a, dtype=float)
    b = np.asarray(counts_b, dtype=float)
    p = a / a.sum()
    q = b / b.sum()
    mid = (p + q) / 2
    left = p > 0
    right = q > 0
    return float(
        (np.sum(p[left] * np.log2(p[left] / mid[left]))
         + np.sum(q[right] * np.log2(q[right] / mid[right]))) / 2
    )


def bh_adjust(p_values):
    """任务三 第一步：在单站点的19个特征范围内执行BH校正。"""
    p = np.asarray(p_values, dtype=float)
    q = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    running = 1.0
    for i in range(len(order)-1, -1, -1):
        running = min(running, p[order[i]] * len(order) / (i+1))
        q[order[i]] = running
    return q


def feature_outcome_association(frame, permutations):
    """任务三 第一步：站点内按结局分组的特征分布JS，估计稀有阳性造成的随机基线。

    只有17个阳性时，JS散度会有较大有限样本正偏。
    因此通过站点内置换标签（保持阳性人数不变）计算空假设参考分布。
    该关联不是P(Y|X)的因果效应，也不能证明标签错误。
    """
    records = []
    for site, part in frame.groupby(ENVIRONMENT):
        site = str(site)
        y = part["label"].to_numpy(dtype=int)

        for feature in FEATURES:
            values = (
                part[feature]
                .astype("string")
                .fillna("<MISSING>")
            )
            category, levels = pd.factorize(values, sort=True)
            n_categories = len(levels)

            def js_for_label(candidate_y):
                positives = np.bincount(
                    category[candidate_y == 1], minlength=n_categories
                )
                negatives = np.bincount(
                    category[candidate_y == 0], minlength=n_categories
                )
                return js_divergence(positives, negatives)

            observed = js_for_label(y)
            seed_token = f"{SEED}|{site}|{feature}".encode()
            seed = int(hashlib.sha256(seed_token).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)

            null_values = np.asarray(
                [js_for_label(rng.permutation(y)) for _ in range(permutations)],
                dtype=float,
            )
            null_mean = float(np.mean(null_values))
            p_value = (
                float((1 + np.count_nonzero(null_values >= observed))
                      / (permutations + 1))
            )
            records.append({
                "site_code": site,
                "held_out_region": str(part[REGION].iloc[0]),
                "feature": feature,
                "sample_count": len(y),
                "positive_count": int(y.sum()),
                "category_count": n_categories,
                "observed_label_conditional_js_bits": observed,
                "random_label_null_mean_js_bits": null_mean,
                "excess_js_over_null_bits": observed - null_mean,
                "permutation_p_value": p_value,
            })

    result = pd.DataFrame(records)
    result["bh_q_within_site"] = np.nan
    for site, idx in result.groupby("site_code").groups.items():
        result.loc[idx, "bh_q_within_site"] = bh_adjust(
            result.loc[idx, "permutation_p_value"].to_numpy()
        )
    return result


def matched_site_auc_bootstrap(frame, prediction_map, n_bootstraps):
    """任务三 第一步：比较220202与360102的AUC差异及分层bootstrap区间。

    每个站点固定阳性/阴性人数，各模型复用同一抽样索引。
    区间只描述这批训练完成的模型在当前站点样本上的抽样波动，
    不包括训练随机种子、站点总体或标签测量误差。
    """
    sites = {}
    for site in MATCHED_PAIR:
        part = frame.loc[frame[ENVIRONMENT] == site]
        if len(part) != 400 or int(part["label"].sum()) != 17:
            raise ValueError(
                f"配对案例预期{site}有400人、17阳性，实际"
                f"{len(part)}人、{int(part['label'].sum())}阳性。"
            )
        ids = part["sample_id"].to_numpy(dtype=np.int64)
        y = part["label"].to_numpy(dtype=int)
        scores = {
            model: predictions.loc[ids].to_numpy(dtype=float)
            for model, predictions in prediction_map.items()
        }
        sites[site] = (y, scores)

    rng = np.random.default_rng(SEED)
    auc_samples = {
        (site, model): []
        for site in MATCHED_PAIR for model in MODELS
    }
    for _ in range(n_bootstraps):
        for site in MATCHED_PAIR:
            y, scores = sites[site]
            positive_idx = np.flatnonzero(y == 1)
            negative_idx = np.flatnonzero(y == 0)
            sample = np.concatenate(
                [
                    rng.choice(positive_idx, size=len(positive_idx), replace=True),
                    rng.choice(negative_idx, size=len(negative_idx), replace=True),
                ]
            )
            for model in MODELS:
                auc_samples[(site, model)].append(
                    float(roc_auc_score(y[sample], scores[model][sample]))
                )

    records = []
    for model in MODELS:
        low = np.asarray(auc_samples[(MATCHED_PAIR[0], model)])
        high = np.asarray(auc_samples[(MATCHED_PAIR[1], model)])
        difference = high - low
        observed = float(
            roc_auc_score(*(
                sites[MATCHED_PAIR[1]][0],
                sites[MATCHED_PAIR[1]][1][model],
            ))
            - roc_auc_score(*(
                sites[MATCHED_PAIR[0]][0],
                sites[MATCHED_PAIR[0]][1][model],
            ))
        )
        records.append({
            "model": model,
            "low_auc_site": MATCHED_PAIR[0],
            "high_auc_site": MATCHED_PAIR[1],
            "observed_auc_difference_high_minus_low": observed,
            "bootstrap_q025": float(np.quantile(difference, 0.025)),
            "bootstrap_q975": float(np.quantile(difference, 0.975)),
            "bootstrap_repeats": n_bootstraps,
            "interpretation": "exploratory_stratified_bootstrap_fixed_models",
        })

    return pd.DataFrame(records)


# ==================== 三、输出结果 ====================

def draw_figures(site_metrics, feature_association, output):
    """任务三 第一步：生成易读的英文图表。"""
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for model in MODELS:
        subset = site_metrics.loc[site_metrics["model"] == model]
        ax.scatter(
            subset["positive_rate"],
            subset["auc"],
            label=model,
            alpha=0.78,
        )
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_xlabel("Monitoring-site positive rate")
    ax.set_ylabel("Monitoring-site ROC-AUC")
    ax.set_title("Prevalence alone does not explain site-level ROC-AUC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "prevalence_vs_site_auc.png", dpi=250)
    plt.close(fig)

    focus = site_metrics.loc[
        site_metrics["site_code"].isin(MATCHED_PAIR)
    ].copy()
    pivot = focus.pivot(index="model", columns="site_code", values="auc")
    pivot = pivot.reindex(list(MODELS))
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot.bar(ax=ax, width=0.75)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_ylabel("ROC-AUC")
    ax.set_xlabel("Algorithm")
    ax.set_title("Two sites: same prevalence, very different ranking")
    fig.tight_layout()
    fig.savefig(figures / "matched_low_prevalence_site_auc.png", dpi=250)
    plt.close(fig)

    focus_assoc = feature_association.loc[
        feature_association["site_code"].isin(MATCHED_PAIR)
    ]
    table = focus_assoc.pivot(
        index="feature",
        columns="site_code",
        values="excess_js_over_null_bits",
    )
    fig, ax = plt.subplots(figsize=(7, 8))
    image = ax.imshow(table.to_numpy(), aspect="auto", interpolation="nearest")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(table.columns)
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_yticklabels(table.index)
    ax.set_title("Feature-label JS above permutation null (exploratory)")
    fig.colorbar(image, ax=ax, label="Excess JS divergence (bits)")
    fig.tight_layout()
    fig.savefig(figures / "matched_site_feature_label_association.png", dpi=250)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--bootstraps", type=int, default=BOOTSTRAPS)
    args = parser.parse_args()

    if args.permutations < 100 or args.bootstraps < 200:
        raise ValueError("正式分析至少使用100次标签置换和200次bootstrap。")
    if not args.data.is_file():
        raise FileNotFoundError(f"缺少模型输入数据：{args.data}")
    if args.output.exists():
        raise FileExistsError(f"输出目录已存在，禁止覆盖：{args.output}")
    if file_sha256(args.data) != EXPECTED_DATA_SHA256:
        raise RuntimeError("当前数据SHA256与历史模型不一致，停止分析。")

    frame = attach_regions(read_raw(args.data, encoding="gbk"))
    if len(frame) != 4769 or frame[ENVIRONMENT].nunique() != 12:
        raise ValueError("数据样本数或监测点数与研究设计不一致。")

    # 任务三 第一步：计算期间只在内存使用个体sample_id，不导出个体结果。
    predictions, manifest = load_and_verify_predictions(frame)
    site_metrics = compute_site_metrics(frame, predictions)
    agreement = calculate_model_rank_agreement(frame, predictions)
    association = feature_outcome_association(frame, args.permutations)
    matched_auc = matched_site_auc_bootstrap(frame, predictions, args.bootstraps)

    matched_rows = site_metrics.loc[
        site_metrics["site_code"].isin(
            MATCHED_PAIR + OTHER_LOW_PREVALENCE_SITES
        )
    ].copy()

    pair_assoc = association.loc[
        association["site_code"].isin(MATCHED_PAIR)
    ].pivot(
        index="feature",
        columns="site_code",
        values=[
            "observed_label_conditional_js_bits",
            "excess_js_over_null_bits",
            "permutation_p_value",
            "bh_q_within_site",
        ],
    )
    pair_assoc.columns = [
        f"{name}_{site}" for name, site in pair_assoc.columns
    ]
    pair_assoc = pair_assoc.reset_index()

    args.output.mkdir(parents=True, exist_ok=False)
    for filename, table in (
        ("site_model_score_diagnostics.csv", site_metrics),
        ("site_model_rank_agreement.csv", agreement),
        ("site_feature_label_association.csv", association),
        ("matched_low_prevalence_sites.csv", matched_rows),
        ("matched_site_auc_bootstrap.csv", matched_auc),
        ("matched_site_feature_label_contrast.csv", pair_assoc),
    ):
        table.to_csv(args.output / filename, index=False, encoding="utf-8-sig")

    draw_figures(site_metrics, association, args.output)
    write_json(
        args.output / "experiment_metadata.json",
        {
            "task": "Task 3 Step 1: Monitoring-site failure diagnostics",
            "data_file": str(args.data.resolve()),
            "data_sha256": file_sha256(args.data),
            "model_names": list(MODELS),
            "regions": list(REGIONS),
            "site_count": int(frame[ENVIRONMENT].nunique()),
            "samples": int(len(frame)),
            "features": list(FEATURES),
            "matched_pair_sites": list(MATCHED_PAIR),
            "low_prevalence_comparator_sites": list(OTHER_LOW_PREVALENCE_SITES),
            "random_seed": SEED,
            "label_permutations": args.permutations,
            "auc_bootstraps": args.bootstraps,
            "individual_records_exported": False,
            "models_retrained": False,
            "all_saved_predictions_checked_by_sample_id_label_site": True,
            "methodology": {
                "site_auc": "ROC-AUC of each completed fixed LORO model at its held-out site",
                "score_gap": "mean predicted probability in cases minus noncases",
                "top_decile": "top ceil(0.1*n) risk scores within each site",
                "feature_label_js": "JS between X given Y=1 and X given Y=0 within each site",
                "null_js": "site-label permutation preserving the number of positive outcomes",
                "bootstrap": "site-wise stratified fixed-model sampling; paired across models",
            },
            "limitations": [
                "All site outcomes have already been explored; these are hypothesis-generating findings.",
                "Only 17 positive subjects occur in both matched comparison sites.",
                "AUC sampling intervals do not include training-seed variation or population-level site uncertainty.",
                "Feature-label association is not the conditional P(Y|X,E) effect and is not causal.",
                "Site-level feature associations can be unstable, with 19 multiple comparisons per site.",
                "Prediction mean can be miscalibrated because the historical losses use positive class weights.",
                "Matched pair is selected after reviewing earlier outcomes; it is exploratory, not confirmatory.",
            ],
            "model_prediction_sources": manifest,
        },
    )

    print("=" * 76)
    print("任务三 第一步：监测点预测失败诊断完成；未重新训练任何模型。")
    print(f"结果目录：{args.output}")
    print("\n同样仅17个阳性、400个样本的两个监测点：")
    print(
        site_metrics.loc[
            site_metrics["site_code"].isin(MATCHED_PAIR),
            [
                "site_code",
                "model",
                "auc",
                "auprc",
                "mean_score_gap_positive_minus_negative",
                "top_decile_positive_count",
            ],
        ].to_string(index=False)
    )
    print("\n两个监测点的AUC差异Bootstrap：")
    print(matched_auc.to_string(index=False))
    print("\n所有输出均为聚合统计，不包含个人标识信息。")


if __name__ == "__main__":
    main()

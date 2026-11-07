"""任务三 第4步：源域与两个失败案例对照站点的特征—结局关联迁移。

仅检查留出中部地区折。以训练源域标准差统一阳性减阴性的均值差，
保留源站点结果以检查合并混杂。数值编码方向不等于临床方向或因果效应。
不拟合模型，不修改数据，不选择特征；标签仅用于回顾性描述分析。
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
from elderly_health.data import ENVIRONMENT, FEATURES, attach_regions, read_raw  # noqa: E402
from task3_step3_fixed_representation_probes import align_raw  # noqa: E402
from task3_step2_v5_representation_diagnostics import (  # noqa: E402
    MODEL_ROOT, EXPECTED_DATA_SHA256, load_fold, sha256_file,
)

OUTPUT = ROOT / "outputs" / "analysis_2611" / "task3_step4_feature_association_transfer"
FOCUS = ("220202", "360102")
SEED = 26110304


def stable_rng(*tokens):
    """任务三 第4步：为每个组和特征固定独立随机流。"""
    token = "|".join(map(str, (SEED,) + tokens)).encode("utf-8")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8], "little"))


def association(part, feature, scale, group, repeats):
    """任务三 第4步：完整观测值均值差及分层bootstrap；缺失不填补。"""
    x = part[feature].to_numpy(dtype=float)
    y = part["label"].to_numpy(dtype=int)
    values = [x[(y == k) & np.isfinite(x)] for k in (0, 1)]
    row = {"group": group, "feature": feature, "n": len(part),
           "negative_total": int((y == 0).sum()), "positive_total": int((y == 1).sum()),
           "negative_observed": len(values[0]), "positive_observed": len(values[1]),
           "negative_missing": int(((y == 0) & ~np.isfinite(x)).sum()),
           "positive_missing": int(((y == 1) & ~np.isfinite(x)).sum()),
           "source_train_sd": float(scale), "direction_scope": "numeric_code_only"}
    valid = min(map(len, values)) >= 2 and np.isfinite(scale) and scale > 1e-12
    row["estimable"] = bool(valid)
    row["negative_mean"] = float(values[0].mean()) if len(values[0]) else np.nan
    row["positive_mean"] = float(values[1].mean()) if len(values[1]) else np.nan
    draws = np.full(repeats, np.nan)
    if valid:
        rng = stable_rng(group, feature)
        means = []
        for v in values:
            # 任务三 第4步：分批抽样，避免完整源域矩阵占用过多内存。
            chunks = []
            for start in range(0, repeats, 100):
                count = min(100, repeats - start)
                chunks.append(rng.choice(v, size=(count, len(v)), replace=True).mean(axis=1))
            means.append(np.concatenate(chunks))
        draws = (means[1] - means[0]) / scale
    row["standardized_mean_gap"] = (row["positive_mean"] - row["negative_mean"]) / scale if valid else np.nan
    row["ci_lower"] = float(np.quantile(draws, 0.025)) if valid else np.nan
    row["ci_upper"] = float(np.quantile(draws, 0.975)) if valid else np.nan
    return row, draws


def wilson_interval(positive, total):
    """任务三 第4步：各取值组阳性率的Wilson描述区间。"""
    p, z = positive / total, 1.959963984540054
    denominator = 1 + z*z / total
    center = (p + z*z / (2*total)) / denominator
    radius = z * np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return max(0.0, center-radius), min(1.0, center+radius)


def category_rates(groups, train):
    """任务三 第4步：以训练源域确定取值组；高基数变量按源域五分位分箱。"""
    rows, definitions = [], {}
    for feature in FEATURES:
        levels = np.sort(train[feature].dropna().unique())
        if len(levels) <= 20:
            definitions[feature] = {"method": "source_train_levels", "levels": levels.tolist()}
            def encode(series):
                result = series.map(lambda v: format(float(v), ".12g") if pd.notna(v) else "MISSING")
                result.loc[series.notna() & ~series.isin(levels)] = "UNSEEN_SOURCE"
                return result
        else:
            edges = np.unique(np.quantile(train[feature].dropna(), [0.2, 0.4, 0.6, 0.8]))
            bins = np.concatenate(([-np.inf], edges, [np.inf]))
            definitions[feature] = {"method": "source_train_quantiles", "finite_edges": edges.tolist()}
            def encode(series):
                return pd.cut(series, bins=bins, include_lowest=True).astype("string").fillna("MISSING")
        for group, part in groups.items():
            encoded = encode(part[feature])
            for category in sorted(encoded.unique()):
                labels = part.loc[encoded.eq(category), "label"].to_numpy(dtype=int)
                n, positives = len(labels), int(labels.sum())
                low, high = wilson_interval(positives, n)
                rows.append({"group": group, "feature": feature, "value_group": str(category),
                             "n": n, "positive_count": positives, "positive_rate": positives/n,
                             "wilson_lower": low, "wilson_upper": high})
    return pd.DataFrame(rows), definitions


def draw_figures(stats, contrasts, output):
    """任务三 第4步：绘制方向热力图、源站点对照及迁移差值区间。"""
    figures = output / "figures"
    figures.mkdir()
    order = ["SOURCE_TRAIN", "SOURCE_VALIDATION", *FOCUS]
    pivot = stats.pivot(index="feature", columns="group", values="standardized_mean_gap").reindex(index=FEATURES, columns=order)
    matrix = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
    finite = matrix.compressed()
    limit = max(0.1, float(np.max(np.abs(finite)))) if len(finite) else 1.0
    fig, ax = plt.subplots(figsize=(9, 10))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("lightgrey")
    plot = ax.imshow(matrix, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels(FEATURES)
    for i in range(len(FEATURES)):
        for j in range(len(order)):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.2f}" if np.isfinite(value) else "NA", ha="center", va="center", fontsize=8)
    ax.set_title("Feature-outcome direction: numeric-code contrast only")
    fig.colorbar(plot, ax=ax, label="(Positive mean - negative mean) / source-training SD")
    fig.tight_layout()
    fig.savefig(figures / "feature_association_direction.png", dpi=220)
    plt.close(fig)

    source_sites = sorted(g for g in stats["group"].unique() if g.startswith("SOURCE_SITE_"))
    site_table = stats.pivot(index="feature", columns="group", values="standardized_mean_gap").reindex(index=FEATURES, columns=source_sites + list(FOCUS))
    fig, ax = plt.subplots(figsize=(12, 9))
    plot = ax.imshow(np.ma.masked_invalid(site_table.to_numpy(dtype=float)), cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(site_table.columns)))
    ax.set_xticklabels([v.replace("SOURCE_SITE_", "S:") for v in site_table.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels(FEATURES)
    ax.set_title("Source-training sites and two target sites (grey: not estimable)")
    fig.colorbar(plot, ax=ax, label="Signed mean gap / source-training SD")
    fig.tight_layout()
    fig.savefig(figures / "source_site_association_heterogeneity.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 10), sharey=True)
    for ax, site in zip(axes, FOCUS):
        table = contrasts.loc[contrasts["site_code"] == site].set_index("feature").reindex(FEATURES)
        y = np.arange(len(FEATURES))
        ax.hlines(y, table["delta_ci_lower"], table["delta_ci_upper"], color="tab:blue")
        ax.scatter(table["target_minus_source_gap"], y, s=18)
        ax.axvline(0, color="grey", linestyle="--")
        ax.set_yticks(y)
        ax.set_yticklabels(FEATURES)
        ax.set_title(f"{site}: target minus source-training contrast")
        ax.set_xlabel("Difference in standardized mean gap (95% bootstrap CI)")
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(figures / "target_source_association_differences.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data" / "data_mission2.csv")
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bootstraps", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"固定输出目录已存在，禁止覆盖：{args.output}")
    if args.bootstraps < 200:
        raise ValueError("正式诊断至少使用200次bootstrap。")
    if sha256_file(args.data) != EXPECTED_DATA_SHA256:
        raise ValueError("数据哈希与历史模型不一致。")
    frame = attach_regions(read_raw(args.data, encoding="gbk"))
    if len(frame) != 4769 or frame[ENVIRONMENT].nunique() != 12:
        raise ValueError("研究样本数或监测点数异常。")
    fold = load_fold("central", args.model_root)
    if tuple(fold["config"]["feature_names"]) != tuple(FEATURES):
        raise ValueError("模型特征顺序不一致。")
    train, val, test = [align_raw(frame, fold[key]) for key in ("train", "validation", "test")]
    if set(pd.concat([train, val, test])["sample_id"]) != set(frame["sample_id"]):
        raise ValueError("历史划分没有覆盖全部研究样本。")
    groups = {"SOURCE_TRAIN": train, "SOURCE_VALIDATION": val}
    groups.update({"SOURCE_SITE_" + str(site): part for site, part in train.groupby(ENVIRONMENT)})
    for site in FOCUS:
        part = test.loc[test[ENVIRONMENT].eq(site)]
        if len(part) != 400 or int(part["label"].sum()) != 17:
            raise ValueError(f"{site}不符合400例、17例阳性的历史记录。")
        groups[site] = part
    rows, samples = [], {}
    scales = train[list(FEATURES)].std(ddof=1)
    for group, part in groups.items():
        print(f"任务三 第4步：分析 {group} 的19个特征关联", flush=True)
        for feature in FEATURES:
            row, draws = association(part, feature, scales[feature], group, args.bootstraps)
            rows.append(row)
            samples[(group, feature)] = draws
    stats = pd.DataFrame(rows)
    indexed = stats.set_index(["group", "feature"])
    contrasts = []
    for site in FOCUS:
        for feature in FEATURES:
            source = indexed.loc[("SOURCE_TRAIN", feature)]
            target = indexed.loc[(site, feature)]
            valid = bool(source["estimable"] and target["estimable"])
            difference = samples[(site, feature)] - samples[("SOURCE_TRAIN", feature)]
            s, t = source["standardized_mean_gap"], target["standardized_mean_gap"]
            contrasts.append({"site_code": site, "feature": feature, "estimable": valid,
                              "source_gap": s, "target_gap": t,
                              "target_minus_source_gap": t-s,
                              "delta_ci_lower": float(np.quantile(difference, 0.025)) if valid else np.nan,
                              "delta_ci_upper": float(np.quantile(difference, 0.975)) if valid else np.nan,
                              "opposite_point_estimate_signs": bool(s*t < 0) if valid else None,
                              "smaller_target_absolute_gap": bool(abs(t) < abs(s)) if valid else None})
    contrasts = pd.DataFrame(contrasts)
    rates, definitions = category_rates(groups, train)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, table in (("feature_association_by_group.csv", stats), ("target_source_association_contrasts.csv", contrasts),
                        ("feature_value_group_positive_rates.csv", rates)):
        table.to_csv(args.output / name, index=False, encoding="utf-8-sig")
    draw_figures(stats, contrasts, args.output)
    metadata = {"task": "Task 3 Step 4 - Feature-outcome association transfer", "held_out_region": "central",
                "data_sha256": sha256_file(args.data), "models_trained": False, "features_selected": False,
                "source_reference": "historical source train only; validation shown separately",
                "missing_policy": "feature-wise complete cases, class counts and missing counts exported; no imputation",
                "scale": "fixed source-train sample SD (ddof=1), not pooled-within-class SD",
                "groups": {name: len(part) for name, part in groups.items()}, "seed": SEED,
                "bootstrap_repeats": args.bootstraps, "individual_records_exported": False,
                "value_group_definitions": definitions, "historical_artifact_hashes": fold["artifacts"],
                "limitations": [
                    "All associations are unadjusted and observational; no causal effect or full conditional-shift test is estimated.",
                    "For nominal categories, numeric-code direction depends on arbitrary encoding; use value-group rates for interpretation.",
                    "physical_activity provenance and smoke coding remain unresolved; no clinical direction is asserted.",
                    "Source pooling may mix within-site and between-site associations; source-site results are provided separately.",
                    "Bootstrap samples observed positive/negative values independently for each feature and group, treating source SD as fixed.",
                    "Intervals exclude site-population, training, missingness and source-scale uncertainty; they are unadjusted exploratory intervals.",
                    "Opposite point signs alone do not establish a genuine reversal; insignificant association is not evidence of absence.",
                    "Source-constant features are not estimable on this scale; a target-constant observed feature has gap zero if source SD is valid.",
                    "Value-group Wilson intervals are descriptive and unadjusted; sparse groups and 17 positives per target site limit precision.",
                    "No individual features, predictions or identifiers are exported; value-group outputs contain aggregate counts only.",
                ]}
    (args.output / "experiment_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"任务三 第4步完成：未训练模型；输出目录：{args.output}")


if __name__ == "__main__":
    main()

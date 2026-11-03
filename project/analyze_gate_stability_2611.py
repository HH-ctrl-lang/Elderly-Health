"""2611 / Task 1, step 2: stability of CausalGate feature rankings.

Run from the repository root after completing step 1:
    python project/analyze_gate_stability_2611.py

Reads only aggregate gate scores. Does not load or retrain a model, and
never accesses individual-level health records.
"""
import argparse
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "outputs/analysis_2611/gate_step1/gate_matrix.csv"
DEFAULT_OUTPUT = ROOT / "outputs/analysis_2611/gate_step2"
THRESHOLDS = (0.45, 0.50, 0.55)
TOP_K = 5


def read_gate_matrix(path):
    """Check expected fold/feature layout, scores and grouping."""
    frame = pd.read_csv(path, encoding="utf-8-sig", index_col=0)
    if frame.shape != (15, 19):
        raise ValueError(f"Expected a 15x19 gate matrix, received {frame.shape}")
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError("Duplicate fold or feature names in input")
    if not all(label.startswith("site:") for label in frame.index[:12]):
        raise ValueError("Expected the first 12 rows to be LOEO sites")
    if set(frame.index[12:]) != {"region:east", "region:central", "region:west"}:
        raise ValueError("Expected the last 3 rows to be LORO regions")
    try:
        frame = frame.astype(float)
    except ValueError as exc:
        raise ValueError("All gate scores must be numeric") from exc
    values = frame.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("All 285 scores must be finite values in [0, 1]")
    return frame


def group_name(fold):
    return "LOEO_site" if fold.startswith("site:") else "LORO_region"


def rank_matrix(gates):
    """Rank 1 means highest allocation to z_c, NOT causal importance."""
    return gates.rank(axis=1, method="average", ascending=False)


def topk(gates, fold, k=TOP_K):
    return set(gates.loc[fold].nlargest(k, keep="first").index)


def jaccard(a, b):
    union = a | b
    return len(a & b) / len(union) if union else float("nan")


def pairwise_results(gates, ranks):
    records = []
    for level, subset in (
        ("LOEO_site", list(gates.index[:12])),
        ("LORO_region", list(gates.index[12:])),
    ):
        for left, right in combinations(subset, 2):
            rl = ranks.loc[left].to_numpy()
            rr = ranks.loc[right].to_numpy()
            # Constant gates are a special case: rank similarity not meaningful.
            if gates.loc[left].nunique() == 1 or gates.loc[right].nunique() == 1:
                rho = float("nan")
            else:
                rho = float(np.corrcoef(rl, rr)[0, 1])
            selected_l = set(gates.columns[gates.loc[left] > 0.50])
            selected_r = set(gates.columns[gates.loc[right] > 0.50])
            records.append(
                {
                    "level": level,
                    "fold_a": left,
                    "fold_b": right,
                    "spearman_rank_correlation": rho,
                    "top3_jaccard": jaccard(topk(gates, left, 3), topk(gates, right, 3)),
                    "top5_jaccard": jaccard(topk(gates, left, 5), topk(gates, right, 5)),
                    "threshold_0_5_jaccard": jaccard(selected_l, selected_r),
                }
            )
    return pd.DataFrame(records)


def feature_summary(gates, ranks):
    records = []
    for feature in gates.columns:
        record = {"feature": feature}
        for level, subset in (
            ("loeo", gates.iloc[:12]),
            ("loro", gates.iloc[12:]),
        ):
            r = ranks.loc[subset.index, feature]
            g = subset[feature]
            record[f"{level}_mean_gate"] = float(g.mean())
            record[f"{level}_mean_rank"] = float(r.mean())
            record[f"{level}_sd_rank"] = float(r.std(ddof=1))
            record[f"{level}_top3_count"] = sum(
                feature in topk(gates, fold, 3) for fold in subset.index
            )
            record[f"{level}_top5_count"] = sum(
                feature in topk(gates, fold, 5) for fold in subset.index
            )
            record[f"{level}_above_0_50_count"] = int((g > 0.50).sum())
            record[f"{level}_above_0_55_count"] = int((g > 0.55).sum())
        records.append(record)
    return pd.DataFrame(records).sort_values(
        ["loro_top5_count", "loeo_top5_count", "loeo_mean_rank"],
        ascending=[False, False, True],
    )


def threshold_sensitivity(gates):
    rows = []
    for fold in gates.index:
        for t in THRESHOLDS:
            selected = gates.columns[gates.loc[fold] > t].tolist()
            rows.append(
                {
                    "level": group_name(fold),
                    "held_out": fold,
                    "threshold": t,
                    "selected_count": len(selected),
                    "selected_features": ";".join(selected),
                }
            )
    return pd.DataFrame(rows)


def plot_ranks(ranks, path):
    fig, ax = plt.subplots(figsize=(17, 7))
    img = ax.imshow(
        ranks.to_numpy(),
        vmin=1,
        vmax=len(ranks.columns),
        cmap="viridis_r",
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_yticks(np.arange(len(ranks)))
    ax.set_yticklabels(ranks.index, fontsize=9)
    ax.set_xticks(np.arange(len(ranks.columns)))
    ax.set_xticklabels(ranks.columns, rotation=55, ha="right", fontsize=8)
    ax.axhline(11.5, color="white", linewidth=2)
    ax.set_title("CausalGate within-fold rankings (1 = largest gate score)")
    cb = fig.colorbar(img, ax=ax, fraction=0.019, pad=0.012)
    cb.set_label("Within-fold rank, NOT feature or causal importance")
    fig.subplots_adjust(left=0.12, right=0.96, top=0.91, bottom=0.30)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_thresholds(thresholds, path):
    labels = thresholds["held_out"].drop_duplicates().tolist()
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 4.8))
    for i, threshold in enumerate(THRESHOLDS):
        counts = thresholds.loc[thresholds.threshold == threshold, "selected_count"]
        ax.bar(x + (i - 1) * width, counts.to_numpy(), width, label=f"gate > {threshold:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Number of features above threshold")
    ax.set_ylim(0, 19)
    ax.axvline(11.5, linewidth=1.1, linestyle="--", color="gray")
    ax.legend(ncol=3)
    ax.set_title("How a small threshold change alters the selected set")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(input_path, output):
    gates = read_gate_matrix(input_path)
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")
    ranks = rank_matrix(gates)
    feature_stats = feature_summary(gates, ranks)
    pairs = pairwise_results(gates, ranks)
    sensitivity = threshold_sensitivity(gates)
    group_stats = pairs.groupby("level", sort=False).agg(
        comparisons=("spearman_rank_correlation", "size"),
        mean_rank_spearman=("spearman_rank_correlation", "mean"),
        min_rank_spearman=("spearman_rank_correlation", "min"),
        max_rank_spearman=("spearman_rank_correlation", "max"),
        mean_top5_jaccard=("top5_jaccard", "mean"),
        mean_threshold_0_5_jaccard=("threshold_0_5_jaccard", "mean"),
    ).reset_index()
    output.mkdir(parents=True)
    def save(table, name):
        table.to_csv(output / name, index=False, encoding="utf-8-sig")
    save(feature_stats, "gate_rank_summary.csv")
    save(pairs, "gate_pairwise_similarity.csv")
    save(group_stats, "gate_pairwise_summary.csv")
    save(sensitivity, "gate_threshold_sensitivity.csv")
    ranks.to_csv(output / "gate_rank_matrix.csv", index_label="held_out", encoding="utf-8-sig")
    plot_ranks(ranks, output / "gate_rank_heatmap.png")
    plot_thresholds(sensitivity, output / "gate_threshold_sensitivity.png")

    common_loro = set.intersection(*(topk(gates, fold) for fold in gates.index[12:]))
    lines = [
        "# CausalGate stability diagnostics (descriptive, seed=42)",
        "",
        "This is a ranking/allocation analysis, **not** an assessment of causal effects or predictive feature importance.",
        "",
        "## Pairwise summary",
        "",
        "| Group | Pair count | Mean Spearman rank correlation | Mean top-5 Jaccard |",
        "|---|---:|---:|---:|",
    ]
    for _, r in group_stats.iterrows():
        lines.append(
            f"| {r.level} | {int(r.comparisons)} | "
            f"{r.mean_rank_spearman:.4f} | {r.mean_top5_jaccard:.4f} |"
        )
    lines.extend([
        "",
        "## Common LORO top-5 features",
        "",
        ", ".join(sorted(common_loro)) or "None",
        "",
        "## Caveats",
        "",
        "- All results are seed=42 and use overlapping source training samples between folds.",
        "- Rank consistency is relative, even when all gate values remain close to 0.5.",
        "- A score above 0.5 reflects allocation to z_c; it is not a hard selected feature.",
        "- The V5 joint prediction head uses z_c and z_s, so gate scores are not feature importances.",
        "- Threshold sensitivity is a display diagnostic, not hyperparameter tuning.",
        "- Do not select a threshold or algorithm from held-out test performance.",
        "",
    ])
    (output / "gate_stability_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Read 15 folds x 19 features from", input_path)
    print(group_stats.to_string(index=False))
    print("Common LORO top-5 features:", ", ".join(sorted(common_loro)))
    print("Results saved in:", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.input.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()

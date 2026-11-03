"""2611 / 任务1：总结现有CausalGate分数

仅导出聚合后的模型级门控值；不包含单条记录。
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent
REPO = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
from elderly_health.data import FEATURES  # noqa: E402


def read_folds(root):
    specs = (
        (
            "LOEO_site",
            root / "outputs/v2-region-loeo-selection91-42",
            12,
        ),
        (
            "LORO_region",
            root / "outputs/v2-region-loro-selection91-42",
            3,
        ),
    )
    records = []
    for level, directory, expected in specs:
        paths = list(directory.glob("*/v5/metrics.json"))
        if len(paths) != expected:
            raise FileNotFoundError(
                f"Expected {expected} {level} metric files in {directory}; "
                f"found {len(paths)}. Run git lfs pull / check the results checkout."
            )
        if level == "LORO_region":
            order = {"east": 0, "central": 1, "west": 2}
            paths.sort(key=lambda p: order.get(p.parent.parent.name, 99))
        else:
            paths.sort(key=lambda p: p.parent.parent.name)
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            gate = data.get("gate")
            if not isinstance(gate, dict) or set(gate) != set(FEATURES):
                raise ValueError(f"Missing or mismatched 19-feature gate: {path}")
            values = np.asarray([gate[f] for f in FEATURES], dtype=float)
            if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
                raise ValueError(f"Invalid gate score in {path}")
            selected = set(data.get("selected_features", []))
            if selected != {f for f, value in zip(FEATURES, values) if value > 0.5}:
                raise ValueError(f"selected_features inconsistent with >0.5: {path}")
            records.append(
                {
                    "level": level,
                    "held_out": path.parent.parent.name,
                    "values": values,
                    "metric_path": str(path.relative_to(root)),
                    "auc": data.get("main", {}).get("auc"),
                }
            )
    return records


def draw_heatmap(matrix, labels, output, vmin, vmax, title):
    fig, ax = plt.subplots(figsize=(17, 7))
    im = ax.imshow(
        matrix,
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(FEATURES)))
    ax.set_xticklabels(FEATURES, rotation=52, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axhline(11.5, color="black", linewidth=1.2)
    ax.set_title(title, fontsize=12, pad=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.018)
    cb.set_label("Gate score (allocation to z_c); 0.5 = equal split")
    fig.subplots_adjust(left=0.11, right=0.96, top=0.91, bottom=0.29)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(root, output):
    records = read_folds(root)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing analysis folder: {output}"
        )
    output.mkdir(parents=True)

    gate_long = []
    fold_rows = []
    matrix = []
    labels = []
    for rec in records:
        vals = rec["values"]
        labels.append(f"{'site' if rec['level'] == 'LOEO_site' else 'region'}:{rec['held_out']}")
        matrix.append(vals)
        fold_rows.append(
            {
                "level": rec["level"],
                "held_out": rec["held_out"],
                "test_auc": rec["auc"],
                "gate_mean": vals.mean(),
                "gate_sd": vals.std(ddof=1),
                "gate_min": vals.min(),
                "gate_max": vals.max(),
                "gate_range": vals.max() - vals.min(),
                "selected_gt_0_5_count": int((vals > 0.5).sum()),
                "strong_gt_0_6_count": int((vals > 0.6).sum()),
                "metric_path": rec["metric_path"],
            }
        )
        for f, score in zip(FEATURES, vals):
            gate_long.append(
                {
                    "level": rec["level"],
                    "held_out": rec["held_out"],
                    "feature": f,
                    "gate_weight": float(score),
                    "selected_gt_0_5": int(score > 0.5),
                }
            )
    pd.DataFrame(gate_long).to_csv(
        output / "gate_weights_long.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(fold_rows).to_csv(
        output / "gate_fold_summary.csv", index=False, encoding="utf-8-sig"
    )

    matrix = np.asarray(matrix)
    stats = []
    for index, feature in enumerate(FEATURES):
        all_values = matrix[:, index]
        site_values = matrix[:12, index]
        region_values = matrix[12:, index]
        stats.append(
            {
                "feature": feature,
                "mean_all": all_values.mean(),
                "sd_all": all_values.std(ddof=1),
                "min_all": all_values.min(),
                "max_all": all_values.max(),
                "mean_loeo": site_values.mean(),
                "sd_loeo": site_values.std(ddof=1),
                "selected_loeo_12": int((site_values > 0.5).sum()),
                "mean_loro": region_values.mean(),
                "sd_loro": region_values.std(ddof=1),
                "selected_loro_3": int((region_values > 0.5).sum()),
                "gt_0_6_all_15": int((all_values > 0.6).sum()),
            }
        )
    pd.DataFrame(stats).to_csv(
        output / "gate_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(matrix, index=labels, columns=FEATURES).to_csv(
        output / "gate_matrix.csv", index_label="held_out", encoding="utf-8-sig"
    )

    draw_heatmap(
        matrix, labels, output / "gate_heatmap_full.png",
        0.0, 1.0, "CausalGate across held-out sites and regions (full 0-1 scale)",
    )
    draw_heatmap(
        matrix, labels, output / "gate_heatmap_zoom.png",
        0.3, 0.7,
        "CausalGate (zoomed color scale 0.3-0.7; visual differences amplified)",
    )

    print(f"Read {len(records)} folds: 12 LOEO sites and 3 LORO regions")
    print(f"Global gate range: {matrix.min():.4f} to {matrix.max():.4f}")
    print(f"Scores above 0.6: {(matrix > 0.6).sum()} / {matrix.size}")
    print(f"Output saved to: {output}")
    print("Next: inspect gate_summary.csv and both heatmaps; do not infer causality.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "outputs/analysis_2611/gate_step1",
    )
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()

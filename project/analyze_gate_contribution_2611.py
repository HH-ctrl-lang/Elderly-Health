"""任务1，第3步：使用固定的V5检查点衡量特征依赖度。

不重新训练，也不进行参数调优。在源验证集和未见区域测试划分上，对每个归一化特征在站点内进行置换，分别针对主预测头和仅zc预测头。所报告的AUC/AP变化是描述性的预测依赖，而非因果关系。

从代码仓库根目录运行：
git lfs pull
python project/analyze_gate_contribution_2611.py
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
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "project/src"))

from elderly_health.data import FEATURES  # noqa: E402
from elderly_health.models import CausalDisentangledModel  # noqa: E402
from elderly_health.training import predict  # noqa: E402

SOURCE = ROOT / "outputs/v2-region-loro-selection91-42"
DEFAULT_OUTPUT = ROOT / "outputs/analysis_2611/gate_step3"
REGIONS = ("east", "central", "west")


def require_lfs_data(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required result missing: {path}")
    with path.open("rb") as stream:
        if b"version https://git-lfs.github.com/spec/v1" in stream.read(140):
            raise RuntimeError(f"Git LFS pointer, not actual data: {path}. Run git lfs pull.")


def metric(y, p):
    if np.unique(y).size != 2:
        raise ValueError("Both binary classes are required to calculate ROC-AUC.")
    return {
        "auc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
    }


def restore_fold(region):
    folder = SOURCE / region / "v5"
    model_path = folder / "best_model.pt"
    split_path = folder / "splits.npz"
    require_lfs_data(model_path)
    require_lfs_data(split_path)
    metadata = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    config = metadata["config"]
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if config["model"] != "v5":
        raise ValueError(f"Expected V5 checkpoint in {folder}")
    if list(checkpoint["feature_names"]) != list(FEATURES):
        raise ValueError(f"Checkpoint feature order mismatch: {folder}")
    if checkpoint["config"] != config:
        raise ValueError(f"Checkpoint and exported configuration differ: {folder}")
    model = CausalDisentangledModel(
        input_dim=len(FEATURES),
        num_envs=len(checkpoint["env_map"]),
        latent_dim=config["latent_dim"],
        gate_init_std=config["gate_init_std"],
        gate_temperature=config["gate_temperature"],
        joint=True,
    ).cpu()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    splits = {}
    with np.load(split_path, allow_pickle=False) as archive:
        for subset in ("val", "test"):
            x = np.asarray(archive[f"X_{subset}"], dtype=np.float32)
            y = np.asarray(archive[f"y_{subset}"], dtype=np.float32)
            sites = np.asarray(archive[f"site_env_{subset}"]).astype(str)
            if x.shape[1] != len(FEATURES) or len(x) != len(y) or len(y) != len(sites):
                raise ValueError(f"Unexpected feature and label sizes: {folder}/{subset}")
            splits[subset] = x, y, sites

    # Check saved predictions to rule out reconstructing a different model.
    for subset, (x, y, _sites) in splits.items():
        recorded = pd.read_csv(
            folder / f"representations_{subset}.csv",
            usecols=["y_prob", "zc_prob", "cognitive_impairment"],
        )
        if len(recorded) != len(y) or not np.array_equal(
            recorded["cognitive_impairment"].to_numpy(), y
        ):
            raise ValueError(f"Saved labels or ordering mismatch: {folder}/{subset}")
        for head, column in (("main", "y_prob"), ("zc", "zc_prob")):
            current = predict(model, x, torch.device("cpu"), head=head)
            error = np.max(np.abs(current - recorded[column].to_numpy(dtype=float)))
            if error > 2e-5:
                raise ValueError(
                    f"Restored predictions differ: {region}/{subset}/{head}, "
                    f"max absolute error {error:.7g}"
                )
    return model, splits


def permute_within_sites(x, sites, feature_idx, seed):
    """Only change one feature; preserve each site's marginal distribution."""
    rng = np.random.default_rng(seed)
    modified = x.copy()
    for site in np.unique(sites):
        indices = np.flatnonzero(sites == site)
        if len(indices) > 1:
            modified[indices, feature_idx] = x[rng.permutation(indices), feature_idx]
    return modified


def plot_contribution(data, head, output):
    subset = data[(data["subset"] == "test") & (data["head"] == head)]
    order = (
        subset.groupby("feature")["mean_auc_drop"]
        .mean().sort_values(ascending=False).index.tolist()
    )
    matrix = subset.pivot(index="feature", columns="region", values="mean_auc_drop")
    matrix = matrix.loc[order]
    fig, ax = plt.subplots(figsize=(13, 7.5))
    axis_x = np.arange(len(order))
    for i, region in enumerate(REGIONS):
        ax.bar(
            axis_x + (i-1) * 0.25,
            matrix[region].to_numpy(),
            width=0.24,
            label="Held out " + region,
        )
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xticks(axis_x)
    ax.set_xticklabels(order, rotation=65, ha="right", fontsize=8)
    ax.set_ylabel("Original AUC minus shuffled AUC (positive = deterioration)")
    ax.set_title("Fixed-model feature permutation, " + head + " head")
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run(output, repeats):
    if repeats < 2:
        raise ValueError("--repeats must be at least 2")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite analysis output: {output}")
    torch.set_num_threads(2)
    permutation_rows = []
    baseline_rows = []
    for fold_idx, region in enumerate(REGIONS):
        print("Verifying saved model and predictions for", region, flush=True)
        model, splits = restore_fold(region)
        for subset, (x, y, sites) in splits.items():
            for head in ("main", "zc"):
                original = predict(model, x, torch.device("cpu"), head=head)
                base = metric(y, original)
                baseline_rows.append({
                    "region": region,
                    "subset": subset,
                    "head": head,
                    "n": len(y),
                    "positive_rate": float(y.mean()),
                    "original_auc": base["auc"],
                    "original_auprc": base["auprc"],
                })
                for feature_index, feature in enumerate(FEATURES):
                    for repeat in range(repeats):
                        seed = (
                            2611000 + fold_idx * 100000
                            + (0 if subset == "val" else 50000)
                            + feature_index * 100 + repeat
                        )
                        disturbed_x = permute_within_sites(
                            x, sites, feature_index, seed
                        )
                        disturbed = metric(
                            y, predict(
                                model, disturbed_x, torch.device("cpu"), head=head
                            )
                        )
                        permutation_rows.append({
                            "region": region,
                            "subset": subset,
                            "head": head,
                            "feature": feature,
                            "auc_drop": base["auc"] - disturbed["auc"],
                            "auprc_drop": base["auprc"] - disturbed["auprc"],
                        })
        print("Completed", region, flush=True)

    summary = (
        pd.DataFrame(permutation_rows)
        .groupby(["region", "subset", "head", "feature"], sort=False)
        .agg(
            mean_auc_drop=("auc_drop", "mean"),
            sd_auc_drop=("auc_drop", "std"),
            mean_auprc_drop=("auprc_drop", "mean"),
            sd_auprc_drop=("auprc_drop", "std"),
            n_permutations=("auc_drop", "size"),
        )
        .reset_index()
    )
    output.mkdir(parents=True)
    pd.DataFrame(baseline_rows).to_csv(
        output / "original_metrics.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        output / "feature_contribution_summary.csv",
        index=False, encoding="utf-8-sig"
    )
    plot_contribution(summary, "main", output / "main_auc_drop_by_region.png")
    plot_contribution(summary, "zc", output / "zc_auc_drop_by_region.png")
    (output / "README.md").write_text(
        "\n".join([
            "# Experiment 2611, step 3: permutation-based feature reliance",
            "",
            "- Fixed existing V5 checkpoints; no retraining, tuning or model selection.",
            "- Features permuted independently within each original site.",
            "- Source validation and held-out test are reported separately.",
            "- Main head uses both z_c and z_s; zc head uses z_c only.",
            "- Positive AUC/AP drop means worse predictions after perturbation.",
            "- A negative drop means shuffling improved the evaluated metric.",
            "- SD across permutations reflects shuffling randomness only.",
            "- Correlated features can make permutation effects hard to interpret.",
            "- This is not proof of feature importance as a causal effect.",
            "- Never select parameters or report an optimized model from held-out test results.",
            "",
        ]),
        encoding="utf-8",
    )
    print(f"Success: wrote {len(summary)} aggregate rows to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    run(args.output.resolve(), args.repeats)


if __name__ == "__main__":
    main()

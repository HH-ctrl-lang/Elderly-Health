"""任务五 第2步：固定其余设置，仅检查MMD权重0、0.1、1、10的影响。

先从第1步冻结权重复核0和1的结果，再各训练三个试验的0.1和10组。
不根据真实潜变量成绩选择权重，不做新的健康数据实验。
运行：python project/src/analysis/task5_step2_mmd_weight_control.py
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from task5_step1_invariance_disentanglement_pilot import (
    ARMS, ROOT, SEED, make_data, build_autoencoder, fit_model, diagnose,
)

HISTORY = ROOT / "outputs/analysis_2611/task5_step1_invariance_disentanglement_pilot"
OUTPUT = ROOT / "outputs/analysis_2611/task5_step2_mmd_weight_control"
WEIGHTS = (0.0, 0.1, 1.0, 10.0)


def sha256(path):
    """任务五 第2步：记录实际读取的历史文件与代码版本。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """任务五 第2步：保存配置与校验，不生成阶段小结。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def check_protocol(metadata):
    """任务五 第2步：拒绝把不同预算或不同方法的结果混进同一对照。"""
    required = {"samples_per_environment": 6000, "test_samples_per_environment": 2000,
                "epochs": 100, "batch_size_per_environment": 256, "mmd_weight": 1.0,
                "seed": 42, "health_data_used": False,
                "latent_truth_used_for_autoencoder_training": False,
                "reproduction_level": "independent_downscaled_mechanism_reproduction"}
    if any(metadata.get(key) != value for key, value in required.items()):
        raise ValueError("第1步的实验设置与预定对照不一致。")
    if metadata.get("mmd_kernel_gammas") != [0.5, 1.0, 2.0]:
        raise ValueError("核函数设置发生变化。")
    expected_means = np.random.default_rng(SEED).uniform(2, 5, size=3)
    if not np.array_equal(np.asarray(metadata.get("nintervention_means")), expected_means):
        raise ValueError("模拟环境发生变化。")
    # 任务五 第2步：允许Windows换行差别，但不接受未知的历史程序版本。
    if metadata.get("script_sha256") not in {
        "f46c4de438c9c712587a5d090cf113a910b95d3c54713a44182287496a5e1f40",
        "294342d9c38dabbd00cd0b860c8f545687eea40bfc3d9d08686ef0ddee413a32",
    }:
        raise ValueError("第1步脚本版本未知，请先核对历史结果。")
    return expected_means


def compare_recovery(previous, current, trial, arm):
    """任务五 第2步：冻结模型重新读出，核对12项历史R2，任何失败都停止。"""
    subset = previous.loc[previous.trial.eq(trial) & previous.arm.eq(arm)]
    if len(subset) != 12 or subset.duplicated(["representation", "latent"]).any():
        raise ValueError("历史读出表不完整或重复。")
    old = subset.set_index(["representation", "latent"]).test_r2
    checks = []
    for row in current:
        reference = float(old.loc[(row["representation"], row["latent"])])
        error = abs(reference-row["test_r2"])
        if not np.isclose(reference, row["test_r2"], atol=2e-5, rtol=0):
            raise RuntimeError(f"冻结模型复核失败：trial={trial}, {arm}, {row['representation']}, {row['latent']}, error={error}")
        checks.append({"trial": trial, "arm": arm, "representation": row["representation"],
                       "latent": row["latent"], "absolute_r2_error": error, "passed": True})
    return checks


def draw_figures(recovery, diagnostics, output):
    """任务五 第2步：同时展示目标信息保留、变化变量残留与分布差异。"""
    figures = output / "figures"
    figures.mkdir()
    part = recovery.loc[recovery.representation.eq("invariant_block_1_3")]
    positions = np.arange(len(WEIGHTS))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, latent in zip(axes, ("z1", "z3", "z2")):
        for trial in range(3):
            rows = part.loc[part.trial.eq(trial) & part.latent.eq(latent)].set_index("mmd_weight").loc[list(WEIGHTS)]
            ax.plot(positions, rows.test_r2, marker="o", alpha=0.6, label=f"Trial {trial}")
        means = part.loc[part.latent.eq(latent)].groupby("mmd_weight").test_r2.mean().reindex(WEIGHTS)
        ax.plot(positions, means, color="black", linewidth=2, marker="s", label="Mean")
        ax.set_xticks(positions)
        ax.set_xticklabels([str(w) for w in WEIGHTS])
        ax.set_xlabel("MMD weight (categorical spacing)")
        ax.set_ylabel("Held-out R2 from block [h1, h3]")
        ax.set_title(f"{latent}: {'retention' if latent != 'z2' else 'remaining information'}")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "mmd_weight_vs_latent_recovery.png", dpi=220)
    plt.close(fig)
    # 任务五 第2步：两环境的块MMD记录重复；只取观测环境一行进行试验平均。
    block = diagnostics.loc[diagnostics.domain.eq("observational")]
    means = block.groupby("mmd_weight").heldout_block_mmd_standardized.mean().reindex(WEIGHTS)
    recon = diagnostics.groupby("mmd_weight").reconstruction_mse.mean().reindex(WEIGHTS)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, values, label in ((axes[0], means, "Standardized block MMD squared"),
                              (axes[1], recon, "Held-out reconstruction MSE")):
        ax.plot(positions, values, marker="o")
        ax.set_xticks(positions)
        ax.set_xticklabels([str(w) for w in WEIGHTS])
        ax.set_xlabel("MMD weight (categorical spacing)")
        ax.set_ylabel(label)
    fig.tight_layout()
    fig.savefig(figures / "mmd_weight_alignment_and_reconstruction.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=HISTORY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"禁止覆盖已有结果：{args.output}")
    import torch
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    metadata_path = args.history / "experiment_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    means = check_protocol(metadata)
    if not json.loads((args.history / "completion.json").read_text(encoding="utf-8")).get("completed"):
        raise ValueError("第1步未完成。")
    old = pd.read_csv(args.history / "latent_recovery.csv")
    if len(old) != 72 or not np.isfinite(old.test_r2.to_numpy()).all():
        raise ValueError("第1步读出表不完整。")
    args.output.mkdir(parents=True, exist_ok=False)
    source_hashes = {name: sha256(args.history / name)
                     for name in ("experiment_metadata.json", "completion.json", "latent_recovery.csv")}
    for trial in range(3):
        for arm in ARMS:
            name = f"trial_{trial}_{arm}.pt"
            path = args.history / name
            if not path.is_file():
                raise FileNotFoundError(f"缺少第1步权重：{path}；请检查历史输出及git lfs pull。")
            source_hashes[name] = sha256(path)
    write_json(args.output / "experiment_design.json", {
        "task": "Task 5 Step 2: MMD weight control", "mmd_weights": list(WEIGHTS),
        "new_training_weights": [0.1, 10.0], "reused_frozen_weights": [0.0, 1.0],
        "health_data_used": False, "source_files_sha256": source_hashes,
        "unchanged_budget": {k: metadata[k] for k in ("samples_per_environment", "test_samples_per_environment", "epochs", "batch_size_per_environment")},
        "code_sha256": {"runner": sha256(Path(__file__)),
                        "shared_pilot": sha256(Path(__file__).with_name("task5_step1_invariance_disentanglement_pilot.py"))},
        "torch_version": torch.__version__, "numpy_version": np.__version__,
        "limitations": ["Exploratory sensitivity analysis with three shifts, not exact paper-number reproduction.",
                        "Latent truth is used only by frozen evaluation probes, never by autoencoder fitting.",
                        "No weight selection, no winner-only reporting, and no claim of transfer to discrete health data.",
                        "Cannot explain effects of kernel bandwidth, training duration or probe capacity because they are held fixed."]})
    rows, diagnostics, histories, checks = [], [], [], []
    prepared = {}
    # 任务五 第2步：先完成全部旧模型复核，再开始任何新增训练。
    for trial, mu in enumerate(means):
        data = make_data(6000, 2000, trial, float(mu))
        prepared[trial] = data
        train_x, test_x, train_z, test_z, _, _ = data
        for weight, arm in ((0.0, ARMS[0]), (1.0, ARMS[1])):
            print(f"任务五 第2步：复核旧模型 trial={trial}, lambda={weight}", flush=True)
            path = args.history / f"trial_{trial}_{arm}.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            expected_info = {"trial": trial, "shifted_mean": float(mu), "arm": arm}
            if checkpoint["info"] != expected_info:
                raise ValueError("历史权重的试验信息不一致。")
            model = build_autoencoder()
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            info = {**expected_info, "mmd_weight": weight, "origin": "reused_frozen_step1"}
            result, diag = diagnose(model, train_x, test_x, train_z, test_z, info)
            checks.extend(compare_recovery(old, result, trial, arm))
            rows.extend(result)
            diagnostics.extend(diag)
    pd.DataFrame(checks).to_csv(args.output / "historical_reproduction_checks.csv", index=False, encoding="utf-8-sig")
    init_checks = []
    for trial, mu in enumerate(means):
        train_x, test_x, train_z, test_z, _, _ = prepared[trial]
        hashes = []
        for weight in (0.1, 10.0):
            arm = f"mmd_{weight:g}"
            print(f"任务五 第2步：新增训练 trial={trial}, lambda={weight}", flush=True)
            model, history, initial_hash = fit_model(train_x, arm, 100, 256, SEED+trial, mmd_weight=weight)
            hashes.append(initial_hash)
            info = {"trial": trial, "shifted_mean": float(mu), "arm": arm,
                    "mmd_weight": weight, "origin": "new_training_step2"}
            result, diag = diagnose(model, train_x, test_x, train_z, test_z, info)
            rows.extend(result)
            diagnostics.extend(diag)
            histories.extend([{**info, **record} for record in history])
            torch.save({"state_dict": model.state_dict(), "info": info,
                        "initial_weights_sha256": initial_hash}, args.output / f"trial_{trial}_{arm}.pt")
            for name, content in (("latent_recovery.csv", rows), ("representation_diagnostics.csv", diagnostics),
                                  ("new_training_history.csv", histories)):
                pd.DataFrame(content).to_csv(args.output / name, index=False, encoding="utf-8-sig")
        if len(set(hashes)) != 1:
            raise RuntimeError("新增两组的初始权重不同。")
        init_checks.append({"trial": trial, "same_new_arm_initial_weights": True,
                            "initial_weights_sha256": hashes[0], "passed": True})
    result, diag = pd.DataFrame(rows), pd.DataFrame(diagnostics)
    if len(result) != 144 or len(diag) != 24 or len(checks) != 72 or not np.isfinite(result.test_r2.to_numpy()).all():
        raise RuntimeError("结果数量不完整或存在非有限值。")
    pd.DataFrame(init_checks).to_csv(args.output / "initialization_checks.csv", index=False, encoding="utf-8-sig")
    result.groupby(["mmd_weight", "representation", "latent"]).test_r2.agg(["mean", "min", "max"]).reset_index().to_csv(
        args.output / "latent_recovery_summary.csv", index=False, encoding="utf-8-sig")
    draw_figures(result, diag, args.output)
    write_json(args.output / "completion.json", {"completed": True, "new_training_runs": 6,
                                                 "historical_models_verified": 6, "weights": list(WEIGHTS)})
    print(f"任务五 第2步完成，请提交整个目录：{args.output}")


if __name__ == "__main__":
    main()

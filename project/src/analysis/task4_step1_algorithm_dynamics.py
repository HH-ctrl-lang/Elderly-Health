"""任务四 第一步：RDM、VREx、GroupDRO训练机制与跨地区表现对照。

本脚本只读取已完成的MLP、V5、RDM、VREx、GroupDRO历史配置、
training.json和metrics.json，不训练模型、不重新计算个人预测。

研究目标：
- 对照三种领域泛化算法的优化目标和实际训练轨迹；
- 检查正则/风险权重是否真正激活、是否出现环境权重极端集中；
- 比较三个留出地区的总体与监测点表现；
- 将算法差异与任务三已发现的共同失败站点联系起来。

解释边界：
- 当前结果只有一个随机种子；
- 三种算法的优化目标、部分学习率和batch策略不同；
- 本分析用于理解算法行为，不作为严格的算法优劣定论。
"""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from elderly_health.data import FEATURES

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parent
DATA_PATH = PROJECT / "data" / "data_mission2.csv"
OUTPUT_DIR = ROOT / "outputs" / "analysis_2611" / "task4_step1_algorithm_dynamics"

EXPECTED_DATA_SHA256 = (
    "969f0c1c82faafcfa402f37572e4a8ceb324492062cf1e590465f82dc6b5b6de"
)
REGIONS = ("east", "central", "west")
MODELS = ("MLP", "V5", "RDM", "VREx", "GroupDRO")

MODEL_PATHS = {
    "MLP": ("v2-region-loro-selection91-42", "baseline"),
    "V5": ("v2-region-loro-selection91-42", "v5"),
    "RDM": ("external-rdm-region-loro-42", None),
    "VREx": ("external-vrex-region-loro-42", None),
    "GroupDRO": ("external-groupdro-region-loro-42", None),
}

METRIC_COLUMNS = (
    "auc", "auprc", "brier", "log_loss",
    "balanced_accuracy", "f1", "precision", "recall",
)


# ==================== 一、基础函数 ====================

def sha256_file(path):
    """任务四 第一步：计算输入文件SHA256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def model_directory(model, region):
    root, subdirectory = MODEL_PATHS[model]
    base = ROOT / "outputs" / root / region
    return base / subdirectory if subdirectory else base


def finite_or_nan(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


# ==================== 二、读取和核验历史结果 ====================

def load_one_model_region(model, region):
    """任务四 第一步：读取一个模型/留出地区的配置、训练历史和评价结果。"""
    folder = model_directory(model, region)
    config_path = folder / "config.json"
    training_path = folder / "training.json"
    metrics_path = folder / "metrics.json"

    for path in (config_path, training_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(f"缺少历史结果文件：{path}")

    config_record = read_json(config_path)
    training = read_json(training_path)
    evaluation = read_json(metrics_path)

    config = config_record.get("config", config_record)
    data_hash = (
        config_record.get("input_sha256")
        or config_record.get("input_sha256")
        or config_record.get("input_sha256")
    )
    if data_hash is None:
        data_hash = config_record.get("input_sha256")

    history = training.get("history", [])
    best_epoch = int(training["best_epoch"])
    best_record = next(
        (row for row in history if int(row.get("epoch", -1)) == best_epoch),
        None,
    )
    if best_record is None:
        raise ValueError(f"{model}/{region}找不到最佳epoch记录。")

    main_metrics = evaluation.get("main")
    if not isinstance(main_metrics, dict):
        raise ValueError(f"{model}/{region} metrics.json缺少main指标。")

    return {
        "model": model,
        "region": region,
        "folder": folder,
        "config_record": config_record,
        "config": config,
        "training": training,
        "evaluation": evaluation,
        "best_record": best_record,
        "history": history,
        "data_hash": data_hash,
    }


# ==================== 三、提取总体指标与训练机制 ====================

def build_metric_rows(records):
    """任务四 第一步：提取每个模型/地区的总体测试集指标。"""
    rows = []
    site_rows = []

    for record in records:
        model = record["model"]
        region = record["region"]
        main = record["evaluation"]["main"]

        row = {
            "model": model,
            "held_out_region": region,
            "n": int(main["n"]),
            "positive_rate": float(main["positive_rate"]),
            "threshold": finite_or_nan(main.get("threshold")),
        }
        for metric in METRIC_COLUMNS:
            row[metric] = finite_or_nan(main.get(metric))
        rows.append(row)

        per_site = record["evaluation"].get("per_site", {})
        for site, values in per_site.items():
            item = {
                "model": model,
                "held_out_region": region,
                "site_code": str(site),
                "n": int(values["n"]),
                "positive_rate": float(values["positive_rate"]),
            }
            for metric in METRIC_COLUMNS:
                item[metric] = finite_or_nan(values.get(metric))
            site_rows.append(item)

    return pd.DataFrame(rows), pd.DataFrame(site_rows)


def build_training_rows(records):
    """任务四 第一步：提取最佳轮次的算法特有训练状态。"""
    rows = []
    group_rows = []
    gate_rows = []

    for record in records:
        model = record["model"]
        region = record["region"]
        config = record["config"]
        training = record["training"]
        best = record["best_record"]
        history = record["history"]

        row = {
            "model": model,
            "held_out_region": region,
            "best_epoch": int(training["best_epoch"]),
            "best_validation_auc": finite_or_nan(training.get("best_val_auc")),
            "last_epoch": int(history[-1]["epoch"]) if history else np.nan,
            "history_length": len(history),
            "selection_start_epoch": config.get("selection_start_epoch"),
            "seed": config.get("seed"),
            "device": config.get("device"),
            "learning_rate": config.get("learning_rate"),
            "batch_size": config.get("batch_size"),
            "per_environment_batch_size": config.get("per_environment_batch_size"),
            "positive_weight": config.get("positive_weight", config.get("pos_weight")),
            "best_updates": best.get("updates"),
        }

        if model == "RDM":
            weight = finite_or_nan(best.get("matching_weight"))
            matching_penalty = finite_or_nan(best.get("matching_penalty"))
            variance_penalty = finite_or_nan(best.get("variance_penalty"))
            variance_weight = finite_or_nan(config.get("variance_weight", 0.0))
            row.update({
                "erm_loss_at_best": finite_or_nan(best.get("erm_loss")),
                "matching_penalty_at_best": matching_penalty,
                "matching_weight_at_best": weight,
                "matching_contribution_at_best": weight * matching_penalty,
                "variance_penalty_at_best": variance_penalty,
                "variance_weight": variance_weight,
                "variance_contribution_at_best": variance_weight * variance_penalty,
                "penalty_anneal_updates": config.get("rdm_penalty_anneal_iters"),
                "training_domains": ";".join(training.get("training_domains", [])),
            })

        elif model == "VREx":
            variance = finite_or_nan(best.get("risk_variance"))
            anneal_steps = int(config.get("penalty_anneal_steps", 0))
            updates = int(best.get("updates", 0))
            lambda_value = float(config.get("vrex_lambda", 10.0))
            active_weight = lambda_value if updates >= anneal_steps else 1.0
            row.update({
                "mean_domain_risk_at_best": finite_or_nan(best.get("mean_risk")),
                "domain_risk_variance_at_best": variance,
                "effective_penalty_weight_at_best": active_weight,
                "variance_penalty_contribution_at_best": active_weight * variance,
                "penalty_anneal_updates": anneal_steps,
                "training_domains": ";".join(training.get("training_domains", [])),
            })

        elif model == "GroupDRO":
            weights = best.get("group_weights", training.get("selected_group_weights", []))
            groups = training.get("training_groups", [])
            row.update({
                "robust_loss_at_best": finite_or_nan(best.get("robust_loss")),
                "mean_group_loss_at_best": finite_or_nan(best.get("mean_group_loss")),
                "worst_group_loss_at_best": finite_or_nan(best.get("worst_group_loss")),
                "max_group_weight_at_best": max(weights) if weights else np.nan,
                "min_group_weight_at_best": min(weights) if weights else np.nan,
                "training_domains": ";".join(groups),
            })
            for group, weight in zip(groups, weights):
                group_rows.append({
                    "held_out_region": region,
                    "group": group,
                    "selected_group_weight": float(weight),
                    "model": model,
                })

        elif model == "V5":
            mask = np.asarray(best.get("mask", []), dtype=float)
            row.update({
                "gate_mean_at_best": float(np.mean(mask)) if mask.size else np.nan,
                "gate_std_at_best": float(np.std(mask)) if mask.size else np.nan,
                "gate_min_at_best": float(np.min(mask)) if mask.size else np.nan,
                "gate_max_at_best": float(np.max(mask)) if mask.size else np.nan,
                "gate_below_045_count": int(np.sum(mask < 0.45)) if mask.size else np.nan,
                "gate_above_055_count": int(np.sum(mask > 0.55)) if mask.size else np.nan,
                "train_sparse_at_best": finite_or_nan(best.get("train_sparse")),
                "train_irm_at_best": finite_or_nan(best.get("train_irm")),
                "train_environment_at_best": finite_or_nan(best.get("train_environment")),
                "train_independence_at_best": finite_or_nan(best.get("train_independence")),
            })
            for index, score in enumerate(mask):
                gate_rows.append({
                    "held_out_region": region,
                    "feature_index": index,
                    "feature": FEATURES[index],
                    "gate_score": float(score),
                })

        elif model == "MLP":
            row.update({
                "train_prediction_at_best": finite_or_nan(best.get("train_prediction")),
                "selection_start_epoch": config.get("selection_start_epoch"),
            })

        rows.append(row)

    return (
        pd.DataFrame(rows),
        pd.DataFrame(group_rows),
        pd.DataFrame(gate_rows),
    )


def build_pairwise_deltas(metrics_table):
    """任务四 第一步：计算各方法相对同地区MLP的性能差值。"""
    rows = []
    for region in REGIONS:
        base = metrics_table.loc[
            (metrics_table["held_out_region"] == region)
            & (metrics_table["model"] == "MLP")
        ]
        if len(base) != 1:
            raise ValueError(f"{region}缺少唯一MLP基线。")
        baseline = base.iloc[0]

        for model in MODELS:
            if model == "MLP":
                continue
            current = metrics_table.loc[
                (metrics_table["held_out_region"] == region)
                & (metrics_table["model"] == model)
            ]
            if len(current) != 1:
                raise ValueError(f"{region}/{model}指标不唯一。")
            row = {
                "held_out_region": region,
                "model": model,
                "baseline_model": "MLP",
            }
            for metric in METRIC_COLUMNS:
                row[f"delta_{metric}"] = (
                    float(current.iloc[0][metric]) - float(baseline[metric])
                )
            rows.append(row)
    return pd.DataFrame(rows)


# ==================== 四、绘图 ====================

def draw_comparison_figures(metrics_table, training_table, output):
    """任务四 第一步：生成地区表现与算法机制图。"""
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    for metric in ("auc", "auprc", "brier"):
        pivot = metrics_table.pivot(
            index="model",
            columns="held_out_region",
            values=metric,
        ).reindex(MODELS)

        fig, ax = plt.subplots(figsize=(9, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_xlabel("Algorithm")
        ax.set_ylabel(metric.upper())
        if metric in ("auc", "auprc"):
            ax.set_ylim(0, 1)
        ax.set_title(f"Leave-one-region-out {metric.upper()}")
        ax.legend(title="Held-out region")
        fig.tight_layout()
        fig.savefig(
            figure_dir / f"region_{metric}_comparison.png",
            dpi=250,
            bbox_inches="tight",
        )
        plt.close(fig)

    # 任务四 第一步：不同算法的优化量纲不同，分别绘图，不共用纵轴。
    plot_specs = [
        ("RDM", "matching_contribution_at_best", "MMD matching penalty contribution",
         "rdm_matching_penalty_contribution.png"),
        ("VREx", "variance_penalty_contribution_at_best", "Risk variance penalty contribution",
         "vrex_variance_penalty_contribution.png"),
        ("GroupDRO", "max_group_weight_at_best", "Maximum selected source-group weight",
         "groupdro_max_group_weight.png"),
        ("V5", "gate_mean_at_best", "Mean V5 gate score",
         "v5_gate_mean_by_region.png"),
    ]
    for model, column, ylabel, filename in plot_specs:
        subset = training_table.loc[
            training_table["model"] == model
        ].sort_values("held_out_region")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(
            subset["held_out_region"],
            subset[column],
            marker="o",
        )
        ax.set_xlabel("Held-out region")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{model}: {ylabel}")
        fig.tight_layout()
        fig.savefig(
            figure_dir / filename,
            dpi=250,
            bbox_inches="tight",
        )
        plt.close(fig)


# ==================== 五、主程序 ====================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(f"缺少数据文件：{args.data}")
    if args.output.exists():
        raise FileExistsError(f"固定输出目录已存在，禁止覆盖：{args.output}")

    data_hash = sha256_file(args.data)
    if data_hash != EXPECTED_DATA_SHA256:
        raise RuntimeError("数据SHA256与历史LORO实验不一致，停止汇总。")

    records = []
    for model in MODELS:
        for region in REGIONS:
            records.append(load_one_model_region(model, region))

    hashes = {
        record["data_hash"]
        for record in records
        if record["data_hash"] is not None
    }
    if hashes and hashes != {EXPECTED_DATA_SHA256}:
        raise RuntimeError(f"历史模型数据版本不一致：{hashes}")

    metrics_table, site_table = build_metric_rows(records)
    training_table, group_table, gate_table = build_training_rows(records)
    deltas = build_pairwise_deltas(metrics_table)

    if len(metrics_table) != 15 or len(training_table) != 15:
        raise RuntimeError("期望15条地区级模型记录。")
    if len(site_table) != 60:
        raise RuntimeError(f"期望60条监测点记录，实际{len(site_table)}。")

    args.output.mkdir(parents=True, exist_ok=False)

    metrics_table.to_csv(
        args.output / "algorithm_region_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    site_table.to_csv(
        args.output / "algorithm_site_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    training_table.to_csv(
        args.output / "algorithm_training_dynamics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    deltas.to_csv(
        args.output / "algorithm_deltas_vs_mlp.csv",
        index=False,
        encoding="utf-8-sig",
    )
    group_table.to_csv(
        args.output / "groupdro_selected_weights.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gate_table.to_csv(
        args.output / "v5_gate_scores_at_best_epoch.csv",
        index=False,
        encoding="utf-8-sig",
    )

    draw_comparison_figures(metrics_table, training_table, args.output)

    metadata = {
        "task": "Task 4 Step 1 - Algorithm Dynamics Comparison",
        "data_sha256": data_hash,
        "models": list(MODELS),
        "regions": list(REGIONS),
        "training_runs_repeated": False,
        "historical_metrics_and_logs_reused": True,
        "random_seed": 42,
        "selection_rule": (
            "Use each historical run's saved best checkpoint selected by source validation AUC."
        ),
        "important_comparability_limits": [
            "Only one random seed is available.",
            "RDM, VREx and GroupDRO optimize different objectives and have algorithm-specific hyperparameters.",
            "MLP/V5 use the project training pipeline; external algorithms use per-environment minibatches.",
            "All three external methods are tabular binary adaptations of published algorithms, not exact reproductions of original image benchmarks.",
            "Test metrics are reused historical results; no new test-based tuning is performed.",
        ],
        "algorithm_objectives": {
            "RDM": "ERM plus multiscale Gaussian-kernel MMD matching the worst-mean-risk environment loss distribution to pooled losses; penalty activates after 1500 updates.",
            "VREx": "Mean source-domain risk plus lambda times variance of domain risks; penalty weight changes from 1 to 10 after 500 updates.",
            "GroupDRO": "Multiplicative adversarial update of source-group weights; robust loss is the weighted sum of group risks.",
        },
        "outputs_are_aggregate": True,
    }
    write_json(args.output / "experiment_metadata.json", metadata)

    print("=" * 72)
    print("任务四 第一步：算法训练机制与跨地区表现汇总完成")
    print(f"结果目录：{args.output}")
    print("\n地区级测试AUC：")
    print(
        metrics_table.pivot(
            index="model", columns="held_out_region", values="auc"
        ).reindex(MODELS).to_string()
    )
    print("\n算法最佳轮次的优化状态：")
    print(training_table[
        [
            "model",
            "held_out_region",
            "best_epoch",
            "best_validation_auc",
            "matching_contribution_at_best",
            "variance_penalty_contribution_at_best",
            "max_group_weight_at_best",
            "gate_mean_at_best",
        ]
    ].to_string(index=False))


if __name__ == "__main__":
    main()

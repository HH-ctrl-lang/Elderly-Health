"""任务二：实验一补充诊断（不重新训练）。

研究问题：
1. 分布加权对跨地区 AUC / AUPRC / Brier 的改善是否稳健？
2. 东部模型在0.5阈值下召回率为0，是风险排序失效还是概率尺度/阈值不适用？
3. 分布加权能否改善校准及监测点间表现？

所有预测模型参数均从已有实验读取；只对已保存的加权MLP进行验证集推理，
绝不重新训练。目标地区健康标签仅用于最终评价和事后不确定性分析。
源地区验证集用于选择探索性阈值，不能在目标地区测试集上调阈值。
"""

import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "project"
sys.path.insert(0, str(PROJECT / "src"))

from elderly_health.data import (  # noqa: E402
    REGION,
    attach_regions,
    held_out_split,
    prepare_splits,
    read_raw,
)
from elderly_health.evaluation import metrics  # noqa: E402
from elderly_health.models import SimpleMLP  # noqa: E402
from elderly_health.training import predict  # noqa: E402


# ==================== 一、路径和复现实验参数 ====================

DATA_PATH = PROJECT / "data" / "data_mission2.csv"
HISTORICAL_ROOT = ROOT / "outputs" / "v2-region-loro-selection91-42"
WEIGHTED_ROOT = (
    ROOT / "outputs" / "analysis_2611" / "experiment1b_covariate_reweighting"
)
OUTPUT_DIR = (
    ROOT / "outputs" / "analysis_2611" / "experiment1c_prediction_diagnostics"
)

REGIONS = ("east", "central", "west")
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20261104
N_CALIBRATION_BINS = 8


# ==================== 二、输入验证 ====================

def sha256_file(path):
    """任务二，补充诊断：校验输入数据版本。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    """任务二，补充诊断：读取实验元数据。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    """任务二，补充诊断：保存复现信息。"""
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def verify_predictions(frame, split, region):
    """任务二，补充诊断：核对保存的目标样本、标签和监测点。"""
    required = {
        "sample_id",
        "site_code",
        "cognitive_impairment",
        "probability_unweighted_reused",
        "probability_weighted",
    }
    if not required.issubset(frame.columns):
        raise ValueError(
            f"{region}目标预测缺少字段：{sorted(required - set(frame.columns))}"
        )

    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{region}目标预测存在重复sample_id。")

    aligned = frame.set_index("sample_id").reindex(
        split.sample_ids.astype(np.int64)
    )

    if aligned[["site_code", "cognitive_impairment"]].isna().any().any():
        raise ValueError(f"{region}目标预测缺少样本或标签。")

    if len(frame) != len(split.sample_ids) or not np.array_equal(
        aligned.index.to_numpy(dtype=np.int64),
        split.sample_ids.astype(np.int64),
    ):
        raise ValueError(f"{region}目标预测行数或样本顺序不匹配。")

    if not np.array_equal(
        aligned["cognitive_impairment"].astype(int).to_numpy(),
        split.y.astype(int),
    ):
        raise ValueError(f"{region}目标预测健康结局与输入数据不一致。")

    codes = aligned["site_code"].astype(str).str.replace(
        r"\.0$", "", regex=True
    ).to_numpy()

    if not np.array_equal(codes, split.site_env.astype(str)):
        raise ValueError(f"{region}目标预测监测点与输入数据不一致。")

    for key in ("probability_unweighted_reused", "probability_weighted"):
        values = aligned[key].to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"{region}目标预测概率无效：{key}")

    return aligned.reset_index()


def load_and_align_baseline_validation(region, val_split, historical_root):
    """任务二，补充诊断：复用历史未加权MLP源域验证预测。"""
    path = historical_root / region / "baseline" / "representations_val.csv"
    val = pd.read_csv(path, encoding="utf-8-sig")

    if val["sample_id"].duplicated().any() or len(val) != len(val_split.y):
        raise ValueError(f"{region}历史验证集预测行数/ID异常。")

    if not (val["split"].astype(str) == "val").all():
        raise ValueError(f"{region}历史验证预测包含非验证样本。")

    aligned = val.set_index("sample_id").reindex(
        val_split.sample_ids.astype(np.int64)
    )

    if aligned["y_prob"].isna().any():
        raise ValueError(f"{region}历史验证集缺少样本预测。")

    if not np.array_equal(
        aligned["cognitive_impairment"].astype(int).to_numpy(),
        val_split.y.astype(int),
    ):
        raise ValueError(f"{region}历史验证集标签不匹配。")

    sites = aligned["environment_code"].astype(str).str.replace(
        r"\.0$", "", regex=True
    ).to_numpy()

    if not np.array_equal(sites, val_split.site_env.astype(str)):
        raise ValueError(f"{region}历史验证集站点不匹配。")

    return aligned["y_prob"].to_numpy(dtype=float)


def weighted_validation_predictions(
    region, val_split, test_split, saved_test, weighted_root
):
    """任务二，补充诊断：加载既有checkpoint，仅在源域验证集推理。

    同时核对目标地区推理与已提交的保存预测一致，以验证预处理、
    网络结构、checkpoint和样本顺序没有发生变化。
    """
    checkpoint_path = weighted_root / region / "model_weighted.pt"
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model = SimpleMLP(val_split.x.shape[1])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    observed_test = saved_test["probability_weighted"].to_numpy(dtype=float)
    reconstructed_test = predict(model, test_split.x, torch.device("cpu"))

    max_error = float(np.max(np.abs(observed_test - reconstructed_test)))
    if max_error > 2e-5:
        raise RuntimeError(
            f"{region}权重模型重新推理与已保存目标预测不一致，"
            f"最大绝对误差={max_error:.8f}。"
            "请勿在未确认checkpoint与预处理一致前继续。"
        )

    predictions = predict(model, val_split.x, torch.device("cpu"))
    return predictions, max_error, int(checkpoint["best_epoch"])


# ==================== 三、阈值与校准 ====================

def choose_source_validation_threshold(y, probabilities):
    """任务二，补充诊断：只使用源域验证集选择探索性Youden阈值。"""
    if len(np.unique(y)) != 2:
        raise ValueError("源域验证集不同时包含两类，无法选阈值。")

    fpr, tpr, thresholds = roc_curve(
        np.asarray(y, dtype=int),
        np.asarray(probabilities, dtype=float),
    )

    # 任务二，补充诊断：排除ROC起点的无穷大阈值。
    finite = np.isfinite(thresholds)
    candidate = np.flatnonzero(finite)
    if len(candidate) == 0:
        raise RuntimeError("源域验证集没有可用的有限阈值。")

    # 任务二，补充诊断：最大化灵敏度+特异度-1，同分时使用较高阈值。
    youden = tpr - fpr
    index = sorted(
        candidate,
        key=lambda i: (youden[i], thresholds[i]),
        reverse=True,
    )[0]
    return float(thresholds[index]), float(youden[index])


def calibration_bins(y, scores, region, model_name):
    """任务二，补充诊断：采用等频分组，描述性评价校准误差。"""
    data = pd.DataFrame(
        {"outcome": np.asarray(y, dtype=int), "score": np.asarray(scores)}
    )
    data["bin"] = pd.qcut(
        data["score"],
        q=N_CALIBRATION_BINS,
        duplicates="drop",
    )
    grouped = (
        data.groupby("bin", observed=True)
        .agg(
            n=("outcome", "size"),
            mean_predicted=("score", "mean"),
            observed_rate=("outcome", "mean"),
        )
        .reset_index(drop=True)
    )
    grouped["absolute_calibration_gap"] = (
        grouped["mean_predicted"] - grouped["observed_rate"]
    ).abs()
    grouped["region"] = region
    grouped["model"] = model_name

    ece = float(
        (
            grouped["n"] / len(y) * grouped["absolute_calibration_gap"]
        ).sum()
    )
    return grouped, ece


# ==================== 四、配对重抽样 ====================

def three_metric_deltas(y, p_old, p_new):
    """任务二，补充诊断：计算加权减未加权指标差值。"""
    if len(np.unique(y)) < 2:
        raise ValueError("重抽样中健康结局只有一个类别。")

    return {
        "auc_delta": (
            roc_auc_score(y, p_new) - roc_auc_score(y, p_old)
        ),
        "auprc_delta": (
            average_precision_score(y, p_new)
            - average_precision_score(y, p_old)
        ),
        "brier_delta": (
            brier_score_loss(y, p_new) - brier_score_loss(y, p_old)
        ),
    }


def paired_hierarchical_bootstrap(y, p_old, p_new, sites, seed):
    """任务二，补充诊断：按站点重抽样并在站点内重抽样个体。

    仅4个监测点，区间只能用于探索性稳定性诊断，不能被解释为
    独立监测点充足时的严格总体置信区间或显著性检验。
    """
    rng = np.random.default_rng(seed)
    sites = np.asarray(sites).astype(str)
    site_names = np.unique(sites)
    site_indices = {
        site: np.flatnonzero(sites == site)
        for site in site_names
    }

    observed = three_metric_deltas(y, p_old, p_new)
    records = []

    for iteration in range(BOOTSTRAP_N):
        chosen_sites = rng.choice(
            site_names, size=len(site_names), replace=True
        )
        sample_blocks = []
        for site in chosen_sites:
            indices = site_indices[site]
            sample_blocks.append(
                rng.choice(indices, size=len(indices), replace=True)
            )

        indices = np.concatenate(sample_blocks)
        if np.unique(y[indices]).size != 2:
            continue

        records.append(three_metric_deltas(
            y[indices], p_old[indices], p_new[indices]
        ))

    if len(records) < BOOTSTRAP_N * 0.90:
        raise RuntimeError("配对Bootstrap有效重抽样过少。")

    rows = []
    for key in ("auc_delta", "auprc_delta", "brier_delta"):
        values = np.asarray([record[key] for record in records])
        low, high = np.quantile(values, [0.025, 0.975])
        rows.append(
            {
                "metric": key,
                "observed_delta": float(observed[key]),
                "bootstrap_ci_low": float(low),
                "bootstrap_ci_high": float(high),
                "fraction_positive": float((values > 0).mean()),
                "valid_bootstrap_replicates": len(records),
                "bootstrap_sites": len(site_names),
            }
        )
    return pd.DataFrame(rows)


# ==================== 五、绘图与主程序 ====================

def plot_reliability(region, data, output_dir):
    """任务二，补充诊断：绘制目标地区校准曲线。"""
    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, group in data.groupby("model", sort=False):
        ax.plot(
            group["mean_predicted"],
            group["observed_rate"],
            marker="o",
            label=model_name,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted probability",
        ylabel="Observed event rate",
        title=f"Calibration: held-out {region}",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"calibration_{region}.png", dpi=250)
    plt.close(fig)


def main():
    """任务二，补充诊断：复用已提交结果，不重新训练任何模型。"""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--historical", type=Path, default=HISTORICAL_ROOT)
    parser.add_argument("--weighted", type=Path, default=WEIGHTED_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    historical_root = args.historical
    weighted_root = args.weighted

    if args.output.exists():
        raise FileExistsError(
            f"固定实验输出目录已存在，为避免覆盖历史结果，请检查：{args.output}"
        )

    metadata = load_json(
        weighted_root / "experiment_metadata.json"
    )
    actual_hash = sha256_file(args.data)
    if metadata["input_sha256"] != actual_hash:
        raise RuntimeError("输入数据SHA256与加权实验不一致。")

    if metadata.get("baseline_retrained") is not False:
        raise RuntimeError("加权实验元数据未声明复用历史baseline。")

    if metadata.get("target_outcome_used_for_weight_estimation") is not False:
        raise RuntimeError("目标Y参与了权重估计，拒绝开展无监督适配诊断。")

    if metadata.get("model_config", {}).get("seed") != 42:
        raise RuntimeError("当前仅支持历史LORO seed=42 配对诊断。")

    frame = attach_regions(read_raw(args.data, encoding="gbk"))

    bootstrap_rows = []
    threshold_rows = []
    threshold_metric_rows = []
    calibration_rows = []
    summary_rows = []

    # 任务二，补充诊断：所有校验均通过后才创建输出目录。
    # 以下步骤仅在内存计算统计量、进行已有模型推理，不训练。
    prepared = {}

    for idx, region in enumerate(REGIONS):
        train_frame, val_frame, target_frame = held_out_split(
            frame, region, seed=42, column=REGION
        )
        (train_split, val_split, test_split), _ = prepare_splits(
            [train_frame, val_frame, target_frame],
            environment_column=REGION,
        )

        saved = pd.read_csv(
            weighted_root / region / "target_predictions.csv",
            encoding="utf-8-sig",
        )
        target = verify_predictions(saved, test_split, region)

        y_test = test_split.y.astype(int)
        old_test = target["probability_unweighted_reused"].to_numpy(dtype=float)
        new_test = target["probability_weighted"].to_numpy(dtype=float)

        old_val = load_and_align_baseline_validation(
            region, val_split, historical_root
        )
        new_val, consistency_error, best_epoch = weighted_validation_predictions(
            region, val_split, test_split, target, weighted_root
        )

        t_old, j_old = choose_source_validation_threshold(
            val_split.y.astype(int), old_val
        )
        t_new, j_new = choose_source_validation_threshold(
            val_split.y.astype(int), new_val
        )

        # 任务二，补充诊断：在每个模型相同的源地区验证集选阈值。
        for name, val_score, test_score, threshold, youden in (
            ("MLP_unweighted", old_val, old_test, t_old, j_old),
            ("MLP_weighted", new_val, new_test, t_new, j_new),
        ):
            threshold_rows.append(
                {
                    "region": region,
                    "model": name,
                    "selection_data": "source_validation",
                    "criterion": "Youden_J",
                    "selected_threshold": threshold,
                    "source_validation_youden_j": youden,
                    "source_validation_auc": roc_auc_score(
                        val_split.y.astype(int), val_score
                    ),
                    "target_labels_used_for_selection": False,
                }
            )
            for threshold_name, value in (
                ("fixed_0.5", 0.5),
                ("source_validation_youden", threshold),
            ):
                threshold_metric_rows.append(
                    {
                        "region": region,
                        "model": name,
                        "threshold_rule": threshold_name,
                        **metrics(y_test, test_score, threshold=value),
                    }
                )

            bins, ece = calibration_bins(
                y_test, test_score, region, name
            )
            calibration_rows.extend(bins.to_dict(orient="records"))
            summary_rows.append(
                {
                    "region": region,
                    "model": name,
                    "target_n": len(y_test),
                    "target_positive_n": int(y_test.sum()),
                    "target_positive_rate": float(y_test.mean()),
                    "mean_predicted_probability": float(test_score.mean()),
                    "maximum_predicted_probability": float(test_score.max()),
                    "target_predictions_above_0_5": int(
                        (test_score > 0.5).sum()
                    ),
                    "ece_quantile_bins": ece,
                    "brier": brier_score_loss(y_test, test_score),
                    "weighted_checkpoint_best_epoch": (
                        best_epoch if name == "MLP_weighted" else np.nan
                    ),
                    "weighted_model_prediction_consistency_error": (
                        consistency_error if name == "MLP_weighted" else np.nan
                    ),
                }
            )

        boot = paired_hierarchical_bootstrap(
            y_test,
            old_test,
            new_test,
            test_split.site_env,
            BOOTSTRAP_SEED + idx,
        )
        boot.insert(0, "region", region)
        bootstrap_rows.extend(boot.to_dict(orient="records"))
        prepared[region] = True

    args.output.mkdir(parents=True, exist_ok=False)

    bootstrap_table = pd.DataFrame(bootstrap_rows)
    threshold_table = pd.DataFrame(threshold_rows)
    threshold_metrics = pd.DataFrame(threshold_metric_rows)
    calibration_table = pd.DataFrame(calibration_rows)
    summary_table = pd.DataFrame(summary_rows)

    for name, table in (
        ("paired_bootstrap_ci.csv", bootstrap_table),
        ("validation_selected_thresholds.csv", threshold_table),
        ("target_metrics_by_threshold.csv", threshold_metrics),
        ("calibration_bins.csv", calibration_table),
        ("diagnostic_summary.csv", summary_table),
    ):
        table.to_csv(
            args.output / name,
            index=False,
            encoding="utf-8-sig",
        )

    for region in REGIONS:
        plot_reliability(
            region,
            calibration_table.loc[calibration_table["region"] == region],
            args.output,
        )

    write_json(
        args.output / "experiment_metadata.json",
        {
            "experiment": "Task 2 Experiment 1C - Prediction Diagnostics",
            "source_data_sha256": actual_hash,
            "historical_baseline": str(historical_root.resolve()),
            "weighted_experiment": str(weighted_root.resolve()),
            "data_retrained": False,
            "model_retrained": False,
            "only_saved_weighted_model_inference": True,
            "threshold_selected_on": "source_validation_only",
            "threshold_selection": "Youden_J_exploratory",
            "target_outcome_used_for_threshold_selection": False,
            "target_outcome_used_for_evaluation": True,
            "paired_bootstrap": {
                "method": "two_stage_site_individual_with_replacement",
                "replicates": BOOTSTRAP_N,
                "seed": BOOTSTRAP_SEED,
                "note": (
                    "Only four sites per target region; intervals are exploratory "
                    "and not definitive hypothesis tests."
                ),
            },
            "calibration": {
                "method": "equal_frequency_bin_ECE",
                "bins": N_CALIBRATION_BINS,
                "note": "ECE is descriptive and sensitive to grouping.",
            },
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    )

    print("=" * 65)
    print("任务二：实验一补充诊断完成（没有重新训练模型）")
    print(f"结果位置：{args.output}")
    print("=" * 65)
    print("\n三个地区的配对Bootstrap结果：")
    print(bootstrap_table.to_string(index=False))
    print("\n源域阈值及目标区分类指标：")
    print(
        threshold_metrics[
            ["region", "model", "threshold_rule", "threshold", "recall",
             "precision", "f1", "balanced_accuracy"]
        ].to_string(index=False)
    )
    print("\n校准与预测概率摘要：")
    print(summary_table.to_string(index=False))


if __name__ == "__main__":
    main()

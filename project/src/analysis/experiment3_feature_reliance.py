"""任务二，实验三：模型特征依赖与跨监测点稳定性分析。

复用历史LORO训练完成的MLP baseline与V5因果解耦模型，不重新训练。
对每个留出地区中的每个监测点，分别置换原始19个输入特征，
经该模型保存的预处理器转换后进行预测。

输出：
1. 每个模型×监测点×特征的重复置换重要性；
2. 跨监测点特征依赖稳定性；
3. V5内部gate分数与置换重要性的对应关系；
4. 特征分布偏移与模型特征依赖之间的探索性关联。

解释边界：
- 置换重要性表示模型预测对输入信息的依赖，不是因果效应；
- 单特征置换可能破坏相关特征组合，手机变量另做联合置换；
- 12个监测点的相关性仅作探索性描述；
- 目标标签仅用于计算预测指标，不参与置换或模型选择。
"""

import hashlib
import json
import pickle
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

from elderly_health.config import TrainConfig  # noqa: E402
from elderly_health.data import (  # noqa: E402
    ENVIRONMENT,
    FEATURES,
    REGION,
    attach_regions,
    held_out_split,
    read_raw,
)
from elderly_health.evaluation import metrics  # noqa: E402
from elderly_health.models import CausalDisentangledModel, SimpleMLP  # noqa: E402
from elderly_health.training import get_device, predict  # noqa: E402


# ==================== 一、实验配置 ====================

DATA_PATH = PROJECT / "data" / "data_mission2.csv"
MODEL_ROOT = ROOT / "outputs" / "v2-region-loro-selection91-42"
SHIFT_DETAIL_PATH = (
    ROOT / "outputs" / "analysis_2611"
    / "experiment2_site_heterogeneity"
    / "site_feature_shift_detail.csv"
)
OUTPUT_DIR = (
    ROOT / "outputs" / "analysis_2611"
    / "experiment3_feature_reliance"
)

REGIONS = ("east", "central", "west")
MODELS = ("baseline", "v5")
N_PERMUTATIONS = 30
RANDOM_SEED = 20261105
PREDICTION_TOLERANCE = 2e-5

# 任务二，实验三：三个语义相关的手机使用变量另做联合置换。
JOINT_GROUPS = {
    "phone_use_bundle": (
        "if_use_mobile_phone",
        "using_phone_year",
        "hours_phone_per_day",
    )
}

METRIC_KEYS = ("auc", "auprc", "brier", "log_loss")


# ==================== 二、通用函数 ====================

def sha256_file(path):
    """任务二，实验三：计算输入文件SHA256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """任务二，实验三：保存实验配置和复现信息。"""
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def normalize_metrics(y, probability):
    """任务二，实验三：统一计算基准与置换后的预测指标。"""
    result = metrics(np.asarray(y, dtype=int), np.asarray(probability, dtype=float))
    return {key: result[key] for key in METRIC_KEYS}


def stable_seed(region, site, feature, repeat):
    """任务二，实验三：为每次置换生成跨模型一致的随机种子。"""
    token = f"{RANDOM_SEED}|{region}|{site}|{feature}|{repeat}".encode("utf-8")
    value = int(hashlib.sha256(token).hexdigest()[:8], 16)
    return value % (2**32 - 1)


def bh_adjust(p_values):
    """任务二，实验三：Benjamini-Hochberg多重检验校正。"""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p_values), np.nan)
    valid = np.flatnonzero(np.isfinite(p_values))
    if len(valid) == 0:
        return adjusted

    order = valid[np.argsort(p_values[valid])]
    total = len(order)
    running = 1.0

    for i in range(total - 1, -1, -1):
        rank = i + 1
        running = min(running, p_values[order[i]] * total / rank)
        adjusted[order[i]] = min(running, 1.0)

    return adjusted


# ==================== 三、读取原始数据和历史模型 ====================

def load_data(path):
    """任务二，实验三：读取原始数据并核验12个监测点。"""
    frame = attach_regions(read_raw(path, encoding="gbk"))
    site_table = frame[[ENVIRONMENT, REGION]].drop_duplicates()

    if site_table[ENVIRONMENT].duplicated().any():
        raise ValueError("同一监测点映射到了多个地区。")

    counts = site_table[REGION].value_counts().to_dict()
    if counts != {"east": 4, "central": 4, "west": 4}:
        raise ValueError(f"监测点构成不符合当前12站点设计：{counts}")

    return frame


def load_saved_model(model_dir, model_name, expected_data_hash):
    """任务二，实验三：从历史LORO目录加载已训练模型和预处理器。"""
    config_path = model_dir / "config.json"
    checkpoint_path = model_dir / "best_model.pt"
    preprocessor_path = model_dir / "preprocessor.pkl"
    training_path = model_dir / "training.json"

    for path in (config_path, checkpoint_path, preprocessor_path, training_path):
        if not path.is_file():
            raise FileNotFoundError(f"历史模型文件缺失：{path}")

    config_record = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if config_record.get("input_sha256") != expected_data_hash:
        raise RuntimeError(f"{model_dir}使用的数据版本与当前输入不一致。")

    if config_record.get("feature_names") != list(FEATURES):
        raise RuntimeError(f"{model_dir}的特征顺序与当前项目不一致。")

    if config_record.get("holdout_level") != "region":
        raise RuntimeError(f"{model_dir}不是留一地区实验模型。")

    if checkpoint.get("feature_names") != list(FEATURES):
        raise RuntimeError(f"{model_dir}checkpoint特征顺序不一致。")

    model_config = TrainConfig(**checkpoint["config"])

    if model_config.model != model_name:
        raise RuntimeError(
            f"{model_dir}模型类型不一致：预期{model_name}，"
            f"实际{model_config.model}。"
        )

    with preprocessor_path.open("rb") as stream:
        preprocessor = pickle.load(stream)

    if tuple(preprocessor.imputer.feature_names_in_) != tuple(FEATURES):
        raise RuntimeError(f"{model_dir}预处理器特征顺序不一致。")

    if model_name == "baseline":
        model = SimpleMLP(len(FEATURES))
    elif model_name == "v5":
        env_map = checkpoint.get("env_map")
        if not isinstance(env_map, dict) or len(env_map) != 2:
            raise RuntimeError(
                f"{model_dir}V5 checkpoint缺少预期的两个源地区环境映射。"
            )
        model = CausalDisentangledModel(
            input_dim=len(FEATURES),
            num_envs=len(env_map),
            latent_dim=model_config.latent_dim,
            gate_init_std=model_config.gate_init_std,
            gate_temperature=model_config.gate_temperature,
            joint=True,
        )
    else:
        raise ValueError(f"当前未实现的模型类型：{model_name}")

    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    device = get_device(model_config.device)
    model.to(device)

    training_record = json.loads(training_path.read_text(encoding="utf-8"))
    best_epoch = int(training_record["best_epoch"])

    gate_scores = None
    if model_name == "v5":
        history = training_record.get("history", [])
        selected = [
            item for item in history
            if int(item.get("epoch", -1)) == best_epoch
        ]
        if len(selected) != 1 or "mask" not in selected[0]:
            raise RuntimeError(
                f"{model_dir}找不到最佳epoch={best_epoch}对应的gate记录。"
            )
        gate_scores = np.asarray(selected[0]["mask"], dtype=float)
        if len(gate_scores) != len(FEATURES):
            raise RuntimeError("V5 gate长度与19个输入特征不一致。")

    return {
        "model": model,
        "preprocessor": preprocessor,
        "config": model_config,
        "device": device,
        "best_epoch": best_epoch,
        "gate_scores": gate_scores,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "preprocessor_sha256": sha256_file(preprocessor_path),
    }


def load_saved_predictions(model_dir, target_frame):
    """任务二，实验三：读取并按sample_id对齐历史测试集预测。"""
    path = model_dir / "representations_test.csv"
    if not path.is_file():
        raise FileNotFoundError(f"历史测试预测缺失：{path}")

    table = pd.read_csv(path, encoding="utf-8-sig")
    required = {"sample_id", "environment_code", "cognitive_impairment", "y_prob"}
    if not required.issubset(table.columns):
        raise ValueError(f"{path}缺少字段：{sorted(required - set(table.columns))}")

    if table["sample_id"].duplicated().any():
        raise ValueError(f"{path}存在重复sample_id。")

    expected_ids = target_frame["sample_id"].astype(np.int64).to_numpy()
    table = table.set_index("sample_id").reindex(expected_ids)

    if table["y_prob"].isna().any():
        raise RuntimeError(f"{path}未覆盖当前LORO测试样本。")

    if not np.array_equal(
        table["cognitive_impairment"].astype(int).to_numpy(),
        target_frame["label"].astype(int).to_numpy(),
    ):
        raise RuntimeError(f"{path}的测试标签与当前原始数据不一致。")

    actual_sites = (
        table["environment_code"].astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .to_numpy()
    )
    expected_sites = target_frame[ENVIRONMENT].astype(str).to_numpy()

    if not np.array_equal(actual_sites, expected_sites):
        raise RuntimeError(f"{path}的监测点与当前原始数据不一致。")

    return table["y_prob"].to_numpy(dtype=float)


# ==================== 四、基准预测核验 ====================

def load_fold_context(frame, region, model_name, model_root, expected_data_hash):
    """任务二，实验三：加载留出地区模型、预处理器和测试样本。"""
    train_frame, val_frame, target_frame = held_out_split(
        frame,
        region,
        seed=42,
        column=REGION,
    )

    model_dir = model_root / region / model_name
    artifacts = load_saved_model(model_dir, model_name, expected_data_hash)

    raw_x = target_frame[list(FEATURES)].copy()
    transformed_x = artifacts["preprocessor"].transform(raw_x)

    saved_probability = load_saved_predictions(model_dir, target_frame)
    recomputed_probability = predict(
        artifacts["model"],
        transformed_x,
        artifacts["device"],
    )

    max_error = float(
        np.max(np.abs(saved_probability - recomputed_probability))
    )

    if max_error > PREDICTION_TOLERANCE:
        raise RuntimeError(
            f"{region}/{model_name}基准预测无法复核，"
            f"最大概率误差={max_error:.8f}。"
        )

    return {
        **artifacts,
        "train_frame": train_frame,
        "validation_frame": val_frame,
        "target_frame": target_frame,
        "target_x": transformed_x,
        "baseline_probability": recomputed_probability,
        "prediction_max_abs_error": max_error,
        "model_dir": model_dir,
    }


# ==================== 五、单特征与联合置换 ====================

def make_permutation_groups():
    """任务二，实验三：建立19个单特征组及一个手机使用联合组。"""
    groups = [
        (feature, (feature,))
        for feature in FEATURES
    ]
    for group_name, members in JOINT_GROUPS.items():
        if not set(members).issubset(FEATURES):
            raise ValueError(f"联合置换组{group_name}包含未知特征。")
        groups.append((group_name, tuple(members)))
    return groups


def permute_raw_features(site_frame, members, seed):
    """任务二，实验三：在原始样本层面置换特征，再使用固定预处理器转换。"""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(site_frame))
    permuted = site_frame.copy()

    # 联合置换使用相同的行置换，保留组内变量的联合结构。
    for feature in members:
        permuted[feature] = site_frame[feature].to_numpy()[order]

    return permuted


def calculate_site_importance(region, site, model_name, context, n_permutations):
    """任务二，实验三：计算一个模型在一个监测点上的重复置换重要性。"""
    target = context["target_frame"]
    site_mask = target[ENVIRONMENT].astype(str).to_numpy() == str(site)
    site_frame = target.loc[site_mask].copy()

    if site_frame.empty:
        raise ValueError(f"{region}/{site}没有测试样本。")

    y = site_frame["label"].astype(int).to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError(
            f"{region}/{site}测试样本只有一个结局类别，"
            "无法计算AUC置换重要性。"
        )

    all_prob = context["baseline_probability"]
    base_prob = all_prob[site_mask]
    baseline_metrics = normalize_metrics(y, base_prob)

    groups = make_permutation_groups()
    repeat_rows = []

    for group_name, members in groups:
        for repeat in range(n_permutations):
            seed = stable_seed(region, site, group_name, repeat)
            permuted_raw = permute_raw_features(
                site_frame,
                members,
                seed,
            )

            permuted_x = context["preprocessor"].transform(
                permuted_raw[list(FEATURES)]
            )
            permuted_probability = predict(
                context["model"],
                permuted_x,
                context["device"],
            )
            permuted_metrics = normalize_metrics(y, permuted_probability)

            repeat_rows.append(
                {
                    "held_out_region": region,
                    "site_code": str(site),
                    "model": model_name,
                    "feature": group_name,
                    "feature_members": ";".join(members),
                    "repeat": repeat + 1,
                    "baseline_auc": baseline_metrics["auc"],
                    "permuted_auc": permuted_metrics["auc"],
                    "delta_auc": baseline_metrics["auc"] - permuted_metrics["auc"],
                    "baseline_auprc": baseline_metrics["auprc"],
                    "permuted_auprc": permuted_metrics["auprc"],
                    "delta_auprc": (
                        baseline_metrics["auprc"] - permuted_metrics["auprc"]
                    ),
                    "baseline_brier": baseline_metrics["brier"],
                    "permuted_brier": permuted_metrics["brier"],
                    "delta_brier": (
                        permuted_metrics["brier"] - baseline_metrics["brier"]
                    ),
                    "baseline_log_loss": baseline_metrics["log_loss"],
                    "permuted_log_loss": permuted_metrics["log_loss"],
                    "delta_log_loss": (
                        permuted_metrics["log_loss"] - baseline_metrics["log_loss"]
                    ),
                    "permutation_seed": seed,
                }
            )

    return baseline_metrics, repeat_rows


# ==================== 六、汇总和稳定性分析 ====================

def summarize_importance(repeat_table):
    """任务二，实验三：汇总置换均值、标准差及置换分布区间。"""
    metric_columns = [
        "delta_auc",
        "delta_auprc",
        "delta_brier",
        "delta_log_loss",
    ]

    grouped = repeat_table.groupby(
        ["held_out_region", "site_code", "model", "feature", "feature_members"],
        dropna=False,
    )

    rows = []
    for keys, group in grouped:
        row = dict(
            zip(
                ["held_out_region", "site_code", "model", "feature", "feature_members"],
                keys,
            )
        )
        row["n_permutations"] = len(group)

        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
            row[f"{metric}_q025"] = float(np.quantile(values, 0.025))
            row[f"{metric}_q975"] = float(np.quantile(values, 0.975))

        rows.append(row)

    return pd.DataFrame(rows)


def calculate_stability(summary):
    """任务二，实验三：计算每个模型的跨监测点特征依赖稳定性。"""
    individual = summary.loc[
        ~summary["feature"].isin(JOINT_GROUPS)
    ].copy()

    rows = []
    for model_name in MODELS:
        model_data = individual.loc[individual["model"] == model_name]

        for feature in FEATURES:
            values = model_data.loc[
                model_data["feature"] == feature,
                "delta_auc_mean",
            ].dropna().to_numpy(dtype=float)

            rows.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "site_count": len(values),
                    "mean_delta_auc": float(np.mean(values)) if len(values) else np.nan,
                    "std_delta_auc": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                    "median_delta_auc": float(np.median(values)) if len(values) else np.nan,
                    "positive_importance_site_count": int((values > 0).sum()),
                    "positive_importance_site_rate": (
                        float((values > 0).mean()) if len(values) else np.nan
                    ),
                    "mean_delta_auprc": float(
                        model_data.loc[
                            model_data["feature"] == feature,
                            "delta_auprc_mean",
                        ].mean()
                    ),
                    "mean_delta_brier": float(
                        model_data.loc[
                            model_data["feature"] == feature,
                            "delta_brier_mean",
                        ].mean()
                    ),
                }
            )

    stability = pd.DataFrame(rows)

    rank_rows = []
    for model_name in MODELS:
        pivot = individual.loc[
            individual["model"] == model_name
        ].pivot(
            index="site_code",
            columns="feature",
            values="delta_auc_mean",
        )

        correlations = []
        for site_a, site_b in combinations(pivot.index, 2):
            pair = pivot.loc[[site_a, site_b]].T.dropna()
            if len(pair) < 4:
                continue
            rho = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic
            if np.isfinite(rho):
                correlations.append(float(rho))

        rank_rows.append(
            {
                "model": model_name,
                "site_count": int(pivot.shape[0]),
                "pairwise_site_rank_correlations": len(correlations),
                "median_pairwise_spearman": (
                    float(np.median(correlations)) if correlations else np.nan
                ),
                "mean_pairwise_spearman": (
                    float(np.mean(correlations)) if correlations else np.nan
                ),
            }
        )

    return stability, pd.DataFrame(rank_rows)


def correlate_shift_and_reliance(summary, shift_path):
    """任务二，实验三：探索特征分布偏移与模型依赖是否对应。"""
    shift = pd.read_csv(shift_path, encoding="utf-8-sig")
    # 任务二，实验三：CSV可能把纯数字监测点代码推断为int64，
    # 而置换结果中的site_code为字符串；统一为六位代码字符串后再合并。
    shift["site_code"] = (
        pd.to_numeric(shift["site_code"], errors="raise")
        .astype("int64")
        .astype(str)
        .str.zfill(6)
    )
    required = {"site_code", "feature", "js_divergence_bits"}
    if not required.issubset(shift.columns):
        raise ValueError(
            f"实验二特征偏移表缺少字段：{sorted(required - set(shift.columns))}"
        )

    importance = summary.loc[
        summary["feature"].isin(FEATURES)
    ].copy()

    merged = importance.merge(
        shift[list(required)],
        on=["site_code", "feature"],
        how="inner",
        # 左表包含baseline和V5两种模型，因此每个site-feature有两行；
        # 右表每个site-feature只有一行，属于多对一合并。
        validate="many_to_one",
    )

    rows = []
    for model_name in MODELS:
        for outcome in ("delta_auc_mean", "delta_auprc_mean", "delta_brier_mean"):
            for feature in FEATURES:
                subset = merged.loc[
                    (merged["model"] == model_name)
                    & (merged["feature"] == feature),
                    ["js_divergence_bits", outcome],
                ].dropna()

                if (
                    len(subset) < 5
                    or subset["js_divergence_bits"].nunique() < 2
                    or subset[outcome].nunique() < 2
                ):
                    rho, p_value = np.nan, np.nan
                else:
                    result = spearmanr(
                        subset["js_divergence_bits"],
                        subset[outcome],
                    )
                    rho, p_value = float(result.statistic), float(result.pvalue)

                rows.append(
                    {
                        "model": model_name,
                        "feature": feature,
                        "shift_measure": "site_vs_loro_source_js",
                        "importance_measure": outcome,
                        "site_count": len(subset),
                        "spearman_rho": rho,
                        "nominal_p_value": p_value,
                        "interpretation": "exploratory_only_not_causal",
                    }
                )

    result_table = pd.DataFrame(rows)
    result_table["bh_q_within_model_metric"] = np.nan

    for model_name in MODELS:
        for outcome in ("delta_auc_mean", "delta_auprc_mean", "delta_brier_mean"):
            mask = (
                (result_table["model"] == model_name)
                & (result_table["importance_measure"] == outcome)
            )
            result_table.loc[mask, "bh_q_within_model_metric"] = bh_adjust(
                result_table.loc[mask, "nominal_p_value"].to_numpy()
            )

    return result_table


# ==================== 七、绘图 ====================

def create_figures(summary, stability, rank_stability, gate_table, output_dir):
    """任务二，实验三：生成特征依赖与跨监测点稳定性图。"""
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    # 任务二，实验三：全局平均特征置换重要性。
    global_mean = (
        stability.groupby(["model", "feature"])["mean_delta_auc"]
        .mean()
        .reset_index()
    )

    for model_name in MODELS:
        subset = global_mean.loc[global_mean["model"] == model_name].nlargest(
            12, "mean_delta_auc"
        ).sort_values("mean_delta_auc")

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(subset["feature"], subset["mean_delta_auc"])
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_xlabel("Mean permutation importance (baseline AUC - permuted AUC)")
        ax.set_title(f"Feature reliance across sites: {model_name}")
        fig.tight_layout()
        fig.savefig(
            figure_dir / f"global_feature_importance_{model_name}.png",
            dpi=250,
            bbox_inches="tight",
        )
        plt.close(fig)

    # 任务二，实验三：12站点×19特征重要性热力图。
    for model_name in MODELS:
        pivot = summary.loc[
            (summary["model"] == model_name)
            & summary["feature"].isin(FEATURES)
        ].pivot(
            index="site_code",
            columns="feature",
            values="delta_auc_mean",
        )

        fig, ax = plt.subplots(figsize=(14, 6))
        image = ax.imshow(
            pivot.to_numpy(),
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=60, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f"Site-level feature reliance: {model_name}")
        fig.colorbar(
            image,
            ax=ax,
            label="AUC decrease after feature permutation",
        )
        fig.tight_layout()
        fig.savefig(
            figure_dir / f"site_feature_importance_{model_name}.png",
            dpi=250,
            bbox_inches="tight",
        )
        plt.close(fig)

    # 任务二，实验三：V5 gate与经验置换重要性对照。
    if not gate_table.empty:
        gate_mean = (
            gate_table.groupby(["model", "feature"])["gate_score"]
            .agg(gate_score="mean", gate_score_std="std")
            .reset_index()
        )
        merged = gate_mean.merge(
            stability.loc[stability["model"] == "v5"],
            on=["model", "feature"],
            how="inner",
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(
            merged["gate_score"],
            merged["mean_delta_auc"],
            s=65,
        )
        for row in merged.itertuples(index=False):
            ax.annotate(
                row.feature,
                (row.gate_score, row.mean_delta_auc),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=8,
            )
        ax.set_xlabel("V5 gate score at selected epoch")
        ax.set_ylabel("Mean permutation importance (AUC decrease)")
        ax.set_title("V5 internal gate vs empirical feature reliance")
        fig.tight_layout()
        fig.savefig(
            figure_dir / "v5_gate_vs_permutation_importance.png",
            dpi=250,
            bbox_inches="tight",
        )
        plt.close(fig)

    # 任务二，实验三：跨监测点特征重要性排序一致性。
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        rank_stability["model"],
        rank_stability["median_pairwise_spearman"],
    )
    ax.set_ylabel("Median pairwise Spearman correlation")
    ax.set_title("Cross-site feature reliance rank stability")
    ax.set_ylim(-1, 1)
    fig.tight_layout()
    fig.savefig(
        figure_dir / "feature_reliance_rank_stability.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)


# ==================== 八、主程序 ====================

def main():
    """任务二，实验三：复用MLP/V5历史模型开展重复特征置换。"""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--shift-detail", type=Path, default=SHIFT_DETAIL_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--permutations", type=int, default=N_PERMUTATIONS)
    args = parser.parse_args()

    if args.permutations < 10:
        raise ValueError("建议至少进行10次置换；正式默认设置为30次。")
    if not args.data.is_file():
        raise FileNotFoundError(f"找不到原始数据：{args.data}")
    if not args.shift_detail.is_file():
        raise FileNotFoundError(f"找不到实验二特征偏移明细：{args.shift_detail}")
    if args.output.exists():
        raise FileExistsError(
            f"固定结果目录已存在，为避免覆盖：{args.output}"
        )

    data_hash = sha256_file(args.data)
    frame = load_data(args.data)

    raw_rows = []
    baseline_rows = []
    gate_rows = []
    model_manifests = []

    # 任务二，实验三：先加载并核验所有历史模型，避免运行到一半才发现缺档。
    contexts = {}
    for region in REGIONS:
        for model_name in MODELS:
            model_dir = args.model_root / region / model_name
            context = load_fold_context(
                frame, region, model_name, args.model_root, data_hash
            )
            contexts[(region, model_name)] = context

            baseline_rows.append(
                {
                    "held_out_region": region,
                    "model": model_name,
                    "sample_count": len(context["target_frame"]),
                    "site_count": int(
                        context["target_frame"][ENVIRONMENT].nunique()
                    ),
                    "best_epoch": context["best_epoch"],
                    "prediction_max_abs_error": context["prediction_max_abs_error"],
                    **normalize_metrics(
                        context["target_frame"]["label"].to_numpy(dtype=int),
                        context["baseline_probability"],
                    ),
                }
            )

            if model_name == "v5":
                gate_rows.extend(
                    {
                        "held_out_region": region,
                        "model": model_name,
                        "feature": feature,
                        "gate_score": float(score),
                        "best_epoch": context["best_epoch"],
                    }
                    for feature, score in zip(FEATURES, context["gate_scores"])
                )

            model_manifests.append(
                {
                    "held_out_region": region,
                    "model": model_name,
                    "model_directory": str(model_dir),
                    "best_epoch": context["best_epoch"],
                    "checkpoint_sha256": context["checkpoint_sha256"],
                    "preprocessor_sha256": context["preprocessor_sha256"],
                    "prediction_max_abs_error": context["prediction_max_abs_error"],
                }
            )

    # 任务二，实验三：对每个站点、每个模型计算30次置换重要性。
    for region in REGIONS:
        target = contexts[(region, "baseline")]["target_frame"]
        sites = sorted(target[ENVIRONMENT].astype(str).unique())

        for site in sites:
            print(f"分析监测点 {site}（留出地区：{region}）", flush=True)

            for model_name in MODELS:
                baseline_metrics, rows = calculate_site_importance(
                    region,
                    site,
                    model_name,
                    contexts[(region, model_name)],
                    args.permutations,
                )

                raw_rows.extend(rows)

    raw_table = pd.DataFrame(raw_rows)
    summary_table = summarize_importance(raw_table)
    stability_table, rank_stability = calculate_stability(summary_table)
    gate_table = pd.DataFrame(gate_rows)

    shift_correlations = correlate_shift_and_reliance(
        summary_table,
        args.shift_detail,
    )

    # 任务二，实验三：全部历史模型和预测核验通过后，才创建固定输出目录。
    args.output.mkdir(parents=True, exist_ok=False)

    raw_table.to_csv(
        args.output / "permutation_importance_repeats.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary_table.to_csv(
        args.output / "site_feature_importance_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stability_table.to_csv(
        args.output / "cross_site_feature_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rank_stability.to_csv(
        args.output / "feature_rank_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(baseline_rows).to_csv(
        args.output / "baseline_prediction_recheck.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gate_table.to_csv(
        args.output / "v5_gate_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )
    shift_correlations.to_csv(
        args.output / "feature_shift_reliance_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )

    create_figures(
        summary_table,
        stability_table,
        rank_stability,
        gate_table,
        args.output,
    )

    metadata = {
        "experiment": "Task 2 Experiment 3 - Feature Reliance and Cross-Site Stability",
        "data_file": str(args.data.resolve()),
        "data_sha256": data_hash,
        "model_root": str(args.model_root.resolve()),
        "shift_detail_file": str(args.shift_detail.resolve()),
        "shift_detail_sha256": sha256_file(args.shift_detail),
        "models": list(MODELS),
        "regions": list(REGIONS),
        "features": list(FEATURES),
        "permutations_per_site_feature_model": int(args.permutations),
        "random_seed": RANDOM_SEED,
        "permutation_unit": "raw feature column values within each held-out monitoring site",
        "preprocessing": "historical fold-specific saved preprocessor; no refit",
        "joint_permutation_groups": {
            name: list(members) for name, members in JOINT_GROUPS.items()
        },
        "baseline_prediction_tolerance": PREDICTION_TOLERANCE,
        "models_retrained": False,
        "target_labels_used_for_permutation": False,
        "target_labels_used_only_for_metrics": True,
        "permutation_importance_definition": {
            "delta_auc": "baseline AUC minus permuted AUC; positive means predictive discrimination decreases",
            "delta_auprc": "baseline AUPRC minus permuted AUPRC; positive means ranking quality decreases",
            "delta_brier": "permuted Brier minus baseline Brier; positive means probability error increases",
            "delta_log_loss": "permuted log loss minus baseline log loss; positive means probability error increases",
        },
        "limitations": [
            "Permutation importance is predictive reliance, not a causal effect.",
            "Single-feature permutation can disrupt correlations among inputs.",
            "The phone-use bundle is jointly permuted to retain within-bundle structure.",
            "Only 12 monitoring sites are available; stability summaries are exploratory.",
            "Permutation-repeat quantiles describe permutation variability, not population confidence intervals.",
            "Only persisted baseline MLP and V5 checkpoints are included; other model variants are not silently substituted.",
        ],
        "model_artifacts": model_manifests,
        "python_note": "This script reuses saved checkpoints and preprocessors and does not train models.",
    }

    write_json(args.output / "experiment_metadata.json", metadata)

    print("=" * 70)
    print("任务二，实验三：模型特征依赖与跨监测点稳定性分析完成")
    print(f"结果目录：{args.output}")
    print(f"置换记录：{len(raw_table)}")
    print("\n跨监测点特征依赖稳定性：")
    print(stability_table.sort_values(
        ["model", "mean_delta_auc"], ascending=[True, False]
    ).groupby("model").head(10).to_string(index=False))
    print("\n特征重要性排序一致性：")
    print(rank_stability.to_string(index=False))


if __name__ == "__main__":
    main()

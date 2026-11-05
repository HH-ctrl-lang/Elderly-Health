"""任务二，实验四：候选因果特征组的受控消融与跨环境验证。

最终目的不是简单提高AUC，而是筛选具有医学合理性、跨环境稳定性、
且不依赖明显环境代理信息的因果候选健康特征。

设计：
- 三个LORO留出地区：east / central / west；
- 两类模型：baseline MLP / V5因果解耦模型；
- 同一留出折内，完整19特征模型作为本轮重训对照；
- 每次仅移除一个预先定义的候选特征组；
- 同一模型类型、地区、对照/消融条件使用相同seed和训练参数；
- 训练/验证/测试样本划分完全相同；
- 缺失值填补和标准化只在源域训练集拟合；
- 目标地区标签只用于最终测试评价，不参与特征选择、训练或checkpoint选择。

注意：
- IADL可能与认知功能存在构念重叠或处于认知变化的下游路径，单独消融，
  不预先把它认定为因果特征或泄漏变量。
- 消融后性能下降只说明模型依赖该信息，不证明该变量具有因果效应。
- V5的sparsity='mean'，其门控均值惩罚保持原定义；target_sparsity在该模式下不参与损失。
"""

import hashlib
import json
import pickle
import platform
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))

from elderly_health.config import TrainConfig  # noqa: E402
from elderly_health.data import (  # noqa: E402
    ENVIRONMENT,
    FEATURES,
    REGION,
    SplitData,
    attach_regions,
    held_out_split,
    read_raw,
)
from elderly_health.evaluation import metrics  # noqa: E402
from elderly_health.models import CausalDisentangledModel, SimpleMLP  # noqa: E402
from elderly_health.training import fit, get_device, predict  # noqa: E402


# ==================== 一、实验配置 ====================

DATA_PATH = PROJECT / "data" / "data_mission2.csv"
CONFIG_PATH = PROJECT / "configs" / "v2_region_v1_selection91.json"
DICTIONARY_PATH = PROJECT / "data" / "25年变量字典.xlsx"

OUTPUT_DIR = (
    ROOT / "outputs" / "analysis_2611"
    / "experiment4_candidate_feature_ablation"
)

REGIONS = ("east", "central", "west")
MODELS = ("baseline", "v5")
SEED = 42

# 每次只移除一组，避免把多个不同机制混在同一个消融对照中。
ABLATION_GROUPS = {
    "iadl_single": {
        "features": ("IADL",),
        "hypothesis": "功能状态/可能的结局近端信息",
        "risk": (
            "需检查IADL量表与认知结局是否存在测量构念重叠，"
            "以及IADL是否可能位于认知变化的下游路径。"
        ),
    },
    "depression_sleep": {
        "features": ("depression", "sleep_duration"),
        "hypothesis": "心理与睡眠健康信息",
        "risk": "两变量可能受共同健康状态或未测量因素影响。",
    },
    "phone_use_bundle": {
        "features": (
            "if_use_mobile_phone",
            "using_phone_year",
            "hours_phone_per_day",
        ),
        "hypothesis": "数字设备使用与生活方式信息",
        "risk": "手机使用可能同时反映年龄、教育、社会参与和环境条件。",
    },
    "socioeconomic_context": {
        "features": ("residence", "education", "wealth"),
        "hypothesis": "社会经济与环境代理信息",
        "risk": "这些变量可能既包含真实社会决定因素，也包含样本构成/环境代理信息。",
    },
}

# 候选变量的中英文检索词，用于从变量字典中抽取原始定义行。
DICTIONARY_ALIASES = {
    "IADL": ["IADL", "日间功能障碍", "工具性日常生活", "日常生活能力", "家务"],
    "depression": ["depression", "是否抑郁", "抑郁"],
    "sleep_duration": ["sleep_duration", "睡眠时长", "睡眠时间"],
    "if_use_mobile_phone": ["if_use_mobile_phone", "是否使用手机", "使用手机"],
    "using_phone_year": ["using_phone_year", "手机使用年限", "使用手机年限"],
    "hours_phone_per_day": ["hours_phone_per_day", "每日手机使用", "每天使用手机"],
    "residence": ["residence", "居住类型", "城乡", "居住地"],
    "education": ["education", "受教育程度", "教育程度", "文化程度"],
    "wealth": ["wealth", "财富状况", "家庭财富", "收入"],
}

METRIC_KEYS = (
    "auc",
    "auprc",
    "brier",
    "log_loss",
    "balanced_accuracy",
    "f1",
    "precision",
    "recall",
    "accuracy",
)


# ==================== 二、基础工具 ====================

def sha256_file(path):
    """任务二，实验四：计算输入文件SHA256。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """任务二，实验四：保存可复核的实验记录。"""
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def safe_metrics(y, probability):
    """任务二，实验四：兼容单站点仅含一个结局类别的情况。"""
    result = metrics(
        np.asarray(y, dtype=int),
        np.asarray(probability, dtype=float),
    )
    return {key: result.get(key) for key in METRIC_KEYS} | {
        "n": int(result["n"]),
        "positive_rate": float(result["positive_rate"]),
    }


def make_model(model_name, input_dim, num_envs, config):
    """任务二，实验四：按项目历史模型结构实例化模型。"""
    if model_name == "baseline":
        return SimpleMLP(input_dim)

    if model_name == "v5":
        return CausalDisentangledModel(
            input_dim=input_dim,
            num_envs=num_envs,
            latent_dim=config.latent_dim,
            gate_init_std=config.gate_init_std,
            gate_temperature=config.gate_temperature,
            joint=True,
        )

    raise ValueError(f"不支持的模型：{model_name}")


def prepare_feature_splits(frames, feature_names):
    """任务二，实验四：动态特征子集预处理，拟合仅使用源域训练集。"""
    train_frame, validation_frame, test_frame = frames

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    train_imputed = imputer.fit_transform(train_frame[list(feature_names)])
    train_x = scaler.fit_transform(train_imputed).astype(np.float32)

    def transform(frame):
        values = imputer.transform(frame[list(feature_names)])
        return scaler.transform(values).astype(np.float32)

    splits = []
    for frame, x in (
        (train_frame, train_x),
        (validation_frame, transform(validation_frame)),
        (test_frame, transform(test_frame)),
    ):
        splits.append(
            SplitData(
                x=x,
                y=frame["label"].to_numpy(dtype=np.float32),
                env=frame[REGION].to_numpy(dtype=str),
                sample_ids=frame["sample_id"].to_numpy(dtype=np.int64),
                site_env=frame[ENVIRONMENT].to_numpy(dtype=str),
            )
        )

    preprocessor = {
        "feature_names": tuple(feature_names),
        "imputer": imputer,
        "scaler": scaler,
    }
    return splits, preprocessor


# ==================== 三、变量字典审计 ====================

def audit_variable_dictionary(path):
    """任务二，实验四：从Excel变量字典提取候选变量的原始定义行。

    该函数不自动推断编码方向，只保留字典原文供研究者复核。
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"未找到变量字典：{path}。为避免盲目解释候选特征，停止实验。"
        )

    workbook = pd.ExcelFile(path)
    rows = []

    for sheet_name in workbook.sheet_names:
        table = pd.read_excel(
            workbook,
            sheet_name=sheet_name,
            dtype=str,
            header=None,
        ).fillna("")

        for row_index, values in table.iterrows():
            raw_cells = [
                str(value).strip()
                for value in values.tolist()
                if str(value).strip()
            ]
            if not raw_cells:
                continue

            row_text = " | ".join(raw_cells).lower()

            for feature, aliases in DICTIONARY_ALIASES.items():
                matched_aliases = [
                    alias for alias in aliases
                    if alias.lower() in row_text
                ]
                if matched_aliases:
                    rows.append(
                        {
                            "feature": feature,
                            "sheet": sheet_name,
                            "excel_row_1based": int(row_index) + 1,
                            "matched_aliases": ";".join(matched_aliases),
                            "dictionary_raw_row": " | ".join(raw_cells),
                            "manual_definition_review_required": True,
                        }
                    )

    result = pd.DataFrame(
        rows,
        columns=[
            "feature",
            "sheet",
            "excel_row_1based",
            "matched_aliases",
            "dictionary_raw_row",
            "manual_definition_review_required",
        ],
    )

    found = set(result["feature"]) if not result.empty else set()
    summary = []
    for feature, aliases in DICTIONARY_ALIASES.items():
        matches = result.loc[result["feature"] == feature] if not result.empty else pd.DataFrame()
        summary.append(
            {
                "feature": feature,
                "dictionary_match_found": feature in found,
                "matched_row_count": len(matches),
                "aliases_searched": ";".join(aliases),
                "coding_direction_confirmed": False,
                "manual_review_required": True,
            }
        )

    return result, pd.DataFrame(summary)


def validate_ablation_groups():
    """任务二，实验四：检查候选组没有重叠或包含未知特征。"""
    used = []
    for group_name, specification in ABLATION_GROUPS.items():
        members = tuple(specification["features"])
        if not members:
            raise ValueError(f"候选组{group_name}为空。")
        unknown = set(members) - set(FEATURES)
        if unknown:
            raise ValueError(f"候选组{group_name}包含未知特征：{sorted(unknown)}")
        overlap = set(members).intersection(used)
        if overlap:
            raise ValueError(f"候选组之间存在重叠：{sorted(overlap)}")
        used.extend(members)

    return True


# ==================== 四、训练与评价 ====================

def train_one_condition(
    model_name,
    condition_name,
    feature_names,
    frames,
    config,
    output_dir,
):
    """任务二，实验四：训练一个全特征对照或单组消融模型。"""
    train_frame, validation_frame, test_frame = frames
    splits, preprocessor = prepare_feature_splits(frames, feature_names)
    train_split, validation_split, test_split = splits

    model, training = fit(
        train_split,
        validation_split,
        config,
    )

    device = get_device(config.device)
    test_probability = predict(model, test_split.x, device)
    validation_probability = predict(model, validation_split.x, device)

    overall = safe_metrics(test_split.y, test_probability)

    site_rows = []
    for site in sorted(set(test_split.site_env.astype(str))):
        mask = test_split.site_env.astype(str) == site
        site_rows.append(
            {
                "held_out_region": str(test_frame[REGION].iloc[0]),
                "site_code": site,
                "model": model_name,
                "condition": condition_name,
                **safe_metrics(test_split.y[mask], test_probability[mask]),
            }
        )

    condition_dir = output_dir / model_name / condition_name
    condition_dir.mkdir(parents=True, exist_ok=False)

    # 保存模型、预处理器、预测和训练历史，保证后续可复核。
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config.to_dict(),
            "feature_names": list(feature_names),
            "model_name": model_name,
            "condition": condition_name,
            "best_epoch": int(training["best_epoch"]),
            "env_map": training["env_map"],
        },
        condition_dir / "best_model.pt",
    )

    with (condition_dir / "preprocessor.pkl").open("wb") as stream:
        pickle.dump(preprocessor, stream)

    pd.DataFrame(
        {
            "sample_id": test_split.sample_ids,
            "site_code": test_split.site_env,
            "held_out_region": test_split.env,
            "cognitive_impairment": test_split.y.astype(int),
            "predicted_probability": test_probability,
            "split": "test",
        }
    ).to_csv(
        condition_dir / "predictions_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        {
            "sample_id": validation_split.sample_ids,
            "site_code": validation_split.site_env,
            "cognitive_impairment": validation_split.y.astype(int),
            "predicted_probability": validation_probability,
            "split": "validation",
        }
    ).to_csv(
        condition_dir / "predictions_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(training["history"]).to_csv(
        condition_dir / "training_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    gate_scores = None
    if model_name == "v5":
        selected = [
            row for row in training["history"]
            if int(row["epoch"]) == int(training["best_epoch"])
        ]
        if len(selected) != 1 or "mask" not in selected[0]:
            raise RuntimeError(
                f"{model_name}/{condition_name}找不到最佳轮次Gate记录。"
            )
        gate_scores = selected[0]["mask"]

    write_json(
        condition_dir / "condition_metadata.json",
        {
            "model": model_name,
            "condition": condition_name,
            "features": list(feature_names),
            "removed_features": sorted(set(FEATURES) - set(feature_names)),
            "config": config.to_dict(),
            "best_epoch": int(training["best_epoch"]),
            "best_validation_auc": float(training["best_val_auc"]),
            "selection_start_epoch": int(training["selection_start_epoch"]),
            "preprocessor_fit_on": "source training split only",
            "target_labels_used_for_training": False,
            "target_labels_used_for_checkpoint_selection": False,
            "gate_scores_at_best_epoch": gate_scores,
            "test_metrics": overall,
        },
    )

    metric_row = {
        "held_out_region": str(test_frame[REGION].iloc[0]),
        "model": model_name,
        "condition": condition_name,
        "feature_count": len(feature_names),
        "removed_features": ";".join(sorted(set(FEATURES) - set(feature_names))),
        "best_epoch": int(training["best_epoch"]),
        **overall,
    }

    return metric_row, site_rows, gate_scores


# ==================== 五、结果汇总 ====================

def build_ablation_deltas(metrics_table, site_table):
    """任务二，实验四：相对同地区同模型全特征对照计算性能差值。"""
    rows = []

    for (region, model_name), group in metrics_table.groupby(
        ["held_out_region", "model"]
    ):
        baseline_rows = group.loc[group["condition"] == "full_features"]
        if len(baseline_rows) != 1:
            raise RuntimeError(f"{region}/{model_name}缺少唯一全特征对照。")
        baseline = baseline_rows.iloc[0]

        for _, row in group.loc[group["condition"] != "full_features"].iterrows():
            item = {
                "held_out_region": region,
                "model": model_name,
                "condition": row["condition"],
                "removed_features": row["removed_features"],
            }

            for metric in METRIC_KEYS:
                old = baseline[metric]
                new = row[metric]
                item[f"baseline_{metric}"] = old
                item[f"ablation_{metric}"] = new
                item[f"delta_{metric}"] = (
                    float(new) - float(old)
                    if pd.notna(old) and pd.notna(new)
                    else np.nan
                )

            item["auc_retention"] = (
                float(row["auc"] / baseline["auc"])
                if pd.notna(row["auc"]) and pd.notna(baseline["auc"])
                and float(baseline["auc"]) != 0
                else np.nan
            )
            item["auprc_retention"] = (
                float(row["auprc"] / baseline["auprc"])
                if pd.notna(row["auprc"]) and pd.notna(baseline["auprc"])
                and float(baseline["auprc"]) != 0
                else np.nan
            )
            rows.append(item)

    return pd.DataFrame(rows)


def summarize_cross_region(deltas):
    """任务二，实验四：汇总各消融组在三个留出地区的变化方向。"""
    rows = []
    for (model, condition), group in deltas.groupby(["model", "condition"]):
        rows.append(
            {
                "model": model,
                "condition": condition,
                "region_count": int(group["held_out_region"].nunique()),
                "mean_delta_auc": float(group["delta_auc"].mean()),
                "min_delta_auc": float(group["delta_auc"].min()),
                "max_delta_auc": float(group["delta_auc"].max()),
                "regions_auc_improved": int((group["delta_auc"] > 0).sum()),
                "mean_delta_auprc": float(group["delta_auprc"].mean()),
                "mean_delta_brier": float(group["delta_brier"].mean()),
                "mean_auc_retention": float(group["auc_retention"].mean()),
            }
        )
    return pd.DataFrame(rows)


# ==================== 六、绘图 ====================

def create_figures(metrics_table, deltas, output_dir):
    """任务二，实验四：绘制消融前后性能变化。"""
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    for model_name in MODELS:
        subset = deltas.loc[deltas["model"] == model_name]
        pivot = subset.pivot(
            index="condition",
            columns="held_out_region",
            values="delta_auc",
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel("AUC change vs full-feature control")
        ax.set_xlabel("Ablation group")
        ax.set_title(f"Candidate feature ablation: {model_name}")
        ax.legend(title="Held-out region")
        fig.tight_layout()
        fig.savefig(
            figure_dir / f"ablation_auc_delta_{model_name}.png",
            dpi=250,
            bbox_inches="tight",
        )
        plt.close(fig)

    for model_name in MODELS:
        subset = metrics_table.loc[metrics_table["model"] == model_name]
        pivot = subset.pivot(
            index="condition",
            columns="held_out_region",
            values="auc",
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("AUC")
        ax.set_ylim(0, 1)
        ax.set_title(f"Cross-region AUC by ablation: {model_name}")
        ax.legend(title="Held-out region")
        fig.tight_layout()
        fig.savefig(
            figure_dir / f"ablation_auc_absolute_{model_name}.png",
            dpi=250,
            bbox_inches="tight",
        )
        plt.close(fig)


# ==================== 七、主程序 ====================

def main():
    """任务二，实验四：执行三地区、两模型、四组单独消融。"""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--dictionary", type=Path, default=DICTIONARY_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default=None)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(
            f"固定实验目录已存在，为避免覆盖：{args.output}"
        )
    for path in (args.data, args.config, args.dictionary):
        if not path.is_file():
            raise FileNotFoundError(f"缺少必要输入文件：{path}")

    validate_ablation_groups()

    # 任务二，实验四：先审计字典并记录原文，不自动解释编码方向。
    dictionary_detail, dictionary_summary = audit_variable_dictionary(
        args.dictionary
    )

    frame = attach_regions(read_raw(args.data, encoding="gbk"))
    config_values = json.loads(args.config.read_text(encoding="utf-8"))
    base_config = TrainConfig(**config_values)

    if base_config.environment_granularity != "region":
        raise ValueError("当前消融实验要求region环境粒度。")
    if base_config.seed != SEED:
        raise ValueError("历史LORO基线使用seed=42，当前配置不一致。")
    if base_config.device != "cpu":
        raise ValueError("本阶段先保持历史CPU训练设置。")

    device = args.device or base_config.device
    threads = args.threads or base_config.threads

    if device != base_config.device or threads != base_config.threads:
        raise ValueError(
            "为保证控制变量，本次不允许更改历史LORO设备或线程数。"
        )

    all_metrics = []
    all_site_metrics = []
    all_gate_rows = []
    all_training_rows = []
    all_config_rows = []

    # 任务二，实验四：在任何训练开始前固定所有地区的样本划分。
    region_frames = {}
    for region in REGIONS:
        splits = held_out_split(
            frame,
            region,
            seed=SEED,
            column=REGION,
        )
        region_frames[region] = splits

    total_conditions = len(REGIONS) * len(MODELS) * (
        1 + len(ABLATION_GROUPS)
    )
    print("=" * 72)
    print("任务二，实验四：候选因果特征组受控消融")
    print("=" * 72)
    print(f"数据样本：{len(frame)}")
    print(f"输入特征：{len(FEATURES)}")
    print(f"候选消融组：{len(ABLATION_GROUPS)}")
    print(f"预计模型训练次数：{total_conditions}")
    print("每个地区/模型均包含同轮次全特征对照，不复用不同随机初始化的旧模型。")
    print("目标地区标签仅用于最终测试评价。")

    # 所有字典审计和配置先写入内存，正式训练结束后一次性创建输出目录。
    for region in REGIONS:
        frames = region_frames[region]

        for model_name in MODELS:
            if model_name == "baseline":
                model_config = replace(
                    base_config,
                    model="baseline",
                    selection_start_epoch=None,
                )
            else:
                model_config = replace(
                    base_config,
                    model="v5",
                )

            conditions = [("full_features", tuple(FEATURES))]
            for group_name, specification in ABLATION_GROUPS.items():
                removed = set(specification["features"])
                kept = tuple(feature for feature in FEATURES if feature not in removed)
                conditions.append((group_name, kept))

            for condition_name, feature_names in conditions:
                print(
                    f"\n[{region}] {model_name} / {condition_name} "
                    f"({len(feature_names)} features)",
                    flush=True,
                )

                # 每个模型和条件均由fit内部重置同一个seed，形成配对控制。
                # 全特征控制与每个消融组均在同一地区、同一划分上独立训练。
                metric_row, site_rows, gate_scores = train_one_condition(
                    model_name=model_name,
                    condition_name=condition_name,
                    feature_names=feature_names,
                    frames=frames,
                    config=model_config,
                    output_dir=args.output / region,
                )

                all_metrics.append(metric_row)
                all_site_metrics.extend(site_rows)

                if gate_scores is not None:
                    all_gate_rows.extend(
                        {
                            "held_out_region": region,
                            "model": model_name,
                            "condition": condition_name,
                            "feature": feature,
                            "gate_score": float(score),
                        }
                        for feature, score in zip(feature_names, gate_scores)
                    )

                all_config_rows.append(
                    {
                        "held_out_region": region,
                        "model": model_name,
                        "condition": condition_name,
                        "feature_count": len(feature_names),
                        "features": ";".join(feature_names),
                        "config": json.dumps(
                            model_config.to_dict(),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    }
                )

    metrics_table = pd.DataFrame(all_metrics)
    site_table = pd.DataFrame(all_site_metrics)
    gate_table = pd.DataFrame(all_gate_rows)
    config_table = pd.DataFrame(all_config_rows)

    deltas = build_ablation_deltas(metrics_table, site_table)
    cross_region = summarize_cross_region(deltas)

    # 任务二，实验四：所有训练完成后创建固定输出目录，避免半成品被误认为完整结果。
    args.output.mkdir(parents=True, exist_ok=False)

    dictionary_detail.to_csv(
        args.output / "candidate_variable_dictionary_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )
    dictionary_summary.to_csv(
        args.output / "candidate_variable_dictionary_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metrics_table.to_csv(
        args.output / "region_model_ablation_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    site_table.to_csv(
        args.output / "site_model_ablation_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    deltas.to_csv(
        args.output / "ablation_performance_deltas.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cross_region.to_csv(
        args.output / "cross_region_ablation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gate_table.to_csv(
        args.output / "ablation_gate_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )
    config_table.to_csv(
        args.output / "all_model_configs.csv",
        index=False,
        encoding="utf-8-sig",
    )

    create_figures(metrics_table, deltas, args.output)

    metadata = {
        "experiment": "Task 2 Experiment 4 - Candidate Causal Feature Ablation",
        "research_goal": (
            "Identify medically plausible, cross-environment stable candidate health "
            "features; predictive ablation is evidence about model reliance, not causal proof."
        ),
        "data_sha256": sha256_file(args.data),
        "config_sha256": sha256_file(args.config),
        "dictionary_sha256": sha256_file(args.dictionary),
        "sample_count": int(len(frame)),
        "site_count": int(frame[ENVIRONMENT].nunique()),
        "features": list(FEATURES),
        "regions": list(REGIONS),
        "models": list(MODELS),
        "seed": SEED,
        "split": "same held_out_split(seed=42, column=environment_region) for all conditions",
        "preprocessing": "feature-subset median imputation and StandardScaler; fit on source training rows only",
        "full_feature_control": (
            "retrained in the same run for each region/model with the same seed and "
            "configuration as the corresponding ablation conditions"
        ),
        "ablation_groups": {
            name: {
                "features": list(spec["features"]),
                "hypothesis": spec["hypothesis"],
                "risk": spec["risk"],
            }
            for name, spec in ABLATION_GROUPS.items()
        },
        "v5_sparsity_policy": (
            "Preserve historical sparsity='mean' and lambda_sparse; target_sparsity "
            "is inactive in mean mode, so no dimension-dependent retuning is introduced."
        ),
        "target_labels_used_for_training": False,
        "target_labels_used_for_checkpoint_selection": False,
        "target_labels_used_for_final_evaluation": True,
        "dictionary_coding_direction_confirmed": False,
        "dictionary_matches": dictionary_summary.to_dict(orient="records"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "training_runs": int(len(metrics_table)),
        "limitations": [
            "One random seed is used to preserve comparability with historical LORO; repeat seeds are needed for robustness.",
            "Ablation changes model input information and may alter optimization; paired same-seed controls reduce but do not eliminate this issue.",
            "IADL may be outcome-proximal or share measurement constructs with cognition; dictionary and clinical review are required.",
            "A performance decrease after ablation shows predictive reliance, not a causal effect.",
            "Only three held-out regions and twelve monitoring sites are available.",
        ],
    }
    write_json(args.output / "experiment_metadata.json", metadata)

    print("\n" + "=" * 72)
    print("实验四完成")
    print(f"结果目录：{args.output}")
    print("\n各消融组相对全特征对照的跨地区结果：")
    print(cross_region.to_string(index=False))
    print("\n字典匹配情况（编码方向仍需人工复核）：")
    print(dictionary_summary.to_string(index=False))


if __name__ == "__main__":
    main()

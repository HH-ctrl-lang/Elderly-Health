"""任务五 第4步：MIR/VIR二分类公式适配的一次有限筛选，由用户运行。

固定原19维输入和地区留一划分；用来源训练站点估计无标签均值/协方差。
只按来源验证AUC选择预定强度，留出地区不参与选模型、特征或约束。
这是预测层面适配，不是反因果假设已成立的证明，也不直接做表征解耦。
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
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
from elderly_health.data import FEATURES, ENVIRONMENT, REGION, read_raw, attach_regions, held_out_split, prepare_splits
from elderly_health.anti_causal_regularization import environment_matrices, fit_logistic, probabilities

OUTPUT = ROOT / "outputs/analysis_2611/task5_step4_mir_vir_health_screen"
REGIONS = ("east", "central", "west")
GAMMAS = (0.0, 0.01, 0.1, 1.0, 10.0)
DATA_SHA256 = "969f0c1c82faafcfa402f37572e4a8ceb324492062cf1e590465f82dc6b5b6de"


def digest(path):
    """任务五 第4步：记录数据、历史参考和代码版本。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    """任务五 第4步：保存可核查记录，不生成阶段结论。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def metrics(y, p):
    """任务五 第4步：评价排序和概率误差，数据只在最终评价中使用。"""
    return {"n": len(y), "positive_count": int(y.sum()), "positive_rate": float(y.mean()),
            "auc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1]))}


def choose_candidate(candidates):
    """任务五 第4步：验证AUC优先，完全相同则取较小强度；允许选择gamma=0。"""
    return min(candidates, key=lambda row: (-row["source_val_auc"], row["gamma"]))


def macro_metrics(site):
    """任务五 第4步：站点等权宏指标，不用人数多的站点淹没小站点。"""
    records = []
    for scope in (*REGIONS, "ALL"):
        part = site if scope == "ALL" else site.loc[site.held_out_region.eq(scope)]
        for model, group in part.groupby("model"):
            records.append({"scope": scope, "model": model, "site_count": len(group),
                            **group[["auc", "auprc", "brier"]].mean().to_dict()})
    return pd.DataFrame(records)


def draw_figures(site, macro, output):
    """任务五 第4步：只比较验证集选定版本，不能从留出结果挑参数。"""
    figures = output / "figures"
    figures.mkdir()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ("auc", "auprc")):
        macro.pivot(index="scope", columns="model", values=metric).reindex([*REGIONS, "ALL"]).plot.bar(ax=ax, rot=0)
        ax.set_ylabel(f"Equal-site macro {metric.upper()}")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "region_and_global_macro_performance.png", dpi=220)
    plt.close(fig)
    pivot = site.loc[site.model.ne("MLP_historical_reference")].pivot(
        index=["held_out_region", "site_code"], columns="model", values="auc")
    order = [(r, s) for r in REGIONS for s in sorted(site.loc[site.held_out_region.eq(r), "site_code"].unique())]
    pivot = pivot.reindex(pd.MultiIndex.from_tuples(order, names=["held_out_region", "site_code"]))
    changes = pd.DataFrame({name: pivot[name]-pivot["linear_control"] for name in ("MIR", "VIR")}, index=pivot.index)
    fig, ax = plt.subplots(figsize=(12, 5))
    changes.plot.bar(ax=ax)
    ax.set_xticklabels([f"{r}: {s}" for r, s in order], rotation=55, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Site AUC: selected method minus identical linear control")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(figures / "site_auc_changes.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data/data_mission2.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"禁止覆盖已有结果：{args.output}")
    if digest(args.data) != DATA_SHA256:
        raise ValueError("数据版本不同。")
    frame = attach_regions(read_raw(args.data, encoding="gbk"))
    if len(frame) != 4769 or frame[ENVIRONMENT].nunique() != 12:
        raise ValueError("样本数或站点数与历史研究不同。")
    reference_path = ROOT / "outputs/analysis_2611/task3_step1_site_failure_diagnostics/site_model_score_diagnostics.csv"
    reference = pd.read_csv(reference_path, dtype={"site_code": str})
    reference = reference.loc[reference.model.eq("MLP")].copy()
    if len(reference) != 12 or reference.site_code.duplicated().any():
        raise ValueError("历史MLP参考不完整。")
    for row in reference.to_dict("records"):
        part = frame.loc[frame[ENVIRONMENT].eq(row["site_code"])]
        if len(part) != row["n"] or int(part.label.sum()) != row["positive_count"] or set(part[REGION]) != {row["held_out_region"]}:
            raise ValueError("历史参考样本不一致。")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "experiment_design.json", {
        "paper": "https://proceedings.mlr.press/v306/saengkyongam26a.html",
        "checked_pdf_version": "arXiv:2602.17187v2, 2026-06-17; equations 9,10 and Section6.1",
        "reproduction_level": "independent_binary_prediction_adaptation_not_published_benchmark_replication",
        "gammas": list(GAMMAS), "selection": "pooled source validation AUC; exact tie chooses smaller gamma; zero allowed",
        "seed": 42, "input_features": list(FEATURES), "ridge": 1e-5,
        "loss": "unweighted binary cross entropy; equal source sites; unpenalized intercept",
        "penalty_environments": "eight source training sites, not target sites",
        "individual_records_exported": False, "data_sha256": DATA_SHA256,
        "historical_reference_sha256": digest(reference_path),
        "code_sha256": {"runner": digest(Path(__file__)), "adapter": digest(PROJECT / "src/elderly_health/anti_causal_regularization.py")},
        "limitations": ["No causal representation split and no causal identification claim.",
                        "Anti-causal SCM and environment-independent outcome assumptions are not established for the health data.",
                        "No target features are used to estimate environment moments; no target labels used for selection.",
                        "All source sites are labeled here, unlike the paper's main few-labeled-many-unlabeled setting.",
                        "Binary log-loss, free intercept and fixed small ridge are explicit adaptations; squared-loss robustness theorems do not automatically transfer.",
                        "Numeric codes are retained; changes in category means/covariances need not reflect causal perturbations.",
                        "Historical MLP differs in architecture, grouping and positive weighting; compare ranking only as a reference.",
                        "One split; assess value before further repetitions or parameter searches."]})
    rows, pooled, validation, selections, spectra, coefficients = [], [], [], [], [], []
    for region in REGIONS:
        frames = held_out_split(frame, region, seed=42, column=REGION)
        (train, val, test), preprocessor = prepare_splits(frames, environment_column=REGION)
        if set(train.sample_ids) & set(val.sample_ids) or set(test.sample_ids) & (set(train.sample_ids) | set(val.sample_ids)):
            raise RuntimeError("训练、验证或测试样本交叉。")
        matrices, groups, means, covariances = environment_matrices(train.x, train.site_env)
        if len(groups) != 8 or set(groups) & set(test.site_env):
            raise RuntimeError("正则矩阵必须仅来自8个来源训练站点。")
        directory = args.output / region
        directory.mkdir()
        np.savez(directory / "source_moments_and_preprocessing.npz", MIR=matrices["MIR"], VIR=matrices["VIR"],
                 site_means=means, site_covariances=covariances, source_sites=np.asarray(groups),
                 median=preprocessor.imputer.statistics_, mean=preprocessor.scaler.mean_, scale=preprocessor.scaler.scale_)
        write_json(directory / "split_manifest.json", {name: {"n": len(split.y), "positive_count": int(split.y.sum()),
                   "sample_ids_sha256": hashlib.sha256(np.asarray(split.sample_ids, dtype=np.int64).tobytes()).hexdigest()}
                   for name, split in (("train", train), ("validation", val), ("test", test))})
        baseline, report = fit_logistic(train.x, train.y, train.site_env, np.zeros((19, 19)), 0.0)
        baseline_auc = float(roc_auc_score(val.y, probabilities(baseline, val.x)))
        selected = {"linear_control": (baseline, 0.0, report)}
        for method, matrix in matrices.items():
            eigenvalues = np.linalg.eigvalsh(matrix)
            spectra.append({"held_out_region": region, "method": method, "rank": int(np.linalg.matrix_rank(matrix)),
                            "trace": float(np.trace(matrix)), "min_eigenvalue": float(eigenvalues.min()),
                            "max_eigenvalue": float(eigenvalues.max()), "source_site_count": len(groups)})
            candidates, fitted = [], {}
            for gamma in GAMMAS:
                theta, result = (baseline, report) if gamma == 0 else fit_logistic(train.x, train.y, train.site_env, matrix, gamma)
                score = float(roc_auc_score(val.y, probabilities(theta, val.x)))
                item = {"held_out_region": region, "method": method, "gamma": gamma, "source_val_auc": score, **result}
                validation.append(item)
                candidates.append(item)
                fitted[gamma] = (theta, result)
            choice = choose_candidate(candidates)
            theta, result = fitted[choice["gamma"]]
            selected[method] = (theta, choice["gamma"], result)
        # 任务五 第4步：两种方法都完成来源验证选择后，才评价留出地区。
        for model, (theta, gamma, result) in selected.items():
            score = float(roc_auc_score(val.y, probabilities(theta, val.x)))
            if model != "linear_control" and score < baseline_auc:
                raise RuntimeError("候选含gamma=0，来源验证成绩不应低于对照。")
            selections.append({"held_out_region": region, "model": model, "gamma": gamma, "source_val_auc": score})
            write_json(directory / f"{model}.json", {"model": model, "gamma": gamma, "coef": theta[:-1].tolist(),
                       "intercept": float(theta[-1]), "source_val_auc": score, "optimizer": result})
            coefficients.extend({"held_out_region": region, "model": model, "feature": feature, "coefficient": float(value)}
                                for feature, value in zip(FEATURES, theta[:-1]))
            p = probabilities(theta, test.x)
            pooled.append({"held_out_region": region, "model": model, **metrics(test.y, p)})
            for code in sorted(set(test.site_env)):
                mask = test.site_env == code
                rows.append({"held_out_region": region, "model": model, "site_code": str(code), **metrics(test.y[mask], p[mask])})
        print(f"任务五 第4步：{region}完成，参数仅按来源验证AUC选定。", flush=True)
    site = pd.DataFrame(rows)
    if len(site) != 36:
        raise RuntimeError("站点结果数量错误。")
    macro = macro_metrics(site)
    old = macro_metrics(reference.assign(model="MLP_historical_reference"))
    # 任务五 第4步：历史MLP使用正类权重，避免直接展示不同损失下的Brier优劣。
    old["brier"] = np.nan
    macro = pd.concat([macro, old], ignore_index=True)
    differences = []
    for scope in (*REGIONS, "ALL"):
        block = macro.loc[macro.scope.eq(scope)].set_index("model")
        for method in ("MIR", "VIR"):
            for ref in ("linear_control", "MLP_historical_reference"):
                differences.append({"scope": scope, "method": method, "reference": ref,
                                    **{f"delta_{metric}": float(block.loc[method, metric]-block.loc[ref, metric]) for metric in ("auc", "auprc")}})
    for name, table in (("site_metrics.csv", site), ("site_macro_metrics.csv", macro),
                        ("candidate_macro_differences.csv", pd.DataFrame(differences)),
                        ("pooled_region_metrics.csv", pd.DataFrame(pooled)),
                        ("source_validation_candidates.csv", pd.DataFrame(validation)),
                        ("selected_parameters.csv", pd.DataFrame(selections)),
                        ("source_penalty_matrix_diagnostics.csv", pd.DataFrame(spectra)),
                        ("coefficients_not_causal_importance.csv", pd.DataFrame(coefficients))):
        table.to_csv(args.output / name, index=False, encoding="utf-8-sig")
    draw_figures(site, macro, args.output)
    write_json(args.output / "completion.json", {"completed": True, "optimizer_fits": 27,
                                                 "target_evaluated_models": 9, "sites": 12})
    print(f"任务五 第4步完成，请提交整个目录：{args.output}")


if __name__ == "__main__":
    main()

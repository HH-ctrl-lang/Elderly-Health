"""任务3 第7步：原始低维特征预测筛选，可选显式指定TabPFN-v2检查点。

经典线性、稀疏线性、浅层树是对照，TabPFN-v2对应Nature 2025候选。
本步不等于因果解耦，也不通过预测分数证明任何变量的因果作用。
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT = Path(__file__).resolve().parents[3]
ROOT = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
from elderly_health.data import FEATURES, REGION, read_raw, attach_regions, held_out_split, prepare_splits
from elderly_health.evaluation import metrics

EXPECTED_HASH = "969f0c1c82faafcfa402f37572e4a8ceb324492062cf1e590465f82dc6b5b6de"


def score_rows(region, name, split, probability, stage):
    """任务3 第7步：合并地区与站点指标分别保存，不混合评价口径。"""
    rows = [{"region": region, "model": name, "scope": "region_pooled", "stage": stage, **metrics(split.y, probability)}]
    for site in sorted(set(split.site_env)):
        mask = split.site_env == site
        rows.append({"region": region, "model": name, "scope": "site", "stage": stage, "site_code": site, **metrics(split.y[mask], probability[mask])})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=PROJECT / "data/data_mission2.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/analysis_2611/task3_step7_low_dimensional_method_screen")
    parser.add_argument("--tabpfn-checkpoint", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != EXPECTED_HASH:
        raise ValueError("固定CSV版本不一致")
    checkpoint_hash, package_version = None, None
    if args.tabpfn_checkpoint:
        if not args.tabpfn_checkpoint.is_file():
            raise FileNotFoundError("请显式指定官方TabPFN-v2分类检查点文件")
        # 任务3 第7步：明确指定v2权重，禁止默认版本更新后仍将结果写作Nature 2025。
        if "v2" not in args.tabpfn_checkpoint.name.lower() or "v2." in args.tabpfn_checkpoint.name.lower():
            raise ValueError("请使用TabPFN-v2分类权重；v2.5、v2.6等版本须另立实验")
        from tabpfn import TabPFNClassifier
        checkpoint_hash = hashlib.sha256(args.tabpfn_checkpoint.read_bytes()).hexdigest()
        package_version = importlib.metadata.version("tabpfn")
        # 官方CPU数据量保护仅影响耗时；不截取样本以避免改变比较人群。
        os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
    args.output.mkdir(parents=True, exist_ok=False)
    frame = attach_regions(read_raw(args.data))
    all_rows, selections, coefficients = [], [], []
    for region in ("east", "central", "west"):
        frames = held_out_split(frame, region, seed=42, column=REGION)
        (train, val, test), _ = prepare_splits(frames, environment_column=REGION)
        for penalty in ("l2", "l1"):
            candidates = []
            # 任务3 第7步：仅用来源验证站点宏AUC选C；目标标签不参与选模。
            for c in (0.1, 1.0, 10.0):
                model = LogisticRegression(penalty=penalty, C=c, solver="liblinear", max_iter=5000, random_state=42)
                model.fit(train.x, train.y)
                p = model.predict_proba(val.x)[:, 1]
                per_site = score_rows(region, penalty, val, p, "source_validation")
                macro = np.mean([r["auc"] for r in per_site if r["scope"] == "site" and r["auc"] is not None])
                selections.append({"region": region, "model": penalty, "C": c, "source_macro_auc": float(macro)})
                candidates.append((macro, -c, model))
            _, neg_c, model = max(candidates, key=lambda x: (x[0], x[1]))
            name = f"logistic_{penalty}"
            for stage, split in (("source_validation", val), ("test", test)):
                all_rows += score_rows(region, name, split, model.predict_proba(split.x)[:, 1], stage)
            for feature, coef in zip(FEATURES, model.coef_[0]):
                coefficients.append({"region": region, "model": name, "C": -neg_c, "feature": feature, "coefficient": float(coef), "selected": bool(abs(coef) > 1e-8)})
        model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, l2_regularization=1.0, early_stopping=False, random_state=42)
        model.fit(train.x, train.y)
        for stage, split in (("source_validation", val), ("test", test)):
            all_rows += score_rows(region, "shallow_histgb", split, model.predict_proba(split.x)[:, 1], stage)
        if args.tabpfn_checkpoint:
            print(f"任务3 第7步：{region} TabPFN-v2；CPU可能耗时较长", flush=True)
            model = TabPFNClassifier(model_path=str(args.tabpfn_checkpoint.resolve()), device=args.device, n_estimators=4, random_state=42, ignore_pretraining_limits=True)
            model.fit(train.x, train.y)
            for stage, split in (("source_validation", val), ("test", test)):
                all_rows += score_rows(region, "tabpfn_v2", split, model.predict_proba(split.x)[:, list(model.classes_).index(1)], stage)
        pd.DataFrame(all_rows).to_csv(args.output / "metrics.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(selections).to_csv(args.output / "source_only_selection.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(coefficients).to_csv(args.output / "feature_coefficients.csv", index=False, encoding="utf-8-sig")
    table = pd.DataFrame(all_rows)
    sites = table.loc[(table.scope == "site") & (table.stage == "test")]
    sites.groupby(["region", "model"])[["auc", "auprc", "brier"]].mean().to_csv(args.output / "region_site_macro.csv", encoding="utf-8-sig")
    sites.groupby("model")[["auc", "auprc", "brier"]].mean().to_csv(args.output / "global_site_macro.csv", encoding="utf-8-sig")
    # 任务3 第7步：三个地区完整展示，避免仅报告东部最好的一种模型。
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = args.output / "figures"
    figures.mkdir()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key in zip(axes, ("auc", "auprc")):
        sites.groupby(["region", "model"])[key].mean().unstack().reindex(["east", "central", "west"]).plot.bar(ax=ax, rot=0)
        ax.set_ylabel(f"Equal-site macro {key.upper()}")
    fig.tight_layout()
    fig.savefig(figures / "low_dimensional_region_comparison.png", dpi=180)
    plt.close(fig)
    metadata = {"input_sha256": EXPECTED_HASH, "split_seed": 42, "sklearn": sklearn.__version__, "tabpfn_version": package_version, "checkpoint_sha256": checkpoint_hash, "checkpoint": str(args.tabpfn_checkpoint), "tabpfn_executed": bool(args.tabpfn_checkpoint), "source_training_only": True, "source_validation_used_for_logistic_C": True, "target_used_for_selection": False, "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "limitations": "原始数值编码不变；本步只筛预测能力；稀疏系数不是因果特征；经典基线不是2025新方法；不等于稀疏IRM或ISD复现。TabPFN-v2权重需人工核对官方来源。"}
    (args.output / "experiment_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"任务3 第7步完成：{args.output}")


if __name__ == "__main__":
    main()

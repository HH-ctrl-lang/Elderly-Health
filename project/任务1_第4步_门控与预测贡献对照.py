"""任务1，第4步：对照CausalGate分配权重与特征的实际预测贡献。

研究目标：
1. 比较门控权重与置换AUC/AUPRC变化的排名是否一致。
2. 对比来源验证集、留出地区测试集和两个预测头。
3. 整理若干预先指定的典型特征，形成任务1阶段小结。

运行：python project/任务1_第4步_门控与预测贡献对照.py
说明：只读取任务1前三步的汇总CSV，不训练、不改变模型、不读取个人样本。
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt, font_manager
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "outputs" / "analysis_2611"
GATE_FILE = BASE / "gate_step1" / "gate_matrix.csv"
CONTRIBUTION_FILE = BASE / "gate_step3" / "feature_contribution_summary.csv"
OUTPUT = BASE / "task1_step4_gate_alignment"

REGIONS = ("east", "central", "west")
HEADS = ("main", "zc")
SUBSETS = ("val", "test")
KEY_FEATURES = (
    "if_use_mobile_phone", "IADL", "age",
    "residence", "depression", "sleep_duration",
)

# 图表优先使用可用的中文字体；未安装时使用英文绘图标题，避免方框乱码。
FONT_NAMES = (
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
    "Noto Sans CJK JP", "WenQuanYi Micro Hei", "PingFang SC",
)


def configure_font():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in FONT_NAMES:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def load_tables():
    # 只提取三个留一地区折；站点折留待跨环境稳定性分析使用。
    matrix = pd.read_csv(GATE_FILE, encoding="utf-8-sig", index_col=0)
    region_keys = ["region:" + name for name in REGIONS]
    if not all(name in matrix.index for name in region_keys):
        raise ValueError("门控矩阵缺少东部、中部或西部留出折")
    if len(matrix.columns) != 19:
        raise ValueError("期望19个原始输入特征")
    gate = matrix.loc[region_keys].copy().astype(float)
    gate.index = pd.Index(REGIONS, name="region")
    gate = gate.reset_index().melt(
        id_vars="region", var_name="feature", value_name="gate_weight"
    )
    gate["gate_rank"] = gate.groupby("region")["gate_weight"].rank(
        ascending=False, method="average"
    )
    if not np.isfinite(gate["gate_weight"]).all():
        raise ValueError("门控表包含无效数字")

    # 置换数据必须完整：3地区 × 2集合 × 2预测头 × 19特征。
    imp = pd.read_csv(CONTRIBUTION_FILE, encoding="utf-8-sig")
    keys = ["region", "subset", "head", "feature"]
    if len(imp) != 3 * 2 * 2 * 19 or imp.duplicated(keys).any():
        raise ValueError("预测贡献表行数不对或包含重复的地区/数据集/预测头/特征")
    if (set(imp["region"]) != set(REGIONS)
            or set(imp["subset"]) != set(SUBSETS)
            or set(imp["head"]) != set(HEADS)):
        raise ValueError("预测贡献表中地区、数据集或预测头不完整")

    joined = imp.merge(gate, on=["region", "feature"], how="inner", validate="many_to_one")
    if len(joined) != len(imp) or joined[["gate_weight", "mean_auc_drop", "mean_auprc_drop"]].isna().any().any():
        raise ValueError("门控权重与预测贡献无法完整对应")
    for col in ("gate_weight", "mean_auc_drop", "mean_auprc_drop"):
        if not np.isfinite(joined[col].to_numpy(dtype=float)).all():
            raise ValueError(f"发现无效数值：{col}")
    joined["auc_drop_rank"] = joined.groupby(["region", "subset", "head"])[
        "mean_auc_drop"
    ].rank(ascending=False, method="average")
    joined["auprc_drop_rank"] = joined.groupby(["region", "subset", "head"])[
        "mean_auprc_drop"
    ].rank(ascending=False, method="average")
    return joined


def summarize_correlations(joined):
    # 相关系数只比较同一地区、同一预测头下19个特征的排序。
    rows = []
    for (region, subset, head), group in joined.groupby(
        ["region", "subset", "head"], sort=False
    ):
        rows.append({
            "留出地区": region,
            "数据集": subset,
            "预测头": head,
            "特征数量": len(group),
            "门控与AUC下降_Spearman": group["gate_weight"].corr(
                group["mean_auc_drop"], method="spearman"
            ),
            "门控与AUPRC下降_Spearman": group["gate_weight"].corr(
                group["mean_auprc_drop"], method="spearman"
            ),
            "AUC下降为正的特征数": int((group["mean_auc_drop"] > 0).sum()),
        })
    return pd.DataFrame(rows)


def plot_alignment(joined, head, use_cn):
    # 每张图三列分别表示三个留出地区；横轴是门控值，纵轴是AUC下降量。
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.5), sharex=True, sharey=True)
    short = {
        "if_use_mobile_phone": "mobile",
        "IADL": "IADL",
        "age": "age",
        "residence": "residence",
    }
    for ax, region in zip(axes, REGIONS):
        subset = joined[
            (joined["region"] == region)
            & (joined["subset"] == "test")
            & (joined["head"] == head)
        ]
        ax.scatter(
            subset["gate_weight"], subset["mean_auc_drop"],
            s=32, color="#567F9B", alpha=0.85,
        )
        for feature, label in short.items():
            row = subset[subset["feature"] == feature].iloc[0]
            ax.annotate(
                label, (row["gate_weight"], row["mean_auc_drop"]),
                xytext=(3, 4), textcoords="offset points", fontsize=8,
            )
        ax.axhline(0, color="#777777", linewidth=0.9)
        rho = subset["gate_weight"].corr(subset["mean_auc_drop"], method="spearman")
        if use_cn:
            ax.set_title(f"留出{dict(east='东部', central='中部', west='西部')[region]} | 排名相关={rho:.3f}")
            ax.set_xlabel("CausalGate门控权重")
        else:
            ax.set_title(f"Held-out {region} | Spearman={rho:.3f}")
            ax.set_xlabel("CausalGate weight")
        ax.grid(alpha=0.13)
    axes[0].set_ylabel("打乱特征后AUC下降量" if use_cn else "Original AUC - shuffled AUC")
    fig.suptitle(
        f"任务1，第4步：门控与预测贡献对照（{head}头）"
        if use_cn else f"Task 1, Step 4: gate weight vs AUC reliance ({head} head)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(
        OUTPUT / f"任务1_第4步_{head}_门控贡献散点图.png",
        dpi=180, bbox_inches="tight"
    )
    plt.close(fig)


def write_report(correlations, joined):
    # 报告中的数字只来自CSV；不将相关性解释成因果效应。
    lines = [
        "# 任务1，第4步：门控分配与预测贡献一致性分析",
        "",
        "## 一、分析方法",
        "",
        "使用任务1第1步的连续门控权重与第3步的特征置换结果，"
        "分别在东部、中部、西部、来源验证集和留出测试集、main与zc预测头中，"
        "计算19个特征的门控权重与AUC/AUPRC下降量之间的Spearman排名相关系数。",
        "",
        "AUC下降量 = 原始AUC - 打乱该特征后的AUC。正值表示预测排序变差；"
        "负值表示置换后排序反而改善。门控分数衡量输入在两条分支间的分配，"
        "不等于因果重要性或最终预测贡献。",
        "",
        "## 二、留出地区测试集的结果",
        "",
        "| 留出地区 | 主头：门控/AUC排名相关 | zc头：门控/AUC排名相关 |",
        "|---|---:|---:|",
    ]
    for region in REGIONS:
        test = correlations[
            (correlations["留出地区"] == region) & (correlations["数据集"] == "test")
        ]
        main = test[test["预测头"] == "main"].iloc[0]
        zc = test[test["预测头"] == "zc"].iloc[0]
        lines.append(
            f"| {region} | {main['门控与AUC下降_Spearman']:.3f} | "
            f"{zc['门控与AUC下降_Spearman']:.3f} |"
        )
    lines.extend([
        "",
        "## 三、解释与限制",
        "",
        "- 三个地区的门控排名与置换预测贡献并非高度一致；不能把0.5以上当作真实因果变量。",
        "- 应同时查看IADL、年龄、是否使用手机、城乡居住类型等对照案例，而不是只看Top-5。",
        "- 置换在原站点内部完成，尽量保留站点分布，但可能破坏变量间的真实相关结构。",
        "- 来源验证集与留出测试集的作用可能不同，表明存在跨环境的不稳定性，尚不能确定具体原因。",
        "- 五次置换的标准差只描述随机置换的波动，不是训练种子间方差或置信区间。",
        "- 所有训练模型均固定，不依据留出测试集调整门控或模型参数。",
        "- **结论范围：描述性模型诊断，不识别因果效应。**",
        "",
        "## 四、后续研究建议",
        "",
        "任务1可以先归档小结。下一阶段按照导师要求，"
        "优先分析东部与中、西部的阳性率、19个变量分布、缺失情况和标签内差异，"
        "再判断预测效果较差是否与地区数据分布差异有关。",
        "",
    ])
    (OUTPUT / "任务1_第4步_阶段小结.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"输出目录已存在，为保护原始结果不覆盖：{OUTPUT}"
        )
    joined = load_tables()
    correlation = summarize_correlations(joined)
    OUTPUT.mkdir(parents=True)
    joined.to_csv(
        OUTPUT / "任务1_第4步_门控与贡献逐特征对照.csv",
        index=False, encoding="utf-8-sig"
    )
    correlation.to_csv(
        OUTPUT / "任务1_第4步_相关性汇总.csv",
        index=False, encoding="utf-8-sig"
    )
    joined[joined["feature"].isin(KEY_FEATURES)].to_csv(
        OUTPUT / "任务1_第4步_重点特征对照.csv",
        index=False, encoding="utf-8-sig"
    )
    use_cn = configure_font()
    for head in HEADS:
        plot_alignment(joined, head, use_cn)
    write_report(correlation, joined)
    print("任务1，第4步：完成门控与预测贡献的一致性分析")
    print("处理地区：东部、中部、西部；19个特征，2种预测头，2个数据集")
    print("输出目录：", OUTPUT)
    print("下一步：阅读中文阶段小结与两张散点图，不要根据测试集重新选特征。")


if __name__ == "__main__":
    main()

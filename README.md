# 老年认知健康：因果表征与跨环境泛化实验

本仓库包含历史V4/V5因果表征探索，以及经过整理的19特征老年认知功能受损预测流程。**当前完成阶段：实验2610（2026年10月）**，重点研究CausalGate软约束、环境粒度与开源域泛化方法的探索性表现。

> 当前结果属于观察性数据上的预测/域泛化实验，不构成健康变量的因果效应识别；所谓`z_c`“因果表示”仍为待验证的模型设计假设。

## 文档导航

- **[阶段成果总结](doc/实验2610_阶段性总结_20261021.md)**：研究逻辑、V1/V2与MLP、VREx、RDM、GroupDRO的结果与结论。
- **[研究方法与复现指南](doc/研究方法与复现指南.md)**：代码清理要点、网络与损失、数据切分、完整命令、外部论文来源及注意事项。
- [源代码](project/src/elderly_health/) · [实验配置](project/configs/) · [已提交结果](outputs/) · [自动化测试](tests/)

## 数据及环境

- 输入：`project/data/data_mission2.csv`（GBK）；目标`cognitive_impairment`，原值1/2转换为0/1。
- 4,769个有效标签样本、19个数值编码输入特征；12个监测站点，以及按研究方案划分的东部（北京/山东）、中部（吉林/江西）、西部（广西/宁夏）。
- 地区定义是**本研究数据分组约定**，不是通用行政区划标准；正式报告时仍需核对调查方案。
- 参考运行环境：Python 3.10.20、PyTorch 2.4.1+cpu；**不需要GPU**。
- 建议：先创建Python环境并执行`python -m pip install -r requirements.txt`，再执行`git lfs pull`。Git LFS指针不是实际的模型/数据内容。

## 主要运行入口

从仓库根目录执行：

```bash
python -m unittest discover -s tests -v

# V1：12个站点作为训练环境；V2：3地区作为训练环境
python project/run.py train --config project/configs/v2_site_v1_selection91.json --output outputs/recheck-site-42
python project/run.py train --config project/configs/v2_region_v1_selection91.json --output outputs/recheck-region-42

# 留一监测站点：各12折
python project/run.py loeo --holdout-level site --config project/configs/v2_site_v1_selection91.json --output outputs/recheck-site-loeo-42
python project/run.py loeo --holdout-level site --config project/configs/v2_region_v1_selection91.json --output outputs/recheck-region-loeo-42

# 留一地区：各3折
python project/run.py loeo --holdout-level region --config project/configs/v2_region_v1_selection91.json --output outputs/recheck-v2-loro-42

# 开源方法表格二分类适配：均为留一地区3折
python project/run_external_vrex.py --mode loro --environment-granularity region --output outputs/recheck-vrex-loro-42
python project/run_external_rdm.py --mode loro --environment-granularity region --output outputs/recheck-rdm-loro-42
python project/run_external_groupdro.py --mode loro --environment-granularity region --output outputs/recheck-groupdro-loro-42
```

各入口拒绝覆盖已有目录。以上`recheck-`路径供未来独立复核；**本分支已有正式结果，不必重新运行**。正式结果及更多参数见[复现指南](doc/研究方法与复现指南.md)。

## 方法边界

- V5由连续CausalGate、`z_c`/`z_s`双分支、联合分类头与辅助头组成，结合自定义IRM(v11)、环境分类和HSIC等约束。
- `gate>0.5`只用于特征展示，不是硬性筛选；`mask(1-mask)`为门控极化惩罚，非真正信息熵。
- LOEO/LORO检验未见监测站点或未见地区的**预测泛化**，并不能证明`z_c`或筛出的变量有因果效应。
- 外部VREx/RDM/GroupDRO移植了作者**核心优化机制**，同时适配CPU、19维输入和二分类，并非原图像/文本论文的完整复现。

仓库历史脚本与旧研究记录保留在Git历史和`project/legacy/`。如需恢复旧的V4说明或早期研究设想，可从本分支整理前提交`38a77cf`检索。新阶段研究应另建分支，不把尚未完成的实验写成已有成果。

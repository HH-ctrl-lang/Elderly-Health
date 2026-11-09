# 任务3 第7步：低维方法筛选与2025年以来文献

本轮固定19维CSV、原数值编码和来源训练/验证划分；三个地区逐一留出。目标是先确认简单原特征方法是否优于已有复杂解耦结果，再决定深入实现哪篇不变性方法。本轮不继续重编码或追溯原工作簿。

## 已核查的近期文献

| 论文 | 正式发表 | 与当前任务关系 | 当前决定 |
| --- | --- | --- | --- |
| [Computationally Efficient Methods for Invariant Feature Selection with Sparsity](https://proceedings.mlr.press/v286/du25b.html) | UAI 2025 | 在特征层面用稀疏IRM与迭代硬阈值寻找不变特征，比强行构造两块深层表示更接近当前问题 | 下一优先候选；需核对完整目标函数、理论数据条件与二分类适配，当前代码未实现此论文 |
| [Invariant Subspace Decomposition](https://jmlr.org/papers/v26/24-0699.html) | JMLR 2025 | 将线性条件分成不变和时间变化残余成分；可启发低维空间处理 | 时间变化线性条件与当前县区二分类不等价；暂不直接套用或声称理论保证 |
| [Accurate predictions on small data with a tabular foundation model](https://www.nature.com/articles/s41586-024-08328-6) | Nature，2025-01-08 | TabPFN-v2用于小样本表格预测；本数据规模适合做有限预测参照 | 已写可选本地模型分支；不宣称因果解耦或跨地区稳健性已成立 |
| [Understanding the Limits of Deep Tabular Methods with Temporal Shift](https://proceedings.mlr.press/v267/cai25j.html) | ICML 2025 | 强调划分/验证政策和表格表示在偏移下的影响 | 当前没有时间轴，不能把县区编号当时间或直接套用其时间嵌入 |

这些论文提供候选机制和比较对象，没有证明“低维且语义明确必然导致表示崩溃”。此前V5维度集中也不是该命题的证据。稀疏逻辑回归系数不等于已识别的因果特征。

## 运行

现有环境直接运行三个经典对照，无新依赖：

```bash
python project/src/analysis/task3/low_dimensional_method_screen.py
```

输出 `outputs/analysis_2611/task3_step7_low_dimensional_method_screen/`。逻辑回归L2和L1的C均从0.1/1/10按来源验证站点宏AUC选择；浅树结构固定。所有模型只拟合来源训练样本。三个经典对照不是2025新算法，也不是论文稀疏IRM的复现。

TabPFN分支需要额外安装官方 `tabpfn` 包并准备官方 **v2分类检查点**，建议独立环境，避免改变现有PyTorch 2.4.1环境。官方来源：[PriorLabs/TabPFN](https://github.com/PriorLabs/TabPFN)、[Prior-Labs/TabPFN-v2-clf](https://huggingface.co/Prior-Labs/TabPFN-v2-clf)。记录真实包版本和检查点SHA256；不能默认加载2026年的最新版后称为Nature 2025复现。

```bash
python project/src/analysis/task3/low_dimensional_method_screen.py --tabpfn-checkpoint /absolute/path/to/tabpfn-v2-classifier.ckpt --output outputs/analysis_2611/task3_step7_low_dimensional_with_tabpfn
```

Windows将路径替换为自己的本地文件路径。CPU保持完整来源训练人群，耗时可能较长；有兼容CUDA环境时可加 `--device cuda`。本次未执行TabPFN，不能视作已跑通的候选结果；版本/API或检查点不兼容时应保留错误后核对，不自动换模型或抽样改人群。

## 如何阅读

- `metrics.csv` 同时记录来源验证和留出地区测试，地区合并与站点评价分开。
- `region_site_macro.csv`、`global_site_macro.csv` 是站点等权平均；比较任务3历史站点宏结果时必须对齐该口径。
- `feature_coefficients.csv` 保存原特征稀疏系数；不称作因果效应。
- `source_only_selection.csv` 保存每个候选C的来源验证结果。
- `experiment_metadata.json` 明确TabPFN是否实际运行。无检查点运行只有经典对照。

这一步先回答“更适合表格数据的预测方法是否有效”。若预测提升，仍需另外验证是否来自更好的表示、是否跨地区稳定，不能据此认定复杂隐层发生崩溃。本轮模型损失未设置pos_weight=3，与历史MLP是算法筛选而非单一机制控制；要归因于结构需追加统一损失的对照。

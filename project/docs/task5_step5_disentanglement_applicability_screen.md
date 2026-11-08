# 任务5 第5步：新解耦方法的适用性筛选

核查日期：2026-11-08。仅文献与代码入口核查；本步没有训练模型，也没有生成新的实验结果。

## 目的与当前决定

目的仍是解释跨地区预测为什么不稳定，并判断解耦能否改善真实数据的预测。恢复隐藏因素的理论保证，不等于提高我们的认知健康预测效果。

本轮核查三个候选后，暂不增加新的复现训练。原因是关键前提与当前数据尚不匹配，而不是证明这些方法没有价值。保留离散潜变量方法作为后备，不继续调条件MMD或MIR/VIR参数。

## 已有结果决定投入范围

- 第3步：条件MMD相对同结构对照，全局站点宏AUC约增加0.0014，AUPRC约下降0.0021；没有明确提升。
- 第4步：MIR相对线性对照，全局站点宏AUC增加0.001539，AUPRC增加0.002305；只在留出西部时选择非零约束，东部和中部不变。
- VIR在三个留出地区均选择零约束，因此与线性对照相同。
- 这些是单次既定划分下的观测差异，未据此证明统计显著或因果有效。

结果表：[第4步差值](../../outputs/analysis_2611/task5_step4_mir_vir_health_screen/candidate_macro_differences.csv)。

相关图：
- [各站点AUC变化](../../outputs/analysis_2611/task5_step4_mir_vir_health_screen/figures/site_auc_changes.png)：结合差值表，MIR的改善集中在西部，不能概括为所有地区共同受益。
- [地区与总体表现](../../outputs/analysis_2611/task5_step4_mir_vir_health_screen/figures/region_and_global_macro_performance.png)：必须比较候选与同结构对照；不能把线性模型本身相对历史MLP的差异全部记到MIR名下。

## 候选一：CREATOR

**Linear Causal Representation Learning by Topological Ordering, Pruning, and Disentanglement**，Chen、Liu、Wang，ICML 2026。

- [会议正式页面](https://proceedings.mlr.press/v306/chen26cj.html)
- [全文](https://arxiv.org/html/2509.22553v2)
- [作者代码](https://github.com/utulie/code_for_linear_crl_paper_creator)

方法先恢复潜变量的先后关系，再删去不必要的图连接，最后进一步分离潜变量。其设定包括线性潜在结构、跨环境不变的线性混合、独立外生噪声且至多一个高斯分量，以及环境变化的非退化条件。定理2.7还要求至少d个环境，d为潜变量维数。

当前判断：暂不实施。我们的类别/有序变量不是已被验证的连续线性混合观测；不同站点也不能自动视作满足论文非退化条件。每个留出地区实验只有8个训练站点；如果把19个观测变量直接当作19维潜变量，环境数量也不满足该定理要求。选更小d虽然可以缓解数量限制，却不能同时解决其他前提。以上属于适用性判断，不是实验失败结论。

代码入口已确认：main_select.py为合成实验，story_experiments2.py为真实数据实验，main_hard_topo.py为消融。尚未执行或完成逐函数审计。

## 候选二：函数正交解耦

**Unsupervised Disentanglement Without Compromises: How Functional Orthogonality Enforces Identifiability**，Simon、Frossard、De Vleeschouwer，ICML 2026。

- [会议正式页面](https://proceedings.mlr.press/v306/simon26b.html)
- [全文](https://arxiv.org/pdf/2606.21385)

方法要求不同隐藏因素对观测产生的局部变化方向正交，并用带正交约束的归一化流进行验证。其识别结果依赖潜变量定义域允许所有因素取值组合；它本身不要求因果假设，因此不能直接把分出的因素称为因果变量。

当前判断：暂不实施。论文的连续生成映射及潜变量取值组合条件，尚未在当前混合类型数据中建立。不能仅凭观测变量存在取值限制就断言潜变量条件必然违反；同样不能默认这些条件成立。把类别编码加噪声后直接套流模型，会引入新的数据处理和方法选择，当前收益依据不足。

## 候选三：DCRL

**Discrete Causal Representation Learning**，Zhang、Wang、Gu，2026年预印本。

- [作者论文列表与发表状态](https://yuqigu.github.io/publications/)
- [作者提供的全文](https://yuqigu.github.io/assets/pdf/DCRL_Mar07_2026.pdf)
- arXiv编号：2603.25017。

方法用离散潜变量图与连接潜变量、观测指标的稀疏测量图，解释连续、计数及二元观测。流程是估计生成模型、重采样潜变量，再发现潜变量之间的图结构。

当前判断：保留为后备，暂不训练。作者列表仍归入Preprints，不能称为已发表顶会论文。它需要适当的测量模型及识别条件；我们的19个变量不是已经设计好的重复测量指标，能否满足条件需要单独论证。论文重点也是潜在结构恢复，而非跨地区分类提升。因此不因“支持离散数据”就直接认定适合本项目。

## 后续范围

1. 新增训练前，先说明候选如何适配类别/有序特征、是否需要额外数据、怎样与同结构无约束模型比较。
2. 若继续采用表示学习，优先保留原始变量预测路径，再检查额外表示是否有增益；这是下一步实验设计方向，并非本步完成的实现或文献结论。
3. 新模型的选择与预处理只用训练来源数据；留出地区仅用于最终评估。固定既有划分，报告12站点及3地区，不只关注一个站点。
4. 既有220202与360102的案例分析方向继续保留。若全局方法没有更有价值的信号，可返回该方向。
5. 本步仅新增本文档，没有修改数据、训练脚本或既有结果，不需要用户运行命令。

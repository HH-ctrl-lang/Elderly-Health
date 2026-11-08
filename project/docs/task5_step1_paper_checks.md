# 任务五 第1步：两篇文献核查与解耦机制复现

本步骤重点是因果解耦。研究实验由用户运行，不在助手环境运行。

## 已核查的原始来源

1. Saengkyongam et al. **Anti-causal domain generalization: Leveraging unlabeled data**.
   ICML 2026，PMLR 306。
   - 正式页面：https://proceedings.mlr.press/v306/saengkyongam26a.html
   - 本次核对全文版本：https://arxiv.org/pdf/2602.17187 （v2，2026-06-17，24页）。
   - 检查位置：Setting 3.1、Remark 3.2、§4–6、附录E。
2. Yao et al. **Unifying Causal Representation Learning with the Invariance Principle**.
   ICLR 2025。
   - 正式全文：https://proceedings.iclr.cc/paper_files/paper/2025/file/85381f4549b5ddf1d48e2e287d7d3d15-Paper-Conference.pdf
   - 检查位置：§2–3、§5.1–5.2、附录F.1–F.2、Table 3。
   - 原文明确链接的示例：https://github.com/CausalLearningAI/ISTAnt/blob/main/experiments/invariance.ipynb
   - 本次看到的示例仓库main树SHA：46cee608eae29c699119557595aa9009ac6216c7。

## 方法和适用边界

| 问题 | 第1篇：ICML 2026 | 第3篇：ICLR 2025 |
|---|---|---|
| 核心 | 从无标签训练环境的均值/协方差变化构造MIR/VIR正则 | 信息保留与已知不变性相结合，识别潜变量块 |
| 是否直接做表征拆分 | MIR/VIR本身不学习因果/非因果双编码器 | §5.2使用自编码器和指定潜变量块的MMD |
| 关键边界 | Setting 3.1不允许预测变量导致结局；不是必须每个特征都由结局直接造成 | 需有足够的不变性信息；任意地区编号不能自动充当已知干预 |
| 与当前健康数据的差距 | 年龄等变量与因果方向、环境对结局的影响均需另行论证 | 类别编码未被连续变量理论覆盖；真实潜变量未知，不能直接验证其恢复率 |
| 本轮状态 | 方法及假设已核查，保留后续对照，不声称已复现 | 提供§5.2缩小规模的独立机制复现，等待用户运行 |

第1篇Remark 3.2允许X与Y仅由共同潜在状态关联。附录E还讨论条件于观测W的扩展，
但需要W阻断环境到结局的路径，不能任意指定年龄/性别便宣称满足。
本轮没有确认第1篇的官方实现仓库，不以搜索未找到来断言作者没有公开代码。

第3篇不是一套可以直接替换现有模型的统一新网络。其§5.1生态实验用的是V-REx；
本轮关注更接近解耦问题的§5.2连续低维模拟实验。原文结论中明确将离散变量和有限样本保证留作后续研究。

## 本轮可执行步骤

入口：`project/src/analysis/task5_step1_invariance_disentanglement_pilot.py`

```bash
python project/src/analysis/task5_step1_invariance_disentanglement_pilot.py
```

输出：`outputs/analysis_2611/task5_step1_invariance_disentanglement_pilot/`

- 使用原文F.1的三变量链参数。改变z2时，下游z3继续使用旧z2；这叫nintervention，不能当作真实因果干预。
- 用同一个可逆混合函数生成观测，两组使用相同数据、初始参数及独立打乱的环境批次。
- 比较自编码器重建，与自编码器重建加指定块[h1,h3]的MMD约束。
- 真实z不进入编码器训练。训练固定100轮，不能根据潜变量恢复成绩选模型。
- 冻结训练后的模型，再用固定核岭读出器在独立模拟测试集评价z1、z2、z3的信息。
- 同时输出完整表征、原始观测和真实[z1,z3]的参考读出成绩。
- 原文是块识别，不要求h1单独等于z1或h3单独等于z3；不能据此声称恢复了因果图。

## 与原论文的差别，必须在分析时保留

本脚本是**独立、缩小规模的机制复现**，不是作者代码的原样执行，也不用于复现原文具体R²数字。

| 项目 | 原文Table 3 | 本轮 |
|---|---|---|
| 样本规模 | 200,000 | 每环境训练6,000，独立评估2,000 |
| 训练轮次 | 500 | 100 |
| 批量 | 4,000 | 每环境256 |
| 混合 | 三层可逆MLP，表中隐藏宽128 | 三层3×3正交映射，中间可逆LeakyReLU，保证全局可逆 |
| 编码器/解码器 | 三层MLP，隐藏128 | 同为三层，隐藏128，三维潜空间 |
| 优化器 | Adam，学习率1e-4 | 相同；CPU，两线程 |

另行明确的实现选择：训练集观测标准化；LeakyReLU斜率0.2；固定多尺度高斯核；
KRR输入及目标仅按读出器训练样本标准化，alpha=0.01，gamma=1/输入维数，最多1024个训练读出样本。
这些设置不能冒充原文未完整给出的参数。三次试验使用三个独立抽取的nintervention均值，结果仅作描述。

## 读取结果时关注什么

1. `latent_recovery.csv`：指定块是否保留z1、z3的信息，以及是否仍包含z2信息；结合两个参考输入判断。
2. `representation_diagnostics.csv`：重建误差、各维标准差、数值秩，以及原始/标准化后的块MMD和线性环境读出AUC。
3. `training_history.csv`：约束变小是否伴随重建能力变差。MMD单独下降不算解耦成功。
4. 两张图只展示模拟结果，不预设一定复现成功。三次试验都保留，不挑最好的一次。

只有用户运行并提交这些结果后，再讨论扩大规模或是否有理由向19维健康数据适配。
本轮不对健康数据使用MMD，也不把跨地区分布一致当作因果变量证据。

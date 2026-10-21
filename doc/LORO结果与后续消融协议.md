# LORO留一地区结果与机制消融协议（2026-10-21）

## 目的与数据口径

本文件总结已提交的 `outputs/v2-site-loro-selection91-42`、`outputs/v2-region-loro-selection91-42`；
两组配置共享原始数据SHA256：`969f0c1c82faafcfa402f37572e4a8ceb324492062cf1e590465f82dc6b5b6de`。
外部留出东部1570人（阳性237）、中部1598人（阳性142）、西部1601人（阳性293），三组测试样本互不重叠，总计4769人；
每折仅在其他地区的训练/验证数据上拟合预处理与选择模型。
`site`配置根据剩余监测站点建模环境，`region`配置根据剩余两个地区建模环境。
两个配置中的MLP基线每折结果一致。

## LORO各地区结果

|留出地区|MLP AUC|站点V5 AUC|地区V5 AUC|MLP AP|站点V5 AP|地区V5 AP|
|---|---:|---:|---:|---:|---:|---:|
|east|0.6563|0.6897|0.6835|0.2607|0.3053|0.3013|
|central|0.7407|0.7388|0.7418|0.2255|0.2133|0.2132|
|west|0.7495|0.7444|0.7480|0.3647|0.4090|0.4158|
|三地区等权宏平均|0.7155|0.7243|0.7244|0.2836|0.3092|0.3101|

注意：三地区宏平均与LOEO的12站点宏平均是不同聚合单位，不能直接据此推断LORO条件比LOEO更容易或更难。
同一地区的两个V5环境粒度配置没有稳定的性能胜负；与MLP比较则东、西地区AP可能有收益，中部没有。
LORO只用seed=42且只有三个地区，不能认定模型普遍改进跨区域泛化。
所有结果是外部预测能力，不证明任一健康特征的因果效应。

## 表征与门控诊断

LORO各折所选V5模型第91轮后才允许保存，已检查选中轮次均不早于第91轮。
当前门控>0.5所选特征在三个外部地区间明显变化，且部分折没有超过0.5的特征。
`zc`为模型内经过辅助标签损失训练的预测头，不能视为独立训练的因果检验。
LORO测试集全部是未见地区，现有环境探针`n_known=0`输出`null`，
并非无环境信息。不可把测试集探针缺失当作模型的解耦成功证据。

## 下一阶段：预先规定的单因素机制消融

首选`environment_granularity=region`，沿用已完成的`v2_region_v1_selection91.json`完整约束模型，
每项仅关闭一个损失模块，不同时更改数据、随机种子、优化器、主预测结构或模型选择策略：

|实验配置|关闭内容|修改|
|---|---|---|
|`ablation_region_no_irm_selection91.json`|项目自定义v11梯度方差惩罚|irm_mode=none,lambda_irm=0|
|`ablation_region_no_hsic_selection91.json`|HSIC独立性约束|independence=none,lambda_independence=0|
|`ablation_region_no_env_selection91.json`|环境分类辅助损失|lambda_env=0|

上述三项不是三种全新算法，而是完整V5方法的三个单模块消融。
运行一个完整的三地区LORO实验会训练3个V5加3个MLP；当前CLI自动重算MLP，已知完整模型的MLP结果可以用来比较，重复计算仅增加CPU成本。

```bash
git switch refactor/code-cleanup
git pull --ff-only origin refactor/code-cleanup
python -m unittest discover -s tests -v

# 优先执行IRM模块消融，完成并审查后再决定是否执行其余两项
python project/run.py loeo --holdout-level region --config project/configs/ablation_region_no_irm_selection91.json --output outputs/ablation-region-no-irm-loro-42

python project/run.py loeo --holdout-level region --config project/configs/ablation_region_no_hsic_selection91.json --output outputs/ablation-region-no-hsic-loro-42

python project/run.py loeo --holdout-level region --config project/configs/ablation_region_no_env_selection91.json --output outputs/ablation-region-no-env-loro-42
```

每个新输出目录应当不存在，避免覆盖历史结果。
先检查全部三地区、各自V5与MLP的结果文件，以及训练日志与配置记录。
主指标按地区分别报告AUC、AP、Brier，辅助指标包括Recall、F1、主预测与zc预测差异、门控连续分数。
优先按方向和一致性解释，不仅依据哪个配置在测试集数值最高来选新超参数。
考虑已多次分析同一外部数据，需要以后固定验证集调参协议和更多随机种子/独立队列，
不把这些反复使用的测试集当作新的调参集。

## 待检验的研究问题

1. IRM(v11)是否增加了跨地区预测能力，还是对某些地区起负作用？
2. HSIC是否实现更清晰的信息分离，代价是否为预测性能下降？
3. 环境分类辅助任务是否真正帮助两个表征承担不同角色？
4. 若去掉某一约束后结果改善，这是一项消融发现，并不自动说明学到真实因果表示。

后续再基于明确结果决定是否扩展到`site`训练粒度、多种子和外部方法（如VREx/GroupDRO）。

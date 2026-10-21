# 《实验2610》开源方法试用（三）：GroupDRO

## 这次试用要回答什么？

仅探索当训练更关注**高预测损失的地区**时，能否改善新地区上的老年认知受损预测。不要求严格统计证明，也不直接推断因果关系。

**论文**：Sagawa et al., *Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization*, ICLR 2020
- 原文：https://arxiv.org/abs/1911.08731
- 作者官方代码：https://github.com/kohpangwei/group_DRO
- 固定源SHA：`cbbc1c5b06844e46b87e264326b56056d2a437d1`
- 源文件：`loss.py` -> `LossComputer.compute_robust_loss`
- MIT许可全文：`project/external/GROUPDRO_LICENSE.txt`
- 本项目最小适配：`project/src/elderly_health/groupdro_adapter.py`
- 运行入口：`project/run_external_groupdro.py`

核心代码保留作者的乘法式动态组权重：

```text
weights[g] <- weights[g] * exp(robust_step_size * group_loss[g])
weights <- weights / sum(weights)
robust_loss <- sum(weights[g] * group_loss[g])
```

组风险采用逐组样本的平均二元交叉熵。为了与当前老年健康任务兼容，我们使用此前的19特征SimpleMLP、正例权重为3的二分类BCE、每个训练环境32条的均衡小批次、Adam、来源验证集选最佳模型和CPU。
**上述网络、采样、正例损失权重和优化器均为本项目适配，不是原论文全部默认设置，不能称为官方图像或文本实验原样复现。**

对训练中每个组的权重和组风险，保存到`training.json`。每个外部地区只使用其他地区来训练、拟合缺失值和标准化；不会使用外部测试地区做模型选择。

## 第一步：更新分支并做基础检查

在本地仓库根目录执行：

```bash
git fetch origin
git switch feature/experiment2610-open-source-baselines
git pull --ff-only origin feature/experiment2610-open-source-baselines
git lfs pull

python -m unittest discover -s tests -v
python -m compileall -q project/src/elderly_health project/run_external_groupdro.py
```

如果测试未通过，请保留报错；不要直接运行正式实验。

## 第二步：仅留出东部进行三轮试运行

用于发现代码和数据接口错误，不计入论文结果：

```bash
python project/run_external_groupdro.py --mode loro --held-out east --epochs 3 --selection-start-epoch 1 --output outputs/external-groupdro-smoke-east-42
```

## 第三步：正式试用，完整留出东、中、西三个地区

```bash
python project/run_external_groupdro.py --mode loro --environment-granularity region --output outputs/external-groupdro-region-loro-42
```

默认seed=42、300轮上限、第91轮后选最佳验证模型，LORO三个留出地区各训练一个网络。如果CPU耗时较长，不需要为了“更快”随意降低正式运行轮次。

如正式试用后有兴趣，也可以之后探索`--environment-granularity site`，使留一地区时剩余8个监测站点成为训练组，但**本轮不要求**。

## 查看指标

结果目录`outputs/external-groupdro-region-loro-42/summary.csv`自动汇总三个地区的AUC、AUPRC、Recall、F1和Brier；每个地区另有`training.json`、`metrics.json`、`predictions_test.csv`、`best_model.pt`和配置溯源。

观察两点即可：
1. 最困难地区的AUC/AUPRC是否有所改善，还是只在已有较容易的地区改善？
2. 相比此前V5、VREx和RDM，模型是否存在明显的召回率或概率预测质量问题？

参考已提交：
- `outputs/v2-region-loro-selection91-42/summary.csv`：MLP、V5
- `outputs/external-vrex-region-loro-42/summary.csv`：VREx
- `outputs/external-rdm-region-loro-42/summary.csv`：RDM

这是探索性跨方法比较，不需要本轮做多种子或显著性论证。原GroupDRO官方代码有CUDA硬编码，我们的版本已更改为设备无关张量，**尚未在用户真实CPU环境执行验证**，请先运行测试。上传含健康调查逐人预测结果前，务必核实公开仓库的数据授权与脱敏要求。

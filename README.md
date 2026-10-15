# 老年认知健康实验代码

本仓库包含历史实验及整理后的19特征预测流程。当前维护入口为 `project/run.py`，
适用于 Python 3.10、PyTorch 2.4.1 CPU。所有训练从原始CSV开始，训练集单独拟合
缺失值填充和标准化，实验产物写入独立目录。

## 环境

在现有 Python 3.10.20、PyTorch 2.4.1+cpu 环境中安装其余依赖即可：

```bash
python -m pip install -r requirements.txt
```

如需新建环境，请先安装CPU版PyTorch，再安装其余依赖：

```bash
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

因果发现的历史脚本另需各自使用的算法库，不在本次核心依赖中。当前主流程不需要
DirectML、CUDA、networkx或Optuna。Git LFS文件需 `git lfs pull` 后才是实际数据。
不要把LFS指针文本当作NPZ或模型文件读取。

## 运行

以下命令从仓库根目录执行。默认输入是 `project/data/data_mission2.csv`，默认设备是CPU。
每次指定新的输出目录；已存在的输出目录会被拒绝，防止覆盖实验。

```bash
# 检查预处理，保存3337/716/716划分、样本行ID和训练集拟合对象
python project/run.py preprocess --seed 42 --output outputs/preprocess-42

# 同一数据划分的普通MLP对照
python project/run.py train --model baseline --seed 42 --output outputs/baseline-42

# V5：联合zc+zs主预测，以及单独zc预测头，无图先验
python project/run.py train --config project/configs/v5.json --output outputs/v5-42

# 显式开启修复后的图先验；图缺失或变量不匹配时直接报错
python project/run.py train --config project/configs/v5.json --graph project/data/preprocessed/causal_graph.dot --output outputs/v5-graph-42

# V4：仅zc主预测，使用标准IRM和Cosine
python project/run.py train --config project/configs/v4.json --output outputs/v4-42

# 放松门控约束：软均值稀疏、关闭图损失
python project/run.py train --config project/configs/v1_relaxed.json --output outputs/relaxed-42

# 一个留站点折的试运行，自动用相同数据行训练MLP对照
python project/run.py loeo --held-out 110113 --config project/configs/v5.json --output outputs/loeo-one-42

# 完整12折；每折从原始数据重新拟合预处理
python project/run.py loeo --config project/configs/v5.json --output outputs/loeo-42
```

可用 `--input` 和 `--encoding` 指定原始CSV。训练超参数放在JSON中，通过 `--config` 传入。
命令行的 `--model`、`--seed`、`--epochs`、`--device`、`--threads`、`--probe-epochs`、
`--irm-mode` 可覆盖配置。未知JSON键直接报错。短训练只用于检查流程：

```bash
python project/run.py train --epochs 3 --probe-epochs 0 --output outputs/smoke
```

3轮检查不会进入第20/50轮启用的正则阶段。检查全部损失的反向传播请运行测试。
V1预设只改变稀疏模式和图权重，不调整IRM、HSIC、预测头及其他超参数。
环境粒度的地区/省份映射及外部算法暂未纳入这次整理，地区归属需要按调查方案确认。

## 目录

| 路径 | 用途 |
|---|---|
| `project/run.py` | 统一入口 |
| `project/src/elderly_health/config.py` | 验证训练配置 |
| `project/src/elderly_health/data.py` | 原始数据、划分及训练集预处理 |
| `project/src/elderly_health/models.py` | 公共编码器、V4/V5和MLP |
| `project/src/elderly_health/losses.py` | IRM、Cosine及HSIC |
| `project/src/elderly_health/graph.py` | DOT角色、图损失及Jaccard |
| `project/src/elderly_health/training.py` | 训练、早停和最佳权重恢复 |
| `project/src/elderly_health/evaluation.py` | 指标、探针及表征导出 |
| `project/src/elderly_health/cli.py` | 实验组织及LOEO |
| `project/legacy/` | 本次替换入口的原始代码快照，只供追溯 |
| `project/src/train_v1-v3/`、`predata*.py`、`causaldiscovery/` | 尚未迁移的历史探索代码 |
| `outputs/` | 新实验输出，Git忽略 |

`project/src/train/train_v4.py`、`train_v5.py`、`loeo_validation.py`及新预处理旧路径
已变为兼容命令入口，必须传 `--output`。它们不再提供原脚本的全局变量或训练函数API。
已有自动化脚本应迁移到统一入口，原代码和旧记录均保留。

## 输出和解释

每次训练生成配置及输入SHA256、环境和依赖版本、预处理对象、数据划分、
最佳权重、逐轮损失、评估指标及三个数据划分的预测/表征CSV。
CSV的 `sample_id` 是原始CSV的零起始行号，跨划分和运行保持一致，不替代调查个人ID。
新NPZ和CSV的标签统一为0/1，并显式记录编码；不要交给旧版按1/2处理的加载器。

`metrics.json` 分别记录联合主预测与zc头预测；V4两者使用同一个头。
环境探针使用训练集拟合，报告多数类基线和平衡准确率；留出环境不在训练类集合中时，
该探针指标记为null，并报告未见样本数，不能当作“环境信息为零”。
`zc`结果是训练过的专用预测头结果，并非新训练的独立标签探针。
`graph_jaccard`仅表示特征选择与先验父节点的一致性，不能直接证明因果性。
HSIC、IRM和跨站点验证同样不保证识别真实因果效应。

本次保留数值编码的特征表示、固定HSIC带宽，以及V5的双头训练设计。
`v11`保留原项目名称，指最后一层梯度方差的自定义惩罚，不是已核实的标准IRM版本。
`mask*(1-mask)`是极化惩罚，不是真正的Shannon熵。门控在预测热身阶段依旧参与学习，
与旧代码的实际行为相同，未实现注释所称“冻结0.5”。

## 检查

```bash
python -m unittest discover -s tests -v
python -m compileall -q project/src/elderly_health project/run.py
python -m pip install -r requirements-dev.txt
ruff check project/src/elderly_health project/run.py project/src/train project/src/loeo_validation.py project/src/preprocess_data tests
ruff format --check project/src/elderly_health project/run.py project/src/train project/src/loeo_validation.py project/src/preprocess_data tests
```

无PyTorch时，数据、配置和图解析测试可以运行，训练与梯度测试会明确跳过。
历史结果无需重写，修正后的新结果应独立编号；详见 `doc/代码整理说明.md`。


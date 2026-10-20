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

## 完整约束阶段的选模对照

JSON可设置 `selection_start_epoch`，使用日志中的一基轮次（第1轮为1）。
例如 `91` 表示前90轮仍正常训练、验证和调整学习率，但只从第91轮起
按验证AUC保存最佳权重，并从该窗口开始计算早停的未改善次数。
默认值 `null` 保留旧策略：解耦模型若进入IRM阶段，从第51轮起选模；
不足51轮的短训练及普通MLP从第1轮选模。

当前 `irm_start=50`、`ramp_epochs=40` 使用零基训练索引，故第91轮的
约束渐增系数才达到1。新配置不改变任何训练目标、约束时序或数据划分，
只改变选模窗口及其早停计数。早停计数改变可能延长总训练轮数。

```bash
python project/run.py train --config project/configs/v1_full_polar0.json --output outputs/v1-full-polar0-42
python project/run.py train --config project/configs/v1_full_polar3.json --output outputs/v1-full-polar3-42
python project/run.py train --config project/configs/v1_full_polar30.json --output outputs/v1-full-polar30-42
```

三组均为均值稀疏、无图、seed=42，只在极化权重0/3/30之间变化。
不要用 `--epochs 3` 覆盖这些配置：训练轮数小于显式选模起点会报错，
短训练应使用原有配置或把选模起点同步设小。`training.json` 新增实际
`selection_start_epoch`、每轮是否可选，以及稀疏/不变性渐增系数，方便核验。
91是根据训练时序确定的对照起点，不是根据测试集挑选的最佳超参数。

## 2026年10月：V1软门控与V2三地区环境实验

`project/configs/v1_relaxed.json` 已明确使用均值软稀疏（`sparsity=mean`）、
无图先验（`lambda_graph=0`）及零极化惩罚（`lambda_entropy=0`）。
其余双表示、V5预测头、IRM(v11)和HSIC的实现不变。
门控在训练中始终是连续分数，不因0.5阈值发生硬截断。
**请不要把此次配置改动后的新结果与原同名配置的历史结果混为同一条件。**

V2新增 `environment_granularity`，默认 `site`（12监测点），可选
`region`（东、中、西三个地区）。地区编码为研究方案的明确假设：
北京11、山东37 -> east；吉林22、江西36 -> central；广西45、宁夏64 -> west。
映射仅由站点码前两位生成，需与原调查方案/老师核对，尤其吉林所属的
研究区域分组；未知省份或非六位编码会报错。它不是一个预测特征，不覆盖
原始`environment_code`，原站点信息仍在预测CSV和NPZ中。
研究环境标签由原站点转为地区时，改变的是分组标签，不是样本或预测特征。

使用相同配置与seed做公平对照（从仓库根目录）：

```bash
python -m unittest discover -s tests -v
python -m compileall -q project/src/elderly_health project/run.py

# V1软门控：未完成时可单独训练（已存在输出目录不可复用）
python project/run.py train --config project/configs/v1_relaxed.json --output outputs/v1-relaxed-zero-polar-42

# V2: 仅改变环境粒度，其余超参数、训练/验证/测试划分保持一致
python project/run.py train --config project/configs/v2_site_v1.json --output outputs/v2-site-v1-42
python project/run.py train --config project/configs/v2_region_v1.json --output outputs/v2-region-v1-42

# 外部泛化测试：按原12站点留一；两套模型用相同的外部站点和输入行
python project/run.py loeo --held-out 110113 --holdout-level site --config project/configs/v2_site_v1.json --output outputs/v2-site-loeo-one-42
python project/run.py loeo --held-out 110113 --holdout-level site --config project/configs/v2_region_v1.json --output outputs/v2-region-loeo-one-42

# 上述单折运行通过后，再执行完整12站点留一对照
python project/run.py loeo --holdout-level site --config project/configs/v2_site_v1.json --output outputs/v2-site-loeo-all-42
python project/run.py loeo --holdout-level site --config project/configs/v2_region_v1.json --output outputs/v2-region-loeo-all-42

# 附加研究：真正留下一个完整地区（共3折），不是上述12站点对照的替代
python project/run.py loeo --holdout-level region --config project/configs/v2_region_v1.json --output outputs/v2-region-leave-region-out-42
```

`loeo`默认外部留出粒度依旧是`site`；`--holdout-level region`才把
一个地区的全部样本一起留出。地区留出下，环境探针在测试地区上
`n_known=0`时指标为null，不意味着`z_c`成功去除了环境信息。
`summary.csv`包含外部留出单位、环境训练粒度和分类指标；
地区配置的`metrics.json`还附加`per_site`，防止地区聚合掩盖站点差异。
样本的目标标签和划分逻辑不变；不能由这两组实验推断因果效应。

**注意：** 当前v11是自定义梯度方差惩罚，不是已证实的标准IRM。
若研究发现地区粒度改善了环境探针或站点泛化，仍必须验证不同随机种子、
以验证集选择超参、评估站点异质性，并确认认知障碍标签是否来自正确的
MMSE判定规则。测试集不得用于选模。Git LFS的.pt/.npz指针需拉取真实数据。

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

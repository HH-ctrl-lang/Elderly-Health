# V4 Causal Disentanglement + Graph-Guided Prior

> 历史V4实验说明。当前维护入口、行为修复及运行命令请参阅[仓库README](../README.md)
> 和[代码整理说明](../doc/代码整理说明.md)。下文旧命令及方法结论不代表整理后的实现。

## 项目结构

```
project/
├── data/
│   ├── data_mission2.csv              # 原始数据 (GBK编码)
│   ├── preprocessed/
│   │   ├── preprocessed_data.npz      # 预处理后数据 (train/val/test)
│   │   ├── preprocessed_data.pt       # torch格式
│   │   ├── scaler.pkl                 # StandardScaler
│   │   ├── feature_names.txt          # 19个特征名
│   │   ├── best_params.json           # 最优超参 (Bayes搜索后保存)
│   │   ├── best_model_v4.pth          # 训练好的模型权重
│   │   ├── roc_curve_v4.png           # ROC曲线
│   │   ├── loeo_results.png           # LOEO结果图
│   │   └── causal_graph.dot           # 因果图 (DOT格式)
│   └── 25年变量字典.xlsx              # 变量字典
├── src/
│   ├── preprocess_data/
│   │   └── preprocess_data.py         # 预处理脚本
│   ├── train/
│   │   ├── train_v4.py                # V4训练主脚本
│   │   ├── train_catal.py             # CATAL扩展 (实验性)
│   │   └── graph_prior.py             # 因果图解析 & L_graph损失
│   ├── causaldiscovery/
│   │   ├── code_pc.py                 # PC算法因果发现
│   │   ├── code_notears.py            # NOTEARS因果发现
│   │   ├── code_rfci.py               # RFCI因果发现
│   │   └── causalgrpah.dot            # 用户准备的因果图
│   └── loeo_validation.py             # Leave-One-Environment-Out验证
└── README.md
```

## 数据流

```
data_mission2.csv (23列, 4769样本, GBK编码)
    │
    ▼ preprocess_data.py
    │   · ID/environment_name丢弃
    │   · using_phone_year/hours_phone_per_day NaN→0
    │   · 其余NaN中位数填充
    │   · StandardScaler标准化
    │   · 70/15/15分层划分
    │
    ▼ preprocessed_data.npz (19特征, 12环境)
    │
    ├──→ train_v4.py (V4因果解耦训练)
    │    │  Phase 1 [0-20]:  纯预测 (mask冻结)
    │    │  Phase 2 [20-50]: +sparse+entropy+graph
    │    │  Phase 3 [50+]:   +IRM+env+cosine
    │    │  IRM入场时best_val_auc重置
    │    │  早停: warmup内不触发, IRM缓冲50epoch
    │    │
    │    ▼ best_model_v4.pth + roc_curve_v4.png
    │
    └──→ loeo_validation.py (LOEO外部验证)
         12轮, 每轮hold-out 1个环境
         对比 Causal V4 vs Baseline MLP
         输出 mask稳定性 + Jaccard一致性
```

## 19个特征

| # | 特征 | 类型 | 含义 |
|---|------|------|------|
| 1 | residence | 2类 | 城乡 (1城市/2乡村) |
| 2 | age | 2类 | 年龄分组 (1:65-80/2:80+) |
| 3 | gender | 2类 | 性别 (1男/2女) |
| 4 | education | 7类 | 文化程度 (1未上学~7本科+) |
| 5 | marital_status | 6类 | 婚姻状态 |
| 6 | living_alone | 2类 | 独居 (1是/2否) |
| 7 | wealth | 4类 | 财富状况 (1不好~4很好) |
| 8 | chronic_disease | 3类 | 慢病数量 (0无/1个/2+) |
| 9 | social_activity | 5类 | 社会活动频率 |
| 10 | IADL | 3类 | 工具性日常生活活动能力 |
| 11 | depression | 2类 | 抑郁 (GDS) |
| 12 | sleep_quality | 2类 | 睡眠质量 |
| 13 | sleep_duration | 3类 | 睡眠时长 |
| 14 | physical_activity | 2类 | 身体活动 |
| 15 | if_use_mobile_phone | 2类 | 是否使用手机 |
| 16 | using_phone_year | 连续 | 手机使用年数 (NaN=0) |
| 17 | hours_phone_per_day | 连续 | 每日手机时长 (NaN=0) |
| 18 | smoke | 4类 | 吸烟 |
| 19 | alcohol | 2类 | 饮酒 |

目标变量: `cognitive_impairment` (1=无, 2=有, 正例率~14%)

环境: `environment_code` (12个监测点, 每点381-403样本)

## 模型架构

```
x (19维)
  │
  ├── CausalGate: sigmoid(T × raw_mask) → mask ∈ [0,1]^19
  │     ├── x_c = x ⊙ mask       (因果特征)
  │     └── x_s = x ⊙ (1-mask)   (混杂特征)
  │
  ├── Encoder_c: Linear(19→64)→LayerNorm→ReLU→Dropout→Linear(64→32)→LayerNorm→ReLU
  │     └── z_c (32维因果表示)
  │           └── Predictor: Linear(32→32)→ReLU→Dropout→Linear(32→1) → ŷ
  │
  └── Encoder_s: 同结构
        └── z_s (32维混杂表示)
              └── EnvClassifier: Linear(32→32)→ReLU→Linear(32→12) → ê
```

## 损失函数 (7项)

| 损失 | 符号 | 权重 | 作用 |
|------|------|------|------|
| Prediction | L_pred | 1.0 | BCEWithLogitsLoss(pos_weight) |
| IRM | L_IRM | 0.01 | z_c预测能力跨环境梯度一致 |
| Environment | L_env | 1.0 | z_s必须编码环境信息 (交叉熵) |
| Sparsity | L_sparse | 可调 | mask均值接近target_sparsity |
| Cosine Independence | L_cosine | 0.02 | z_c与z_s列空间正交 |
| Entropy | L_entropy | 可调 | mask值推向0/1两极 |
| **Graph Prior** | **L_graph** | **0.1** | **因果图结构软约束** |

## 因果图先验机制

### 输入格式 (causal_graph.dot)

```dot
digraph {
  age -> cognitive_impairment;
  social_activity -> cognitive_impairment;
  depression -> cognitive_impairment;
  sleep_quality -> cognitive_impairment;
  using_phone_year -> cognitive_impairment;
  residence -> education;
  gender -> smoke;
  ...  (其他边)
}
```

### 解析逻辑

graph_prior.py 自动:
1. 解析DOT文件提取所有有向边
2. 识别直接指向target (cognitive_impairment) 的节点 → **parent** (鼓励mask→1)
3. 识别target指向的节点 → **child** (不加惩罚)
4. 其他节点 → **unrelated** (弱鼓励mask→0, weight=0.3)

### L_graph 计算

```
对parent特征:    loss += (mask - 1.0)² × 1.0     (强鼓励高mask)
对unrelated特征: loss += (mask - 0.0)² × 0.3     (弱鼓励低mask)
```

**关键设计**: parent权重(1.0) > unrelated权重(0.3)。图是软先验——数据信号可以override。

### 训练时机

```
Phase 1 (epoch 0-20):  无图先验 (纯预测warmup)
Phase 2 (epoch 20-50): 图先验ramp-up (40epoch线性增加到满权重)
Phase 3 (epoch 50+):   图先验满权重 + IRM + Env + Cosine
```

## 运行命令

### 1. 预处理
```bash
cd project/
python src/preprocess_data/preprocess_data.py
# 输出: data/preprocessed/preprocessed_data.npz
```

### 2. V4训练 (默认超参)
```bash
cd project/
python src/train/train_v4.py
# 输出: best_model_v4.pth, roc_curve_v4.png
```

### 3. V4训练 (Bayesian优化)
```bash
cd project/
python src/train/train_v4.py --bayes --trials 40 --timeout 7200
# 输出: best_params.json, best_model_v4.pth
```

### 4. LOEO外部验证
```bash
cd project/
python src/loeo_validation.py
# 输出: loeo_results.png, 12轮AUC对比
# 注意: LOEO自动读取best_params.json
```

### 5. 更新因果图后重新训练

编辑 `data/preprocessed/causal_graph.dot` → 直接运行步骤2

## 关键结果 (V4 + Graph Prior)

### 单次训练
- Test AUC: 0.7405
- 因果特征 (>0.5): 6/19
- Mask分化: 6个>0.7, 13个<0.2

### 因果特征排序 (本次)
| 排名 | 特征 | Mask | 图角色 | 说明 |
|------|------|------|--------|------|
| 1 | sleep_duration | 0.88 | unrelated | 图遗漏，数据强推 |
| 2 | living_alone | 0.88 | unrelated | 图遗漏 |
| 3 | social_activity | 0.86 | **parent** | 图✓ 数据✓ |
| 4 | sleep_quality | 0.84 | **parent** | 图✓ 数据✓ |
| 5 | if_use_mobile_phone | 0.82 | unrelated | 图遗漏 |
| 6 | IADL | 0.77 | unrelated | 图遗漏 (LOEO也确认) |

### 图与数据的矛盾
| 图说是parent但数据排除 | 图说是unrelated但数据选中 |
|------------------------|--------------------------|
| age (mask=0.19) | sleep_duration (0.88) |
| depression (0.15) | living_alone (0.88) |
| using_phone_year (0.17) | if_use_mobile_phone (0.82) |
| | IADL (0.77) |

### LOEO结果 (V4无图先验, 12轮)
- Causal V4: 0.715 ± 0.075
- Baseline MLP: 0.699 ± 0.090
- 最稳定特征: education (μ=0.83), IADL (0.75), sleep_duration (0.75)
- Top-8 Jaccard: 0.45

## 方法定位

本方法的核心贡献:

1. **端到端因果特征选择**: CausalGate嵌入式特征选择，联合优化预测+不变性+稀疏性
2. **环境解耦对抗训练**: 利用12个监测点作为自然实验，分离因果/混杂特征
3. **图结构引导的软约束**: 因果图提供结构骨架，CausalGate在数据信号和理论先验间平衡
4. **跨环境可迁移性**: LOEO验证选出的因果特征在unseen环境上的稳定性
5. **图-数据矛盾的发现**: CausalGate override图结构的能力揭示了因果发现可能遗漏的路径

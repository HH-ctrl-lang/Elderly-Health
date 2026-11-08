# 任务2 第2步：变量来源与编码追查

本步只处理阶段核对中的第一项补充：变量定义和来源。没有训练模型，不改原始文件或历史结果，也不直接纠正有疑问的编码。

## 已从代码确认的边界

- `project/legacy/preprocess_data.py`读取已存在的`data_mission2.csv`，不是从原始问卷生成该CSV的完整程序。
- `project/src/predata.py`与`predata_env.py`是另一条原始问卷预处理流程，其中存在活动、吸烟、手机字段，但不能因此认定它们就是当前19特征的完整来源。
- 当前审计表仍记录城乡69条差异、体力活动来源未确认、吸烟编码与字典冲突和手机文字编码疑问。本步核验这些问题，不预先判定解决。
- 原任务2第1步通过merge匹配，缺失编码可能互相连接；新脚本排除缺失编码。因此有效匹配数可能少于原来的4769，差异数量也必须按新输出解释。禁止按行号补配。

## 运行

在仓库根目录执行：

```bash
git pull origin feature/experiment2611-deep-diagnostics
python project/src/analysis/task2_step2_variable_provenance_followup.py
```

需要本地文件：

- `project/data/数据.xlsx`：原始工作簿，不要上传。
- `project/data/25年变量字典.xlsx`。
- `project/data/data_mission2.csv`：历史模型输入。

依赖pandas、numpy、openpyxl、matplotlib，不需要PyTorch。若文件位置不同，可用`--raw`、`--dictionary`、`--derived`指定。输出默认是`outputs/analysis_2611/task2_step2_variable_provenance_followup`。目录已存在时停止，避免覆盖；失败后重试应先将不完整输出移开，或用`--output`指定另一目录。

## 代码结构

主脚本分为四部分：

1. **读取与匹配**：根据非缺失个人编码一对一连接；编码仅保留在内存。重复编码会停止运行。
2. **数值核验**：统计城乡差异所在站点及编码变化；核对吸烟原始值和模型值；检查手机使用回答与后续问题缺失、填零的关系；对四个预先指定的活动字段进行交叉计数。
3. **证据摘录**：读取字典全部工作表中的相关定义，摘录限定预处理脚本的相关代码行；不执行旧脚本。
4. **保存**：只保存汇总表、两张图、输入文件哈希和匹配数量，不导出个人编码、逐人数据、预测或标签。

测试文件`tests/test_task2_variable_provenance_followup.py`用虚构数据检查：缺失编码不匹配、乱序仍正确匹配、重复编码停止、缺失与零区分、汇总计数正确、候选不会自动升级为已确认来源。

## 输出与如何判断

| 输出 | 用途 |
| --- | --- |
| `completion.json` | 匹配数量、未匹配数量和文件版本哈希；核对本次分母 |
| `field_comparison.csv` | 已知直接对应字段与手机缺失填零规则是否逐样本一致 |
| `residence_code_pairs_by_site.csv` | 城乡差异是哪些编码之间的变化、发生在哪些站点 |
| `residence_mismatch_by_site.csv`及同名图 | 差异数量和比例的站点分布 |
| `smoke_code_pairs_by_site.csv` | 吸烟1至4编码是否完整保留；不推断3、4的含义 |
| `phone_skip_pattern_counts.csv`及`phone_followup_missingness.png` | 手机回答与后续题目缺失、填零的对应关系 |
| `activity_candidate_comparison.csv`与`activity_candidate_code_pairs.csv` | 体力活动与四个候选字段的数值关系；不进行阈值搜索 |
| `dictionary_evidence.csv` | 字典定义的具体出处，包含工作表和行号 |
| `preprocessing_code_evidence.csv` | 当前限定源码中的相关行，作为追溯入口而非完整历史搜索 |

数值完全一致只能支持“可能直接对应”，不能证明实际来源。跳题模式可以提供旁证，不能代替原问卷定义。若仍找不到生成代码或问卷定义，最终应明确记为未确认，不猜测、不自动重编码、不更换特征重新训练。

运行后只需提交上述固定输出目录。等结果核对完成，再决定是否需要寻找原问卷/旧版生成脚本；此时不推进阶段报告或稳定性重复训练。

"""任务2，第1步：核对原始问卷、变量字典与当前模型输入的对应关系。

运行示例：
python project/analyze_region_data_audit_2611.py

本脚本只输出汇总统计和字段对应核验结果，不导出个人编码或逐人记录。
data_mission2.csv 仅作为“已有模型输入”的核验对象，不作为原始数据来源。
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = ROOT / "project/data/数据.xlsx"
DEFAULT_DICT = ROOT / "project/data/25年变量字典.xlsx"
DEFAULT_DERIVED = ROOT / "project/data/data_mission2.csv"
DEFAULT_OUTPUT = ROOT / "outputs/analysis_2611/task2_step1"

# 任务2，第1步：保留当前研究使用的地区划分，按监测点代码前两位映射。
PROVINCE_TO_REGION = {
    "11": "east",
    "37": "east",
    "22": "central",
    "36": "central",
    "45": "west",
    "64": "west",
}

# 任务2，第1步：记录已经能够从原始字段或问卷定义中确认的模型特征来源。
# 物理活动变量暂时不猜测来源，留待追踪原始构造规则。
FEATURE_RULES = [
    ("residence", "城乡", "直接使用", "1=城市；2=乡村", "已知"),
    ("age", "年龄分组", "直接使用分组值；不是实际年龄", "1=65至80岁；2=80岁及以上", "已知"),
    ("gender", "性别", "直接使用", "1=男；2=女", "已知"),
    ("education", "文化程度", "直接使用", "1至7级", "已知"),
    ("marital_status", "婚姻状态", "直接使用", "1至6类", "已知"),
    ("living_alone", "独居", "直接使用", "1=是；2=否", "已知"),
    ("wealth", "财富状况", "直接使用", "1至4级", "已知"),
    ("chronic_disease", "慢病数量分组", "直接使用分组值；不是慢病原始数量", "1=无；2=1个；3=2个及以上", "已知"),
    ("social_activity", "社会活动", "直接使用", "1=几乎每天；5=不参加", "已知"),
    ("IADL", "工具日常活动受损", "直接使用受损等级；不是IALD总分", "1=能力完好；2=轻度失能；3=中重度失能", "已知"),
    ("depression", "是否抑郁", "直接使用二分类结果；不是抑郁总分", "1=否；2=是", "已知"),
    ("sleep_quality", "PSQI", "PSQI大于5记为2，否则记为1；缺失保留", "1=PSQI不超过5；2=PSQI大于5", "需逐行核验"),
    ("sleep_duration", "夜间睡眠时长", "不超过4小时记为1；9小时及以上记为2；5至8小时记为3", "1=不超过4小时；2=9小时及以上；3=5至8小时", "需逐行核验"),
    ("physical_activity", "", "当前仓库未找到可复现的原始字段构造规则", "暂不解释1和2的含义", "未确认"),
    ("if_use_mobile_phone", "智能手机", "直接使用", "数据库编码1、2；字典文字说明存在笔误", "编码文字需核实"),
    ("using_phone_year", "G2", "未使用手机的缺失值填为0", "1至5级；0表示未使用手机", "需逐行核验"),
    ("hours_phone_per_day", "G4_1", "未使用手机的缺失值填为0；实际为时长等级", "1至5级；0表示未使用手机", "需逐行核验"),
    ("smoke", "吸烟", "直接使用原始值", "原始数据出现1至4；变量字典仅列1、2", "编码冲突待核实"),
    ("alcohol", "饮酒", "直接使用", "1=是；2=否", "已知"),
]

# 任务2，第1步：当前模型输入中除19个特征外的标签、环境和样本索引来源。
AUX_RULES = [
    ("cognitive_impairment", "1=无，2=有", "直接使用；1=无受损，2=有受损"),
    ("environment_code", "监测点代码", "直接使用六位监测点代码"),
    ("ID", "个人编码", "仅用于原始记录匹配，不是模型特征"),
]

def normalize_id(series):
    """任务2，第1步：统一个人编码格式，只在内存中用于匹配，不导出。"""
    value = series.astype("string").str.strip()
    value = value.str.replace(r"\.0$", "", regex=True)
    return value.mask(value.isin(["", "nan", "None", "<NA>"]))


def numeric(series):
    """任务2，第1步：把可转换的字段转成数值，保留无法转换的缺失。"""
    return pd.to_numeric(series, errors="coerce")


def same_values(expected, observed):
    """任务2，第1步：逐样本比较两个字段，允许双方同时缺失。"""
    left = numeric(expected).to_numpy(dtype=float)
    right = numeric(observed).to_numpy(dtype=float)
    both_missing = np.isnan(left) & np.isnan(right)
    both_present_equal = (
        np.isfinite(left) & np.isfinite(right) & np.isclose(left, right, atol=1e-8)
    )
    return both_missing | both_present_equal


def load_inputs(raw_path, dict_path, derived_path):
    if not raw_path.is_file():
        raise FileNotFoundError(
            f"找不到原始工作簿：{raw_path}\n"
            "请把本地原始数据.xlsx放到该路径；不要将它提交到Git。"
        )
    if not dict_path.is_file():
        raise FileNotFoundError(f"找不到变量字典：{dict_path}")
    if not derived_path.is_file():
        raise FileNotFoundError(
            f"找不到已有模型输入：{derived_path}\n"
            "请先从本地备份恢复data_mission2.csv。"
        )

    raw = pd.read_excel(raw_path, sheet_name=0)
    dictionary = pd.read_excel(dict_path, sheet_name=0)
    derived = pd.read_csv(derived_path, dtype={"ID": "string"})

    required_raw = {
        "个人编码", "监测点名称", "监测点代码", "1=无，2=有",
        *[rule[1] for rule in FEATURE_RULES if rule[1]],
        "PSQI", "夜间睡眠时长", "G2", "G4_1",
    }
    missing_raw = required_raw - set(raw.columns)
    if missing_raw:
        raise ValueError(f"原始工作簿缺少必要字段：{sorted(missing_raw)}")

    required_derived = {
        "ID", "environment_code", "cognitive_impairment",
        *[rule[0] for rule in FEATURE_RULES],
    }
    missing_derived = required_derived - set(derived.columns)
    if missing_derived:
        raise ValueError(f"已有模型输入缺少必要字段：{sorted(missing_derived)}")

    raw = raw.copy()
    derived = derived.copy()
    raw["__match_id"] = normalize_id(raw["个人编码"])
    derived["__match_id"] = normalize_id(derived["ID"])

    raw_ids = raw["__match_id"].dropna()
    derived_ids = derived["__match_id"].dropna()
    if raw_ids.duplicated().any() or derived_ids.duplicated().any():
        raise ValueError("个人编码存在重复，不能进行一对一来源核验。")

    matched = raw.merge(
        derived,
        on="__match_id",
        how="inner",
        suffixes=("_raw", "_model"),
        validate="one_to_one",
    )
    return raw, dictionary, derived, matched


def expected_model_values(feature, source, matched):
    """任务2，第1步：按已知规则生成应有的模型输入值。"""
    if feature in {
        "residence", "age", "gender", "education", "marital_status",
        "living_alone", "wealth", "chronic_disease", "social_activity",
        "IADL", "depression", "if_use_mobile_phone", "smoke", "alcohol",
    }:
        return matched[source]

    if feature == "sleep_quality":
        psqi = numeric(matched["PSQI"])
        return pd.Series(
            np.where(psqi.isna(), np.nan, np.where(psqi > 5, 2, 1)),
            index=matched.index,
        )

    if feature == "sleep_duration":
        hours = numeric(matched["夜间睡眠时长"])
        result = pd.Series(np.nan, index=matched.index, dtype=float)
        result.loc[hours <= 4] = 1
        result.loc[hours >= 9] = 2
        result.loc[hours.between(5, 8)] = 3
        return result

    if feature == "using_phone_year":
        return numeric(matched["G2"]).fillna(0)

    if feature == "hours_phone_per_day":
        return numeric(matched["G4_1"]).fillna(0)

    return None


def build_mapping_audit(raw, dictionary, derived, matched):
    """任务2，第1步：逐样本核验19个特征的原始来源和加工规则。"""
    rows = []
    for feature, source, transform, coding, status in FEATURE_RULES:
        model_values = numeric(matched[feature]) if len(matched) else pd.Series(dtype=float)
        if not source:
            rows.append({
                "model_feature": feature,
                "raw_source": "未确认",
                "transformation": transform,
                "coding_note": coding,
                "status": status,
                "matched_rows": len(matched),
                "exact_match_rows": np.nan,
                "mismatch_rows": np.nan,
                "raw_missing_rows": np.nan,
                "model_missing_rows": int(numeric(derived[feature]).isna().sum()),
            })
            continue

        expected = expected_model_values(feature, source, matched)
        observed = matched[feature]
        equal = same_values(expected, observed)
        mismatch = int((~equal).sum())

        final_status = status
        if feature == "smoke":
            final_status = "字段对应可核验，但字典编码与原始取值冲突"
        elif mismatch == 0 and status == "需逐行核验":
            final_status = "逐样本核验一致"
        elif mismatch > 0:
            final_status = "存在不一致，需追溯构造规则"

        rows.append({
            "model_feature": feature,
            "raw_source": source,
            "transformation": transform,
            "coding_note": coding,
            "status": final_status,
            "matched_rows": len(matched),
            "exact_match_rows": int(equal.sum()),
            "mismatch_rows": mismatch,
            "raw_missing_rows": int(raw[source].isna().sum()) if source in raw else np.nan,
            "model_missing_rows": int(numeric(derived[feature]).isna().sum()),
        })

    # 任务2，第1步：单独核验标签、环境代码与样本索引来源。
    for model_name, raw_name, note in AUX_RULES:
        expected = matched[raw_name]
        observed = matched[model_name]
        equal = same_values(expected, observed)
        rows.append({
            "model_feature": model_name,
            "raw_source": raw_name,
            "transformation": note,
            "coding_note": note,
            "status": "逐样本核验一致" if equal.all() else "存在不一致，需检查",
            "matched_rows": len(matched),
            "exact_match_rows": int(equal.sum()),
            "mismatch_rows": int((~equal).sum()),
            "raw_missing_rows": int(raw[raw_name].isna().sum()),
            "model_missing_rows": int(derived[model_name].isna().sum()),
        })
    return pd.DataFrame(rows)


def build_codebook_audit(raw, dictionary):
    """任务2，第1步：对照变量字典的编码说明与原始数据实际取值。"""
    rows = []
    names = ["城乡", "年龄分组", "性别", "文化程度", "婚姻状态", "独居",
             "财富状况", "慢病数量分组", "社会活动", "工具日常活动受损",
             "是否抑郁", "智能手机", "G2", "G4_1", "吸烟", "饮酒",
             "锻炼", "高强度活动", "中等强度体力活动", "户外活动"]
    dict_copy = dictionary.copy()
    dict_copy["变量_标准名"] = dict_copy["变量"].astype("string").str.strip()

    for name in names:
        source = raw[name] if name in raw else None
        found = dict_copy[dict_copy["变量_标准名"] == name]
        if source is None:
            observed = "原始工作簿无此字段"
            missing_n = np.nan
        else:
            codes = sorted(numeric(source).dropna().unique().tolist())
            observed = ";".join(str(int(v)) if float(v).is_integer() else str(v) for v in codes)
            missing_n = int(source.isna().sum())

        if found.empty:
            description, db_codes = "", ""
        else:
            row = found.iloc[0]
            description = "" if pd.isna(row.get("值描述")) else str(row.get("值描述"))
            db_codes = "" if pd.isna(row.get("对应数据库的值")) else str(row.get("对应数据库的值"))

        issue = ""
        if name == "吸烟" and observed == "1;2;3;4":
            issue = "字典仅列1、2，但原始数据出现1、2、3、4；需确认3、4含义"
        elif name == "智能手机":
            issue = "字典文字将否误写为1；数据库编码和原始数据均为1、2"
        rows.append({
            "raw_field": name,
            "dictionary_description": description,
            "dictionary_database_codes": db_codes,
            "observed_raw_codes": observed,
            "missing_rows": missing_n,
            "issue": issue,
        })
    return pd.DataFrame(rows)


def build_site_table(raw):
    """任务2，第1步：检查12个监测点及地区映射，不进行显著性检验。"""
    frame = raw.copy()
    frame["site_code"] = frame["监测点代码"].astype("string").str.replace(r"\.0$", "", regex=True)
    frame["province_prefix"] = frame["site_code"].str[:2]
    frame["region"] = frame["province_prefix"].map(PROVINCE_TO_REGION)
    if frame["region"].isna().any():
        bad = sorted(frame.loc[frame["region"].isna(), "site_code"].dropna().unique())
        raise ValueError(f"以下监测点代码没有地区映射：{bad}")
    frame["label"] = numeric(frame["1=无，2=有"])
    table = (
        frame.groupby(["region", "site_code", "监测点名称"], dropna=False)
        .agg(
            sample_count=("label", "size"),
            no_impairment_count=("label", lambda x: int((x == 1).sum())),
            impairment_count=("label", lambda x: int((x == 2).sum())),
        )
        .reset_index()
    )
    table["positive_rate"] = table["impairment_count"] / table["sample_count"]
    return table.sort_values(["region", "site_code"])


def build_overview(raw, derived, matched):
    """任务2，第1步：输出不含个人身份信息的原始数据概况。"""
    target = numeric(raw["1=无，2=有"])
    return pd.DataFrame([
        {"item": "原始样本数", "value": len(raw)},
        {"item": "原始字段数", "value": len(raw.columns) - 1},
        {"item": "原始个人编码缺失数", "value": int(raw["个人编码"].isna().sum())},
        {"item": "原始个人编码唯一数", "value": int(raw["个人编码"].nunique(dropna=True))},
        {"item": "监测点数量", "value": int(raw["监测点代码"].nunique(dropna=True))},
        {"item": "目标标签缺失数", "value": int(target.isna().sum())},
        {"item": "无认知功能受损人数", "value": int((target == 1).sum())},
        {"item": "有认知功能受损人数", "value": int((target == 2).sum())},
        {"item": "现有模型输入样本数", "value": len(derived)},
        {"item": "可通过个人编码匹配的样本数", "value": len(matched)},
        {"item": "无法通过个人编码匹配的原始记录数", "value": len(raw) - len(matched)},
    ])


def run(raw_path, dict_path, derived_path, output):
    if output.exists():
        raise FileExistsError(f"输出目录已存在，为避免覆盖而停止：{output}")

    raw, dictionary, derived, matched = load_inputs(raw_path, dict_path, derived_path)
    output.mkdir(parents=True)

    mapping = build_mapping_audit(raw, dictionary, derived, matched)
    codebook = build_codebook_audit(raw, dictionary)
    sites = build_site_table(raw)
    overview = build_overview(raw, derived, matched)

    mapping.to_csv(output / "feature_mapping_audit.csv", index=False, encoding="utf-8-sig")
    codebook.to_csv(output / "codebook_consistency.csv", index=False, encoding="utf-8-sig")
    sites.to_csv(output / "site_distribution.csv", index=False, encoding="utf-8-sig")
    overview.to_csv(output / "raw_data_overview.csv", index=False, encoding="utf-8-sig")

    print("任务2，第1步：原始数据与模型特征来源核对完成")
    print(f"原始数据：{len(raw)}行，{len(raw.columns)-1}个字段")
    print(f"匹配样本：{len(matched)}行；未匹配记录仅统计数量，不导出个人编码")
    print(f"监测点：{raw['监测点代码'].nunique()}个")
    print("注意：physical_activity来源尚未确认；吸烟编码与变量字典不一致。")
    print("输出目录：", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="本地原始数据.xlsx路径")
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICT, help="25年变量字典路径")
    parser.add_argument("--derived", type=Path, default=DEFAULT_DERIVED, help="当前模型输入CSV，仅用于来源核验")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(
        args.raw.resolve(),
        args.dictionary.resolve(),
        args.derived.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()

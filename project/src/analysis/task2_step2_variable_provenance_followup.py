"""任务2 第2步：追查变量来源与编码，只导出汇总，不训练或修改数据。

在仓库根目录运行：python project/src/analysis/task2_step2_variable_provenance_followup.py
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
FIELDS = {
    "residence": "城乡", "smoke": "吸烟",
    "if_use_mobile_phone": "智能手机", "using_phone_year": "G2",
    "hours_phone_per_day": "G4_1",
}
ACTIVITY_FIELDS = ["锻炼", "高强度活动", "中等强度体力活动", "户外活动"]
KEYWORDS = ["physical_activity", "residence", "smoke", "if_use_mobile_phone",
            "data_mission2", "城乡", "体力活动", "锻炼", "吸烟", "智能手机"]


def normalize_id(values):
    """任务2 第2步：编码只在内存用于匹配，缺失编码不参与连接。"""
    result = values.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return result.mask(result.isin(["", "nan", "None", "<NA>"]))


def match_records(raw, model):
    """任务2 第2步：只允许非缺失编码一对一匹配，不按行号猜测。"""
    left, right = raw.copy(), model.copy()
    left["__key"] = normalize_id(left["个人编码"])
    right["__key"] = normalize_id(right["ID"])
    for frame in (left, right):
        if frame.loc[frame["__key"].notna(), "__key"].duplicated().any():
            raise ValueError("非缺失个人编码重复，停止核验；请在本地排查，勿上传逐人记录。")
    # 任务2 第2步：保留命名空间，避免原始列和模型列同名而混淆。
    left = left.loc[left["__key"].notna()].set_index("__key").add_prefix("raw::")
    right = right.loc[right["__key"].notna()].set_index("__key").add_prefix("model::")
    joined = left.join(right, how="inner", validate="one_to_one")
    overview = {
        "raw_rows": len(raw), "model_rows": len(model), "valid_id_matches": len(joined),
        "raw_missing_id": int(normalize_id(raw["个人编码"]).isna().sum()),
        "model_missing_id": int(normalize_id(model["ID"]).isna().sum()),
        "raw_unmatched_rows_including_missing_id": len(raw) - len(joined),
        "model_unmatched_rows_including_missing_id": len(model) - len(joined),
    }
    if joined.empty:
        raise ValueError("没有有效编码匹配，停止核验。")
    return joined, overview


def numeric(values):
    """任务2 第2步：不能把非数值文本悄悄当作缺失。"""
    result = pd.to_numeric(values, errors="coerce")
    if (values.notna() & result.isna()).any() or np.isinf(result.dropna()).any():
        raise ValueError(f"核验字段包含无法解析的数值：{values.name}；未导出原值。")
    return result.astype(float)


def equality(left, right):
    a, b = numeric(left), numeric(right)
    return (a.isna() & b.isna()) | (a.notna() & b.notna() & a.eq(b))


def codes(values):
    """任务2 第2步：汇总编码显示为字符串，使缺失值也计入统计。"""
    return numeric(values).map(lambda x: "missing" if pd.isna(x) else f"{x:g}")


def grouped_counts(frame, columns):
    return frame.groupby(columns, dropna=False).size().rename("count").reset_index()


def build_tables(matched):
    """任务2 第2步：构建城乡、吸烟、手机及体力活动候选的汇总核验表。"""
    m = matched
    site = codes(m["model::environment_code"])
    rows = []
    for feature, source in FIELDS.items():
        a, b = m[f"raw::{source}"], m[f"model::{feature}"]
        expected = numeric(a).fillna(0) if source in ["G2", "G4_1"] else numeric(a)
        eq = equality(expected, b)
        rows.append({"model_feature": feature, "raw_field": source,
                     "comparison_rule": "missing_to_zero" if source in ["G2", "G4_1"] else "direct",
                     "matched_rows": len(m), "equal_rows": int(eq.sum()),
                     "mismatch_rows": int((~eq).sum()),
                     "raw_missing_rows": int(a.isna().sum()),
                     "model_missing_rows": int(b.isna().sum())})
    tables = {"field_comparison": pd.DataFrame(rows)}
    residence = pd.DataFrame({"site": site, "raw_code": codes(m["raw::城乡"]),
                              "model_code": codes(m["model::residence"])})
    tables["residence_code_pairs_by_site"] = grouped_counts(residence, ["site", "raw_code", "model_code"])
    residence["mismatch"] = ~equality(m["raw::城乡"], m["model::residence"])
    tables["residence_mismatch_by_site"] = residence.groupby("site").agg(
        matched_rows=("mismatch", "size"), mismatch_rows=("mismatch", "sum")).reset_index()
    tables["residence_mismatch_by_site"]["mismatch_fraction"] = (
        tables["residence_mismatch_by_site"]["mismatch_rows"] /
        tables["residence_mismatch_by_site"]["matched_rows"])
    smoke = pd.DataFrame({"site": site, "raw_code": codes(m["raw::吸烟"]),
                          "model_code": codes(m["model::smoke"])})
    tables["smoke_code_pairs_by_site"] = grouped_counts(smoke, ["site", "raw_code", "model_code"])
    phone = pd.DataFrame({"raw_phone_code": codes(m["raw::智能手机"]),
                          "model_phone_code": codes(m["model::if_use_mobile_phone"]),
                          "g2_missing": m["raw::G2"].isna(),
                          "g4_missing": m["raw::G4_1"].isna(),
                          "model_year_zero": numeric(m["model::using_phone_year"]).eq(0),
                          "model_hours_zero": numeric(m["model::hours_phone_per_day"]).eq(0)})
    tables["phone_skip_pattern_counts"] = grouped_counts(phone, list(phone.columns))
    # 任务2 第2步：明确限定候选，不搜索最优阈值，不据一致率认定来源或因果性。
    candidates, cross = [], []
    for field in ACTIVITY_FIELDS:
        key = f"raw::{field}"
        if key not in m:
            candidates.append({"raw_field": field, "available": False,
                               "provenance_confirmed": False})
            continue
        a, b = numeric(m[key]), numeric(m["model::physical_activity"])
        valid = a.notna() & b.notna()
        candidates.append({"raw_field": field, "available": True, "matched_rows": len(m),
                           "both_present_rows": int(valid.sum()),
                           "direct_equal_on_present": int(a[valid].eq(b[valid]).sum()),
                           "raw_missing_rows": int(a.isna().sum()),
                           "provenance_confirmed": False})
        frame = pd.DataFrame({"raw_field": field, "raw_code": codes(a), "model_code": codes(b)})
        cross.append(grouped_counts(frame, ["raw_field", "raw_code", "model_code"]))
    tables["activity_candidate_comparison"] = pd.DataFrame(candidates)
    tables["activity_candidate_code_pairs"] = pd.concat(cross, ignore_index=True) if cross else pd.DataFrame(
        columns=["raw_field", "raw_code", "model_code", "count"])
    return tables


def dictionary_evidence(path):
    """任务2 第2步：只摘录相关变量的字典定义，读取全部工作表。"""
    targets = set(FIELDS.values()) | set(ACTIVITY_FIELDS)
    rows = []
    for sheet, frame in pd.read_excel(path, sheet_name=None).items():
        if "变量" not in frame:
            continue
        names = frame["变量"].ffill().astype("string").str.strip()
        for i in frame.index[names.isin(targets)]:
            rows.append({"sheet": sheet, "excel_row": int(i) + 2, "raw_field": names.loc[i],
                         "description": frame.loc[i].get("值描述", ""),
                         "database_codes": frame.loc[i].get("对应数据库的值", "")})
    return pd.DataFrame(rows, columns=["sheet", "excel_row", "raw_field", "description", "database_codes"])


def code_evidence():
    """任务2 第2步：读取预处理源码，不导入或执行旧脚本。"""
    paths = [ROOT / "project/legacy/preprocess_data.py", ROOT / "project/src/predata.py",
             ROOT / "project/src/predata_env.py", ROOT / "project/src/preprocess_data/preprocess_data.py"]
    rows = []
    for path in paths:
        if not path.is_file():
            rows.append({"path": str(path.relative_to(ROOT)), "line": 0, "code": "FILE_NOT_FOUND"})
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if any(word in line for word in KEYWORDS):
                rows.append({"path": str(path.relative_to(ROOT)), "line": i, "code": line.strip()})
    return pd.DataFrame(rows)


def save_figures(tables, output):
    """任务2 第2步：图中文字使用英文，图只展示汇总计数。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir()
    frame = tables["residence_mismatch_by_site"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(frame["site"], frame["mismatch_rows"], color="#4477aa")
    ax.set(title="Residence disagreements in valid-ID matched records", ylabel="Record count", xlabel="Site")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output / "residence_mismatch_by_site.png", dpi=160)
    plt.close(fig)
    frame = tables["phone_skip_pattern_counts"].groupby("raw_phone_code").agg(
        total=("count", "sum"))
    skip = tables["phone_skip_pattern_counts"]
    both = skip[skip["g2_missing"] & skip["g4_missing"]].groupby("raw_phone_code")["count"].sum()
    frame["both_followups_missing"] = both.reindex(frame.index, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    frame.plot.bar(ax=ax, color=["#4477aa", "#eeaa33"])
    ax.set(title="Phone follow-up missingness by raw response code", xlabel="Raw phone code", ylabel="Record count")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(output / "phone_followup_missingness.png", dpi=160)
    plt.close(fig)


def read_model(path):
    for encoding in ["utf-8-sig", "gbk"]:
        try:
            return pd.read_csv(path, encoding=encoding, dtype={"ID": "string"})
        except UnicodeDecodeError:
            continue
    raise ValueError("模型CSV无法按UTF-8或GBK读取。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "project/data/数据.xlsx")
    parser.add_argument("--dictionary", type=Path, default=ROOT / "project/data/25年变量字典.xlsx")
    parser.add_argument("--derived", type=Path, default=ROOT / "project/data/data_mission2.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/analysis_2611/task2_step2_variable_provenance_followup")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("输出目录已存在，请保留旧结果并使用--output指定新目录。")
    for path in [args.raw, args.dictionary, args.derived]:
        if not path.is_file():
            raise FileNotFoundError(f"找不到本地文件：{path}；请恢复本地备份，不要上传原始问卷。")
    raw = pd.read_excel(args.raw, dtype={"个人编码": "string"})
    model = read_model(args.derived)
    for frame, required in [(raw, {"个人编码", *FIELDS.values()}),
                            (model, {"ID", "environment_code", "physical_activity", *FIELDS})]:
        if not required.issubset(frame.columns):
            raise ValueError(f"缺少核验字段：{sorted(required - set(frame.columns))}")
    matched, overview = match_records(raw, model)
    tables = build_tables(matched)
    tables["dictionary_evidence"] = dictionary_evidence(args.dictionary)
    tables["preprocessing_code_evidence"] = code_evidence()
    args.output.mkdir(parents=True)
    for name, table in tables.items():
        table.to_csv(args.output / f"{name}.csv", index=False, encoding="utf-8-sig")
    save_figures(tables, args.output / "figures")
    metadata = {"task": "任务2 第2步", "matching": overview,
                "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for name, path in [("raw", args.raw), ("dictionary", args.dictionary), ("derived", args.derived)]},
                "model_trained": False, "input_modified": False, "individual_records_exported": False,
                "activity_candidate_match_is_not_provenance_proof": True,
                "phone_skip_patterns_are_not_questionnaire_definition_proof": True}
    (args.output / "completion.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("任务2 第2步核验完成：仅输出汇总，不修改原始数据，不训练模型。")
    print(f"有效编码匹配：{len(matched)}；原始未匹配：{overview['raw_unmatched_rows_including_missing_id']}")
    print(f"输出目录：{args.output}")


if __name__ == "__main__":
    main()

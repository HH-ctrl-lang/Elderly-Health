"""任务2 第2步：用虚构记录验证匹配和计数，不读取真实研究数据。"""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

PATH = Path(__file__).resolve().parents[1] / "project/src/analysis/task2_step2_variable_provenance_followup.py"
SPEC = importlib.util.spec_from_file_location("provenance_followup", PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class ProvenanceTests(unittest.TestCase):
    def test_missing_ids_never_match_and_order_is_not_used(self):
        raw = pd.DataFrame({"个人编码": ["a", None, "b"], "城乡": [1, 1, 2]})
        model = pd.DataFrame({"ID": ["b", None, "a"], "residence": [2, 2, 1]})
        matched, info = audit.match_records(raw, model)
        self.assertEqual(info["valid_id_matches"], 2)
        self.assertEqual(info["raw_missing_id"], 1)
        self.assertTrue(audit.equality(matched["raw::城乡"], matched["model::residence"]).all())

    def test_duplicates_after_normalization_stop(self):
        with self.assertRaises(ValueError):
            audit.match_records(pd.DataFrame({"个人编码": [" 12", "12.0"]}),
                                pd.DataFrame({"ID": ["12"]}))

    def test_missing_missing_equal_but_missing_zero_different(self):
        result = audit.equality(pd.Series([np.nan, np.nan, 1]), pd.Series([np.nan, 0, 1]))
        self.assertEqual(result.tolist(), [True, False, True])

    def test_text_is_not_silently_missing(self):
        with self.assertRaises(ValueError):
            audit.numeric(pd.Series(["unknown", "1"]))

    def test_aggregate_counts_and_candidate_not_confirmed(self):
        raw = pd.DataFrame({"个人编码": ["a", "b", "c"], "城乡": [1, 1, 2],
                            "吸烟": [1, 3, 4], "智能手机": [1, 2, 2],
                            "G2": [2, np.nan, np.nan], "G4_1": [3, np.nan, np.nan],
                            "锻炼": [1, 2, 1]})
        model = pd.DataFrame({"ID": ["a", "b", "c"], "residence": [1, 2, 2],
                              "smoke": [1, 3, 4], "if_use_mobile_phone": [1, 2, 2],
                              "using_phone_year": [2, 0, 0], "hours_phone_per_day": [3, 0, 0],
                              "physical_activity": [1, 2, 1], "environment_code": [110113]*3})
        matched, _ = audit.match_records(raw, model)
        tables = audit.build_tables(matched)
        self.assertEqual(tables["residence_mismatch_by_site"]["mismatch_rows"].sum(), 1)
        comp = tables["field_comparison"].set_index("model_feature")
        self.assertEqual(comp.loc["using_phone_year", "mismatch_rows"], 0)
        self.assertEqual(tables["phone_skip_pattern_counts"]["count"].sum(), 3)
        self.assertFalse(tables["activity_candidate_comparison"]["provenance_confirmed"].any())
        for frame in tables.values():
            self.assertNotIn("ID", frame.columns)
            self.assertNotIn("__key", frame.columns)
            self.assertNotIn("个人编码", frame.columns)


if __name__ == "__main__":
    unittest.main()

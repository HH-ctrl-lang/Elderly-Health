"""Read raw data, split first, and fit preprocessing on the training rows only."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET = "cognitive_impairment"
ENVIRONMENT = "environment_code"
REGION = "environment_region"
# Three-region study assumption: Beijing/Shandong east, Jilin/Jiangxi central,
# Guangxi/Ningxia west. This is deliberately an explicit, auditable mapping.
PROVINCE_TO_REGION = {
    "11": "east",
    "37": "east",
    "22": "central",
    "36": "central",
    "45": "west",
    "64": "west",
}

FEATURES = (
    "residence",
    "age",
    "gender",
    "education",
    "marital_status",
    "living_alone",
    "wealth",
    "chronic_disease",
    "social_activity",
    "IADL",
    "depression",
    "sleep_quality",
    "sleep_duration",
    "physical_activity",
    "if_use_mobile_phone",
    "using_phone_year",
    "hours_phone_per_day",
    "smoke",
    "alcohol",
)


def read_raw(path, encoding="gbk"):
    """Retain source-row identity. Do not silently convert invalid labels to negatives."""
    frame = pd.read_csv(path, encoding=encoding)
    required = set(FEATURES) | {TARGET, ENVIRONMENT}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = frame.loc[frame[TARGET].notna()].copy()
    if not frame[TARGET].isin([1, 2]).all():
        raise ValueError("Target must use 1=no impairment, 2=impairment")
    if frame[ENVIRONMENT].isna().any():
        raise ValueError("Environment codes cannot be missing")
    frame[ENVIRONMENT] = frame[ENVIRONMENT].astype(str).str.replace(r"\.0$", "", regex=True)
    for column in FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if np.isinf(frame[list(FEATURES)].to_numpy(dtype=float)).any():
        raise ValueError("Features cannot contain infinity")
    # Original CSV row number is stable even when the source ID is missing or duplicated.
    frame["sample_id"] = frame.index.to_numpy(dtype=np.int64)
    frame["label"] = (frame[TARGET] == 2).astype(np.float32)
    if frame.empty or frame["label"].nunique() != 2:
        raise ValueError("Both outcome classes are required")
    return frame


def attach_regions(frame):
    """Add research-defined regions without replacing original six-digit site codes."""
    codes = frame[ENVIRONMENT].astype(str)
    provinces = codes.str[:2]
    invalid = ~codes.str.fullmatch(r"[0-9]{6}") | ~provinces.isin(PROVINCE_TO_REGION)
    if invalid.any():
        bad_codes = sorted(codes[invalid].unique().tolist())
        raise ValueError(f"Region mapping is undefined for site codes: {bad_codes}")
    result = frame.copy()
    result[REGION] = provinces.map(PROVINCE_TO_REGION)
    return result


def random_split(frame, seed=42):
    train_val, test = train_test_split(
        frame,
        test_size=0.15,
        stratify=frame["label"],
        random_state=seed,
    )
    train, val = train_test_split(
        train_val,
        test_size=0.15 / 0.85,
        stratify=train_val["label"],
        random_state=seed,
    )
    return train, val, test


def held_out_split(frame, held_out, seed=42, column=ENVIRONMENT):
    test = frame.loc[frame[column].astype(str) == str(held_out)]
    source = frame.loc[frame[column].astype(str) != str(held_out)]
    if test.empty or source.empty:
        raise ValueError(f"Invalid held-out environment: {held_out}")
    train, val = train_test_split(
        source,
        test_size=0.15,
        stratify=source["label"],
        random_state=seed,
    )
    return train, val, test


class FeaturePreprocessor:
    """Preserve the existing numeric coding; learn imputation and scaling from source rows.

    Phone non-use and unknown phone duration are not conflated: zeros must already be
    coded by the raw-data preparation rules. Missing values use training medians.
    """

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, frame):
        values = frame[list(FEATURES)]
        empty = values.columns[values.isna().all()].tolist()
        if empty:
            raise ValueError(f"Entirely missing training features: {empty}")
        self.scaler.fit(self.imputer.fit_transform(values))
        self.fitted = True
        return self

    def transform(self, frame):
        if not self.fitted:
            raise RuntimeError("Fit preprocessing on training rows first")
        return self.scaler.transform(self.imputer.transform(frame[list(FEATURES)])).astype(
            np.float32
        )


@dataclass
class SplitData:
    x: np.ndarray
    y: np.ndarray
    env: np.ndarray
    sample_ids: np.ndarray
    site_env: np.ndarray | None = None


def prepare_splits(frames, environment_column=ENVIRONMENT):
    preprocessor = FeaturePreprocessor().fit(frames[0])
    splits = [
        SplitData(
            preprocessor.transform(frame),
            frame["label"].to_numpy(dtype=np.float32),
            frame[environment_column].to_numpy(dtype=str),
            frame["sample_id"].to_numpy(),
            frame[ENVIRONMENT].to_numpy(dtype=str),
        )
        for frame in frames
    ]
    return splits, preprocessor


def save_splits(path, splits):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"feature_names": np.asarray(FEATURES)}
    for name, split in zip(("train", "val", "test"), splits):
        if split.site_env is not None:
            arrays[f"site_env_{name}"] = split.site_env
        arrays.update(
            {
                f"X_{name}": split.x,
                f"y_{name}": split.y,
                f"env_{name}": split.env,
                f"sample_id_{name}": split.sample_ids,
            }
        )
    # New files explicitly use 0/1 outcomes; do not pass them to archived 1/2 loaders.
    arrays["label_encoding"] = np.asarray("0=no impairment;1=impairment")
    np.savez_compressed(path, **arrays)

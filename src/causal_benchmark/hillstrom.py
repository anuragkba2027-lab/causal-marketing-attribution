"""Loading, validating, and preprocessing the Hillstrom randomized email test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

HILLSTROM_SOURCE_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
HILLSTROM_SHA256 = "0E5893329D8B93CEFECC571777672028290AB69865718020C78C7284F291AECE"

CONTROL_ARM = "No E-Mail"
TREATMENT_ARM = "Mens E-Mail"
EXCLUDED_ARM = "Womens E-Mail"
PRIMARY_OUTCOME = "visit"

NUMERIC_FEATURES = ["recency", "history", "mens", "womens", "newbie"]
CATEGORICAL_FEATURES = ["zip_code", "channel"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
SOURCE_COLUMNS = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
    "segment",
    "visit",
    "conversion",
    "spend",
]


def load_hillstrom_raw(path: str | Path = "data/raw/hillstrom_email_rct.csv") -> pd.DataFrame:
    """Load the unchanged source CSV and enforce its documented schema."""

    raw_path = Path(path)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Missing {raw_path}. Run scripts/download_hillstrom.py from the repository root."
        )
    frame = pd.read_csv(raw_path)
    if frame.columns.tolist() != SOURCE_COLUMNS:
        raise ValueError(
            f"Unexpected Hillstrom schema. Expected {SOURCE_COLUMNS}, got {frame.columns.tolist()}"
        )
    return frame


def validate_hillstrom_raw(frame: pd.DataFrame) -> dict[str, object]:
    """Return high-signal data-quality evidence and fail on causal blockers."""

    expected_arms = {CONTROL_ARM, TREATMENT_ARM, EXCLUDED_ARM}
    actual_arms = set(frame["segment"].dropna().unique())
    blockers: list[str] = []
    if len(frame) != 64_000:
        blockers.append(f"Expected 64,000 rows; found {len(frame):,}")
    if frame.isna().any().any():
        blockers.append("Required source columns contain missing values")
    if actual_arms != expected_arms:
        blockers.append(f"Unexpected treatment arms: {sorted(actual_arms)}")
    for column in ["mens", "womens", "newbie", "visit", "conversion"]:
        if not set(frame[column].unique()).issubset({0, 1}):
            blockers.append(f"{column} is not binary")
    if not frame["recency"].between(1, 12).all():
        blockers.append("recency falls outside the documented 1-12 month range")
    if (frame[["history", "spend"]] < 0).any().any():
        blockers.append("history or spend contains negative values")
    if blockers:
        raise ValueError("; ".join(blockers))

    arm_counts = frame["segment"].value_counts().to_dict()
    return {
        "rows": len(frame),
        "columns": frame.shape[1],
        "missing_cells": int(frame.isna().sum().sum()),
        "exact_repeated_rows": int(frame.duplicated().sum()),
        "treatment_arm_counts": arm_counts,
        "visit_rate": float(frame["visit"].mean()),
        "conversion_rate": float(frame["conversion"].mean()),
        "mean_spend": float(frame["spend"].mean()),
    }


def prepare_hillstrom_contrast(
    frame: pd.DataFrame,
    *,
    treatment_arm: str = TREATMENT_ARM,
    control_arm: str = CONTROL_ARM,
    random_state: int = 42,
    test_size: float = 0.30,
) -> pd.DataFrame:
    """Create the pre-specified binary RCT contrast and shared holdout split.

    The Women's email arm is excluded rather than collapsed into treatment because
    it is a substantively different intervention.
    """

    validate_hillstrom_raw(frame)
    contrast = frame.loc[frame["segment"].isin([control_arm, treatment_arm])].copy()
    contrast.insert(0, "customer_id", contrast.index.to_numpy(dtype=int) + 1)
    contrast["treatment"] = (contrast["segment"] == treatment_arm).astype(int)

    train_ids, test_ids = train_test_split(
        contrast.index,
        test_size=test_size,
        random_state=random_state,
        stratify=contrast["treatment"],
    )
    contrast["split"] = "train"
    contrast.loc[test_ids, "split"] = "test"
    assert len(train_ids) + len(test_ids) == len(contrast)
    return contrast.reset_index(drop=True)


def load_or_create_hillstrom_contrast(
    raw_path: str | Path = "data/raw/hillstrom_email_rct.csv",
    processed_path: str | Path = "data/processed/hillstrom_mens_email_vs_control.csv",
    random_state: int = 42,
) -> pd.DataFrame:
    """Load a processed contrast or deterministically rebuild it from raw data."""

    output_path = Path(processed_path)
    if output_path.exists():
        return pd.read_csv(output_path)
    contrast = prepare_hillstrom_contrast(
        load_hillstrom_raw(raw_path), random_state=random_state
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contrast.to_csv(output_path, index=False)
    return contrast


def build_feature_preprocessor() -> ColumnTransformer:
    """Create leakage-safe numeric passthrough and categorical one-hot encoding."""

    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_hillstrom_arrays(frame: pd.DataFrame) -> dict[str, object]:
    """Fit preprocessing on train only and return the paired modeling arrays."""

    train = frame.loc[frame["split"] == "train"].copy()
    test = frame.loc[frame["split"] == "test"].copy()
    preprocessor = build_feature_preprocessor()
    X_train = preprocessor.fit_transform(train[MODEL_FEATURES])
    X_test = preprocessor.transform(test[MODEL_FEATURES])
    feature_names = preprocessor.get_feature_names_out().tolist()

    return {
        "X_train": np.asarray(X_train, dtype=float),
        "X_test": np.asarray(X_test, dtype=float),
        "t_train": train["treatment"].to_numpy(dtype=int),
        "t_test": test["treatment"].to_numpy(dtype=int),
        "y_train": train[PRIMARY_OUTCOME].to_numpy(dtype=float),
        "y_test": test[PRIMARY_OUTCOME].to_numpy(dtype=float),
        "test_frame": test.reset_index(drop=True),
        "feature_names": feature_names,
        "preprocessor": preprocessor,
    }


def standardized_mean_differences(frame: pd.DataFrame) -> pd.DataFrame:
    """Check randomization balance after fitting the declared preprocessing."""

    preprocessor = build_feature_preprocessor()
    matrix = preprocessor.fit_transform(frame[MODEL_FEATURES])
    names = preprocessor.get_feature_names_out()
    treatment = frame["treatment"].to_numpy(dtype=int)
    treated = matrix[treatment == 1]
    control = matrix[treatment == 0]
    pooled_sd = np.sqrt((treated.var(axis=0, ddof=1) + control.var(axis=0, ddof=1)) / 2)
    difference = treated.mean(axis=0) - control.mean(axis=0)
    smd = np.divide(difference, pooled_sd, out=np.zeros_like(difference), where=pooled_sd > 0)
    return pd.DataFrame({"feature": names, "smd": smd, "abs_smd": np.abs(smd)}).sort_values(
        "abs_smd", ascending=False
    )


def experimental_effect_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize unadjusted randomized effects for each campaign versus control."""

    rows: list[dict[str, float | str]] = []
    control = frame.loc[frame["segment"] == CONTROL_ARM]
    for arm in [TREATMENT_ARM, EXCLUDED_ARM]:
        treated = frame.loc[frame["segment"] == arm]
        for outcome in ["visit", "conversion", "spend"]:
            rows.append(
                {
                    "campaign": arm,
                    "outcome": outcome,
                    "treated_mean": float(treated[outcome].mean()),
                    "control_mean": float(control[outcome].mean()),
                    "rct_difference": float(treated[outcome].mean() - control[outcome].mean()),
                }
            )
    return pd.DataFrame(rows)

import hashlib
from pathlib import Path

import numpy as np

from causal_benchmark.hillstrom import (
    HILLSTROM_SHA256,
    MODEL_FEATURES,
    SOURCE_COLUMNS,
    load_hillstrom_raw,
    load_or_create_hillstrom_contrast,
    split_hillstrom_arrays,
    standardized_mean_differences,
    validate_hillstrom_raw,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "hillstrom_email_rct.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "hillstrom_mens_email_vs_control.csv"


def test_raw_hillstrom_file_matches_source_and_passes_quality_checks():
    checksum = hashlib.sha256(RAW_PATH.read_bytes()).hexdigest().upper()
    assert checksum == HILLSTROM_SHA256

    raw = load_hillstrom_raw(RAW_PATH)
    quality = validate_hillstrom_raw(raw)
    assert raw.shape == (64_000, 12)
    assert raw.columns.tolist() == SOURCE_COLUMNS
    assert quality["missing_cells"] == 0
    assert sum(quality["treatment_arm_counts"].values()) == 64_000


def test_processed_contrast_has_shared_balanced_holdout():
    data = load_or_create_hillstrom_contrast(RAW_PATH, PROCESSED_PATH)
    assert len(data) == 42_613
    assert data["customer_id"].is_unique
    assert data["treatment"].value_counts().to_dict() == {1: 21_307, 0: 21_306}
    assert data["split"].value_counts().to_dict() == {"train": 29_829, "test": 12_784}
    holdout = data.loc[data["split"] == "test", "treatment"]
    assert holdout.value_counts().to_dict() == {0: 6_392, 1: 6_392}
    assert standardized_mean_differences(data)["abs_smd"].max() < 0.05


def test_preprocessing_is_fit_on_train_and_excludes_post_treatment_columns():
    data = load_or_create_hillstrom_contrast(RAW_PATH, PROCESSED_PATH)
    arrays = split_hillstrom_arrays(data)
    assert arrays["X_train"].shape == (29_829, 11)
    assert arrays["X_test"].shape == (12_784, 11)
    assert np.isfinite(arrays["X_train"]).all()
    assert set(MODEL_FEATURES) == {
        "recency",
        "history",
        "mens",
        "womens",
        "newbie",
        "zip_code",
        "channel",
    }
    assert set(arrays["feature_names"]).isdisjoint(
        {"segment", "treatment", "visit", "conversion", "spend", "split"}
    )

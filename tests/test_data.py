import numpy as np

from causal_benchmark.data import FEATURE_COLUMNS, generate_coupon_retention_data, split_arrays


def test_generation_is_deterministic_and_valid():
    first = generate_coupon_retention_data(n_samples=600, random_state=7)
    second = generate_coupon_retention_data(n_samples=600, random_state=7)
    assert first.equals(second)
    assert set(FEATURE_COLUMNS).issubset(first.columns)
    assert first["observed_propensity"].between(0.05, 0.95).all()
    assert first["true_cate"].between(-1, 1).all()
    assert set(first["split"]) == {"train", "test"}
    assert first["customer_id"].is_unique
    assert not first.isna().any().any()
    assert set(first["coupon"]) == {0, 1}
    assert set(first["retained_90d"]) == {0, 1}
    assert set(FEATURE_COLUMNS).isdisjoint(
        {"coupon", "retained_90d", "true_cate", "observed_propensity"}
    )


def test_split_excludes_ground_truth_from_features():
    arrays = split_arrays(generate_coupon_retention_data(n_samples=600))
    assert arrays["X_train"].shape[1] == len(FEATURE_COLUMNS)
    assert len(arrays["true_cate_test"]) == len(arrays["X_test"])
    assert not np.isnan(arrays["X_train"]).any()

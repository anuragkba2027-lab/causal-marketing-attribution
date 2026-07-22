import numpy as np

from causal_benchmark.rct_evaluation import (
    evaluate_rct_predictions,
    randomized_difference,
    uplift_calibration_table,
)


def test_randomized_difference_matches_known_contrast():
    result = randomized_difference(
        outcome=np.array([1, 1, 0, 0, 0, 0, 0, 0]),
        treatment=np.array([1, 1, 1, 1, 0, 0, 0, 0]),
    )
    assert result["estimate"] == 0.5
    assert result["treated_n"] == 4
    assert result["control_n"] == 4
    assert result["ci_lower"] < result["estimate"] < result["ci_upper"]


def test_calibration_and_rct_metrics_are_finite():
    rng = np.random.default_rng(42)
    n = 2_000
    feature = rng.normal(size=n)
    treatment = rng.binomial(1, 0.5, size=n)
    true_effect = 0.06 + 0.04 * (feature > 0)
    probability = np.clip(0.12 + treatment * true_effect, 0, 1)
    outcome = rng.binomial(1, probability)
    prediction = true_effect + rng.normal(0, 0.01, size=n)

    calibration = uplift_calibration_table(outcome, treatment, prediction)
    assert len(calibration) == 10
    assert calibration["customers"].sum() == n

    metrics = evaluate_rct_predictions(
        outcome,
        treatment,
        prediction,
        model="test",
        library="test",
        fit_seconds=0.1,
    )
    numeric_keys = [
        "rct_ate",
        "predicted_ate",
        "ate_abs_error",
        "calibration_rmse",
        "qini_score",
        "auuc_score",
        "policy_incremental_visits_per_1000",
    ]
    assert np.isfinite([metrics[key] for key in numeric_keys]).all()
    assert 0 <= metrics["policy_email_rate"] <= 1


def test_rct_evaluation_rejects_shape_mismatch():
    try:
        evaluate_rct_predictions(
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([0.1]),
            model="bad",
            library="test",
            fit_seconds=0.0,
        )
    except ValueError as error:
        assert "lengths must match" in str(error)
    else:
        raise AssertionError("Expected ValueError")

import numpy as np

from causal_benchmark.evaluation import evaluate_cate_predictions


def test_perfect_predictions_have_zero_error_and_regret():
    truth = np.array([-0.03, 0.02, 0.10, 0.20])
    metrics = evaluate_cate_predictions(
        truth,
        truth,
        model="perfect",
        library="test",
        fit_seconds=0.0,
    )
    assert metrics["pehe"] == 0.0
    assert metrics["cate_mae"] == 0.0
    assert abs(metrics["policy_regret_per_1000"]) < 1e-12


def test_shape_mismatch_fails_fast():
    try:
        evaluate_cate_predictions(
            np.array([0.1, 0.2]),
            np.array([0.1]),
            model="bad",
            library="test",
            fit_seconds=0.0,
        )
    except ValueError as error:
        assert "Shape mismatch" in str(error)
    else:
        raise AssertionError("Expected ValueError")

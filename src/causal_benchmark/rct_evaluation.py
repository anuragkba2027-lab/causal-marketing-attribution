"""Held-out randomized-experiment validation for CATE predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from causalml.metrics import auuc_score, qini_score
from scipy.stats import norm, spearmanr


def _as_1d(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float).squeeze().reshape(-1)


def randomized_difference(
    outcome: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Difference in means and normal-approximation CI under random assignment."""

    y = _as_1d(outcome)
    w = np.asarray(treatment, dtype=int).reshape(-1)
    treated = y[w == 1]
    control = y[w == 0]
    estimate = float(treated.mean() - control.mean())
    standard_error = float(
        np.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control))
    )
    critical_value = float(norm.ppf(1 - alpha / 2))
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "ci_lower": estimate - critical_value * standard_error,
        "ci_upper": estimate + critical_value * standard_error,
        "treated_n": int(len(treated)),
        "control_n": int(len(control)),
        "treated_mean": float(treated.mean()),
        "control_mean": float(control.mean()),
    }


def uplift_calibration_table(
    outcome: np.ndarray,
    treatment: np.ndarray,
    predicted_cate: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Estimate randomized uplift inside predicted-CATE quantiles."""

    frame = pd.DataFrame(
        {
            "outcome": _as_1d(outcome),
            "treatment": np.asarray(treatment, dtype=int).reshape(-1),
            "predicted_cate": _as_1d(predicted_cate),
        }
    )
    ranked = frame["predicted_cate"].rank(method="first")
    frame["uplift_decile"] = pd.qcut(ranked, q=n_bins, labels=False) + 1
    rows: list[dict[str, float | int]] = []
    for decile, group in frame.groupby("uplift_decile", sort=True):
        estimate = randomized_difference(group["outcome"], group["treatment"])
        rows.append(
            {
                "uplift_decile": int(decile),
                "customers": int(len(group)),
                "treated_n": int(estimate["treated_n"]),
                "control_n": int(estimate["control_n"]),
                "mean_predicted_cate": float(group["predicted_cate"].mean()),
                "observed_rct_uplift": float(estimate["estimate"]),
                "uplift_ci_lower": float(estimate["ci_lower"]),
                "uplift_ci_upper": float(estimate["ci_upper"]),
            }
        )
    return pd.DataFrame(rows)


def evaluate_rct_predictions(
    outcome: np.ndarray,
    treatment: np.ndarray,
    predicted_cate: np.ndarray,
    *,
    model: str,
    library: str,
    fit_seconds: float,
    assignment_probability: float = 0.5,
) -> dict[str, float | str | bool]:
    """Compare model-implied effects with an untouched randomized holdout."""

    y = _as_1d(outcome)
    w = np.asarray(treatment, dtype=int).reshape(-1)
    prediction = _as_1d(predicted_cate)
    if not (len(y) == len(w) == len(prediction)):
        raise ValueError("Outcome, treatment, and CATE prediction lengths must match")
    if not np.isfinite(prediction).all():
        raise ValueError(f"{library}/{model} produced non-finite CATE predictions")

    benchmark = randomized_difference(y, w)
    calibration = uplift_calibration_table(y, w, prediction)
    calibration_rmse = float(
        np.sqrt(
            np.mean(
                (
                    calibration["mean_predicted_cate"]
                    - calibration["observed_rct_uplift"]
                )
                ** 2
            )
        )
    )
    calibration_rank = float(
        spearmanr(
            calibration["mean_predicted_cate"], calibration["observed_rct_uplift"]
        ).statistic
    )
    if np.isnan(calibration_rank):
        calibration_rank = 0.0

    score_frame = pd.DataFrame({"y": y, "w": w, "prediction": prediction})
    qini = float(
        qini_score(
            score_frame,
            outcome_col="y",
            treatment_col="w",
            treatment_effect_col="true_effect_not_observed",
        )["prediction"]
    )
    auuc = float(
        auuc_score(
            score_frame,
            outcome_col="y",
            treatment_col="w",
            treatment_effect_col="true_effect_not_observed",
        )["prediction"]
    )

    policy = prediction > 0
    ipw_effect = w * y / assignment_probability - (1 - w) * y / (
        1 - assignment_probability
    )
    policy_value = float(np.mean(policy * ipw_effect) * 1_000)
    treat_rate = float(policy.mean())
    efficiency = policy_value / treat_rate if treat_rate > 0 else 0.0

    top_cutoff = np.quantile(prediction, 0.70)
    top_group = prediction >= top_cutoff
    top_effect = randomized_difference(y[top_group], w[top_group])

    predicted_ate = float(prediction.mean())
    return {
        "library": library,
        "model": model,
        "rct_ate": float(benchmark["estimate"]),
        "rct_ci_lower": float(benchmark["ci_lower"]),
        "rct_ci_upper": float(benchmark["ci_upper"]),
        "predicted_ate": predicted_ate,
        "ate_abs_error": abs(predicted_ate - float(benchmark["estimate"])),
        "ate_within_rct_95ci": bool(
            benchmark["ci_lower"] <= predicted_ate <= benchmark["ci_upper"]
        ),
        "calibration_rmse": calibration_rmse,
        "calibration_rank": calibration_rank,
        "qini_score": qini,
        "auuc_score": auuc,
        "top_30pct_rct_uplift": float(top_effect["estimate"]),
        "policy_email_rate": treat_rate,
        "policy_incremental_visits_per_1000": policy_value,
        "incremental_visits_per_1000_emailed": efficiency,
        "fit_seconds": float(fit_seconds),
    }

"""Ground-truth CATE and marketing-policy evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _as_1d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.squeeze(array).reshape(-1)


def evaluate_cate_predictions(
    true_cate: np.ndarray,
    predicted_cate: np.ndarray,
    *,
    model: str,
    library: str,
    fit_seconds: float,
    retained_customer_value: float = 120.0,
    coupon_cost: float = 10.0,
) -> dict[str, float | str]:
    """Compute accuracy, ranking, and oracle policy-value metrics.

    Monetary metrics are simulation outputs under explicit unit-economics
    assumptions. They are not claims about a real production campaign.
    """

    truth = _as_1d(true_cate)
    prediction = _as_1d(predicted_cate)
    if truth.shape != prediction.shape:
        raise ValueError(f"Shape mismatch: truth={truth.shape}, prediction={prediction.shape}")
    if not np.isfinite(prediction).all():
        raise ValueError(f"{library}/{model} produced non-finite CATE predictions")

    error = prediction - truth
    threshold = coupon_cost / retained_customer_value
    policy = prediction > threshold
    oracle_policy = truth > threshold
    unit_net_value = truth * retained_customer_value - coupon_cost
    policy_value = float(np.mean(policy * unit_net_value) * 1_000)
    oracle_value = float(np.mean(oracle_policy * unit_net_value) * 1_000)
    treat_all_value = float(np.mean(unit_net_value) * 1_000)
    targeted = int(policy.sum())
    incremental_value = float(np.sum(policy * truth * retained_customer_value))
    spend = float(targeted * coupon_cost)

    rank_correlation = float(spearmanr(truth, prediction).statistic)
    if np.isnan(rank_correlation):
        rank_correlation = 0.0

    return {
        "library": library,
        "model": model,
        "pehe": float(np.sqrt(np.mean(error**2))),
        "cate_mae": float(np.mean(np.abs(error))),
        "ate_true": float(np.mean(truth)),
        "ate_estimate": float(np.mean(prediction)),
        "ate_abs_error": float(abs(np.mean(prediction) - np.mean(truth))),
        "spearman_rank": rank_correlation,
        "effect_sign_accuracy": float(np.mean((prediction > 0) == (truth > 0))),
        "fit_seconds": float(fit_seconds),
        "policy_treat_rate": float(np.mean(policy)),
        "policy_value_per_1000": policy_value,
        "oracle_value_per_1000": oracle_value,
        "policy_regret_per_1000": oracle_value - policy_value,
        "treat_all_value_per_1000": treat_all_value,
        "value_vs_treat_all_per_1000": policy_value - treat_all_value,
        "policy_roi": (incremental_value / spend - 1.0) if spend else 0.0,
    }


def build_calibration_table(
    true_cate: np.ndarray,
    predicted_cate: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compare predicted and true CATE across predicted-uplift quantiles."""

    table = pd.DataFrame(
        {"true_cate": _as_1d(true_cate), "predicted_cate": _as_1d(predicted_cate)}
    )
    table["predicted_uplift_decile"] = pd.qcut(
        table["predicted_cate"], q=n_bins, labels=False, duplicates="drop"
    )
    return (
        table.groupby("predicted_uplift_decile", as_index=False)
        .agg(
            customers=("true_cate", "size"),
            mean_predicted_cate=("predicted_cate", "mean"),
            mean_true_cate=("true_cate", "mean"),
        )
        .sort_values("predicted_uplift_decile", ascending=False)
    )


def naive_difference_in_means(frame: pd.DataFrame) -> float:
    """Observed treated-control retention difference (intentionally biased)."""

    treated = frame.loc[frame["coupon"] == 1, "retained_90d"].mean()
    control = frame.loc[frame["coupon"] == 0, "retained_90d"].mean()
    return float(treated - control)

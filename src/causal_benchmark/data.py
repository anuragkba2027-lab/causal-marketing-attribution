"""Deterministic marketing data with observable CATE ground truth.

The data-generating process mimics historical coupon targeting. Treatment is
confounded by pre-treatment customer attributes, while both potential response
probabilities are retained for direct evaluation of CATE estimates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "recency_days",
    "orders_12m",
    "avg_order_value",
    "tenure_months",
    "app_sessions_30d",
    "email_open_rate",
    "price_sensitivity",
    "prior_coupon_rate",
]

OUTCOME_COLUMN = "retained_90d"
TREATMENT_COLUMN = "coupon"
TRUE_CATE_COLUMN = "true_cate"


def generate_coupon_retention_data(
    n_samples: int = 8_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate an observational coupon dataset with known treatment effects.

    The outcome is 90-day retention. The treatment is a discount coupon. The
    assignment mechanism intentionally targets customers using variables that
    also drive retention and responsiveness, creating selection bias.

    ``true_cate`` is E[Y(1)-Y(0)|X] and is known exactly from the simulation.
    It is a probability-point difference, not the unobservable realized
    individual treatment effect.
    """

    if n_samples < 500:
        raise ValueError("n_samples must be at least 500 for stable treatment arms")

    rng = np.random.default_rng(random_state)

    recency_days = np.clip(rng.gamma(shape=2.2, scale=27.0, size=n_samples), 1, 180)
    orders_12m = np.clip(rng.poisson(lam=4.5, size=n_samples) + 1, 1, 22)
    avg_order_value = np.clip(rng.lognormal(mean=4.25, sigma=0.48, size=n_samples), 15, 350)
    tenure_months = rng.integers(2, 73, size=n_samples)
    app_sessions_30d = np.clip(rng.negative_binomial(n=3, p=0.34, size=n_samples), 0, 35)
    email_open_rate = rng.beta(a=2.2, b=3.4, size=n_samples)
    price_sensitivity = rng.beta(a=2.0, b=2.1, size=n_samples)
    prior_coupon_rate = rng.beta(a=1.5, b=4.2, size=n_samples)

    # Fixed, documented transforms keep the DGP reproducible and interpretable.
    z_recency = (recency_days - 60.0) / 45.0
    z_orders = (orders_12m - 5.5) / 3.5
    z_value = (np.log(avg_order_value) - 4.25) / 0.48
    z_tenure = (tenure_months - 36.0) / 22.0
    z_sessions = (app_sessions_30d - 5.8) / 5.0

    baseline_log_odds = (
        -0.50
        - 0.34 * z_recency
        + 0.22 * z_orders
        + 0.18 * z_value
        + 0.15 * z_tenure
        + 0.18 * z_sessions
        + 0.48 * email_open_rate
        - 0.16 * price_sensitivity
        + 0.10 * np.sin(z_orders)
    )

    # Heterogeneous coupon response: price-sensitive, reachable, recently
    # inactive users can be persuadable; habitual coupon users and high-value
    # loyal customers can have weak or negative incremental response.
    coupon_log_odds_shift = (
        -0.62
        + 1.45 * price_sensitivity
        + 0.78 * email_open_rate
        + 0.30 * np.maximum(z_recency, 0)
        - 0.70 * prior_coupon_rate
        - 0.28 * np.maximum(z_value, 0)
        - 0.18 * np.maximum(z_orders, 0)
        + 0.20 * price_sensitivity * email_open_rate
    )

    p_retained_control = expit(baseline_log_odds)
    p_retained_treated = expit(baseline_log_odds + coupon_log_odds_shift)
    true_cate = p_retained_treated - p_retained_control

    # Historical marketers preferentially coupon customers who look responsive.
    # This satisfies observed-confounder ignorability because every assignment
    # driver below is included in FEATURE_COLUMNS.
    assignment_log_odds = (
        -0.85
        + 1.25 * price_sensitivity
        + 0.95 * email_open_rate
        - 0.35 * z_recency
        + 0.65 * z_orders
        + 0.45 * z_sessions
        + 0.42 * prior_coupon_rate
    )
    propensity = np.clip(expit(assignment_log_odds), 0.05, 0.95)
    coupon = rng.binomial(1, propensity)

    observed_probability = np.where(coupon == 1, p_retained_treated, p_retained_control)
    retained_90d = rng.binomial(1, observed_probability)

    frame = pd.DataFrame(
        {
            "customer_id": np.arange(1, n_samples + 1),
            "recency_days": recency_days.round(2),
            "orders_12m": orders_12m,
            "avg_order_value": avg_order_value.round(2),
            "tenure_months": tenure_months,
            "app_sessions_30d": app_sessions_30d,
            "email_open_rate": email_open_rate.round(6),
            "price_sensitivity": price_sensitivity.round(6),
            "prior_coupon_rate": prior_coupon_rate.round(6),
            "coupon": coupon,
            "retained_90d": retained_90d,
            "observed_propensity": propensity,
            "p_retained_control": p_retained_control,
            "p_retained_treated": p_retained_treated,
            "true_cate": true_cate,
        }
    )
    return _assign_split(frame, random_state=random_state)


def _assign_split(frame: pd.DataFrame, random_state: int) -> pd.DataFrame:
    """Add one shared stratified train/test split used by both libraries."""

    train_ids, test_ids = train_test_split(
        frame.index,
        test_size=0.30,
        random_state=random_state,
        stratify=frame[TREATMENT_COLUMN],
    )
    frame = frame.copy()
    frame["split"] = "train"
    frame.loc[test_ids, "split"] = "test"
    assert len(train_ids) + len(test_ids) == len(frame)
    return frame


def load_or_create_data(
    path: str | Path = "data/processed/coupon_retention_benchmark.csv",
    n_samples: int = 8_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load the benchmark CSV or create it deterministically if absent."""

    data_path = Path(path)
    if data_path.exists():
        return pd.read_csv(data_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_coupon_retention_data(n_samples=n_samples, random_state=random_state)
    frame.to_csv(data_path, index=False)
    return frame


def split_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray | pd.DataFrame]:
    """Return the shared modeling arrays without leaking ground truth into X."""

    train = frame.loc[frame["split"] == "train"].copy()
    test = frame.loc[frame["split"] == "test"].copy()
    return {
        "X_train": train[FEATURE_COLUMNS].to_numpy(dtype=float),
        "X_test": test[FEATURE_COLUMNS].to_numpy(dtype=float),
        "t_train": train[TREATMENT_COLUMN].to_numpy(dtype=int),
        "t_test": test[TREATMENT_COLUMN].to_numpy(dtype=int),
        "y_train": train[OUTCOME_COLUMN].to_numpy(dtype=float),
        "y_test": test[OUTCOME_COLUMN].to_numpy(dtype=float),
        "true_cate_test": test[TRUE_CATE_COLUMN].to_numpy(dtype=float),
        "test_frame": test.reset_index(drop=True),
    }

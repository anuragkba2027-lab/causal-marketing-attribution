"""Uber CausalML model benchmark on the shared marketing dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from causalml.inference.meta import (
    BaseDRRegressor,
    BaseRRegressor,
    BaseSRegressor,
    BaseTRegressor,
    BaseXRegressor,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import split_arrays
from .evaluation import evaluate_cate_predictions
from .hillstrom import split_hillstrom_arrays
from .rct_evaluation import evaluate_rct_predictions


@dataclass
class BenchmarkResult:
    """Metrics, predictions, and fitted estimators from one library run."""

    metrics: pd.DataFrame
    predictions: pd.DataFrame
    estimators: dict[str, object]


def _forest(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=25,
        max_features=0.8,
        # Single-threaded fits keep notebook artifacts bitwise reproducible.
        n_jobs=1,
        random_state=seed,
    )


def _propensity_model(seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2_000, random_state=seed),
    )


def _r_learner(forest_seed: int, cv_seed: int) -> BaseRRegressor:
    """Build an R-Learner without spawning cross-validation worker processes."""

    estimator = BaseRRegressor(
        learner=_forest(forest_seed), n_fold=3, random_state=cv_seed
    )
    # BaseRRegressor 0.16 does not expose BaseRLearner.cv_n_jobs in its constructor.
    estimator.cv_n_jobs = 1
    return estimator


def run_causalml_benchmark(
    frame: pd.DataFrame,
    *,
    artifact_dir: str | Path = "artifacts",
    random_state: int = 42,
) -> BenchmarkResult:
    """Fit S/T/X/R/DR learners and persist test-set predictions and metrics."""

    arrays = split_arrays(frame)
    X_train = np.asarray(arrays["X_train"])
    X_test = np.asarray(arrays["X_test"])
    treatment = np.asarray(arrays["t_train"])
    outcome = np.asarray(arrays["y_train"])
    truth = np.asarray(arrays["true_cate_test"])
    test_frame = arrays["test_frame"]

    propensity_model = _propensity_model(random_state)
    propensity_model.fit(X_train, treatment)
    p_train = np.clip(propensity_model.predict_proba(X_train)[:, 1], 0.03, 0.97)
    p_test = np.clip(propensity_model.predict_proba(X_test)[:, 1], 0.03, 0.97)

    estimators: dict[str, object] = {
        "S-Learner": BaseSRegressor(learner=_forest(random_state)),
        "T-Learner": BaseTRegressor(learner=_forest(random_state + 1)),
        "X-Learner": BaseXRegressor(learner=_forest(random_state + 2)),
        "R-Learner": _r_learner(random_state + 3, random_state),
        "DR-Learner": BaseDRRegressor(learner=_forest(random_state + 4)),
    }

    metrics_rows: list[dict[str, float | str]] = []
    prediction_frame = pd.DataFrame(
        {
            "customer_id": test_frame["customer_id"].to_numpy(),
            "true_cate": truth,
        }
    )

    for model_name, estimator in estimators.items():
        started = perf_counter()
        if model_name == "DR-Learner":
            estimator.fit(X_train, treatment, outcome, p=p_train, seed=random_state)
            prediction = estimator.predict(X_test, p=p_test)
        elif model_name in {"X-Learner", "R-Learner"}:
            estimator.fit(X_train, treatment, outcome, p=p_train)
            prediction = estimator.predict(X_test, p=p_test)
        else:
            estimator.fit(X_train, treatment, outcome)
            prediction = estimator.predict(X_test, verbose=False)
        fit_seconds = perf_counter() - started
        prediction = np.asarray(prediction, dtype=float).squeeze().reshape(-1)
        prediction_frame[model_name] = prediction
        metrics_rows.append(
            evaluate_cate_predictions(
                truth,
                prediction,
                model=model_name,
                library="CausalML",
                fit_seconds=fit_seconds,
            )
        )

    metrics = pd.DataFrame(metrics_rows).sort_values("pehe").reset_index(drop=True)
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "synthetic_causalml_metrics.csv", index=False)
    prediction_frame.to_csv(output_dir / "synthetic_causalml_predictions.csv", index=False)
    return BenchmarkResult(metrics=metrics, predictions=prediction_frame, estimators=estimators)


def run_causalml_rct_benchmark(
    frame: pd.DataFrame,
    *,
    artifact_dir: str | Path = "artifacts",
    random_state: int = 42,
) -> BenchmarkResult:
    """Fit five CausalML estimators and validate them on the randomized holdout."""

    arrays = split_hillstrom_arrays(frame)
    X_train = np.asarray(arrays["X_train"])
    X_test = np.asarray(arrays["X_test"])
    treatment_train = np.asarray(arrays["t_train"])
    treatment_test = np.asarray(arrays["t_test"])
    outcome_train = np.asarray(arrays["y_train"])
    outcome_test = np.asarray(arrays["y_test"])
    test_frame = arrays["test_frame"]

    # The pairwise Hillstrom contrast was randomized 1:1 by design.
    p_train = np.full(len(X_train), 0.5, dtype=float)
    p_test = np.full(len(X_test), 0.5, dtype=float)
    estimators: dict[str, object] = {
        "S-Learner": BaseSRegressor(learner=_forest(random_state)),
        "T-Learner": BaseTRegressor(learner=_forest(random_state + 1)),
        "X-Learner": BaseXRegressor(learner=_forest(random_state + 2)),
        "R-Learner": _r_learner(random_state + 3, random_state),
        "DR-Learner": BaseDRRegressor(learner=_forest(random_state + 4)),
    }

    rows: list[dict[str, float | str | bool]] = []
    predictions = pd.DataFrame(
        {
            "customer_id": test_frame["customer_id"].to_numpy(),
            "treatment": treatment_test,
            "outcome": outcome_test,
        }
    )
    for model_name, estimator in estimators.items():
        started = perf_counter()
        if model_name == "DR-Learner":
            estimator.fit(
                X_train,
                treatment_train,
                outcome_train,
                p=p_train,
                seed=random_state,
            )
            prediction = estimator.predict(X_test, p=p_test)
        elif model_name in {"X-Learner", "R-Learner"}:
            estimator.fit(X_train, treatment_train, outcome_train, p=p_train)
            prediction = estimator.predict(X_test, p=p_test)
        else:
            estimator.fit(X_train, treatment_train, outcome_train)
            prediction = estimator.predict(X_test, verbose=False)
        fit_seconds = perf_counter() - started
        prediction = np.asarray(prediction, dtype=float).squeeze().reshape(-1)
        predictions[model_name] = prediction
        rows.append(
            evaluate_rct_predictions(
                outcome_test,
                treatment_test,
                prediction,
                model=model_name,
                library="CausalML",
                fit_seconds=fit_seconds,
            )
        )

    metrics = pd.DataFrame(rows).sort_values("qini_score", ascending=False).reset_index(
        drop=True
    )
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "causalml_metrics.csv", index=False)
    predictions.to_csv(output_dir / "causalml_predictions.csv", index=False)
    return BenchmarkResult(metrics=metrics, predictions=predictions, estimators=estimators)

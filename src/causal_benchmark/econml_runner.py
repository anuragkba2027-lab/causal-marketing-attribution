"""Microsoft Research/PyWhy EconML benchmark on the shared dataset."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from econml.dr import DRLearner
from econml.metalearners import SLearner, TLearner, XLearner
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .causalml_runner import BenchmarkResult
from .data import split_arrays
from .evaluation import evaluate_cate_predictions
from .hillstrom import split_hillstrom_arrays
from .rct_evaluation import evaluate_rct_predictions


def _outcome_forest(seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=25,
        max_features=0.8,
        # Single-threaded fits keep notebook artifacts bitwise reproducible.
        n_jobs=1,
        random_state=seed,
    )


def _outcome_classifier(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=25,
        max_features=0.8,
        n_jobs=1,
        random_state=seed,
    )


def _propensity(seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2_000, random_state=seed),
    )


def run_econml_benchmark(
    frame: pd.DataFrame,
    *,
    artifact_dir: str | Path = "artifacts",
    random_state: int = 42,
) -> BenchmarkResult:
    """Fit S/T/X/DR/CausalForestDML and persist shared holdout results."""

    arrays = split_arrays(frame)
    X_train = np.asarray(arrays["X_train"])
    X_test = np.asarray(arrays["X_test"])
    treatment = np.asarray(arrays["t_train"])
    outcome = np.asarray(arrays["y_train"])
    truth = np.asarray(arrays["true_cate_test"])
    test_frame = arrays["test_frame"]

    estimators: dict[str, object] = {
        "S-Learner": SLearner(overall_model=_outcome_forest(random_state)),
        "T-Learner": TLearner(
            models=[_outcome_forest(random_state + 1), _outcome_forest(random_state + 2)]
        ),
        "X-Learner": XLearner(
            models=[_outcome_forest(random_state + 3), _outcome_forest(random_state + 4)],
            cate_models=[_outcome_forest(random_state + 5), _outcome_forest(random_state + 6)],
            propensity_model=_propensity(random_state),
        ),
        "DR-Learner": DRLearner(
            model_propensity=_propensity(random_state),
            model_regression=_outcome_classifier(random_state + 7),
            model_final=_outcome_forest(random_state + 8),
            discrete_outcome=True,
            cv=3,
            min_propensity=0.03,
            random_state=random_state,
        ),
        "CausalForestDML": CausalForestDML(
            model_y=_outcome_classifier(random_state + 9),
            model_t=_propensity(random_state),
            discrete_outcome=True,
            discrete_treatment=True,
            cv=3,
            n_estimators=160,
            min_samples_leaf=20,
            max_depth=12,
            max_features=0.8,
            inference=False,
            n_jobs=1,
            random_state=random_state,
        ),
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
        estimator.fit(outcome, treatment, X=X_train)
        prediction = estimator.effect(X_test)
        fit_seconds = perf_counter() - started
        prediction = np.asarray(prediction, dtype=float).squeeze().reshape(-1)
        prediction_frame[model_name] = prediction
        metrics_rows.append(
            evaluate_cate_predictions(
                truth,
                prediction,
                model=model_name,
                library="EconML",
                fit_seconds=fit_seconds,
            )
        )

    metrics = pd.DataFrame(metrics_rows).sort_values("pehe").reset_index(drop=True)
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "synthetic_econml_metrics.csv", index=False)
    prediction_frame.to_csv(output_dir / "synthetic_econml_predictions.csv", index=False)
    return BenchmarkResult(metrics=metrics, predictions=prediction_frame, estimators=estimators)


def _randomized_propensity() -> DummyClassifier:
    """Estimate only the fold-specific randomized treatment prior."""

    return DummyClassifier(strategy="prior")


def run_econml_rct_benchmark(
    frame: pd.DataFrame,
    *,
    artifact_dir: str | Path = "artifacts",
    random_state: int = 42,
) -> BenchmarkResult:
    """Fit five EconML estimators and validate them on the randomized holdout."""

    arrays = split_hillstrom_arrays(frame)
    X_train = np.asarray(arrays["X_train"])
    X_test = np.asarray(arrays["X_test"])
    treatment_train = np.asarray(arrays["t_train"])
    treatment_test = np.asarray(arrays["t_test"])
    outcome_train = np.asarray(arrays["y_train"])
    outcome_test = np.asarray(arrays["y_test"])
    test_frame = arrays["test_frame"]

    estimators: dict[str, object] = {
        "S-Learner": SLearner(overall_model=_outcome_forest(random_state)),
        "T-Learner": TLearner(
            models=[_outcome_forest(random_state + 1), _outcome_forest(random_state + 2)]
        ),
        "X-Learner": XLearner(
            models=[_outcome_forest(random_state + 3), _outcome_forest(random_state + 4)],
            cate_models=[_outcome_forest(random_state + 5), _outcome_forest(random_state + 6)],
            propensity_model=_randomized_propensity(),
        ),
        "DR-Learner": DRLearner(
            model_propensity=_randomized_propensity(),
            model_regression=_outcome_classifier(random_state + 7),
            model_final=_outcome_forest(random_state + 8),
            discrete_outcome=True,
            cv=3,
            min_propensity=0.03,
            random_state=random_state,
        ),
        "CausalForestDML": CausalForestDML(
            model_y=_outcome_classifier(random_state + 9),
            model_t=_randomized_propensity(),
            discrete_outcome=True,
            discrete_treatment=True,
            cv=3,
            n_estimators=160,
            min_samples_leaf=30,
            max_depth=12,
            max_features=0.8,
            inference=False,
            n_jobs=1,
            random_state=random_state,
        ),
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
        estimator.fit(outcome_train, treatment_train, X=X_train)
        prediction = np.asarray(estimator.effect(X_test), dtype=float).squeeze().reshape(-1)
        fit_seconds = perf_counter() - started
        predictions[model_name] = prediction
        rows.append(
            evaluate_rct_predictions(
                outcome_test,
                treatment_test,
                prediction,
                model=model_name,
                library="EconML",
                fit_seconds=fit_seconds,
            )
        )

    metrics = pd.DataFrame(rows).sort_values("qini_score", ascending=False).reset_index(
        drop=True
    )
    output_dir = Path(artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "econml_metrics.csv", index=False)
    predictions.to_csv(output_dir / "econml_predictions.csv", index=False)
    return BenchmarkResult(metrics=metrics, predictions=predictions, estimators=estimators)

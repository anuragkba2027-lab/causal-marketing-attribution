"""Cross-library result loading and paired prediction comparisons."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_benchmark_artifacts(
    artifact_dir: str | Path = "artifacts",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and validate outputs produced by both library notebooks."""

    root = Path(artifact_dir)
    metric_paths = [root / "causalml_metrics.csv", root / "econml_metrics.csv"]
    prediction_paths = [root / "causalml_predictions.csv", root / "econml_predictions.csv"]
    missing = [str(path) for path in metric_paths + prediction_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run notebooks 01 and 02 first. Missing artifacts: " + ", ".join(missing)
        )

    metrics = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True)
    causal_predictions = pd.read_csv(prediction_paths[0])
    econ_predictions = pd.read_csv(prediction_paths[1])
    keys = ["customer_id", "treatment", "outcome"]
    if not causal_predictions[keys].equals(
        econ_predictions[keys]
    ):
        raise ValueError("Libraries were not evaluated on the identical holdout rows")
    return metrics, causal_predictions, econ_predictions


def paired_family_comparison(
    metrics: pd.DataFrame,
    shared_families: tuple[str, ...] = ("S-Learner", "T-Learner", "X-Learner", "DR-Learner"),
    metric: str = "qini_score",
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Create a side-by-side RCT-validation comparison for shared model families."""

    shared = metrics.loc[metrics["model"].isin(shared_families)].copy()
    table = shared.pivot(index="model", columns="library", values=metric)
    required = {"CausalML", "EconML"}
    if not required.issubset(table.columns):
        raise ValueError(f"Expected libraries {sorted(required)} in metric artifacts")
    table[f"absolute_{metric}_gap"] = np.abs(table["CausalML"] - table["EconML"])
    if higher_is_better:
        table["better_library"] = np.where(
            table["CausalML"] > table["EconML"], "CausalML", "EconML"
        )
    else:
        table["better_library"] = np.where(
            table["CausalML"] < table["EconML"], "CausalML", "EconML"
        )
    return table.reset_index().sort_values(f"absolute_{metric}_gap")

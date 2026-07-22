"""Consistent plots shared by all notebooks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def set_plot_style() -> None:
    """Apply one accessible visual style."""

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.figsize"] = (9, 5)
    plt.rcParams["axes.titleweight"] = "bold"


def plot_metric_comparison(
    metrics: pd.DataFrame,
    metric: str = "pehe",
    *,
    title: str | None = None,
):
    """Create a zero-based comparison bar chart."""

    order = metrics.sort_values(metric)["model"]
    libraries = metrics["library"].drop_duplicates().tolist()
    if len(libraries) == 1:
        ax = sns.barplot(
            data=metrics, x=metric, y="model", order=order, color="#4C78A8"
        )
    else:
        palette = {"CausalML": "#4C78A8", "EconML": "#F58518"}
        ax = sns.barplot(
            data=metrics, x=metric, y="model", hue="library", order=order, palette=palette
        )
    ax.set_xlim(left=0)
    ax.set_title(title or f"Model comparison: {metric}")
    if metric == "pehe":
        ax.set_xlabel("PEHE (retention-probability points; lower is better)")
    else:
        ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("")
    return ax


def plot_rct_metric(
    metrics: pd.DataFrame,
    metric: str,
    *,
    title: str,
    x_label: str,
    higher_is_better: bool = True,
):
    """Plot an RCT validation metric with an honest zero reference."""

    ordered = metrics.sort_values(metric, ascending=not higher_is_better)
    libraries = ordered["library"].drop_duplicates().tolist()
    if len(libraries) == 1:
        ax = sns.barplot(data=ordered, x=metric, y="model", color="#4C78A8")
    else:
        ax = sns.barplot(
            data=ordered,
            x=metric,
            y="model",
            hue="library",
            palette={"CausalML": "#4C78A8", "EconML": "#F58518"},
        )
    ax.axvline(0, color="#374151", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("")
    return ax


def plot_ate_against_rct(metrics: pd.DataFrame, *, title: str):
    """Compare every model-implied ATE with the randomized holdout interval."""

    ordered = metrics.sort_values(["library", "predicted_ate"]).reset_index(drop=True)
    labels = ordered["model"].astype(str)
    if ordered["library"].nunique() > 1:
        labels = ordered["library"].astype(str) + " · " + labels
    colors = ordered["library"].map(
        {"CausalML": "#4C78A8", "EconML": "#F58518"}
    ).fillna("#4C78A8")
    positions = np.arange(len(ordered))
    figure, ax = plt.subplots(figsize=(9, max(4.5, 0.48 * len(ordered))))
    lower = float(ordered["rct_ci_lower"].iloc[0])
    upper = float(ordered["rct_ci_upper"].iloc[0])
    estimate = float(ordered["rct_ate"].iloc[0])
    ax.axvspan(lower, upper, color="#D1D5DB", alpha=0.65, label="RCT 95% CI")
    ax.axvline(estimate, color="#111827", linestyle="--", label="RCT estimate")
    ax.scatter(ordered["predicted_ate"], positions, c=colors, s=55, zorder=3)
    ax.set_yticks(positions, labels)
    ax.set_xlabel("Estimated visit uplift")
    ax.set_ylabel("")
    ax.set_title(title)
    ax.legend(loc="best")
    return ax


def plot_uplift_calibration(calibration: pd.DataFrame, *, title: str):
    """Plot predicted and randomized uplift by predicted-effect decile."""

    figure, ax = plt.subplots(figsize=(9, 5))
    errors = np.vstack(
        [
            calibration["observed_rct_uplift"] - calibration["uplift_ci_lower"],
            calibration["uplift_ci_upper"] - calibration["observed_rct_uplift"],
        ]
    )
    ax.errorbar(
        calibration["uplift_decile"],
        calibration["observed_rct_uplift"],
        yerr=errors,
        fmt="o-",
        color="#4C78A8",
        capsize=3,
        label="Observed RCT uplift (95% CI)",
    )
    ax.plot(
        calibration["uplift_decile"],
        calibration["mean_predicted_cate"],
        "s--",
        color="#F58518",
        label="Mean predicted CATE",
    )
    ax.axhline(0, color="#374151", linewidth=1)
    ax.set_xticks(calibration["uplift_decile"])
    ax.set_xlabel("Predicted-effect decile (low to high)")
    ax.set_ylabel("Visit uplift")
    ax.set_title(title)
    ax.legend(loc="best")
    return ax

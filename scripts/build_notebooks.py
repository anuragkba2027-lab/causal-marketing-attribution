"""Build the three reader-facing Hillstrom RCT notebooks with nbformat.

Run from the repository root:
    .venv/Scripts/python scripts/build_notebooks.py
"""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def notebook(cells: list, title: str):
    book = nbf.v4.new_notebook(cells=cells)
    book.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (causal-ml)",
            "language": "python",
            "name": "causal-ml",
        },
        "language_info": {"name": "python", "version": "3.11"},
        "title": title,
    }
    return book


common_setup = """
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from sklearn.exceptions import DataConversionWarning

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

RAW_PATH = ROOT / "data" / "raw" / "hillstrom_email_rct.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "hillstrom_mens_email_vs_control.csv"
ARTIFACT_DIR = ROOT / "artifacts"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", message="IProgress not found.*")
"""


causalml_cells = [
    md(
        """
        # Uber CausalML on the Hillstrom randomized email experiment

        ## tl;dr

        The untouched holdout estimates that a Men's Email increased visit rate by
        **7.35 percentage points** (95% CI: **6.13–8.57 pp**) versus No Email. All
        five CausalML estimators produced an average predicted effect inside that
        interval. The **T-Learner** ranked customers best by normalized Qini
        (**0.0266**), while the **X-Learner** had the closest average-effect estimate
        (absolute error **0.0028**).

        These are reproducible RCT validation results—not evidence of an 18% ROI gain.
        """
    ),
    md(
        """
        ## Context & Methods

        The [Hillstrom email challenge](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)
        contains 64,000 customers randomly assigned to Men's Email, Women's Email,
        or No Email. This notebook compares CausalML S-, T-, X-, R-, and DR-Learners
        on one pre-specified binary intervention: **Men's Email versus No Email**.

        ### Identification and validation contract

        - Random assignment identifies the campaign's average causal effect without
          a no-unmeasured-confounding assumption.
        - The Women's Email arm is excluded—not merged—because it is a different treatment.
        - Models use only pre-treatment features; the primary outcome is a visit in
          the following two weeks.
        - The preprocessor is fitted on training rows only. A stratified 30% holdout
          is never used for model fitting.
        - An RCT validates average and group-level uplift. It cannot expose both
          potential outcomes for one customer, so individual PEHE is not reported.
        """
    ),
    md("## Data"),
    code(common_setup),
    code(
        """
        from causal_benchmark.hillstrom import (
            experimental_effect_table,
            load_hillstrom_raw,
            load_or_create_hillstrom_contrast,
            standardized_mean_differences,
            validate_hillstrom_raw,
        )
        from causal_benchmark.plotting import set_plot_style

        set_plot_style()
        raw = load_hillstrom_raw(RAW_PATH)
        quality = validate_hillstrom_raw(raw)
        display(pd.Series({
            "rows": quality["rows"],
            "columns": quality["columns"],
            "missing_cells": quality["missing_cells"],
            "exact_repeated_rows_retained": quality["exact_repeated_rows"],
            "overall_visit_rate": quality["visit_rate"],
            "overall_conversion_rate": quality["conversion_rate"],
        }, name="value").to_frame().style.format(precision=4))

        arm_counts = pd.Series(quality["treatment_arm_counts"], name="customers")
        display(arm_counts.to_frame())
        """
    ),
    md(
        """
        The source has no customer identifier and contains repeated combinations of
        coarse attributes and outcomes. Those rows are valid randomized units, so
        they are retained and assigned stable row-based IDs. The source spelling
        `Surburban` is also preserved to keep the raw file unchanged.
        """
    ),
    code(
        """
        effects = experimental_effect_table(raw)
        display(effects.style.format({
            "treated_mean": "{:.4f}",
            "control_mean": "{:.4f}",
            "rct_difference": "{:.4f}",
        }))
        """
    ),
    code(
        """
        from causal_benchmark.hillstrom import split_hillstrom_arrays
        from causal_benchmark.rct_evaluation import randomized_difference

        data = load_or_create_hillstrom_contrast(RAW_PATH, PROCESSED_PATH)
        arrays = split_hillstrom_arrays(data)
        holdout = randomized_difference(arrays["y_test"], arrays["t_test"])
        balance = standardized_mean_differences(data)

        design = pd.Series({
            "binary_contrast_customers": len(data),
            "training_customers": int((data["split"] == "train").sum()),
            "holdout_customers": int((data["split"] == "test").sum()),
            "model_features_after_encoding": len(arrays["feature_names"]),
            "maximum_absolute_SMD": balance["abs_smd"].max(),
            "holdout_rct_visit_uplift": holdout["estimate"],
            "holdout_ci_lower": holdout["ci_lower"],
            "holdout_ci_upper": holdout["ci_upper"],
        }, name="value")
        display(design.to_frame().style.format(precision=4))
        display(balance.head(11).style.format({"smd": "{:.4f}", "abs_smd": "{:.4f}"}))
        """
    ),
    md(
        """
        Preprocessing passes through `recency`, `history`, `mens`, `womens`, and
        `newbie`, and one-hot encodes `zip_code` and `channel`. `history_segment`
        is omitted because it deterministically bins `history`. Treatment, campaign
        label, customer ID, outcomes, and split are never model features.
        """
    ),
    md("## Results\n\n### 1. Fit five CausalML estimators"),
    code(
        """
        from causal_benchmark.causalml_runner import run_causalml_rct_benchmark

        result = run_causalml_rct_benchmark(data, artifact_dir=ARTIFACT_DIR)
        metric_columns = [
            "model", "predicted_ate", "ate_abs_error", "ate_within_rct_95ci",
            "qini_score", "auuc_score", "calibration_rmse",
            "top_30pct_rct_uplift", "fit_seconds",
        ]
        display(result.metrics[metric_columns].style.format(precision=4))
        """
    ),
    md("### 2. Compare targeting rank with an RCT-based Qini score"),
    code(
        """
        from causal_benchmark.plotting import plot_rct_metric

        plot_rct_metric(
            result.metrics,
            "qini_score",
            title="CausalML normalized Qini on the randomized holdout",
            x_label="Normalized Qini (higher is better)",
        )
        plt.show()
        """
    ),
    md("### 3. Compare model-implied ATEs with the experimental interval"),
    code(
        """
        from causal_benchmark.plotting import plot_ate_against_rct

        plot_ate_against_rct(
            result.metrics,
            title="CausalML average-effect estimates against the RCT benchmark",
        )
        plt.show()
        """
    ),
    md("### 4. Inspect uplift calibration for the best-ranked estimator"),
    code(
        """
        from causal_benchmark.plotting import plot_uplift_calibration
        from causal_benchmark.rct_evaluation import uplift_calibration_table

        best_model = result.metrics.sort_values("qini_score", ascending=False).iloc[0]["model"]
        calibration = uplift_calibration_table(
            result.predictions["outcome"],
            result.predictions["treatment"],
            result.predictions[best_model],
        )
        display(calibration.style.format(precision=4))
        plot_uplift_calibration(
            calibration,
            title=f"{best_model} predicted and observed uplift by decile",
        )
        plt.show()
        """
    ),
    md("### 5. Express policy performance in observed visits—not dollars"),
    code(
        """
        policy_columns = [
            "model", "policy_email_rate", "policy_incremental_visits_per_1000",
            "incremental_visits_per_1000_emailed", "top_30pct_rct_uplift",
        ]
        display(result.metrics[policy_columns].sort_values(
            "policy_incremental_visits_per_1000", ascending=False
        ).style.format(precision=3))
        """
    ),
    md("## Takeaways"),
    code(
        """
        qini_winner = result.metrics.sort_values("qini_score", ascending=False).iloc[0]
        ate_winner = result.metrics.sort_values("ate_abs_error").iloc[0]
        inside = int(result.metrics["ate_within_rct_95ci"].sum())
        print(
            f"Best CausalML ranking: {qini_winner['model']} "
            f"(Qini={qini_winner['qini_score']:.4f})."
        )
        print(
            f"Closest model-implied ATE: {ate_winner['model']} "
            f"(absolute error={ate_winner['ate_abs_error']:.4f})."
        )
        print(f"ATE estimates inside the RCT 95% CI: {inside}/{len(result.metrics)}.")
        """
    ),
    md(
        """
        The campaign has broad positive lift, so most learned policies email nearly
        everyone. The T-Learner's top 30% has higher observed uplift than the overall
        holdout, but Qini is small and decile intervals are wide. This supports a
        cautious targeting result—not a claim that individualized causal effects are known.
        """
    ),
]


econml_cells = [
    md(
        """
        # Microsoft/PyWhy EconML on the Hillstrom randomized email experiment

        ## tl;dr

        The package is **EconML** (not “EcoML”). On the same untouched holdout, all
        five EconML average-effect estimates fall within the experimental 95% CI.
        The **T-Learner** has the highest normalized Qini (**0.0183**), while
        **CausalForestDML** has the lowest decile calibration RMSE (**0.0257**).

        DML reduces nuisance-model bias through orthogonalization and cross-fitting;
        it is not guaranteed to win every finite-sample uplift ranking.
        """
    ),
    md(
        """
        ## Context & Methods

        This notebook compares EconML S-, T-, X-, and DR-Learners with
        `CausalForestDML`, a nonlinear double/debiased machine-learning estimator.
        The data design, feature allowlist, split, base-forest capacity, primary
        outcome, and RCT validation functions are shared with notebook 01.

        Random assignment is represented by a constant-prior propensity model for
        orthogonal estimators. No propensity model is allowed to invent treatment
        selection that the experiment did not have.
        """
    ),
    md("## Data"),
    code(common_setup),
    code(
        """
        from causal_benchmark.hillstrom import (
            load_or_create_hillstrom_contrast,
            split_hillstrom_arrays,
            standardized_mean_differences,
        )
        from causal_benchmark.plotting import set_plot_style
        from causal_benchmark.rct_evaluation import randomized_difference

        set_plot_style()
        data = load_or_create_hillstrom_contrast(RAW_PATH, PROCESSED_PATH)
        arrays = split_hillstrom_arrays(data)
        holdout = randomized_difference(arrays["y_test"], arrays["t_test"])
        summary = pd.Series({
            "binary_contrast_customers": len(data),
            "training_customers": int((data["split"] == "train").sum()),
            "holdout_customers": int((data["split"] == "test").sum()),
            "treated_holdout_customers": holdout["treated_n"],
            "control_holdout_customers": holdout["control_n"],
            "rct_visit_uplift": holdout["estimate"],
            "rct_ci_lower": holdout["ci_lower"],
            "rct_ci_upper": holdout["ci_upper"],
            "maximum_absolute_SMD": standardized_mean_differences(data)["abs_smd"].max(),
        }, name="value")
        display(summary.to_frame().style.format(precision=4))
        print("Encoded features:", ", ".join(arrays["feature_names"]))
        """
    ),
    md(
        """
        The primary binary contrast has 42,613 customers. A seeded, treatment-stratified
        70/30 split creates the same 29,829 training rows and 12,784 holdout rows for
        every implementation. The holdout contains exactly 6,392 customers per arm.
        """
    ),
    md("## Results\n\n### 1. Fit meta-learners, DR-Learner, and CausalForestDML"),
    code(
        """
        from causal_benchmark.econml_runner import run_econml_rct_benchmark

        result = run_econml_rct_benchmark(data, artifact_dir=ARTIFACT_DIR)
        metric_columns = [
            "model", "predicted_ate", "ate_abs_error", "ate_within_rct_95ci",
            "qini_score", "auuc_score", "calibration_rmse", "calibration_rank",
            "fit_seconds",
        ]
        display(result.metrics[metric_columns].style.format(precision=4))
        """
    ),
    md("### 2. Compare uplift ranking on the randomized holdout"),
    code(
        """
        from causal_benchmark.plotting import plot_rct_metric

        plot_rct_metric(
            result.metrics,
            "qini_score",
            title="EconML normalized Qini on the randomized holdout",
            x_label="Normalized Qini (higher is better)",
        )
        plt.show()
        """
    ),
    md("### 3. Compare model-implied ATEs with the experimental interval"),
    code(
        """
        from causal_benchmark.plotting import plot_ate_against_rct

        plot_ate_against_rct(
            result.metrics,
            title="EconML average-effect estimates against the RCT benchmark",
        )
        plt.show()
        """
    ),
    md("### 4. Inspect CausalForestDML calibration"),
    code(
        """
        from causal_benchmark.plotting import plot_uplift_calibration
        from causal_benchmark.rct_evaluation import uplift_calibration_table

        calibration = uplift_calibration_table(
            result.predictions["outcome"],
            result.predictions["treatment"],
            result.predictions["CausalForestDML"],
        )
        display(calibration.style.format(precision=4))
        plot_uplift_calibration(
            calibration,
            title="CausalForestDML predicted and observed uplift by decile",
        )
        plt.show()
        """
    ),
    md("### 5. Separate three different definitions of ‘best’"),
    code(
        """
        leaders = pd.DataFrame([
            result.metrics.sort_values("qini_score", ascending=False).iloc[0],
            result.metrics.sort_values("calibration_rmse").iloc[0],
            result.metrics.sort_values("ate_abs_error").iloc[0],
        ], index=["Qini ranking", "Calibration RMSE", "ATE absolute error"])
        display(leaders[["model", "qini_score", "calibration_rmse", "ate_abs_error"]]
                .style.format(precision=4))
        """
    ),
    md("## Takeaways"),
    code(
        """
        qini_winner = result.metrics.sort_values("qini_score", ascending=False).iloc[0]
        calibrated = result.metrics.sort_values("calibration_rmse").iloc[0]
        inside = int(result.metrics["ate_within_rct_95ci"].sum())
        print(
            f"Best EconML ranking: {qini_winner['model']} "
            f"(Qini={qini_winner['qini_score']:.4f})."
        )
        print(
            f"Lowest calibration RMSE: {calibrated['model']} "
            f"({calibrated['calibration_rmse']:.4f})."
        )
        print(f"ATE estimates inside the RCT 95% CI: {inside}/{len(result.metrics)}.")
        """
    ),
    md(
        """
        CausalForestDML calibrates the ten uplift groups best on this holdout, but
        the T-Learner ranks customers somewhat better by Qini. The distinction is
        useful in interviews: DML is an estimation framework, whereas the winning
        metric depends on the actual marketing decision.
        """
    ),
]


comparison_cells = [
    md(
        """
        # CausalML versus EconML: paired Hillstrom RCT comparison

        ## tl;dr

        Across ten implementations on exactly the same holdout:

        - **CausalML T-Learner** has the highest Qini (**0.0266**).
        - **EconML CausalForestDML** has the lowest calibration RMSE (**0.0257**).
        - **CausalML X-Learner** is closest to the RCT ATE (error **0.0028**).
        - All **10/10** model-implied ATEs lie inside the RCT 95% confidence interval.

        This is a model-validation result, not a production ROI result.
        """
    ),
    md(
        """
        ## Context & Methods

        The comparison is paired: both libraries receive identical customer rows,
        features, treatment, outcome, split, and evaluation functions. S/T/X/DR
        families are compared directly. CausalML R-Learner and EconML
        CausalForestDML remain in the overall ranking as library-specific additions.

        ### How an RCT validates CATE models

        The experiment directly validates the average effect and provides unbiased,
        noisy effect estimates for model-ranked groups. Qini/AUUC test whether higher
        predicted effects correspond to greater randomized lift. Calibration RMSE
        compares predicted versus observed uplift across ten equal-sized groups.
        None of these recovers an individual's unobserved counterfactual.
        """
    ),
    md("## Data"),
    code(common_setup),
    code(
        """
        from causal_benchmark.comparison import (
            load_benchmark_artifacts,
            paired_family_comparison,
        )
        from causal_benchmark.plotting import set_plot_style

        set_plot_style()
        try:
            metrics, causal_predictions, econ_predictions = load_benchmark_artifacts(
                ARTIFACT_DIR
            )
        except FileNotFoundError:
            from causal_benchmark.causalml_runner import run_causalml_rct_benchmark
            from causal_benchmark.econml_runner import run_econml_rct_benchmark
            from causal_benchmark.hillstrom import load_or_create_hillstrom_contrast

            data = load_or_create_hillstrom_contrast(RAW_PATH, PROCESSED_PATH)
            run_causalml_rct_benchmark(data, artifact_dir=ARTIFACT_DIR)
            run_econml_rct_benchmark(data, artifact_dir=ARTIFACT_DIR)
            metrics, causal_predictions, econ_predictions = load_benchmark_artifacts(
                ARTIFACT_DIR
            )

        benchmark = metrics.iloc[0]
        print(f"Paired randomized holdout rows: {len(causal_predictions):,}")
        print(
            f"RCT visit uplift: {benchmark['rct_ate']:.4f} "
            f"(95% CI {benchmark['rct_ci_lower']:.4f} to {benchmark['rct_ci_upper']:.4f})"
        )
        """
    ),
    md("## Results\n\n### 1. Rank all ten implementations by RCT-based Qini"),
    code(
        """
        overall = metrics.sort_values("qini_score", ascending=False).reset_index(drop=True)
        display(overall[[
            "library", "model", "qini_score", "auuc_score", "calibration_rmse",
            "ate_abs_error", "ate_within_rct_95ci", "fit_seconds",
        ]].style.format(precision=4))
        """
    ),
    code(
        """
        from causal_benchmark.plotting import plot_rct_metric

        chart_data = overall.copy()
        chart_data["model"] = chart_data["library"] + " · " + chart_data["model"]
        plot_rct_metric(
            chart_data,
            "qini_score",
            title="Cross-library normalized Qini on the randomized holdout",
            x_label="Normalized Qini (higher is better)",
        )
        plt.show()
        """
    ),
    md("### 2. Compare shared learner families directly"),
    code(
        """
        family_table = paired_family_comparison(metrics, metric="qini_score")
        display(family_table.style.format({
            "CausalML": "{:.4f}",
            "EconML": "{:.4f}",
            "absolute_qini_score_gap": "{:.4f}",
        }))
        """
    ),
    md("### 3. Check all model-implied average effects against the RCT"),
    code(
        """
        from causal_benchmark.plotting import plot_ate_against_rct

        plot_ate_against_rct(
            metrics,
            title="Ten model-implied ATEs against one experimental benchmark",
        )
        plt.show()
        """
    ),
    md("### 4. Compare targeting, calibration, and average-effect leaders"),
    code(
        """
        leader_rows = []
        for criterion, metric_name, ascending in [
            ("Targeting rank", "qini_score", False),
            ("Group calibration", "calibration_rmse", True),
            ("Average effect", "ate_abs_error", True),
        ]:
            row = metrics.sort_values(metric_name, ascending=ascending).iloc[0]
            leader_rows.append({
                "criterion": criterion,
                "library": row["library"],
                "model": row["model"],
                "qini_score": row["qini_score"],
                "calibration_rmse": row["calibration_rmse"],
                "ate_abs_error": row["ate_abs_error"],
            })
        display(pd.DataFrame(leader_rows).style.format(precision=4))
        """
    ),
    md("### 5. Evaluate deployable policies without fabricating economics"),
    code(
        """
        policy = metrics[[
            "library", "model", "policy_email_rate",
            "policy_incremental_visits_per_1000",
            "incremental_visits_per_1000_emailed", "top_30pct_rct_uplift",
        ]].sort_values("policy_incremental_visits_per_1000", ascending=False)
        display(policy.style.format(precision=3))
        """
    ),
    md("## Takeaways"),
    code(
        """
        qini_winner = metrics.sort_values("qini_score", ascending=False).iloc[0]
        calibration_winner = metrics.sort_values("calibration_rmse").iloc[0]
        ate_winner = metrics.sort_values("ate_abs_error").iloc[0]
        inside = int(metrics["ate_within_rct_95ci"].sum())
        print(
            f"Qini leader: {qini_winner['library']} {qini_winner['model']} "
            f"({qini_winner['qini_score']:.4f})."
        )
        print(
            f"Calibration leader: {calibration_winner['library']} "
            f"{calibration_winner['model']} "
            f"({calibration_winner['calibration_rmse']:.4f})."
        )
        print(
            f"Closest ATE: {ate_winner['library']} {ate_winner['model']} "
            f"(error={ate_winner['ate_abs_error']:.4f})."
        )
        print(f"ATE estimates inside the RCT 95% CI: {inside}/{len(metrics)}.")
        """
    ),
    md(
        """
        The cross-library conclusion is deliberately nuanced. Simple T-Learners
        rank best here; CausalForestDML calibrates ranked groups best; X-Learner is
        closest on the campaign average. Because nearly every predicted effect is
        positive, targeting saves few emails. A future business experiment would
        need campaign cost and visit/customer value to make an ROI claim.
        """
    ),
]


outputs = {
    "01_causalml_benchmark.ipynb": notebook(causalml_cells, "CausalML Hillstrom RCT"),
    "02_econml_benchmark.ipynb": notebook(econml_cells, "EconML Hillstrom RCT"),
    "03_cross_library_comparison.ipynb": notebook(
        comparison_cells, "CausalML versus EconML on Hillstrom"
    ),
}

for filename, book in outputs.items():
    nbf.write(book, NOTEBOOK_DIR / filename)
    print(f"Wrote {NOTEBOOK_DIR / filename}")

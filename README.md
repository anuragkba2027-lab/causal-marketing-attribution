# Causal marketing attribution: EconML vs CausalML

A complete, reproducible benchmark of heterogeneous treatment-effect models on the
**Hillstrom randomized email experiment**. The repository compares estimators within
Uber CausalML, within Microsoft/PyWhy EconML, and across both packages on the same
untouched RCT holdout.

> The correct package name is **EconML**, not “EcoML.” The project needs no API key,
> paid data, Gemini API, or cloud service.

## What the project answers

1. How do S-, T-, X-, R-, DR-, and DML-family estimators differ?
2. Do their campaign-average predictions agree with a real randomized experiment?
3. Which estimators rank customers best by experimentally observed uplift?
4. Are the conclusions consistent across CausalML and EconML?

## Why the Hillstrom dataset

The [original MineThatData challenge](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)
randomly assigned 64,000 customers approximately equally to Men's Email, Women's
Email, or No Email. It records pre-campaign shopping history and whether the customer
visited, converted, and spent money during the following two weeks.

The primary analysis pre-specifies **Men's Email vs No Email** and uses **visit** as
the outcome. This yields 42,613 customers. Women's Email is excluded—not collapsed
into treatment—because it is a different intervention. Visit is preferred over
conversion for CATE validation because conversion has only 578 positive events in
the full experiment.

Randomization gives credible ground truth for the average treatment effect and
unbiased uplift estimates for model-ranked groups. It does **not** reveal both
potential outcomes for an individual, so the real-data notebooks do not claim
individual PEHE or exact customer-level ground truth.

## Model matrix

| Notebook | Package | Estimators |
|---|---|---|
| `01_causalml_benchmark.ipynb` | Uber CausalML 0.16 | S, T, X, R, DR |
| `02_econml_benchmark.ipynb` | Microsoft/PyWhy EconML 0.16 | S, T, X, DR, CausalForestDML |
| `03_cross_library_comparison.ipynb` | Both | paired shared-family and overall comparison |

S/T/X/DR are compared directly across packages. R-Learner is retained as a
CausalML-specific orthogonal estimator; `CausalForestDML` is EconML's nonlinear
double/debiased machine-learning representative.

## Verified default-seed results

The shared 30% holdout contains 12,784 customers—6,392 per arm. Its randomized
visit-rate difference is **0.0735** (95% CI **0.0613 to 0.0857**).

| Validation objective | Best implementation | Result |
|---|---|---:|
| Customer ranking | CausalML T-Learner | normalized Qini 0.0266 |
| Group calibration | EconML CausalForestDML | decile RMSE 0.0257 |
| Campaign-average agreement | CausalML X-Learner | ATE absolute error 0.0028 |

All 10 model-implied ATEs fall inside the holdout RCT's 95% confidence interval.
The best-ranked CausalML T-Learner's top 30% has an observed randomized visit uplift
of about 9.49 percentage points, versus 7.35 points overall. However, Qini scores
are small and most learned policies email almost everyone, so the evidence for
strong persuadable segmentation is modest.

## Data and preprocessing

- Raw source: `data/raw/hillstrom_email_rct.csv` (64,000 × 12).
- SHA-256: `0E5893329D8B93CEFECC571777672028290AB69865718020C78C7284F291AECE`.
- No missing values; all randomized rows are retained.
- A stable row-based ID is added because the source has no customer ID.
- Seed 42 creates one treatment-stratified 70/30 train/holdout split.
- Numeric features: `recency`, `history`, `mens`, `womens`, `newbie`.
- One-hot features: `zip_code`, `channel`.
- `history_segment` is excluded because it deterministically bins `history`.
- Treatment, campaign label, outcomes, ID, and split are excluded from X.
- Preprocessing is fitted on training rows only.

The maximum absolute standardized mean difference across encoded baseline features
is 0.0137, consistent with successful randomization. See
[`docs/INTERVIEW_GUIDE.md`](docs/INTERVIEW_GUIDE.md) for a concise interview-ready
data explanation.

## Repository layout

```text
Causal ML/
├── artifacts/                         # persisted RCT predictions and metrics
├── data/
│   ├── raw/hillstrom_email_rct.csv
│   └── processed/hillstrom_mens_email_vs_control.csv
├── docs/
│   ├── INTERVIEW_GUIDE.md
│   └── METHODOLOGY.md
├── notebooks/
│   ├── 01_causalml_benchmark.ipynb
│   ├── 02_econml_benchmark.ipynb
│   └── 03_cross_library_comparison.ipynb
├── scripts/
│   ├── build_notebooks.py
│   └── download_hillstrom.py
├── src/causal_benchmark/
│   ├── hillstrom.py
│   ├── rct_evaluation.py
│   ├── causalml_runner.py
│   ├── econml_runner.py
│   └── comparison.py
├── tests/
└── CV_POINTERS.md
```

The previous synthetic coupon dataset remains available as an optional exact-CATE
stress test, but it is not the primary benchmark.

## Setup

Both packages run together on **Python 3.11 + scikit-learn 1.6.1**.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m ipykernel install --user --name causal-ml --display-name "Python 3 (causal-ml)"
```

If the raw CSV is not present:

```powershell
.\.venv\Scripts\python.exe scripts\download_hillstrom.py
```

Build and execute the notebooks:

```powershell
.\.venv\Scripts\python.exe scripts\build_notebooks.py
.\.venv\Scripts\python.exe -m nbconvert --execute --to notebook --inplace notebooks\01_causalml_benchmark.ipynb --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=causal-ml
.\.venv\Scripts\python.exe -m nbconvert --execute --to notebook --inplace notebooks\02_econml_benchmark.ipynb --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=causal-ml
.\.venv\Scripts\python.exe -m nbconvert --execute --to notebook --inplace notebooks\03_cross_library_comparison.ipynb --ExecutePreprocessor.timeout=600 --ExecutePreprocessor.kernel_name=causal-ml
```

Run QA:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests scripts
```

## Evaluation metrics

- **RCT ATE agreement:** mean predicted CATE versus the randomized difference in means.
- **Normalized Qini / AUUC:** ranking quality using observed outcomes and assignment.
- **Uplift calibration RMSE:** predicted versus randomized uplift across ten bins.
- **Top-30% RCT uplift:** experimental lift in the highest-scored customers.
- **IPW policy value:** estimated incremental visits per 1,000 eligible customers.
- **Fit time:** local descriptive runtime, not a portable speed claim.

No monetary ROI is calculated because the data has no email cost, visit value, or
real deployment budget. [`CV_POINTERS.md`](CV_POINTERS.md) separates verified claims
from unsupported ones.

## References

- [Hillstrom experiment description and data dictionary](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)
- [EconML documentation](https://www.pywhy.org/EconML/)
- [Uber CausalML documentation](https://causalml.readthedocs.io/en/stable/)
- Künzel et al., *Metalearners for Estimating Heterogeneous Treatment Effects* (2019)
- Chernozhukov et al., *Double/Debiased Machine Learning for Treatment and Structural Parameters* (2018)

## License

Project code is MIT; see [`LICENSE`](LICENSE). The source page does not state a
formal software-style license for the dataset, so verify redistribution requirements
before republishing the raw CSV elsewhere.

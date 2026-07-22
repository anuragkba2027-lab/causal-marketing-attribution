# Methodology and validation contract

## Causal question

Among customers eligible for the Hillstrom email experiment, how much does sending
a **Men's Email**, rather than **No Email**, change the probability of visiting the
website within two weeks, and can models identify groups with different uplift?

The estimand is the conditional average treatment effect:

`CATE(x) = E[visit(1) - visit(0) | X=x]`.

## Experimental design

The original dataset has 64,000 customers randomized approximately equally among
Men's Email, Women's Email, and No Email. The primary binary contrast retains 21,307
Men's Email customers and 21,306 No Email customers. Women's Email is a separate
intervention and is not pooled with Men's Email.

Randomization supports:

- consistency/SUTVA for the defined campaign contrast;
- exchangeability by design;
- positivity through approximately 1:1 conditional assignment in the retained arms.

A seeded, treatment-stratified 70/30 split produces 29,829 training and 12,784
holdout observations. The holdout has 6,392 customers per arm.

## Preprocessing

Only pre-treatment features are used:

| Type | Variables | Handling |
|---|---|---|
| Numeric/binary | `recency`, `history`, `mens`, `womens`, `newbie` | passthrough |
| Categorical | `zip_code`, `channel` | one-hot encoding, unknowns ignored |
| Redundant | `history_segment` | excluded; deterministic bin of `history` |
| Forbidden | `segment`, treatment, outcomes, ID, split | excluded from X |

The column transformer is fitted on training data only. The raw source's 6,562
exact repeated rows are retained: no customer ID is supplied, and repeated coarse
attribute/outcome combinations do not establish duplicate experimental units.

## Estimators

| Library | Estimators | Role |
|---|---|---|
| CausalML 0.16 | S, T, X, R, DR | marketing/uplift meta-learners and orthogonal learners |
| EconML 0.16 | S, T, X, DR, CausalForestDML | shared meta-learners plus nonlinear DML |

Random forests use matched capacity where APIs allow it. CausalML orthogonal
estimators receive the known 0.5 pairwise treatment probability. EconML DR and DML
estimators use a fold-specific constant-prior treatment model, reflecting randomized
assignment rather than learned selection.

## What is validatable

The holdout randomized difference in visit rates is 0.0735 with a normal-approximate
95% CI of 0.0613–0.0857. It is used as the campaign-average reference.

Individual `visit(1) - visit(0)` is never observed. Therefore:

- no real-data PEHE or individual CATE MAE is claimed;
- average predicted CATE is compared with the experimental ATE and CI;
- Qini/AUUC assess ordering through randomized outcomes;
- calibration bins compare mean predicted CATE with experimental uplift;
- top-ranked group uplift is estimated by treatment-control differences;
- learned policy value uses inverse-probability weighting.

Calibration and subgroup effects remain noisy because each bin is a smaller
experiment. Their confidence intervals and modest Qini values must be interpreted
alongside point estimates.

## Data-quality and leakage checks

1. The raw CSV must match its 64,000 × 12 schema and stored SHA-256 checksum.
2. Required fields have no missing values and binary fields contain only 0/1.
3. All three treatment arms have the documented labels.
4. The maximum encoded-feature absolute standardized mean difference is below 0.05
   (observed: 0.0137).
5. Train preprocessing is never fitted on the holdout.
6. Both libraries' prediction files must contain identical customer IDs, treatments,
   and outcomes.
7. Tests validate source integrity, split counts, feature exclusion, metric shapes,
   and finite outputs.

## Limitations

- One campaign and one period do not establish transportability to another business.
- The experiment validates group effects, not each customer's counterfactual.
- Qini and calibration comparisons can change with sampling noise or hyperparameters.
- `visit` is a proxy for business value; the project does not observe profit.
- Email cost and customer value are absent, so an ROI improvement cannot be inferred.
- Normal-approximation intervals are adequate at this sample size but are not the
  only possible inferential choice.

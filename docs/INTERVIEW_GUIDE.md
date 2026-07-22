# Interview guide: dataset and preprocessing

## 60-second dataset explanation

“I used the Hillstrom email dataset because it comes from a randomized marketing
experiment, which gives a credible benchmark for causal validation. It contains
64,000 customers randomized roughly one-third each to a Men's Email, a Women's
Email, or No Email. The pre-treatment variables describe recency, historical spend,
prior men's and women's purchases, whether the customer is new, location type, and
purchase channel. The post-treatment outcomes are visit, conversion, and spend over
the next two weeks.

I pre-specified Men's Email versus No Email, giving 42,613 customers, and used visit
as the primary outcome because conversion is very sparse. I kept 70% for training
and an untouched 30% randomized holdout for validation. The holdout experiment shows
a 7.35-percentage-point visit lift, with a 95% interval from 6.13 to 8.57 points.”

## 60-second preprocessing explanation

“The source has no missing values. I preserved every randomized row—even repeated
attribute combinations—because there is no customer ID proving they are duplicates.
I added a stable row-based ID and encoded Men's Email as treatment 1 and No Email as
0. I excluded the Women's arm rather than merging it, because it is a different
intervention.

For modeling, I passed through recency, historical spend, prior men's and women's
purchases, and new-customer status. I one-hot encoded zip type and channel. I dropped
`history_segment` because it duplicates the information in numeric history, and I
excluded treatment, outcomes, campaign labels, IDs, and split columns to prevent
leakage. The preprocessor is fitted only on training rows. Finally, I checked
randomization balance: the largest absolute standardized mean difference was 0.0137.”

## Why `visit`, not `conversion` or `spend`?

- `visit` is binary, directly related to email engagement, and has enough events for
  subgroup validation.
- `conversion` has only 578 positives in all 64,000 rows, making decile effects noisy.
- `spend` is highly zero-inflated and skewed; it is useful as a secondary outcome but
  requires more careful distributional modeling.

## How the RCT validates the models

The untouched holdout provides four progressively harder checks:

1. **ATE agreement:** Does mean predicted CATE match the randomized visit difference?
2. **Calibration:** Within predicted-uplift deciles, do predictions match randomized lift?
3. **Ranking:** Do Qini/AUUC improve when customers are ordered by predicted CATE?
4. **Policy:** How many incremental visits does the learned email rule create per
   1,000 eligible customers using inverse-probability weighting?

The important caveat is that an RCT still observes only one outcome per customer.
It validates averages and ranked groups, but not exact individual treatment effects.

## Results to remember

- RCT holdout ATE: **0.0735**; 95% CI: **[0.0613, 0.0857]**.
- All 10 model ATE estimates are within the experimental interval.
- CausalML T-Learner: best Qini, **0.0266**.
- EconML CausalForestDML: best calibration RMSE, **0.0257**.
- CausalML X-Learner: smallest ATE error, **0.0028**.
- CausalML T-Learner top 30% observed uplift: **0.0949**.

## Questions an interviewer may ask

**Why did a simple T-Learner beat DML on Qini?**

DML targets bias reduction through orthogonal nuisance estimation; it does not
guarantee the best finite-sample ranking. Here treatment was already randomized and
the main effect was broadly positive, so the complexity advantage was limited.

**Why not train and validate on the full experiment?**

That would reuse outcomes for both fitting and model selection. A fixed holdout gives
a cleaner external check of average effects, calibration, and ranking.

**Did you correct selection bias?**

The primary Hillstrom analysis does not need observational selection-bias correction
because treatment was randomized. R/DR/DML still residualize nuisance components,
but the honest CV statement is “validated on an RCT,” not “corrected historical
selection bias.”

**Can you claim persuadable customers?**

Only cautiously. The best model's top-ranked group has larger observed uplift, but
Qini is modest and subgroup intervals are wide. A targeted follow-up RCT would be
needed before a strong production claim.

**Can you claim an ROI improvement?**

No. The dataset has no email-delivery cost, visit value, or deployed budget. The
repository reports incremental visits rather than invented dollars.

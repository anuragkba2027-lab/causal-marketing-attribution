# CV pointer evidence ledger

Use only claims marked **Verified**. Refresh the figures after changing the dataset,
split, estimators, seed, features, or evaluation code.

| Status | Candidate CV pointer | Evidence |
|---|---|---|
| Verified | Built an end-to-end causal marketing benchmark across 10 CATE implementations in Uber CausalML and Microsoft/PyWhy EconML using a 64K-customer randomized email experiment. | Three executed notebooks; Hillstrom source checksum; metric artifacts |
| Verified | Designed a leakage-safe Men's Email vs No Email analysis on 42,613 customers with train-only one-hot preprocessing and a shared 12,784-customer randomized holdout. | `hillstrom.py`; notebook data sections; tests |
| Verified | Validated model-implied campaign lift against an experimental visit uplift of 7.35 pp (95% CI 6.13–8.57 pp); all 10 model ATE estimates fell inside the RCT interval. | Both metric CSVs; comparison notebook |
| Verified | Benchmarked S/T/X/DR across both packages plus CausalML R-Learner and EconML CausalForestDML using Qini, AUUC, randomized uplift calibration, ATE error, and IPW policy value. | Comparison notebook; shared RCT evaluator |
| Verified | Achieved the best holdout uplift ranking with CausalML T-Learner (normalized Qini 0.0266); its top-scored 30% showed 9.49 pp randomized visit lift. | `artifacts/causalml_metrics.csv` |
| Verified | Achieved the lowest decile uplift-calibration RMSE (0.0257) with EconML CausalForestDML and the lowest ATE error (0.0028) with CausalML X-Learner. | Cross-library artifact comparison |
| Verified | Engineered a reproducible Python 3.11 repository with source checksum validation, common experiment splits, deterministic preprocessing, tests, and three executable notebooks. | Repository and QA output |
| Unsupported—do not use | “Corrected selection bias in historical coupon assignment.” | Primary data is randomized; there is no observational assignment bias to correct. |
| Unsupported—do not use | “Identified true individual persuadables.” | The RCT does not reveal both potential outcomes for one customer. |
| Unsupported—do not use | “Increased marketing ROI by 18%.” | No deployment, treatment cost, visit value, finance baseline, or realized-spend evidence. |

## Concise résumé-ready options

1. **Experiment validation:** Benchmarked 10 heterogeneous-treatment-effect models
   across EconML and CausalML on a 64K-customer randomized email experiment; all
   model ATEs matched the holdout RCT's 95% confidence interval.
2. **Targeting:** Built an RCT-validated uplift pipeline on 42.6K customers, with a
   CausalML T-Learner achieving 0.0266 normalized Qini and 9.49 pp visit lift in the
   top-scored 30%.
3. **DML comparison:** Compared meta-learners, doubly robust estimation, and
   CausalForestDML under identical splits and features; CausalForestDML delivered the
   best randomized decile-calibration RMSE (0.0257).
4. **Engineering:** Developed a reproducible Python causal-inference repository with
   dataset checksum validation, leakage-safe preprocessing, cross-library artifact
   checks, automated tests, and three executable notebooks.

## Suggested combined two-bullet version

- Benchmarked 10 CATE implementations across EconML and CausalML on a 64K-customer
  randomized email experiment, validating every model-implied ATE against the RCT's
  7.35 pp visit uplift (95% CI: 6.13–8.57 pp).
- Built a leakage-safe uplift evaluation pipeline using Qini, AUUC, randomized
  calibration, and IPW policy value; achieved 0.0266 Qini with a CausalML T-Learner
  and 0.0257 calibration RMSE with EconML CausalForestDML.

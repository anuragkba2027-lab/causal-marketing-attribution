# Data provenance

## Hillstrom randomized email experiment

- **Creator:** Kevin Hillstrom, MineThatData
- **Original challenge:** https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html
- **Official CSV:** http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv
- **Downloaded file:** `raw/hillstrom_email_rct.csv`
- **Rows / source columns:** 64,000 / 12
- **SHA-256:** `0E5893329D8B93CEFECC571777672028290AB69865718020C78C7284F291AECE`

The raw file is preserved unchanged. `processed/hillstrom_mens_email_vs_control.csv`
is deterministically produced by selecting Men's Email and No Email, adding a
row-based customer ID, encoding treatment, and assigning the shared 70/30 split.

The creator publicly released the data for an analytics challenge but the source
page does not state a formal software-style data license. Verify redistribution
requirements before publishing the raw CSV outside this repository.

## Synthetic coupon benchmark

`processed/coupon_retention_benchmark.csv` was generated locally by
`src/causal_benchmark/data.py`. It is retained only as an optional direct-CATE
stress test; the Hillstrom RCT is now the primary project dataset.

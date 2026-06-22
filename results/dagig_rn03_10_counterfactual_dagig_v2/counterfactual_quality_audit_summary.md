# Counterfactual Quality Audit

Rows marked `overall_cf_quality=fail` should be excluded from DAG-IG v2 scoring/training.

| split | n | pass | fail | fail_rate |
| --- | --- | --- | --- | --- |
| train | 458 | 327 | 131 | 0.28602620087336245 |
| dev | 98 | 74 | 24 | 0.24489795918367346 |
| test | 64 | 47 | 17 | 0.265625 |

Failed sample ids: 172

# DAG-IG v3 Score Predictiveness

| split | comparison | score_key | target_key | n | spearman | spearman_ci_low | spearman_ci_high | pearson | auc | auc_ci_low | auc_ci_high | top_bottom_gap | score_mean | target_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | R_search -> R@5 | R_search_v3 | retrieval_r5 | 1796 | 0.7632 | 0.7433 | 0.7808 | 0.7541 | 0.9473 | 0.9357 | 0.9560 | 0.8530 | 0.3139 | 0.6169 |
| train | R_search -> MRR | R_search_v3 | retrieval_mrr | 1796 | 0.7362 | 0.7150 | 0.7548 | 0.7506 | None | None | None | 0.7417 | 0.3139 | 0.5070 |
| train | DAGIG_total -> evidence_support | DAGIG_v3_total | evidence_support | 1832 | 0.4950 | 0.4575 | 0.5272 | 0.4683 | 0.8079 | 0.7855 | 0.8254 | 0.5590 | 0.3181 | 0.6861 |
| train | DAGIG_total -> supported_answer | DAGIG_v3_total | supported_answer | 1832 | 0.0658 | 0.0226 | 0.1090 | 0.0831 | 0.5633 | 0.5208 | 0.6048 | 0.0568 | 0.3181 | 0.0999 |
| dev | R_search -> R@5 | R_search_v3 | retrieval_r5 | 383 | 0.7467 | 0.7028 | 0.7871 | 0.7371 | 0.9401 | 0.9124 | 0.9612 | 0.8316 | 0.3445 | 0.6240 |
| dev | R_search -> MRR | R_search_v3 | retrieval_mrr | 383 | 0.7426 | 0.6981 | 0.7805 | 0.7572 | None | None | None | 0.7749 | 0.3445 | 0.5196 |
| dev | DAGIG_total -> evidence_support | DAGIG_v3_total | evidence_support | 392 | 0.4769 | 0.3841 | 0.5450 | 0.4523 | 0.7980 | 0.7482 | 0.8419 | 0.5306 | 0.3582 | 0.6913 |
| dev | DAGIG_total -> supported_answer | DAGIG_v3_total | supported_answer | 392 | 0.0954 | -0.0181 | 0.1925 | 0.0946 | 0.5910 | 0.4970 | 0.6777 | 0.0816 | 0.3582 | 0.1020 |
| test | R_search -> R@5 | R_search_v3 | retrieval_r5 | 251 | 0.7478 | 0.6658 | 0.8041 | 0.7546 | 0.9303 | 0.8956 | 0.9604 | 0.8387 | 0.2661 | 0.5737 |
| test | R_search -> MRR | R_search_v3 | retrieval_mrr | 251 | 0.7280 | 0.6811 | 0.7682 | 0.7506 | None | None | None | 0.6587 | 0.2661 | 0.4760 |
| test | DAGIG_total -> evidence_support | DAGIG_v3_total | evidence_support | 256 | 0.5177 | 0.4054 | 0.5943 | 0.5002 | 0.8147 | 0.7547 | 0.8651 | 0.5938 | 0.1946 | 0.6562 |
| test | DAGIG_total -> supported_answer | DAGIG_v3_total | supported_answer | 256 | 0.0078 | -0.0943 | 0.1185 | 0.0134 | 0.5082 | 0.3964 | 0.6098 | 0.0156 | 0.1946 | 0.0820 |

## Training Gate

| split | score_key | target_key | metric | value | threshold | passed |
| --- | --- | --- | --- | --- | --- | --- |
| dev | R_search_v3 | retrieval_r5 | spearman | 0.7467 | 0.2000 | True |
| dev | DAGIG_v3_total | evidence_support | spearman | 0.4769 | 0.2000 | True |
| dev | DAGIG_v3_total | supported_answer | auc | 0.5910 | 0.6000 | False |

Gate passed: False

# DAG-IG v3.1 Score Predictiveness

| split | comparison | score_key | target_key | n | spearman | spearman_ci_low | spearman_ci_high | pearson | auc | auc_ci_low | auc_ci_high | top_bottom_gap | score_mean | target_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | R_search_v3 -> R@5 | R_search_v3 | retrieval_r5 | 1796 | 0.7632 | 0.7433 | 0.7808 | 0.7541 | 0.9473 | 0.9357 | 0.9560 | 0.8530 | 0.3139 | 0.6169 |
| train | R_search_v3 -> MRR | R_search_v3 | retrieval_mrr | 1796 | 0.7362 | 0.7150 | 0.7548 | 0.7506 | None | None | None | 0.7417 | 0.3139 | 0.5070 |
| train | R_evidence -> evidence_support | R_evidence | evidence_support | 1832 | 0.2461 | 0.2043 | 0.2864 | 0.2928 | 0.6252 | 0.6025 | 0.6495 | 0.2402 | -0.0427 | 0.6861 |
| train | R_answer_v31 -> supported_answer_v2 | R_answer_v31 | supported_answer_v2 | 1832 | 0.3801 | 0.1843 | 0.5208 | 0.4195 | 0.6299 | 0.5555 | 0.7152 | 0.0131 | 0.0016 | 0.0213 |
| train | DAGIG_v31_total -> evidence_support | DAGIG_v31_total | evidence_support | 1832 | 0.5414 | 0.5095 | 0.5701 | 0.5002 | 0.8368 | 0.8195 | 0.8540 | 0.5721 | 0.0590 | 0.6861 |
| train | DAGIG_v31_total -> supported_answer_v2 | DAGIG_v31_total | supported_answer_v2 | 1832 | -0.0487 | -0.1028 | 0.0048 | -0.0909 | 0.4026 | 0.2861 | 0.5147 | -0.0153 | 0.0590 | 0.0213 |
| train | DAGIG_v31_total -> answer F1 | DAGIG_v31_total | answer_f1_v2 | 1832 | 0.0837 | 0.0392 | 0.1203 | 0.0032 | None | None | None | 0.0157 | 0.0590 | 0.1298 |
| train | search_evidence_only -> evidence_support | DAGIG_v31_search_evidence_only | evidence_support | 1832 | 0.6103 | 0.5833 | 0.6349 | 0.5390 | 0.8773 | 0.8604 | 0.8903 | 0.6026 | 0.1002 | 0.6861 |
| train | no_answer -> R@5 | DAGIG_v31_no_answer | retrieval_r5 | 1796 | 0.6236 | 0.5935 | 0.6515 | 0.5728 | 0.8703 | 0.8555 | 0.8849 | 0.7127 | 0.0870 | 0.6169 |
| train | no_answer -> evidence_support | DAGIG_v31_no_answer | evidence_support | 1832 | 0.5386 | 0.5085 | 0.5659 | 0.5049 | 0.8351 | 0.8190 | 0.8519 | 0.5786 | 0.0772 | 0.6861 |
| dev | R_search_v3 -> R@5 | R_search_v3 | retrieval_r5 | 383 | 0.7467 | 0.6925 | 0.7858 | 0.7371 | 0.9401 | 0.9087 | 0.9635 | 0.8316 | 0.3445 | 0.6240 |
| dev | R_search_v3 -> MRR | R_search_v3 | retrieval_mrr | 383 | 0.7426 | 0.7035 | 0.7789 | 0.7572 | None | None | None | 0.7749 | 0.3445 | 0.5196 |
| dev | R_evidence -> evidence_support | R_evidence | evidence_support | 392 | 0.3038 | 0.1924 | 0.3841 | 0.3613 | 0.6570 | 0.6127 | 0.7038 | 0.2959 | -0.0144 | 0.6913 |
| dev | R_answer_v31 -> supported_answer_v2 | R_answer_v31 | supported_answer_v2 | 392 | 0.3170 | 0.0063 | 0.6204 | 0.4006 | 0.6710 | 0.5051 | 0.8763 | 0.0306 | -0.0023 | 0.0153 |
| dev | DAGIG_v31_total -> evidence_support | DAGIG_v31_total | evidence_support | 392 | 0.5171 | 0.4381 | 0.5739 | 0.4987 | 0.8231 | 0.7758 | 0.8627 | 0.5612 | 0.0697 | 0.6913 |
| dev | DAGIG_v31_total -> supported_answer_v2 | DAGIG_v31_total | supported_answer_v2 | 392 | -0.0013 | -0.1271 | 0.1140 | -0.0584 | 0.4970 | 0.0179 | 0.7965 | 0.0000 | 0.0697 | 0.0153 |
| dev | DAGIG_v31_total -> answer F1 | DAGIG_v31_total | answer_f1_v2 | 392 | 0.0667 | -0.0477 | 0.1565 | -0.0359 | None | None | None | 0.0102 | 0.0697 | 0.1550 |
| dev | search_evidence_only -> evidence_support | DAGIG_v31_search_evidence_only | evidence_support | 392 | 0.6122 | 0.5489 | 0.6634 | 0.5752 | 0.8806 | 0.8433 | 0.9103 | 0.6531 | 0.1286 | 0.6913 |
| dev | no_answer -> R@5 | DAGIG_v31_no_answer | retrieval_r5 | 383 | 0.5794 | 0.5107 | 0.6404 | 0.5436 | 0.8453 | 0.8042 | 0.8787 | 0.6632 | 0.1015 | 0.6240 |
| dev | no_answer -> evidence_support | DAGIG_v31_no_answer | evidence_support | 392 | 0.5145 | 0.4218 | 0.5884 | 0.4990 | 0.8215 | 0.7788 | 0.8600 | 0.5714 | 0.0894 | 0.6913 |
| test | R_search_v3 -> R@5 | R_search_v3 | retrieval_r5 | 251 | 0.7478 | 0.6788 | 0.8049 | 0.7546 | 0.9303 | 0.8907 | 0.9640 | 0.8387 | 0.2661 | 0.5737 |
| test | R_search_v3 -> MRR | R_search_v3 | retrieval_mrr | 251 | 0.7280 | 0.6806 | 0.7611 | 0.7506 | None | None | None | 0.6587 | 0.2661 | 0.4760 |
| test | R_evidence -> evidence_support | R_evidence | evidence_support | 256 | 0.4072 | 0.3112 | 0.5135 | 0.4031 | 0.7050 | 0.6394 | 0.7652 | 0.4531 | -0.0878 | 0.6562 |
| test | R_answer_v31 -> supported_answer_v2 | R_answer_v31 | supported_answer_v2 | 256 | 0.4107 | 0.0039 | 1.0000 | 0.4107 | 0.6680 | 0.5000 | 1.0000 | 0.0156 | 0.0000 | 0.0117 |
| test | DAGIG_v31_total -> evidence_support | DAGIG_v31_total | evidence_support | 256 | 0.5615 | 0.4904 | 0.6364 | 0.5399 | 0.8412 | 0.7921 | 0.8811 | 0.6562 | 0.0354 | 0.6562 |
| test | DAGIG_v31_total -> supported_answer_v2 | DAGIG_v31_total | supported_answer_v2 | 256 | -0.1397 | -0.2194 | -0.0555 | -0.1258 | 0.1252 | 0.0476 | 0.2441 | -0.0469 | 0.0354 | 0.0117 |
| test | DAGIG_v31_total -> answer F1 | DAGIG_v31_total | answer_f1_v2 | 256 | 0.0106 | -0.1066 | 0.1571 | -0.0863 | None | None | None | -0.0335 | 0.0354 | 0.1389 |
| test | search_evidence_only -> evidence_support | DAGIG_v31_search_evidence_only | evidence_support | 256 | 0.6492 | 0.5680 | 0.7043 | 0.6115 | 0.8927 | 0.8498 | 0.9341 | 0.7031 | 0.0578 | 0.6562 |
| test | no_answer -> R@5 | DAGIG_v31_no_answer | retrieval_r5 | 251 | 0.6702 | 0.6032 | 0.7373 | 0.6335 | 0.8912 | 0.8494 | 0.9302 | 0.7419 | 0.0592 | 0.5737 |
| test | no_answer -> evidence_support | DAGIG_v31_no_answer | evidence_support | 256 | 0.5549 | 0.4687 | 0.6393 | 0.5392 | 0.8373 | 0.7851 | 0.8798 | 0.6250 | 0.0501 | 0.6562 |

## Training Gate

| split | score_key | target_key | metric | value | threshold | passed |
| --- | --- | --- | --- | --- | --- | --- |
| dev | R_search_v3 | retrieval_r5 | spearman | 0.7467 | 0.2000 | True |
| dev | R_evidence | evidence_support | spearman | 0.3038 | 0.3500 | False |
| dev | R_answer_v31 | supported_answer_v2 | auc | 0.6710 | 0.6000 | True |
| dev | DAGIG_v31_total | evidence_support | spearman | 0.5171 | 0.2000 | True |
| dev | DAGIG_v31_total | supported_answer_v2 | auc | 0.4970 | 0.6000 | False |

Gate passed: False

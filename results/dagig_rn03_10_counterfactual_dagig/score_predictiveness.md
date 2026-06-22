# Counterfactual DAG-IG Score Predictiveness

Higher Spearman/top-bottom/AUC means the edge score is more predictive of downstream success.

| split | comparison | score_key | target_key | n | spearman | top_bottom_gap | auc | score_mean | target_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | DAGIG_total vs R@5 | DAGIG_total | retrieval_r5 | 1796 | 0.2192 | 0.2829 | 0.6302 | 0.1039 | 0.6169 |
| train | DAGIG_total vs evidence_support | DAGIG_total | evidence_support | 1832 | 0.2056 | 0.2445 | 0.6279 | 0.0860 | 0.6861 |
| train | DAGIG_total vs answer F1 | DAGIG_total | answer_f1 | 1832 | 0.0596 | 0.0271 | None | 0.0860 | 0.1206 |
| train | R_ground vs IoU | R_ground | ground_iou | 1832 | 0.5514 | 0.2897 | None | -0.0699 | 0.2813 |
| train | R_ground vs center-hit | R_ground | ground_center_hit | 1832 | 0.5505 | 0.6943 | 0.8289 | -0.0699 | 0.6294 |
| train | R_search vs R@5 | R_search | retrieval_r5 | 1796 | 0.2368 | 0.2962 | 0.6267 | 0.0724 | 0.6169 |
| train | R_search vs MRR | R_search | retrieval_mrr | 1796 | 0.2178 | 0.2301 | None | 0.0724 | 0.5070 |
| train | R_evidence vs evidence_support | R_evidence | evidence_support | 1832 | 0.2461 | 0.2402 | 0.6252 | -0.0427 | 0.6861 |
| dev | DAGIG_total vs R@5 | DAGIG_total | retrieval_r5 | 383 | 0.1159 | 0.1263 | 0.5690 | 0.1193 | 0.6240 |
| dev | DAGIG_total vs evidence_support | DAGIG_total | evidence_support | 392 | 0.1303 | 0.0918 | 0.5814 | 0.0931 | 0.6913 |
| dev | DAGIG_total vs answer F1 | DAGIG_total | answer_f1 | 392 | 0.0366 | 0.0181 | None | 0.0931 | 0.1419 |
| dev | R_ground vs IoU | R_ground | ground_iou | 392 | 0.4904 | 0.2469 | None | -0.0733 | 0.2654 |
| dev | R_ground vs center-hit | R_ground | ground_center_hit | 392 | 0.3996 | 0.5612 | 0.7358 | -0.0733 | 0.6046 |
| dev | R_search vs R@5 | R_search | retrieval_r5 | 383 | 0.0311 | 0.0000 | 0.5164 | 0.0710 | 0.6240 |
| dev | R_search vs MRR | R_search | retrieval_mrr | 383 | 0.0222 | 0.0054 | None | 0.0710 | 0.5196 |
| dev | R_evidence vs evidence_support | R_evidence | evidence_support | 392 | 0.3038 | 0.2959 | 0.6570 | -0.0144 | 0.6913 |
| test | DAGIG_total vs R@5 | DAGIG_total | retrieval_r5 | 251 | 0.2309 | 0.2419 | 0.6348 | -0.0096 | 0.5737 |
| test | DAGIG_total vs evidence_support | DAGIG_total | evidence_support | 256 | 0.2055 | 0.1719 | 0.6249 | -0.0214 | 0.6562 |
| test | DAGIG_total vs answer F1 | DAGIG_total | answer_f1 | 256 | -0.0558 | -0.0671 | None | -0.0214 | 0.1362 |
| test | R_ground vs IoU | R_ground | ground_iou | 256 | 0.5219 | 0.2290 | None | -0.0744 | 0.3131 |
| test | R_ground vs center-hit | R_ground | ground_center_hit | 256 | 0.5224 | 0.6406 | 0.8088 | -0.0744 | 0.6094 |
| test | R_search vs R@5 | R_search | retrieval_r5 | 251 | 0.1815 | 0.3226 | 0.5942 | 0.0388 | 0.5737 |
| test | R_search vs MRR | R_search | retrieval_mrr | 251 | 0.1398 | 0.1868 | None | 0.0388 | 0.4760 |
| test | R_evidence vs evidence_support | R_evidence | evidence_support | 256 | 0.4072 | 0.4531 | 0.7050 | -0.0878 | 0.6562 |

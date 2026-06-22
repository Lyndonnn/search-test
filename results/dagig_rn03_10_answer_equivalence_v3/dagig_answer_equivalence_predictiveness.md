# DAG-IG With Answer Equivalence v3 Predictiveness

| split | comparison | score_key | target_key | n | spearman | pearson | auc | top_bottom_gap | score_mean | target_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | DAGIG_v31_total -> supported_answer_soft | DAGIG_v31_total | supported_answer_soft | 1832 | 0.0062 | -0.0383 | None | 0.0015 | 0.0590 | 0.0750 |
| train | DAGIG_v31_total -> supported_answer_hard_v3 | DAGIG_v31_total | supported_answer_hard | 1832 | -0.0517 | -0.0988 | 0.4055 | -0.0240 | 0.0590 | 0.0257 |
| train | search_evidence_only -> evidence_support | DAGIG_v31_search_evidence_only | evidence_support | 1832 | 0.6103 | 0.5390 | 0.8773 | 0.6026 | 0.1002 | 0.6861 |
| train | R_answer_v31 -> answer_correct_score | R_answer_v31 | answer_correct_score | 1832 | 0.0454 | 0.0951 | None | -0.0044 | 0.0016 | 0.0866 |
| train | R_answer_v31 -> supported_answer_soft | R_answer_v31 | supported_answer_soft | 1832 | 0.0770 | 0.1336 | None | -0.0073 | 0.0016 | 0.0750 |
| train | answer F1 -> answer_correct_score | answer_f1_v2 | answer_correct_score | 1832 | 0.1345 | 0.4702 | None | 0.1461 | 0.1298 | 0.0866 |
| dev | DAGIG_v31_total -> supported_answer_soft | DAGIG_v31_total | supported_answer_soft | 392 | -0.1257 | -0.1101 | None | -0.0431 | 0.0697 | 0.0728 |
| dev | DAGIG_v31_total -> supported_answer_hard_v3 | DAGIG_v31_total | supported_answer_hard | 392 | 0.0112 | -0.0197 | 0.5322 | 0.0000 | 0.0697 | 0.0102 |
| dev | search_evidence_only -> evidence_support | DAGIG_v31_search_evidence_only | evidence_support | 392 | 0.6122 | 0.5752 | 0.8806 | 0.6531 | 0.1286 | 0.6913 |
| dev | R_answer_v31 -> answer_correct_score | R_answer_v31 | answer_correct_score | 392 | -0.1263 | -0.1081 | None | -0.0806 | -0.0023 | 0.0934 |
| dev | R_answer_v31 -> supported_answer_soft | R_answer_v31 | supported_answer_soft | 392 | -0.1371 | -0.0590 | None | -0.0686 | -0.0023 | 0.0728 |
| dev | answer F1 -> answer_correct_score | answer_f1_v2 | answer_correct_score | 392 | 0.1656 | 0.3687 | None | 0.1158 | 0.1550 | 0.0934 |
| test | DAGIG_v31_total -> supported_answer_soft | DAGIG_v31_total | supported_answer_soft | 256 | 0.0032 | -0.0737 | None | -0.0368 | 0.0354 | 0.0587 |
| test | DAGIG_v31_total -> supported_answer_hard_v3 | DAGIG_v31_total | supported_answer_hard | 256 | -0.0742 | -0.0607 | 0.3274 | -0.0312 | 0.0354 | 0.0156 |
| test | search_evidence_only -> evidence_support | DAGIG_v31_search_evidence_only | evidence_support | 256 | 0.6492 | 0.6115 | 0.8927 | 0.7031 | 0.0578 | 0.6562 |
| test | R_answer_v31 -> answer_correct_score | R_answer_v31 | answer_correct_score | 256 | -0.1334 | -0.1963 | None | -0.1120 | 0.0000 | 0.0895 |
| test | R_answer_v31 -> supported_answer_soft | R_answer_v31 | supported_answer_soft | 256 | -0.1504 | -0.2350 | None | -0.0231 | 0.0000 | 0.0587 |
| test | answer F1 -> answer_correct_score | answer_f1_v2 | answer_correct_score | 256 | 0.3264 | 0.5348 | None | 0.2319 | 0.1389 | 0.0895 |

## Positive Rate By Answer Type

| split | answer_type | n | supported_answer_hard_v3_positive_rate | supported_answer_soft_mean | semantic_equivalent_rate | answer_correct_score_mean |
| --- | --- | --- | --- | --- | --- | --- |
| dev | address | 30 | 0.0000 | 0.1633 | 0.0000 | 0.2780 |
| dev | company_id | 75 | 0.0133 | 0.0133 | 0.0133 | 0.0133 |
| dev | count | 53 | 0.0189 | 0.0189 | 0.0189 | 0.0189 |
| dev | date | 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dev | email | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dev | entity_name | 33 | 0.0606 | 0.3417 | 0.0606 | 0.3911 |
| dev | other | 15 | 0.0000 | 0.4114 | 0.0667 | 0.5277 |
| dev | phone | 131 | 0.0000 | 0.0000 | 0.0076 | 0.0076 |
| dev | price | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dev | title | 15 | 0.0000 | 0.2793 | 0.0000 | 0.2968 |
| test | address | 24 | 0.0000 | 0.2594 | 0.0417 | 0.4251 |
| test | company_id | 23 | 0.0870 | 0.0870 | 0.0870 | 0.0870 |
| test | count | 32 | 0.0625 | 0.0625 | 0.0625 | 0.0625 |
| test | date | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| test | email | 12 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| test | entity_name | 17 | 0.0000 | 0.1829 | 0.2353 | 0.3805 |
| test | other | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| test | phone | 118 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| test | price | 11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| test | title | 6 | 0.0000 | 0.2838 | 0.1667 | 0.3739 |
| train | address | 131 | 0.0000 | 0.2345 | 0.0534 | 0.3015 |
| train | company_id | 225 | 0.0178 | 0.0178 | 0.0178 | 0.0178 |
| train | count | 220 | 0.1000 | 0.1000 | 0.1000 | 0.1000 |
| train | date | 56 | 0.0536 | 0.0536 | 0.0536 | 0.0536 |
| train | description | 4 | 0.0000 | 0.0500 | 0.0000 | 0.2742 |
| train | email | 103 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| train | entity_name | 118 | 0.0254 | 0.2423 | 0.0593 | 0.2878 |
| train | other | 95 | 0.0737 | 0.2858 | 0.0842 | 0.3203 |
| train | phone | 655 | 0.0046 | 0.0046 | 0.0046 | 0.0046 |
| train | price | 154 | 0.0065 | 0.0065 | 0.0130 | 0.0130 |
| train | title | 71 | 0.0563 | 0.2495 | 0.0563 | 0.2770 |

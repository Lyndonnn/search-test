# Grounded DAG-IG RL Report

## 1. Starting Point

- SFT initializer: `checkpoints/dagig_rn03_10_grounded/ground_action_sft_7b_lora`
- Previous direct-bbox autonomous IoU>=0.3: about 2.65%.
- Ground-action SFT + DINO had already established the grounded action interface: dev IoU>=0.3 about 42.86%, test about 45.31%.
- Re-evaluated SFT dev/test in this run: dev IoU>=0.3=0.42857142857142855, test IoU>=0.3=0.453125.

## 2. RL Data

- Source: `data/dagig_rn03_10_grounded_rl/grounded_rl_train/dev/test.jsonl`.
- Review-needed rows remain excluded because these files are built from the 620 hard-pass ground-action rows only.
- Leakage-sensitive rows are retained for diagnostics and excluded from RL training reward computation by the trainer.

## 3. Reward Definitions

- `outcome_only`: answer EM/F1 with malformed penalty.
- `outcome_plus_ground_penalty`: answer reward plus penalties for malformed/missing ground, DINO miss, extreme box, or missing search.
- `generic_process`: format, non-empty ground, DINO detection/center proxy, query specificity, retrieval, evidence, and answer rewards without dependency gating.
- `dagig_grounded`: dependency-aware components R_ground/R_observe/R_search/R_evidence/R_answer/R_cost; answer reward is gated by evidence support and search/observe rewards are gated by upstream grounded process quality.

## 4. Low-Budget Setup

- Initial adapter: grounded SFT LoRA.
- Diagnostic budget: default 64 train samples, 20 steps, 2 rollouts/prompt, temperature 0.2.
- This run is intended to test credit-assignment signal direction, not produce final paper-scale RL numbers.

## 5. Missing Variants

- none

## 5b. Collapsed / Partial Variants

- rl_grounded_outcome_plus_ground_penalty_lowtemp_7b_lora: early-stopped/collapsed marker `checkpoints/dagig_rn03_10_grounded_rl/rl_grounded_outcome_plus_ground_penalty_lowtemp_7b_lora.FAILED`

Collapsed variants are shown only as diagnostics; they should not be treated as successful RL runs.

## 6. Dev/Test Comparison

# Grounded RL Comparison

| method | split | format_valid | mean_iou | iou_ge_0_3 | center_hit | R@1 | R@5 | MRR | query_anchor_hit | evidence_support | unsupported_rate | spurious_success | EM | F1 | reward_total | R_ground | R_observe | R_search | R_evidence | R_answer | R_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ground_action_sft_initializer | dev | 0.9898 | 0.2828 | 0.4286 | 0.6429 | 0.4124 | 0.6082 | 0.5041 | 0.7347 | 0.7143 | 0.0102 | 0.0102 | 0.0510 | 0.1617 | 2.0078 | 0.6153 | 0.2031 | 0.8949 | 0.2143 | 0.0813 | -0.0010 |
| ground_action_sft_initializer | test | 0.9688 | 0.3102 | 0.4531 | 0.5938 | 0.3710 | 0.5645 | 0.4573 | 0.5938 | 0.6562 | 0.0000 | 0.0000 | 0.0156 | 0.1573 | 1.7918 | 0.6312 | 0.2000 | 0.7438 | 0.1562 | 0.0636 | -0.0031 |
| rl_grounded_outcome_only_lowtemp_7b_lora | dev | 0.9796 | 0.2806 | 0.4082 | 0.6633 | 0.4375 | 0.6354 | 0.5257 | 0.7449 | 0.7041 | 0.0102 | 0.0102 | 0.0510 | 0.1431 | 2.0454 | 0.6051 | 0.2418 | 0.9204 | 0.2041 | 0.0765 | -0.0026 |
| rl_grounded_outcome_only_lowtemp_7b_lora | test | 1.0000 | 0.2903 | 0.4219 | 0.5469 | 0.3750 | 0.5469 | 0.4628 | 0.5938 | 0.6719 | 0.0000 | 0.0000 | 0.0312 | 0.1550 | 1.8049 | 0.5828 | 0.2078 | 0.7750 | 0.1719 | 0.0674 | 0.0000 |
| rl_grounded_outcome_plus_ground_penalty_lowtemp_7b_lora | dev | 0.9592 | 0.2809 | 0.4184 | 0.6531 | 0.4526 | 0.6421 | 0.5361 | 0.7449 | 0.7245 | 0.0102 | 0.0102 | 0.0408 | 0.1468 | 2.0075 | 0.6041 | 0.2051 | 0.9071 | 0.2245 | 0.0713 | -0.0046 |
| rl_grounded_outcome_plus_ground_penalty_lowtemp_7b_lora | test | 0.9844 | 0.2886 | 0.4219 | 0.5469 | 0.3651 | 0.5714 | 0.4541 | 0.5781 | 0.6562 | 0.0000 | 0.0000 | 0.0156 | 0.1425 | 1.7124 | 0.5828 | 0.1703 | 0.7422 | 0.1562 | 0.0624 | -0.0016 |
| rl_grounded_generic_process_lowtemp_7b_lora | dev | 0.9796 | 0.2839 | 0.4184 | 0.6531 | 0.4375 | 0.6354 | 0.5256 | 0.7449 | 0.7551 | 0.0000 | 0.0000 | 0.0306 | 0.1380 | 2.1361 | 0.6143 | 0.2827 | 0.9020 | 0.2551 | 0.0846 | -0.0026 |
| rl_grounded_generic_process_lowtemp_7b_lora | test | 1.0000 | 0.3014 | 0.4531 | 0.5938 | 0.3906 | 0.5781 | 0.4700 | 0.5781 | 0.6719 | 0.0000 | 0.0000 | 0.0312 | 0.1434 | 1.8869 | 0.6219 | 0.2703 | 0.7594 | 0.1719 | 0.0635 | 0.0000 |
| rl_grounded_dagig_lowtemp_7b_lora | dev | 0.9796 | 0.2835 | 0.4184 | 0.6429 | 0.4062 | 0.6354 | 0.5072 | 0.7653 | 0.7143 | 0.0102 | 0.0102 | 0.0306 | 0.1522 | 2.0766 | 0.6051 | 0.2929 | 0.9061 | 0.2143 | 0.0608 | -0.0026 |
| rl_grounded_dagig_lowtemp_7b_lora | test | 0.9844 | 0.2890 | 0.4375 | 0.5469 | 0.4127 | 0.5873 | 0.4906 | 0.5938 | 0.6719 | 0.0000 | 0.0000 | 0.0156 | 0.1625 | 1.8399 | 0.5984 | 0.2375 | 0.7703 | 0.1719 | 0.0634 | -0.0016 |


## 7. Reward Components

| method | split | reward_total | R_ground | R_observe | R_search | R_evidence | R_answer | R_cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ground_action_sft_initializer | dev | 2.007806531688344 | 0.6153061224489796 | 0.2030612244897959 | 0.8948979591836734 | 0.21428571428571427 | 0.08127591944344618 | -0.0010204081632653062 |
| rl_grounded_outcome_only_lowtemp_7b_lora | dev | 2.0453607348288734 | 0.6051020408163266 | 0.24183673469387756 | 0.9204081632653062 | 0.20408163265306123 | 0.07648318380846518 | -0.0025510204081632655 |
| rl_grounded_outcome_plus_ground_penalty_lowtemp_7b_lora | dev | 2.0075221554611997 | 0.6040816326530613 | 0.20510204081632652 | 0.9071428571428571 | 0.22448979591836735 | 0.07129766566528113 | -0.004591836734693878 |
| rl_grounded_generic_process_lowtemp_7b_lora | dev | 2.1361352940233176 | 0.6142857142857143 | 0.2826530612244898 | 0.9020408163265307 | 0.25510204081632654 | 0.08460468177841969 | -0.0025510204081632655 |
| rl_grounded_dagig_lowtemp_7b_lora | dev | 2.076611179356385 | 0.6051020408163266 | 0.29285714285714287 | 0.9061224489795918 | 0.21428571428571427 | 0.060794852825772745 | -0.0025510204081632655 |
| ground_action_sft_initializer | test | 1.7917550155758843 | 0.63125 | 0.2 | 0.74375 | 0.15625 | 0.06363001557588438 | -0.003125 |
| rl_grounded_outcome_only_lowtemp_7b_lora | test | 1.8049178943637632 | 0.5828125 | 0.2078125 | 0.775 | 0.171875 | 0.06741789436376316 | 0.0 |
| rl_grounded_outcome_plus_ground_penalty_lowtemp_7b_lora | test | 1.712365223624415 | 0.5828125 | 0.1703125 | 0.7421875 | 0.15625 | 0.0623652236244148 | -0.0015625 |
| rl_grounded_generic_process_lowtemp_7b_lora | test | 1.8868887469760631 | 0.621875 | 0.2703125 | 0.759375 | 0.171875 | 0.06345124697606315 | 0.0 |
| rl_grounded_dagig_lowtemp_7b_lora | test | 1.8399130972146405 | 0.5984375 | 0.2375 | 0.7703125 | 0.171875 | 0.06335059721464037 | -0.0015625 |

## 8. Interpretation

- dev: DAG-IG improves over both outcome-only and generic-process on query_anchor_hit, F1.
- test: DAG-IG improves over both outcome-only and generic-process on R@5, MRR, F1.
- Overall: this low-budget run does not establish a clean DAG-IG win. Generic-process has stronger reward_total/evidence_support on dev and stronger reward_total/center-hit on test, while DAG-IG shows some search-side gains.
- Answer EM remains very low across all methods; answer-level credit assignment is not solved by this diagnostic run.

The grounded-interface evidence is already complete: replacing direct bbox generation with a ground-expression plus GroundingDINO interface substantially improves localization. The new RL evidence should be interpreted only as a low-budget diagnostic for long-horizon credit assignment.

## 9. Failure Analysis

- If DAG-IG improves process metrics but not EM/F1, the likely issue is downstream retrieval/answering rather than the grounding interface.
- If malformed rate rises above baselines, the reward or rollout budget is not stabilizing the action grammar.
- If grounding IoU drops while reward increases, the current proxies are too weak and need a stronger verifier or more explicit anti-spurious penalties.

## 10. Next Steps

- Run more steps and at least 3 seeds for any variant that improves process metrics here.
- Add a stronger evidence verifier before claiming answer-level credit assignment.
- Fix or review leakage-sensitive rows before scaling RL.
- Keep direct-bbox as a negative localization baseline; it is not competitive with the grounded action interface.

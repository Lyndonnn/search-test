# Counterfactual DAG-IG Report

## Executive Conclusion

This completes the requested counterfactual edge-level DAG-IG diagnostic pipeline on top of the grounded SFT checkpoint.

The grounded action interface is clearly successful: direct numeric bbox generation had IoU>=0.3 around 2.65%, while model `<ground>` expression plus GroundingDINO reaches about 42.86% dev and 45.31% test. However, the new counterfactual DAG-IG preference model does not consistently beat the generic-process baseline on long-horizon credit-assignment metrics. It is useful diagnostic evidence, not yet AAAI-level evidence that DAG-IG wins.

## Starting Point

- Project: `/storage/zhengxiang/search-test`
- Base model: Qwen2.5-VL-7B-Instruct, unchanged.
- Starting checkpoint: `checkpoints/dagig_rn03_10_grounded/ground_action_sft_7b_lora`
- Counterfactual preference checkpoint: `checkpoints/dagig_rn03_10_counterfactual_dagig/dagig_dpo_7b_lora`
- GroundingDINO implementation: local official GroundingDINO, not HuggingFace GroundingDINO.
- GroundingDINO weights: `third_party/GroundingDINO_weights/groundingdino_swint_ogc.pth`

## Baselines

`previous direct-bbox autonomous`

- Directly emitted bbox numbers.
- Did not use GroundingDINO as the action interface.
- Not fully fair against grounded-action methods because the action space is different.
- Kept as a negative localization baseline.

`teacher expression + GroundingDINO`

- Uses teacher `ground_expression`.
- GroundingDINO converts the expression to bbox/crop.
- This is a feasibility/reference line, not a trained model.

`ground-action SFT initializer`

- Qwen LoRA trained to emit `<ground>`, `<observe>`, `<search_decision>`, `<search>`, `<evidence>`, `<answer>`.
- Uses GroundingDINO in evaluation to convert the predicted expression to bbox/crop.
- Initializes all later RL/preference baselines.

`outcome-only low-budget RL`

- Same model/init/data/eval stack.
- Training reward only uses answer EM/F1 plus malformed penalty.
- Training reward does not use DINO.
- Final evaluation still uses GroundingDINO because the method emits grounded expressions.

`outcome-plus-ground-penalty low-budget RL`

- Same setup as outcome-only.
- Adds basic grounding penalties.
- Uses GroundingDINO-derived penalties when reward mode requires grounding checks.

`generic-process low-budget RL`

- Generic process reward over format, grounding, search, evidence, and answer.
- Uses GroundingDINO in training reward and final evaluation.
- No edge-level counterfactual dependency scoring.

`old heuristic DAG-IG low-budget RL`

- Uses hand-written dependency gates for ground, observe, search, evidence, answer.
- Uses GroundingDINO in training reward and final evaluation.
- This was diagnostic, not paper-level RL evidence.

`new counterfactual DAG-IG rejection SFT`

- Samples K=4 rollouts from grounded SFT.
- Builds typed counterfactuals for ground, crop, observe, search, and evidence.
- Scores edge credit as `Score(v | real u) - Score(v | counterfactual u)`.
- Builds preference pairs from high-vs-low DAG-IG rollouts within the same prompt.
- Trains rejection SFT from the grounded SFT checkpoint. This is not full DPO, but it follows the prompt option "DPO / rejection SFT" and is the safer step before full RL.

## Fairness

The fair comparison set is: SFT initializer, outcome-only RL, outcome-plus-ground-penalty RL, generic-process RL, old heuristic DAG-IG RL, and new counterfactual DAG-IG rejection SFT.

They share the same base model, same grounded SFT initializer, same dev/test splits, same output format, and same final GroundingDINO-based evaluation. The reward differs by method, which is the experimental variable.

The old direct-bbox autonomous run is not fully fair because it uses a different action interface. It is used only to show that direct bbox emission is a weak interface.

## Completed Artifacts

Scripts:

- `scripts/dagig_train/29_build_typed_counterfactuals.py`
- `scripts/dagig_train/30_score_counterfactual_dagig.py`
- `scripts/dagig_train/31_generate_sft_rollouts_for_dagig.py`
- `scripts/dagig_train/32_validate_dagig_score_predictiveness.py`
- `scripts/dagig_train/33_build_dagig_preference_pairs.py`
- `scripts/dagig_train/34_train_rejection_sft_lora.py`
- `scripts/dagig_train/34_run_dagig_dpo_or_rejection_sft.sh`
- `scripts/dagig_train/35_eval_counterfactual_dagig_model.py`

Data and rollouts:

| artifact | train | dev | test |
| --- | ---: | ---: | ---: |
| counterfactual rows | 458 | 98 | 64 |
| SFT rollouts, K=4 | 1832 | 392 | 256 |
| scored rollouts | 1832 | 392 | 256 |
| preference pairs | 372 | 75 | n/a |

Train preference pairs were used for rejection SFT. Dev pairs are diagnostic only.

## Grounded Interface Result

| method | split | mean IoU | IoU>=0.3 | center-hit |
| --- | --- | ---: | ---: | ---: |
| previous direct-bbox autonomous | old eval | 0.0358 | 0.0265 | 0.1327 |
| teacher expression + GroundingDINO | dev | 0.3042 | 0.4694 | 0.7041 |
| teacher expression + GroundingDINO | test | 0.3766 | 0.5469 | 0.7344 |
| SFT model expression + GroundingDINO | dev | 0.2828 | 0.4286 | 0.6429 |
| SFT model expression + GroundingDINO | test | 0.3102 | 0.4531 | 0.5938 |

This is the strongest positive result: the grounded action interface works much better than direct bbox generation.

## DAG-IG Score Predictiveness

| split | score -> target | Spearman | AUC / gap |
| --- | --- | ---: | ---: |
| train | DAGIG_total -> R@5 | 0.2192 | AUC 0.6302 |
| dev | DAGIG_total -> R@5 | 0.1159 | AUC 0.5690 |
| test | DAGIG_total -> R@5 | 0.2309 | AUC 0.6348 |
| train | R_ground -> IoU | 0.5514 | gap 0.2897 |
| dev | R_ground -> IoU | 0.4904 | gap 0.2469 |
| test | R_ground -> IoU | 0.5219 | gap 0.2290 |
| dev | R_search -> R@5 | 0.0311 | AUC 0.5164 |
| test | R_search -> R@5 | 0.1815 | AUC 0.5942 |
| dev | R_evidence -> evidence_support | 0.3038 | AUC 0.6570 |
| test | R_evidence -> evidence_support | 0.4072 | AUC 0.7050 |
| dev | DAGIG_total -> answer F1 | 0.0366 | weak |
| test | DAGIG_total -> answer F1 | -0.0558 | weak/negative |

Interpretation: the counterfactual scorer is strongest for grounding, moderately useful for evidence, weak for search on dev, and not predictive of answer F1.

## Final Same-Evaluator Comparison

All rows below use the same final counterfactual scorer and the same GroundingDINO evaluation path.

| method | split | IoU>=0.3 | center_hit | R@5 | MRR | evidence_support | unsupported | spurious | EM | F1 | DAGIG_total |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT initializer | dev | 0.4286 | 0.6429 | 0.6082 | 0.5041 | 0.7143 | 0.0102 | 0.0102 | 0.0510 | 0.1617 | 0.1298 |
| outcome-only RL | dev | 0.4082 | 0.6633 | 0.6354 | 0.5257 | 0.7041 | 0.0102 | 0.0102 | 0.0510 | 0.1431 | 0.1769 |
| outcome+ground-penalty RL | dev | 0.4184 | 0.6531 | 0.6421 | 0.5361 | 0.7245 | 0.0102 | 0.0102 | 0.0408 | 0.1468 | 0.1249 |
| generic-process RL | dev | 0.4184 | 0.6531 | 0.6354 | 0.5256 | 0.7551 | 0.0000 | 0.0000 | 0.0306 | 0.1380 | 0.1863 |
| old heuristic DAG-IG RL | dev | 0.4184 | 0.6429 | 0.6354 | 0.5072 | 0.7143 | 0.0102 | 0.0102 | 0.0306 | 0.1522 | 0.1958 |
| counterfactual DAG-IG rejection SFT | dev | 0.4082 | 0.6531 | 0.6186 | 0.5132 | 0.7347 | 0.0000 | 0.0000 | 0.0612 | 0.1642 | 0.1641 |
| SFT initializer | test | 0.4531 | 0.5938 | 0.5645 | 0.4573 | 0.6562 | 0.0000 | 0.0000 | 0.0156 | 0.1573 | 0.0022 |
| outcome-only RL | test | 0.4219 | 0.5469 | 0.5469 | 0.4628 | 0.6719 | 0.0000 | 0.0000 | 0.0312 | 0.1550 | -0.0588 |
| outcome+ground-penalty RL | test | 0.4219 | 0.5469 | 0.5714 | 0.4541 | 0.6562 | 0.0000 | 0.0000 | 0.0156 | 0.1425 | -0.0976 |
| generic-process RL | test | 0.4531 | 0.5938 | 0.5781 | 0.4700 | 0.6719 | 0.0000 | 0.0000 | 0.0312 | 0.1434 | 0.0172 |
| old heuristic DAG-IG RL | test | 0.4375 | 0.5469 | 0.5873 | 0.4906 | 0.6719 | 0.0000 | 0.0000 | 0.0156 | 0.1625 | -0.0901 |
| counterfactual DAG-IG rejection SFT | test | 0.4531 | 0.6094 | 0.5714 | 0.4571 | 0.6562 | 0.0000 | 0.0000 | 0.0156 | 0.1635 | -0.0095 |

## Interpretation

Grounding:

- Counterfactual rejection SFT does not improve dev IoU>=0.3 over generic-process.
- On test it ties generic-process on IoU>=0.3 and has better center-hit: 0.6094 vs 0.5938.

Search:

- Counterfactual rejection SFT does not improve R@5/MRR over generic-process.
- Dev R@5: 0.6186 vs generic 0.6354.
- Test R@5: 0.5714 vs generic 0.5781.
- This prevents a clean DAG-IG credit-assignment claim.

Evidence:

- Counterfactual rejection SFT improves over SFT on dev evidence_support, but does not beat generic-process.
- Dev: 0.7347 vs generic 0.7551.
- Test: 0.6562 vs generic 0.6719.

Answer:

- Counterfactual rejection SFT has best dev EM/F1 in this table.
- On test it has best F1 by a small margin.
- These answer-side gains are small and not backed by stronger retrieval/evidence metrics.

Spurious/unsupported:

- Counterfactual rejection SFT has zero unsupported/spurious rate under the current verifier.
- Generic-process also has zero unsupported/spurious, so this is not unique.

## Prompt Compliance

Completed:

- Built typed counterfactuals for train/dev/test.
- Implemented edge-level counterfactual DAG-IG scorer.
- Sampled K=4 rollouts from grounded SFT for train/dev/test.
- Scored all requested edges.
- Validated score predictiveness with Spearman, top-bottom gap, and AUC.
- Built train/dev preference pairs.
- Trained rejection SFT from the grounded SFT checkpoint.
- Evaluated SFT initializer, outcome-only RL, outcome+ground-penalty RL, generic-process RL, old heuristic DAG-IG RL, and new counterfactual DAG-IG rejection SFT.

Deviation:

- The checkpoint path is named `dagig_dpo_7b_lora`, but the method is rejection SFT, not full DPO. This follows the prompt's "DPO / rejection SFT" option.
- I did not proceed to full RL after rejection SFT, because the prompt explicitly asked not to simply increase RL steps and to do preference learning first.

## What Is Missing For AAAI

- True DPO or full RL after preference learning.
- Multiple seeds and statistical tests.
- Larger rollout budget than K=4.
- Stronger search edge scorer; current search-edge predictiveness is weak on dev.
- Stronger evidence verifier or entailment judge.
- A no-ground search baseline in the same RN03_10 setting.
- Counterfactual quality audit; average `R_ground` is negative because some counterfactual ground expressions are easier for DINO than sampled model expressions.
- Equal-compute ablations: only-ground, only-search, only-evidence, no-counterfactual, and stronger outcome-only.

## Bottom Line

The grounded interface result is strong and usable. The counterfactual DAG-IG result is not yet strong enough to claim success over generic-process. It gives a clear debugging direction: improve the search/evidence verifier and counterfactual quality, then rerun preference learning and full RL with multiple seeds.


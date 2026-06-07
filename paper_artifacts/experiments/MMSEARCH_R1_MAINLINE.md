# MMSearch-R1 Mainline Protocol for DAG-IG

This file defines the paper-relevant mainline. Toy DAG-IG diagnostics are useful for unit tests, but they are not evidence for the paper claim.

## Problem

Outcome-only MMSearch-R1 training rewards the final answer and applies format/search penalties. On our FVQA-128 debug run, 50-step outcome-only GRPO improved validation reward/score, but the validation search ratios collapsed to zero:

```text
val reward: 0.3023
val score:  0.2266
text search ratio:  0.0
image search ratio: 0.0
```

This is the exact failure mode DAG-IG should address. A model can improve short-run answer score by avoiding tools, because tool calls add cost, formatting risk, latency, and noisy observations. That is not a search agent.

## MMSearch-R1 Baseline We Must Match

Use the upstream MMSearch-R1 protocol as the anchor:

- Base model: Qwen2.5-VL-Instruct, with 7B as the paper-scale target and 3B as the debug target.
- Training: veRL GRPO with multi-turn MMSearch rollout.
- Tools: image search and text search; max generation rounds is 3 in the full protocol.
- Dataset: FVQA is the first practical debug dataset. The paper-scale path should increase from FVQA-128 to FVQA-1k/5k before claiming trends.
- Metrics: answer score/accuracy and search ratio must be tracked together.

The minimum publishable comparison is not "DAG-IG reward can be computed." It is:

```text
Direct / pretrained MMSearch-R1 rollout
Outcome-only GRPO
Outcome-only GRPO with no search penalty
Search-positive shaping sanity baseline
DAG-IG-Lite GRPO
```

All methods must use the same data split, model, tool backend, max rounds, batch size, and validation script.

## Acceptance Criteria

DAG-IG is useful only if it improves the Pareto point:

1. Answer score or accuracy is higher than pretrained/direct and competitive with outcome-only.
2. Search ratio does not collapse to 0 and does not saturate at 100%.
3. Search failure ratio stays low.
4. The gain remains when FVQA grows from 128 to 1k+ samples.
5. Reward diagnostics show that future-action credit is active on examples where local answer IG is weak but search enables the next action.

If DAG-IG raises search ratio but lowers answer score sharply, it is just encouraging search, not better search. If it raises answer score while search remains zero, it is not validating the method.

## Current Status

Completed:

- MMSearch-R1 pinned environment and veRL version are running on double A800.
- FVQA debug parquet preparation works.
- Pretrained/val-only and outcome-only GRPO debug loops run end-to-end.
- 50-step outcome-only GRPO demonstrates the search-collapse failure mode.

Immediate gap:

- DAG-IG is not yet injected into the MMSearch-R1 GRPO reward manager.
- Current DAG-IG logprob/counterfactual code lives in `projects/dagig_mmsearch`; it must be bridged into `mmsearch_r1/workers/multimodal/reward/` without replacing the upstream training loop.

## Next Experiments

Run in this order.

1. `pretrained_val`: verify the current base model score/search ratio on the same FVQA val split.
2. `outcome_50`: reproduce the existing 50-step outcome-only result.
3. `outcome_no_search_penalty_50`: determine whether search collapse is mainly caused by explicit search penalty or by sparse final-answer reward.
4. `search_bonus_sanity_50`: a non-paper sanity baseline that gives a small correct-search bonus. If this cannot increase search ratio, the rollout/prompt/data path is the bottleneck, not DAG-IG.
5. `dagig_lite_50`: true method hook. Add action-span reward to search actions using local/future IG diagnostics, then compare against the same matrix.

## Storage Policy

AutoDL data disk is small. Keep:

- `/root/autodl-tmp/search-test`
- `/root/autodl-tmp/dagig/hf_cache`
- `mmsearch_r1/data/fvqa_debug_*.pq`
- `mmsearch_r1/data/fvqa_debug_images`
- compact results under `/root/autodl-tmp/dagig/results_keep`

Delete:

- optimizer checkpoints
- `/tmp/ray`
- pip/build caches
- duplicate validation JSONs after compact summaries are copied


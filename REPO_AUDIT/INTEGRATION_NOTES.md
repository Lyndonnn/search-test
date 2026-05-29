# DAG-IG Integration Notes

## Current Best Integration Strategy

The repository already contains MMSearch-R1 and veRL. DAG-IG should therefore be integrated as a reward and trace adapter, not as a replacement for the existing trainer.

Recommended path:

1. Keep `projects/dagig_mmsearch/` as the research sandbox.
2. Represent every tool call as `reward.types.ToolStep`.
3. Convert MMSearch-R1 multi-turn rollout outputs into `reward.types.Trajectory`.
4. Score local answer IG and future-action IG with a frozen reference policy.
5. Inject DAG-IG rewards only into the target action spans, using the existing multi-turn response mask/span data when available.

## Existing Components To Reuse

- Existing text and image search tools under `mmsearch_r1/utils/tools/`.
- Existing MMSearch-R1 rollout logic under `mmsearch_r1/workers/multimodal/rollout/`.
- Existing veRL reward manager pattern under `mmsearch_r1/workers/multimodal/reward/`.
- Existing answer scoring under `mmsearch_r1/utils/reward_score_mm/`.
- Existing veRL-compatible dataset format under `mmsearch_r1/utils/dataset/`.

## Adapter Boundary

The DAG-IG adapter should consume:

- `sample_id`
- question text
- image references
- gold answers
- full prompt and response
- per-step tool type
- action text and token span
- raw observation
- evidence summary
- context before action
- context after observation

The adapter should emit:

- per-step `StepReward`
- token reward vector aligned to response tokens
- diagnostics for JSONL logging, tables, and figures

## First Prototype Scope

The implemented first stage supports:

- `text_search`
- `image_search`
- `stop`
- typed counterfactual sampling
- rule-based self-evidence summaries
- local answer IG
- next-step future-action IG
- DAG-IG-Lite propagation
- gate reward
- tool cost penalty
- action-span-only token reward injection
- smoke-testable toy rollouts without GPU or external search services

## Later A800/A100 Integration Work

- Replace toy logprob scorer with frozen Qwen2.5-VL reference-policy logprob.
- Save real action spans from MMSearch-R1 rollout.
- Add crop/OCR/select traces to the rollout.
- Add full DAG top-k dependency propagation.
- Add veRL reward manager adapter for production RL.


# FINAL_STATUS

Updated: 2026-06-02

## Current Environment

- Working directory: `/Users/lyndon/Desktop/search-test`
- Python: 3.9.6
- Platform: macOS-15.7.3-arm64-arm-64bit
- Current run mode: local CPU smoke prototype

## GPU Information

- `cuda_available=False`
- No local CUDA GPU detected in this workspace.
- A800/A100 production training has not been run in this environment.

## Successfully Run Commands

- `make prepare_data`
- `make build_indexes`
- `make smoke`
- `make agent_rollout_smoke`
- `make eval_all`
- `make audit`
- `make make_figures`
- `make setup`
- `make autodl_check`
- `bash projects/dagig_mmsearch/scripts/clone_third_party.sh` completed with handled clone failures and wrote status output.

## Failed Commands

- First `make eval_all` failed at `make_figures` because Matplotlib aborted on unwritable font/cache directories.
- `make make_figures` failed once for the same Matplotlib abort.
- Escalated network retry for `bash projects/dagig_mmsearch/scripts/clone_third_party.sh` timed out waiting for approval.

## Failure Reasons And Fixes

- Matplotlib failure: replaced figure generation with a pure Pillow/fallback renderer and set local cache directories in `make_figures.sh`.
- GitHub clone failure: current sandbox could not resolve `github.com`; references are documented in `third_party/README.md` and the shallow clone script is ready for an environment with network access.

## Generated Result Paths

- `REPO_AUDIT/tree.txt`
- `REPO_AUDIT/REPO_AUDIT.md`
- `REPO_AUDIT/INTEGRATION_NOTES.md`
- `data/processed/toy_vqa.jsonl`
- `data/indexes/text_corpus.jsonl`
- `data/indexes/image_corpus.jsonl`
- `results/baselines/direct_vqa_smoke.jsonl`
- `results/baselines/prompted_search_smoke.jsonl`
- `results/direct_vqa/direct_vqa_smoke.jsonl`
- `results/prompted_search/prompted_search_smoke.jsonl`
- `results/outcome_rl/outcome_rl_smoke.jsonl`
- `results/local_ig/local_ig_smoke.jsonl`
- `results/dagig_lite/dagig_reward_debug.jsonl`
- `results/dagig_lite/dagig_lite_smoke.jsonl`
- `results/agent_rollout/agentic_rollout_smoke.jsonl`
- `paper_artifacts/tables/main_table.csv`
- `paper_artifacts/tables/efficiency_table.csv`
- `paper_artifacts/tables/ablation_table.csv`
- `paper_artifacts/tables/attribution_diagnostic.csv`
- `paper_artifacts/tables/agentic_rollout_smoke.csv`
- `paper_artifacts/figures/accuracy_vs_toolcalls.png`
- `paper_artifacts/figures/local_ig_hist.png`
- `paper_artifacts/figures/future_action_ig_hist.png`
- `paper_artifacts/figures/dependency_heatmap.png`
- `paper_artifacts/figures/reward_by_tool_type.png`
- `paper_artifacts/figures/case_study_dagig.png`
- `paper_artifacts/case_studies/toy_dagig_case.md`

## DAG-IG-Lite Status

- DAG-IG-Lite minimal implementation is complete for CPU smoke:
  - typed `ToolStep` and `Trajectory`
  - typed counterfactual pool
  - rule-based self-evidence summaries
  - Local IG
  - Future Action IG
  - next-step DAG-IG-Lite propagation
  - action-span-only token reward injection
  - gate reward
  - tool cost penalty
  - JSONL rollout/debug logging
  - tables and figures
  - reference-policy reward smoke entrypoint
  - FVQA small-data adapter with Colab finalization-abort tolerance
- reference reward ablation entrypoint
- reference ablation step-level delta diagnostic table
- lightweight agentic rollout smoke entrypoint
- Qwen/HF model-agent rollout entrypoint
- robust action parser for multi-JSON model outputs
- placeholder search-action repair for `"action":"query"` outputs
- optional DAG-IG reward scoring on model-agent trajectories
- two-turn non-oracle model-agent rollout entrypoint
- answer-field stripping for model-visible search observations
- answer-text redaction for gold-derived diagnostic snippets
- final-answer-specific parser for second-turn stop generation
- 38 passing smoke tests
- Production A800 training is not yet complete because this local environment has no CUDA GPU and no model download was attempted.

## Next Most Important Command

On AutoDL A800:

```bash
source projects/dagig_mmsearch/scripts/autodl_env.sh
make autodl_check
make hf_probe
make setup
make prepare_data
make build_indexes
make smoke
make train_dagig_lite
make reference_logprob_smoke
```

`make hf_probe` checks Qwen model metadata access before the full download. `make reference_logprob_smoke` is the first non-toy reward-scoring step: it loads `Qwen/Qwen2.5-VL-3B-Instruct` as a frozen HF reference policy and computes DAG-IG rewards on a tiny batch.

Next real-data diagnostic command:

```bash
python3 -m pip install datasets
DAGIG_REAL_DATASET=fvqa DAGIG_REAL_SPLIT=train DAGIG_REAL_LIMIT=32 make prepare_real_data
DAGIG_REF_SAMPLES_JSONL=data/processed/fvqa_train_small.jsonl \
DAGIG_REF_TEXT_INDEX=data/indexes/fvqa_train_text_corpus.jsonl \
DAGIG_REF_IMAGE_INDEX=data/indexes/fvqa_train_image_corpus.jsonl \
DAGIG_REF_METHOD=reference_logprob_fvqa_train \
DAGIG_REF_SMOKE_LIMIT=32 \
DAGIG_REF_CF_SAMPLES=2 \
make reference_logprob_smoke
```

After the FVQA 32/cf4 reference diagnostic is stable, run reward ablations:

```bash
DAGIG_REF_SAMPLES_JSONL=data/processed/fvqa_train_small.jsonl \
DAGIG_REF_TEXT_INDEX=data/indexes/fvqa_train_text_corpus.jsonl \
DAGIG_REF_IMAGE_INDEX=data/indexes/fvqa_train_image_corpus.jsonl \
DAGIG_ABLATION_LIMIT=32 \
DAGIG_ABLATION_CF_SAMPLES=4 \
DAGIG_ABLATION_METHOD_PREFIX=reference_ablation_fvqa_train \
make reference_ablation
```

Key output:

- `paper_artifacts/tables/reference_ablation.csv`
- `paper_artifacts/tables/reference_ablation_delta.csv`

Then run lightweight agent rollout:

```bash
DAGIG_AGENT_LIMIT=32 \
DAGIG_AGENT_SAMPLES_JSONL=data/processed/fvqa_train_small.jsonl \
DAGIG_AGENT_TEXT_INDEX=data/indexes/fvqa_train_text_corpus.jsonl \
DAGIG_AGENT_IMAGE_INDEX=data/indexes/fvqa_train_image_corpus.jsonl \
make agent_rollout_smoke
```

Then run raw Qwen model-agent rollout:

```bash
DAGIG_MODEL_AGENT_LIMIT=32 \
DAGIG_MODEL_AGENT_SAMPLES_JSONL=data/processed/fvqa_train_small.jsonl \
DAGIG_MODEL_AGENT_TEXT_INDEX=data/indexes/fvqa_train_text_corpus.jsonl \
DAGIG_MODEL_AGENT_IMAGE_INDEX=data/indexes/fvqa_train_image_corpus.jsonl \
make model_agent_rollout
```

The raw one-turn rollout is diagnostic only because it still turns tool `answer` fields into a stop answer. The non-oracle MMSearch-R1-style rollout is:

```bash
DAGIG_MODEL_AGENT_LIMIT=32 \
DAGIG_MODEL_AGENT_SAMPLES_JSONL=data/processed/fvqa_train_small.jsonl \
DAGIG_MODEL_AGENT_TEXT_INDEX=data/indexes/fvqa_train_text_corpus.jsonl \
DAGIG_MODEL_AGENT_IMAGE_INDEX=data/indexes/fvqa_train_image_corpus.jsonl \
make model_agent_two_turn
```

If raw Qwen copies placeholder JSON or emits multiple JSON objects, rerun after this parser update. To attach DAG-IG rewards to model-agent trajectories:

```bash
DAGIG_MODEL_AGENT_SCORE_REWARD=1 \
DAGIG_MODEL_AGENT_CF_SAMPLES=2 \
DAGIG_MODEL_AGENT_LIMIT=32 \
DAGIG_MODEL_AGENT_SAMPLES_JSONL=data/processed/fvqa_train_small.jsonl \
DAGIG_MODEL_AGENT_TEXT_INDEX=data/indexes/fvqa_train_text_corpus.jsonl \
DAGIG_MODEL_AGENT_IMAGE_INDEX=data/indexes/fvqa_train_image_corpus.jsonl \
make model_agent_rollout
```

Use the same `DAGIG_MODEL_AGENT_SCORE_REWARD=1` flag with `make model_agent_two_turn` after two-turn invalid-action and accuracy diagnostics are stable.

## A100 Multi-Card Expansion Readiness

- Not ready for production A100 multi-card expansion yet.
- The config scaffold exists in `projects/dagig_mmsearch/configs/dagig_full_qwen25vl_7b_a100x4.yaml`.
- Required before starting full A100 work:
  - validate real reference-policy logprob scoring on A800 with `make reference_logprob_smoke`
  - validate FVQA 32/cf4 ablation table with `make reference_ablation`
  - inspect raw Qwen model-agent invalid/under-search behavior with `make model_agent_rollout`
  - validate non-oracle two-turn trajectories with `make model_agent_two_turn`
  - veRL reward-manager adapter
  - real MMSearch-R1 trajectory span extraction
  - crop/OCR/select rollout traces
  - full DAG top-k edge propagation

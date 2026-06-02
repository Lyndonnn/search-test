# DAG-IG: Dependency-Aware Information Gain for Multimodal Search Agents

DAG-IG extends answer-centered information gain from text search agents to typed multimodal tool-use trajectories.

The key limitation of local IG is that it only asks whether the current observation increases the probability of the final answer. In multimodal search, early steps often act as enablers: an image search may enable a later select action, a crop may enable OCR, and OCR may enable a better text-search query. DAG-IG adds future-action information gain so these early actions receive credit when their observations make later true action spans more likely.

## Contributions

1. We generalize answer-centered information gain from text search to typed multimodal tool-use trajectories.
2. We introduce future-action information gain, which treats later tool actions as hindsight subgoals and measures whether an earlier observation enables them.
3. We propose DAG-IG-Lite, a low-cost next-step dependency credit propagation method requiring no human subgoal labels.
4. We provide a reproducible sandbox training and evaluation pipeline for multimodal search agents with typed counterfactuals, self-evidence summaries, and attribution diagnostics.

## Method

For each tool step `i`, local IG is:

```text
g_i = log p(a* | C_i_real) - E_{O_i_cf} log p(a* | C_i_replace_i)
```

Future-action IG for a later action span `A_j` is:

```text
d_i_j = ReLU(log p(A_j | H_j_real) - E_{O_i_cf} log p(A_j | H_j_replace_i))
```

DAG-IG-Lite only scores adjacent dependencies:

```text
R_i = g_i + lambda_dep * d_i_i+1 * max(g_i+1, 0)
R_last = g_last
```

Rewards are injected only into the source action span:

```text
bonus_token = alpha * R_i / len(A_i)
```

This action length normalization prevents query-length reward hacking.

## Implemented First Stage

- Tool types: `text_search`, `image_search`, `stop`, plus smoke-test stubs for `crop`, `ocr`, and `select`.
- Typed counterfactual pool with per-tool replacement metadata.
- Rule-based self-evidence summaries.
- Local IG scorer.
- Future-action IG scorer.
- DAG-IG-Lite reward aggregation.
- Gate reward and tool cost penalty.
- Toy rollouts and reproducible smoke tests.
- Table, figure, and case-study generation.

## Commands

```bash
make setup
make audit
make prepare_data
make build_indexes
make smoke
make eval_nosearch
make eval_prompted
make train_outcome
make train_local_ig
make train_dagig_lite
make eval_all
make make_tables
make make_figures
```

## AutoDL A800 Next Step

After cloning on AutoDL, source the environment helper so model downloads and logs go to the data disk:

```bash
cd /root/search-test
source projects/dagig_mmsearch/scripts/autodl_env.sh
make autodl_check
make hf_probe
```

Then run the current CPU-safe smoke:

```bash
make smoke
make train_dagig_lite
```

To test real frozen-reference logprob scoring with Qwen2.5-VL-3B on the A800:

```bash
source projects/dagig_mmsearch/scripts/autodl_env.sh
make hf_probe
make reference_logprob_smoke
```

On AutoDL, `autodl_env.sh` defaults to `HF_ENDPOINT=https://hf-mirror.com` and downloads `Qwen/Qwen2.5-VL-3B-Instruct` into `/root/autodl-tmp/dagig/hf_cache`. Override the endpoint when needed:

```bash
export HF_ENDPOINT=https://huggingface.co
```

The reference smoke scores a 2-sample DAG-IG batch with `cf_samples=1` by default, writing `results/dagig_lite/reference_logprob_smoke.jsonl` and `paper_artifacts/tables/reference_logprob_smoke.csv`.

## Real Small-Data Diagnostic

After toy reward diagnostics are stable, prepare a small HF VQA split and score it with the same reference-policy path:

```bash
source projects/dagig_mmsearch/scripts/autodl_env.sh
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

The adapter also accepts any HF dataset name through `DAGIG_REAL_DATASET=<hf_org/dataset>`. It streams the dataset by default, extracts common `question`, `answer(s)`, `prompt`, `reward_model`, and image columns into the DAG-IG `VQASample` schema, then builds local diagnostic search indexes from gold evidence. This is a reward-diagnostic bridge, not yet the final production retrieval setup.

The current implementation is intentionally CPU-smoke-testable. A800/A100 training should replace the toy logprob scorer with a frozen Qwen2.5-VL reference-policy scorer and attach the reward adapter to the existing MMSearch-R1/veRL rollout path.

## Ablation And Agent Rollout

After FVQA 32-sample reference scoring is stable with `cf_samples=4`, run controlled reward ablations on the same trajectories:

```bash
DAGIG_REF_SAMPLES_JSONL=data/processed/fvqa_train_small.jsonl \
DAGIG_REF_TEXT_INDEX=data/indexes/fvqa_train_text_corpus.jsonl \
DAGIG_REF_IMAGE_INDEX=data/indexes/fvqa_train_image_corpus.jsonl \
DAGIG_ABLATION_LIMIT=32 \
DAGIG_ABLATION_CF_SAMPLES=4 \
DAGIG_ABLATION_METHOD_PREFIX=reference_ablation_fvqa_train \
make reference_ablation
```

This writes per-variant JSONL files under `results/ablations/` and the summary table `paper_artifacts/tables/reference_ablation.csv`. The default variants are `local_ig_only`, `dagig_lite`, `dagig_no_gate`, `dagig_no_cost`, and `lambda_dep` values `0`, `0.25`, `0.5`, `1.0`.

It also writes `paper_artifacts/tables/reference_ablation_delta.csv`, a step-level comparison between `local_ig_only` and `dagig_lite`. Key fields:

- `future_edge_active`: whether `d_i->i+1 > 0`
- `future_credit_eligible`: whether `d_i->i+1 > 0` and `g_i+1 > 0`
- `total_reward_delta`: how much more reward DAG-IG-Lite gives the step than Local-IG only

To run the lightweight agent rollout smoke without loading Qwen:

```bash
DAGIG_AGENT_LIMIT=32 \
DAGIG_AGENT_SAMPLES_JSONL=data/processed/fvqa_train_small.jsonl \
DAGIG_AGENT_TEXT_INDEX=data/indexes/fvqa_train_text_corpus.jsonl \
DAGIG_AGENT_IMAGE_INDEX=data/indexes/fvqa_train_image_corpus.jsonl \
make agent_rollout_smoke
```

This writes `results/agent_rollout/agentic_rollout_smoke.jsonl` and `paper_artifacts/tables/agentic_rollout_smoke.csv`. To score ablations on these agentic trajectories instead of fixed prompted trajectories, add:

```bash
DAGIG_ABLATION_ROLLOUT_MODE=agentic make reference_ablation
```

To run the first true model-agent action-generation pass with Qwen:

```bash
DAGIG_MODEL_AGENT_LIMIT=32 \
DAGIG_MODEL_AGENT_SAMPLES_JSONL=data/processed/fvqa_train_small.jsonl \
DAGIG_MODEL_AGENT_TEXT_INDEX=data/indexes/fvqa_train_text_corpus.jsonl \
DAGIG_MODEL_AGENT_IMAGE_INDEX=data/indexes/fvqa_train_image_corpus.jsonl \
make model_agent_rollout
```

This writes `results/model_agent/model_agent_rollout.jsonl` and `paper_artifacts/tables/model_agent_rollout.csv`. Unlike the smoke fallback, this default mode does not force a search when the model stops early and does not repair invalid JSON. Use it to measure raw `invalid_action_rate`, under-search, tool choice, and query quality before training.

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
```

Then run the current CPU-safe smoke:

```bash
make smoke
make train_dagig_lite
```

To test real frozen-reference logprob scoring with Qwen2.5-VL-3B on the A800:

```bash
source projects/dagig_mmsearch/scripts/autodl_env.sh
make reference_logprob_smoke
```

This downloads `Qwen/Qwen2.5-VL-3B-Instruct` into `/root/autodl-tmp/dagig/hf_cache` when running on AutoDL. It scores a 2-sample DAG-IG batch with `cf_samples=1` by default, writing `results/dagig_lite/reference_logprob_smoke.jsonl` and `paper_artifacts/tables/reference_logprob_smoke.csv`.

The current implementation is intentionally CPU-smoke-testable. A800/A100 training should replace the toy logprob scorer with a frozen Qwen2.5-VL reference-policy scorer and attach the reward adapter to the existing MMSearch-R1/veRL rollout path.

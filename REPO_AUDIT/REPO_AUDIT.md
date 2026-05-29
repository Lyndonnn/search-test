# Repository Audit: search-test

Audit date: 2026-05-29

## Summary

This repository already contains a working MMSearch-R1 style research codebase:

- `mmsearch_r1/` contains multimodal search agents, tools, datasets, rewards, rollouts, workers, prompts, and trainer entrypoints.
- `verl/` is vendored and provides the core RL trainer, rollout workers, reward manager interfaces, datasets, and distributed execution stack.
- `scripts/` contains local sanity/eval/data-prep scripts for the existing MMSearch-R1 prototype.
- `configs/exp/m0_sanity.yaml` contains a lightweight sanity configuration for Qwen2.5-VL-3B.

DAG-IG should be added as an isolated research subproject under `projects/dagig_mmsearch/` and later integrated through adapters into the existing MMSearch-R1 reward/rollout path. The current implementation should not rewrite `mmsearch_r1/` or `verl/`.

## Current Directory Structure

See `REPO_AUDIT/tree.txt` for the full tree snapshot.

Important top-level paths:

- `README.md`: MMSearch-R1 project overview and setup notes.
- `requirements.txt`: root Python dependencies.
- `mmsearch_r1/agents/`: existing grounded zoom/search/verify prototype.
- `mmsearch_r1/utils/tools/`: existing `text_search`, `image_search`, offline search, SerpApi backend.
- `mmsearch_r1/utils/dataset/`: veRL-compatible multimodal RL datasets.
- `mmsearch_r1/utils/reward_score_mm/`: answer and format reward utilities.
- `mmsearch_r1/workers/multimodal/rollout/`: multi-turn MMSearch rollout code.
- `mmsearch_r1/workers/multimodal/reward/`: reward manager implementations.
- `mmsearch_r1/trainer/multimodal/`: PPO/GRPO trainer entrypoints and configs.
- `verl/`: vendored veRL training framework.
- `scripts/`: local sanity, debug, data preparation, and evaluation scripts.
- `configs/`: existing experiment config directory.

## Existing MMSearch / Multimodal-Search-R1 Style Code

The repository already has the main MMSearch-R1 components:

- Multi-turn rollout: `mmsearch_r1/workers/multimodal/rollout/vllm_rollout_spmd.py`
- Text search tool: `mmsearch_r1/utils/tools/text_search.py`
- Image search tool: `mmsearch_r1/utils/tools/image_search.py`
- Offline retrieval backend: `mmsearch_r1/utils/tools/offline_search.py`
- Reward manager: `mmsearch_r1/workers/multimodal/reward/mmsearch_r1.py`
- Reward scoring utilities: `mmsearch_r1/utils/reward_score_mm/mmsearch_r1_score.py`
- Multimodal dataset: `mmsearch_r1/utils/dataset/mm_rl_dataset.py`
- GRPO/PPO entrypoint: `mmsearch_r1/trainer/multimodal/main_ppo.py`
- Training script: `mmsearch_r1/scripts/run_mmsearch_r1_grpo.sh`

## Existing Trainer / Rollout / Reward / Tool / Eval / Data Loader

- Trainer: present via `mmsearch_r1/trainer/multimodal` and `verl/verl/trainer`.
- Rollout: present via `mmsearch_r1/workers/multimodal/rollout` with multi-turn text/image search.
- Reward: present via answer correctness, format, and search penalties in `mmsearch_r1/workers/multimodal/reward`.
- Tools: present for text and image search; grounded zoom prototype includes crop-like region logic.
- Eval scripts: present in `scripts/eval_*` and `scripts/run_grounded_zoom_search_verify.py`.
- Data loader: present via veRL parquet dataset in `mmsearch_r1/utils/dataset/mm_rl_dataset.py`.

## Existing Requirements / Environment / Makefile / Scripts

- `requirements.txt` exists and is tailored to MMSearch-R1 + veRL.
- `environment.yml` did not exist before DAG-IG scaffolding.
- `Makefile` did not exist before DAG-IG scaffolding.
- `scripts/` exists with local utilities, but no unified DAG-IG experiment entrypoints.

## Reusable Files

- `mmsearch_r1/utils/tools/text_search.py` and `image_search.py` can be wrapped by DAG-IG tool adapters.
- `mmsearch_r1/workers/multimodal/rollout/vllm_rollout_spmd.py` can later emit DAG-IG `ToolStep` traces.
- `mmsearch_r1/workers/multimodal/reward/mmsearch_r1.py` can later call DAG-IG reward aggregation and inject step-token rewards.
- `mmsearch_r1/utils/reward_score_mm/mmsearch_r1_score.py` can be reused for EM/sub-EM and answer extraction.
- `mmsearch_r1/utils/dataset/mm_rl_dataset.py` can be reused for veRL parquet training data.
- `configs/exp/m0_sanity.yaml` is useful for local Qwen2.5-VL-3B sanity runs.

## Files To Avoid Modifying

These files are core upstream-like code or existing research baselines and should not be changed for the first DAG-IG prototype:

- `verl/**`
- `mmsearch_r1/trainer/**`
- `mmsearch_r1/workers/**`
- `mmsearch_r1/utils/dataset/**`
- `mmsearch_r1/utils/tools/**`
- `mmsearch_r1/utils/reward_score_mm/**`
- Existing scripts in `scripts/`

If integration is needed later, add thin adapter modules and minimal config hooks instead of rewriting these files.

## Recommended New Code Location

Use the isolated subproject:

`projects/dagig_mmsearch/`

This keeps DAG-IG reward math, typed counterfactuals, smoke tests, toy data, tables, and figures separate from existing MMSearch-R1 training code. The first-stage integration path is:

1. Build and test DAG-IG-Lite with toy trajectories and local deterministic logprob scorers.
2. Wrap existing MMSearch-R1 tools through `projects/dagig_mmsearch/src/tools`.
3. Convert existing rollout traces into DAG-IG `Trajectory` and `ToolStep` records.
4. Add a reward-manager adapter that injects DAG-IG token rewards into veRL response spans.
5. Keep full GRPO/A800 training scripts as reproducible shell entrypoints.


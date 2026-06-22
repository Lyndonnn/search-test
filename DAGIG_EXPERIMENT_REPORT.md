# DAG-IG-R1 Experiment Report

Date: 2026-06-19

This run used only the cloned repository at `/storage/zhengxiang/search-test` for code development and execution. The dataset zip and goal prompt were used as input references; no code was reused from `/storage/zhengxiang/code/00-agentic-rl-search/dagig_r1`.

## Data

Dataset:

- Zip: `/storage/zhengxiang/code/00-agentic-rl-search/pix2fact_dagig_1k_gpt54_teacher_clean_package.zip`
- Unpacked package: `data/pix2fact_dagig_1k_gpt54_teacher_clean_package`
- Main training file: `data/pix2fact_dagig_1k_gpt54_teacher_clean_package/data/pix2fact_dagig_train_AB_clean_split.jsonl`

Sanity check outputs:

- `results/dagig_train/sanity/dataset_summary.json`
- `results/dagig_train/sanity/random_examples.md`
- `results/dagig_train/sanity/contact_sheet.jpg`

Verified:

- 880 Tier A/B rows.
- Split counts: 660 train, 132 dev, 88 test.
- Training weights: Tier A = 1.0, Tier B = 0.6.
- 1760 model-input images verified: full image plus crop for every row.
- Tier C/D rows are not in the main training file.

## Implemented Pipeline

Scripts under `scripts/dagig_train/` now support the clean package schema:

- `00_inspect_package.py`: strict package/image/reward sanity checks.
- `01_build_sft_data.py`: segment-weighted SFT JSONL for `uniform_sft`, `outcome_only_sft`, `local_ig_sft`, `dagig_sft`, and `dagig_action_only_sft`.
- `02_train_lora_qwen_vl.py`: Qwen2.5-VL LoRA SFT with token-level segment loss weights.
- `03_eval_chain.py`: oracle-crop chain generation and tag-level metrics.
- `04_eval_query_retrieval.py`: BM25 supporting-evidence retrieval diagnostics.
- `05_build_retrieval_corpus.py`: fixed offline evidence corpus.
- `06_score_rollouts.py`: outcome-only, search-penalty, generic-process, text-IG, and DAG-IG reward scoring.
- `07_make_tables.py`: aggregate CSV/Markdown tables.
- `08_failure_analysis.py`: failure taxonomy and qualitative examples.
- `09_train_grpo_lora_qwen_vl.py`: minimal offline GRPO-style LoRA training.
- `10_build_autonomous_sft_data.py`: full-image autonomous SFT examples with `<locate>`.
- `11_eval_autonomous_locate.py`: bbox IoU and center-hit diagnostics for `<locate>`.

Matrix scripts:

- `run_sft_matrix_7b.sh`
- `run_eval_sft_matrix_7b.sh`
- `run_rl_matrix_7b.sh`
- `run_eval_rl_matrix_7b.sh`

## Training

Backbone:

- `/data/zhengxiang/code/dagig/models/Qwen2.5-VL-7B-Instruct`

Environment:

- `conda run -n dagig-sft`
- CUDA visible in this environment: 8 x A100.

SFT:

- LoRA rank 32, alpha 64.
- 2 epochs.
- Five SFT variants trained in parallel on GPUs 0-4.
- Checkpoints: `checkpoints/dagig_train/*_sft_7b_lora`

Final SFT train losses:

| Variant | Train loss |
| --- | ---: |
| uniform_sft | 1.127 |
| outcome_only_sft | 0.418 |
| local_ig_sft | 1.172 |
| dagig_sft | 1.112 |
| dagig_action_only_sft | 0.6026 |

RL:

- Initial adapter: `checkpoints/dagig_train/dagig_sft_7b_lora`.
- Reward modes: outcome-only, outcome + search penalty, text-IG, DAG-IG.
- Low-temperature fixed run: 20 steps, rollout_n=2, limit=64, temperature=0.2.
- Checkpoints: `checkpoints/dagig_train/rl_*_lowtemp_7b_lora`

RL rollout reward diagnostics:

| Reward mode | Avg rollout reward | Nonzero rollouts | Steps with reward variance |
| --- | ---: | ---: | ---: |
| outcome_only | 0.0750 | 3/40 | 1/20 |
| outcome_plus_search_penalty | 0.0950 | 4/40 | 2/20 |
| text_ig | 0.4432 | 40/40 | 12/20 |
| dagig | 0.4690 | 40/40 | 17/20 |

An earlier high-temperature RL attempt was retained as a diagnostic under `checkpoints/dagig_train/rl_*_7b_lora`; it produced mostly invalid rollouts and zero advantage. The low-temperature run fixed this.

Autonomous full-image SFT:

- Data: `data/dagig_autonomous_sft/autonomous_dagig_sft_{train,dev,test}.jsonl`
- Input mode: full image only, with `<locate>` before observe/search/evidence/answer.
- LoRA rank 32, alpha 64.
- 1 epoch.
- Checkpoint: `checkpoints/dagig_train/autonomous_dagig_sft_7b_lora`
- Final train loss: 1.258.

## Dev Results

Primary dev set: 132 Pix2Fact-DAGIG oracle-crop examples.

### SFT

| Method | Answer EM | Answer F1 | Query anchor hit | Evidence R@1 | Evidence R@5 | Evidence MRR | Spurious success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform_sft | 0.0682 | 0.2108 | 0.7879 | 0.3712 | 0.5682 | 0.4590 | 0.0379 |
| outcome_only_sft | 0.0682 | 0.1849 | 0.7879 | 0.3561 | 0.5379 | 0.4416 | 0.0530 |
| local_ig_sft | 0.0530 | 0.1888 | 0.8106 | 0.3864 | 0.5758 | 0.4690 | 0.0303 |
| dagig_sft | 0.0758 | 0.2067 | 0.8258 | 0.3788 | 0.5758 | 0.4686 | 0.0303 |
| dagig_action_only_sft | 0.0682 | 0.1863 | 0.8258 | 0.3939 | 0.5758 | 0.4723 | 0.0303 |

### RL Low-Temperature

| Method | Answer EM | Answer F1 | Query anchor hit | Evidence R@1 | Evidence R@5 | Evidence MRR | Spurious success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rl_outcome_only_lowtemp | 0.0758 | 0.2141 | 0.8182 | 0.3939 | 0.5985 | 0.4832 | 0.0379 |
| rl_outcome_plus_search_penalty_lowtemp | 0.0606 | 0.1960 | 0.8182 | 0.4015 | 0.5909 | 0.4848 | 0.0227 |
| rl_text_ig_lowtemp | 0.0758 | 0.2161 | 0.8030 | 0.3939 | 0.5909 | 0.4828 | 0.0379 |
| rl_dagig_lowtemp | 0.0606 | 0.2018 | 0.8106 | 0.3939 | 0.5985 | 0.4848 | 0.0227 |

Observation:

- DAG-IG RL and outcome-only RL tie on Evidence R@5 at 0.5985 in this small-budget run.
- DAG-IG RL improves spurious success rate over outcome-only RL: 0.0227 vs 0.0379.
- DAG-IG RL has the strongest process-reward training signal among RL modes: highest average rollout reward and most steps with group variance.

### Autonomous Full-Image SFT

This is a pilot full-image setting without oracle crops. The model emits a valid `<locate>` box almost every time, but the boxes do not overlap the target objects.

| Method | Answer EM | Answer F1 | Query anchor hit | Evidence R@1 | Evidence R@5 | Evidence MRR | Valid format | Spurious success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| autonomous_dagig_sft | 0.0455 | 0.1524 | 0.3258 | 0.2652 | 0.3712 | 0.3189 | 0.9848 | 0.0455 |

Locate diagnostics:

| Method | Valid bbox | Raw pixel mean IoU | Qwen 0-1000 mean IoU | Input-pixel mean IoU | Best interpreted mean IoU | Success@IoU0.3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| autonomous_dagig_sft | 1.0000 | 0.0000 | 0.0010 | 0.0011 | 0.0017 | 0.0000 |

Observation:

- The poor locate score is not only a coordinate-scale mismatch: raw original-pixel, Qwen-style 0-1000 normalized, and resized-input-pixel interpretations all have 0.0 success at IoU 0.3.
- Because localization fails, full-image query anchoring and answer accuracy are substantially below the oracle-crop SFT/RL runs.

## Outputs

Key output locations:

- SFT checkpoints: `checkpoints/dagig_train/*_sft_7b_lora`
- RL checkpoints: `checkpoints/dagig_train/rl_*_lowtemp_7b_lora`
- Autonomous checkpoint: `checkpoints/dagig_train/autonomous_dagig_sft_7b_lora`
- SFT/RL eval summaries: `results/dagig_train/*_eval.csv`
- Reward summaries: `results/dagig_train/*_reward_summary.csv`
- Query retrieval summaries: `results/dagig_train/*_query_retrieval.csv`
- Autonomous locate summary: `results/dagig_train/autonomous_dagig_sft_7b_locate.csv`
- Combined table: `results/dagig_train/tables/all_results.md`
- Failure summary: `results/dagig_train/failure_analysis/failure_summary.csv`
- Qualitative examples: `results/dagig_train/failure_analysis/qualitative_examples.md`

## Limitations

- The primary SFT/RL comparison is oracle-crop controlled. It isolates observe/query/evidence/answer behavior from localization.
- A full-image autonomous SFT pilot was trained and evaluated, but it did not learn reliable pixel localization. Scaling this setting likely needs stronger locate supervision, coordinate-convention alignment, or a dedicated grounding stage.
- RL was run with a small budget for feasibility: 20 steps and rollout_n=2. The scripts support scaling this budget.
- Live search was not used. All search metrics use the fixed offline Pix2Fact evidence corpus.
- Public benchmark evaluation for MMSearch, MMSearch-Plus, WebQA, and InfoSeek is scaffolded conceptually by the pipeline but was not executed in this run.

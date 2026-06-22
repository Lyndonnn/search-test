# RN03-10 Pix2Fact-DAGIG Experiment Report

Date: 2026-06-20

This run repeats the DAG-IG pipeline on the newly provided Google Drive package:

- Downloaded package: `data/new_drive_package/drive_file`
- Extracted package: `data/pix2fact_dagig_rn03_10_paper_audited_package`
- Main file: `data/pix2fact_dagig_rn03_10_paper_audited_package/data/pix2fact_dagig_rn03_10_paper_audited_train_AB_clean_split.jsonl`
- Manifest: `data/pix2fact_dagig_rn03_10_paper_audited_package/MANIFEST_PAPER_AUDITED_RN03_10.json`
- Codebase: `/storage/zhengxiang/search-test`
- The old `/storage/zhengxiang/code/00-agentic-rl-search/dagig_r1` codebase was not used as source code.

## Data

The new package is a region-normalized RN03-10 dataset. The audited manifest reports:

- Accepted rows: 781
- Rejected rows: 99
- Split: train 589, dev 113, test 79
- Tiers: A 549, B 232
- Bbox format: `qwen_0_1000_xyxy_on_region_normalized_image`
- Data rule: raw box count exactly 1, RN03-10 geometry, GPT/VLM semantic audit passed

Sanity check outputs:

- `results/dagig_rn03_10/sanity/dataset_summary.json`
- `results/dagig_rn03_10/sanity/random_examples.md`
- `results/dagig_rn03_10/sanity/contact_sheet.jpg`

Retrieval corpus:

- Path: `data/dagig_rn03_10_retrieval`
- Samples: 781
- Docs: 1028
- Supporting docs: 827
- Samples without support targets: 0

## Implemented Pipeline Updates

RN03-10 differs from the previous package because it has one region-normalized image input and no separate `crop_model_input`. I updated the repo scripts accordingly:

- `scripts/dagig_train/00_inspect_package.py`: manifest fallback and configurable image keys.
- `scripts/dagig_train/01_build_sft_data.py`: supports one-image RN examples without oracle crop.
- `scripts/dagig_train/10_build_autonomous_sft_data.py`: uses RN qwen-0-1000 locate boxes.
- `scripts/dagig_train/run_sft_matrix_7b.sh`: parameterized data/output/log roots.
- `scripts/dagig_train/run_eval_sft_matrix_7b.sh`: parameterized package/main/retrieval paths.
- `scripts/dagig_train/run_rl_matrix_7b.sh`: parameterized train file and retrieval corpus used by RL rewards.
- `scripts/dagig_train/run_eval_rl_matrix_7b.sh`: parameterized package/main/retrieval paths.

## Training Artifacts

Cold-start SFT adapters:

- `checkpoints/dagig_rn03_10/uniform_sft_7b_lora`
- `checkpoints/dagig_rn03_10/outcome_only_sft_7b_lora`
- `checkpoints/dagig_rn03_10/local_ig_sft_7b_lora`
- `checkpoints/dagig_rn03_10/dagig_sft_7b_lora`
- `checkpoints/dagig_rn03_10/dagig_action_only_sft_7b_lora`

Low-budget RL adapters, all initialized from `dagig_sft_7b_lora` with 20 steps, 64 samples, and 2 rollouts:

- `checkpoints/dagig_rn03_10/rl_outcome_only_lowtemp_7b_lora`
- `checkpoints/dagig_rn03_10/rl_outcome_plus_search_penalty_lowtemp_7b_lora`
- `checkpoints/dagig_rn03_10/rl_generic_process_lowtemp_7b_lora`
- `checkpoints/dagig_rn03_10/rl_text_ig_lowtemp_7b_lora`
- `checkpoints/dagig_rn03_10/rl_dagig_lowtemp_7b_lora`

Autonomous RN adapter:

- `checkpoints/dagig_rn03_10/autonomous_rn03_10_dagig_sft_7b_lora`
- Final train loss: 1.203

## Controlled SFT Dev Results

These rows use the RN image input and the non-autonomous chain format. Retrieval columns are from the standalone query-retrieval evaluator.

| Method | EM | F1 | Anchor hit | Query specificity | R@1 | R@5 | MRR | Unsupported |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uniform_sft | 3.54 | 17.24 | 78.76 | 92.61 | 41.59 | 58.41 | 49.45 | 1.77 |
| outcome_only_sft | 4.42 | 16.99 | 76.99 | 83.64 | 40.71 | 56.64 | 48.85 | 0.88 |
| local_ig_sft | 5.31 | 17.55 | 71.68 | 77.25 | 43.36 | 53.98 | 48.85 | 2.65 |
| dagig_sft | 6.19 | 19.45 | 78.76 | 91.57 | 42.48 | 56.64 | 49.49 | 2.65 |
| dagig_action_only_sft | 5.31 | 17.12 | 73.45 | 89.46 | 40.71 | 54.87 | 48.48 | 0.88 |

Main controlled-SFT observation: `dagig_sft` gives the best answer EM/F1 among the five SFT variants, while `uniform_sft` has stronger standalone R@5. `dagig_sft` keeps anchor hit high while improving answer quality over outcome-only.

## Low-Budget RL Dev Results

These rows are reward-summary metrics from the offline corpus evaluator. This is a small diagnostic RL budget, not a final convergence run.

| Method | EM | F1 | Anchor hit | R@1 | R@5 | MRR | Unsupported | Spurious | DAG-IG reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dagig_sft init | 6.19 | 19.45 | 78.76 | 43.36 | 61.06 | 50.90 | 0.88 | 4.42 | 45.44 |
| outcome_only_rl | 6.19 | 19.24 | 80.53 | 44.25 | 61.95 | 51.85 | 0.00 | 4.42 | 46.16 |
| search_penalty_rl | 6.19 | 21.79 | 79.65 | 43.36 | 61.95 | 51.24 | 0.88 | 4.42 | 46.52 |
| generic_process_rl | 5.31 | 18.35 | 79.65 | 43.36 | 60.18 | 50.96 | 0.88 | 3.54 | 45.14 |
| text_ig_rl | 7.08 | 20.88 | 79.65 | 43.36 | 61.06 | 51.58 | 0.88 | 5.31 | 45.88 |
| dagig_rl | 5.31 | 20.63 | 78.76 | 43.36 | 61.06 | 51.18 | 0.88 | 3.54 | 46.01 |

Main RL observation: at this very small 20-step budget, DAG-IG RL is not a decisive winner on answer/retrieval. It does reduce spurious success relative to outcome-only, search-penalty, and text-IG, and improves F1 over the SFT initializer, but search-penalty has the highest scalar DAG-IG reward and text-IG has the highest EM/F1. This should be treated as a smoke-scale RL result, not final evidence.

## Autonomous RN Dev Results

Autonomous input uses the RN image plus question and predicts:

`<locate> -> <observe> -> <search_decision> -> <search> -> <evidence> -> <answer>`

Answer/query/retrieval:

- EM: 7.08
- F1: 17.11
- Query anchor hit: 78.76
- Query specificity: 92.84
- Reward-scored retrieval: R@1 44.25, R@5 65.49, MRR 53.36
- Standalone query retrieval: R@1 42.48, R@5 61.06, MRR 51.58
- Unsupported answer: 0.88 by reward scorer, 1.77 by chain evaluator
- Spurious success: 2.65
- Search-call rate: 100.00

Locate:

- Valid bbox rate: 100.00
- Mean IoU, qwen-0-1000 interpretation: 3.58
- Center-hit rate: 13.27
- IoU >= 0.3 success: 2.65
- Best-interpretation mean IoU: 7.50
- Best-interpretation IoU >= 0.3 success: 6.19

Main autonomous observation: the RN data improves the downstream query/retrieval setting, with the autonomous chain reaching the best reward-scored retrieval R@5 in this run. However, locate itself is still weak. The model emits valid boxes, but the boxes usually do not overlap the gold target well.

## Failure Analysis

Failure summary is in:

- `results/dagig_rn03_10/failure_analysis/failure_summary.csv`
- `results/dagig_rn03_10/failure_analysis/qualitative_examples.md`

Selected rows:

| Method | OK | Answer reasoning failure | Observation failure | Query-anchor failure | Retrieval failure | Spurious/unsupported |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| autonomous | 7.96 | 37.17 | 23.89 | 13.27 | 16.81 | 0.88 |
| dagig_sft | 8.85 | 45.13 | 0.88 | 18.58 | 23.01 | 2.65 |
| outcome_only_rl | 8.85 | 46.02 | 0.88 | 16.81 | 24.78 | 2.65 |
| search_penalty_rl | 10.62 | 44.25 | 0.88 | 18.58 | 22.12 | 3.54 |
| generic_process_rl | 9.73 | 43.36 | 0.88 | 17.70 | 23.89 | 1.77 |
| text_ig_rl | 8.85 | 45.13 | 0.88 | 17.70 | 23.89 | 3.54 |
| dagig_rl | 10.62 | 43.36 | 1.77 | 17.70 | 23.01 | 2.65 |

## Conclusions

1. The new RN03-10 package is usable and cleaner for the visual target-size objective: all accepted examples are final audited rows with one raw box and target area in the 3-10 percent range.
2. Controlled SFT still supports the credit-assignment story: `dagig_sft` improves answer EM/F1 over outcome-only and uniform training while keeping query anchor hit high.
3. The autonomous RN chain gives the strongest retrieval R@5, but localization remains the main bottleneck. Valid boxes are produced, but mean IoU is only 3.58 and IoU>=0.3 success is 2.65.
4. Low-budget RL is only partially informative. DAG-IG reduces spurious success versus several baselines, but does not dominate answer accuracy or retrieval at 20 steps. More RL budget and seeds are needed before claiming a reward-design win.
5. The next data-side improvement should target locate supervision and image construction: make the target occupy a reliably visible fraction, preserve high-frequency text/logo details, and include harder negative/context cases where the model must choose the correct nearby object rather than copying generic boxes.

## Not Run In This Pass

- Public benchmarks: MMSearch, MMSearch-Plus, WebQA, InfoSeek.
- Cached/live web search.
- Multi-seed RL.
- Full-budget autonomous RL with actual crop-execute environment.
- Direct VLM and RAG baselines.


<h1 align="center">Multimodal-Search-R1: Incentivizing LMMs to Search</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2506.20670">Paper</a> ｜ 
  <a href="https://www.lmms-lab.com/posts/mmsearch_r1">Blog</a> ｜
  <a href="https://huggingface.co/lmms-lab/MMSearch-R1-7B">Model</a> ｜
  <a href="https://huggingface.co/datasets/lmms-lab/FVQA">Data</a>
</p>

## Overview
<p align="center">
  <img src="assets/mmsearch_r1_overview.png" alt="Overview of MMSearch-R1" width="800">
</p>

**MMSearch-R1** is an end-to-end RL framework that enables LMMs to perform on-demand, multi-turn search with real-world multimodal search tools.

## News
- [25.07.30] [MMSearch-R1-7B](https://huggingface.co/lmms-lab/MMSearch-R1-7B) Model and [FactualVQA(FVQA)](https://huggingface.co/datasets/lmms-lab/FVQA) Dataset **(including all cached image search results)** now released on [Huggingface](https://huggingface.co/collections/lmms-lab/mmsearch-r1-6889e975d8651ce2554b1b3e)!
- [25.06.26] Paper released on [ArXiv](https://arxiv.org/abs/2506.20670) and [Huggingface](https://huggingface.co/papers/2506.20670)!
- [25.06.18] [Blog](https://www.lmms-lab.com/posts/mmsearch_r1) and code are updated!

## Table of Content
- [Installation](#installation)
- [Multimodal Search Tool Implementation](#multimodal-search-tool-implemention)
- [Data Construction](#data-construction)
- [Train & Eval](#train--eval)

## Installation
Quickstart (Colab or local):
```bash
git clone https://github.com/Lyndonnn/search-test.git
cd search-test
make mmsearch_setup_baseline
make mmsearch_cuda_preflight
bash scripts/run_m0_sanity.sh
```

The baseline setup deliberately does not install `./verl`. The original
MMSearch-R1 repository pins veRL commit
`8e9e73723fd1cc729bedb3bbcf915060afbda91d`; this fork checks that commit out
under `third_party/mmsearch_r1_verl/` and installs a locked compatibility stack
inside `.venv-mmsearch-r1`.

For Google Colab:
```bash
PREPARE_FVQA_DEBUG=1 FVQA_TRAIN_LIMIT=16 FVQA_TEST_LIMIT=8 \
  bash scripts/bootstrap_colab.sh
```
This installs the isolated environment, checks CUDA/BF16/NCCL/vLLM, and prepares
streamed FVQA debug parquets. `INSTALL_FLASH_ATTN=1` is optional after the
locked eager-attention baseline is stable.

If you want a one-shot Colab setup for the current research stage, use:
```bash
from google.colab import drive
drive.mount('/content/drive')
```
```bash
%cd /content
!rm -rf search-test
!git clone https://github.com/Lyndonnn/search-test.git
%cd search-test
!PREPARE_FVQA_DEBUG=1 FVQA_TRAIN_LIMIT=16 FVQA_TEST_LIMIT=8 bash scripts/bootstrap_colab.sh
```
This creates `.venv-mmsearch-r1`, installs the original pinned veRL checkout,
and prepares `mmsearch_r1/data/fvqa_debug_train.pq` plus
`mmsearch_r1/data/fvqa_debug_val.pq`.

The repository includes `mmsearch_r1/data/mini_data.pq` for sanity/debug runs.
Larger parquet datasets are expected to be prepared locally in veRL format and are not
bundled with this repo.

Locked baseline versions are recorded in `requirements-mmsearch-r1.txt`.

## Multimodal Search Tool Implemention
We draw inspiration from open-sourced implementation [OpenDeepResearcher](https://github.com/mshumer/OpenDeepResearcher/blob/main/open_deep_researcher.ipynb), which integrates [SerpApi](https://serpapi.com/), [JINA Reader](https://jina.ai/reader/), and LLM-based summarization to retrieve and condense web content relevant to a given question. Currently, MMSearch-R1 includes two types of search tools: an image search tool and a text search tool.
- **Image Search Tool:** This tool is built solely on SerpAPI. The model provides the image (via URL or other form) to the tool, which is responsible for retrieving the top-k visually relevant web pages. The tool returns a sequence of interleaved thumbnails and titles extracted from those pages.
- **Text Search Tool:** This tool combines SerpAPI, JINA Reader, and Qwen3-32B for summarization. The model submits a text query, and SerpAPI retrieves the top-k relevant web page URLs. JINA Reader parses and cleans the content of those pages, and Qwen3-32B generates summaries based on the original query. The tool ultimately returns a list of summarized passages from the top-k relevant webpages with their respective links.

⚠️⚠️⚠️ Before initiating formal training, you are expected to build your own search tool pipeline under the `mmsearch_r1/utils/tools/` directory and invoke it appropriately during the multi-turn rollout process.

## Data Construction
Both the training and validation datasets follow the format defined by veRL. We provide an example dataset under directory `mmsearch_r1/data` as a reference to help you prepare your own training data.

Important:
- `mmsearch_r1/data/mini_data.pq` is included for sanity checks
- larger parquet files should be generated or downloaded locally
- this repo intentionally does not ship the previous 2k parquet pointer because it broke fresh clones

Prepare a small streamed FVQA debug/val parquet first:
```bash
FVQA_TRAIN_LIMIT=16 FVQA_VAL_LIMIT=8 make mmsearch_prepare_fvqa_debug
```

If you are using Colab with the isolated virtualenv, avoid importing `pandas` in the notebook kernel directly. Instead, run helper scripts through the virtualenv:
```bash
source .venv-mmsearch-r1/bin/activate && python scripts/export_verl_sample.py --parquet mmsearch_r1/data/fvqa_debug_val.pq --index 0 --output-dir outputs/fvqa_demo
```

The generated parquet follows the veRL-style object columns used by MMSearch-R1:
- `prompt`: one-turn chat list, e.g. `[{"role": "user", "content": "question text"}]`
- `images`: list of image payload dicts, e.g. `[{"bytes": ...}]`
- `reward_model`: dict with `ground_truth`, optional `candidate_answers`, and `style`
- `data_source`: should contain `mmsearch_r1` for the current reward dispatcher
- `image_urls`: optional logging field
- `extra_info`: optional metadata, typically including `index` and `question_id`

## Train & Eval
We recommend use the command below for unified training and evaluation:
```bash
bash mmsearch_r1/scripts/run_mmsearch_r1_grpo.sh
```
We highlight the important configurations for training the Multi-Round Search LMMs:
- `actor_rollout_ref.rollout.name`: should be `vllm_multiturn_mmsearch` for multi-turn search rollout;
- `actor_rollout_ref.actor.use_multi_turn_response_mask`: should be `True`, as we use it to refine the original `response_mask` for accurate loss calculation.
- `actor_rollout_ref.rollout.max_gen_round`: The max number of turns during rollout;
- `data.max_response_length`: The max response length for each turn;
- `actor_rollout_ref.rollout.response_length_total`: The max conversation length for all turns (except the user prompt in the first turn);

For evaluation only, configure these parameters in the above script:
```bash
...
trainer.val_files=${path_to_val_data} \
trainer.val_only=True \
trainer.val_only_save_dir=${path_to_save_dir} \
trainer.val_generations_to_log_to_wandb=64 # num of val generations to log, this should be larger than the size of val dataset for complete saving
```
The model's responses will be saved in JSON format under `${path_to_save_dir}`, which can be used for subsequent analysis and evaluation.

### Mainline Reproduction Path For This Fork

The fastest useful path is to reproduce the MMSearch-R1 pipeline first, then replace the reward manager with DAG-IG. Do not treat the DAG-IG toy diagnostics as the paper baseline.

1. Install and verify the original locked baseline environment:
```bash
make mmsearch_setup_baseline
make mmsearch_cuda_preflight
make mmsearch_check_overrides
```

2. Prepare FVQA in the veRL/MMSearch-R1 parquet format:
```bash
make mmsearch_prepare_fvqa_debug
```

3. Run validation-only multi-turn MMSearch-R1 search rollout on one A100/A800:
```bash
make mmsearch_val_only
```
This uses `Qwen/Qwen2.5-VL-3B-Instruct`, `vllm_multiturn_mmsearch`, `use_multi_turn_response_mask=True`, and saves generations under `results/mmsearch_r1/val_only_a100_debug`. If `SERPAPI_API_KEY` is not set, it automatically uses `MMSEARCH_OFFLINE_PARQUET` for offline FVQA retrieval.

4. Run a small GRPO baseline through the same official training stack:
```bash
make mmsearch_grpo_a100_debug
```
This is the first training-loop target to compare against DAG-IG. The default is intentionally small: 3B, one GPU, `rollout.n=2`, `max_gen_round=2`, and `total_training_steps=20`.

5. Scale to the original baseline only after the debug path is stable:
```bash
MMSEARCH_MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct \
N_GPUS=4 \
TRAIN_BATCH_SIZE=8 \
ROLLOUT_N=4 \
MAX_GEN_ROUND=3 \
TOTAL_TRAINING_STEPS=1000 \
bash mmsearch_r1/scripts/run_mmsearch_r1_grpo_a100_debug.sh
```

For real web/image search, export `SERPAPI_API_KEY`. Without it, the scripts use offline FVQA retrieval only, which is useful for pipeline debugging but is not the final paper setting.

The DAG-IG integration point is `mmsearch_r1/workers/multimodal/reward/mmsearch_r1.py`: MMSearch-R1 currently places an outcome reward on the final valid response token; DAG-IG should replace/augment this with action-span rewards from the multi-turn response mask and tool observations while keeping the same data, rollout, and GRPO training loop.

## ToDo
- [x] Model and Datasets
- [x] Inference script example

## Acknowledgement
We sincerely thank these repositories for providing helpful open-source resources: [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [veRL](https://github.com/volcengine/verl), [OpenDeepResearcher](https://github.com/mshumer/OpenDeepResearcher), [cfpark00/verl](https://github.com/cfpark00/verl/tree/multi_turn_rollout), [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [MMSearch](https://github.com/CaraJ7/MMSearch).

## Citation
```
@article{wu2025mmsearch,
  title={MMSearch-R1: Incentivizing LMMs to Search},
  author={Wu, Jinming and Deng, Zihao and Li, Wei and Liu, Yiding and You, Bo and Li, Bo and Ma, Zejun and Liu, Ziwei},
  journal={arXiv preprint arXiv:2506.20670},
  year={2025}
}
```

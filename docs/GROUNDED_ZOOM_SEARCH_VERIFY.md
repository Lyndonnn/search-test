**Grounded Zoom-Search-Verify**

This document repositions the project from a crop-as-preprocessing mindset to an explicit local-evidence agent design:

`question -> propose_regions -> zoom_region -> local_probe -> search -> verify -> answer`

**Repo Audit**

Current local repo:
- Inference entry: [mmsearch_r1/scripts/inference_torch_demo.py](/Users/lyndon/Desktop/search-test/mmsearch_r1/scripts/inference_torch_demo.py)
  - `load_model_and_processor(...)`
  - `generate_response(...)`
  - `run_mmsearch_demo(...)`
- Search tools:
  - [mmsearch_r1/utils/tools/image_search.py](/Users/lyndon/Desktop/search-test/mmsearch_r1/utils/tools/image_search.py)
  - [mmsearch_r1/utils/tools/text_search.py](/Users/lyndon/Desktop/search-test/mmsearch_r1/utils/tools/text_search.py)
  - [mmsearch_r1/utils/tools/offline_search.py](/Users/lyndon/Desktop/search-test/mmsearch_r1/utils/tools/offline_search.py)
- Evaluation utilities:
  - [scripts/eval_fvqa_debug.py](/Users/lyndon/Desktop/search-test/scripts/eval_fvqa_debug.py)
  - [scripts/eval_pix2fact_local_evidence.py](/Users/lyndon/Desktop/search-test/scripts/eval_pix2fact_local_evidence.py)
- Training entry:
  - [mmsearch_r1/trainer/multimodal/main_ppo.py](/Users/lyndon/Desktop/search-test/mmsearch_r1/trainer/multimodal/main_ppo.py)
  - [mmsearch_r1/trainer/multimodal/ray_trainer.py](/Users/lyndon/Desktop/search-test/mmsearch_r1/trainer/multimodal/ray_trainer.py)

Public Vision-DeepResearch audit from its README and tree:
- Inference/eval entry sits under `rllm/eval/run_eval.sh` and `rllm/eval/eval_runner.py`
- RL workflow sits under `rllm/vision_deepresearch_async_workflow/`
- Data preparation scripts live under `rllm/vision_deepresearch_async_workflow/data_prepare/`
- SFT launch scripts live under `ms-swift/run/`
- The reusable ideas for this repo are:
  - long-horizon asynchronous tool workflow
  - explicit visual->text bridge
  - multi-scale crop + search traces
- The pieces that should remain local additions here are:
  - explicit zoom action
  - ROI-level evidence cache and trace schema
  - OCR/caption/search routing before final verification

**Proposed Architecture**

New inference-time controller:
- Module: [mmsearch_r1/agents/grounded_zoom_search_verify.py](/Users/lyndon/Desktop/search-test/mmsearch_r1/agents/grounded_zoom_search_verify.py)
- State:
  - question
  - original image
  - region history
  - current candidate regions
  - local evidence cache
  - search results cache
  - webpage evidence cache
  - tool budget usage
- Current action implementation status:
  - implemented:
    - `propose_regions`
    - `zoom_region`
    - `ocr_region`
    - `caption_region`
    - `crop_image_search`
    - `text_search`
    - `verify_evidence`
    - `answer`
  - reserved for later:
    - `visit_webpage`
    - `summarize_webpage`

**First Patch Set**

- Add a new agent package:
  - [mmsearch_r1/agents/__init__.py](/Users/lyndon/Desktop/search-test/mmsearch_r1/agents/__init__.py)
  - [mmsearch_r1/agents/grounded_zoom_search_verify.py](/Users/lyndon/Desktop/search-test/mmsearch_r1/agents/grounded_zoom_search_verify.py)
- Add a runnable CLI:
  - [scripts/run_grounded_zoom_search_verify.py](/Users/lyndon/Desktop/search-test/scripts/run_grounded_zoom_search_verify.py)
- Keep training untouched for now.
- Keep all current eval/inference code backward compatible.

**How To Run**

Single image:
```bash
source .venv-colab/bin/activate && python scripts/run_grounded_zoom_search_verify.py \
  --model-path lmms-lab/MMSearch-R1-7B \
  --image /path/to/image.png \
  --question "Your question" \
  --output-json outputs/gzsv/result.json \
  --trace-jsonl outputs/gzsv/trace.jsonl
```

Parquet sample:
```bash
source .venv-colab/bin/activate && python scripts/run_grounded_zoom_search_verify.py \
  --model-path lmms-lab/MMSearch-R1-7B \
  --parquet /path/to/pix2fact_week1_clean.parquet \
  --index 0 \
  --image-dir /path/to/raw_images/pix2fact \
  --output-json outputs/gzsv/sample0.json \
  --trace-jsonl outputs/gzsv/sample0.jsonl
```

**Suggested Next Experiments**

1. Whole image vs oracle crop vs grounded zoom-search-verify on 20 Pix2Fact samples.
2. Split by `visual_perception_type` to see which categories benefit from local probing.
3. Measure:
   - region proposal quality
   - clue extraction quality
   - query usefulness
   - final answer quality
4. Once traces are stable, add a trajectory schema for SFT/RL using the JSONL events.

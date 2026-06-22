#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ZIP_PATH="${ZIP_PATH:-pix2fact_dagig_rn03_10_ground_expr_gpt54_v3_FULL_WITH_IMAGES.zip}"
DRIVE_ID="${DRIVE_ID:-11bXtyZmxEb9LFAC2Q-fwdr5TlMjfLHF4}"
PACKAGE_ROOT="${PACKAGE_ROOT:-data/dagig_rn03_10_ground_expr_v3_full}"
RESULT_ROOT="${RESULT_ROOT:-results/dagig_rn03_10_grounded}"
CONDA_ENV="${CONDA_ENV:-dagig-sft}"
GPU="${GPU:-0}"
BOX_THRESHOLD="${BOX_THRESHOLD:-0.10}"
TEXT_THRESHOLD="${TEXT_THRESHOLD:-0.10}"
MODEL="${MODEL:-/data/zhengxiang/code/dagig/models/Qwen2.5-VL-7B-Instruct}"

run() {
  echo
  echo "== $* =="
  "$@"
}

if [ ! -f "$ZIP_PATH" ]; then
  run gdown "$DRIVE_ID" -O "$ZIP_PATH"
fi

if [ ! -d "$PACKAGE_ROOT" ]; then
  mkdir -p "$PACKAGE_ROOT"
  run unzip -q "$ZIP_PATH" -d "$PACKAGE_ROOT"
fi

run conda run -n "$CONDA_ENV" python scripts/dagig_train/16_validate_ground_expr_full_package.py \
  --package_root "$PACKAGE_ROOT" \
  --out_dir "$RESULT_ROOT"

run env CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$CONDA_ENV" python scripts/dagig_train/17_groundingdino_eval_ground_expr.py \
  --input_jsonl "$PACKAGE_ROOT/data/pix2fact_dagig_rn03_10_ground_expr_gpt54_v3_dev.jsonl" \
  --package_root "$PACKAGE_ROOT" \
  --out_dir "$RESULT_ROOT/grounding/smoke_dev_10" \
  --box_threshold 0.25 \
  --text_threshold 0.25 \
  --device cuda \
  --max_rows 10

run env CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$CONDA_ENV" python scripts/dagig_train/18_groundingdino_threshold_grid.py \
  --input_jsonl "$PACKAGE_ROOT/data/pix2fact_dagig_rn03_10_ground_expr_gpt54_v3_dev.jsonl" \
  --package_root "$PACKAGE_ROOT" \
  --out_dir "$RESULT_ROOT/grounding" \
  --device cuda

for split in train dev test; do
  run env CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$CONDA_ENV" python scripts/dagig_train/17_groundingdino_eval_ground_expr.py \
    --input_jsonl "$PACKAGE_ROOT/data/pix2fact_dagig_rn03_10_ground_expr_gpt54_v3_${split}.jsonl" \
    --package_root "$PACKAGE_ROOT" \
    --out_dir "$RESULT_ROOT/grounding/final_${split}" \
    --box_threshold "$BOX_THRESHOLD" \
    --text_threshold "$TEXT_THRESHOLD" \
    --device cuda
done

run conda run -n "$CONDA_ENV" python scripts/dagig_train/19_make_grounding_contact_sheets.py \
  --dev_results "$RESULT_ROOT/grounding/final_dev/grounding_results.jsonl" \
  --test_results "$RESULT_ROOT/grounding/final_test/grounding_results.jsonl" \
  --out_dir "$RESULT_ROOT/contact_sheets"

run conda run -n "$CONDA_ENV" python scripts/dagig_train/20_build_ground_action_sft_data.py \
  --package_root "$PACKAGE_ROOT" \
  --grounding_train "$RESULT_ROOT/grounding/final_train/grounding_results.jsonl" \
  --grounding_dev "$RESULT_ROOT/grounding/final_dev/grounding_results.jsonl" \
  --grounding_test "$RESULT_ROOT/grounding/final_test/grounding_results.jsonl" \
  --out_dir data/dagig_rn03_10_grounded

run conda run -n "$CONDA_ENV" python scripts/dagig_train/21_validate_ground_action_sft_data.py \
  --data_dir data/dagig_rn03_10_grounded \
  --package_root "$PACKAGE_ROOT" \
  --out_dir "$RESULT_ROOT"

run env CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$CONDA_ENV" python scripts/dagig_train/02_train_lora_qwen_vl.py \
  --train_file data/dagig_rn03_10_grounded/ground_action_train.jsonl \
  --output_dir checkpoints/dagig_rn03_10_grounded/ground_action_sft_7b_lora_sanity \
  --model_name_or_path "$MODEL" \
  --epochs 1 \
  --lr 2e-5 \
  --lora_r 32 \
  --lora_alpha 64 \
  --batch_size 1 \
  --grad_accum 16 \
  --max_length 2048 \
  --max_pixels 262144 \
  --logging_steps 1 \
  --max_steps 10 \
  --save_steps 10 \
  --limit 64

run env CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$CONDA_ENV" python scripts/dagig_train/02_train_lora_qwen_vl.py \
  --train_file data/dagig_rn03_10_grounded/ground_action_train.jsonl \
  --output_dir checkpoints/dagig_rn03_10_grounded/ground_action_sft_7b_lora \
  --model_name_or_path "$MODEL" \
  --epochs 4 \
  --lr 2e-5 \
  --lora_r 32 \
  --lora_alpha 64 \
  --batch_size 1 \
  --grad_accum 16 \
  --max_length 2048 \
  --max_pixels 262144 \
  --logging_steps 5 \
  --save_steps 50

for split in dev test; do
  run env CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$CONDA_ENV" python scripts/dagig_train/23_eval_ground_action_sft.py \
    --eval_file "data/dagig_rn03_10_grounded/ground_action_${split}.jsonl" \
    --output_predictions_jsonl "$RESULT_ROOT/eval/ground_action_sft_${split}_predictions.jsonl" \
    --output_metrics_json "$RESULT_ROOT/eval/ground_action_sft_${split}_metrics.json" \
    --summary_md "$RESULT_ROOT/eval/ground_action_sft_${split}_summary.md" \
    --model_name_or_path "$MODEL" \
    --adapter_dir checkpoints/dagig_rn03_10_grounded/ground_action_sft_7b_lora \
    --package_root "$PACKAGE_ROOT" \
    --box_threshold "$BOX_THRESHOLD" \
    --text_threshold "$TEXT_THRESHOLD" \
    --device cuda \
    --max_new_tokens 384 \
    --require_beats_old_direct_bbox
done

run conda run -n "$CONDA_ENV" python scripts/dagig_train/24_write_grounded_report.py \
  --result_root "$RESULT_ROOT" \
  --checkpoint_dir checkpoints/dagig_rn03_10_grounded/ground_action_sft_7b_lora

echo
echo "Grounded pipeline complete."
echo "Report: $RESULT_ROOT/GROUNDED_EXPERIMENT_REPORT.md"

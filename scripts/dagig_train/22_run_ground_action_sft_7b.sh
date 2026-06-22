#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-/data/zhengxiang/code/dagig/models/Qwen2.5-VL-7B-Instruct}"
CONDA_ENV="${CONDA_ENV:-dagig-sft}"
GPU="${GPU:-0}"
MODE="${MODE:-full}"
TRAIN_FILE="${TRAIN_FILE:-data/dagig_rn03_10_grounded/ground_action_train.jsonl}"
DEV_FILE="${DEV_FILE:-data/dagig_rn03_10_grounded/ground_action_dev.jsonl}"
OUT_DIR="${OUT_DIR:-checkpoints/dagig_rn03_10_grounded/ground_action_sft_7b_lora}"
LOG_FILE="${LOG_FILE:-results/dagig_rn03_10_grounded/train_logs/ground_action_sft_7b.log}"
EPOCHS="${EPOCHS:-4}"
LR="${LR:-2e-5}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_PIXELS="${MAX_PIXELS:-262144}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
MAX_STEPS="${MAX_STEPS:-0}"
LIMIT="${LIMIT:-0}"

if [ "$MODE" = "sanity" ]; then
  OUT_DIR="${OUT_DIR}_sanity"
  LOG_FILE="${LOG_FILE%.log}_sanity.log"
  EPOCHS="${EPOCHS:-1}"
  MAX_STEPS="${MAX_STEPS:-20}"
  LIMIT="${LIMIT:-64}"
fi

if [ ! -f "$TRAIN_FILE" ]; then
  echo "Missing train file: $TRAIN_FILE" >&2
  exit 2
fi
if [ ! -f "$DEV_FILE" ]; then
  echo "Missing dev file: $DEV_FILE" >&2
  exit 2
fi

mkdir -p "$(dirname "$LOG_FILE")" "$OUT_DIR"

echo "mode=$MODE"
echo "model=$MODEL"
echo "train_file=$TRAIN_FILE"
echo "dev_file=$DEV_FILE"
echo "out_dir=$OUT_DIR"
echo "log_file=$LOG_FILE"
echo "gpu=$GPU"

cmd=(
  conda run -n "$CONDA_ENV" python scripts/dagig_train/02_train_lora_qwen_vl.py
  --train_file "$TRAIN_FILE"
  --output_dir "$OUT_DIR"
  --model_name_or_path "$MODEL"
  --epochs "$EPOCHS"
  --lr "$LR"
  --lora_r "$LORA_R"
  --lora_alpha "$LORA_ALPHA"
  --batch_size 1
  --grad_accum "$GRAD_ACCUM"
  --max_length "$MAX_LENGTH"
  --max_pixels "$MAX_PIXELS"
  --logging_steps "$LOGGING_STEPS"
)
if [ "$MAX_STEPS" != "0" ]; then
  cmd+=(--max_steps "$MAX_STEPS" --save_steps "$MAX_STEPS")
fi
if [ "$LIMIT" != "0" ]; then
  cmd+=(--limit "$LIMIT")
fi

if env CUDA_VISIBLE_DEVICES="$GPU" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 "${cmd[@]}" >"$LOG_FILE" 2>&1; then
  echo "training finished: $OUT_DIR"
  echo "log: $LOG_FILE"
else
  status=$?
  echo "training failed with status $status; tail follows:" >&2
  tail -n 120 "$LOG_FILE" >&2 || true
  exit "$status"
fi

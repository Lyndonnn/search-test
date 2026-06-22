#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-/data/zhengxiang/code/dagig/models/Qwen2.5-VL-7B-Instruct}"
CONDA_ENV="${CONDA_ENV:-dagig-sft}"
OUT_ROOT="${OUT_ROOT:-checkpoints/dagig_train}"
LOG_ROOT="${LOG_ROOT:-logs/dagig_train}"
DATA_DIR="${DATA_DIR:-data/dagig_sft}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-2e-5}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_PIXELS="${MAX_PIXELS:-262144}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"

IFS=' ' read -r -a VARIANTS <<< "${VARIANTS:-uniform_sft outcome_only_sft local_ig_sft dagig_sft dagig_action_only_sft}"
IFS=' ' read -r -a GPUS <<< "${GPUS:-0 1 2 3 4}"

if [ "${#GPUS[@]}" -lt "${#VARIANTS[@]}" ]; then
  echo "Need at least as many GPUS as VARIANTS: ${#GPUS[@]} < ${#VARIANTS[@]}" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

pids=()
names=()
for idx in "${!VARIANTS[@]}"; do
  variant="${VARIANTS[$idx]}"
  gpu="${GPUS[$idx]}"
  train_file="${DATA_DIR}/${variant}_train.jsonl"
  out_dir="${OUT_ROOT}/${variant}_7b_lora"
  log_file="${LOG_ROOT}/sft_${variant}_7b.log"
  if [ ! -f "$train_file" ]; then
    echo "Missing train file: $train_file" >&2
    exit 2
  fi
  if [ -f "${out_dir}/adapter_config.json" ]; then
    echo "skip ${variant}: existing ${out_dir}/adapter_config.json"
    continue
  fi
  echo "launch ${variant} on GPU ${gpu}; log=${log_file}"
  (
    env CUDA_VISIBLE_DEVICES="$gpu" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
      conda run -n "$CONDA_ENV" python scripts/dagig_train/02_train_lora_qwen_vl.py \
        --train_file "$train_file" \
        --output_dir "$out_dir" \
        --model_name_or_path "$MODEL" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --lora_r "$LORA_R" \
        --lora_alpha "$LORA_ALPHA" \
        --batch_size 1 \
        --grad_accum "$GRAD_ACCUM" \
        --max_length "$MAX_LENGTH" \
        --max_pixels "$MAX_PIXELS" \
        --logging_steps "$LOGGING_STEPS"
  ) >"$log_file" 2>&1 &
  pids+=("$!")
  names+=("$variant")
done

status=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  name="${names[$idx]}"
  if wait "$pid"; then
    echo "done ${name}"
  else
    echo "failed ${name}; tail follows" >&2
    tail -n 80 "${LOG_ROOT}/sft_${name}_7b.log" >&2 || true
    status=1
  fi
done
exit "$status"

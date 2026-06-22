#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-/data/zhengxiang/code/dagig/models/Qwen2.5-VL-7B-Instruct}"
CONDA_ENV="${CONDA_ENV:-dagig-sft}"
INIT_ADAPTER="${INIT_ADAPTER:-checkpoints/dagig_train/dagig_sft_7b_lora}"
OUT_ROOT="${OUT_ROOT:-checkpoints/dagig_train}"
LOG_ROOT="${LOG_ROOT:-logs/dagig_train}"
TRAIN_FILE="${TRAIN_FILE:-data/dagig_sft/dagig_sft_train.jsonl}"
RETRIEVAL_DIR="${RETRIEVAL_DIR:-data/dagig_retrieval}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
MAX_STEPS="${MAX_STEPS:-50}"
LIMIT="${LIMIT:-128}"
ROLLOUT_N="${ROLLOUT_N:-4}"
LR="${LR:-1e-6}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_PIXELS="${MAX_PIXELS:-262144}"

IFS=' ' read -r -a REWARD_MODES <<< "${REWARD_MODES:-outcome_only outcome_plus_search_penalty text_ig dagig}"
IFS=' ' read -r -a GPUS <<< "${GPUS:-0 1 2 3}"

if [ ! -f "${INIT_ADAPTER}/adapter_config.json" ]; then
  echo "Missing INIT_ADAPTER=${INIT_ADAPTER}/adapter_config.json" >&2
  exit 2
fi
if [ "${#GPUS[@]}" -lt "${#REWARD_MODES[@]}" ]; then
  echo "Need at least as many GPUS as REWARD_MODES: ${#GPUS[@]} < ${#REWARD_MODES[@]}" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

pids=()
names=()
for idx in "${!REWARD_MODES[@]}"; do
  mode="${REWARD_MODES[$idx]}"
  gpu="${GPUS[$idx]}"
  out_dir="${OUT_ROOT}/rl_${mode}${RUN_SUFFIX}_7b_lora"
  log_file="${LOG_ROOT}/rl_${mode}${RUN_SUFFIX}_7b.log"
  if [ -f "${out_dir}/adapter_config.json" ]; then
    echo "skip ${mode}: existing ${out_dir}/adapter_config.json"
    continue
  fi
  echo "launch rl ${mode} on GPU ${gpu}; log=${log_file}"
  (
    env CUDA_VISIBLE_DEVICES="$gpu" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
      conda run -n "$CONDA_ENV" python scripts/dagig_train/09_train_grpo_lora_qwen_vl.py \
        --train_file "$TRAIN_FILE" \
        --output_dir "$out_dir" \
        --model_name_or_path "$MODEL" \
        --init_adapter_dir "$INIT_ADAPTER" \
        --corpus_jsonl "${RETRIEVAL_DIR}/corpus.jsonl" \
        --targets_json "${RETRIEVAL_DIR}/targets.json" \
        --reward_mode "$mode" \
        --rollout_n "$ROLLOUT_N" \
        --max_steps "$MAX_STEPS" \
        --limit "$LIMIT" \
        --lr "$LR" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --max_length "$MAX_LENGTH" \
        --max_pixels "$MAX_PIXELS"
  ) >"$log_file" 2>&1 &
  pids+=("$!")
  names+=("$mode")
done

status=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  name="${names[$idx]}"
  if wait "$pid"; then
    echo "done rl ${name}"
  else
    echo "failed rl ${name}; tail follows" >&2
    tail -n 80 "${LOG_ROOT}/rl_${name}${RUN_SUFFIX}_7b.log" >&2 || true
    status=1
  fi
done
exit "$status"

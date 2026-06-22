#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_ENV="${CONDA_ENV:-dagig-sft}"
TRAIN_FILE="${TRAIN_FILE:-data/dagig_rn03_10_grounded_rl/grounded_rl_train.jsonl}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/data/zhengxiang/code/dagig/models/Qwen2.5-VL-7B-Instruct}"
INIT_ADAPTER="${INIT_ADAPTER:-checkpoints/dagig_rn03_10_grounded/ground_action_sft_7b_lora}"
OUT_ROOT="${OUT_ROOT:-checkpoints/dagig_rn03_10_grounded_rl}"
LOG_ROOT="${LOG_ROOT:-results/dagig_rn03_10_grounded_rl/train_logs}"
LIMIT="${LIMIT:-64}"
MAX_STEPS="${MAX_STEPS:-20}"
ROLLOUT_N="${ROLLOUT_N:-2}"
TEMP="${TEMP:-0.2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
LR="${LR:-1e-6}"
GPUS="${GPUS:-0 1 2 3}"
PARALLEL="${PARALLEL:-1}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

VARIANT_NAMES=(
  "rl_grounded_outcome_only_lowtemp_7b_lora"
  "rl_grounded_outcome_plus_ground_penalty_lowtemp_7b_lora"
  "rl_grounded_generic_process_lowtemp_7b_lora"
  "rl_grounded_dagig_lowtemp_7b_lora"
)
REWARD_MODES=(
  "outcome_only"
  "outcome_plus_ground_penalty"
  "generic_process"
  "dagig_grounded"
)

read -r -a GPU_LIST <<< "$GPUS"

run_variant() {
  local idx="$1"
  local name="${VARIANT_NAMES[$idx]}"
  local mode="${REWARD_MODES[$idx]}"
  local gpu="${GPU_LIST[$((idx % ${#GPU_LIST[@]}))]}"
  local out_dir="$OUT_ROOT/$name"
  local log_file="$LOG_ROOT/${name}.log"
  echo "=== RUN $name mode=$mode gpu=$gpu ==="
  set +e
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    conda run -n "$CONDA_ENV" python scripts/dagig_train/27_train_grounded_rl_lora.py \
      --train_file "$TRAIN_FILE" \
      --output_dir "$out_dir" \
      --model_name_or_path "$MODEL_NAME_OR_PATH" \
      --init_adapter_dir "$INIT_ADAPTER" \
      --reward_mode "$mode" \
      --limit "$LIMIT" \
      --max_steps "$MAX_STEPS" \
      --rollout_n "$ROLLOUT_N" \
      --temperature "$TEMP" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --lr "$LR" \
      --device cuda
  ) 2>&1 | tee "$log_file"
  local status="${PIPESTATUS[0]}"
  set -e
  if [[ "$status" -ne 0 ]]; then
    echo "FAILED $name status=$status log=$log_file" | tee "$OUT_ROOT/${name}.FAILED"
  else
    echo "DONE $name log=$log_file"
  fi
  return "$status"
}

failed=0
if [[ "$PARALLEL" == "1" ]]; then
  pids=()
  for idx in "${!VARIANT_NAMES[@]}"; do
    run_variant "$idx" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
else
  for idx in "${!VARIANT_NAMES[@]}"; do
    if ! run_variant "$idx"; then
      failed=1
    fi
  done
fi

if [[ "$failed" -ne 0 ]]; then
  echo "One or more grounded RL variants failed. Completed variants remain usable for evaluation."
else
  echo "All grounded RL low-budget variants finished."
fi

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_ENV="${CONDA_ENV:-dagig-sft}"
METHOD="${METHOD:-rejection_sft}"
TRAIN_PAIRS="${TRAIN_PAIRS:-data/dagig_rn03_10_counterfactual_dagig/dagig_preference_pairs_train.jsonl}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/data/zhengxiang/code/dagig/models/Qwen2.5-VL-7B-Instruct}"
INIT_ADAPTER="${INIT_ADAPTER:-checkpoints/dagig_rn03_10_grounded/ground_action_sft_7b_lora}"
OUT_DIR="${OUT_DIR:-checkpoints/dagig_rn03_10_counterfactual_dagig/dagig_dpo_7b_lora}"
LOG_DIR="${LOG_DIR:-results/dagig_rn03_10_counterfactual_dagig/train_logs}"
LIMIT="${LIMIT:-0}"
MAX_STEPS="${MAX_STEPS:-100}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-6}"

mkdir -p "$LOG_DIR" "$(dirname "$OUT_DIR")"

if [[ "$METHOD" != "rejection_sft" ]]; then
  echo "METHOD=$METHOD requested, but this runner currently implements the dependency-free rejection_sft path."
  echo "Set METHOD=rejection_sft or replace this script with a TRL DPO trainer once the dependency is pinned."
  exit 2
fi

echo "Running counterfactual DAG-IG rejection SFT"
echo "train pairs: $TRAIN_PAIRS"
echo "init adapter: $INIT_ADAPTER"
echo "output: $OUT_DIR"

conda run -n "$CONDA_ENV" python scripts/dagig_train/34_train_rejection_sft_lora.py \
  --train_pairs "$TRAIN_PAIRS" \
  --output_dir "$OUT_DIR" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --init_adapter_dir "$INIT_ADAPTER" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --limit "$LIMIT" \
  --max_steps "$MAX_STEPS" \
  2>&1 | tee "$LOG_DIR/dagig_rejection_sft_7b.log"

echo "saved: $OUT_DIR"

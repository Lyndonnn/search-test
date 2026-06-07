#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f projects/dagig_mmsearch/scripts/autodl_env.sh ]]; then
  # shellcheck disable=SC1091
  source projects/dagig_mmsearch/scripts/autodl_env.sh
fi

TRAIN_LIMIT="${FVQA_TRAIN_LIMIT:-128}"
VAL_LIMIT="${FVQA_VAL_LIMIT:-128}"
STEPS="${TOTAL_TRAINING_STEPS:-50}"
TEST_FREQ="${TEST_FREQ:-25}"
SAVE_FREQ="${SAVE_FREQ:--1}"
RESULT_ROOT="${DAGIG_DATA_ROOT:-$ROOT/data/cache/dagig_local}/results_keep/mmsearch_mainline"
mkdir -p "$RESULT_ROOT"

echo "MMSearch-R1 mainline matrix"
echo "  train_limit=$TRAIN_LIMIT val_limit=$VAL_LIMIT steps=$STEPS test_freq=$TEST_FREQ save_freq=$SAVE_FREQ"
echo "  result_root=$RESULT_ROOT"

echo "[1/4] Preparing FVQA debug parquet."
FVQA_TRAIN_LIMIT="$TRAIN_LIMIT" FVQA_VAL_LIMIT="$VAL_LIMIT" make mmsearch_prepare_fvqa_debug

echo "[2/4] Pretrained val-only baseline."
make mmsearch_val_only
make mmsearch_summarize_val
cp paper_artifacts/tables/mmsearch_val_only_summary.csv "$RESULT_ROOT/pretrained_val.csv"
if ls results/mmsearch_r1/val_only_a100_debug/val_result_*.json >/dev/null 2>&1; then
  cp "$(ls -t results/mmsearch_r1/val_only_a100_debug/val_result_*.json | head -n 1)" "$RESULT_ROOT/pretrained_val_result.json"
fi

echo "[3/4] Outcome-only GRPO baseline."
WANDB_EXP_NAME="outcome_only_fvqa${TRAIN_LIMIT}_${STEPS}step" \
TOTAL_TRAINING_STEPS="$STEPS" \
TEST_FREQ="$TEST_FREQ" \
SAVE_FREQ="$SAVE_FREQ" \
SEARCH_PENALTY="${SEARCH_PENALTY:-0.1}" \
FORMAT_PENALTY="${FORMAT_PENALTY:-0.1}" \
make mmsearch_grpo_a100_debug
make mmsearch_summarize_val
cp paper_artifacts/tables/mmsearch_val_only_summary.csv "$RESULT_ROOT/outcome_only_${STEPS}step.csv"
if ls results/mmsearch_r1/val_only_a100_debug/val_result_*.json >/dev/null 2>&1; then
  cp "$(ls -t results/mmsearch_r1/val_only_a100_debug/val_result_*.json | head -n 1)" "$RESULT_ROOT/outcome_only_${STEPS}step_val_result.json"
fi

echo "[4/4] Outcome-only without search penalty. This isolates whether collapse comes from explicit tool cost."
WANDB_EXP_NAME="outcome_no_search_penalty_fvqa${TRAIN_LIMIT}_${STEPS}step" \
TOTAL_TRAINING_STEPS="$STEPS" \
TEST_FREQ="$TEST_FREQ" \
SAVE_FREQ="$SAVE_FREQ" \
SEARCH_PENALTY="0.0" \
FORMAT_PENALTY="${FORMAT_PENALTY:-0.1}" \
make mmsearch_grpo_a100_debug
make mmsearch_summarize_val
cp paper_artifacts/tables/mmsearch_val_only_summary.csv "$RESULT_ROOT/outcome_no_search_penalty_${STEPS}step.csv"
if ls results/mmsearch_r1/val_only_a100_debug/val_result_*.json >/dev/null 2>&1; then
  cp "$(ls -t results/mmsearch_r1/val_only_a100_debug/val_result_*.json | head -n 1)" "$RESULT_ROOT/outcome_no_search_penalty_${STEPS}step_val_result.json"
fi

echo "Mainline matrix completed. Compact outputs:"
find "$RESULT_ROOT" -maxdepth 1 -type f -print | sort


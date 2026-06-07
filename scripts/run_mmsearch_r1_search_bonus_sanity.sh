#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f projects/dagig_mmsearch/scripts/autodl_env.sh ]]; then
  # shellcheck disable=SC1091
  source projects/dagig_mmsearch/scripts/autodl_env.sh
fi

STEPS="${TOTAL_TRAINING_STEPS:-50}"
TEST_FREQ="${TEST_FREQ:-25}"
SAVE_FREQ="${SAVE_FREQ:--1}"
BONUS="${SEARCH_ACTION_BONUS:-0.2}"
RESULT_ROOT="${DAGIG_DATA_ROOT:-$ROOT/data/cache/dagig_local}/results_keep/mmsearch_mainline"
LOG_DIR="$RESULT_ROOT/logs/search_bonus_sanity_${STEPS}step"

mkdir -p "$RESULT_ROOT" "$LOG_DIR"

echo "Running MMSearch-R1 search-bonus sanity baseline"
echo "  steps=$STEPS test_freq=$TEST_FREQ save_freq=$SAVE_FREQ bonus=$BONUS"
echo "  log_dir=$LOG_DIR"

WANDB_EXP_NAME="search_bonus_sanity_${STEPS}step" \
TOTAL_TRAINING_STEPS="$STEPS" \
TEST_FREQ="$TEST_FREQ" \
SAVE_FREQ="$SAVE_FREQ" \
ROLLOUT_LOG_DIR="$LOG_DIR" \
SEARCH_PENALTY="${SEARCH_PENALTY:-0.0}" \
FORMAT_PENALTY="${FORMAT_PENALTY:-0.1}" \
REWARD_SHAPING_MODE="search_success_shaping" \
SEARCH_ACTION_BONUS="$BONUS" \
SEARCH_ACTION_BONUS_CORRECT_ONLY="${SEARCH_ACTION_BONUS_CORRECT_ONLY:-True}" \
bash mmsearch_r1/scripts/run_mmsearch_r1_grpo_a100_debug.sh

python3 scripts/extract_mmsearch_train_metrics.py \
  --input "$LOG_DIR/final_validation.json" \
  --method "search_bonus_sanity_${STEPS}step" \
  --output-csv "$RESULT_ROOT/search_bonus_sanity_${STEPS}step.csv" \
  --output-json "$RESULT_ROOT/search_bonus_sanity_${STEPS}step_metrics.json"

echo "wrote $RESULT_ROOT/search_bonus_sanity_${STEPS}step.csv"


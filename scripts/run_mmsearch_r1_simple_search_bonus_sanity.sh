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
PROMPT_TAG="normal"
PROMPT_PATH="${USER_PROMPT_ROUND_1:-}"

mkdir -p "$RESULT_ROOT"

if [[ "${DAGIG_USE_SEARCH_REQUIRED_PROMPT:-0}" == "1" && -z "$PROMPT_PATH" ]]; then
  PROMPT_TAG="search_required"
  PROMPT_PATH="mmsearch_r1/prompts/round_1_user_prompt_qwenvl_search_required.pkl"
  python3 scripts/create_mmsearch_search_required_prompts.py --output "$PROMPT_PATH"
fi

LOG_DIR="$RESULT_ROOT/logs/simple_search_bonus_${PROMPT_TAG}_${STEPS}step"
mkdir -p "$LOG_DIR"

echo "Running MMSearch-R1 simple search bonus sanity run"
echo "  prompt=$PROMPT_TAG steps=$STEPS test_freq=$TEST_FREQ bonus=$BONUS"
echo "  log_dir=$LOG_DIR"

env_args=(
  WANDB_EXP_NAME="simple_search_bonus_${PROMPT_TAG}_${STEPS}step"
  TOTAL_TRAINING_STEPS="$STEPS"
  TEST_FREQ="$TEST_FREQ"
  SAVE_FREQ="$SAVE_FREQ"
  ROLLOUT_LOG_DIR="$LOG_DIR"
  SEARCH_PENALTY="${SEARCH_PENALTY:-0.0}"
  FORMAT_PENALTY="${FORMAT_PENALTY:-0.1}"
  REWARD_SHAPING_MODE="search_success_shaping"
  SEARCH_ACTION_BONUS="$BONUS"
  SEARCH_ACTION_BONUS_CORRECT_ONLY="${SEARCH_ACTION_BONUS_CORRECT_ONLY:-False}"
)
if [[ -n "$PROMPT_PATH" ]]; then
  env_args+=(USER_PROMPT_ROUND_1="$PROMPT_PATH")
fi

env "${env_args[@]}" bash mmsearch_r1/scripts/run_mmsearch_r1_grpo_a100_debug.sh

python3 scripts/extract_mmsearch_train_metrics.py \
  --input "$LOG_DIR/final_validation.json" \
  --method "simple_search_bonus_${PROMPT_TAG}_${STEPS}step" \
  --output-csv "$RESULT_ROOT/simple_search_bonus_${PROMPT_TAG}_${STEPS}step.csv" \
  --output-json "$RESULT_ROOT/simple_search_bonus_${PROMPT_TAG}_${STEPS}step_metrics.json"

echo "wrote $RESULT_ROOT/simple_search_bonus_${PROMPT_TAG}_${STEPS}step.csv"

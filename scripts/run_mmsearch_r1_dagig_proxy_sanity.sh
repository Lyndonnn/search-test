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
LOG_DIR="$RESULT_ROOT/logs/dagig_proxy_search_required_${STEPS}step"
PROMPT_PATH="mmsearch_r1/prompts/round_1_user_prompt_qwenvl_search_required.pkl"

mkdir -p "$RESULT_ROOT" "$LOG_DIR"
python3 scripts/create_mmsearch_search_required_prompts.py --output "$PROMPT_PATH"

echo "Running MMSearch-R1 DAGIG proxy search-required sanity run"
echo "  purpose=selective action-span credit for evidence-supported search actions"
echo "  steps=$STEPS test_freq=$TEST_FREQ save_freq=$SAVE_FREQ bonus=$BONUS"
echo "  prompt=$PROMPT_PATH"
echo "  log_dir=$LOG_DIR"

WANDB_EXP_NAME="dagig_proxy_search_required_${STEPS}step" \
TOTAL_TRAINING_STEPS="$STEPS" \
TEST_FREQ="$TEST_FREQ" \
SAVE_FREQ="$SAVE_FREQ" \
ROLLOUT_LOG_DIR="$LOG_DIR" \
USER_PROMPT_ROUND_1="$PROMPT_PATH" \
SEARCH_PENALTY="${SEARCH_PENALTY:-0.0}" \
FORMAT_PENALTY="${FORMAT_PENALTY:-0.1}" \
REWARD_SHAPING_MODE="dagig_lite_proxy" \
SEARCH_ACTION_BONUS="$BONUS" \
SEARCH_ACTION_BONUS_CORRECT_ONLY="${SEARCH_ACTION_BONUS_CORRECT_ONLY:-False}" \
bash mmsearch_r1/scripts/run_mmsearch_r1_grpo_a100_debug.sh

python3 scripts/extract_mmsearch_train_metrics.py \
  --input "$LOG_DIR/final_validation.json" \
  --method "dagig_proxy_search_required_${STEPS}step" \
  --output-csv "$RESULT_ROOT/dagig_proxy_search_required_${STEPS}step.csv" \
  --output-json "$RESULT_ROOT/dagig_proxy_search_required_${STEPS}step_metrics.json"

echo "wrote $RESULT_ROOT/dagig_proxy_search_required_${STEPS}step.csv"

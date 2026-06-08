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
BONUS="${DAGIG_OFFLINE_SEARCH_BONUS:-0.2}"
RELABEL_PATH="${DAGIG_OFFLINE_RELABEL_PATH:-results/dagig_offline/dependency_relabel_selected.jsonl}"
RESULT_ROOT="${DAGIG_DATA_ROOT:-$ROOT/data/cache/dagig_local}/results_keep/mmsearch_mainline"
LOG_DIR="$RESULT_ROOT/logs/dagig_offline_${STEPS}step"
PROMPT_PATH="${USER_PROMPT_ROUND_1:-}"

if [[ ! -s "$RELABEL_PATH" ]]; then
  echo "Missing DAG-IG selected relabel file: $RELABEL_PATH" >&2
  echo "Run first:" >&2
  echo "  DAGIG_RELABEL_LIMIT=128 DAGIG_RELABEL_CF_SAMPLES=1 DAGIG_RELABEL_USE_REFERENCE=1 make offline_dependency_relabel" >&2
  exit 1
fi

mkdir -p "$RESULT_ROOT" "$LOG_DIR"

if [[ "${DAGIG_USE_SEARCH_REQUIRED_PROMPT:-1}" == "1" && -z "$PROMPT_PATH" ]]; then
  PROMPT_PATH="mmsearch_r1/prompts/round_1_user_prompt_qwenvl_search_required.pkl"
  python3 scripts/create_mmsearch_search_required_prompts.py --output "$PROMPT_PATH"
fi

echo "Running MMSearch-R1 DAG-IG offline shaping sanity run"
echo "  purpose=MMSearch-R1 GRPO with offline DAG-IG selected-edge action credit"
echo "  steps=$STEPS test_freq=$TEST_FREQ save_freq=$SAVE_FREQ bonus=$BONUS"
echo "  relabel_path=$RELABEL_PATH"
echo "  prompt=${PROMPT_PATH:-<default-mmsearch-r1>}"
echo "  log_dir=$LOG_DIR"

env_args=(
  WANDB_EXP_NAME="dagig_offline_${STEPS}step"
  TOTAL_TRAINING_STEPS="$STEPS"
  TEST_FREQ="$TEST_FREQ"
  SAVE_FREQ="$SAVE_FREQ"
  ROLLOUT_LOG_DIR="$LOG_DIR"
  SEARCH_PENALTY="${SEARCH_PENALTY:-0.0}"
  FORMAT_PENALTY="${FORMAT_PENALTY:-0.1}"
  REWARD_SHAPING_MODE="dagig_offline"
  DAGIG_OFFLINE_RELABEL_PATH="$RELABEL_PATH"
  DAGIG_OFFLINE_SEARCH_BONUS="$BONUS"
  DAGIG_OFFLINE_CORRECT_ONLY="${DAGIG_OFFLINE_CORRECT_ONLY:-False}"
  DAGIG_OFFLINE_BONUS_TOOL="${DAGIG_OFFLINE_BONUS_TOOL:-}"
  DAGIG_OFFLINE_WEIGHT_KEY="${DAGIG_OFFLINE_WEIGHT_KEY:-constant}"
)
if [[ -n "$PROMPT_PATH" ]]; then
  env_args+=(USER_PROMPT_ROUND_1="$PROMPT_PATH")
fi

env "${env_args[@]}" bash mmsearch_r1/scripts/run_mmsearch_r1_grpo_a100_debug.sh

python3 scripts/extract_mmsearch_train_metrics.py \
  --input "$LOG_DIR/final_validation.json" \
  --method "dagig_offline_${STEPS}step" \
  --output-csv "$RESULT_ROOT/dagig_offline_${STEPS}step.csv" \
  --output-json "$RESULT_ROOT/dagig_offline_${STEPS}step_metrics.json"

echo "wrote $RESULT_ROOT/dagig_offline_${STEPS}step.csv"

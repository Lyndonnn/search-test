#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/projects/dagig_mmsearch/src:${PYTHONPATH:-}"

if [ -f projects/dagig_mmsearch/scripts/autodl_env.sh ]; then
  # shellcheck disable=SC1091
  source projects/dagig_mmsearch/scripts/autodl_env.sh
fi

SAMPLES_JSONL="${DAGIG_MODEL_AGENT_SAMPLES_JSONL:-${DAGIG_REF_SAMPLES_JSONL:-}}"
TEXT_INDEX="${DAGIG_MODEL_AGENT_TEXT_INDEX:-${DAGIG_REF_TEXT_INDEX:-}}"
IMAGE_INDEX="${DAGIG_MODEL_AGENT_IMAGE_INDEX:-${DAGIG_REF_IMAGE_INDEX:-}}"

ARGS=(
  --config "${DAGIG_MODEL_AGENT_CONFIG:-${DAGIG_REF_CONFIG:-projects/dagig_mmsearch/configs/dagig_lite_qwen25vl_3b_a800.yaml}}"
  --limit "${DAGIG_MODEL_AGENT_LIMIT:-8}"
  --output "${DAGIG_MODEL_AGENT_OUTPUT:-results/model_agent/model_agent_rollout.jsonl}"
  --table-output "${DAGIG_MODEL_AGENT_TABLE_OUTPUT:-paper_artifacts/tables/model_agent_rollout.csv}"
  --method "${DAGIG_MODEL_AGENT_METHOD:-model_agent_qwen25vl3b}"
  --max-new-tokens "${DAGIG_MODEL_AGENT_MAX_NEW_TOKENS:-96}"
  --answer-max-new-tokens "${DAGIG_MODEL_AGENT_ANSWER_MAX_NEW_TOKENS:-64}"
  --temperature "${DAGIG_MODEL_AGENT_TEMPERATURE:-0.0}"
  --rollout-mode "${DAGIG_MODEL_AGENT_ROLLOUT_MODE:-one_turn_oracle}"
  --cf-samples "${DAGIG_MODEL_AGENT_CF_SAMPLES:-2}"
)

if [ -n "$SAMPLES_JSONL" ]; then
  ARGS+=(--samples-jsonl "$SAMPLES_JSONL")
fi
if [ -n "$TEXT_INDEX" ]; then
  ARGS+=(--text-index "$TEXT_INDEX")
fi
if [ -n "$IMAGE_INDEX" ]; then
  ARGS+=(--image-index "$IMAGE_INDEX")
fi
if [ "${DAGIG_MODEL_AGENT_SCRIPTED_DIRECT_STOP:-0}" = "1" ]; then
  ARGS+=(--scripted-direct-stop)
fi
if [ "${DAGIG_MODEL_AGENT_FORCE_SEARCH_WHEN_NEEDED:-0}" = "1" ]; then
  ARGS+=(--force-search-when-needed)
fi
if [ "${DAGIG_MODEL_AGENT_FALLBACK_ON_INVALID:-0}" = "1" ]; then
  ARGS+=(--fallback-on-invalid)
fi
if [ "${DAGIG_MODEL_AGENT_SCORE_REWARD:-0}" = "1" ]; then
  ARGS+=(--score-reward)
fi

python3 -m eval.run_model_agent_rollout "${ARGS[@]}"

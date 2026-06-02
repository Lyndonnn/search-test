#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/projects/dagig_mmsearch/src:${PYTHONPATH:-}"

if [ -f projects/dagig_mmsearch/scripts/autodl_env.sh ]; then
  # shellcheck disable=SC1091
  source projects/dagig_mmsearch/scripts/autodl_env.sh
fi

SAMPLES_JSONL="${DAGIG_AGENT_SAMPLES_JSONL:-${DAGIG_REF_SAMPLES_JSONL:-}}"
TEXT_INDEX="${DAGIG_AGENT_TEXT_INDEX:-${DAGIG_REF_TEXT_INDEX:-}}"
IMAGE_INDEX="${DAGIG_AGENT_IMAGE_INDEX:-${DAGIG_REF_IMAGE_INDEX:-}}"

ARGS=(
  --limit "${DAGIG_AGENT_LIMIT:-8}"
  --output "${DAGIG_AGENT_OUTPUT:-results/agent_rollout/agentic_rollout_smoke.jsonl}"
  --table-output "${DAGIG_AGENT_TABLE_OUTPUT:-paper_artifacts/tables/agentic_rollout_smoke.csv}"
  --method "${DAGIG_AGENT_METHOD:-agentic_search}"
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

python3 -m eval.run_agentic_rollout "${ARGS[@]}"

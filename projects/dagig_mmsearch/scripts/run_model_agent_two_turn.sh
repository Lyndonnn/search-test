#!/usr/bin/env bash
set -euo pipefail

export DAGIG_MODEL_AGENT_ROLLOUT_MODE="${DAGIG_MODEL_AGENT_ROLLOUT_MODE:-two_turn_non_oracle}"
export DAGIG_MODEL_AGENT_METHOD="${DAGIG_MODEL_AGENT_METHOD:-model_agent_two_turn_qwen25vl3b}"
export DAGIG_MODEL_AGENT_OUTPUT="${DAGIG_MODEL_AGENT_OUTPUT:-results/model_agent/model_agent_two_turn.jsonl}"
export DAGIG_MODEL_AGENT_TABLE_OUTPUT="${DAGIG_MODEL_AGENT_TABLE_OUTPUT:-paper_artifacts/tables/model_agent_two_turn.csv}"

bash "$(dirname "${BASH_SOURCE[0]}")/run_model_agent_rollout.sh"

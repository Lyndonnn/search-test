#!/usr/bin/env bash
set -euo pipefail

DATASET="${DAGIG_REAL_DATASET:-fvqa}"
SPLIT="${DAGIG_REAL_SPLIT:-train}"
export DAGIG_MODEL_AGENT_SAMPLES_JSONL="${DAGIG_MODEL_AGENT_SAMPLES_JSONL:-data/processed/${DATASET}_${SPLIT}_small.jsonl}"
export DAGIG_MODEL_AGENT_TEXT_INDEX="${DAGIG_MODEL_AGENT_TEXT_INDEX:-data/indexes/${DATASET}_${SPLIT}_nonleaky_text_corpus.jsonl}"
export DAGIG_MODEL_AGENT_IMAGE_INDEX="${DAGIG_MODEL_AGENT_IMAGE_INDEX:-data/indexes/${DATASET}_${SPLIT}_nonleaky_image_corpus.jsonl}"
export DAGIG_MODEL_AGENT_METHOD="${DAGIG_MODEL_AGENT_METHOD:-model_agent_two_turn_nonleaky_qwen25vl3b}"
export DAGIG_MODEL_AGENT_OUTPUT="${DAGIG_MODEL_AGENT_OUTPUT:-results/model_agent/model_agent_two_turn_nonleaky.jsonl}"
export DAGIG_MODEL_AGENT_TABLE_OUTPUT="${DAGIG_MODEL_AGENT_TABLE_OUTPUT:-paper_artifacts/tables/model_agent_two_turn_nonleaky.csv}"
export DAGIG_MODEL_AGENT_REDACT_OBSERVATION_ANSWERS="${DAGIG_MODEL_AGENT_REDACT_OBSERVATION_ANSWERS:-1}"

bash "$(dirname "${BASH_SOURCE[0]}")/run_model_agent_two_turn.sh"

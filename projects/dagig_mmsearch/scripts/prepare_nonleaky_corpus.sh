#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/projects/dagig_mmsearch/src:${PYTHONPATH:-}"

if [ -f projects/dagig_mmsearch/scripts/autodl_env.sh ]; then
  # shellcheck disable=SC1091
  source projects/dagig_mmsearch/scripts/autodl_env.sh
fi

DATASET="${DAGIG_REAL_DATASET:-fvqa}"
SPLIT="${DAGIG_REAL_SPLIT:-train}"
SAMPLES="${DAGIG_NONLEAKY_SAMPLES_JSONL:-data/processed/${DATASET}_${SPLIT}_small.jsonl}"
TEXT_INDEX="${DAGIG_NONLEAKY_TEXT_INDEX:-data/indexes/${DATASET}_${SPLIT}_nonleaky_text_corpus.jsonl}"
IMAGE_INDEX="${DAGIG_NONLEAKY_IMAGE_INDEX:-data/indexes/${DATASET}_${SPLIT}_nonleaky_image_corpus.jsonl}"

ARGS=(
  --samples-jsonl "$SAMPLES"
  --text-index "$TEXT_INDEX"
  --image-index "$IMAGE_INDEX"
  --max-snippet-chars "${DAGIG_NONLEAKY_MAX_SNIPPET_CHARS:-360}"
)

if [ "${DAGIG_NONLEAKY_INCLUDE_QUESTION_DOCS:-0}" = "1" ]; then
  ARGS+=(--include-question-docs)
fi

python3 -m data.prepare_nonleaky_corpus "${ARGS[@]}"

echo "To run two-turn non-oracle with this corpus:"
echo "DAGIG_MODEL_AGENT_SAMPLES_JSONL=$SAMPLES \\"
echo "DAGIG_MODEL_AGENT_TEXT_INDEX=$TEXT_INDEX \\"
echo "DAGIG_MODEL_AGENT_IMAGE_INDEX=$IMAGE_INDEX \\"
echo "make model_agent_two_turn"

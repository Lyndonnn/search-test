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
LIMIT="${DAGIG_REAL_LIMIT:-32}"
OUT="data/processed/${DATASET}_${SPLIT}_small.jsonl"
TEXT_INDEX="data/indexes/${DATASET}_${SPLIT}_text_corpus.jsonl"
IMAGE_INDEX="data/indexes/${DATASET}_${SPLIT}_image_corpus.jsonl"

set +e
python3 -m data.real_vqa_adapter \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --limit "$LIMIT" \
  --out "$OUT" \
  --text-index "$TEXT_INDEX" \
  --image-index "$IMAGE_INDEX"
status=$?
set -e

if [ "$status" -ne 0 ]; then
  if [ -s "$OUT" ] && [ -s "$TEXT_INDEX" ] && [ -s "$IMAGE_INDEX" ]; then
    echo "prepare_real_data warning: Python exited with status $status after writing outputs; continuing."
  else
    exit "$status"
  fi
fi

echo "To score this small dataset:"
echo "DAGIG_REF_SAMPLES_JSONL=data/processed/${DATASET}_${SPLIT}_small.jsonl \\"
echo "DAGIG_REF_TEXT_INDEX=data/indexes/${DATASET}_${SPLIT}_text_corpus.jsonl \\"
echo "DAGIG_REF_IMAGE_INDEX=data/indexes/${DATASET}_${SPLIT}_image_corpus.jsonl \\"
echo "DAGIG_REF_METHOD=reference_logprob_${DATASET}_${SPLIT} \\"
echo "DAGIG_REF_SMOKE_LIMIT=$LIMIT DAGIG_REF_CF_SAMPLES=2 make reference_logprob_smoke"

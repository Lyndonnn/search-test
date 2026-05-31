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

python3 -m data.real_vqa_adapter \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --limit "$LIMIT" \
  --out "data/processed/${DATASET}_${SPLIT}_small.jsonl" \
  --text-index "data/indexes/${DATASET}_${SPLIT}_text_corpus.jsonl" \
  --image-index "data/indexes/${DATASET}_${SPLIT}_image_corpus.jsonl"

echo "To score this small dataset:"
echo "DAGIG_REF_SAMPLES_JSONL=data/processed/${DATASET}_${SPLIT}_small.jsonl \\"
echo "DAGIG_REF_TEXT_INDEX=data/indexes/${DATASET}_${SPLIT}_text_corpus.jsonl \\"
echo "DAGIG_REF_IMAGE_INDEX=data/indexes/${DATASET}_${SPLIT}_image_corpus.jsonl \\"
echo "DAGIG_REF_METHOD=reference_logprob_${DATASET}_${SPLIT} \\"
echo "DAGIG_REF_SMOKE_LIMIT=$LIMIT DAGIG_REF_CF_SAMPLES=2 make reference_logprob_smoke"

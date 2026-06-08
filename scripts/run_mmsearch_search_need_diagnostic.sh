#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f projects/dagig_mmsearch/scripts/autodl_env.sh ]]; then
  # shellcheck disable=SC1091
  source projects/dagig_mmsearch/scripts/autodl_env.sh
fi

RESULT_ROOT="${DAGIG_DATA_ROOT:-$ROOT/data/cache/dagig_local}/results_keep/mmsearch_mainline"
DIRECT_VAL_RESULT="${DIRECT_VAL_RESULT:-$RESULT_ROOT/val_only_direct/val_result_latest.json}"
SEARCH_VAL_RESULT="${SEARCH_VAL_RESULT:-$RESULT_ROOT/val_only_image_required/val_result_latest.json}"
OUT_CSV="${SEARCH_NEED_OUTPUT_CSV:-paper_artifacts/tables/search_need_diagnostic.csv}"
OUT_JSON="${SEARCH_NEED_OUTPUT_JSON:-paper_artifacts/tables/search_need_diagnostic.json}"
OUT_SAMPLES="${SEARCH_NEED_OUTPUT_SAMPLES_CSV:-paper_artifacts/tables/search_need_samples.csv}"

if [[ ! -s "$DIRECT_VAL_RESULT" ]]; then
  echo "Missing direct val_result: $DIRECT_VAL_RESULT" >&2
  echo "Run direct val-only first, then copy the newest val_result_*.json to that path." >&2
  exit 1
fi

if [[ ! -s "$SEARCH_VAL_RESULT" ]]; then
  echo "Missing forced-search val_result: $SEARCH_VAL_RESULT" >&2
  echo "Run image-required val-only first, then copy the newest val_result_*.json to that path." >&2
  exit 1
fi

python3 scripts/analyze_mmsearch_search_need.py \
  --direct "$DIRECT_VAL_RESULT" \
  --search "$SEARCH_VAL_RESULT" \
  --method "${SEARCH_NEED_METHOD:-fvqa_debug_direct_vs_image_required}" \
  --output-csv "$OUT_CSV" \
  --output-json "$OUT_JSON" \
  --output-samples-csv "$OUT_SAMPLES"

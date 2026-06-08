#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/projects/dagig_mmsearch/src:${PYTHONPATH:-}"

if [[ -f projects/dagig_mmsearch/scripts/autodl_env.sh ]]; then
  # shellcheck disable=SC1091
  source projects/dagig_mmsearch/scripts/autodl_env.sh
fi

if [[ "${DAGIG_RELABEL_USE_REFERENCE:-0}" == "1" ]]; then
  if ! python3 - <<'PY' >/dev/null 2>&1
import transformers
PY
  then
    PY_TAG="$(python3 - <<'PY'
import sys
print(f"py{sys.version_info.major}{sys.version_info.minor}")
PY
)"
    DAGIG_REFERENCE_VENV="${DAGIG_REFERENCE_VENV:-$DAGIG_DATA_ROOT/venvs/dagig-reference-$PY_TAG}"
    if [[ ! -f "$DAGIG_REFERENCE_VENV/bin/activate" ]]; then
      echo "Missing DAG-IG reference env: $DAGIG_REFERENCE_VENV" >&2
      echo "Run once: make dagig_setup_reference_env" >&2
      exit 1
    fi
    # shellcheck disable=SC1090
    source "$DAGIG_REFERENCE_VENV/bin/activate"
  fi
fi

ARGS=(
  --config "${DAGIG_RELABEL_CONFIG:-projects/dagig_mmsearch/configs/dagig_lite_qwen25vl_3b_a800.yaml}"
  --limit "${DAGIG_RELABEL_LIMIT:-32}"
  --parquet "${DAGIG_RELABEL_PARQUET:-mmsearch_r1/data/fvqa_debug_train.pq}"
  --cf-samples "${DAGIG_RELABEL_CF_SAMPLES:-2}"
  --search-topk "${DAGIG_RELABEL_SEARCH_TOPK:-5}"
  --output "${DAGIG_RELABEL_OUTPUT:-results/dagig_offline/dependency_relabel.jsonl}"
  --selected-output "${DAGIG_RELABEL_SELECTED_OUTPUT:-results/dagig_offline/dependency_relabel_selected.jsonl}"
  --edge-csv "${DAGIG_RELABEL_EDGE_CSV:-paper_artifacts/tables/offline_dependency_edges.csv}"
  --selected-edge-csv "${DAGIG_RELABEL_SELECTED_EDGE_CSV:-paper_artifacts/tables/offline_dependency_edges_selected.csv}"
  --summary-csv "${DAGIG_RELABEL_SUMMARY_CSV:-paper_artifacts/tables/offline_dependency_summary.csv}"
  --method "${DAGIG_RELABEL_METHOD:-offline_dependency_relabel}"
)

if [[ -n "${DAGIG_RELABEL_SAMPLES_JSONL:-}" ]]; then
  ARGS+=(--samples-jsonl "$DAGIG_RELABEL_SAMPLES_JSONL")
fi
if [[ -n "${DAGIG_RELABEL_TEXT_INDEX:-}" ]]; then
  ARGS+=(--text-index "$DAGIG_RELABEL_TEXT_INDEX")
fi
if [[ -n "${DAGIG_RELABEL_IMAGE_INDEX:-}" ]]; then
  ARGS+=(--image-index "$DAGIG_RELABEL_IMAGE_INDEX")
fi
if [[ "${DAGIG_RELABEL_USE_REFERENCE:-0}" == "1" ]]; then
  ARGS+=(--use-reference)
fi
if [[ "${DAGIG_RELABEL_KEEP_EARLY_ANSWER:-0}" == "1" ]]; then
  ARGS+=(--keep-early-answer)
fi
if [[ "${DAGIG_RELABEL_NO_AUTO_INDEX:-0}" == "1" ]]; then
  ARGS+=(--no-auto-index)
fi

python3 -m eval.run_offline_dependency_relabel "${ARGS[@]}"

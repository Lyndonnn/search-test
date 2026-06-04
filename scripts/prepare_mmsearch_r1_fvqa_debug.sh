#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

if [[ -f projects/dagig_mmsearch/scripts/autodl_env.sh ]]; then
  # shellcheck disable=SC1091
  source projects/dagig_mmsearch/scripts/autodl_env.sh
fi

FVQA_TRAIN_SPLIT="${FVQA_TRAIN_SPLIT:-train}"
FVQA_VAL_SPLIT="${FVQA_VAL_SPLIT:-test}"
FVQA_TRAIN_LIMIT="${FVQA_TRAIN_LIMIT:-128}"
FVQA_VAL_LIMIT="${FVQA_VAL_LIMIT:-128}"
FVQA_TRAIN_OFFSET="${FVQA_TRAIN_OFFSET:-0}"
FVQA_VAL_OFFSET="${FVQA_VAL_OFFSET:-0}"
FVQA_STREAMING="${FVQA_STREAMING:-1}"
TRAIN_OUT="${TRAIN_DATA_PATH:-mmsearch_r1/data/fvqa_debug_train.pq}"
VAL_OUT="${VAL_DATA_PATH:-mmsearch_r1/data/fvqa_debug_val.pq}"

mkdir -p mmsearch_r1/data

run_prepare() {
  local split="$1"
  local limit="$2"
  local offset="$3"
  local out="$4"
  local streaming_args=()
  if [[ "$FVQA_STREAMING" == "1" ]]; then
    streaming_args+=(--streaming)
  fi

  set +e
  python3 scripts/prepare_fvqa_verl.py \
    --split "$split" \
    --limit "$limit" \
    --offset "$offset" \
    --out "$out" \
    --print-sample \
    "${streaming_args[@]}"
  local status=$?
  set -e

  if [[ "$status" -ne 0 ]]; then
    if [[ -s "$out" ]]; then
      echo "prepare warning: python exited with status $status after writing $out; continuing."
    else
      exit "$status"
    fi
  fi
}

run_prepare "$FVQA_TRAIN_SPLIT" "$FVQA_TRAIN_LIMIT" "$FVQA_TRAIN_OFFSET" "$TRAIN_OUT"
run_prepare "$FVQA_VAL_SPLIT" "$FVQA_VAL_LIMIT" "$FVQA_VAL_OFFSET" "$VAL_OUT"

echo "Prepared MMSearch-R1 FVQA debug data:"
echo "  train: $TRAIN_OUT"
echo "  val:   $VAL_OUT"
echo
echo "Next baseline checks:"
echo "  make mmsearch_val_only"
echo "  make mmsearch_grpo_a100_debug"

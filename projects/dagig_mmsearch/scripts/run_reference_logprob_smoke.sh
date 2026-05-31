#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/projects/dagig_mmsearch/src:${PYTHONPATH:-}"

if [ -f projects/dagig_mmsearch/scripts/autodl_env.sh ]; then
  # shellcheck disable=SC1091
  source projects/dagig_mmsearch/scripts/autodl_env.sh
fi

ARGS=(
  --config "${DAGIG_REF_CONFIG:-projects/dagig_mmsearch/configs/dagig_lite_qwen25vl_3b_a800.yaml}"
  --limit "${DAGIG_REF_SMOKE_LIMIT:-2}"
  --cf-samples "${DAGIG_REF_CF_SAMPLES:-1}"
  --output "${DAGIG_REF_OUTPUT:-results/dagig_lite/reference_logprob_smoke.jsonl}"
  --method "${DAGIG_REF_METHOD:-reference_logprob_smoke}"
)

if [ -n "${DAGIG_REF_SAMPLES_JSONL:-}" ]; then
  ARGS+=(--samples-jsonl "$DAGIG_REF_SAMPLES_JSONL")
fi
if [ -n "${DAGIG_REF_TEXT_INDEX:-}" ]; then
  ARGS+=(--text-index "$DAGIG_REF_TEXT_INDEX")
fi
if [ -n "${DAGIG_REF_IMAGE_INDEX:-}" ]; then
  ARGS+=(--image-index "$DAGIG_REF_IMAGE_INDEX")
fi

python3 -m eval.run_reference_logprob_smoke "${ARGS[@]}"

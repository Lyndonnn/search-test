#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/projects/dagig_mmsearch/src:${PYTHONPATH:-}"

if [ -f projects/dagig_mmsearch/scripts/autodl_env.sh ]; then
  # shellcheck disable=SC1091
  source projects/dagig_mmsearch/scripts/autodl_env.sh
fi

SAMPLES_JSONL="${DAGIG_ABLATION_SAMPLES_JSONL:-${DAGIG_REF_SAMPLES_JSONL:-}}"
TEXT_INDEX="${DAGIG_ABLATION_TEXT_INDEX:-${DAGIG_REF_TEXT_INDEX:-}}"
IMAGE_INDEX="${DAGIG_ABLATION_IMAGE_INDEX:-${DAGIG_REF_IMAGE_INDEX:-}}"

ARGS=(
  --config "${DAGIG_ABLATION_CONFIG:-${DAGIG_REF_CONFIG:-projects/dagig_mmsearch/configs/dagig_lite_qwen25vl_3b_a800.yaml}}"
  --limit "${DAGIG_ABLATION_LIMIT:-${DAGIG_REF_SMOKE_LIMIT:-32}}"
  --cf-samples "${DAGIG_ABLATION_CF_SAMPLES:-${DAGIG_REF_CF_SAMPLES:-4}}"
  --rollout-mode "${DAGIG_ABLATION_ROLLOUT_MODE:-prompted}"
  --output-dir "${DAGIG_ABLATION_OUTPUT_DIR:-results/ablations}"
  --table-output "${DAGIG_ABLATION_TABLE_OUTPUT:-paper_artifacts/tables/reference_ablation.csv}"
  --delta-output "${DAGIG_ABLATION_DELTA_OUTPUT:-paper_artifacts/tables/reference_ablation_delta.csv}"
  --method-prefix "${DAGIG_ABLATION_METHOD_PREFIX:-reference_ablation}"
  --variants "${DAGIG_ABLATION_VARIANTS:-local_ig_only,dagig_lite,dagig_no_gate,dagig_no_cost,lambda_0,lambda_025,lambda_05,lambda_1}"
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

python3 -m eval.run_reference_ablation "${ARGS[@]}"

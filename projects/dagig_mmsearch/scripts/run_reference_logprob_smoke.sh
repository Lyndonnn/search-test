#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/projects/dagig_mmsearch/src:${PYTHONPATH:-}"

if [ -f projects/dagig_mmsearch/scripts/autodl_env.sh ]; then
  # shellcheck disable=SC1091
  source projects/dagig_mmsearch/scripts/autodl_env.sh
fi

python3 -m eval.run_reference_logprob_smoke \
  --config projects/dagig_mmsearch/configs/dagig_lite_qwen25vl_3b_a800.yaml \
  --limit "${DAGIG_REF_SMOKE_LIMIT:-2}" \
  --cf-samples "${DAGIG_REF_CF_SAMPLES:-1}"


#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/projects/dagig_mmsearch/src:${PYTHONPATH:-}"

if [ -f projects/dagig_mmsearch/scripts/autodl_env.sh ]; then
  # shellcheck disable=SC1091
  source projects/dagig_mmsearch/scripts/autodl_env.sh
fi

python3 -m utils.hf_probe --repo-id "${DAGIG_REF_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"

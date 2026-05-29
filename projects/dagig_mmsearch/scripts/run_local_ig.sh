#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/projects/dagig_mmsearch/src:${PYTHONPATH:-}"

python3 -m train.rl_grpo --stage local_ig --config projects/dagig_mmsearch/configs/local_ig_qwen25vl_3b_a800.yaml
echo "Local-IG smoke outputs saved to results/local_ig"


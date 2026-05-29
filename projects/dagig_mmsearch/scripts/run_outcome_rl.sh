#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/projects/dagig_mmsearch/src:${PYTHONPATH:-}"

python3 -m train.rl_grpo --stage outcome --config projects/dagig_mmsearch/configs/outcome_only_qwen25vl_3b_a800.yaml
echo "outcome-only RL smoke outputs saved to results/outcome_rl"


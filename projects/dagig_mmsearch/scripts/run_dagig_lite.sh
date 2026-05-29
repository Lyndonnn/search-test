#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/projects/dagig_mmsearch/src:${PYTHONPATH:-}"

python3 -m eval.run_agent_eval
python3 -m train.rl_grpo --stage dagig_lite --config projects/dagig_mmsearch/configs/dagig_lite_qwen25vl_3b_a800.yaml
echo "DAG-IG-Lite smoke outputs saved to results/dagig_lite"


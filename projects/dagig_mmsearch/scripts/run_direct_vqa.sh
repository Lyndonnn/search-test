#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/projects/dagig_mmsearch/src:${PYTHONPATH:-}"

python3 -m eval.run_direct_vqa
echo "direct VQA outputs saved to results/direct_vqa and paper_artifacts/tables/direct_vqa.csv"


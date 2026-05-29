#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/projects/dagig_mmsearch/src:${PYTHONPATH:-}"

python3 -m data.dataset_mixer
python3 -m data.prepare_fvqa
python3 -m data.prepare_infoseek
python3 -m data.prepare_mmsearch_plus

echo "prepared toy data under data/processed and indexes under data/indexes"


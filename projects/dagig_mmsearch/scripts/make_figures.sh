#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/projects/dagig_mmsearch/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="$ROOT/data/cache/matplotlib"
export XDG_CACHE_HOME="$ROOT/data/cache/xdg"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

python3 -m eval.make_figures
echo "figures saved to paper_artifacts/figures"

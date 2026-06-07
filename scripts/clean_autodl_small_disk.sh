#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f projects/dagig_mmsearch/scripts/autodl_env.sh ]]; then
  # shellcheck disable=SC1091
  source projects/dagig_mmsearch/scripts/autodl_env.sh
fi

DAGIG_DATA_ROOT="${DAGIG_DATA_ROOT:-/root/autodl-tmp/dagig}"

echo "Before cleanup:"
df -h || true
du -sh "$DAGIG_DATA_ROOT"/* 2>/dev/null | sort -h || true

mkdir -p "$DAGIG_DATA_ROOT/results_keep"

echo "Deleting large temporary artifacts. Model cache and FVQA parquet/images are kept."
rm -rf /tmp/ray
rm -rf "$DAGIG_DATA_ROOT/tmp"
rm -rf "$DAGIG_DATA_ROOT/pip_cache"
rm -rf "$DAGIG_DATA_ROOT/xdg_cache"
rm -rf "$DAGIG_DATA_ROOT/matplotlib"
rm -rf "$DAGIG_DATA_ROOT/checkpoints/mmsearch_r1/grpo_a100_debug"
rm -rf checkpoints/mmsearch_r1/grpo_a100_debug
rm -rf "$HOME/.cache/pip"

find "$ROOT" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "$ROOT" -type d -name ".pytest_cache" -prune -exec rm -rf {} +

echo "After cleanup:"
df -h || true
du -sh "$DAGIG_DATA_ROOT"/* 2>/dev/null | sort -h || true


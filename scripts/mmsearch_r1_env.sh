#!/usr/bin/env bash

# Source this file from MMSearch-R1 baseline commands. It ensures imports use
# the exact veRL commit pinned by the original MMSearch-R1 repository.

if [[ -z "${ROOT:-}" ]]; then
  _MMSEARCH_R1_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "$_MMSEARCH_R1_ENV_DIR/.." && pwd)"
fi

MMSEARCH_R1_VERL_COMMIT="8e9e73723fd1cc729bedb3bbcf915060afbda91d"
MMSEARCH_R1_VERL_ROOT="${MMSEARCH_R1_VERL_ROOT:-$ROOT/third_party/mmsearch_r1_verl}"
MMSEARCH_R1_VENV="${MMSEARCH_R1_VENV:-$ROOT/.venv-mmsearch-r1}"

if [[ ! -x "$MMSEARCH_R1_VENV/bin/python" ]] \
  || [[ ! -f "$MMSEARCH_R1_VENV/bin/activate" ]] \
  || ! "$MMSEARCH_R1_VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "Missing or incomplete isolated MMSearch-R1 environment: $MMSEARCH_R1_VENV"
  echo "Run: make mmsearch_setup_baseline"
  return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "$MMSEARCH_R1_VENV/bin/activate"

if [[ ! -f "$MMSEARCH_R1_VERL_ROOT/verl/__init__.py" ]]; then
  echo "Missing exact MMSearch-R1 veRL checkout: $MMSEARCH_R1_VERL_ROOT"
  echo "Run: make mmsearch_setup_baseline"
  return 1 2>/dev/null || exit 1
fi

actual_commit="$(git -C "$MMSEARCH_R1_VERL_ROOT" rev-parse HEAD 2>/dev/null || true)"
if [[ "$actual_commit" != "$MMSEARCH_R1_VERL_COMMIT" ]]; then
  echo "Wrong MMSearch-R1 veRL commit: ${actual_commit:-<unknown>}"
  echo "Expected: $MMSEARCH_R1_VERL_COMMIT"
  echo "Run: make mmsearch_setup_baseline"
  return 1 2>/dev/null || exit 1
fi

export MMSEARCH_R1_VERL_ROOT
export MMSEARCH_R1_VERL_COMMIT
export MMSEARCH_R1_VENV
export PYTHONPATH="$MMSEARCH_R1_VERL_ROOT:$ROOT:${PYTHONPATH:-}"

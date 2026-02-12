#!/usr/bin/env bash
set -euo pipefail

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"

python3 - <<'PY' || {
import importlib
import sys
importlib.import_module("verl")
PY
} || {
  echo "[m0] Installing verl from submodule..."
  git submodule update --init --recursive
  python3 -m pip install -e ./verl
}

python3 scripts/run_m0_sanity_nohydra.py "$@"

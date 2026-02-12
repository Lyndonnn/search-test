#!/usr/bin/env bash
set -euo pipefail

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"

if ! python3 - <<'PY'
import importlib
importlib.import_module("verl")
PY
then
  echo "[m0] Installing verl from submodule..."
  git submodule update --init --recursive
  python3 -m pip install -e ./verl
fi

python3 scripts/run_m0_sanity_nohydra.py "$@"

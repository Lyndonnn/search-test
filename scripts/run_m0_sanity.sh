#!/usr/bin/env bash
set -euo pipefail

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"

ensure_verl_submodule() {
  if [[ -f "verl/setup.py" || -f "verl/pyproject.toml" ]]; then
    return 0
  fi

  echo "[m0] Initializing verl submodule..."
  git submodule sync --recursive
  git submodule update --init --recursive

  if [[ ! -f "verl/setup.py" && ! -f "verl/pyproject.toml" ]]; then
    echo "[m0] ERROR: verl submodule is still empty."
    echo "[m0] Run: git submodule update --init --recursive"
    echo "[m0] Then verify: ls -la verl"
    exit 1
  fi
}

ensure_verl_submodule

if ! python3 - <<'PY'
import importlib, os, sys
repo_root = os.getcwd()
expected = os.path.join(repo_root, "verl")
try:
    m = importlib.import_module("verl")
    path = os.path.abspath(getattr(m, "__file__", ""))
    if not path.startswith(os.path.abspath(expected)):
        raise SystemExit(1)
except Exception:
    raise SystemExit(1)
PY
then
  echo "[m0] Installing verl from submodule..."
  python3 -m pip uninstall -y verl || true
  python3 -m pip install -e ./verl
fi

python3 scripts/run_m0_sanity_nohydra.py "$@"

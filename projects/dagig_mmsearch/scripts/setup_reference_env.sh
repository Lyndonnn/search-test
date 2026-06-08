#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

if [[ -f projects/dagig_mmsearch/scripts/autodl_env.sh ]]; then
  # shellcheck disable=SC1091
  source projects/dagig_mmsearch/scripts/autodl_env.sh
fi

PYTHON_BIN="${DAGIG_REFERENCE_PYTHON:-python3}"
PY_TAG="$("$PYTHON_BIN" - <<'PY'
import sys
print(f"py{sys.version_info.major}{sys.version_info.minor}")
PY
)"
DAGIG_REFERENCE_VENV="${DAGIG_REFERENCE_VENV:-$DAGIG_DATA_ROOT/venvs/dagig-reference-$PY_TAG}"

venv_ready() {
  [[ -x "$DAGIG_REFERENCE_VENV/bin/python" ]] \
    && [[ -f "$DAGIG_REFERENCE_VENV/bin/activate" ]] \
    && "$DAGIG_REFERENCE_VENV/bin/python" -m pip --version >/dev/null 2>&1
}

if ! venv_ready; then
  mkdir -p "$(dirname "$DAGIG_REFERENCE_VENV")"
  echo "Creating persistent DAG-IG reference env: $DAGIG_REFERENCE_VENV"
  "$PYTHON_BIN" -m venv --system-site-packages "$DAGIG_REFERENCE_VENV"
fi

# shellcheck disable=SC1091
source "$DAGIG_REFERENCE_VENV/bin/activate"

needs_install=0
if [[ "${DAGIG_REFERENCE_FORCE_INSTALL:-0}" == "1" ]]; then
  needs_install=1
elif ! python - <<'PY' >/dev/null 2>&1
import pyarrow
import qwen_vl_utils
import transformers
PY
then
  needs_install=1
fi

if [[ "$needs_install" == "1" ]]; then
  python -m pip install -U pip setuptools wheel packaging
  python -m pip install \
    "transformers==4.51.0" \
    accelerate \
    safetensors \
    sentencepiece \
    qwen_vl_utils \
    pillow \
    PyYAML \
    "numpy<2.3" \
    "pyarrow==19.0.1" \
    datasets \
    tqdm \
    requests \
    "huggingface_hub>=0.30.0"
fi

python - <<'PY'
import importlib
import sys

print("dagig_reference_env_ready=True")
print(f"python={sys.version.split()[0]}")
for name in ("torch", "transformers", "pyarrow", "qwen_vl_utils"):
    module = importlib.import_module(name)
    print(f"{name}={getattr(module, '__version__', 'installed')}")
PY

echo "To reuse after reboot:"
echo "  source $DAGIG_REFERENCE_VENV/bin/activate"

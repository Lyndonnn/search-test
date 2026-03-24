#!/usr/bin/env bash
set -euo pipefail

# Usage in Colab:
#   REPO_URL=https://github.com/Lyndonnn/search-test.git \
#   bash scripts/bootstrap_colab.sh
#
# Optional:
#   REPO_DIR=search-test bash scripts/bootstrap_colab.sh
#   MINI_DATA_URL=https://.../mini_data.tar.gz bash scripts/bootstrap_colab.sh
#   VENV_DIR=.venv-colab bash scripts/bootstrap_colab.sh
#   INSTALL_FLASH_ATTN=1 bash scripts/bootstrap_colab.sh
#   PREPARE_FVQA_DEBUG=1 bash scripts/bootstrap_colab.sh

REPO_URL="${REPO_URL:-https://github.com/Lyndonnn/search-test.git}"
REPO_DIR="${REPO_DIR:-search-test}"
MINI_DATA_URL="${MINI_DATA_URL:-}"
VENV_DIR="${VENV_DIR:-.venv-colab}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
PREPARE_FVQA_DEBUG="${PREPARE_FVQA_DEBUG:-0}"
FVQA_TRAIN_LIMIT="${FVQA_TRAIN_LIMIT:-100}"
FVQA_TEST_LIMIT="${FVQA_TEST_LIMIT:-100}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${REPO_URL}" ]]; then
  echo "ERROR: Set REPO_URL to your repo clone URL."
  exit 1
fi

if [[ -f "${CURRENT_REPO_ROOT}/scripts/bootstrap_colab.sh" ]] && [[ -f "${CURRENT_REPO_ROOT}/README.md" ]]; then
  echo "[1/5] Using existing repo at ${CURRENT_REPO_ROOT}..."
  cd "${CURRENT_REPO_ROOT}"
else
  echo "[1/5] Cloning repo..."
  git clone "${REPO_URL}" "${REPO_DIR}"
  cd "${REPO_DIR}"
fi

echo "[2/5] Checking vendored verl..."
if [[ ! -f "verl/setup.py" && ! -f "verl/pyproject.toml" ]]; then
  echo "ERROR: ./verl is missing from the repository."
  exit 1
fi

echo "[3/6] Creating isolated virtualenv..."
if ! python3 -m venv "${VENV_DIR}"; then
  echo "[3/6] python3 -m venv failed, falling back to virtualenv..."
  python3 -m pip install -U virtualenv
  python3 -m virtualenv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
python -m pip install -U pip setuptools wheel packaging ninja
python -m pip install --no-cache-dir --force-reinstall numpy==1.26.4 pandas==2.2.2 pyarrow==19.0.1
python -m pip install -r requirements.txt
python -m pip install -e ./verl

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  echo "[4/6] Installing flash-attn in ${VENV_DIR}..."
  MAX_JOBS="${MAX_JOBS:-4}" python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
else
  echo "[4/6] Skipping flash-attn (INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN})."
fi

echo "[5/6] Optional data bootstrap..."
if [[ -f "mmsearch_r1/data/mini_data.pq" ]]; then
  echo "Using checked-in mmsearch_r1/data/mini_data.pq"
elif [[ -n "${MINI_DATA_URL}" ]]; then
  mkdir -p "mmsearch_r1/data"
  curl -L "${MINI_DATA_URL}" -o /tmp/mini_data.tar.gz
  tar -xzf /tmp/mini_data.tar.gz -C "mmsearch_r1/data" --strip-components=1
else
  echo "Skipping mini_data bootstrap."
fi

if [[ "${PREPARE_FVQA_DEBUG}" == "1" ]]; then
  mkdir -p "mmsearch_r1/data"
  python scripts/prepare_fvqa_verl.py \
    --split train \
    --limit "${FVQA_TRAIN_LIMIT}" \
    --out mmsearch_r1/data/fvqa_debug_train.pq
  python scripts/prepare_fvqa_verl.py \
    --split test \
    --limit "${FVQA_TEST_LIMIT}" \
    --out mmsearch_r1/data/fvqa_debug_val.pq
  echo "Prepared FVQA debug parquets:"
  echo "  mmsearch_r1/data/fvqa_debug_train.pq"
  echo "  mmsearch_r1/data/fvqa_debug_val.pq"
fi

echo "[6/6] Verifying environment..."
python - <<'PY'
import numpy
import pandas
import pyarrow
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("pyarrow", pyarrow.__version__)
PY

echo "Ready."
echo "Use the isolated interpreter to avoid Colab preinstalled package conflicts:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  python -c \"import numpy, pandas, pyarrow; print(numpy.__version__, pandas.__version__, pyarrow.__version__)\""
if [[ "${PREPARE_FVQA_DEBUG}" == "1" ]]; then
  echo "Next: run Pix2Fact/FVQA evaluation helpers from the venv."
  echo "  source ${VENV_DIR}/bin/activate && python scripts/eval_pix2fact_local_evidence.py --help"
else
  echo "Optional: PREPARE_FVQA_DEBUG=1 bash scripts/bootstrap_colab.sh"
fi

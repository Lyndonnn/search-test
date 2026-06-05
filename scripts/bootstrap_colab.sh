#!/usr/bin/env bash
set -euo pipefail

# Usage in Colab:
#   REPO_URL=https://github.com/Lyndonnn/search-test.git \
#   bash scripts/bootstrap_colab.sh
#
# Optional:
#   REPO_DIR=search-test bash scripts/bootstrap_colab.sh
#   MINI_DATA_URL=https://.../mini_data.tar.gz bash scripts/bootstrap_colab.sh
#   VENV_DIR=.venv-mmsearch-r1 bash scripts/bootstrap_colab.sh
#   INSTALL_FLASH_ATTN=1 bash scripts/bootstrap_colab.sh
#   PREPARE_FVQA_DEBUG=1 bash scripts/bootstrap_colab.sh

REPO_URL="${REPO_URL:-https://github.com/Lyndonnn/search-test.git}"
REPO_DIR="${REPO_DIR:-search-test}"
MINI_DATA_URL="${MINI_DATA_URL:-}"
VENV_DIR="${VENV_DIR:-.venv-mmsearch-r1}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
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

echo "[2/4] Installing the locked MMSearch-R1 baseline environment..."
MMSEARCH_R1_VENV="$VENV_DIR" INSTALL_FLASH_ATTN="$INSTALL_FLASH_ATTN" \
  bash scripts/setup_mmsearch_r1_baseline_env.sh

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[3/4] Optional data bootstrap..."
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
  FVQA_TRAIN_LIMIT="$FVQA_TRAIN_LIMIT" FVQA_VAL_LIMIT="$FVQA_TEST_LIMIT" \
    bash scripts/prepare_mmsearch_r1_fvqa_debug.sh
fi

echo "[4/4] Verifying environment..."
ROOT="$(pwd)"
export ROOT
# shellcheck disable=SC1091
source scripts/mmsearch_r1_env.sh
python scripts/check_mmsearch_cuda_stack.py \
  --require-nccl \
  --require-vllm \
  --require-exact-verl \
  --require-locked-versions

echo "Ready."
echo "Next:"
echo "  make mmsearch_val_only"
echo "  make mmsearch_grpo_a100_debug"

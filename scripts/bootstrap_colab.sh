#!/usr/bin/env bash
set -euo pipefail

# Usage in Colab:
#   REPO_URL=https://github.com/Lyndonnn/search-test.git \
#   MINI_DATA_URL=https://.../mini_data.tar.gz \
#   bash scripts/bootstrap_colab.sh

REPO_URL="${REPO_URL:-https://github.com/Lyndonnn/search-test.git}"
REPO_DIR="${REPO_DIR:-mmsearch-zoom-agent}"
MINI_DATA_URL="${MINI_DATA_URL:-}"
MINI_DATA_DIR="${MINI_DATA_DIR:-data/mini_data}"

if [[ -z "${REPO_URL}" ]]; then
  echo "ERROR: Set REPO_URL to your repo clone URL."
  exit 1
fi

if [[ -z "${MINI_DATA_URL}" ]]; then
  echo "ERROR: Set MINI_DATA_URL to the mini_data archive URL."
  exit 1
fi

echo "[1/4] Cloning repo..."
git clone --recurse-submodules "${REPO_URL}" "${REPO_DIR}"
cd "${REPO_DIR}"

echo "[2/5] Syncing submodules..."
git submodule sync --recursive
git submodule update --init --recursive

echo "[3/5] Installing dependencies..."
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
python3 -m pip install -e ./verl

echo "[4/5] Downloading mini_data..."
mkdir -p "${MINI_DATA_DIR}"
curl -L "${MINI_DATA_URL}" -o /tmp/mini_data.tar.gz
tar -xzf /tmp/mini_data.tar.gz -C "${MINI_DATA_DIR}" --strip-components=1

echo "[5/5] Ready."
echo "Next: run M0 sanity"
echo "  python3 -m mmsearch_r1.trainer.multimodal.main_ppo +exp=m0_sanity"

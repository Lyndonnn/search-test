#!/usr/bin/env bash
set -euo pipefail

# Usage in Colab:
#   REPO_URL=https://github.com/Lyndonnn/search-test.git \
#   bash scripts/bootstrap_colab.sh
#
# Optional:
#   REPO_DIR=search-test bash scripts/bootstrap_colab.sh
#   MINI_DATA_URL=https://.../mini_data.tar.gz bash scripts/bootstrap_colab.sh

REPO_URL="${REPO_URL:-https://github.com/Lyndonnn/search-test.git}"
REPO_DIR="${REPO_DIR:-mmsearch-zoom-agent}"
MINI_DATA_URL="${MINI_DATA_URL:-}"

if [[ -z "${REPO_URL}" ]]; then
  echo "ERROR: Set REPO_URL to your repo clone URL."
  exit 1
fi

echo "[1/5] Cloning repo..."
git clone "${REPO_URL}" "${REPO_DIR}"
cd "${REPO_DIR}"

echo "[2/5] Checking vendored verl..."
if [[ ! -f "verl/setup.py" && ! -f "verl/pyproject.toml" ]]; then
  echo "ERROR: ./verl is missing from the repository."
  exit 1
fi

echo "[3/5] Installing dependencies..."
python3 -m pip install -U pip
python3 -m pip install --no-cache-dir --force-reinstall numpy==1.26.4 pandas==2.2.2 pyarrow==19.0.1
python3 -m pip install -r requirements.txt
python3 -m pip install -e ./verl

echo "[4/5] Preparing sanity data..."
if [[ -f "mmsearch_r1/data/mini_data.pq" ]]; then
  echo "Using checked-in mmsearch_r1/data/mini_data.pq"
elif [[ -n "${MINI_DATA_URL}" ]]; then
  mkdir -p "mmsearch_r1/data"
  curl -L "${MINI_DATA_URL}" -o /tmp/mini_data.tar.gz
  tar -xzf /tmp/mini_data.tar.gz -C "mmsearch_r1/data" --strip-components=1
else
  echo "ERROR: mini_data.pq is missing and MINI_DATA_URL is not set."
  exit 1
fi

echo "[5/5] Ready."
echo "If Colab previously had incompatible numpy/pandas wheels loaded, restart the runtime once before importing datasets/pandas."
echo "Next: run M0 sanity"
echo "  bash scripts/run_m0_sanity.sh"

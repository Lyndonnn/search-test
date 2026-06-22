#!/usr/bin/env bash
set -euo pipefail

# Install the official GroundingDINO repo for DAG-IG RN03-10 grounding checks.
#
# This script intentionally does not download the checkpoint by default. The
# checkpoint is large and should be managed explicitly on the data disk.
#
# Expected checkpoint for smoke tests:
#   third_party/GroundingDINO_weights/groundingdino_swint_ogc.pth
#
# Usage:
#   bash scripts/dagig_train/00_setup_groundingdino_env.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THIRD_PARTY_DIR="${THIRD_PARTY_DIR:-${REPO_ROOT}/third_party}"
GROUNDINGDINO_REPO="${GROUNDINGDINO_REPO:-${THIRD_PARTY_DIR}/GroundingDINO}"
GROUNDINGDINO_GIT_URL="${GROUNDINGDINO_GIT_URL:-https://github.com/IDEA-Research/GroundingDINO.git}"
GROUNDINGDINO_WEIGHTS_DIR="${GROUNDINGDINO_WEIGHTS_DIR:-${THIRD_PARTY_DIR}/GroundingDINO_weights}"
GROUNDINGDINO_CHECKPOINT="${GROUNDINGDINO_CHECKPOINT:-${GROUNDINGDINO_WEIGHTS_DIR}/groundingdino_swint_ogc.pth}"

echo "repo_root=${REPO_ROOT}"
echo "third_party_dir=${THIRD_PARTY_DIR}"
echo "groundingdino_repo=${GROUNDINGDINO_REPO}"
echo "groundingdino_checkpoint=${GROUNDINGDINO_CHECKPOINT}"

mkdir -p "${THIRD_PARTY_DIR}" "${GROUNDINGDINO_WEIGHTS_DIR}"

if [[ ! -d "${GROUNDINGDINO_REPO}/.git" ]]; then
  echo "Cloning GroundingDINO..."
  git clone "${GROUNDINGDINO_GIT_URL}" "${GROUNDINGDINO_REPO}"
else
  echo "GroundingDINO repo already exists."
fi

echo "Installing GroundingDINO editable package..."
python3 -m pip install -e "${GROUNDINGDINO_REPO}"

CONFIG_PATH="${GROUNDINGDINO_REPO}/groundingdino/config/GroundingDINO_SwinT_OGC.py"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: GroundingDINO config missing: ${CONFIG_PATH}" >&2
  exit 2
fi

if [[ ! -f "${GROUNDINGDINO_CHECKPOINT}" ]]; then
  cat >&2 <<EOF
WARNING: checkpoint is missing:
  ${GROUNDINGDINO_CHECKPOINT}

Place the checkpoint there before running:
  python3 scripts/dagig_train/01_groundingdino_smoke_test.py

This script does not auto-download the checkpoint to avoid accidental storage
pressure and unverified mirrors.
EOF
else
  echo "Checkpoint exists: ${GROUNDINGDINO_CHECKPOINT}"
fi

echo "GroundingDINO setup finished."

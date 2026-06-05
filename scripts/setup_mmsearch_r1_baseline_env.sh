#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f projects/dagig_mmsearch/scripts/autodl_env.sh ]]; then
  # shellcheck disable=SC1091
  source projects/dagig_mmsearch/scripts/autodl_env.sh
fi

MMSEARCH_R1_VERL_COMMIT="8e9e73723fd1cc729bedb3bbcf915060afbda91d"
MMSEARCH_R1_VERL_ROOT="${MMSEARCH_R1_VERL_ROOT:-$ROOT/third_party/mmsearch_r1_verl}"
MMSEARCH_R1_VENV="${MMSEARCH_R1_VENV:-$ROOT/.venv-mmsearch-r1}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"

mkdir -p "$ROOT/third_party"

if [[ ! -d "$MMSEARCH_R1_VERL_ROOT/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/volcengine/verl.git "$MMSEARCH_R1_VERL_ROOT"
fi

git -C "$MMSEARCH_R1_VERL_ROOT" fetch --depth=1 origin "$MMSEARCH_R1_VERL_COMMIT"
git -C "$MMSEARCH_R1_VERL_ROOT" checkout --detach "$MMSEARCH_R1_VERL_COMMIT"

if [[ ! -x "$MMSEARCH_R1_VENV/bin/python" ]]; then
  python3 -m venv "$MMSEARCH_R1_VENV"
fi

# shellcheck disable=SC1091
source "$MMSEARCH_R1_VENV/bin/activate"
python -m pip install -U pip setuptools wheel packaging ninja
python -m pip install --upgrade --force-reinstall -r requirements-mmsearch-r1.txt
python -m pip uninstall -y verl || true
python -m pip install --no-deps -e "$MMSEARCH_R1_VERL_ROOT"

if [[ "$INSTALL_FLASH_ATTN" == "1" ]]; then
  MAX_JOBS="${MAX_JOBS:-4}" python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
fi

export ROOT
# shellcheck disable=SC1091
source scripts/mmsearch_r1_env.sh

python -m pip check
python scripts/check_mmsearch_cuda_stack.py \
  --require-nccl \
  --require-vllm \
  --require-exact-verl \
  --require-locked-versions

echo "MMSearch-R1 baseline environment is ready."
echo "The MMSearch-R1 commands activate the isolated environment automatically."
echo "Next run:"
echo "  make mmsearch_cuda_preflight"
echo "  make mmsearch_val_only"

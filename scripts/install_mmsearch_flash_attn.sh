#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f projects/dagig_mmsearch/scripts/autodl_env.sh ]]; then
  # shellcheck disable=SC1091
  source projects/dagig_mmsearch/scripts/autodl_env.sh
fi

# shellcheck disable=SC1091
source scripts/mmsearch_r1_env.sh

FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.4.post1}"
MAX_JOBS="${MAX_JOBS:-4}"
FLASH_ATTN_TMPDIR="${FLASH_ATTN_TMPDIR:-${DAGIG_DATA_ROOT:-$ROOT/data/cache/dagig_local}/tmp/flash_attn}"
FLASH_ATTN_PIP_CACHE_DIR="${FLASH_ATTN_PIP_CACHE_DIR:-${DAGIG_DATA_ROOT:-$ROOT/data/cache/dagig_local}/pip_cache}"

mkdir -p "$FLASH_ATTN_TMPDIR" "$FLASH_ATTN_PIP_CACHE_DIR"
export TMPDIR="$FLASH_ATTN_TMPDIR"
export TEMP="$FLASH_ATTN_TMPDIR"
export TMP="$FLASH_ATTN_TMPDIR"
export PIP_CACHE_DIR="$FLASH_ATTN_PIP_CACHE_DIR"

python3 - <<'PY'
import platform
import torch

print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"capability={torch.cuda.get_device_capability(0)}")
PY

python3 -m pip install -U "setuptools<81" wheel packaging ninja
MAX_JOBS="$MAX_JOBS" python3 -m pip install \
  "flash-attn==$FLASH_ATTN_VERSION" \
  --no-build-isolation \
  --no-cache-dir

python3 - <<'PY'
from importlib.metadata import version

import flash_attn  # noqa: F401
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input  # noqa: F401

print(f"flash_attn={version('flash-attn')}")
print("flash_attn_bert_padding=ok")
PY

echo "flash-attn is ready for MMSearch-R1."

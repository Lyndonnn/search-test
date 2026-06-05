#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

# shellcheck disable=SC1091
source scripts/mmsearch_r1_env.sh

python3 scripts/run_m0_sanity_nohydra.py "$@"

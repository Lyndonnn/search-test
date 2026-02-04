#!/usr/bin/env bash
set -euo pipefail

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export HYDRA_CONFIG_PATH="${HYDRA_CONFIG_PATH:-configs}"

python3 -m mmsearch_r1.trainer.multimodal.main_ppo +exp=m0_sanity "$@"

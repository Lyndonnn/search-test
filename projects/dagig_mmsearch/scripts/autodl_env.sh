#!/usr/bin/env bash

# Source this file on AutoDL before running model downloads or long jobs:
#   source projects/dagig_mmsearch/scripts/autodl_env.sh

if [ -z "${DAGIG_DATA_ROOT:-}" ]; then
  if [ -d /root/autodl-tmp ] && [ -w /root/autodl-tmp ]; then
    export DAGIG_DATA_ROOT=/root/autodl-tmp/dagig
  else
    export DAGIG_DATA_ROOT="$(pwd)/data/cache/dagig_local"
  fi
fi
export HF_HOME="${HF_HOME:-$DAGIG_DATA_ROOT/hf_cache}"
if [ -z "${HF_ENDPOINT:-}" ] && [ -d /root/autodl-tmp ]; then
  export HF_ENDPOINT="https://hf-mirror.com"
fi
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$DAGIG_DATA_ROOT/xdg_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$DAGIG_DATA_ROOT/matplotlib}"
export WANDB_DIR="${WANDB_DIR:-$DAGIG_DATA_ROOT/logs/wandb}"
export DAGIG_RESULTS_DIR="${DAGIG_RESULTS_DIR:-$DAGIG_DATA_ROOT/results}"
export DAGIG_LOGS_DIR="${DAGIG_LOGS_DIR:-$DAGIG_DATA_ROOT/logs}"
export DAGIG_MODEL_CACHE="${DAGIG_MODEL_CACHE:-$HF_HOME}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p \
  "$DAGIG_DATA_ROOT" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$XDG_CACHE_HOME" \
  "$MPLCONFIGDIR" \
  "$WANDB_DIR" \
  "$DAGIG_RESULTS_DIR" \
  "$DAGIG_LOGS_DIR"

echo "DAGIG_DATA_ROOT=$DAGIG_DATA_ROOT"
echo "HF_HOME=$HF_HOME"
echo "HF_ENDPOINT=${HF_ENDPOINT:-<default huggingface.co>}"
echo "DAGIG_RESULTS_DIR=$DAGIG_RESULTS_DIR"

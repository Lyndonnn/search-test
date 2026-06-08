#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

mkdir -p data/raw data/processed data/indexes data/cache logs results/baselines results/direct_vqa results/prompted_search results/outcome_rl results/local_ig results/dagig_lite paper_artifacts/tables paper_artifacts/figures paper_artifacts/case_studies third_party

if [ -d /root/autodl-tmp ]; then
  mkdir -p /root/autodl-tmp/dagig/{hf_cache,logs,results,xdg_cache,matplotlib}
  echo "Detected AutoDL data disk: /root/autodl-tmp"
  echo "For model downloads and long jobs, run:"
  echo "  source projects/dagig_mmsearch/scripts/autodl_env.sh"
fi

echo "DAG-IG setup directories created under $ROOT"
echo "For reference-policy DAG-IG scoring, run once: make dagig_setup_reference_env"
echo "For MMSearch-R1 baseline training, run: make mmsearch_setup_baseline"

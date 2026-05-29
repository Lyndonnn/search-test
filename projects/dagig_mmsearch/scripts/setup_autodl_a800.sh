#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

mkdir -p data/raw data/processed data/indexes data/cache logs results/baselines results/direct_vqa results/prompted_search results/outcome_rl results/local_ig results/dagig_lite paper_artifacts/tables paper_artifacts/figures paper_artifacts/case_studies third_party

echo "DAG-IG setup directories created under $ROOT"
echo "Install dependencies manually when needed: pip install -r requirements.txt && pip install -e ./verl"


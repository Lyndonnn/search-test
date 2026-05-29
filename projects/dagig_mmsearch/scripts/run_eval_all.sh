#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

bash projects/dagig_mmsearch/scripts/run_direct_vqa.sh
bash projects/dagig_mmsearch/scripts/run_prompted_search.sh
bash projects/dagig_mmsearch/scripts/run_outcome_rl.sh
bash projects/dagig_mmsearch/scripts/run_local_ig.sh
bash projects/dagig_mmsearch/scripts/run_dagig_lite.sh
bash projects/dagig_mmsearch/scripts/make_tables.sh
bash projects/dagig_mmsearch/scripts/make_figures.sh

echo "all DAG-IG smoke evaluations completed"


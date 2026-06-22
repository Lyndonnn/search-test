#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONDA_ENV="${CONDA_ENV:-dagig-sft}"
SMOKE_GPU="${SMOKE_GPU:-0}"
EVAL_GPU="${EVAL_GPU:-0}"

echo "=== 1. artifact check ==="
required_paths=(
  "checkpoints/dagig_rn03_10_grounded/ground_action_sft_7b_lora"
  "data/dagig_rn03_10_grounded/ground_action_train.jsonl"
  "data/dagig_rn03_10_grounded/ground_action_dev.jsonl"
  "data/dagig_rn03_10_grounded/ground_action_test.jsonl"
  "results/dagig_rn03_10_grounded/grounding/final_train/grounding_results.jsonl"
  "results/dagig_rn03_10_grounded/grounding/final_dev/grounding_results.jsonl"
  "results/dagig_rn03_10_grounded/grounding/final_test/grounding_results.jsonl"
)
for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "MISSING: $path" >&2
    exit 2
  fi
  ls -lh "$path"
done

echo "=== 2. build grounded RL data ==="
conda run -n "$CONDA_ENV" python scripts/dagig_train/24_build_grounded_rl_data.py

echo "=== 3. leakage inspection ==="
conda run -n "$CONDA_ENV" python scripts/dagig_train/25_inspect_grounded_rl_leakage.py

echo "=== 4. reward scorer smoke on 5 teacher examples ==="
CUDA_VISIBLE_DEVICES="$SMOKE_GPU" conda run -n "$CONDA_ENV" python scripts/dagig_train/26_score_grounded_rollouts.py \
  --rl_data_jsonl data/dagig_rn03_10_grounded_rl/grounded_rl_dev.jsonl \
  --output_jsonl results/dagig_rn03_10_grounded_rl/reward_smoke/scored_teacher_dev5.jsonl \
  --summary_json results/dagig_rn03_10_grounded_rl/reward_smoke/summary_teacher_dev5.json \
  --reward_mode dagig_grounded \
  --limit 5 \
  --use_target_rollouts \
  --device cuda

echo "=== 5. low-budget grounded RL variants ==="
bash scripts/dagig_train/27_run_grounded_rl_low_budget.sh

echo "=== 6. dev/test evaluation and final report ==="
CUDA_VISIBLE_DEVICES="$EVAL_GPU" conda run -n "$CONDA_ENV" python scripts/dagig_train/28_eval_grounded_rl.py \
  --data_dir data/dagig_rn03_10_grounded_rl \
  --result_root results/dagig_rn03_10_grounded_rl \
  --device cuda

echo
echo "GROUNDED DAG-IG RL PIPELINE FINISHED"
echo
echo "Key outputs:"
echo "- RL data: data/dagig_rn03_10_grounded_rl/"
echo "- leakage report: results/dagig_rn03_10_grounded_rl/leakage/leakage_report.md"
echo "- RL checkpoints: checkpoints/dagig_rn03_10_grounded_rl/"
echo "- dev metrics: results/dagig_rn03_10_grounded_rl/eval/grounded_rl_dev_metrics.csv"
echo "- test metrics: results/dagig_rn03_10_grounded_rl/eval/grounded_rl_test_metrics.csv"
echo "- reward component table: results/dagig_rn03_10_grounded_rl/eval/grounded_rl_reward_components.csv"
echo "- comparison table: results/dagig_rn03_10_grounded_rl/tables/grounded_rl_comparison.md"
echo "- final report: results/dagig_rn03_10_grounded_rl/GROUNDED_DAGIG_RL_REPORT.md"

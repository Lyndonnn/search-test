#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-/data/zhengxiang/code/dagig/models/Qwen2.5-VL-7B-Instruct}"
CONDA_ENV="${CONDA_ENV:-dagig-sft}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints/dagig_train}"
RESULT_ROOT="${RESULT_ROOT:-results/dagig_train}"
LOG_ROOT="${LOG_ROOT:-logs/dagig_train}"
DATA_DIR="${DATA_DIR:-data/dagig_sft}"
PACKAGE_DIR="${PACKAGE_DIR:-data/pix2fact_dagig_1k_gpt54_teacher_clean_package}"
MAIN_FILE="${MAIN_FILE:-data/pix2fact_dagig_train_AB_clean_split.jsonl}"
RETRIEVAL_DIR="${RETRIEVAL_DIR:-data/dagig_retrieval}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_PIXELS="${MAX_PIXELS:-262144}"
LIMIT="${LIMIT:-0}"

IFS=' ' read -r -a VARIANTS <<< "${VARIANTS:-uniform_sft outcome_only_sft local_ig_sft dagig_sft dagig_action_only_sft}"
IFS=' ' read -r -a GPUS <<< "${GPUS:-0 1 2 3 4}"

if [ "${#GPUS[@]}" -lt "${#VARIANTS[@]}" ]; then
  echo "Need at least as many GPUS as VARIANTS: ${#GPUS[@]} < ${#VARIANTS[@]}" >&2
  exit 2
fi

mkdir -p "$RESULT_ROOT" "$LOG_ROOT"

pids=()
names=()
for idx in "${!VARIANTS[@]}"; do
  variant="${VARIANTS[$idx]}"
  gpu="${GPUS[$idx]}"
  eval_file="${DATA_DIR}/${variant}_dev.jsonl"
  adapter_dir="${CKPT_ROOT}/${variant}_7b_lora"
  output_csv="${RESULT_ROOT}/${variant}_7b_eval.csv"
  log_file="${LOG_ROOT}/eval_${variant}_7b.log"
  if [ ! -f "${adapter_dir}/adapter_config.json" ]; then
    echo "Missing adapter: ${adapter_dir}/adapter_config.json" >&2
    exit 2
  fi
  echo "launch eval ${variant} on GPU ${gpu}; log=${log_file}"
  (
    env CUDA_VISIBLE_DEVICES="$gpu" TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
      conda run -n "$CONDA_ENV" python scripts/dagig_train/03_eval_chain.py \
        --eval_file "$eval_file" \
        --output_csv "$output_csv" \
        --model_name_or_path "$MODEL" \
        --adapter_dir "$adapter_dir" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --max_pixels "$MAX_PIXELS" \
        --limit "$LIMIT"
  ) >"$log_file" 2>&1 &
  pids+=("$!")
  names+=("$variant")
done

status=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  name="${names[$idx]}"
  if wait "$pid"; then
    echo "done eval ${name}"
  else
    echo "failed eval ${name}; tail follows" >&2
    tail -n 80 "${LOG_ROOT}/eval_${name}_7b.log" >&2 || true
    status=1
  fi
done
if [ "$status" -ne 0 ]; then
  exit "$status"
fi

for variant in "${VARIANTS[@]}"; do
  python3 scripts/dagig_train/06_score_rollouts.py \
    --examples_jsonl "${DATA_DIR}/${variant}_dev.jsonl" \
    --details_jsonl "${RESULT_ROOT}/${variant}_7b_eval_details.jsonl" \
    --corpus_jsonl "${RETRIEVAL_DIR}/corpus.jsonl" \
    --targets_json "${RETRIEVAL_DIR}/targets.json" \
    --output_jsonl "${RESULT_ROOT}/${variant}_7b_reward_details.jsonl" \
    --summary_csv "${RESULT_ROOT}/${variant}_7b_reward_summary.csv"

  python3 scripts/dagig_train/04_eval_query_retrieval.py \
    --package_dir "$PACKAGE_DIR" \
    --main_file "$MAIN_FILE" \
    --details_csv "${RESULT_ROOT}/${variant}_7b_eval_details.csv" \
    --output_csv "${RESULT_ROOT}/${variant}_7b_query_retrieval.csv" \
    --dedupe \
    --corpus_mode evidence_only
done

failure_args=()
for variant in "${VARIANTS[@]}"; do
  failure_args+=(--details_jsonl "${RESULT_ROOT}/${variant}_7b_reward_details.jsonl")
done
python3 scripts/dagig_train/08_failure_analysis.py "${failure_args[@]}" --out_dir "${RESULT_ROOT}/failure_analysis"
python3 scripts/dagig_train/07_make_tables.py --results_dir "$RESULT_ROOT" --out_dir "${RESULT_ROOT}/tables"

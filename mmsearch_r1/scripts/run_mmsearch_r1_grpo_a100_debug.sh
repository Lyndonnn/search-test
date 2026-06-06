#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

if [[ -f projects/dagig_mmsearch/scripts/autodl_env.sh ]]; then
  # shellcheck disable=SC1091
  source projects/dagig_mmsearch/scripts/autodl_env.sh
fi

# shellcheck disable=SC1091
source scripts/mmsearch_r1_env.sh
python3 scripts/patch_mmsearch_r1_verl_flash_attn.py --reset-first --verl-root "$MMSEARCH_R1_VERL_ROOT"

if [[ "${MMSEARCH_SKIP_PREFLIGHT:-0}" != "1" ]]; then
  python3 scripts/check_mmsearch_cuda_stack.py \
    --require-nccl \
    --require-vllm \
    --require-exact-verl \
    --require-locked-versions
fi

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-mmsearch_r1/data/fvqa_debug_train.pq}"
VAL_DATA_PATH="${VAL_DATA_PATH:-mmsearch_r1/data/fvqa_debug_val.pq}"
MODEL_PATH="${MMSEARCH_MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
N_GPUS="${N_GPUS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
ROLLOUT_N="${ROLLOUT_N:-2}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$((TRAIN_BATCH_SIZE * ROLLOUT_N))}"

if [[ ! -f "$TRAIN_DATA_PATH" || ! -f "$VAL_DATA_PATH" ]]; then
  echo "Missing FVQA parquet. Run: make mmsearch_prepare_fvqa_debug"
  exit 1
fi

if [[ -z "${SERPAPI_API_KEY:-}" && -z "${MMSEARCH_OFFLINE_PARQUET:-}" ]]; then
  export MMSEARCH_OFFLINE_PARQUET="$TRAIN_DATA_PATH"
fi

python3 -m mmsearch_r1.trainer.multimodal.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_DATA_PATH" \
  data.val_files="$VAL_DATA_PATH" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.num_workers="${DATA_NUM_WORKERS:-0}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-2048}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-512}" \
  data.image_key=images \
  data.user_prompt_round_1=mmsearch_r1/prompts/round_1_user_prompt_qwenvl.pkl \
  data.user_prompt_after_image_search=mmsearch_r1/prompts/after_image_search_prompt_qwenvl.pkl \
  data.user_prompt_after_text_search=mmsearch_r1/prompts/after_text_search_prompt_qwenvl.pkl \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.attn_implementation="${ATTN_IMPLEMENTATION:-eager}" \
  actor_rollout_ref.model.disable_monkey_patch="${DISABLE_MONKEY_PATCH:-True}" \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-6}" \
  actor_rollout_ref.actor.optim.lr_sigmoid_decay_warmup=False \
  actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING:-False}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF:-0}" \
  actor_rollout_ref.actor.use_multi_turn_response_mask=True \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.001}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm_multiturn_mmsearch \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.45}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager="${VLLM_ENFORCE_EAGER:-True}" \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.max_gen_round="${MAX_GEN_ROUND:-2}" \
  actor_rollout_ref.rollout.response_length_total="${RESPONSE_LENGTH_TOTAL:-2048}" \
  actor_rollout_ref.rollout.search.topk="${SEARCH_TOPK:-3}" \
  actor_rollout_ref.rollout.search.image_search_limit="${IMAGE_SEARCH_LIMIT:-1}" \
  actor_rollout_ref.rollout.search.text_search_limit="${TEXT_SEARCH_LIMIT:-2}" \
  actor_rollout_ref.rollout.search.parallel_tool_call=False \
  actor_rollout_ref.rollout.search.parallel_tool_call_threads="${PARALLEL_TOOL_CALL_THREADS:-2}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  actor_rollout_ref.rollout.micro_batch_size_per_gpu="${ROLLOUT_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.kl_ctrl.kl_coef="${KL_CTRL_COEF:-0.001}" \
  algorithm.filter_groups.enable=False \
  trainer.logger=['console'] \
  trainer.project_name="${WANDB_PROJECT_NAME:-dagig_mmsearch_r1}" \
  trainer.experiment_name="${WANDB_EXP_NAME:-mmsearch_r1_grpo_a100_debug}" \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ:-20}" \
  trainer.test_freq="${TEST_FREQ:-10}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-20}" \
  trainer.critic_warmup=0 \
  trainer.search_penalty="${SEARCH_PENALTY:-0.1}" \
  trainer.format_penalty="${FORMAT_PENALTY:-0.1}" \
  trainer.reward_mode="${REWARD_MODE:-EM}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN:-True}" \
  trainer.val_generations_to_log_to_wandb="${VAL_GENERATIONS_TO_LOG:-0}" \
  trainer.default_local_dir="${CHECKPOINT_DIR:-checkpoints/mmsearch_r1/grpo_a100_debug}" \
  trainer.rollout_log_dir="${ROLLOUT_LOG_DIR:-logs/mmsearch_r1/grpo_a100_debug}"

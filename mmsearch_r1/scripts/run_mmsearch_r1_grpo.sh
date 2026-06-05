# This script is the main MMSearch-R1 GRPO entrypoint.
# It keeps the original 7B/8-GPU defaults, but no longer assumes the
# upstream repo directory is named `multimodal-search-r1`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# shellcheck disable=SC1091
source scripts/mmsearch_r1_env.sh

if [[ "${MMSEARCH_SKIP_PREFLIGHT:-0}" != "1" ]]; then
    python3 scripts/check_mmsearch_cuda_stack.py \
        --require-nccl \
        --require-vllm \
        --require-exact-verl \
        --require-locked-versions
fi

TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-mmsearch_r1/data/fvqa_debug_train.pq}"
VAL_DATA_PATH="${VAL_DATA_PATH:-mmsearch_r1/data/fvqa_debug_val.pq}"
WANDB_PROJECT_NAME="${WANDB_PROJECT_NAME:-mmsearch_r1}"
WANDB_EXP_NAME="${WANDB_EXP_NAME:-mmsearch_r1_grpo}"
MODEL_PATH="${MMSEARCH_MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
N_GPUS="${N_GPUS:-8}"

if [[ -z "${SERPAPI_API_KEY:-}" && -z "${MMSEARCH_OFFLINE_PARQUET:-}" && -f "$TRAIN_DATA_PATH" ]]; then
    export MMSEARCH_OFFLINE_PARQUET="$TRAIN_DATA_PATH"
fi

python3 -m mmsearch_r1.trainer.multimodal.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA_PATH \
    data.val_files=$VAL_DATA_PATH \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
    data.image_key=images \
    data.user_prompt_round_1=mmsearch_r1/prompts/round_1_user_prompt_qwenvl.pkl \
    data.user_prompt_after_image_search=mmsearch_r1/prompts/after_image_search_prompt_qwenvl.pkl \
    data.user_prompt_after_text_search=mmsearch_r1/prompts/after_text_search_prompt_qwenvl.pkl \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_sigmoid_decay_warmup=True \
    actor_rollout_ref.actor.optim.lr_sigmoid_decay_ratio=0.95 \
    actor_rollout_ref.actor.optim.lr_sigmoid_decay_warmup_steps=45 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_multi_turn_response_mask=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm_multiturn_mmsearch \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.max_gen_round=3 \
    actor_rollout_ref.rollout.response_length_total=8192 \
    actor_rollout_ref.rollout.search.topk=5 \
    actor_rollout_ref.rollout.search.image_search_limit=1 \
    actor_rollout_ref.rollout.search.text_search_limit=2 \
    actor_rollout_ref.rollout.search.parallel_tool_call=True \
    actor_rollout_ref.rollout.search.parallel_tool_call_threads=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$WANDB_PROJECT_NAME \
    trainer.experiment_name=$WANDB_EXP_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=30 \
    +trainer.search_penalty=0.1 \
    +trainer.format_penalty=0.1 \
    +trainer.reward_mode="EM" \
    +trainer.val_before_train=True \
    +algorithm.filter_groups.enable=False

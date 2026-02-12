#!/usr/bin/env bash
set -euo pipefail

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONPATH="${PYTHONPATH:-$(pwd)}"

# Default to HF rollout for maximum compatibility in Colab.
# Set MMSEARCH_M0_USE_VLLM=1 to switch to vLLM multi-turn rollout.
if [[ "${MMSEARCH_M0_USE_VLLM:-0}" == "1" ]]; then
  ROLLOUT_NAME="vllm_multiturn_mmsearch"
  MULTI_TURN_MASK="True"
  EXTRA_ROLLOUT_ARGS=(
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.max_gen_round=1
    actor_rollout_ref.rollout.response_length_total=128
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.search.image_search_limit=0
    actor_rollout_ref.rollout.search.text_search_limit=0
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.fsdp_config.param_offload=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_seqs=1
    actor_rollout_ref.rollout.max_num_batched_tokens=1024
    actor_rollout_ref.rollout.max_model_len=896
  )
else
  ROLLOUT_NAME="hf"
  MULTI_TURN_MASK="False"
  EXTRA_ROLLOUT_ARGS=()
fi

python3 -m mmsearch_r1.trainer.multimodal.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="mmsearch_r1/data/mini_data.pq" \
  data.val_files="mmsearch_r1/data/mini_data.pq" \
  data.train_batch_size=1 \
  data.max_prompt_length=1024 \
  data.max_response_length=64 \
  data.image_key=images \
  data.user_prompt_round_1=mmsearch_r1/prompts/round_1_user_prompt_qwenvl.pkl \
  data.user_prompt_after_text_search=mmsearch_r1/prompts/round_1_user_prompt_qwenvl.pkl \
  data.user_prompt_after_image_search=mmsearch_r1/prompts/round_1_user_prompt_qwenvl.pkl \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
  actor_rollout_ref.actor.optim.lr=2e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_multi_turn_response_mask=${MULTI_TURN_MASK} \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
  trainer.logger=['console'] \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=5 \
  trainer.save_freq=1000000 \
  trainer.test_freq=1000000 \
  +trainer.search_penalty=0.1 \
  +trainer.format_penalty=0.1 \
  +trainer.reward_mode=EM \
  +trainer.val_before_train=False \
  +algorithm.filter_groups.enable=False \
  "${EXTRA_ROLLOUT_ARGS[@]}" \
  "$@"

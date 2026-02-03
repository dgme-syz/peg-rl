#!/usr/bin/env bash
set -euo pipefail

model_path="models/Qwen3-4B"
ppo_mini_batch_size=128 
ppo_micro_batch_size_per_gpu=32 # 32 for A6000
project_name="weight"
save_freq=50

train_batch_size=128
rollout_num=8
num_gpus=8
datetime=$(date +%Y%m%d_%H%M%S)
mul_times=1
model_tag="(pe,mt)=(M,0)"
exp_name="bs@${train_batch_size}_n@${rollout_num}_m@${mul_times}_@${datetime}_@${model_tag}_@${num_gpus}gpus"
dir=./data

# fi
train_file_path=$dir/train/wmt_en2fi_1k.parquet
test_file_path="[$dir/test/wmt24_en-fi_FI.parquet,$dir/test/flores_en2fi.parquet]"
export RAY_raylet_start_wait_time_s=120 

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    ++algorithm.leaf_score_only=False \
    ++algorithm.grpo_child_score_merge_fn=mean \
    ++algorithm.qe_weight=0.0 \
    ++algorithm.merge_weight=1.0 \
    ++algorithm.remove_runtime_qe=False \
    data.train_files=$train_file_path \
    data.val_files=$test_file_path \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=768 \
    data.max_response_length=512 \
    ++data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=256 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.82 \
    actor_rollout_ref.rollout.n=${rollout_num} \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=256 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    ++reward_model.train_reward_manager="mt_train" \
    ++reward_model.val_reward_manager="mt_val" \
    ++ray_kwargs.ray_init.ignore_reinit_error=True \
    ++workflow.repeat_times=8 \
    ++workflow.data_divisor=1 \
    ++workflow.mt_only=False \
	++workflow.test_mt_only=False \
    ++workflow.tokenizer_path=$model_path \
    ++workflow.use_test_prompt=True \
    ++workflow.dynamic_mode=False \
    ++workflow.remove_gradient_depth_list=[1] \
    comet_model.n=1 \
    embedding.enable=False \
    custom_reward_function.reward_kwargs.mul_times=${mul_times} \
    custom_reward_function.reward_kwargs.score_limit=95 \
    custom_reward_function.reward_kwargs.thinking_check=False \
    trainer.val_before_train=True \
    trainer.logger="[console,tensorboard]" \
    trainer.project_name=$project_name \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=$num_gpus \
    trainer.nnodes=1 \
    trainer.default_local_dir="${exp_name}" \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=$save_freq \
    trainer.test_freq=10 \
    trainer.total_epochs=15

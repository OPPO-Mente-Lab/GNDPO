#!/bin/bash
set -x

# Set the task name
CURRENT_PATH=$(pwd)
PROJECT_NAME=internvl3_5_4b_custom
TASK_NAME=internvl3_5_4b_on_policy_distill_bs32_minibs32_n16_geo3k_0421
TASK_NAME="${TASK_NAME%.*}"
echo "TASK_NAME: $TASK_NAME"
echo "PROJECT_NAME: $PROJECT_NAME"

export OUTPUT_PATH=${CURRENT_PATH}/verl_internvl_work_dirs/${PROJECT_NAME}/${TASK_NAME}
export TENSORBOARD_DIR=${OUTPUT_PATH}/tensorboard
export JOBLOG=${OUTPUT_PATH}/training.log

# Create output directory if it does not exist
mkdir -p ${OUTPUT_PATH}
mkdir -p ${TENSORBOARD_DIR}

# Set up environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Dynamically compute the number of nodes and total GPU engines
NUM_GPUS_PER_NODE=8
MICRO_TRAIN_BATCH_SIZE=2
MICRO_ROLLOUT_BATCH_SIZE=2
ROLLOUT_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=16
TENSOR_PARALLEL=1
SEQUENCE_PARALLEL=1
PPO_MINI_BATCH_SIZE=32

NPROC_PER_NODE=8
use_dynamic_bsz=True

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=on_policy_distill \
    data.train_files=data/geo3k/train.parquet \
    data.val_files=data/geo3k/test.parquet \
    data.train_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.max_prompt_length=8192 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=8 \
    data.truncation='error' \
    data.image_key=images \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path=OpenGVLab/InternVL3_5-4B-Instruct \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_TRAIN_BATCH_SIZE} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${SEQUENCE_PARALLEL} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_ROLLOUT_BATCH_SIZE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_PARALLEL} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=${N_SAMPLES_PER_PROMPT} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_TRAIN_BATCH_SIZE} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${SEQUENCE_PARALLEL} \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    reward_model.enable=True \
    reward_model.strategy=fsdp \
    reward_model.on_policy_distill=True \
    reward_model.use_dynamic_bsz=True \
    reward_model.forward_max_token_len_per_gpu=32768 \
    reward_model.micro_batch_size_per_gpu=1 \
    reward_model.ulysses_sequence_parallel_size=${SEQUENCE_PARALLEL} \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.fsdp_size=16 \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.model.path=OpenGVLab/InternVL3_5-8B \
    reward_model.model.input_tokenizer=null \
    reward_model.model.trust_remote_code=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.default_local_dir=${OUTPUT_PATH} \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${TASK_NAME} \
    trainer.n_gpus_per_node=${NPROC_PER_NODE} \
    trainer.nnodes=4 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.validation_data_dir=${OUTPUT_PATH}/val_data \
    trainer.val_before_train=True \
    trainer.rollout_data_dir=${OUTPUT_PATH}/rollouts \
    trainer.total_epochs=1 2>&1 | tee ${JOBLOG}

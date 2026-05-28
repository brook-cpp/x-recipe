#!/bin/bash
set -xeuo pipefail
mkdir -p logs

########################### Quick Config ###########################
project_name='GRPO-Qwen3.5-35b-A3B-BASE-MATH'
exp_name='30B_test_megatron'
adv_estimator=grpo
RAY_ADDRESS=http://[fdbd:dc02:2a:204::22]:9764
WORKING_DIR=./
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-2}
NPUS_PER_NODE=${NPUS_PER_NODE:-16}

# Paths
# very important! please modify the max_position_embeddings in config.json to 32768 after downloading from huggingface
MODEL_PATH=/mnt/bn/bandai-hl/shared/models/Qwen3-30B-A3B
TRAIN_FILE=/mnt/bn/bandai-hl/shared/data/dapo-math/dapo-math-17k.parquet
TEST_FILE=/mnt/bn/bandai-hl/shared/data/dapo-math/aime-2024.parquet
CKPTS_DIR=./

########################### Parameter Arrays ###########################
train_batch_size=${train_batch_size:-64} # 256 512
ppo_mini_batch_size=${ppo_mini_batch_size:-32}
# Response length parameters
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 20))
enable_overlong_buffer=True #False #True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
filter_overlong_prompts_workers=64

DATA=(
    data.train_files=${TRAIN_FILE}
    data.val_files=${TEST_FILE}
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.truncation='error'
    data.filter_overlong_prompts=${enable_overlong_buffer}
    data.filter_overlong_prompts_workers=${filter_overlong_prompts_workers}
)
use_remove_padding=${use_remove_padding:-True} # True False
MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding}
)
use_dynamic_bsz=True # False True
COMMON_ACTOR=(
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
)

rollout_name="vllm"
gen_tp=${gen_tp:-4}
gpu_memory_utilization=${gpu_memory_utilization:-0.75}
n_resp_per_prompt=${n_resp_per_prompt:-8}
enforce_eager=${enforce_eager:-False} # True False
rollout_dtype=${rollout_dtype:-bfloat16} # bfloat16 float16
ROLLOUT=(
    actor_rollout_ref.rollout.name=${rollout_name}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.enforce_eager=${enforce_eager}
    actor_rollout_ref.rollout.dtype=${rollout_dtype}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.rollout.skip.enable=False
    actor_rollout_ref.rollout.skip.dump_dir=/mnt/bn/bandai-hl/users/lihaozhe/skip_rollout
    actor_rollout_ref.rollout.skip.max_dump_step=3
)

# actor_rollout_ref.rollout.skip.enable=False
# actor_rollout_ref.rollout.skip.dump_dir=/mnt/bn/bandai-hl/users/lihaozhe/skip_rollout
# actor_rollout_ref.rollout.skip.max_dump_step=1

COMMON_REF=(
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
)

ALGORITHM=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=False
)
train_tp=${train_tp:-2}
train_pp=${train_pp:-1}
train_cp=${train_cp:-1}
train_ep=${train_ep:-8}
train_etp=${train_etp:-1}
ALL_OFFLOAD=${ALL_OFFLOAD:-True}
train_dtype=${train_dtype:-bfloat16} # bfloat16 float16
chunk_entropy=${chunk_entropy:-False}
MEGATRON_ACTOR=(
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True
    actor_rollout_ref.actor.megatron.use_remove_padding=${use_remove_padding}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${train_etp}
    actor_rollout_ref.actor.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.optimizer_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.grad_offload=${ALL_OFFLOAD}
    actor_rollout_ref.actor.megatron.dtype=${train_dtype}
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
    actor_rollout_ref.actor.megatron.entropy_from_logits_with_chunking=${chunk_entropy}
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.01
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_z_loss_coeff=0.001
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_moe_token_permute_and_unpermute=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True

)
# +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=False
MEGATRON_REF=(
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp}
    actor_rollout_ref.ref.megatron.param_offload=${ALL_OFFLOAD}
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=False
    actor_rollout_ref.ref.megatron.entropy_from_logits_with_chunking=${chunk_entropy}
)
# rollout_rs
# rollout_is parameters
calculate_log_probs=True
rollout_is=null # token sequence
rollout_is_threshold=none # none 2.0
rollout_rs=seq_mean_k1 #token_k1/k2/k3 seq_sum_k1/k2/k3 seq_mean_k1/k2/k3 seq_max_k1/k2/k3
rollout_rs_threshold="0.999_1.001" # none 0.999 K1 KL modes (*k1): Use "lower_upper" strings (e.g. "0.7_1.3").
# rollout_rs_threshold_lower=0.999
# rollout_token_veto_threshold=1e-4
rollout_is_batch_normalize=False
ROLLOUT_IS=(
    actor_rollout_ref.rollout.calculate_log_probs=${calculate_log_probs}
    algorithm.rollout_correction.rollout_is=${rollout_is}
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold}
    algorithm.rollout_correction.rollout_is_batch_normalize=${rollout_is_batch_normalize}
    algorithm.rollout_correction.rollout_rs=${rollout_rs}
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold}
)
ref_offload=True
actor_offload=True
fsdp_size=-1 # 32
sp_size=4
FSDP_ACTOR=(
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_offload}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_offload}
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size}
    actor_rollout_ref.actor.fsdp_config.dtype=${train_dtype}
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size}
    actor_rollout_ref.actor.grad_clip=1.0
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)
FSDP_REF=(
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_offload}
    actor_rollout_ref.ref.fsdp_config.fsdp_size=${fsdp_size}
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size}
)
using_megatron=True # False #  True
# ACTOR=( "${MEGATRON_ACTOR[@]}" "${COMMON_ACTOR[@]}" "${ROLLOUT_IS[@]}" )
if [ ${using_megatron} = True ]; then
    ACTOR=( "${MEGATRON_ACTOR[@]}" "${COMMON_ACTOR[@]}" )
    REF=( "${MEGATRON_REF[@]}" "${COMMON_REF[@]}" )
    config_name=ppo_megatron_trainer.yaml
    exp_name=${exp_name}_megatron_tp${train_tp}pp${train_pp}cp${train_cp}ep${train_ep}etp${train_etp}-${rollout_rs}
else
    ACTOR=( "${FSDP_ACTOR[@]}" "${COMMON_ACTOR[@]}" )
    REF=( "${FSDP_REF[@]}" "${COMMON_REF[@]}" )
    config_name=ppo_trainer.yaml
    exp_name=${exp_name}_fsdp_sp${sp_size}
fi
exp_name=${exp_name}_${rollout_name}_tp${gen_tp}_bs${train_batch_size}

save_freq=10
test_freq=5

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${exp_name}
    trainer.n_gpus_per_node=${NPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.val_before_train=False
    trainer.test_freq=${test_freq}
    trainer.total_epochs=15
    trainer.device='npu'
    trainer.default_local_dir=${CKPTS_DIR}
)
using_profiler=False # False or True
if [ ${using_profiler} = True ]; then
    PROFILE_STEPS="[1]"
    DISCRETE=False # False or True
    PROFILE_RANKS_ALL=True # False or True
    PROFILE_RANKS="[1,2]"
    total_training_steps=2
    PROFILER_CONFIG=(
        actor_rollout_ref.actor.profiler.enable=True
        actor_rollout_ref.actor.profiler.ranks=$PROFILE_RANKS
        actor_rollout_ref.actor.profiler.all_ranks=$PROFILE_RANKS_ALL
        actor_rollout_ref.rollout.profiler.enable=True
        actor_rollout_ref.rollout.profiler.ranks=$PROFILE_RANKS
        actor_rollout_ref.rollout.profiler.all_ranks=$PROFILE_RANKS_ALL
        global_profiler.tool=nsys #nsys
        global_profiler.steps=$PROFILE_STEPS
        global_profiler.global_tool_config.nsys.discrete=$DISCRETE
        trainer.total_training_steps=${total_training_steps}
    )

    exp_name=${exp_name}_profiler
    TRAINER=( "${PROFILER_CONFIG[@]}" "${TRAINER[@]}" "trainer.experiment_name=${exp_name}")
fi
########################### Launch ###########################

ray job submit --runtime-env="${RUNTIME_ENV}" \
    --address "${RAY_ADDRESS}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name="${config_name}" \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@" | tee logs/run_35b_grpo_megatron_bs${train_batch_size}_tp${train_tp}pp${train_pp}cp${train_cp}ep${train_ep}_${max_prompt_length}_to_${max_response_length}_chunk_entropy_${chunk_entropy}_remove_padding_${use_remove_padding}.log


#!/usr/bin/env bash
# HERO training script — reproduces arXiv:2510.07242v1, Table 6 (Qwen3-4B-Base).
#
# Training regimes (set HERO_REGIME):
#   verifiable       — 2k verifiable-only samples
#   hard_to_verify   — 2k hard-to-verify-only samples
#   mixed            — 1k verifiable + 1k hard-to-verify (default, best overall)
#
# Prerequisites:
#   1. Preprocess data:
#        python examples/data_preprocess/openmathreasoning_hero.py \
#            --local_save_dir ~/data/openmathreasoning_hero
#   2. Preprocess eval benchmarks:
#        python examples/data_preprocess/hero_eval_benchmarks.py \
#            --local_save_dir ~/data/hero_eval
#   3. pip install math-verify
#
# Usage:
#   bash examples/hero/run_hero_train.sh
#   HERO_REGIME=verifiable bash examples/hero/run_hero_train.sh

set -euo pipefail
set -x

regime=${HERO_REGIME:-mixed}
data_dir=${HERO_DATA_DIR:-$HOME/data/openmathreasoning_hero}
eval_dir=${HERO_EVAL_DIR:-$HOME/data/hero_eval}

case "$regime" in
    verifiable)
        train_file="$data_dir/train_verifiable.parquet"
        val_file="$data_dir/val_verifiable.parquet"
        hero_alpha=0.05
        hero_beta=0.05
        ;;
    hard_to_verify)
        train_file="$data_dir/train_hard_to_verify.parquet"
        val_file="$data_dir/val_hard_to_verify.parquet"
        hero_alpha=0.1
        hero_beta=0.1
        ;;
    mixed)
        train_file="$data_dir/train_mixed.parquet"
        val_file="$data_dir/val_mixed.parquet"
        hero_alpha=0.1
        hero_beta=0.1
        ;;
    *)
        echo "Unknown regime: $regime. Use verifiable|hard_to_verify|mixed." >&2
        exit 1
        ;;
esac

model_path=${HERO_MODEL_PATH:-Qwen/Qwen3-4B-Base}
trust_remote_code=${HERO_TRUST_REMOTE_CODE:-True}

gpus_per_node=${HERO_GPUS_PER_NODE:-4}
nnodes=${HERO_NNODES:-8}

train_batch_size=${HERO_TRAIN_BATCH_SIZE:-512}
ppo_mini_batch_size=${HERO_PPO_MINI_BATCH_SIZE:-128}
ppo_micro_batch_size=${HERO_PPO_MICRO_BATCH_SIZE_PER_GPU:-2}

max_prompt_length=${HERO_MAX_PROMPT_LENGTH:-1024}
max_response_length=${HERO_MAX_RESPONSE_LENGTH:-8192}

rollout_n=${HERO_ROLLOUT_N:-8}
rollout_tp_size=${HERO_ROLLOUT_TP_SIZE:-2}
rollout_gpu_memory_utilization=${HERO_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}
rollout_temperature=${HERO_ROLLOUT_TEMPERATURE:-1.0}
rollout_log_prob_micro_batch_size=${HERO_ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
ref_log_prob_micro_batch_size=${HERO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
rollout_max_num_seqs=${HERO_ROLLOUT_MAX_NUM_SEQS:-128}

dense_rm_model=${HERO_DENSE_RM_MODEL:-nvidia/AceMath-7B-RM}
rm_enable_resource_pool=${HERO_RM_ENABLE_RESOURCE_POOL:-False}
rm_gpus=${HERO_RM_GPUS_PER_NODE:-$gpus_per_node}
rm_nodes=${HERO_RM_NNODES:-1}
rm_tp_size=${HERO_RM_TP_SIZE:-2}
rm_gpu_memory_utilization=${HERO_RM_GPU_MEMORY_UTILIZATION:-0.5}
rm_max_num_seqs=${HERO_RM_MAX_NUM_SEQS:-128}

hero_w_min=${HERO_W_MIN:-0.4}
hero_w_max=${HERO_W_MAX:-3.0}
hero_k=${HERO_K:-6.0}
hero_sigma_ema=${HERO_SIGMA_EMA:-0.9}

total_epochs=${HERO_TOTAL_EPOCHS:-20}
save_freq=${HERO_SAVE_FREQ:-20}
test_freq=${HERO_TEST_FREQ:-5}
experiment_name=${HERO_EXPERIMENT_NAME:-hero_${regime}_$(date +%Y%m%d_%H%M%S)}
loggers=${HERO_LOGGERS:-'["console"]'}
train_output_dir=${HERO_TRAIN_OUTPUT_DIR:-checkpoints/hero_paper_reproduction/${experiment_name}}
checkpoint_save_contents=${HERO_CHECKPOINT_SAVE_CONTENTS:-"['model','optimizer','extra','hf_model']"}
max_actor_ckpt_to_keep=${HERO_MAX_ACTOR_CKPT_TO_KEEP:-1}
max_critic_ckpt_to_keep=${HERO_MAX_CRITIC_CKPT_TO_KEEP:-1}
resume_mode=${HERO_RESUME_MODE:-disable}

val_files="['$val_file'"
for bench in math500 amc minerva olympiad hardverify_math textbook_reasoning; do
    bench_file="$eval_dir/${bench}.parquet"
    if [[ -f "$bench_file" ]]; then
        val_files="$val_files,'$bench_file'"
    fi
done
val_files="$val_files]"

if [[ ! -f "$train_file" ]]; then
    echo "Missing training data: $train_file" >&2
    echo "Run: python examples/data_preprocess/openmathreasoning_hero.py --local_save_dir $data_dir" >&2
    exit 1
fi

if [[ ! -f "$val_file" ]]; then
    echo "Missing validation data: $val_file" >&2
    echo "Run: python examples/data_preprocess/openmathreasoning_hero.py --local_save_dir $data_dir" >&2
    exit 1
fi

mkdir -p "$train_output_dir"
echo "Training output dir: $train_output_dir"

python3 -m verl.trainer.main_ppo     algorithm.adv_estimator=grpo     data.train_files="['$train_file']"     data.val_files="$val_files"     data.train_batch_size="$train_batch_size"     data.max_prompt_length="$max_prompt_length"     data.max_response_length="$max_response_length"     data.filter_overlong_prompts=True     data.truncation='error'     actor_rollout_ref.model.path="$model_path"     actor_rollout_ref.model.trust_remote_code="$trust_remote_code"     actor_rollout_ref.actor.optim.lr=1e-6     actor_rollout_ref.model.use_remove_padding=True     actor_rollout_ref.model.enable_gradient_checkpointing=True     actor_rollout_ref.actor.ppo_mini_batch_size="$ppo_mini_batch_size"     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$ppo_micro_batch_size"     actor_rollout_ref.actor.use_kl_loss=False     actor_rollout_ref.actor.kl_loss_coef=0.0     actor_rollout_ref.actor.entropy_coeff=0.0     actor_rollout_ref.actor.use_dynamic_bsz=True     actor_rollout_ref.actor.clip_ratio=0.2     actor_rollout_ref.actor.clip_ratio_high=0.28     actor_rollout_ref.actor.checkpoint.save_contents="$checkpoint_save_contents"     actor_rollout_ref.rollout.name=vllm     actor_rollout_ref.rollout.gpu_memory_utilization="$rollout_gpu_memory_utilization"     actor_rollout_ref.rollout.max_num_seqs="$rollout_max_num_seqs"     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$rollout_log_prob_micro_batch_size"     actor_rollout_ref.rollout.tensor_model_parallel_size="$rollout_tp_size"     actor_rollout_ref.rollout.n="$rollout_n"     actor_rollout_ref.rollout.temperature="$rollout_temperature"     actor_rollout_ref.rollout.top_p=0.95     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$ref_log_prob_micro_batch_size"     algorithm.use_kl_in_reward=False     reward.reward_manager.name=hero     reward.reward_model.enable=True     reward.reward_model.enable_resource_pool="$rm_enable_resource_pool"     reward.reward_model.n_gpus_per_node="$rm_gpus"     reward.reward_model.nnodes="$rm_nodes"     reward.reward_model.model_path="$dense_rm_model"     reward.reward_model.rollout.name=vllm     reward.reward_model.rollout.tensor_model_parallel_size="$rm_tp_size"     reward.reward_model.rollout.gpu_memory_utilization="$rm_gpu_memory_utilization"     reward.reward_model.rollout.max_num_seqs="$rm_max_num_seqs"     reward.reward_model.rollout.prompt_length="$max_prompt_length"     reward.reward_model.rollout.response_length="$max_response_length"     reward.reward_kwargs.hero.alpha="$hero_alpha"     reward.reward_kwargs.hero.beta="$hero_beta"     reward.reward_kwargs.hero.w_min="$hero_w_min"     reward.reward_kwargs.hero.w_max="$hero_w_max"     reward.reward_kwargs.hero.k="$hero_k"     reward.reward_kwargs.hero.sigma_ema="$hero_sigma_ema"     trainer.critic_warmup=0     trainer.logger="$loggers"     trainer.project_name='hero_paper_reproduction'     trainer.experiment_name="$experiment_name"     trainer.nnodes="$nnodes"     trainer.n_gpus_per_node="$gpus_per_node"     trainer.save_freq="$save_freq"     trainer.test_freq="$test_freq"     trainer.total_epochs="$total_epochs"     trainer.resume_mode="$resume_mode"     trainer.default_local_dir="$train_output_dir"     trainer.max_actor_ckpt_to_keep="$max_actor_ckpt_to_keep"     trainer.max_critic_ckpt_to_keep="$max_critic_ckpt_to_keep"     "$@"

#!/usr/bin/env bash
# EIF training script — EIF-based One-Step Algorithm (Hybrid_reward.pdf, Algorithm 1).
#
# Training regimes (set EIF_REGIME):
#   verifiable       — 2k verifiable-only samples
#   hard_to_verify   — 2k hard-to-verify-only samples
#   mixed            — 1k verifiable + 1k hard-to-verify (default, matches HERO)
#
# Prerequisites:
#   1. Preprocess data (same as HERO):
#        python examples/data_preprocess/openmathreasoning_hero.py \
#            --local_save_dir ~/data/openmathreasoning_hero
#   2. Preprocess eval benchmarks:
#        python examples/data_preprocess/hero_eval_benchmarks.py \
#            --local_save_dir ~/data/hero_eval
#   3. pip install math-verify openai
#   4. Set NAUTILUS_API_KEY for tau/m LLM endpoints
#
# Usage:
#   bash examples/eif/run_eif_train.sh
#   EIF_REGIME=verifiable bash examples/eif/run_eif_train.sh

set -euo pipefail
set -x

# ─── Data: aligned with HERO (arXiv:2510.07242v1) ────────────────────
regime=${EIF_REGIME:-mixed}
data_dir=${EIF_DATA_DIR:-$HOME/data/openmathreasoning_hero}
eval_dir=${EIF_EVAL_DIR:-$HOME/data/hero_eval}

case "$regime" in
    verifiable)
        train_file="$data_dir/train_verifiable.parquet"
        val_file="$data_dir/val_verifiable.parquet"
        ;;
    hard_to_verify)
        train_file="$data_dir/train_hard_to_verify.parquet"
        val_file="$data_dir/val_hard_to_verify.parquet"
        ;;
    mixed)
        train_file="$data_dir/train_mixed.parquet"
        val_file="$data_dir/val_mixed.parquet"
        ;;
    *)
        echo "Unknown regime: $regime. Use verifiable|hard_to_verify|mixed." >&2
        exit 1
        ;;
esac

# ─── Model ────────────────────────────────────────────────────────────
model_path=${EIF_MODEL_PATH:-Qwen/Qwen3-4B-Base}
trust_remote_code=${EIF_TRUST_REMOTE_CODE:-True}

# ─── Cluster layout ───────────────────────────────────────────────────
gpus_per_node=${EIF_GPUS_PER_NODE:-4}
nnodes=${EIF_NNODES:-8}

# ─── Training hyperparams ─────────────────────────────────────────────
train_batch_size=${EIF_TRAIN_BATCH_SIZE:-512}
ppo_mini_batch_size=${EIF_PPO_MINI_BATCH_SIZE:-128}
ppo_micro_batch_size=${EIF_PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
max_prompt_length=${EIF_MAX_PROMPT_LENGTH:-1024}
max_response_length=${EIF_MAX_RESPONSE_LENGTH:-8192}

# ─── Rollout ──────────────────────────────────────────────────────────
rollout_n=${EIF_ROLLOUT_N:-8}
rollout_tp_size=${EIF_ROLLOUT_TP_SIZE:-2}
rollout_gpu_memory_utilization=${EIF_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}
rollout_temperature=${EIF_ROLLOUT_TEMPERATURE:-1.0}
rollout_log_prob_micro_batch_size=${EIF_ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
ref_log_prob_micro_batch_size=${EIF_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
rollout_max_num_seqs=${EIF_ROLLOUT_MAX_NUM_SEQS:-128}
rollout_max_num_batched_tokens=${EIF_ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((max_prompt_length + max_response_length))}

# ─── Dense RM (AceMath-7B-RM for auxiliary r_i) ───────────────────────
dense_rm_model=${EIF_DENSE_RM_MODEL:-nvidia/AceMath-7B-RM}
rm_enable_resource_pool=${EIF_RM_ENABLE_RESOURCE_POOL:-False}
rm_gpus=${EIF_RM_GPUS_PER_NODE:-$gpus_per_node}
rm_nodes=${EIF_RM_NNODES:-1}
rm_tp_size=${EIF_RM_TP_SIZE:-2}
rm_gpu_memory_utilization=${EIF_RM_GPU_MEMORY_UTILIZATION:-0.5}
rm_max_num_seqs=${EIF_RM_MAX_NUM_SEQS:-128}
rm_prompt_length=${EIF_RM_PROMPT_LENGTH:-$max_prompt_length}
rm_response_length=${EIF_RM_RESPONSE_LENGTH:-$max_response_length}
rm_max_num_batched_tokens=${EIF_RM_MAX_NUM_BATCHED_TOKENS:-$((rm_prompt_length + rm_response_length))}

# ─── tau_LLM / m_LLM endpoints (Algorithm 1 nuisance regressors) ─────
llm_model=${EIF_LLM_MODEL:-qwen3}
llm_base_url=${EIF_LLM_BASE_URL:-https://ellm.nrp-nautilus.io/v1}
llm_api_key_env=${EIF_LLM_API_KEY_ENV:-NAUTILUS_API_KEY}

tau_model=${EIF_TAU_MODEL:-$llm_model}
tau_base_url=${EIF_TAU_BASE_URL:-$llm_base_url}
tau_api_key_env=${EIF_TAU_API_KEY_ENV:-$llm_api_key_env}
tau_temperature=${EIF_TAU_TEMPERATURE:-0.0}
tau_max_tokens=${EIF_TAU_MAX_TOKENS:-16}
tau_concurrency=${EIF_TAU_CONCURRENCY:-128}

m_model=${EIF_M_MODEL:-$llm_model}
m_base_url=${EIF_M_BASE_URL:-$llm_base_url}
m_api_key_env=${EIF_M_API_KEY_ENV:-$llm_api_key_env}
m_temperature=${EIF_M_TEMPERATURE:-0.0}
m_max_tokens=${EIF_M_MAX_TOKENS:-16}
m_concurrency=${EIF_M_CONCURRENCY:-128}

aux_concurrency=${EIF_AUX_CONCURRENCY:-32}

# ─── Experiment ───────────────────────────────────────────────────────
total_epochs=${EIF_TOTAL_EPOCHS:-20}
save_freq=${EIF_SAVE_FREQ:-20}
test_freq=${EIF_TEST_FREQ:-5}
experiment_name=${EIF_EXPERIMENT_NAME:-eif_${regime}_$(date +%Y%m%d_%H%M%S)}
loggers=${EIF_LOGGERS:-'["console"]'}
train_output_dir=${EIF_TRAIN_OUTPUT_DIR:-checkpoints/eif/${experiment_name}}
checkpoint_save_contents=${EIF_CHECKPOINT_SAVE_CONTENTS:-"['model','optimizer','extra','hf_model']"}
max_actor_ckpt_to_keep=${EIF_MAX_ACTOR_CKPT_TO_KEEP:-1}
max_critic_ckpt_to_keep=${EIF_MAX_CRITIC_CKPT_TO_KEEP:-1}
resume_mode=${EIF_RESUME_MODE:-disable}

# ─── Build val_files list: in-domain val + eval benchmarks ────────────
val_files="['$val_file'"
for bench in math500 amc minerva olympiad hardverify_math textbook_reasoning; do
    bench_file="$eval_dir/${bench}.parquet"
    if [[ -f "$bench_file" ]]; then
        val_files="$val_files,'$bench_file'"
    fi
done
val_files="$val_files]"

# ─── Validation ───────────────────────────────────────────────────────
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

if [[ "$dense_rm_model" == "nvidia/AceMath-7B-RM" ]] \
    && [[ "$rm_enable_resource_pool" != "True" ]] \
    && [[ "$rm_tp_size" == "1" ]]; then
    echo "EIF_RM_TP_SIZE=1 is not supported for colocated $dense_rm_model on the shared training pool." >&2
    echo "Use EIF_RM_TP_SIZE>=2, or move the reward model to a separate resource pool." >&2
    exit 1
fi

mkdir -p "$train_output_dir"
echo "Training output dir: $train_output_dir"

# ─── Launch ───────────────────────────────────────────────────────────
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="['$train_file']" \
    data.val_files="$val_files" \
    data.train_batch_size="$train_batch_size" \
    data.max_prompt_length="$max_prompt_length" \
    data.max_response_length="$max_response_length" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.model.trust_remote_code="$trust_remote_code" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="$ppo_mini_batch_size" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$ppo_micro_batch_size" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.checkpoint.save_contents="$checkpoint_save_contents" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization="$rollout_gpu_memory_utilization" \
    actor_rollout_ref.rollout.max_num_seqs="$rollout_max_num_seqs" \
    actor_rollout_ref.rollout.max_num_batched_tokens="$rollout_max_num_batched_tokens" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$rollout_log_prob_micro_batch_size" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$rollout_tp_size" \
    actor_rollout_ref.rollout.n="$rollout_n" \
    actor_rollout_ref.rollout.temperature="$rollout_temperature" \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$ref_log_prob_micro_batch_size" \
    algorithm.use_kl_in_reward=False \
    reward.reward_manager.name=hybrid_eif_online \
    reward.reward_model.enable=True \
    reward.reward_model.enable_resource_pool="$rm_enable_resource_pool" \
    reward.reward_model.n_gpus_per_node="$rm_gpus" \
    reward.reward_model.nnodes="$rm_nodes" \
    reward.reward_model.model_path="$dense_rm_model" \
    reward.reward_model.rollout.name=vllm \
    reward.reward_model.rollout.tensor_model_parallel_size="$rm_tp_size" \
    reward.reward_model.rollout.gpu_memory_utilization="$rm_gpu_memory_utilization" \
    reward.reward_model.rollout.max_num_seqs="$rm_max_num_seqs" \
    reward.reward_model.rollout.max_num_batched_tokens="$rm_max_num_batched_tokens" \
    reward.reward_model.rollout.prompt_length="$rm_prompt_length" \
    reward.reward_model.rollout.response_length="$rm_response_length" \
    reward.reward_kwargs.hybrid_eif_online.aux_concurrency="$aux_concurrency" \
    reward.reward_kwargs.hybrid_eif_online.tau_model="$tau_model" \
    reward.reward_kwargs.hybrid_eif_online.tau_base_url="$tau_base_url" \
    reward.reward_kwargs.hybrid_eif_online.tau_api_key_env="$tau_api_key_env" \
    reward.reward_kwargs.hybrid_eif_online.tau_temperature="$tau_temperature" \
    reward.reward_kwargs.hybrid_eif_online.tau_max_tokens="$tau_max_tokens" \
    reward.reward_kwargs.hybrid_eif_online.tau_concurrency="$tau_concurrency" \
    reward.reward_kwargs.hybrid_eif_online.m_model="$m_model" \
    reward.reward_kwargs.hybrid_eif_online.m_base_url="$m_base_url" \
    reward.reward_kwargs.hybrid_eif_online.m_api_key_env="$m_api_key_env" \
    reward.reward_kwargs.hybrid_eif_online.m_temperature="$m_temperature" \
    reward.reward_kwargs.hybrid_eif_online.m_max_tokens="$m_max_tokens" \
    reward.reward_kwargs.hybrid_eif_online.m_concurrency="$m_concurrency" \
    trainer.critic_warmup=0 \
    trainer.logger="$loggers" \
    trainer.project_name='eif_grpo' \
    trainer.experiment_name="$experiment_name" \
    trainer.nnodes="$nnodes" \
    trainer.n_gpus_per_node="$gpus_per_node" \
    trainer.save_freq="$save_freq" \
    trainer.test_freq="$test_freq" \
    trainer.total_epochs="$total_epochs" \
    trainer.resume_mode="$resume_mode" \
    trainer.default_local_dir="$train_output_dir" \
    trainer.max_actor_ckpt_to_keep="$max_actor_ckpt_to_keep" \
    trainer.max_critic_ckpt_to_keep="$max_critic_ckpt_to_keep" \
    "$@"

#!/usr/bin/env bash
# Unified hybrid-reward GRPO training script.
#
# Supports two reward strategies via ALGORITHM env var:
#   hero  — HERO reward shaping (arXiv:2510.07242v1, Table 6)
#   eif   — EIF one-step estimator (hybrid_eif_online)
#
# Training regimes (set REGIME):
#   verifiable       — 2k verifiable-only samples
#   hard_to_verify   — 2k hard-to-verify-only samples
#   mixed            — 1k verifiable + 1k hard-to-verify (default)
#
# Prerequisites:
#   1. Preprocess data:
#        python examples/data_preprocess/openmathreasoning_hero.py \
#            --local_save_dir ~/data/openmathreasoning_hero
#   2. Preprocess eval benchmarks:
#        python examples/data_preprocess/hero_eval_benchmarks.py \
#            --local_save_dir ~/data/hero_eval
#   3. pip install math-verify
#   4. For EIF: pip install openai, set NAUTILUS_API_KEY
#
# Usage:
#   ALGORITHM=hero bash examples/hybrid_reward/run_train.sh
#   ALGORITHM=eif  bash examples/hybrid_reward/run_train.sh
#   ALGORITHM=hero REGIME=verifiable bash examples/hybrid_reward/run_train.sh

set -euo pipefail
set -x

algorithm=${ALGORITHM:-hero}

# ─── Data ─────────────────────────────────────────────────────────────
regime=${REGIME:-mixed}
data_dir=${DATA_DIR:-$HOME/data/openmathreasoning_hero}
eval_dir=${EVAL_DIR:-$HOME/data/hero_eval}

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
model_path=${MODEL_PATH:-Qwen/Qwen3-4B-Base}
trust_remote_code=${TRUST_REMOTE_CODE:-True}

# ─── Cluster layout ──────────────────────────────────────────────────
gpus_per_node=${GPUS_PER_NODE:-4}
nnodes=${NNODES:-8}

# ─── Training hyperparams ────────────────────────────────────────────
train_batch_size=${TRAIN_BATCH_SIZE:-512}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-128}
ppo_micro_batch_size=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}

# ─── Rollout ─────────────────────────────────────────────────────────
rollout_n=${ROLLOUT_N:-8}
rollout_tp_size=${ROLLOUT_TP_SIZE:-2}
rollout_gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}
rollout_temperature=${ROLLOUT_TEMPERATURE:-1.0}
rollout_log_prob_micro_batch_size=${ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
ref_log_prob_micro_batch_size=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-128}
rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((max_prompt_length + max_response_length))}

# ─── Dense RM (AceMath-7B-RM) ────────────────────────────────────────
dense_rm_model=${DENSE_RM_MODEL:-nvidia/AceMath-7B-RM}
rm_enable_resource_pool=${RM_ENABLE_RESOURCE_POOL:-False}
rm_gpus=${RM_GPUS_PER_NODE:-$gpus_per_node}
rm_nodes=${RM_NNODES:-1}
rm_tp_size=${RM_TP_SIZE:-2}
rm_gpu_memory_utilization=${RM_GPU_MEMORY_UTILIZATION:-0.5}
rm_max_num_seqs=${RM_MAX_NUM_SEQS:-128}
rm_prompt_length=${RM_PROMPT_LENGTH:-$max_prompt_length}
rm_response_length=${RM_RESPONSE_LENGTH:-$max_response_length}
rm_max_num_batched_tokens=${RM_MAX_NUM_BATCHED_TOKENS:-$((rm_prompt_length + rm_response_length))}

# ─── Experiment ──────────────────────────────────────────────────────
total_epochs=${TOTAL_EPOCHS:-20}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}
experiment_name=${EXPERIMENT_NAME:-${algorithm}_${regime}_$(date +%Y%m%d_%H%M%S)}
loggers=${LOGGERS:-'["console"]'}
train_output_dir=${TRAIN_OUTPUT_DIR:-checkpoints/hybrid_reward/${experiment_name}}
checkpoint_save_contents=${CHECKPOINT_SAVE_CONTENTS:-"['model','optimizer','extra','hf_model']"}
max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-1}
max_critic_ckpt_to_keep=${MAX_CRITIC_CKPT_TO_KEEP:-1}
resume_mode=${RESUME_MODE:-disable}

# ─── Algorithm-specific reward config ────────────────────────────────
algo_specific_args=()
case "$algorithm" in
    hero)
        case "$regime" in
            verifiable) hero_alpha=0.05; hero_beta=0.05 ;;
            *)          hero_alpha=0.1;  hero_beta=0.1  ;;
        esac
        hero_w_min=${HERO_W_MIN:-0.4}
        hero_w_max=${HERO_W_MAX:-3.0}
        hero_k=${HERO_K:-6.0}
        hero_sigma_ema=${HERO_SIGMA_EMA:-0.9}
        reward_manager_name=hero
        project_name='hero_paper_reproduction'
        algo_specific_args=(
            reward.reward_kwargs.hero.alpha="$hero_alpha"
            reward.reward_kwargs.hero.beta="$hero_beta"
            reward.reward_kwargs.hero.w_min="$hero_w_min"
            reward.reward_kwargs.hero.w_max="$hero_w_max"
            reward.reward_kwargs.hero.k="$hero_k"
            reward.reward_kwargs.hero.sigma_ema="$hero_sigma_ema"
        )
        ;;
    eif)
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
        reward_manager_name=hybrid_eif_online
        project_name='eif_grpo'
        algo_specific_args=(
            reward.reward_kwargs.hybrid_eif_online.aux_concurrency="$aux_concurrency"
            reward.reward_kwargs.hybrid_eif_online.tau_model="$tau_model"
            reward.reward_kwargs.hybrid_eif_online.tau_base_url="$tau_base_url"
            reward.reward_kwargs.hybrid_eif_online.tau_api_key_env="$tau_api_key_env"
            reward.reward_kwargs.hybrid_eif_online.tau_temperature="$tau_temperature"
            reward.reward_kwargs.hybrid_eif_online.tau_max_tokens="$tau_max_tokens"
            reward.reward_kwargs.hybrid_eif_online.tau_concurrency="$tau_concurrency"
            reward.reward_kwargs.hybrid_eif_online.m_model="$m_model"
            reward.reward_kwargs.hybrid_eif_online.m_base_url="$m_base_url"
            reward.reward_kwargs.hybrid_eif_online.m_api_key_env="$m_api_key_env"
            reward.reward_kwargs.hybrid_eif_online.m_temperature="$m_temperature"
            reward.reward_kwargs.hybrid_eif_online.m_max_tokens="$m_max_tokens"
            reward.reward_kwargs.hybrid_eif_online.m_concurrency="$m_concurrency"
        )
        if [[ "$dense_rm_model" == "nvidia/AceMath-7B-RM" ]] \
            && [[ "$rm_enable_resource_pool" != "True" ]] \
            && [[ "$rm_tp_size" == "1" ]]; then
            echo "RM_TP_SIZE=1 is not supported for colocated $dense_rm_model." >&2
            echo "Use RM_TP_SIZE>=2, or set RM_ENABLE_RESOURCE_POOL=True." >&2
            exit 1
        fi
        ;;
    *)
        echo "Unknown ALGORITHM=$algorithm. Use hero or eif." >&2
        exit 1
        ;;
esac

# ─── Build val_files list ────────────────────────────────────────────
debug_mode=${DEBUG:-}
skip_benchmarks=${SKIP_BENCHMARKS:-}
val_files="['$val_file'"
if [[ -n "$debug_mode" ]]; then
    echo "Debug mode enabled: only using amc benchmark for validation"
    bench_file="$eval_dir/amc.parquet"
    if [[ -f "$bench_file" ]]; then
        val_files="$val_files,'$bench_file'"
    fi
else
    for bench in math500 amc minerva olympiad hardverify_math textbook_reasoning; do
        if [[ -n "$skip_benchmarks" ]] && echo "$skip_benchmarks" | grep -qw "$bench"; then
            echo "Skipping benchmark: $bench"
            continue
        fi
        bench_file="$eval_dir/${bench}.parquet"
        if [[ -f "$bench_file" ]]; then
            val_files="$val_files,'$bench_file'"
        fi
    done
fi
val_files="$val_files]"

# ─── Validation ──────────────────────────────────────────────────────
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

# ─── Launch ──────────────────────────────────────────────────────────
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
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$ppo_max_token_len_per_gpu" \
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
    reward.reward_manager.name="$reward_manager_name" \
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
    "${algo_specific_args[@]}" \
    trainer.critic_warmup=0 \
    trainer.logger="$loggers" \
    trainer.project_name="$project_name" \
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

set -euo pipefail
set -x

# ─── Data: aligned with HERO (arXiv:2510.07242v1) ────────────────────
# Training regimes (set HYBRID_EIF_ONLINE_REGIME):
#   verifiable       — 2k verifiable-only samples
#   hard_to_verify   — 2k hard-to-verify-only samples
#   mixed            — 1k verifiable + 1k hard-to-verify (default, matches HERO)
#
# Prerequisites:
#   1. Preprocess data:
#        python examples/data_preprocess/openmathreasoning_hero.py \
#            --local_save_dir ~/data/openmathreasoning_hero
#   2. Preprocess eval benchmarks:
#        python examples/data_preprocess/hero_eval_benchmarks.py \
#            --local_save_dir ~/data/hero_eval

regime=${HYBRID_EIF_ONLINE_REGIME:-mixed}
data_dir=${HYBRID_EIF_ONLINE_DATA_DIR:-$HOME/data/openmathreasoning_hero}
eval_dir=${HYBRID_EIF_ONLINE_EVAL_DIR:-$HOME/data/hero_eval}

case "$regime" in
    verifiable)
        train_path=${HYBRID_EIF_ONLINE_TRAIN_PATH:-$data_dir/train_verifiable.parquet}
        val_path=${HYBRID_EIF_ONLINE_VAL_PATH:-$data_dir/val_verifiable.parquet}
        ;;
    hard_to_verify)
        train_path=${HYBRID_EIF_ONLINE_TRAIN_PATH:-$data_dir/train_hard_to_verify.parquet}
        val_path=${HYBRID_EIF_ONLINE_VAL_PATH:-$data_dir/val_hard_to_verify.parquet}
        ;;
    mixed)
        train_path=${HYBRID_EIF_ONLINE_TRAIN_PATH:-$data_dir/train_mixed.parquet}
        val_path=${HYBRID_EIF_ONLINE_VAL_PATH:-$data_dir/val_mixed.parquet}
        ;;
    *)
        echo "Unknown regime: $regime. Use verifiable|hard_to_verify|mixed." >&2
        exit 1
        ;;
esac

model_path=${HYBRID_EIF_ONLINE_MODEL_PATH:-Qwen/Qwen3-4B-Base}
trust_remote_code=${HYBRID_EIF_ONLINE_TRUST_REMOTE_CODE:-True}

trainer_gpus=${HYBRID_EIF_ONLINE_TRAINER_GPUS_PER_NODE:-4}
trainer_nodes=${HYBRID_EIF_ONLINE_TRAINER_NNODES:-8}
train_batch_size=${HYBRID_EIF_ONLINE_TRAIN_BATCH_SIZE:-512}
max_prompt_length=${HYBRID_EIF_ONLINE_MAX_PROMPT_LENGTH:-1024}
max_response_length=${HYBRID_EIF_ONLINE_MAX_RESPONSE_LENGTH:-8192}
ppo_mini_batch_size=${HYBRID_EIF_ONLINE_PPO_MINI_BATCH_SIZE:-128}
ppo_micro_batch_size=${HYBRID_EIF_ONLINE_PPO_MICRO_BATCH_SIZE_PER_GPU:-2}
rollout_log_prob_micro_batch_size=${HYBRID_EIF_ONLINE_ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
ref_log_prob_micro_batch_size=${HYBRID_EIF_ONLINE_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
rollout_tp_size=${HYBRID_EIF_ONLINE_ROLLOUT_TP_SIZE:-2}
rollout_n=${HYBRID_EIF_ONLINE_ROLLOUT_N:-8}
rollout_temperature=${HYBRID_EIF_ONLINE_ROLLOUT_TEMPERATURE:-1.0}
rollout_gpu_memory_utilization=${HYBRID_EIF_ONLINE_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}
rollout_max_num_seqs=${HYBRID_EIF_ONLINE_ROLLOUT_MAX_NUM_SEQS:-128}
rollout_max_num_batched_tokens=${HYBRID_EIF_ONLINE_ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((max_prompt_length + max_response_length))}

rm_enable_resource_pool=${HYBRID_EIF_ONLINE_RM_ENABLE_RESOURCE_POOL:-False}
rm_gpus=${HYBRID_EIF_ONLINE_RM_GPUS_PER_NODE:-8}
rm_nodes=${HYBRID_EIF_ONLINE_RM_NNODES:-1}
rm_tp_size=${HYBRID_EIF_ONLINE_RM_TP_SIZE:-2}
rm_gpu_memory_utilization=${HYBRID_EIF_ONLINE_RM_GPU_MEMORY_UTILIZATION:-0.5}
rm_prompt_length=${HYBRID_EIF_ONLINE_RM_PROMPT_LENGTH:-$max_prompt_length}
rm_response_length=${HYBRID_EIF_ONLINE_RM_RESPONSE_LENGTH:-$max_response_length}
rm_max_num_seqs=${HYBRID_EIF_ONLINE_RM_MAX_NUM_SEQS:-128}
rm_max_num_batched_tokens=${HYBRID_EIF_ONLINE_RM_MAX_NUM_BATCHED_TOKENS:-$((rm_prompt_length + rm_response_length))}

dense_rm_model=${HYBRID_EIF_ONLINE_DENSE_RM_MODEL:-nvidia/AceMath-7B-RM}
if [[ "$dense_rm_model" == "AceMath/AceMath-7B-RM" ]]; then
    echo "HYBRID_EIF_ONLINE_DENSE_RM_MODEL=AceMath/AceMath-7B-RM is invalid; using nvidia/AceMath-7B-RM instead." >&2
    dense_rm_model="nvidia/AceMath-7B-RM"
fi

llm_model=${HYBRID_EIF_ONLINE_LLM_MODEL:-qwen3}
llm_base_url=${HYBRID_EIF_ONLINE_LLM_BASE_URL:-https://ellm.nrp-nautilus.io/v1}
llm_api_key_env=${HYBRID_EIF_ONLINE_LLM_API_KEY_ENV:-NAUTILUS_API_KEY}

tau_model=${HYBRID_EIF_ONLINE_TAU_MODEL:-$llm_model}
tau_base_url=${HYBRID_EIF_ONLINE_TAU_BASE_URL:-$llm_base_url}
tau_api_key_env=${HYBRID_EIF_ONLINE_TAU_API_KEY_ENV:-NAUTILUS_API_KEY}
tau_temperature=${HYBRID_EIF_ONLINE_TAU_TEMPERATURE:-0.0}
tau_max_tokens=${HYBRID_EIF_ONLINE_TAU_MAX_TOKENS:-16}
tau_concurrency=${HYBRID_EIF_ONLINE_TAU_CONCURRENCY:-128}

m_model=${HYBRID_EIF_ONLINE_M_MODEL:-$llm_model}
m_base_url=${HYBRID_EIF_ONLINE_M_BASE_URL:-$llm_base_url}
m_api_key_env=${HYBRID_EIF_ONLINE_M_API_KEY_ENV:-$llm_api_key_env}
m_temperature=${HYBRID_EIF_ONLINE_M_TEMPERATURE:-0.0}
m_max_tokens=${HYBRID_EIF_ONLINE_M_MAX_TOKENS:-16}
m_concurrency=${HYBRID_EIF_ONLINE_M_CONCURRENCY:-128}

aux_concurrency=${HYBRID_EIF_ONLINE_AUX_CONCURRENCY:-32}
experiment_name=${HYBRID_EIF_ONLINE_EXPERIMENT_NAME:-hybrid_eif_online_${regime}_$(date +%Y%m%d_%H%M%S)}
loggers=${HYBRID_EIF_ONLINE_LOGGERS:-'["console"]'}
train_output_dir=${HYBRID_EIF_ONLINE_TRAIN_OUTPUT_DIR:-checkpoints/hybrid_eif_online/${experiment_name}}
total_epochs=${HYBRID_EIF_ONLINE_TOTAL_EPOCHS:-20}
save_freq=${HYBRID_EIF_ONLINE_SAVE_FREQ:-20}
test_freq=${HYBRID_EIF_ONLINE_TEST_FREQ:-5}
resume_mode=${HYBRID_EIF_ONLINE_RESUME_MODE:-disable}

# ─── Build val_files list: in-domain val + HERO eval benchmarks ──────
val_files="['$val_path'"
for bench in math500 amc minerva olympiad hardverify_math textbook_reasoning; do
    bench_file="$eval_dir/${bench}.parquet"
    if [[ -f "$bench_file" ]]; then
        val_files="$val_files,'$bench_file'"
    fi
done
val_files="$val_files]"

for required_path in "$train_path" "$val_path"; do
    if [[ ! -f "$required_path" ]]; then
        echo "Missing data file: $required_path" >&2
        echo "Run: python examples/data_preprocess/openmathreasoning_hero.py --local_save_dir $data_dir" >&2
        exit 1
    fi
done

if [[ "$dense_rm_model" == "nvidia/AceMath-7B-RM" ]] \
    && [[ "$rm_enable_resource_pool" != "True" ]] \
    && [[ "$rm_tp_size" == "1" ]]; then
    echo "HYBRID_EIF_ONLINE_RM_TP_SIZE=1 is not supported for colocated $dense_rm_model on the shared 8-GPU training pool." >&2
    echo "Use HYBRID_EIF_ONLINE_RM_TP_SIZE>=2, or move the reward model to a separate resource pool." >&2
    exit 1
fi

mkdir -p "$train_output_dir"
echo "Training output dir: $train_output_dir"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="['$train_path']" \
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
    trainer.project_name='verl_hybrid_eif_online_grpo' \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node="$trainer_gpus" \
    trainer.nnodes="$trainer_nodes" \
    trainer.save_freq="$save_freq" \
    trainer.test_freq="$test_freq" \
    trainer.total_epochs="$total_epochs" \
    trainer.resume_mode="$resume_mode" \
    trainer.default_local_dir="$train_output_dir" \
    "$@"

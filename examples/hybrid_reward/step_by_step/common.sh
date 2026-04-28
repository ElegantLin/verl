#!/usr/bin/env bash
# Shared configuration and helper functions for hybrid reward pipelines.
#
# Provides GPU profiles, directory layout, dataset defaults, and helper
# functions used by all step-by-step scripts.
#
# Usage:
#   source examples/hybrid_reward/step_by_step/common.sh

if [[ -n "${_SHARED_STEP_COMMON_SH:-}" ]]; then
    return 0
fi
_SHARED_STEP_COMMON_SH=1

shared_step_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${shared_step_dir}/../../.." && pwd)

# Import shared utilities (resolve_latest_hf_dir, etc.)
source "$shared_step_dir/../bash_utils.sh"

set_default() {
    local var_name="$1"
    local default_value="$2"
    if [[ -z "${!var_name:-}" ]]; then
        export "$var_name=$default_value"
    fi
}

require_file() {
    local target_path="$1"
    if [[ ! -f "$target_path" ]]; then
        echo "Missing required file: $target_path" >&2
        exit 1
    fi
}

ensure_dirs() {
    mkdir -p "$work_dir" "$source_dir" "$data_dir" "$eval_dir" "$sft_data_dir" "$sft_output_dir" "$train_output_dir" "$eval_output_dir"
}

# ── GPU profile presets ──────────────────────────────────────────────
gpu_profile=${RL_PIPELINE_GPU_PROFILE:-8x24gb}

case "$gpu_profile" in
    8x24|8x24gb)
        set_default RL_PIPELINE_GPUS_PER_NODE 8
        set_default RL_PIPELINE_NNODES 1
        set_default RL_PIPELINE_GEN_GPUS_PER_NODE 8
        set_default RL_PIPELINE_GEN_NNODES 1
        set_default RL_PIPELINE_SFT_GPUS_PER_NODE 8
        set_default RL_PIPELINE_SFT_NNODES 1
        set_default RL_PIPELINE_EVAL_GPUS_PER_NODE 8
        set_default RL_PIPELINE_EVAL_NNODES 1
        set_default RL_PIPELINE_MAX_PROMPT_LENGTH 1024
        set_default RL_PIPELINE_MAX_RESPONSE_LENGTH 2048
        set_default RL_PIPELINE_TRAIN_BATCH_SIZE 64
        set_default RL_PIPELINE_PPO_MINI_BATCH_SIZE 64
        set_default RL_PIPELINE_PPO_MICRO_BATCH_SIZE_PER_GPU 2
        set_default RL_PIPELINE_ROLLOUT_N 4
        set_default RL_PIPELINE_ROLLOUT_TP_SIZE 2
        set_default RL_PIPELINE_ROLLOUT_GPU_MEMORY_UTILIZATION 0.35
        set_default RL_PIPELINE_ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU 2
        set_default RL_PIPELINE_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU 2
        set_default RL_PIPELINE_ROLLOUT_MAX_NUM_SEQS 128
        set_default RL_PIPELINE_RM_GPUS_PER_NODE 8
        set_default RL_PIPELINE_RM_NNODES 1
        set_default RL_PIPELINE_RM_TP_SIZE 2
        set_default RL_PIPELINE_RM_GPU_MEMORY_UTILIZATION 0.3
        set_default RL_PIPELINE_RM_MAX_NUM_SEQS 128
        set_default RL_PIPELINE_GEN_TP_SIZE 2
        set_default RL_PIPELINE_GEN_GPU_MEMORY_UTILIZATION 0.5
        set_default RL_PIPELINE_SOURCE_GENERATION_N 1
        set_default RL_PIPELINE_EVAL_TP_SIZE 2
        set_default RL_PIPELINE_EVAL_GPU_MEMORY_UTILIZATION 0.5
        set_default RL_PIPELINE_EVAL_N_SAMPLES 8
        set_default RL_PIPELINE_SFT_TRAIN_BATCH_SIZE 32
        set_default RL_PIPELINE_SFT_MICRO_BATCH_SIZE_PER_GPU 1
        set_default RL_PIPELINE_SFT_MAX_LENGTH 4096
        set_default RL_PIPELINE_SFT_MAX_TOKEN_LEN_PER_GPU 6144
        ;;
    4x80|4x80gb)
        set_default RL_PIPELINE_GPUS_PER_NODE 4
        set_default RL_PIPELINE_NNODES 1
        set_default RL_PIPELINE_GEN_GPUS_PER_NODE 4
        set_default RL_PIPELINE_GEN_NNODES 1
        set_default RL_PIPELINE_SFT_GPUS_PER_NODE 4
        set_default RL_PIPELINE_SFT_NNODES 1
        set_default RL_PIPELINE_EVAL_GPUS_PER_NODE 4
        set_default RL_PIPELINE_EVAL_NNODES 1
        set_default RL_PIPELINE_MAX_PROMPT_LENGTH 1024
        set_default RL_PIPELINE_MAX_RESPONSE_LENGTH 4096
        set_default RL_PIPELINE_TRAIN_BATCH_SIZE 128
        set_default RL_PIPELINE_PPO_MINI_BATCH_SIZE 64
        set_default RL_PIPELINE_PPO_MICRO_BATCH_SIZE_PER_GPU 4
        set_default RL_PIPELINE_ROLLOUT_N 8
        set_default RL_PIPELINE_ROLLOUT_TP_SIZE 2
        set_default RL_PIPELINE_ROLLOUT_GPU_MEMORY_UTILIZATION 0.7
        set_default RL_PIPELINE_ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU 4
        set_default RL_PIPELINE_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU 4
        set_default RL_PIPELINE_ROLLOUT_MAX_NUM_SEQS 128
        set_default RL_PIPELINE_RM_GPUS_PER_NODE 4
        set_default RL_PIPELINE_RM_NNODES 1
        set_default RL_PIPELINE_RM_TP_SIZE 2
        set_default RL_PIPELINE_RM_GPU_MEMORY_UTILIZATION 0.6
        set_default RL_PIPELINE_RM_MAX_NUM_SEQS 128
        set_default RL_PIPELINE_GEN_TP_SIZE 2
        set_default RL_PIPELINE_GEN_GPU_MEMORY_UTILIZATION 0.75
        set_default RL_PIPELINE_SOURCE_GENERATION_N 1
        set_default RL_PIPELINE_EVAL_TP_SIZE 2
        set_default RL_PIPELINE_EVAL_GPU_MEMORY_UTILIZATION 0.75
        set_default RL_PIPELINE_EVAL_N_SAMPLES 8
        set_default RL_PIPELINE_SFT_TRAIN_BATCH_SIZE 64
        set_default RL_PIPELINE_SFT_MICRO_BATCH_SIZE_PER_GPU 2
        set_default RL_PIPELINE_SFT_MAX_LENGTH 6144
        set_default RL_PIPELINE_SFT_MAX_TOKEN_LEN_PER_GPU 8192
        ;;
    2x80|2x80gb)
        set_default RL_PIPELINE_GPUS_PER_NODE 2
        set_default RL_PIPELINE_NNODES 1
        set_default RL_PIPELINE_GEN_GPUS_PER_NODE 2
        set_default RL_PIPELINE_GEN_NNODES 1
        set_default RL_PIPELINE_SFT_GPUS_PER_NODE 2
        set_default RL_PIPELINE_SFT_NNODES 1
        set_default RL_PIPELINE_EVAL_GPUS_PER_NODE 2
        set_default RL_PIPELINE_EVAL_NNODES 1
        set_default RL_PIPELINE_MAX_PROMPT_LENGTH 1024
        set_default RL_PIPELINE_MAX_RESPONSE_LENGTH 4096
        set_default RL_PIPELINE_TRAIN_BATCH_SIZE 64
        set_default RL_PIPELINE_PPO_MINI_BATCH_SIZE 32
        set_default RL_PIPELINE_PPO_MICRO_BATCH_SIZE_PER_GPU 4
        set_default RL_PIPELINE_ROLLOUT_N 8
        set_default RL_PIPELINE_ROLLOUT_TP_SIZE 2
        set_default RL_PIPELINE_ROLLOUT_GPU_MEMORY_UTILIZATION 0.7
        set_default RL_PIPELINE_ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU 4
        set_default RL_PIPELINE_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU 4
        set_default RL_PIPELINE_ROLLOUT_MAX_NUM_SEQS 64
        set_default RL_PIPELINE_RM_GPUS_PER_NODE 2
        set_default RL_PIPELINE_RM_NNODES 1
        set_default RL_PIPELINE_RM_TP_SIZE 2
        set_default RL_PIPELINE_RM_GPU_MEMORY_UTILIZATION 0.6
        set_default RL_PIPELINE_RM_MAX_NUM_SEQS 64
        set_default RL_PIPELINE_GEN_TP_SIZE 2
        set_default RL_PIPELINE_GEN_GPU_MEMORY_UTILIZATION 0.75
        set_default RL_PIPELINE_SOURCE_GENERATION_N 1
        set_default RL_PIPELINE_EVAL_TP_SIZE 2
        set_default RL_PIPELINE_EVAL_GPU_MEMORY_UTILIZATION 0.75
        set_default RL_PIPELINE_EVAL_N_SAMPLES 8
        set_default RL_PIPELINE_SFT_TRAIN_BATCH_SIZE 32
        set_default RL_PIPELINE_SFT_MICRO_BATCH_SIZE_PER_GPU 2
        set_default RL_PIPELINE_SFT_MAX_LENGTH 6144
        set_default RL_PIPELINE_SFT_MAX_TOKEN_LEN_PER_GPU 8192
        ;;
    *)
        echo "Unknown RL_PIPELINE_GPU_PROFILE=$gpu_profile. Use 8x24gb, 4x80gb, or 2x80gb." >&2
        exit 1
        ;;
esac

# ── Directory layout ─────────────────────────────────────────────────
artifact_root=${RL_PIPELINE_ARTIFACT_ROOT:-$repo_root/data/rl_data_pipeline}
run_name=${RL_PIPELINE_RUN_NAME:-step_by_step_run}
work_dir=${RL_PIPELINE_WORK_DIR:-$artifact_root/$run_name}
source_dir=${RL_PIPELINE_SOURCE_DIR:-$work_dir/source_generation}
data_dir=${RL_PIPELINE_DATA_DIR:-$work_dir/openmathreasoning_hero}
eval_dir=${RL_PIPELINE_EVAL_DIR:-$work_dir/hero_eval}
sft_data_dir=${RL_PIPELINE_SFT_DATA_DIR:-$work_dir/openmathreasoning_hero_sft}
sft_output_dir=${RL_PIPELINE_SFT_OUTPUT_DIR:-$work_dir/checkpoints/cold_start_sft}
algorithm_name=${ALGORITHM:-}
default_train_output_dir=$work_dir/checkpoints/rl
default_eval_output_dir=$work_dir/eval_results
if [[ -n "$algorithm_name" ]]; then
    default_train_output_dir=$work_dir/checkpoints/${algorithm_name}_rl
    default_eval_output_dir=$work_dir/eval_results/${algorithm_name}
fi
train_output_dir=${RL_PIPELINE_TRAIN_OUTPUT_DIR:-$default_train_output_dir}
eval_output_dir=${RL_PIPELINE_EVAL_OUTPUT_DIR:-$default_eval_output_dir}
source_prompts_path=${RL_PIPELINE_SOURCE_PROMPTS_PATH:-$source_dir/source_prompts.parquet}
source_generated_path=${RL_PIPELINE_SOURCE_GENERATED_PATH:-$source_dir/source_generated.parquet}
filtered_tbr_path=${RL_PIPELINE_FILTERED_TBR_PATH:-$eval_dir/textbook_reasoning_filtered.parquet}

# ── Model ────────────────────────────────────────────────────────────
base_model_path=${RL_PIPELINE_BASE_MODEL_PATH:-${RL_PIPELINE_MODEL_PATH:-Qwen/Qwen3-4B-Base}}
trust_remote_code=${RL_PIPELINE_TRUST_REMOTE_CODE:-True}
source_dataset_trust_remote_code=${RL_PIPELINE_SOURCE_DATASET_TRUST_REMOTE_CODE:-False}

# ── Dataset ──────────────────────────────────────────────────────────
dataset_name=${RL_PIPELINE_DATASET:-nvidia/OpenMathReasoning}
dataset_config=${RL_PIPELINE_DATASET_CONFIG:-}
dataset_split=${RL_PIPELINE_DATASET_SPLIT:-cot}
local_dataset_path=${RL_PIPELINE_LOCAL_DATASET_PATH:-}
source_question_col=${RL_PIPELINE_SOURCE_QUESTION_COL:-problem}
source_answer_col=${RL_PIPELINE_SOURCE_ANSWER_COL:-expected_answer}
question_col=${RL_PIPELINE_QUESTION_COL:-question}
answer_col=${RL_PIPELINE_ANSWER_COL:-answer}
problem_type_col=${RL_PIPELINE_PROBLEM_TYPE_COL:-problem_type}
problem_type_value=${RL_PIPELINE_PROBLEM_TYPE_VALUE:-has_answer_extracted}
source_sample_size=${RL_PIPELINE_SOURCE_SAMPLE_SIZE:-40000}
seed=${RL_PIPELINE_SEED:-42}

# ── Generation ───────────────────────────────────────────────────────
gen_nnodes=${RL_PIPELINE_GEN_NNODES:-1}
gen_gpus_per_node=${RL_PIPELINE_GEN_GPUS_PER_NODE:-${RL_PIPELINE_GPUS_PER_NODE}}
gen_tp_size=${RL_PIPELINE_GEN_TP_SIZE:-2}
gen_gpu_memory_utilization=${RL_PIPELINE_GEN_GPU_MEMORY_UTILIZATION:-0.75}
source_generation_n=${RL_PIPELINE_SOURCE_GENERATION_N:-1}
source_temperature=${RL_PIPELINE_SOURCE_TEMPERATURE:-1.0}
source_top_p=${RL_PIPELINE_SOURCE_TOP_P:-0.95}
max_prompt_length=${RL_PIPELINE_MAX_PROMPT_LENGTH:-1024}
max_response_length=${RL_PIPELINE_MAX_RESPONSE_LENGTH:-4096}

# ── Eval benchmarks ──────────────────────────────────────────────────
eval_benchmarks=${RL_PIPELINE_EVAL_BENCHMARKS_BUILD:-"math500 amc minerva olympiad hardverify_math textbook_reasoning"}
hvm_local_path=${RL_PIPELINE_HVM_LOCAL_PATH:-}
tbr_local_path=${RL_PIPELINE_TBR_LOCAL_PATH:-}

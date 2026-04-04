#!/usr/bin/env bash

if [[ -n "${_HERO_STEP_COMMON_SH:-}" ]]; then
    return 0
fi
_HERO_STEP_COMMON_SH=1

hero_step_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${hero_step_dir}/../../.." && pwd)

set_default() {
    local var_name="$1"
    local default_value="$2"
    if [[ -z "${!var_name:-}" ]]; then
        export "$var_name=$default_value"
    fi
}

resolve_latest_hf_dir() {
    local root_dir="$1"
    local suffix="$2"
    local tracker_file="$root_dir/latest_checkpointed_iteration.txt"
    if [[ ! -f "$tracker_file" ]]; then
        return 1
    fi

    local latest_step
    latest_step=$(<"$tracker_file")
    local resolved_path="$root_dir/global_step_${latest_step}${suffix}"
    if [[ ! -d "$resolved_path" ]]; then
        return 1
    fi
    printf '%s\n' "$resolved_path"
}

require_file() {
    local target_path="$1"
    if [[ ! -f "$target_path" ]]; then
        echo "Missing required file: $target_path" >&2
        exit 1
    fi
}

ensure_hero_dirs() {
    mkdir -p "$work_dir" "$source_dir" "$data_dir" "$eval_dir" "$sft_data_dir" "$sft_output_dir" "$train_output_dir" "$eval_output_dir"
}

gpu_profile=${HERO_GPU_PROFILE:-8x24gb}

case "$gpu_profile" in
    8x24|8x24gb)
        set_default HERO_GPUS_PER_NODE 8
        set_default HERO_NNODES 1
        set_default HERO_GEN_GPUS_PER_NODE 8
        set_default HERO_GEN_NNODES 1
        set_default HERO_SFT_GPUS_PER_NODE 8
        set_default HERO_SFT_NNODES 1
        set_default HERO_EVAL_GPUS_PER_NODE 8
        set_default HERO_EVAL_NNODES 1
        set_default HERO_MAX_PROMPT_LENGTH 1024
        set_default HERO_MAX_RESPONSE_LENGTH 2048
        set_default HERO_TRAIN_BATCH_SIZE 64
        set_default HERO_PPO_MINI_BATCH_SIZE 64
        set_default HERO_PPO_MICRO_BATCH_SIZE_PER_GPU 1
        set_default HERO_PPO_MAX_TOKEN_LEN_PER_GPU 6144
        set_default HERO_ROLLOUT_N 4
        set_default HERO_ROLLOUT_TP_SIZE 2
        set_default HERO_ROLLOUT_GPU_MEMORY_UTILIZATION 0.35
        set_default HERO_ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU 2
        set_default HERO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU 2
        set_default HERO_ROLLOUT_MAX_NUM_SEQS 128
        set_default HERO_RM_GPUS_PER_NODE 8
        set_default HERO_RM_NNODES 1
        set_default HERO_RM_TP_SIZE 2
        set_default HERO_RM_GPU_MEMORY_UTILIZATION 0.3
        set_default HERO_RM_MAX_NUM_SEQS 128
        set_default HERO_GEN_TP_SIZE 2
        set_default HERO_GEN_GPU_MEMORY_UTILIZATION 0.5
        set_default HERO_SOURCE_GENERATION_N 1
        set_default HERO_EVAL_TP_SIZE 2
        set_default HERO_EVAL_GPU_MEMORY_UTILIZATION 0.5
        set_default HERO_EVAL_N_SAMPLES 8
        set_default HERO_SFT_TRAIN_BATCH_SIZE 32
        set_default HERO_SFT_MICRO_BATCH_SIZE_PER_GPU 1
        set_default HERO_SFT_MAX_LENGTH 4096
        set_default HERO_SFT_MAX_TOKEN_LEN_PER_GPU 6144
        ;;
    4x80|4x80gb)
        set_default HERO_GPUS_PER_NODE 4
        set_default HERO_NNODES 1
        set_default HERO_GEN_GPUS_PER_NODE 4
        set_default HERO_GEN_NNODES 1
        set_default HERO_SFT_GPUS_PER_NODE 4
        set_default HERO_SFT_NNODES 1
        set_default HERO_EVAL_GPUS_PER_NODE 4
        set_default HERO_EVAL_NNODES 1
        set_default HERO_MAX_PROMPT_LENGTH 1024
        set_default HERO_MAX_RESPONSE_LENGTH 4096
        set_default HERO_TRAIN_BATCH_SIZE 128
        set_default HERO_PPO_MINI_BATCH_SIZE 64
        set_default HERO_PPO_MICRO_BATCH_SIZE_PER_GPU 4
        set_default HERO_ROLLOUT_N 8
        set_default HERO_ROLLOUT_TP_SIZE 2
        set_default HERO_ROLLOUT_GPU_MEMORY_UTILIZATION 0.7
        set_default HERO_ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU 4
        set_default HERO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU 4
        set_default HERO_ROLLOUT_MAX_NUM_SEQS 128
        set_default HERO_RM_GPUS_PER_NODE 4
        set_default HERO_RM_NNODES 1
        set_default HERO_RM_TP_SIZE 2
        set_default HERO_RM_GPU_MEMORY_UTILIZATION 0.6
        set_default HERO_RM_MAX_NUM_SEQS 128
        set_default HERO_GEN_TP_SIZE 2
        set_default HERO_GEN_GPU_MEMORY_UTILIZATION 0.75
        set_default HERO_SOURCE_GENERATION_N 1
        set_default HERO_EVAL_TP_SIZE 2
        set_default HERO_EVAL_GPU_MEMORY_UTILIZATION 0.75
        set_default HERO_EVAL_N_SAMPLES 8
        set_default HERO_SFT_TRAIN_BATCH_SIZE 64
        set_default HERO_SFT_MICRO_BATCH_SIZE_PER_GPU 2
        set_default HERO_SFT_MAX_LENGTH 6144
        set_default HERO_SFT_MAX_TOKEN_LEN_PER_GPU 8192
        ;;
    2x80|2x80gb)
        set_default HERO_GPUS_PER_NODE 2
        set_default HERO_NNODES 1
        set_default HERO_GEN_GPUS_PER_NODE 2
        set_default HERO_GEN_NNODES 1
        set_default HERO_SFT_GPUS_PER_NODE 2
        set_default HERO_SFT_NNODES 1
        set_default HERO_EVAL_GPUS_PER_NODE 2
        set_default HERO_EVAL_NNODES 1
        set_default HERO_MAX_PROMPT_LENGTH 1024
        set_default HERO_MAX_RESPONSE_LENGTH 4096
        set_default HERO_TRAIN_BATCH_SIZE 64
        set_default HERO_PPO_MINI_BATCH_SIZE 32
        set_default HERO_PPO_MICRO_BATCH_SIZE_PER_GPU 4
        set_default HERO_ROLLOUT_N 8
        set_default HERO_ROLLOUT_TP_SIZE 2
        set_default HERO_ROLLOUT_GPU_MEMORY_UTILIZATION 0.7
        set_default HERO_ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU 4
        set_default HERO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU 4
        set_default HERO_ROLLOUT_MAX_NUM_SEQS 64
        set_default HERO_RM_GPUS_PER_NODE 2
        set_default HERO_RM_NNODES 1
        set_default HERO_RM_TP_SIZE 2
        set_default HERO_RM_GPU_MEMORY_UTILIZATION 0.6
        set_default HERO_RM_MAX_NUM_SEQS 64
        set_default HERO_GEN_TP_SIZE 2
        set_default HERO_GEN_GPU_MEMORY_UTILIZATION 0.75
        set_default HERO_SOURCE_GENERATION_N 1
        set_default HERO_EVAL_TP_SIZE 2
        set_default HERO_EVAL_GPU_MEMORY_UTILIZATION 0.75
        set_default HERO_EVAL_N_SAMPLES 8
        set_default HERO_SFT_TRAIN_BATCH_SIZE 32
        set_default HERO_SFT_MICRO_BATCH_SIZE_PER_GPU 2
        set_default HERO_SFT_MAX_LENGTH 6144
        set_default HERO_SFT_MAX_TOKEN_LEN_PER_GPU 8192
        ;;
    *)
        echo "Unknown HERO_GPU_PROFILE=$gpu_profile. Use 8x24gb, 4x80gb, or 2x80gb." >&2
        exit 1
        ;;
esac

artifact_root=${HERO_ARTIFACT_ROOT:-$HOME/data/hero_paper_reproduction}
run_name=${HERO_RUN_NAME:-step_by_step_run}
work_dir=${HERO_WORK_DIR:-$artifact_root/$run_name}
source_dir=${HERO_SOURCE_DIR:-$work_dir/source_generation}
data_dir=${HERO_DATA_DIR:-$work_dir/openmathreasoning_hero}
eval_dir=${HERO_EVAL_DIR:-$work_dir/hero_eval}
sft_data_dir=${HERO_SFT_DATA_DIR:-$work_dir/openmathreasoning_hero_sft}
sft_output_dir=${HERO_SFT_OUTPUT_DIR:-$work_dir/checkpoints/hero_cold_start_sft}
train_output_dir=${HERO_TRAIN_OUTPUT_DIR:-$work_dir/checkpoints/hero_rl}
eval_output_dir=${HERO_EVAL_OUTPUT_DIR:-$work_dir/eval_results}
source_prompts_path=${HERO_SOURCE_PROMPTS_PATH:-$source_dir/source_prompts.parquet}
source_generated_path=${HERO_SOURCE_GENERATED_PATH:-$source_dir/source_generated.parquet}
filtered_tbr_path=${HERO_FILTERED_TBR_PATH:-$eval_dir/textbook_reasoning_filtered.parquet}

base_model_path=${HERO_BASE_MODEL_PATH:-${HERO_MODEL_PATH:-Qwen/Qwen3-4B-Base}}
trust_remote_code=${HERO_TRUST_REMOTE_CODE:-True}
source_dataset_trust_remote_code=${HERO_SOURCE_DATASET_TRUST_REMOTE_CODE:-False}

dataset_name=${HERO_DATASET:-nvidia/OpenMathReasoning}
dataset_config=${HERO_DATASET_CONFIG:-}
dataset_split=${HERO_DATASET_SPLIT:-cot}
local_dataset_path=${HERO_LOCAL_DATASET_PATH:-}
source_question_col=${HERO_SOURCE_QUESTION_COL:-problem}
source_answer_col=${HERO_SOURCE_ANSWER_COL:-expected_answer}
question_col=${HERO_QUESTION_COL:-question}
answer_col=${HERO_ANSWER_COL:-answer}
problem_type_col=${HERO_PROBLEM_TYPE_COL:-problem_type}
problem_type_value=${HERO_PROBLEM_TYPE_VALUE:-has_answer_extracted}
source_sample_size=${HERO_SOURCE_SAMPLE_SIZE:-40000}
seed=${HERO_SEED:-42}

gen_nnodes=${HERO_GEN_NNODES:-1}
gen_gpus_per_node=${HERO_GEN_GPUS_PER_NODE:-${HERO_GPUS_PER_NODE}}
gen_tp_size=${HERO_GEN_TP_SIZE:-2}
gen_gpu_memory_utilization=${HERO_GEN_GPU_MEMORY_UTILIZATION:-0.75}
source_generation_n=${HERO_SOURCE_GENERATION_N:-1}
source_temperature=${HERO_SOURCE_TEMPERATURE:-1.0}
source_top_p=${HERO_SOURCE_TOP_P:-0.95}
max_prompt_length=${HERO_MAX_PROMPT_LENGTH:-1024}
max_response_length=${HERO_MAX_RESPONSE_LENGTH:-4096}

eval_benchmarks=${HERO_EVAL_BENCHMARKS_BUILD:-"math500 amc minerva olympiad hardverify_math textbook_reasoning"}
hvm_local_path=${HERO_HVM_LOCAL_PATH:-}
tbr_local_path=${HERO_TBR_LOCAL_PATH:-}

#!/usr/bin/env bash

if [[ -n "${_EIF_STEP_COMMON_SH:-}" ]]; then
    return 0
fi
_EIF_STEP_COMMON_SH=1

eif_step_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${eif_step_dir}/../../.." && pwd)

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

ensure_eif_dirs() {
    mkdir -p "$work_dir" "$source_dir" "$data_dir" "$eval_dir" "$sft_data_dir" "$sft_output_dir" "$train_output_dir" "$eval_output_dir"
}

gpu_profile=${EIF_GPU_PROFILE:-8x24gb}

case "$gpu_profile" in
    8x24|8x24gb)
        set_default EIF_GPUS_PER_NODE 8
        set_default EIF_NNODES 1
        set_default EIF_GEN_GPUS_PER_NODE 8
        set_default EIF_GEN_NNODES 1
        set_default EIF_SFT_GPUS_PER_NODE 8
        set_default EIF_SFT_NNODES 1
        set_default EIF_EVAL_GPUS_PER_NODE 8
        set_default EIF_EVAL_NNODES 1
        set_default EIF_MAX_PROMPT_LENGTH 1024
        set_default EIF_MAX_RESPONSE_LENGTH 2048
        set_default EIF_TRAIN_BATCH_SIZE 64
        set_default EIF_PPO_MINI_BATCH_SIZE 64
        set_default EIF_PPO_MICRO_BATCH_SIZE_PER_GPU 2
        set_default EIF_ROLLOUT_N 4
        set_default EIF_ROLLOUT_TP_SIZE 2
        set_default EIF_ROLLOUT_GPU_MEMORY_UTILIZATION 0.5
        set_default EIF_ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU 2
        set_default EIF_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU 2
        set_default EIF_ROLLOUT_MAX_NUM_SEQS 128
        set_default EIF_RM_GPUS_PER_NODE 8
        set_default EIF_RM_NNODES 1
        set_default EIF_RM_TP_SIZE 2
        set_default EIF_RM_GPU_MEMORY_UTILIZATION 0.5
        set_default EIF_RM_MAX_NUM_SEQS 128
        set_default EIF_GEN_TP_SIZE 2
        set_default EIF_GEN_GPU_MEMORY_UTILIZATION 0.5
        set_default EIF_SOURCE_GENERATION_N 1
        set_default EIF_EVAL_TP_SIZE 2
        set_default EIF_EVAL_GPU_MEMORY_UTILIZATION 0.5
        set_default EIF_EVAL_N_SAMPLES 8
        set_default EIF_SFT_TRAIN_BATCH_SIZE 32
        set_default EIF_SFT_MICRO_BATCH_SIZE_PER_GPU 1
        set_default EIF_SFT_MAX_LENGTH 4096
        set_default EIF_SFT_MAX_TOKEN_LEN_PER_GPU 6144
        ;;
    4x80|4x80gb)
        set_default EIF_GPUS_PER_NODE 4
        set_default EIF_NNODES 1
        set_default EIF_GEN_GPUS_PER_NODE 4
        set_default EIF_GEN_NNODES 1
        set_default EIF_SFT_GPUS_PER_NODE 4
        set_default EIF_SFT_NNODES 1
        set_default EIF_EVAL_GPUS_PER_NODE 4
        set_default EIF_EVAL_NNODES 1
        set_default EIF_MAX_PROMPT_LENGTH 1024
        set_default EIF_MAX_RESPONSE_LENGTH 4096
        set_default EIF_TRAIN_BATCH_SIZE 128
        set_default EIF_PPO_MINI_BATCH_SIZE 64
        set_default EIF_PPO_MICRO_BATCH_SIZE_PER_GPU 4
        set_default EIF_ROLLOUT_N 8
        set_default EIF_ROLLOUT_TP_SIZE 2
        set_default EIF_ROLLOUT_GPU_MEMORY_UTILIZATION 0.7
        set_default EIF_ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU 4
        set_default EIF_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU 4
        set_default EIF_ROLLOUT_MAX_NUM_SEQS 128
        set_default EIF_RM_GPUS_PER_NODE 4
        set_default EIF_RM_NNODES 1
        set_default EIF_RM_TP_SIZE 2
        set_default EIF_RM_GPU_MEMORY_UTILIZATION 0.6
        set_default EIF_RM_MAX_NUM_SEQS 128
        set_default EIF_GEN_TP_SIZE 2
        set_default EIF_GEN_GPU_MEMORY_UTILIZATION 0.75
        set_default EIF_SOURCE_GENERATION_N 1
        set_default EIF_EVAL_TP_SIZE 2
        set_default EIF_EVAL_GPU_MEMORY_UTILIZATION 0.75
        set_default EIF_EVAL_N_SAMPLES 8
        set_default EIF_SFT_TRAIN_BATCH_SIZE 64
        set_default EIF_SFT_MICRO_BATCH_SIZE_PER_GPU 2
        set_default EIF_SFT_MAX_LENGTH 6144
        set_default EIF_SFT_MAX_TOKEN_LEN_PER_GPU 8192
        ;;
    2x80|2x80gb)
        set_default EIF_GPUS_PER_NODE 2
        set_default EIF_NNODES 1
        set_default EIF_GEN_GPUS_PER_NODE 2
        set_default EIF_GEN_NNODES 1
        set_default EIF_SFT_GPUS_PER_NODE 2
        set_default EIF_SFT_NNODES 1
        set_default EIF_EVAL_GPUS_PER_NODE 2
        set_default EIF_EVAL_NNODES 1
        set_default EIF_MAX_PROMPT_LENGTH 1024
        set_default EIF_MAX_RESPONSE_LENGTH 4096
        set_default EIF_TRAIN_BATCH_SIZE 64
        set_default EIF_PPO_MINI_BATCH_SIZE 32
        set_default EIF_PPO_MICRO_BATCH_SIZE_PER_GPU 4
        set_default EIF_ROLLOUT_N 8
        set_default EIF_ROLLOUT_TP_SIZE 2
        set_default EIF_ROLLOUT_GPU_MEMORY_UTILIZATION 0.7
        set_default EIF_ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU 4
        set_default EIF_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU 4
        set_default EIF_ROLLOUT_MAX_NUM_SEQS 64
        set_default EIF_RM_GPUS_PER_NODE 2
        set_default EIF_RM_NNODES 1
        set_default EIF_RM_TP_SIZE 2
        set_default EIF_RM_GPU_MEMORY_UTILIZATION 0.6
        set_default EIF_RM_MAX_NUM_SEQS 64
        set_default EIF_GEN_TP_SIZE 2
        set_default EIF_GEN_GPU_MEMORY_UTILIZATION 0.75
        set_default EIF_SOURCE_GENERATION_N 1
        set_default EIF_EVAL_TP_SIZE 2
        set_default EIF_EVAL_GPU_MEMORY_UTILIZATION 0.75
        set_default EIF_EVAL_N_SAMPLES 8
        set_default EIF_SFT_TRAIN_BATCH_SIZE 32
        set_default EIF_SFT_MICRO_BATCH_SIZE_PER_GPU 2
        set_default EIF_SFT_MAX_LENGTH 6144
        set_default EIF_SFT_MAX_TOKEN_LEN_PER_GPU 8192
        ;;
    *)
        echo "Unknown EIF_GPU_PROFILE=$gpu_profile. Use 8x24gb, 4x80gb, or 2x80gb." >&2
        exit 1
        ;;
esac

artifact_root=${EIF_ARTIFACT_ROOT:-$HOME/data/eif_reproduction}
run_name=${EIF_RUN_NAME:-step_by_step_run}
work_dir=${EIF_WORK_DIR:-$artifact_root/$run_name}
source_dir=${EIF_SOURCE_DIR:-$work_dir/source_generation}
data_dir=${EIF_DATA_DIR:-$work_dir/openmathreasoning_hero}
eval_dir=${EIF_EVAL_DIR:-$work_dir/hero_eval}
sft_data_dir=${EIF_SFT_DATA_DIR:-$work_dir/openmathreasoning_hero_sft}
sft_output_dir=${EIF_SFT_OUTPUT_DIR:-$work_dir/checkpoints/eif_cold_start_sft}
train_output_dir=${EIF_TRAIN_OUTPUT_DIR:-$work_dir/checkpoints/eif_rl}
eval_output_dir=${EIF_EVAL_OUTPUT_DIR:-$work_dir/eval_results}
source_prompts_path=${EIF_SOURCE_PROMPTS_PATH:-$source_dir/source_prompts.parquet}
source_generated_path=${EIF_SOURCE_GENERATED_PATH:-$source_dir/source_generated.parquet}
filtered_tbr_path=${EIF_FILTERED_TBR_PATH:-$eval_dir/textbook_reasoning_filtered.parquet}

base_model_path=${EIF_BASE_MODEL_PATH:-${EIF_MODEL_PATH:-Qwen/Qwen3-4B-Base}}
trust_remote_code=${EIF_TRUST_REMOTE_CODE:-True}
source_dataset_trust_remote_code=${EIF_SOURCE_DATASET_TRUST_REMOTE_CODE:-False}

dataset_name=${EIF_DATASET:-nvidia/OpenMathReasoning}
dataset_config=${EIF_DATASET_CONFIG:-}
dataset_split=${EIF_DATASET_SPLIT:-cot}
local_dataset_path=${EIF_LOCAL_DATASET_PATH:-}
source_question_col=${EIF_SOURCE_QUESTION_COL:-problem}
source_answer_col=${EIF_SOURCE_ANSWER_COL:-expected_answer}
question_col=${EIF_QUESTION_COL:-question}
answer_col=${EIF_ANSWER_COL:-answer}
problem_type_col=${EIF_PROBLEM_TYPE_COL:-problem_type}
problem_type_value=${EIF_PROBLEM_TYPE_VALUE:-has_answer_extracted}
source_sample_size=${EIF_SOURCE_SAMPLE_SIZE:-40000}
seed=${EIF_SEED:-42}

gen_nnodes=${EIF_GEN_NNODES:-1}
gen_gpus_per_node=${EIF_GEN_GPUS_PER_NODE:-${EIF_GPUS_PER_NODE}}
gen_tp_size=${EIF_GEN_TP_SIZE:-2}
gen_gpu_memory_utilization=${EIF_GEN_GPU_MEMORY_UTILIZATION:-0.75}
source_generation_n=${EIF_SOURCE_GENERATION_N:-1}
source_temperature=${EIF_SOURCE_TEMPERATURE:-1.0}
source_top_p=${EIF_SOURCE_TOP_P:-0.95}
max_prompt_length=${EIF_MAX_PROMPT_LENGTH:-1024}
max_response_length=${EIF_MAX_RESPONSE_LENGTH:-4096}

eval_benchmarks=${EIF_EVAL_BENCHMARKS_BUILD:-"math500 amc minerva olympiad hardverify_math textbook_reasoning"}
hvm_local_path=${EIF_HVM_LOCAL_PATH:-}
tbr_local_path=${EIF_TBR_LOCAL_PATH:-}

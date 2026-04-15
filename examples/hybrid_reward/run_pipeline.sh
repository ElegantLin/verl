#!/usr/bin/env bash
# End-to-end hybrid reward pipeline (HERO or EIF).
#
# Shared stages:
#   data-train -> data-eval -> sft
# Algorithm-specific stages:
#   rl -> eval
#
# Usage:
#   ALGORITHM=hero bash examples/hybrid_reward/run_pipeline.sh
#   ALGORITHM=eif  bash examples/hybrid_reward/run_pipeline.sh

set -euo pipefail
set -x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

algorithm=${ALGORITHM:-hero}

case "$algorithm" in
    hero)
        default_artifact_root=$HOME/data/hero_paper_reproduction
        default_sft_project=hero_paper_reproduction
        ;;
    eif)
        default_artifact_root=$HOME/data/eif_reproduction
        default_sft_project=eif_cold_start
        ;;
    *)
        echo "Unknown ALGORITHM=$algorithm. Use hero or eif." >&2
        exit 1
        ;;
esac

run_name=${RL_PIPELINE_RUN_NAME:-$(date +%Y%m%d_%H%M%S)}
artifact_root=${RL_PIPELINE_ARTIFACT_ROOT:-$default_artifact_root}
work_dir=${RL_PIPELINE_WORK_DIR:-$artifact_root/$run_name}

export ALGORITHM="$algorithm"
export RL_PIPELINE_RUN_NAME="$run_name"
export RL_PIPELINE_ARTIFACT_ROOT="$artifact_root"
export RL_PIPELINE_WORK_DIR="$work_dir"
export RL_PIPELINE_SOURCE_DIR="${RL_PIPELINE_SOURCE_DIR:-$work_dir/source_generation}"
export RL_PIPELINE_DATA_DIR="${RL_PIPELINE_DATA_DIR:-$work_dir/openmathreasoning_hero}"
export RL_PIPELINE_EVAL_DIR="${RL_PIPELINE_EVAL_DIR:-$work_dir/hero_eval}"
export RL_PIPELINE_SFT_DATA_DIR="${RL_PIPELINE_SFT_DATA_DIR:-$work_dir/openmathreasoning_hero_sft}"
export RL_PIPELINE_SFT_OUTPUT_DIR="${RL_PIPELINE_SFT_OUTPUT_DIR:-$work_dir/checkpoints/cold_start_sft}"
export RL_PIPELINE_TRAIN_OUTPUT_DIR="${RL_PIPELINE_TRAIN_OUTPUT_DIR:-$work_dir/checkpoints/${algorithm}_rl}"
export RL_PIPELINE_EVAL_OUTPUT_DIR="${RL_PIPELINE_EVAL_OUTPUT_DIR:-$work_dir/eval_results/${algorithm}}"
export RL_PIPELINE_SOURCE_PROMPTS_PATH="${RL_PIPELINE_SOURCE_PROMPTS_PATH:-$RL_PIPELINE_SOURCE_DIR/source_prompts.parquet}"
export RL_PIPELINE_SOURCE_GENERATED_PATH="${RL_PIPELINE_SOURCE_GENERATED_PATH:-$RL_PIPELINE_SOURCE_DIR/source_generated.parquet}"
export RL_PIPELINE_FILTERED_TBR_PATH="${RL_PIPELINE_FILTERED_TBR_PATH:-$RL_PIPELINE_EVAL_DIR/textbook_reasoning_filtered.parquet}"

export RL_PIPELINE_BASE_MODEL_PATH="${RL_PIPELINE_BASE_MODEL_PATH:-${RL_PIPELINE_MODEL_PATH:-Qwen/Qwen3-4B-Base}}"
export RL_PIPELINE_TRUST_REMOTE_CODE="${RL_PIPELINE_TRUST_REMOTE_CODE:-True}"
export RL_PIPELINE_SOURCE_DATASET_TRUST_REMOTE_CODE="${RL_PIPELINE_SOURCE_DATASET_TRUST_REMOTE_CODE:-False}"

export RL_PIPELINE_DATASET="${RL_PIPELINE_DATASET:-nvidia/OpenMathReasoning}"
export RL_PIPELINE_DATASET_CONFIG="${RL_PIPELINE_DATASET_CONFIG:-}"
export RL_PIPELINE_DATASET_SPLIT="${RL_PIPELINE_DATASET_SPLIT:-cot}"
export RL_PIPELINE_LOCAL_DATASET_PATH="${RL_PIPELINE_LOCAL_DATASET_PATH:-}"
export RL_PIPELINE_SOURCE_QUESTION_COL="${RL_PIPELINE_SOURCE_QUESTION_COL:-problem}"
export RL_PIPELINE_SOURCE_ANSWER_COL="${RL_PIPELINE_SOURCE_ANSWER_COL:-expected_answer}"
export RL_PIPELINE_QUESTION_COL="${RL_PIPELINE_QUESTION_COL:-question}"
export RL_PIPELINE_ANSWER_COL="${RL_PIPELINE_ANSWER_COL:-answer}"
export RL_PIPELINE_PROBLEM_TYPE_COL="${RL_PIPELINE_PROBLEM_TYPE_COL:-problem_type}"
export RL_PIPELINE_PROBLEM_TYPE_VALUE="${RL_PIPELINE_PROBLEM_TYPE_VALUE:-has_answer_extracted}"
export RL_PIPELINE_SOURCE_SAMPLE_SIZE="${RL_PIPELINE_SOURCE_SAMPLE_SIZE:-40000}"
export RL_PIPELINE_SEED="${RL_PIPELINE_SEED:-42}"

export RL_PIPELINE_GEN_NNODES="${RL_PIPELINE_GEN_NNODES:-1}"
export RL_PIPELINE_GEN_GPUS_PER_NODE="${RL_PIPELINE_GEN_GPUS_PER_NODE:-8}"
export RL_PIPELINE_GEN_TP_SIZE="${RL_PIPELINE_GEN_TP_SIZE:-2}"
export RL_PIPELINE_GEN_GPU_MEMORY_UTILIZATION="${RL_PIPELINE_GEN_GPU_MEMORY_UTILIZATION:-0.85}"
export RL_PIPELINE_SOURCE_GENERATION_N="${RL_PIPELINE_SOURCE_GENERATION_N:-1}"
export RL_PIPELINE_SOURCE_TEMPERATURE="${RL_PIPELINE_SOURCE_TEMPERATURE:-1.0}"
export RL_PIPELINE_SOURCE_TOP_P="${RL_PIPELINE_SOURCE_TOP_P:-0.95}"
export RL_PIPELINE_MAX_PROMPT_LENGTH="${RL_PIPELINE_MAX_PROMPT_LENGTH:-1024}"
export RL_PIPELINE_MAX_RESPONSE_LENGTH="${RL_PIPELINE_MAX_RESPONSE_LENGTH:-8192}"

export RL_PIPELINE_ENABLE_COLD_START_SFT="${RL_PIPELINE_ENABLE_COLD_START_SFT:-1}"
export RL_PIPELINE_FILTER_TBR="${RL_PIPELINE_FILTER_TBR:-0}"
export RL_PIPELINE_FORCE_SOURCE_BUILD="${RL_PIPELINE_FORCE_SOURCE_BUILD:-0}"
export RL_PIPELINE_FORCE_SOURCE_GENERATION="${RL_PIPELINE_FORCE_SOURCE_GENERATION:-0}"
export RL_PIPELINE_FORCE_RL_PREPROCESS="${RL_PIPELINE_FORCE_RL_PREPROCESS:-0}"
export RL_PIPELINE_FORCE_EVAL_PREPROCESS="${RL_PIPELINE_FORCE_EVAL_PREPROCESS:-0}"
export RL_PIPELINE_FORCE_TBR_FILTER="${RL_PIPELINE_FORCE_TBR_FILTER:-0}"

export RL_PIPELINE_EVAL_BENCHMARKS_BUILD="${RL_PIPELINE_EVAL_BENCHMARKS_BUILD:-math500 amc minerva olympiad hardverify_math textbook_reasoning}"
export RL_PIPELINE_HVM_LOCAL_PATH="${RL_PIPELINE_HVM_LOCAL_PATH:-}"
export RL_PIPELINE_TBR_LOCAL_PATH="${RL_PIPELINE_TBR_LOCAL_PATH:-}"
export RL_PIPELINE_SFT_PROJECT_NAME="${RL_PIPELINE_SFT_PROJECT_NAME:-$default_sft_project}"

for v in DISABLE_TBR_MATH_VERIFY_FILTER TBR_ANSWER_MODEL TBR_ANSWER_BASE_URL TBR_ANSWER_API_KEY_ENV \
         TBR_ANSWER_JUDGE_MODEL TBR_ANSWER_JUDGE_BASE_URL TBR_ANSWER_JUDGE_API_KEY_ENV \
         TBR_SUITABILITY_MODEL TBR_SUITABILITY_BASE_URL TBR_SUITABILITY_API_KEY_ENV; do
    if [[ -n "${!v:-}" ]]; then
        export "RL_PIPELINE_${v}=${!v}"
    fi
done

source "$script_dir/stage_lib.sh"

run_stage_data_train
run_stage_data_eval

if [[ "${RL_PIPELINE_ENABLE_COLD_START_SFT:-1}" == "1" ]]; then
    run_stage_sft
fi

run_stage_rl

if [[ "${RUN_FINAL_EVAL:-1}" == "1" ]]; then
    unset RL_PIPELINE_MODEL_PATH
    run_stage_eval
fi

echo "Pipeline completed (ALGORITHM=$algorithm)."
echo "Work directory: $work_dir"
echo "RL checkpoint root: $train_output_dir"
if [[ "${RUN_FINAL_EVAL:-1}" == "1" ]]; then
    echo "Evaluation output dir: $eval_output_dir"
fi

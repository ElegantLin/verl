#!/usr/bin/env bash
# End-to-end HERO paper reproduction pipeline.
#
# Stages 1-6 (data preprocessing, candidate generation, SFT) are shared
# with the EIF pipeline via examples/shared/run_data_pipeline.sh.
# This script maps HERO_* env vars to RL_PIPELINE_*, runs the shared
# preprocessing, then executes the HERO-specific training and evaluation.
#
# Stages:
#   1-6. Shared preprocessing (see examples/shared/run_data_pipeline.sh)
#   7. Run HERO RL training
#   8. Run evaluation on the latest exported actor checkpoint

set -euo pipefail
set -x

source examples/shared/bash_utils.sh

# ── Map HERO_* env vars to RL_PIPELINE_* for shared preprocessing ────
run_name=${HERO_RUN_NAME:-$(date +%Y%m%d_%H%M%S)}
artifact_root=${HERO_ARTIFACT_ROOT:-$HOME/data/hero_paper_reproduction}
work_dir=${HERO_WORK_DIR:-$artifact_root/$run_name}

export RL_PIPELINE_RUN_NAME="$run_name"
export RL_PIPELINE_ARTIFACT_ROOT="$artifact_root"
export RL_PIPELINE_WORK_DIR="$work_dir"
export RL_PIPELINE_SOURCE_DIR="${HERO_SOURCE_DIR:-$work_dir/source_generation}"
export RL_PIPELINE_DATA_DIR="${HERO_DATA_DIR:-$work_dir/openmathreasoning_hero}"
export RL_PIPELINE_EVAL_DIR="${HERO_EVAL_DIR:-$work_dir/hero_eval}"
export RL_PIPELINE_SFT_DATA_DIR="${HERO_SFT_DATA_DIR:-$work_dir/openmathreasoning_hero_sft}"
export RL_PIPELINE_SFT_OUTPUT_DIR="${HERO_SFT_OUTPUT_DIR:-$work_dir/checkpoints/hero_cold_start_sft}"
export RL_PIPELINE_TRAIN_OUTPUT_DIR="${HERO_TRAIN_OUTPUT_DIR:-$work_dir/checkpoints/hero_rl}"
export RL_PIPELINE_EVAL_OUTPUT_DIR="${HERO_EVAL_OUTPUT_DIR:-$work_dir/eval_results}"
export RL_PIPELINE_SOURCE_PROMPTS_PATH="${HERO_SOURCE_PROMPTS_PATH:-$RL_PIPELINE_SOURCE_DIR/source_prompts.parquet}"
export RL_PIPELINE_SOURCE_GENERATED_PATH="${HERO_SOURCE_GENERATED_PATH:-$RL_PIPELINE_SOURCE_DIR/source_generated.parquet}"
export RL_PIPELINE_FILTERED_TBR_PATH="${HERO_FILTERED_TBR_PATH:-$RL_PIPELINE_EVAL_DIR/textbook_reasoning_filtered.parquet}"

export RL_PIPELINE_BASE_MODEL_PATH="${HERO_BASE_MODEL_PATH:-${HERO_MODEL_PATH:-Qwen/Qwen3-4B-Base}}"
export RL_PIPELINE_MODEL_PATH="${HERO_MODEL_PATH:-Qwen/Qwen3-4B-Base}"
export RL_PIPELINE_TRUST_REMOTE_CODE="${HERO_TRUST_REMOTE_CODE:-True}"
export RL_PIPELINE_SOURCE_DATASET_TRUST_REMOTE_CODE="${HERO_SOURCE_DATASET_TRUST_REMOTE_CODE:-False}"

export RL_PIPELINE_DATASET="${HERO_DATASET:-nvidia/OpenMathReasoning}"
export RL_PIPELINE_DATASET_CONFIG="${HERO_DATASET_CONFIG:-}"
export RL_PIPELINE_DATASET_SPLIT="${HERO_DATASET_SPLIT:-cot}"
export RL_PIPELINE_LOCAL_DATASET_PATH="${HERO_LOCAL_DATASET_PATH:-}"
export RL_PIPELINE_SOURCE_QUESTION_COL="${HERO_SOURCE_QUESTION_COL:-problem}"
export RL_PIPELINE_SOURCE_ANSWER_COL="${HERO_SOURCE_ANSWER_COL:-expected_answer}"
export RL_PIPELINE_QUESTION_COL="${HERO_QUESTION_COL:-question}"
export RL_PIPELINE_ANSWER_COL="${HERO_ANSWER_COL:-answer}"
export RL_PIPELINE_PROBLEM_TYPE_COL="${HERO_PROBLEM_TYPE_COL:-problem_type}"
export RL_PIPELINE_PROBLEM_TYPE_VALUE="${HERO_PROBLEM_TYPE_VALUE:-has_answer_extracted}"
export RL_PIPELINE_SOURCE_SAMPLE_SIZE="${HERO_SOURCE_SAMPLE_SIZE:-40000}"
export RL_PIPELINE_SEED="${HERO_SEED:-42}"

export RL_PIPELINE_GEN_NNODES="${HERO_GEN_NNODES:-1}"
export RL_PIPELINE_GEN_GPUS_PER_NODE="${HERO_GEN_GPUS_PER_NODE:-8}"
export RL_PIPELINE_GEN_TP_SIZE="${HERO_GEN_TP_SIZE:-2}"
export RL_PIPELINE_GEN_GPU_MEMORY_UTILIZATION="${HERO_GEN_GPU_MEMORY_UTILIZATION:-0.85}"
export RL_PIPELINE_SOURCE_GENERATION_N="${HERO_SOURCE_GENERATION_N:-1}"
export RL_PIPELINE_SOURCE_TEMPERATURE="${HERO_SOURCE_TEMPERATURE:-1.0}"
export RL_PIPELINE_SOURCE_TOP_P="${HERO_SOURCE_TOP_P:-0.95}"
export RL_PIPELINE_MAX_PROMPT_LENGTH="${HERO_MAX_PROMPT_LENGTH:-1024}"
export RL_PIPELINE_MAX_RESPONSE_LENGTH="${HERO_MAX_RESPONSE_LENGTH:-8192}"

export RL_PIPELINE_ENABLE_COLD_START_SFT="${HERO_ENABLE_COLD_START_SFT:-1}"
export RL_PIPELINE_FILTER_TBR="${HERO_FILTER_TBR:-0}"
export RL_PIPELINE_FORCE_SOURCE_BUILD="${HERO_FORCE_SOURCE_BUILD:-0}"
export RL_PIPELINE_FORCE_SOURCE_GENERATION="${HERO_FORCE_SOURCE_GENERATION:-0}"
export RL_PIPELINE_FORCE_RL_PREPROCESS="${HERO_FORCE_RL_PREPROCESS:-0}"
export RL_PIPELINE_FORCE_EVAL_PREPROCESS="${HERO_FORCE_EVAL_PREPROCESS:-0}"
export RL_PIPELINE_FORCE_TBR_FILTER="${HERO_FORCE_TBR_FILTER:-0}"

export RL_PIPELINE_EVAL_BENCHMARKS_BUILD="${HERO_EVAL_BENCHMARKS_BUILD:-math500 amc minerva olympiad hardverify_math textbook_reasoning}"
export RL_PIPELINE_HVM_LOCAL_PATH="${HERO_HVM_LOCAL_PATH:-}"
export RL_PIPELINE_TBR_LOCAL_PATH="${HERO_TBR_LOCAL_PATH:-}"

export RL_PIPELINE_SFT_PROJECT_NAME="hero_paper_reproduction"

# TBR filter vars
for v in DISABLE_TBR_MATH_VERIFY_FILTER TBR_ANSWER_MODEL TBR_ANSWER_BASE_URL TBR_ANSWER_API_KEY_ENV \
         TBR_ANSWER_JUDGE_MODEL TBR_ANSWER_JUDGE_BASE_URL TBR_ANSWER_JUDGE_API_KEY_ENV \
         TBR_SUITABILITY_MODEL TBR_SUITABILITY_BASE_URL TBR_SUITABILITY_API_KEY_ENV; do
    hero_var="HERO_${v}"
    pipeline_var="RL_PIPELINE_${v}"
    if [[ -n "${!hero_var:-}" ]]; then
        export "$pipeline_var=${!hero_var}"
    fi
done

# ═══════════════════════════════════════════════════════════════════════
# Stages 1-6: Shared data preprocessing
# ═══════════════════════════════════════════════════════════════════════
bash examples/shared/run_data_pipeline.sh

data_dir="$RL_PIPELINE_DATA_DIR"
eval_dir="$RL_PIPELINE_EVAL_DIR"
train_model_path="$RL_PIPELINE_TRAIN_MODEL_PATH"
train_output_dir="$RL_PIPELINE_TRAIN_OUTPUT_DIR"
eval_output_dir="$RL_PIPELINE_EVAL_OUTPUT_DIR"

# ═══════════════════════════════════════════════════════════════════════
# Stage 7: HERO RL training
# ═══════════════════════════════════════════════════════════════════════
HERO_MODEL_PATH="$train_model_path" \
HERO_TRUST_REMOTE_CODE="${HERO_TRUST_REMOTE_CODE:-True}" \
HERO_DATA_DIR="$data_dir" \
HERO_EVAL_DIR="$eval_dir" \
HERO_TRAIN_OUTPUT_DIR="$train_output_dir" \
bash examples/hero/run_hero_train.sh

# ═══════════════════════════════════════════════════════════════════════
# Stage 8: Final evaluation
# ═══════════════════════════════════════════════════════════════════════
run_final_eval=${HERO_RUN_FINAL_EVAL:-1}
if [[ "$run_final_eval" == "1" ]]; then
    latest_actor_hf_dir=$(resolve_latest_hf_dir "$train_output_dir" "/actor/huggingface")
    HERO_MODEL_PATH="$latest_actor_hf_dir" \
    HERO_TRUST_REMOTE_CODE="${HERO_TRUST_REMOTE_CODE:-True}" \
    HERO_EVAL_DIR="$eval_dir" \
    HERO_OUTPUT_DIR="$eval_output_dir" \
    bash examples/hero/run_hero_eval.sh
fi

echo "HERO pipeline completed."
echo "Work directory: $work_dir"
echo "RL checkpoint root: $train_output_dir"
if [[ "$run_final_eval" == "1" ]]; then
    echo "Evaluation output dir: $eval_output_dir"
fi

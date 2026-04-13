#!/usr/bin/env bash
# End-to-end EIF pipeline.
#
# Stages 1-6 (data preprocessing, candidate generation, SFT) are shared
# with the HERO pipeline via examples/shared/run_data_pipeline.sh.
# This script maps EIF_* env vars to RL_PIPELINE_*, runs the shared
# preprocessing, then executes the EIF-specific training and evaluation.
#
# Stages:
#   1-6. Shared preprocessing (see examples/shared/run_data_pipeline.sh)
#   7. Run EIF RL training (hybrid_eif_online)
#   8. Run evaluation on the latest exported actor checkpoint

set -euo pipefail
set -x

source examples/shared/bash_utils.sh

# ── Map EIF_* env vars to RL_PIPELINE_* for shared preprocessing ─────
run_name=${EIF_RUN_NAME:-$(date +%Y%m%d_%H%M%S)}
artifact_root=${EIF_ARTIFACT_ROOT:-$HOME/data/eif_reproduction}
work_dir=${EIF_WORK_DIR:-$artifact_root/$run_name}

export RL_PIPELINE_RUN_NAME="$run_name"
export RL_PIPELINE_ARTIFACT_ROOT="$artifact_root"
export RL_PIPELINE_WORK_DIR="$work_dir"
export RL_PIPELINE_SOURCE_DIR="${EIF_SOURCE_DIR:-$work_dir/source_generation}"
export RL_PIPELINE_DATA_DIR="${EIF_DATA_DIR:-$work_dir/openmathreasoning_hero}"
export RL_PIPELINE_EVAL_DIR="${EIF_EVAL_DIR:-$work_dir/hero_eval}"
export RL_PIPELINE_SFT_DATA_DIR="${EIF_SFT_DATA_DIR:-$work_dir/openmathreasoning_hero_sft}"
export RL_PIPELINE_SFT_OUTPUT_DIR="${EIF_SFT_OUTPUT_DIR:-$work_dir/checkpoints/eif_cold_start_sft}"
export RL_PIPELINE_TRAIN_OUTPUT_DIR="${EIF_TRAIN_OUTPUT_DIR:-$work_dir/checkpoints/eif_rl}"
export RL_PIPELINE_EVAL_OUTPUT_DIR="${EIF_EVAL_OUTPUT_DIR:-$work_dir/eval_results}"
export RL_PIPELINE_SOURCE_PROMPTS_PATH="${EIF_SOURCE_PROMPTS_PATH:-$RL_PIPELINE_SOURCE_DIR/source_prompts.parquet}"
export RL_PIPELINE_SOURCE_GENERATED_PATH="${EIF_SOURCE_GENERATED_PATH:-$RL_PIPELINE_SOURCE_DIR/source_generated.parquet}"
export RL_PIPELINE_FILTERED_TBR_PATH="${EIF_FILTERED_TBR_PATH:-$RL_PIPELINE_EVAL_DIR/textbook_reasoning_filtered.parquet}"

export RL_PIPELINE_BASE_MODEL_PATH="${EIF_BASE_MODEL_PATH:-${EIF_MODEL_PATH:-Qwen/Qwen3-4B-Base}}"
export RL_PIPELINE_MODEL_PATH="${EIF_MODEL_PATH:-Qwen/Qwen3-4B-Base}"
export RL_PIPELINE_TRUST_REMOTE_CODE="${EIF_TRUST_REMOTE_CODE:-True}"
export RL_PIPELINE_SOURCE_DATASET_TRUST_REMOTE_CODE="${EIF_SOURCE_DATASET_TRUST_REMOTE_CODE:-False}"

export RL_PIPELINE_DATASET="${EIF_DATASET:-nvidia/OpenMathReasoning}"
export RL_PIPELINE_DATASET_CONFIG="${EIF_DATASET_CONFIG:-}"
export RL_PIPELINE_DATASET_SPLIT="${EIF_DATASET_SPLIT:-cot}"
export RL_PIPELINE_LOCAL_DATASET_PATH="${EIF_LOCAL_DATASET_PATH:-}"
export RL_PIPELINE_SOURCE_QUESTION_COL="${EIF_SOURCE_QUESTION_COL:-problem}"
export RL_PIPELINE_SOURCE_ANSWER_COL="${EIF_SOURCE_ANSWER_COL:-expected_answer}"
export RL_PIPELINE_QUESTION_COL="${EIF_QUESTION_COL:-question}"
export RL_PIPELINE_ANSWER_COL="${EIF_ANSWER_COL:-answer}"
export RL_PIPELINE_PROBLEM_TYPE_COL="${EIF_PROBLEM_TYPE_COL:-problem_type}"
export RL_PIPELINE_PROBLEM_TYPE_VALUE="${EIF_PROBLEM_TYPE_VALUE:-has_answer_extracted}"
export RL_PIPELINE_SOURCE_SAMPLE_SIZE="${EIF_SOURCE_SAMPLE_SIZE:-40000}"
export RL_PIPELINE_SEED="${EIF_SEED:-42}"

export RL_PIPELINE_GEN_NNODES="${EIF_GEN_NNODES:-1}"
export RL_PIPELINE_GEN_GPUS_PER_NODE="${EIF_GEN_GPUS_PER_NODE:-8}"
export RL_PIPELINE_GEN_TP_SIZE="${EIF_GEN_TP_SIZE:-2}"
export RL_PIPELINE_GEN_GPU_MEMORY_UTILIZATION="${EIF_GEN_GPU_MEMORY_UTILIZATION:-0.85}"
export RL_PIPELINE_SOURCE_GENERATION_N="${EIF_SOURCE_GENERATION_N:-1}"
export RL_PIPELINE_SOURCE_TEMPERATURE="${EIF_SOURCE_TEMPERATURE:-1.0}"
export RL_PIPELINE_SOURCE_TOP_P="${EIF_SOURCE_TOP_P:-0.95}"
export RL_PIPELINE_MAX_PROMPT_LENGTH="${EIF_MAX_PROMPT_LENGTH:-1024}"
export RL_PIPELINE_MAX_RESPONSE_LENGTH="${EIF_MAX_RESPONSE_LENGTH:-8192}"

export RL_PIPELINE_ENABLE_COLD_START_SFT="${EIF_ENABLE_COLD_START_SFT:-1}"
export RL_PIPELINE_FILTER_TBR="${EIF_FILTER_TBR:-0}"
export RL_PIPELINE_FORCE_SOURCE_BUILD="${EIF_FORCE_SOURCE_BUILD:-0}"
export RL_PIPELINE_FORCE_SOURCE_GENERATION="${EIF_FORCE_SOURCE_GENERATION:-0}"
export RL_PIPELINE_FORCE_RL_PREPROCESS="${EIF_FORCE_RL_PREPROCESS:-0}"
export RL_PIPELINE_FORCE_EVAL_PREPROCESS="${EIF_FORCE_EVAL_PREPROCESS:-0}"
export RL_PIPELINE_FORCE_TBR_FILTER="${EIF_FORCE_TBR_FILTER:-0}"

export RL_PIPELINE_EVAL_BENCHMARKS_BUILD="${EIF_EVAL_BENCHMARKS_BUILD:-math500 amc minerva olympiad hardverify_math textbook_reasoning}"
export RL_PIPELINE_HVM_LOCAL_PATH="${EIF_HVM_LOCAL_PATH:-}"
export RL_PIPELINE_TBR_LOCAL_PATH="${EIF_TBR_LOCAL_PATH:-}"

export RL_PIPELINE_SFT_PROJECT_NAME="eif_cold_start"

# TBR filter vars
for v in DISABLE_TBR_MATH_VERIFY_FILTER TBR_ANSWER_MODEL TBR_ANSWER_BASE_URL TBR_ANSWER_API_KEY_ENV \
         TBR_ANSWER_JUDGE_MODEL TBR_ANSWER_JUDGE_BASE_URL TBR_ANSWER_JUDGE_API_KEY_ENV \
         TBR_SUITABILITY_MODEL TBR_SUITABILITY_BASE_URL TBR_SUITABILITY_API_KEY_ENV; do
    eif_var="EIF_${v}"
    pipeline_var="RL_PIPELINE_${v}"
    if [[ -n "${!eif_var:-}" ]]; then
        export "$pipeline_var=${!eif_var}"
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
# Stage 7: EIF RL training
# ═══════════════════════════════════════════════════════════════════════
EIF_MODEL_PATH="$train_model_path" \
EIF_TRUST_REMOTE_CODE="${EIF_TRUST_REMOTE_CODE:-True}" \
EIF_DATA_DIR="$data_dir" \
EIF_EVAL_DIR="$eval_dir" \
EIF_TRAIN_OUTPUT_DIR="$train_output_dir" \
bash examples/eif/run_eif_train.sh

# ═══════════════════════════════════════════════════════════════════════
# Stage 8: Final evaluation
# ═══════════════════════════════════════════════════════════════════════
run_final_eval=${EIF_RUN_FINAL_EVAL:-1}
if [[ "$run_final_eval" == "1" ]]; then
    latest_actor_hf_dir=$(resolve_latest_hf_dir "$train_output_dir" "/actor/huggingface")
    EIF_MODEL_PATH="$latest_actor_hf_dir" \
    EIF_TRUST_REMOTE_CODE="${EIF_TRUST_REMOTE_CODE:-True}" \
    EIF_EVAL_DIR="$eval_dir" \
    EIF_OUTPUT_DIR="$eval_output_dir" \
    bash examples/eif/run_eif_eval.sh
fi

echo "EIF pipeline completed."
echo "Work directory: $work_dir"
echo "RL checkpoint root: $train_output_dir"
if [[ "$run_final_eval" == "1" ]]; then
    echo "Evaluation output dir: $eval_output_dir"
fi

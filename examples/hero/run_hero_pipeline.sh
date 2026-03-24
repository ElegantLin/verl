#!/usr/bin/env bash
# End-to-end HERO paper reproduction pipeline.
#
# Stages:
#   1. Build 40k prompt-only OpenMathReasoning source parquet
#   2. Generate candidate responses with the base model
#   3. Build HERO RL train/val splits from generated responses
#   4. Optionally filter TextBookReasoning for paper-style hard-to-verify evaluation
#   5. Build evaluation benchmark parquet files
#   6. Optionally run cold-start SFT
#   7. Run HERO RL training
#   8. Run evaluation on the latest exported actor checkpoint

set -euo pipefail
set -x

resolve_latest_hf_dir() {
    local root_dir="$1"
    local suffix="$2"
    local tracker_file="$root_dir/latest_checkpointed_iteration.txt"
    if [[ ! -f "$tracker_file" ]]; then
        echo "Missing checkpoint tracker: $tracker_file" >&2
        return 1
    fi
    local latest_step
    latest_step=$(<"$tracker_file")
    local resolved_path="$root_dir/global_step_${latest_step}${suffix}"
    if [[ ! -d "$resolved_path" ]]; then
        echo "Missing checkpoint directory: $resolved_path" >&2
        return 1
    fi
    printf '%s\n' "$resolved_path"
}

run_name=${HERO_RUN_NAME:-$(date +%Y%m%d_%H%M%S)}
artifact_root=${HERO_ARTIFACT_ROOT:-$HOME/data/hero_paper_reproduction}
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

dataset_name=${HERO_DATASET:-OpenMathReasoning}
dataset_config=${HERO_DATASET_CONFIG:-}
dataset_split=${HERO_DATASET_SPLIT:-train}
local_dataset_path=${HERO_LOCAL_DATASET_PATH:-}
question_col=${HERO_QUESTION_COL:-question}
answer_col=${HERO_ANSWER_COL:-answer}
problem_type_col=${HERO_PROBLEM_TYPE_COL:-problem_type}
problem_type_value=${HERO_PROBLEM_TYPE_VALUE:-has_answer_extracted}
source_sample_size=${HERO_SOURCE_SAMPLE_SIZE:-40000}
seed=${HERO_SEED:-42}

gen_nnodes=${HERO_GEN_NNODES:-1}
gen_gpus_per_node=${HERO_GEN_GPUS_PER_NODE:-8}
gen_tp_size=${HERO_GEN_TP_SIZE:-2}
gen_gpu_memory_utilization=${HERO_GEN_GPU_MEMORY_UTILIZATION:-0.85}
source_generation_n=${HERO_SOURCE_GENERATION_N:-1}
source_temperature=${HERO_SOURCE_TEMPERATURE:-1.0}
source_top_p=${HERO_SOURCE_TOP_P:-0.95}
max_prompt_length=${HERO_MAX_PROMPT_LENGTH:-1024}
max_response_length=${HERO_MAX_RESPONSE_LENGTH:-8192}

enable_cold_start_sft=${HERO_ENABLE_COLD_START_SFT:-1}
run_final_eval=${HERO_RUN_FINAL_EVAL:-1}
filter_tbr=${HERO_FILTER_TBR:-0}
force_source_build=${HERO_FORCE_SOURCE_BUILD:-0}
force_source_generation=${HERO_FORCE_SOURCE_GENERATION:-0}
force_rl_preprocess=${HERO_FORCE_RL_PREPROCESS:-0}
force_eval_preprocess=${HERO_FORCE_EVAL_PREPROCESS:-0}
force_tbr_filter=${HERO_FORCE_TBR_FILTER:-0}

eval_benchmarks=${HERO_EVAL_BENCHMARKS_BUILD:-"math500 amc minerva olympiad hardverify_math textbook_reasoning"}
hvm_local_path=${HERO_HVM_LOCAL_PATH:-}
tbr_local_path=${HERO_TBR_LOCAL_PATH:-}
tbr_input_for_eval="$tbr_local_path"

mkdir -p "$work_dir" "$source_dir" "$data_dir" "$eval_dir" "$sft_data_dir" "$sft_output_dir" "$train_output_dir" "$eval_output_dir"

if [[ "$force_source_build" == "1" || ! -f "$source_prompts_path" ]]; then
    source_cmd=(
        python3 examples/data_preprocess/openmathreasoning_hero_source.py
        --split "$dataset_split"
        --question_col "$question_col"
        --answer_col "$answer_col"
        --problem_type_col "$problem_type_col"
        --problem_type_value "$problem_type_value"
        --source_sample_size "$source_sample_size"
        --seed "$seed"
        --output_path "$source_prompts_path"
    )
    if [[ -n "$local_dataset_path" ]]; then
        source_cmd+=(--local_dataset_path "$local_dataset_path")
    else
        source_cmd+=(--dataset "$dataset_name")
        if [[ -n "$dataset_config" ]]; then
            source_cmd+=(--dataset_config "$dataset_config")
        fi
        if [[ "$trust_remote_code" == "True" ]]; then
            source_cmd+=(--trust_remote_code)
        fi
    fi
    "${source_cmd[@]}"
fi

if [[ "$force_source_generation" == "1" || ! -f "$source_generated_path" ]]; then
    python3 -m verl.trainer.main_generation_server         trainer.nnodes="$gen_nnodes"         trainer.n_gpus_per_node="$gen_gpus_per_node"         data.train_files="['$source_prompts_path']"         data.prompt_key=prompt         data.output_path="$source_generated_path"         actor_rollout_ref.model.path="$base_model_path"         actor_rollout_ref.model.trust_remote_code="$trust_remote_code"         actor_rollout_ref.rollout.name=vllm         actor_rollout_ref.rollout.n="$source_generation_n"         actor_rollout_ref.rollout.temperature="$source_temperature"         actor_rollout_ref.rollout.top_p="$source_top_p"         actor_rollout_ref.rollout.response_length="$max_response_length"         actor_rollout_ref.rollout.tensor_model_parallel_size="$gen_tp_size"         actor_rollout_ref.rollout.gpu_memory_utilization="$gen_gpu_memory_utilization"
fi

if [[ "$force_rl_preprocess" == "1" || ! -f "$data_dir/train_mixed.parquet" || ! -f "$data_dir/val_mixed.parquet" ]]; then
    preprocess_cmd=(
        python3 examples/data_preprocess/openmathreasoning_hero.py
        --local_dataset_path "$source_generated_path"
        --question_col "$question_col"
        --answer_col "$answer_col"
        --response_col responses
        --problem_type_col "$problem_type_col"
        --problem_type_value "$problem_type_value"
        --seed "$seed"
        --local_save_dir "$data_dir"
    )
    "${preprocess_cmd[@]}"
fi

if [[ "$filter_tbr" == "1" ]]; then
    if [[ "$force_tbr_filter" == "1" || ! -f "$filtered_tbr_path" ]]; then
        tbr_filter_cmd=(
            python3 examples/data_preprocess/hero_filter_textbook_reasoning.py
            --output_path "$filtered_tbr_path"
        )
        if [[ -n "$tbr_local_path" ]]; then
            tbr_filter_cmd+=(--local_dataset_path "$tbr_local_path")
        fi
        if [[ "${HERO_DISABLE_TBR_MATH_VERIFY_FILTER:-0}" == "1" ]]; then
            tbr_filter_cmd+=(--disable_math_verify_filter)
        fi
        if [[ -n "${HERO_TBR_ANSWER_MODEL:-}" ]]; then
            tbr_filter_cmd+=(--answer_model "${HERO_TBR_ANSWER_MODEL}")
        fi
        if [[ -n "${HERO_TBR_ANSWER_BASE_URL:-}" ]]; then
            tbr_filter_cmd+=(--answer_base_url "${HERO_TBR_ANSWER_BASE_URL}")
        fi
        if [[ -n "${HERO_TBR_ANSWER_API_KEY_ENV:-}" ]]; then
            tbr_filter_cmd+=(--answer_api_key_env "${HERO_TBR_ANSWER_API_KEY_ENV}")
        fi
        if [[ -n "${HERO_TBR_ANSWER_JUDGE_MODEL:-}" ]]; then
            tbr_filter_cmd+=(--answer_judge_model "${HERO_TBR_ANSWER_JUDGE_MODEL}")
        fi
        if [[ -n "${HERO_TBR_ANSWER_JUDGE_BASE_URL:-}" ]]; then
            tbr_filter_cmd+=(--answer_judge_base_url "${HERO_TBR_ANSWER_JUDGE_BASE_URL}")
        fi
        if [[ -n "${HERO_TBR_ANSWER_JUDGE_API_KEY_ENV:-}" ]]; then
            tbr_filter_cmd+=(--answer_judge_api_key_env "${HERO_TBR_ANSWER_JUDGE_API_KEY_ENV}")
        fi
        if [[ -n "${HERO_TBR_SUITABILITY_MODEL:-}" ]]; then
            tbr_filter_cmd+=(--suitability_model "${HERO_TBR_SUITABILITY_MODEL}")
        fi
        if [[ -n "${HERO_TBR_SUITABILITY_BASE_URL:-}" ]]; then
            tbr_filter_cmd+=(--suitability_base_url "${HERO_TBR_SUITABILITY_BASE_URL}")
        fi
        if [[ -n "${HERO_TBR_SUITABILITY_API_KEY_ENV:-}" ]]; then
            tbr_filter_cmd+=(--suitability_api_key_env "${HERO_TBR_SUITABILITY_API_KEY_ENV}")
        fi
        "${tbr_filter_cmd[@]}"
    fi
    tbr_input_for_eval="$filtered_tbr_path"
fi

need_eval_preprocess="$force_eval_preprocess"
if [[ "$need_eval_preprocess" != "1" ]]; then
    for bench in $eval_benchmarks; do
        if [[ ! -f "$eval_dir/${bench}.parquet" ]]; then
            need_eval_preprocess=1
            break
        fi
    done
fi

if [[ "$need_eval_preprocess" == "1" ]]; then
    eval_cmd=(python3 examples/data_preprocess/hero_eval_benchmarks.py --local_save_dir "$eval_dir" --benchmarks)
    for bench in $eval_benchmarks; do
        eval_cmd+=("$bench")
    done
    if [[ -n "$hvm_local_path" ]]; then
        eval_cmd+=(--hvm_local_path "$hvm_local_path")
    fi
    if [[ -n "$tbr_input_for_eval" ]]; then
        eval_cmd+=(--tbr_local_path "$tbr_input_for_eval")
    fi
    "${eval_cmd[@]}"
fi

train_model_path="$base_model_path"
if [[ "$enable_cold_start_sft" == "1" ]]; then
    HERO_SFT_INPUT_PATH="$source_generated_path"     HERO_SFT_DATA_DIR="$sft_data_dir"     HERO_SFT_OUTPUT_DIR="$sft_output_dir"     HERO_MODEL_PATH="$base_model_path"     HERO_TRUST_REMOTE_CODE="$trust_remote_code"     bash examples/hero/run_hero_cold_start_sft.sh
    train_model_path=$(resolve_latest_hf_dir "$sft_output_dir" "/huggingface")
fi

HERO_MODEL_PATH="$train_model_path" HERO_TRUST_REMOTE_CODE="$trust_remote_code" HERO_DATA_DIR="$data_dir" HERO_EVAL_DIR="$eval_dir" HERO_TRAIN_OUTPUT_DIR="$train_output_dir" bash examples/hero/run_hero_train.sh

if [[ "$run_final_eval" == "1" ]]; then
    latest_actor_hf_dir=$(resolve_latest_hf_dir "$train_output_dir" "/actor/huggingface")
    HERO_MODEL_PATH="$latest_actor_hf_dir"     HERO_TRUST_REMOTE_CODE="$trust_remote_code"     HERO_EVAL_DIR="$eval_dir"     HERO_OUTPUT_DIR="$eval_output_dir"     bash examples/hero/run_hero_eval.sh
fi

echo "HERO pipeline completed."
echo "Work directory: $work_dir"
echo "RL checkpoint root: $train_output_dir"
if [[ "$run_final_eval" == "1" ]]; then
    echo "Evaluation output dir: $eval_output_dir"
fi

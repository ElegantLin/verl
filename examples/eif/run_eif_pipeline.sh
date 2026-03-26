#!/usr/bin/env bash
# End-to-end EIF pipeline (mirrors HERO pipeline).
#
# Stages:
#   1. Build 40k prompt-only OpenMathReasoning source parquet
#   2. Generate candidate responses with the base model
#   3. Build RL train/val splits from generated responses
#   4. Optionally filter TextBookReasoning for paper-style hard-to-verify evaluation
#   5. Build evaluation benchmark parquet files
#   6. Optionally run cold-start SFT
#   7. Run EIF RL training (hybrid_eif_online)
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

run_name=${EIF_RUN_NAME:-$(date +%Y%m%d_%H%M%S)}
artifact_root=${EIF_ARTIFACT_ROOT:-$HOME/data/eif_reproduction}
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

dataset_name=${EIF_DATASET:-OpenMathReasoning}
dataset_config=${EIF_DATASET_CONFIG:-}
dataset_split=${EIF_DATASET_SPLIT:-train}
local_dataset_path=${EIF_LOCAL_DATASET_PATH:-}
question_col=${EIF_QUESTION_COL:-question}
answer_col=${EIF_ANSWER_COL:-answer}
problem_type_col=${EIF_PROBLEM_TYPE_COL:-problem_type}
problem_type_value=${EIF_PROBLEM_TYPE_VALUE:-has_answer_extracted}
source_sample_size=${EIF_SOURCE_SAMPLE_SIZE:-40000}
seed=${EIF_SEED:-42}

gen_nnodes=${EIF_GEN_NNODES:-1}
gen_gpus_per_node=${EIF_GEN_GPUS_PER_NODE:-8}
gen_tp_size=${EIF_GEN_TP_SIZE:-2}
gen_gpu_memory_utilization=${EIF_GEN_GPU_MEMORY_UTILIZATION:-0.85}
source_generation_n=${EIF_SOURCE_GENERATION_N:-1}
source_temperature=${EIF_SOURCE_TEMPERATURE:-1.0}
source_top_p=${EIF_SOURCE_TOP_P:-0.95}
max_prompt_length=${EIF_MAX_PROMPT_LENGTH:-1024}
max_response_length=${EIF_MAX_RESPONSE_LENGTH:-8192}

enable_cold_start_sft=${EIF_ENABLE_COLD_START_SFT:-1}
run_final_eval=${EIF_RUN_FINAL_EVAL:-1}
filter_tbr=${EIF_FILTER_TBR:-0}
force_source_build=${EIF_FORCE_SOURCE_BUILD:-0}
force_source_generation=${EIF_FORCE_SOURCE_GENERATION:-0}
force_rl_preprocess=${EIF_FORCE_RL_PREPROCESS:-0}
force_eval_preprocess=${EIF_FORCE_EVAL_PREPROCESS:-0}
force_tbr_filter=${EIF_FORCE_TBR_FILTER:-0}

eval_benchmarks=${EIF_EVAL_BENCHMARKS_BUILD:-"math500 amc minerva olympiad hardverify_math textbook_reasoning"}
hvm_local_path=${EIF_HVM_LOCAL_PATH:-}
tbr_local_path=${EIF_TBR_LOCAL_PATH:-}
tbr_input_for_eval="$tbr_local_path"

mkdir -p "$work_dir" "$source_dir" "$data_dir" "$eval_dir" "$sft_data_dir" "$sft_output_dir" "$train_output_dir" "$eval_output_dir"

# ═══════════════════════════════════════════════════════════════════════
# Stage 1: Build source prompts
# ═══════════════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════════════
# Stage 2: Generate candidate responses
# ═══════════════════════════════════════════════════════════════════════
if [[ "$force_source_generation" == "1" || ! -f "$source_generated_path" ]]; then
    python3 -m verl.trainer.main_generation_server \
        trainer.nnodes="$gen_nnodes" \
        trainer.n_gpus_per_node="$gen_gpus_per_node" \
        data.train_files="['$source_prompts_path']" \
        data.prompt_key=prompt \
        data.output_path="$source_generated_path" \
        actor_rollout_ref.model.path="$base_model_path" \
        actor_rollout_ref.model.trust_remote_code="$trust_remote_code" \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.n="$source_generation_n" \
        actor_rollout_ref.rollout.temperature="$source_temperature" \
        actor_rollout_ref.rollout.top_p="$source_top_p" \
        actor_rollout_ref.rollout.response_length="$max_response_length" \
        actor_rollout_ref.rollout.tensor_model_parallel_size="$gen_tp_size" \
        actor_rollout_ref.rollout.gpu_memory_utilization="$gen_gpu_memory_utilization"
fi

# ═══════════════════════════════════════════════════════════════════════
# Stage 3: Build RL train/val splits
# ═══════════════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════════════
# Stage 4: Optional TBR filtering
# ═══════════════════════════════════════════════════════════════════════
if [[ "$filter_tbr" == "1" ]]; then
    if [[ "$force_tbr_filter" == "1" || ! -f "$filtered_tbr_path" ]]; then
        tbr_filter_cmd=(
            python3 examples/data_preprocess/hero_filter_textbook_reasoning.py
            --output_path "$filtered_tbr_path"
        )
        if [[ -n "$tbr_local_path" ]]; then
            tbr_filter_cmd+=(--local_dataset_path "$tbr_local_path")
        fi
        if [[ "${EIF_DISABLE_TBR_MATH_VERIFY_FILTER:-0}" == "1" ]]; then
            tbr_filter_cmd+=(--disable_math_verify_filter)
        fi
        if [[ -n "${EIF_TBR_ANSWER_MODEL:-}" ]]; then
            tbr_filter_cmd+=(--answer_model "${EIF_TBR_ANSWER_MODEL}")
        fi
        if [[ -n "${EIF_TBR_ANSWER_BASE_URL:-}" ]]; then
            tbr_filter_cmd+=(--answer_base_url "${EIF_TBR_ANSWER_BASE_URL}")
        fi
        if [[ -n "${EIF_TBR_ANSWER_API_KEY_ENV:-}" ]]; then
            tbr_filter_cmd+=(--answer_api_key_env "${EIF_TBR_ANSWER_API_KEY_ENV}")
        fi
        if [[ -n "${EIF_TBR_ANSWER_JUDGE_MODEL:-}" ]]; then
            tbr_filter_cmd+=(--answer_judge_model "${EIF_TBR_ANSWER_JUDGE_MODEL}")
        fi
        if [[ -n "${EIF_TBR_ANSWER_JUDGE_BASE_URL:-}" ]]; then
            tbr_filter_cmd+=(--answer_judge_base_url "${EIF_TBR_ANSWER_JUDGE_BASE_URL}")
        fi
        if [[ -n "${EIF_TBR_ANSWER_JUDGE_API_KEY_ENV:-}" ]]; then
            tbr_filter_cmd+=(--answer_judge_api_key_env "${EIF_TBR_ANSWER_JUDGE_API_KEY_ENV}")
        fi
        if [[ -n "${EIF_TBR_SUITABILITY_MODEL:-}" ]]; then
            tbr_filter_cmd+=(--suitability_model "${EIF_TBR_SUITABILITY_MODEL}")
        fi
        if [[ -n "${EIF_TBR_SUITABILITY_BASE_URL:-}" ]]; then
            tbr_filter_cmd+=(--suitability_base_url "${EIF_TBR_SUITABILITY_BASE_URL}")
        fi
        if [[ -n "${EIF_TBR_SUITABILITY_API_KEY_ENV:-}" ]]; then
            tbr_filter_cmd+=(--suitability_api_key_env "${EIF_TBR_SUITABILITY_API_KEY_ENV}")
        fi
        "${tbr_filter_cmd[@]}"
    fi
    tbr_input_for_eval="$filtered_tbr_path"
fi

# ═══════════════════════════════════════════════════════════════════════
# Stage 5: Build eval benchmarks
# ═══════════════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════════════
# Stage 6: Optional cold-start SFT
# ═══════════════════════════════════════════════════════════════════════
train_model_path="$base_model_path"
if [[ "$enable_cold_start_sft" == "1" ]]; then
    EIF_SFT_INPUT_PATH="$source_generated_path" \
    EIF_SFT_DATA_DIR="$sft_data_dir" \
    EIF_SFT_OUTPUT_DIR="$sft_output_dir" \
    EIF_MODEL_PATH="$base_model_path" \
    EIF_TRUST_REMOTE_CODE="$trust_remote_code" \
    bash examples/eif/run_eif_cold_start_sft.sh
    train_model_path=$(resolve_latest_hf_dir "$sft_output_dir" "/huggingface")
fi

# ═══════════════════════════════════════════════════════════════════════
# Stage 7: EIF RL training
# ═══════════════════════════════════════════════════════════════════════
EIF_MODEL_PATH="$train_model_path" \
EIF_TRUST_REMOTE_CODE="$trust_remote_code" \
EIF_DATA_DIR="$data_dir" \
EIF_EVAL_DIR="$eval_dir" \
EIF_TRAIN_OUTPUT_DIR="$train_output_dir" \
bash examples/eif/run_eif_train.sh

# ═══════════════════════════════════════════════════════════════════════
# Stage 8: Final evaluation
# ═══════════════════════════════════════════════════════════════════════
if [[ "$run_final_eval" == "1" ]]; then
    latest_actor_hf_dir=$(resolve_latest_hf_dir "$train_output_dir" "/actor/huggingface")
    EIF_MODEL_PATH="$latest_actor_hf_dir" \
    EIF_TRUST_REMOTE_CODE="$trust_remote_code" \
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

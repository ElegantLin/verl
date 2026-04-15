#!/usr/bin/env bash
# Shared stage functions for the unified hybrid_reward CLI.

if [[ -n "${_HYBRID_REWARD_STAGE_LIB_SH:-}" ]]; then
    return 0
fi
_HYBRID_REWARD_STAGE_LIB_SH=1

stage_lib_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

die() {
    echo "$*" >&2
    exit 1
}

require_option_value() {
    local option_name="$1"
    local option_value="${2:-}"
    if [[ -z "$option_value" ]]; then
        die "Missing value for $option_name"
    fi
}

set_env_default() {
    local var_name="$1"
    local default_value="$2"
    if [[ -z "${!var_name:-}" ]]; then
        export "$var_name=$default_value"
    fi
}

validate_algorithm() {
    local algorithm="$1"
    case "$algorithm" in
        hero|eif) ;;
        *)
            die "Unknown --algorithm=$algorithm. Use hero or eif."
            ;;
    esac
}

reload_stage_common() {
    unset _SHARED_STEP_COMMON_SH
    # shellcheck source=examples/hybrid_reward/step_by_step/common.sh
    source "$stage_lib_dir/step_by_step/common.sh"
}

apply_cli_options() {
    local command="$1"
    shift

    local debug=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --algorithm)
                require_option_value "$1" "${2:-}"
                validate_algorithm "$2"
                export ALGORITHM="$2"
                shift 2
                ;;
            --debug)
                debug=1
                shift
                ;;
            --run-name)
                require_option_value "$1" "${2:-}"
                export RL_PIPELINE_RUN_NAME="$2"
                shift 2
                ;;
            --artifact-root)
                require_option_value "$1" "${2:-}"
                export RL_PIPELINE_ARTIFACT_ROOT="$2"
                shift 2
                ;;
            --work-dir)
                require_option_value "$1" "${2:-}"
                export RL_PIPELINE_WORK_DIR="$2"
                shift 2
                ;;
            --model-path)
                require_option_value "$1" "${2:-}"
                export RL_PIPELINE_MODEL_PATH="$2"
                shift 2
                ;;
            --data-dir)
                require_option_value "$1" "${2:-}"
                export RL_PIPELINE_DATA_DIR="$2"
                shift 2
                ;;
            --eval-dir)
                require_option_value "$1" "${2:-}"
                export RL_PIPELINE_EVAL_DIR="$2"
                shift 2
                ;;
            --train-output-dir)
                require_option_value "$1" "${2:-}"
                export RL_PIPELINE_TRAIN_OUTPUT_DIR="$2"
                shift 2
                ;;
            --eval-output-dir)
                require_option_value "$1" "${2:-}"
                export RL_PIPELINE_EVAL_OUTPUT_DIR="$2"
                shift 2
                ;;
            --gpu-profile)
                require_option_value "$1" "${2:-}"
                export RL_PIPELINE_GPU_PROFILE="$2"
                shift 2
                ;;
            --regime)
                require_option_value "$1" "${2:-}"
                export REGIME="$2"
                shift 2
                ;;
            --filter-tbr)
                export RL_PIPELINE_FILTER_TBR=1
                shift
                ;;
            --no-sft)
                export RL_PIPELINE_ENABLE_COLD_START_SFT=0
                shift
                ;;
            --help|-h)
                shift
                ;;
            *)
                die "Unknown option for $command: $1"
                ;;
        esac
    done

    case "$command" in
        data-train)
            if [[ "$debug" == "1" ]]; then
                set_env_default RL_PIPELINE_SOURCE_SAMPLE_SIZE 1000
                set_env_default RL_PIPELINE_MAX_RESPONSE_LENGTH 1024
            fi
            ;;
        data-eval)
            if [[ "$debug" == "1" ]]; then
                set_env_default RL_PIPELINE_EVAL_BENCHMARKS_BUILD amc
            fi
            ;;
        *)
            if [[ "$debug" == "1" ]]; then
                die "--debug is only supported for data-train and data-eval"
            fi
            ;;
    esac

    case "$command" in
        rl|pipeline)
            if [[ -z "${ALGORITHM:-}" ]]; then
                die "$command requires --algorithm"
            fi
            ;;
        eval)
            if [[ -z "${RL_PIPELINE_MODEL_PATH:-}" && -z "${ALGORITHM:-}" ]]; then
                die "eval requires --algorithm or --model-path"
            fi
            ;;
    esac

    reload_stage_common
}

run_stage_filter_tbr() {
    local force_tbr_filter=${RL_PIPELINE_FORCE_TBR_FILTER:-0}

    if [[ "${RL_PIPELINE_FILTER_TBR:-0}" != "1" ]]; then
        return 0
    fi

    if [[ "$force_tbr_filter" == "1" || ! -f "$filtered_tbr_path" ]]; then
        local tbr_filter_cmd=(
            python3 examples/data_preprocess/hero_filter_textbook_reasoning.py
            --output_path "$filtered_tbr_path"
        )

        if [[ -n "$tbr_local_path" ]]; then
            tbr_filter_cmd+=(--local_dataset_path "$tbr_local_path")
        fi
        if [[ "${RL_PIPELINE_DISABLE_TBR_MATH_VERIFY_FILTER:-0}" == "1" ]]; then
            tbr_filter_cmd+=(--disable_math_verify_filter)
        fi
        if [[ -n "${RL_PIPELINE_TBR_ANSWER_MODEL:-}" ]]; then
            tbr_filter_cmd+=(--answer_model "${RL_PIPELINE_TBR_ANSWER_MODEL}")
        fi
        if [[ -n "${RL_PIPELINE_TBR_ANSWER_BASE_URL:-}" ]]; then
            tbr_filter_cmd+=(--answer_base_url "${RL_PIPELINE_TBR_ANSWER_BASE_URL}")
        fi
        if [[ -n "${RL_PIPELINE_TBR_ANSWER_API_KEY_ENV:-}" ]]; then
            tbr_filter_cmd+=(--answer_api_key_env "${RL_PIPELINE_TBR_ANSWER_API_KEY_ENV}")
        fi
        if [[ -n "${RL_PIPELINE_TBR_ANSWER_JUDGE_MODEL:-}" ]]; then
            tbr_filter_cmd+=(--answer_judge_model "${RL_PIPELINE_TBR_ANSWER_JUDGE_MODEL}")
        fi
        if [[ -n "${RL_PIPELINE_TBR_ANSWER_JUDGE_BASE_URL:-}" ]]; then
            tbr_filter_cmd+=(--answer_judge_base_url "${RL_PIPELINE_TBR_ANSWER_JUDGE_BASE_URL}")
        fi
        if [[ -n "${RL_PIPELINE_TBR_ANSWER_JUDGE_API_KEY_ENV:-}" ]]; then
            tbr_filter_cmd+=(--answer_judge_api_key_env "${RL_PIPELINE_TBR_ANSWER_JUDGE_API_KEY_ENV}")
        fi
        if [[ -n "${RL_PIPELINE_TBR_SUITABILITY_MODEL:-}" ]]; then
            tbr_filter_cmd+=(--suitability_model "${RL_PIPELINE_TBR_SUITABILITY_MODEL}")
        fi
        if [[ -n "${RL_PIPELINE_TBR_SUITABILITY_BASE_URL:-}" ]]; then
            tbr_filter_cmd+=(--suitability_base_url "${RL_PIPELINE_TBR_SUITABILITY_BASE_URL}")
        fi
        if [[ -n "${RL_PIPELINE_TBR_SUITABILITY_API_KEY_ENV:-}" ]]; then
            tbr_filter_cmd+=(--suitability_api_key_env "${RL_PIPELINE_TBR_SUITABILITY_API_KEY_ENV}")
        fi

        "${tbr_filter_cmd[@]}"
    fi

    echo "Filtered TBR parquet: $filtered_tbr_path"
}

run_stage_data_train() {
    apply_cli_options data-train "$@"

    cd "$repo_root"
    ensure_dirs

    if [[ "${RL_PIPELINE_FORCE_SOURCE_BUILD:-0}" == "1" || ! -f "$source_prompts_path" ]]; then
        local source_cmd=(
            python3 examples/data_preprocess/openmathreasoning_hero_source.py
            --split "$dataset_split"
            --question_col "$source_question_col"
            --answer_col "$source_answer_col"
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
            if [[ "$source_dataset_trust_remote_code" == "True" ]]; then
                source_cmd+=(--trust_remote_code)
            fi
        fi

        "${source_cmd[@]}"
    fi

    require_file "$source_prompts_path"

    if [[ "${RL_PIPELINE_FORCE_SOURCE_GENERATION:-0}" == "1" || ! -f "$source_generated_path" ]]; then
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

    require_file "$source_generated_path"

    if [[ "${RL_PIPELINE_FORCE_RL_PREPROCESS:-0}" == "1" || ! -f "$data_dir/train_mixed.parquet" || ! -f "$data_dir/val_mixed.parquet" ]]; then
        python3 examples/data_preprocess/openmathreasoning_hero.py \
            --local_dataset_path "$source_generated_path" \
            --question_col "$question_col" \
            --answer_col "$answer_col" \
            --response_col responses \
            --problem_type_col "$problem_type_col" \
            --problem_type_value "$problem_type_value" \
            --seed "$seed" \
            --local_save_dir "$data_dir"
    fi

    export RL_PIPELINE_DATA_DIR="$data_dir"
    echo "RL data directory: $data_dir"
}

run_stage_data_eval() {
    apply_cli_options data-eval "$@"

    cd "$repo_root"
    ensure_dirs

    run_stage_filter_tbr

    local tbr_input_for_eval=""
    if [[ -f "$filtered_tbr_path" ]]; then
        tbr_input_for_eval="$filtered_tbr_path"
    elif [[ -n "$tbr_local_path" ]]; then
        tbr_input_for_eval="$tbr_local_path"
    fi

    local need_eval_preprocess=${RL_PIPELINE_FORCE_EVAL_PREPROCESS:-0}
    if [[ "$need_eval_preprocess" != "1" ]]; then
        for bench in $eval_benchmarks; do
            if [[ ! -f "$eval_dir/${bench}.parquet" ]]; then
                need_eval_preprocess=1
                break
            fi
        done
    fi

    if [[ "$need_eval_preprocess" == "1" ]]; then
        local eval_cmd=(python3 examples/data_preprocess/hero_eval_benchmarks.py --local_save_dir "$eval_dir" --benchmarks)
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

    export RL_PIPELINE_EVAL_DIR="$eval_dir"
    echo "Eval benchmark directory: $eval_dir"
}

run_stage_sft() {
    apply_cli_options sft "$@"

    cd "$repo_root"
    ensure_dirs
    require_file "$source_generated_path"

    local sft_model_path=${RL_PIPELINE_MODEL_PATH:-$base_model_path}

    RL_PIPELINE_SFT_INPUT_PATH="$source_generated_path" \
    RL_PIPELINE_SFT_DATA_DIR="$sft_data_dir" \
    RL_PIPELINE_SFT_OUTPUT_DIR="$sft_output_dir" \
    RL_PIPELINE_MODEL_PATH="$sft_model_path" \
    RL_PIPELINE_TRUST_REMOTE_CODE="$trust_remote_code" \
    bash examples/hybrid_reward/run_cold_start_sft.sh

    if latest_sft_hf_dir=$(resolve_latest_hf_dir "$sft_output_dir" "/huggingface"); then
        export RL_PIPELINE_TRAIN_MODEL_PATH="$latest_sft_hf_dir"
        echo "Latest cold-start HF model: $latest_sft_hf_dir"
    else
        export RL_PIPELINE_TRAIN_MODEL_PATH="$base_model_path"
    fi
}

run_stage_rl() {
    apply_cli_options rl "$@"

    cd "$repo_root"
    ensure_dirs
    require_file "$data_dir/train_mixed.parquet"
    require_file "$data_dir/val_mixed.parquet"

    local train_model_path=${RL_PIPELINE_MODEL_PATH:-}
    if [[ -z "$train_model_path" ]]; then
        if latest_sft_hf_dir=$(resolve_latest_hf_dir "$sft_output_dir" "/huggingface"); then
            train_model_path="$latest_sft_hf_dir"
        else
            train_model_path="$base_model_path"
        fi
    fi

    ALGORITHM="$ALGORITHM" \
    MODEL_PATH="$train_model_path" \
    TRUST_REMOTE_CODE="$trust_remote_code" \
    DATA_DIR="$data_dir" \
    EVAL_DIR="$eval_dir" \
    TRAIN_OUTPUT_DIR="$train_output_dir" \
    bash examples/hybrid_reward/run_train.sh

    if latest_actor_hf_dir=$(resolve_latest_hf_dir "$train_output_dir" "/actor/huggingface"); then
        echo "Latest trained actor HF checkpoint: $latest_actor_hf_dir"
    fi
}

run_stage_eval() {
    apply_cli_options eval "$@"

    cd "$repo_root"
    ensure_dirs

    local eval_model_path=${RL_PIPELINE_MODEL_PATH:-}
    if [[ -z "$eval_model_path" ]]; then
        if latest_actor_hf_dir=$(resolve_latest_hf_dir "$train_output_dir" "/actor/huggingface"); then
            eval_model_path="$latest_actor_hf_dir"
        else
            echo "Could not find a trained actor HF checkpoint under: $train_output_dir" >&2
            echo "Set --model-path explicitly, or run rl with --algorithm first." >&2
            exit 1
        fi
    fi

    RL_PIPELINE_MODEL_PATH="$eval_model_path" \
    RL_PIPELINE_TRUST_REMOTE_CODE="$trust_remote_code" \
    RL_PIPELINE_EVAL_DIR="$eval_dir" \
    RL_PIPELINE_EVAL_OUTPUT_DIR="$eval_output_dir" \
    bash examples/hybrid_reward/run_eval.sh

    echo "Evaluation output directory: $eval_output_dir"
}

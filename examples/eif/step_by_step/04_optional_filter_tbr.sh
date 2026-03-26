#!/usr/bin/env bash
set -euo pipefail
set -x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=examples/eif/step_by_step/common.sh
source "$script_dir/common.sh"

cd "$repo_root"
ensure_eif_dirs

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

"${tbr_filter_cmd[@]}" "$@"

echo "Filtered TBR parquet: $filtered_tbr_path"

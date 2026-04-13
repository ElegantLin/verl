#!/usr/bin/env bash
set -euo pipefail
set -x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=examples/eif/step_by_step/common.sh
source "$script_dir/common.sh"

# Map EIF_* TBR vars to RL_PIPELINE_*
for v in DISABLE_TBR_MATH_VERIFY_FILTER TBR_ANSWER_MODEL TBR_ANSWER_BASE_URL TBR_ANSWER_API_KEY_ENV \
         TBR_ANSWER_JUDGE_MODEL TBR_ANSWER_JUDGE_BASE_URL TBR_ANSWER_JUDGE_API_KEY_ENV \
         TBR_SUITABILITY_MODEL TBR_SUITABILITY_BASE_URL TBR_SUITABILITY_API_KEY_ENV; do
    eif_var="EIF_${v}"
    pipeline_var="RL_PIPELINE_${v}"
    if [[ -n "${!eif_var:-}" ]]; then
        export "$pipeline_var=${!eif_var}"
    fi
done

bash "$repo_root/examples/shared/step_by_step/04_optional_filter_tbr.sh" "$@"

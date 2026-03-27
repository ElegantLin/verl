#!/usr/bin/env bash
set -euo pipefail
set -x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=examples/hero/step_by_step/common.sh
source "$script_dir/common.sh"

cd "$repo_root"
ensure_hero_dirs

source_cmd=(
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

"${source_cmd[@]}" "$@"

echo "Source prompts saved to: $source_prompts_path"

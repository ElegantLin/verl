#!/usr/bin/env bash
set -euo pipefail
set -x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=examples/eif/step_by_step/common.sh
source "$script_dir/common.sh"

cd "$repo_root"
ensure_eif_dirs
require_file "$source_generated_path"

python3 examples/data_preprocess/openmathreasoning_hero.py \
    --local_dataset_path "$source_generated_path" \
    --question_col "$question_col" \
    --answer_col "$answer_col" \
    --response_col responses \
    --problem_type_col "$problem_type_col" \
    --problem_type_value "$problem_type_value" \
    --seed "$seed" \
    --local_save_dir "$data_dir" \
    "$@"

echo "RL data directory: $data_dir"

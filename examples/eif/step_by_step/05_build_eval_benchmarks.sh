#!/usr/bin/env bash
set -euo pipefail
set -x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=examples/eif/step_by_step/common.sh
source "$script_dir/common.sh"

cd "$repo_root"
ensure_eif_dirs

tbr_input_for_eval=""
if [[ -f "$filtered_tbr_path" ]]; then
    tbr_input_for_eval="$filtered_tbr_path"
elif [[ -n "$tbr_local_path" ]]; then
    tbr_input_for_eval="$tbr_local_path"
fi

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

"${eval_cmd[@]}" "$@"

echo "Eval benchmark directory: $eval_dir"

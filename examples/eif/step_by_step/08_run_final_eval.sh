#!/usr/bin/env bash
set -euo pipefail
set -x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=examples/eif/step_by_step/common.sh
source "$script_dir/common.sh"

cd "$repo_root"
ensure_eif_dirs

eval_model_path=${EIF_MODEL_PATH:-}
if [[ -z "$eval_model_path" ]]; then
    if latest_actor_hf_dir=$(resolve_latest_hf_dir "$train_output_dir" "/actor/huggingface"); then
        eval_model_path="$latest_actor_hf_dir"
    else
        echo "Could not find a trained actor HF checkpoint under: $train_output_dir" >&2
        echo "Set EIF_MODEL_PATH explicitly, or run 07_run_rl_train.sh first." >&2
        exit 1
    fi
fi

EIF_MODEL_PATH="$eval_model_path" \
EIF_TRUST_REMOTE_CODE="$trust_remote_code" \
EIF_EVAL_DIR="$eval_dir" \
EIF_OUTPUT_DIR="$eval_output_dir" \
bash examples/eif/run_eif_eval.sh "$@"

echo "Evaluation output directory: $eval_output_dir"

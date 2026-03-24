#!/usr/bin/env bash
set -euo pipefail
set -x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=examples/hero/step_by_step/common.sh
source "$script_dir/common.sh"

cd "$repo_root"
ensure_hero_dirs

eval_model_path=${HERO_MODEL_PATH:-}
if [[ -z "$eval_model_path" ]]; then
    if latest_actor_hf_dir=$(resolve_latest_hf_dir "$train_output_dir" "/actor/huggingface"); then
        eval_model_path="$latest_actor_hf_dir"
    else
        echo "Could not find a trained actor HF checkpoint under: $train_output_dir" >&2
        echo "Set HERO_MODEL_PATH explicitly, or run 07_run_rl_train.sh first." >&2
        exit 1
    fi
fi

HERO_MODEL_PATH="$eval_model_path" \
HERO_TRUST_REMOTE_CODE="$trust_remote_code" \
HERO_EVAL_DIR="$eval_dir" \
HERO_OUTPUT_DIR="$eval_output_dir" \
bash examples/hero/run_hero_eval.sh "$@"

echo "Evaluation output directory: $eval_output_dir"

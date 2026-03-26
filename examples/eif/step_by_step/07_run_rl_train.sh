#!/usr/bin/env bash
set -euo pipefail
set -x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=examples/eif/step_by_step/common.sh
source "$script_dir/common.sh"

cd "$repo_root"
ensure_eif_dirs
require_file "$data_dir/train_mixed.parquet"
require_file "$data_dir/val_mixed.parquet"

train_model_path=${EIF_MODEL_PATH:-}
if [[ -z "$train_model_path" ]]; then
    if latest_sft_hf_dir=$(resolve_latest_hf_dir "$sft_output_dir" "/huggingface"); then
        train_model_path="$latest_sft_hf_dir"
    else
        train_model_path="$base_model_path"
    fi
fi

EIF_MODEL_PATH="$train_model_path" \
EIF_TRUST_REMOTE_CODE="$trust_remote_code" \
EIF_DATA_DIR="$data_dir" \
EIF_EVAL_DIR="$eval_dir" \
EIF_TRAIN_OUTPUT_DIR="$train_output_dir" \
bash examples/eif/run_eif_train.sh "$@"

if latest_actor_hf_dir=$(resolve_latest_hf_dir "$train_output_dir" "/actor/huggingface"); then
    echo "Latest trained actor HF checkpoint: $latest_actor_hf_dir"
fi

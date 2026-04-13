#!/usr/bin/env bash
set -euo pipefail
set -x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/common.sh"

cd "$repo_root"
ensure_dirs
require_file "$data_dir/train_mixed.parquet"
require_file "$data_dir/val_mixed.parquet"

train_model_path=${RL_PIPELINE_MODEL_PATH:-}
if [[ -z "$train_model_path" ]]; then
    if latest_sft_hf_dir=$(resolve_latest_hf_dir "$sft_output_dir" "/huggingface"); then
        train_model_path="$latest_sft_hf_dir"
    else
        train_model_path="$base_model_path"
    fi
fi

ALGORITHM="${ALGORITHM:-hero}" \
MODEL_PATH="$train_model_path" \
TRUST_REMOTE_CODE="$trust_remote_code" \
DATA_DIR="$data_dir" \
EVAL_DIR="$eval_dir" \
TRAIN_OUTPUT_DIR="$train_output_dir" \
bash examples/hybrid_reward/run_train.sh "$@"

if latest_actor_hf_dir=$(resolve_latest_hf_dir "$train_output_dir" "/actor/huggingface"); then
    echo "Latest trained actor HF checkpoint: $latest_actor_hf_dir"
fi

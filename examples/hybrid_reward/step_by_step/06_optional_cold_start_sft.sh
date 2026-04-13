#!/usr/bin/env bash
set -euo pipefail
set -x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=examples/hybrid_reward/step_by_step/common.sh
source "$script_dir/common.sh"

cd "$repo_root"
ensure_dirs
require_file "$source_generated_path"

sft_model_path=${RL_PIPELINE_MODEL_PATH:-$base_model_path}

RL_PIPELINE_SFT_INPUT_PATH="$source_generated_path" \
RL_PIPELINE_SFT_DATA_DIR="$sft_data_dir" \
RL_PIPELINE_SFT_OUTPUT_DIR="$sft_output_dir" \
RL_PIPELINE_MODEL_PATH="$sft_model_path" \
RL_PIPELINE_TRUST_REMOTE_CODE="$trust_remote_code" \
bash examples/hybrid_reward/run_cold_start_sft.sh "$@"

if latest_sft_hf_dir=$(resolve_latest_hf_dir "$sft_output_dir" "/huggingface"); then
    echo "Latest cold-start HF model: $latest_sft_hf_dir"
fi

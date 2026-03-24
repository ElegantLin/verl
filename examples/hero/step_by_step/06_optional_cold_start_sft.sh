#!/usr/bin/env bash
set -euo pipefail
set -x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=examples/hero/step_by_step/common.sh
source "$script_dir/common.sh"

cd "$repo_root"
ensure_hero_dirs
require_file "$source_generated_path"

sft_model_path=${HERO_MODEL_PATH:-$base_model_path}

HERO_SFT_INPUT_PATH="$source_generated_path" \
HERO_SFT_DATA_DIR="$sft_data_dir" \
HERO_SFT_OUTPUT_DIR="$sft_output_dir" \
HERO_MODEL_PATH="$sft_model_path" \
HERO_TRUST_REMOTE_CODE="$trust_remote_code" \
bash examples/hero/run_hero_cold_start_sft.sh "$@"

if latest_sft_hf_dir=$(resolve_latest_hf_dir "$sft_output_dir" "/huggingface"); then
    echo "Latest cold-start HF model: $latest_sft_hf_dir"
fi

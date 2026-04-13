#!/usr/bin/env bash
set -euo pipefail
set -x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=examples/hero/step_by_step/common.sh
source "$script_dir/common.sh"

# Map remaining HERO SFT vars for the shared cold-start script.
export RL_PIPELINE_MODEL_PATH="${HERO_MODEL_PATH:-$base_model_path}"

bash "$repo_root/examples/shared/step_by_step/06_optional_cold_start_sft.sh" "$@"

#!/usr/bin/env bash
set -euo pipefail
set -x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=examples/eif/step_by_step/common.sh
source "$script_dir/common.sh"

# Map remaining EIF SFT vars for the shared cold-start script.
export RL_PIPELINE_MODEL_PATH="${EIF_MODEL_PATH:-$base_model_path}"

bash "$repo_root/examples/shared/step_by_step/06_optional_cold_start_sft.sh" "$@"

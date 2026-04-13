#!/usr/bin/env bash
# EIF step-by-step common config — thin wrapper around shared common.sh.
#
# Maps EIF_* environment variables to the generic RL_PIPELINE_* prefix
# used by examples/shared/step_by_step/common.sh, then sources it.
# After sourcing, all shared variables (repo_root, data_dir, seed, etc.)
# are available, along with helper functions (set_default, resolve_latest_hf_dir,
# require_file, ensure_dirs).

if [[ -n "${_EIF_STEP_COMMON_SH:-}" ]]; then
    return 0
fi
_EIF_STEP_COMMON_SH=1

eif_step_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Map EIF_* → RL_PIPELINE_* using the shared parametric mapper.
_ALGO_PREFIX=EIF
source "$eif_step_dir/../../../examples/shared/step_by_step/map_algo_vars.sh"

# EIF-specific artifact root default.
: "${RL_PIPELINE_ARTIFACT_ROOT:=$HOME/data/eif_reproduction}"
export RL_PIPELINE_ARTIFACT_ROOT

# Source shared common.sh (provides all variables and helpers).
source "$eif_step_dir/../../../examples/shared/step_by_step/common.sh"

# Backward-compatible alias.
ensure_eif_dirs() { ensure_dirs; }

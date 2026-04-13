#!/usr/bin/env bash
# HERO step-by-step common config — thin wrapper around shared common.sh.
#
# Maps HERO_* environment variables to the generic RL_PIPELINE_* prefix
# used by examples/shared/step_by_step/common.sh, then sources it.
# After sourcing, all shared variables (repo_root, data_dir, seed, etc.)
# are available, along with helper functions (set_default, resolve_latest_hf_dir,
# require_file, ensure_dirs).

if [[ -n "${_HERO_STEP_COMMON_SH:-}" ]]; then
    return 0
fi
_HERO_STEP_COMMON_SH=1

hero_step_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Map HERO_* → RL_PIPELINE_* using the shared parametric mapper.
_ALGO_PREFIX=HERO
source "$hero_step_dir/../../../examples/shared/step_by_step/map_algo_vars.sh"

# HERO-specific artifact root default.
: "${RL_PIPELINE_ARTIFACT_ROOT:=$HOME/data/hero_paper_reproduction}"
export RL_PIPELINE_ARTIFACT_ROOT

# Source shared common.sh (provides all variables and helpers).
source "$hero_step_dir/../../../examples/shared/step_by_step/common.sh"

# Backward-compatible alias.
ensure_hero_dirs() { ensure_dirs; }

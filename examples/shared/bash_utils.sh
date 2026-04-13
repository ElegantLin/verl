#!/usr/bin/env bash
# Shared bash utilities for HERO/EIF pipeline scripts.
#
# Source this file from any pipeline or step-by-step script that needs
# resolve_latest_hf_dir or other shared helpers.

resolve_latest_hf_dir() {
    local root_dir="$1"
    local suffix="$2"
    local tracker_file="$root_dir/latest_checkpointed_iteration.txt"
    if [[ ! -f "$tracker_file" ]]; then
        echo "Missing checkpoint tracker: $tracker_file" >&2
        return 1
    fi
    local latest_step
    latest_step=$(<"$tracker_file")
    local resolved_path="$root_dir/global_step_${latest_step}${suffix}"
    if [[ ! -d "$resolved_path" ]]; then
        echo "Missing checkpoint directory: $resolved_path" >&2
        return 1
    fi
    printf '%s\n' "$resolved_path"
}

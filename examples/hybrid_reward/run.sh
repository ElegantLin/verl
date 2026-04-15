#!/usr/bin/env bash
# Canonical unified CLI for hybrid_reward workflows.

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

usage() {
    cat <<'EOF'
Usage: bash examples/hybrid_reward/run.sh <command> [options]

Commands:
  data-train
  data-eval
  sft
  rl
  eval
  pipeline

Options:
  --algorithm hero|eif
  --debug
  --run-name <name>
  --artifact-root <path>
  --work-dir <path>
  --model-path <path>
  --data-dir <path>
  --eval-dir <path>
  --train-output-dir <path>
  --eval-output-dir <path>
  --gpu-profile <profile>
  --regime <name>
  --filter-tbr
  --no-sft
  --help
EOF
}

wants_help() {
    local arg
    for arg in "$@"; do
        if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
            return 0
        fi
    done
    return 1
}

cmd_data_train() {
    # shellcheck source=examples/hybrid_reward/stage_lib.sh
    source "$script_dir/stage_lib.sh"
    run_stage_data_train "$@"
}

cmd_data_eval() {
    # shellcheck source=examples/hybrid_reward/stage_lib.sh
    source "$script_dir/stage_lib.sh"
    run_stage_data_eval "$@"
}

cmd_sft() {
    # shellcheck source=examples/hybrid_reward/stage_lib.sh
    source "$script_dir/stage_lib.sh"
    run_stage_sft "$@"
}

cmd_rl() {
    # shellcheck source=examples/hybrid_reward/stage_lib.sh
    source "$script_dir/stage_lib.sh"
    run_stage_rl "$@"
}

cmd_eval() {
    # shellcheck source=examples/hybrid_reward/stage_lib.sh
    source "$script_dir/stage_lib.sh"
    run_stage_eval "$@"
}

cmd_pipeline() {
    # shellcheck source=examples/hybrid_reward/stage_lib.sh
    source "$script_dir/stage_lib.sh"
    apply_cli_options pipeline "$@"

    run_stage_data_train
    run_stage_data_eval
    if [[ "${RL_PIPELINE_ENABLE_COLD_START_SFT:-1}" != "0" ]]; then
        run_stage_sft
    fi
    run_stage_rl
    unset RL_PIPELINE_MODEL_PATH
    run_stage_eval
}

main() {
    local command="${1:-}"
    if [[ -z "$command" || "$command" == "help" || "$command" == "--help" || "$command" == "-h" ]]; then
        usage
        return 0
    fi
    shift

    if wants_help "$@"; then
        usage
        return 0
    fi

    case "$command" in
        data-train)
            cmd_data_train "$@"
            ;;
        data-eval)
            cmd_data_eval "$@"
            ;;
        sft)
            cmd_sft "$@"
            ;;
        rl)
            cmd_rl "$@"
            ;;
        eval)
            cmd_eval "$@"
            ;;
        pipeline)
            cmd_pipeline "$@"
            ;;
        *)
            echo "Unknown command: $command" >&2
            usage >&2
            return 1
            ;;
    esac
}

main "$@"

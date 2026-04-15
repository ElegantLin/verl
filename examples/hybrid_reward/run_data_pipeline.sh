#!/usr/bin/env bash
# Data preprocessing pipeline for hybrid reward training (HERO / EIF).
#
# Runs the shared stages:
#   1. Build source prompts
#   2. Generate candidate responses
#   3. Build RL train/val splits
#   4. Optionally filter TextBookReasoning
#   5. Build evaluation benchmark parquet files
#   6. Optionally run cold-start SFT
#
# Configure via RL_PIPELINE_* environment variables.

set -euo pipefail
set -x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/stage_lib.sh"

run_stage_data_train
run_stage_data_eval

if [[ "${RL_PIPELINE_ENABLE_COLD_START_SFT:-1}" == "1" ]]; then
    run_stage_sft
fi

export RL_PIPELINE_DATA_DIR="$data_dir"
export RL_PIPELINE_EVAL_DIR="$eval_dir"
export RL_PIPELINE_TRAIN_MODEL_PATH="${RL_PIPELINE_TRAIN_MODEL_PATH:-${RL_PIPELINE_MODEL_PATH:-$base_model_path}}"
export RL_PIPELINE_TRAIN_OUTPUT_DIR="$train_output_dir"
export RL_PIPELINE_EVAL_OUTPUT_DIR="$eval_output_dir"

echo "Shared data pipeline completed."
echo "Work directory: $work_dir"
echo "RL data directory: $data_dir"
echo "Eval benchmark directory: $eval_dir"
echo "Train model path: $RL_PIPELINE_TRAIN_MODEL_PATH"

#!/usr/bin/env bash
# EIF evaluation wrapper — maps EIF_* → RL_PIPELINE_* and delegates
# to the shared evaluation script (examples/shared/run_eval.sh).
#
# Usage:
#   EIF_MODEL_PATH=/path/to/checkpoint bash examples/eif/run_eif_eval.sh
#
#   # Evaluate only verifiable benchmarks (no external judge needed)
#   EIF_EVAL_BENCHMARKS="math500 amc minerva olympiad" bash examples/eif/run_eif_eval.sh

set -euo pipefail

export RL_PIPELINE_MODEL_PATH="${EIF_MODEL_PATH:-Qwen/Qwen3-4B-Base}"
export RL_PIPELINE_TRUST_REMOTE_CODE="${EIF_TRUST_REMOTE_CODE:-True}"
export RL_PIPELINE_EVAL_DIR="${EIF_EVAL_DIR:-$HOME/data/hero_eval}"
export RL_PIPELINE_EVAL_OUTPUT_DIR="${EIF_OUTPUT_DIR:-$HOME/data/eif_eval_results/$(basename "${EIF_MODEL_PATH:-Qwen_Qwen3-4B-Base}")_$(date +%Y%m%d_%H%M%S)}"
export RL_PIPELINE_EVAL_BENCHMARKS="${EIF_EVAL_BENCHMARKS:-math500 amc minerva olympiad hardverify_math textbook_reasoning}"
export RL_PIPELINE_EVAL_N_SAMPLES="${EIF_EVAL_N_SAMPLES:-8}"
export RL_PIPELINE_EVAL_TEMPERATURE="${EIF_EVAL_TEMPERATURE:-0.6}"
export RL_PIPELINE_EVAL_TOP_P="${EIF_EVAL_TOP_P:-0.95}"
export RL_PIPELINE_EVAL_MAX_PROMPT_LENGTH="${EIF_EVAL_MAX_PROMPT_LENGTH:-1024}"
export RL_PIPELINE_EVAL_MAX_RESPONSE_LENGTH="${EIF_EVAL_MAX_RESPONSE_LENGTH:-8192}"
export RL_PIPELINE_EVAL_TP_SIZE="${EIF_EVAL_TP_SIZE:-2}"
export RL_PIPELINE_EVAL_GPU_MEMORY_UTILIZATION="${EIF_EVAL_GPU_MEMORY_UTILIZATION:-0.85}"
export RL_PIPELINE_EVAL_GPUS_PER_NODE="${EIF_EVAL_GPUS_PER_NODE:-8}"
export RL_PIPELINE_EVAL_NNODES="${EIF_EVAL_NNODES:-1}"
export RL_PIPELINE_EVAL_PRIMARY_RESPONSE_INDEX="${EIF_EVAL_PRIMARY_RESPONSE_INDEX:-0}"
export RL_PIPELINE_JUDGE_MODEL="${EIF_JUDGE_MODEL:-gpt-oss}"
export RL_PIPELINE_JUDGE_BASE_URL="${EIF_JUDGE_BASE_URL:-https://ellm.nrp-nautilus.io/v1}"
export RL_PIPELINE_JUDGE_CONCURRENCY="${EIF_JUDGE_CONCURRENCY:-32}"

exec bash examples/shared/run_eval.sh "$@"

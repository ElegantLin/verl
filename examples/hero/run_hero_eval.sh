#!/usr/bin/env bash
# HERO evaluation wrapper — maps HERO_* → RL_PIPELINE_* and delegates
# to the shared evaluation script (examples/shared/run_eval.sh).
#
# Usage:
#   HERO_MODEL_PATH=/path/to/checkpoint bash examples/hero/run_hero_eval.sh
#
#   # Evaluate only verifiable benchmarks (no external judge needed)
#   HERO_EVAL_BENCHMARKS="math500 amc minerva olympiad" bash examples/hero/run_hero_eval.sh

set -euo pipefail

export RL_PIPELINE_MODEL_PATH="${HERO_MODEL_PATH:-Qwen/Qwen3-4B-Base}"
export RL_PIPELINE_TRUST_REMOTE_CODE="${HERO_TRUST_REMOTE_CODE:-True}"
export RL_PIPELINE_EVAL_DIR="${HERO_EVAL_DIR:-$HOME/data/hero_eval}"
export RL_PIPELINE_EVAL_OUTPUT_DIR="${HERO_OUTPUT_DIR:-$HOME/data/hero_eval_results/$(basename "${HERO_MODEL_PATH:-Qwen_Qwen3-4B-Base}")_$(date +%Y%m%d_%H%M%S)}"
export RL_PIPELINE_EVAL_BENCHMARKS="${HERO_EVAL_BENCHMARKS:-math500 amc minerva olympiad hardverify_math textbook_reasoning}"
export RL_PIPELINE_EVAL_N_SAMPLES="${HERO_EVAL_N_SAMPLES:-8}"
export RL_PIPELINE_EVAL_TEMPERATURE="${HERO_EVAL_TEMPERATURE:-0.6}"
export RL_PIPELINE_EVAL_TOP_P="${HERO_EVAL_TOP_P:-0.95}"
export RL_PIPELINE_EVAL_MAX_PROMPT_LENGTH="${HERO_EVAL_MAX_PROMPT_LENGTH:-1024}"
export RL_PIPELINE_EVAL_MAX_RESPONSE_LENGTH="${HERO_EVAL_MAX_RESPONSE_LENGTH:-8192}"
export RL_PIPELINE_EVAL_TP_SIZE="${HERO_EVAL_TP_SIZE:-2}"
export RL_PIPELINE_EVAL_GPU_MEMORY_UTILIZATION="${HERO_EVAL_GPU_MEMORY_UTILIZATION:-0.85}"
export RL_PIPELINE_EVAL_GPUS_PER_NODE="${HERO_EVAL_GPUS_PER_NODE:-8}"
export RL_PIPELINE_EVAL_NNODES="${HERO_EVAL_NNODES:-1}"
export RL_PIPELINE_EVAL_PRIMARY_RESPONSE_INDEX="${HERO_EVAL_PRIMARY_RESPONSE_INDEX:-0}"
export RL_PIPELINE_JUDGE_MODEL="${HERO_JUDGE_MODEL:-gpt-oss}"
export RL_PIPELINE_JUDGE_BASE_URL="${HERO_JUDGE_BASE_URL:-https://ellm.nrp-nautilus.io/v1}"
export RL_PIPELINE_JUDGE_CONCURRENCY="${HERO_JUDGE_CONCURRENCY:-32}"

exec bash examples/shared/run_eval.sh "$@"

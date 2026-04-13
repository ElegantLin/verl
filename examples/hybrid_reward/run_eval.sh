#!/usr/bin/env bash
# Shared evaluation script for HERO and EIF.
#
# Evaluation protocol (arXiv:2510.07242v1, Section 4.1):
#   - Verifiable tasks (MATH500, AMC, Minerva, Olympiad): pass@1 via math_verify
#   - Hard-to-verify tasks (HVM, TBR): pass@1 via an OpenAI-compatible LLM judge
#   - Decoding: temperature=0.6, top_p=0.95, N=8 candidates, evaluate first decoded output
#
# Prerequisites:
#   1. Preprocess eval benchmarks:
#        python examples/data_preprocess/hero_eval_benchmarks.py \
#            --local_save_dir ~/data/hero_eval
#   2. pip install math-verify
#   3. Set NAUTILUS_API_KEY for the external judge (hard-to-verify eval only)
#
# Usage:
#   RL_PIPELINE_MODEL_PATH=/path/to/checkpoint bash examples/hybrid_reward/run_eval.sh
#
# Configure via RL_PIPELINE_* environment variables.

set -euo pipefail
set -x

# ─── Model to evaluate ───────────────────────────────────────────────
model_path=${RL_PIPELINE_MODEL_PATH:-Qwen/Qwen3-4B-Base}
trust_remote_code=${RL_PIPELINE_TRUST_REMOTE_CODE:-True}

# ─── Eval data ───────────────────────────────────────────────────────
eval_dir=${RL_PIPELINE_EVAL_DIR:-$HOME/data/hero_eval}
output_dir=${RL_PIPELINE_EVAL_OUTPUT_DIR:-$HOME/data/eval_results/$(basename "$model_path")_$(date +%Y%m%d_%H%M%S)}

# ─── Benchmarks to evaluate ──────────────────────────────────────────
all_benchmarks=${RL_PIPELINE_EVAL_BENCHMARKS:-"math500 amc minerva olympiad hardverify_math textbook_reasoning"}

# ─── Generation settings ─────────────────────────────────────────────
n_samples=${RL_PIPELINE_EVAL_N_SAMPLES:-8}
temperature=${RL_PIPELINE_EVAL_TEMPERATURE:-0.6}
top_p=${RL_PIPELINE_EVAL_TOP_P:-0.95}
max_prompt_length=${RL_PIPELINE_EVAL_MAX_PROMPT_LENGTH:-1024}
max_response_length=${RL_PIPELINE_EVAL_MAX_RESPONSE_LENGTH:-8192}
tp_size=${RL_PIPELINE_EVAL_TP_SIZE:-2}
gpu_memory_utilization=${RL_PIPELINE_EVAL_GPU_MEMORY_UTILIZATION:-0.85}
gpus_per_node=${RL_PIPELINE_EVAL_GPUS_PER_NODE:-8}
nnodes=${RL_PIPELINE_EVAL_NNODES:-1}
primary_response_index=${RL_PIPELINE_EVAL_PRIMARY_RESPONSE_INDEX:-0}

# ─── LLM-as-judge settings ───────────────────────────────────────────
judge_model=${RL_PIPELINE_JUDGE_MODEL:-gpt-oss}
judge_base_url=${RL_PIPELINE_JUDGE_BASE_URL:-https://ellm.nrp-nautilus.io/v1}
judge_concurrency=${RL_PIPELINE_JUDGE_CONCURRENCY:-32}

# ─── Resolve judge script path ───────────────────────────────────────
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
judge_script="$script_dir/eval_llm_judge.py"
if [[ ! -f "$judge_script" ]]; then
    echo "ERROR: LLM judge script not found: $judge_script" >&2
    exit 1
fi

mkdir -p "$output_dir"

verifiable_benchmarks=""
hard_benchmarks=""

for bench in $all_benchmarks; do
    bench_file="$eval_dir/${bench}.parquet"
    if [[ ! -f "$bench_file" ]]; then
        echo "WARNING: Benchmark file not found: $bench_file (skipping $bench)" >&2
        continue
    fi

    case "$bench" in
        math500|amc|minerva|olympiad)
            verifiable_benchmarks="$verifiable_benchmarks $bench"
            ;;
        hardverify_math|textbook_reasoning)
            hard_benchmarks="$hard_benchmarks $bench"
            ;;
    esac
done

# ═══════════════════════════════════════════════════════════════════════
# Step 1: Generate responses for each benchmark
# ═══════════════════════════════════════════════════════════════════════
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Step 1: Generating responses                          ║"
echo "╚══════════════════════════════════════════════════════════╝"

for bench in $verifiable_benchmarks $hard_benchmarks; do
    bench_file="$eval_dir/${bench}.parquet"
    gen_output="$output_dir/${bench}_generated.parquet"

    if [[ -f "$gen_output" ]]; then
        echo "Skipping generation for $bench (output exists: $gen_output)"
        continue
    fi

    echo "Generating responses for $bench..."
    python3 -m verl.trainer.main_generation_server \
        trainer.nnodes="$nnodes" \
        trainer.n_gpus_per_node="$gpus_per_node" \
        data.train_files="['$bench_file']" \
        data.prompt_key=prompt \
        data.output_path="$gen_output" \
        actor_rollout_ref.model.path="$model_path" \
        actor_rollout_ref.model.trust_remote_code="$trust_remote_code" \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.n="$n_samples" \
        actor_rollout_ref.rollout.temperature="$temperature" \
        actor_rollout_ref.rollout.top_p="$top_p" \
        actor_rollout_ref.rollout.prompt_length="$max_prompt_length" \
        actor_rollout_ref.rollout.response_length="$max_response_length" \
        actor_rollout_ref.rollout.tensor_model_parallel_size="$tp_size" \
        actor_rollout_ref.rollout.gpu_memory_utilization="$gpu_memory_utilization"
done

# ═══════════════════════════════════════════════════════════════════════
# Step 2: Score verifiable benchmarks with math_verify
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Step 2: Scoring verifiable benchmarks (math_verify)    ║"
echo "╚══════════════════════════════════════════════════════════╝"

for bench in $verifiable_benchmarks; do
    gen_output="$output_dir/${bench}_generated.parquet"

    if [[ ! -f "$gen_output" ]]; then
        echo "WARNING: Missing generated file for $bench, skipping scoring."
        continue
    fi

    echo "Scoring $bench..."
    python3 -m verl.trainer.main_eval \
        data.path="$gen_output" \
        data.prompt_key=prompt \
        data.response_key=responses \
        data.primary_response_index="$primary_response_index" \
        data.data_source_key=data_source \
        data.reward_model_key=reward_model \
        2>&1 | tee "$output_dir/${bench}_eval.log"
done

# ═══════════════════════════════════════════════════════════════════════
# Step 3: Score hard-to-verify benchmarks with LLM judge
# ═══════════════════════════════════════════════════════════════════════
if [[ -n "$hard_benchmarks" ]]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  Step 3: Scoring hard-to-verify (LLM-as-Judge)         ║"
    echo "╚══════════════════════════════════════════════════════════╝"

    for bench in $hard_benchmarks; do
        gen_output="$output_dir/${bench}_generated.parquet"
        judge_output="$output_dir/${bench}_judge_results.json"

        if [[ ! -f "$gen_output" ]]; then
            echo "WARNING: Missing generated file for $bench, skipping judge eval."
            continue
        fi

        echo "Running LLM-as-judge for $bench..."
        judge_args=(
            --input_parquet "$gen_output"
            --judge_model "$judge_model"
            --concurrency "$judge_concurrency"
            --output_path "$judge_output"
            --primary_response_index "$primary_response_index"
        )
        if [[ -n "$judge_base_url" ]]; then
            judge_args+=(--base_url "$judge_base_url")
        fi

        python3 "$judge_script" "${judge_args[@]}" \
            2>&1 | tee "$output_dir/${bench}_judge.log"
    done
fi

# ═══════════════════════════════════════════════════════════════════════
# Step 4: Summary
# ═══════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Evaluation Complete                                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Model: $model_path"
echo "Results directory: $output_dir"
echo ""
echo "Files:"
ls -la "$output_dir/"

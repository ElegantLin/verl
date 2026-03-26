#!/usr/bin/env bash
# EIF evaluation script — same protocol as HERO (arXiv:2510.07242v1).
#
# Evaluation protocol (aligned with HERO Section 4.1):
#   - Verifiable tasks (MATH500, AMC, Minerva, Olympiad): pass@1 via math_verify
#   - Hard-to-verify tasks (HVM, TBR): pass@1 via an OpenAI-compatible LLM judge
#   - Decoding: temperature=0.6, top_p=0.95, N=8 candidates, evaluate first decoded output
#
# Prerequisites:
#   1. Preprocess eval benchmarks:
#        python examples/data_preprocess/hero_eval_benchmarks.py \
#            --local_save_dir ~/data/hero_eval
#   2. pip install math-verify
#   3. Set OPENAI_API_KEY for the external judge (hard-to-verify eval only)
#
# Usage:
#   EIF_MODEL_PATH=/path/to/checkpoint bash examples/eif/run_eif_eval.sh
#
#   # Evaluate only verifiable benchmarks (no external judge needed)
#   EIF_EVAL_BENCHMARKS="math500 amc minerva olympiad" bash examples/eif/run_eif_eval.sh

set -euo pipefail
set -x

# ─── Model to evaluate ───────────────────────────────────────────────
model_path=${EIF_MODEL_PATH:-Qwen/Qwen3-4B-Base}
trust_remote_code=${EIF_TRUST_REMOTE_CODE:-True}

# ─── Eval data ───────────────────────────────────────────────────────
eval_dir=${EIF_EVAL_DIR:-$HOME/data/hero_eval}
output_dir=${EIF_OUTPUT_DIR:-$HOME/data/eif_eval_results/$(basename "$model_path")_$(date +%Y%m%d_%H%M%S)}

# ─── Benchmarks to evaluate ──────────────────────────────────────────
all_benchmarks=${EIF_EVAL_BENCHMARKS:-"math500 amc minerva olympiad hardverify_math textbook_reasoning"}

# ─── Generation settings (aligned with HERO Section 4.1) ─────────────
n_samples=${EIF_EVAL_N_SAMPLES:-8}
temperature=${EIF_EVAL_TEMPERATURE:-0.6}
top_p=${EIF_EVAL_TOP_P:-0.95}
max_prompt_length=${EIF_EVAL_MAX_PROMPT_LENGTH:-1024}
max_response_length=${EIF_EVAL_MAX_RESPONSE_LENGTH:-8192}
tp_size=${EIF_EVAL_TP_SIZE:-2}
gpu_memory_utilization=${EIF_EVAL_GPU_MEMORY_UTILIZATION:-0.85}
gpus_per_node=${EIF_EVAL_GPUS_PER_NODE:-8}
nnodes=${EIF_EVAL_NNODES:-1}
primary_response_index=${EIF_EVAL_PRIMARY_RESPONSE_INDEX:-0}

# ─── LLM-as-judge settings ───────────────────────────────────────────
judge_model=${EIF_JUDGE_MODEL:-gpt-oss}
judge_base_url=${EIF_JUDGE_BASE_URL:-https://ellm.nrp-nautilus.io/v1}
judge_concurrency=${EIF_JUDGE_CONCURRENCY:-32}

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

    # Resolve path to the shared LLM judge script (reuses HERO implementation)
    script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
    repo_root=$(cd -- "$script_dir/../.." && pwd)
    judge_script="$repo_root/examples/hero/eval_hero_llm_judge.py"

    if [[ ! -f "$judge_script" ]]; then
        echo "ERROR: LLM judge script not found: $judge_script" >&2
        exit 1
    fi

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

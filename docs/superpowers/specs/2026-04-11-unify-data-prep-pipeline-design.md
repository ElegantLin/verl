# Unify HERO/EIF Data Preparation Pipeline

**Date:** 2026-04-11
**Status:** Draft
**Goal:** Eliminate duplicated data-prep, SFT, and evaluation scripts between HERO and EIF examples so that identical GPU work runs only once.

## Background

HERO and EIF share the same data preparation pipeline (steps 01–06) and evaluation protocol (step 08). The only algorithmic difference is the RL training step (step 07): HERO uses `reward_manager.name=hero` while EIF uses `reward_manager.name=hybrid_eif_online` with additional tau/m LLM endpoints.

Currently, both `examples/hero/step_by_step/` and `examples/eif/step_by_step/` contain full copies of all 8 step scripts plus `common.sh`. Eight of the ten files are identical modulo `HERO_`/`EIF_` variable prefixes. This duplication means users must run the expensive data-generation step (step 02, ~40k prompts) twice to compare the two algorithms, wasting GPU hours.

## Scope

**In scope:**
- Extract shared steps (01–06, 08) into `examples/shared/step_by_step/`
- Create a shared `common.sh` with algorithm-neutral variable names
- Move `eval_hero_llm_judge.py` to `examples/shared/eval_llm_judge.py`
- Move `run_hero_eval.sh` to `examples/shared/run_eval.sh`
- Merge `run_hero_cold_start_sft.sh` and `run_eif_cold_start_sft.sh` into `examples/shared/run_cold_start_sft.sh`
- Slim down `examples/hero/` and `examples/eif/` to training-only files
- Update READMEs for both algorithms

**Out of scope:**
- Changing the RL training scripts (`run_hero_train.sh`, `run_eif_train.sh`)
- Changing the Python data-preprocessing scripts under `examples/data_preprocess/`
- Changing `verl.trainer.main_generation_server` or `verl.trainer.main_eval`
- HERO baselines (`run_hero_rm_only.sh`, `run_hero_verifier_only.sh`, `run_hero_naive_combine.sh`) — but references to moved files must be updated

## Design

### Directory structure (after)

```
examples/
├── shared/
│   ├── step_by_step/
│   │   ├── common.sh
│   │   ├── 01_build_source_prompts.sh
│   │   ├── 02_generate_source_candidates.sh
│   │   ├── 03_build_rl_data.sh
│   │   ├── 04_optional_filter_tbr.sh
│   │   ├── 05_build_eval_benchmarks.sh
│   │   ├── 06_optional_cold_start_sft.sh
│   │   ├── 08_run_final_eval.sh
│   │   └── README.md
│   ├── run_eval.sh
│   ├── run_cold_start_sft.sh
│   └── eval_llm_judge.py
├── hero/
│   ├── step_by_step/
│   │   ├── common.sh          (sources shared/common.sh, adds HERO training defaults)
│   │   ├── 07_run_rl_train.sh
│   │   └── README.md
│   ├── run_hero_train.sh
│   ├── run_hero_pipeline.sh   (updated to call shared scripts)
│   ├── baseline_reward_fn.py
│   ├── run_hero_rm_only.sh
│   ├── run_hero_verifier_only.sh
│   └── run_hero_naive_combine.sh
└── eif/
    ├── step_by_step/
    │   ├── common.sh          (sources shared/common.sh, adds EIF training defaults)
    │   ├── 07_run_rl_train.sh
    │   └── README.md
    ├── run_eif_train.sh
    └── run_eif_pipeline.sh    (updated to call shared scripts)
```

### Shared data output (generated once)

```
~/data/rl_data_prep/<run>/
├── source_generation/
│   ├── source_prompts.parquet              # step 01
│   └── source_generated.parquet            # step 02 (most expensive)
├── openmathreasoning_hero/                 # name kept for compatibility; data is algorithm-neutral
│   ├── train_verifiable.parquet            # step 03
│   ├── train_hard_to_verify.parquet
│   ├── train_mixed.parquet
│   ├── val_verifiable.parquet
│   ├── val_hard_to_verify.parquet
│   ├── val_mixed.parquet
│   └── meta.json
├── hero_eval/                              # name kept for compatibility
│   ├── math500.parquet                     # step 05
│   ├── amc.parquet
│   ├── minerva.parquet
│   ├── olympiad.parquet
│   ├── hardverify_math.parquet
│   ├── textbook_reasoning.parquet
│   └── benchmark_summary.json
└── cold_start_sft/                         # step 06 (optional)
    └── checkpoints/
```

### Per-algorithm output (training + eval only)

```
~/data/hero_train/<run>/
├── checkpoints/hero_rl/
└── eval_results/

~/data/eif_train/<run>/
├── checkpoints/eif_rl/
└── eval_results/
```

### Shared `common.sh` — variable split

The shared `common.sh` owns **data-prep, generation, SFT, and eval** variables. Per-algorithm `common.sh` files own **RL training** variables only.

**Shared variables** (in `examples/shared/step_by_step/common.sh`, prefix `DATA_PREP_*` / `EVAL_*` / `SFT_*`):

```bash
#!/usr/bin/env bash

if [[ -n "${_SHARED_STEP_COMMON_SH:-}" ]]; then return 0; fi
_SHARED_STEP_COMMON_SH=1

shared_step_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${shared_step_dir}/../../.." && pwd)

set_default() {
    local var_name="$1" default_value="$2"
    if [[ -z "${!var_name:-}" ]]; then export "$var_name=$default_value"; fi
}

resolve_latest_hf_dir() {
    local root_dir="$1" suffix="$2"
    local tracker_file="$root_dir/latest_checkpointed_iteration.txt"
    [[ -f "$tracker_file" ]] || return 1
    local latest_step=$(<"$tracker_file")
    local resolved_path="$root_dir/global_step_${latest_step}${suffix}"
    [[ -d "$resolved_path" ]] || return 1
    printf '%s\n' "$resolved_path"
}

require_file() {
    local target_path="$1"
    if [[ ! -f "$target_path" ]]; then
        echo "Missing required file: $target_path" >&2; exit 1
    fi
}

ensure_dirs() {
    mkdir -p "$work_dir" "$source_dir" "$data_dir" "$eval_dir" \
             "$sft_data_dir" "$sft_output_dir"
}

# ─── GPU profile ────────────────────────────────────────────────
gpu_profile=${DATA_PREP_GPU_PROFILE:-8x24gb}

case "$gpu_profile" in
    8x24|8x24gb)
        # Data prep / generation
        set_default DATA_PREP_MAX_PROMPT_LENGTH 1024
        set_default DATA_PREP_MAX_RESPONSE_LENGTH 2048
        set_default DATA_PREP_GEN_GPUS_PER_NODE 8
        set_default DATA_PREP_GEN_NNODES 1
        set_default DATA_PREP_GEN_TP_SIZE 2
        set_default DATA_PREP_GEN_GPU_MEMORY_UTILIZATION 0.5
        set_default DATA_PREP_SOURCE_GENERATION_N 1
        # SFT
        set_default SFT_GPUS_PER_NODE 8
        set_default SFT_NNODES 1
        set_default SFT_TRAIN_BATCH_SIZE 32
        set_default SFT_MICRO_BATCH_SIZE_PER_GPU 1
        set_default SFT_MAX_LENGTH 4096
        set_default SFT_MAX_TOKEN_LEN_PER_GPU 6144
        # Eval
        set_default EVAL_GPUS_PER_NODE 8
        set_default EVAL_NNODES 1
        set_default EVAL_TP_SIZE 2
        set_default EVAL_GPU_MEMORY_UTILIZATION 0.5
        set_default EVAL_N_SAMPLES 8
        ;;
    4x80|4x80gb)
        set_default DATA_PREP_MAX_PROMPT_LENGTH 1024
        set_default DATA_PREP_MAX_RESPONSE_LENGTH 4096
        set_default DATA_PREP_GEN_GPUS_PER_NODE 4
        set_default DATA_PREP_GEN_NNODES 1
        set_default DATA_PREP_GEN_TP_SIZE 2
        set_default DATA_PREP_GEN_GPU_MEMORY_UTILIZATION 0.75
        set_default DATA_PREP_SOURCE_GENERATION_N 1
        set_default SFT_GPUS_PER_NODE 4
        set_default SFT_NNODES 1
        set_default SFT_TRAIN_BATCH_SIZE 64
        set_default SFT_MICRO_BATCH_SIZE_PER_GPU 2
        set_default SFT_MAX_LENGTH 6144
        set_default SFT_MAX_TOKEN_LEN_PER_GPU 8192
        set_default EVAL_GPUS_PER_NODE 4
        set_default EVAL_NNODES 1
        set_default EVAL_TP_SIZE 2
        set_default EVAL_GPU_MEMORY_UTILIZATION 0.75
        set_default EVAL_N_SAMPLES 8
        ;;
    2x80|2x80gb)
        set_default DATA_PREP_MAX_PROMPT_LENGTH 1024
        set_default DATA_PREP_MAX_RESPONSE_LENGTH 4096
        set_default DATA_PREP_GEN_GPUS_PER_NODE 2
        set_default DATA_PREP_GEN_NNODES 1
        set_default DATA_PREP_GEN_TP_SIZE 2
        set_default DATA_PREP_GEN_GPU_MEMORY_UTILIZATION 0.75
        set_default DATA_PREP_SOURCE_GENERATION_N 1
        set_default SFT_GPUS_PER_NODE 2
        set_default SFT_NNODES 1
        set_default SFT_TRAIN_BATCH_SIZE 32
        set_default SFT_MICRO_BATCH_SIZE_PER_GPU 2
        set_default SFT_MAX_LENGTH 6144
        set_default SFT_MAX_TOKEN_LEN_PER_GPU 8192
        set_default EVAL_GPUS_PER_NODE 2
        set_default EVAL_NNODES 1
        set_default EVAL_TP_SIZE 2
        set_default EVAL_GPU_MEMORY_UTILIZATION 0.75
        set_default EVAL_N_SAMPLES 8
        ;;
    *)
        echo "Unknown DATA_PREP_GPU_PROFILE=$gpu_profile. Use 8x24gb, 4x80gb, or 2x80gb." >&2
        exit 1
        ;;
esac

# ─── Artifact paths ─────────────────────────────────────────────
artifact_root=${DATA_PREP_ARTIFACT_ROOT:-$HOME/data/rl_data_prep}
run_name=${DATA_PREP_RUN_NAME:-step_by_step_run}
work_dir=${DATA_PREP_WORK_DIR:-$artifact_root/$run_name}
source_dir=${DATA_PREP_SOURCE_DIR:-$work_dir/source_generation}
data_dir=${DATA_PREP_DATA_DIR:-$work_dir/openmathreasoning_hero}
eval_dir=${DATA_PREP_EVAL_DIR:-$work_dir/hero_eval}
sft_data_dir=${DATA_PREP_SFT_DATA_DIR:-$work_dir/openmathreasoning_hero_sft}
sft_output_dir=${DATA_PREP_SFT_OUTPUT_DIR:-$work_dir/cold_start_sft}
source_prompts_path=${DATA_PREP_SOURCE_PROMPTS_PATH:-$source_dir/source_prompts.parquet}
source_generated_path=${DATA_PREP_SOURCE_GENERATED_PATH:-$source_dir/source_generated.parquet}
filtered_tbr_path=${DATA_PREP_FILTERED_TBR_PATH:-$eval_dir/textbook_reasoning_filtered.parquet}

# ─── Model ───────────────────────────────────────────────────────
base_model_path=${DATA_PREP_MODEL_PATH:-Qwen/Qwen3-4B-Base}
trust_remote_code=${DATA_PREP_TRUST_REMOTE_CODE:-True}
source_dataset_trust_remote_code=${DATA_PREP_SOURCE_DATASET_TRUST_REMOTE_CODE:-False}

# ─── Dataset ─────────────────────────────────────────────────────
dataset_name=${DATA_PREP_DATASET:-nvidia/OpenMathReasoning}
dataset_config=${DATA_PREP_DATASET_CONFIG:-}
dataset_split=${DATA_PREP_DATASET_SPLIT:-cot}
local_dataset_path=${DATA_PREP_LOCAL_DATASET_PATH:-}
source_question_col=${DATA_PREP_SOURCE_QUESTION_COL:-problem}
source_answer_col=${DATA_PREP_SOURCE_ANSWER_COL:-expected_answer}
question_col=${DATA_PREP_QUESTION_COL:-question}
answer_col=${DATA_PREP_ANSWER_COL:-answer}
problem_type_col=${DATA_PREP_PROBLEM_TYPE_COL:-problem_type}
problem_type_value=${DATA_PREP_PROBLEM_TYPE_VALUE:-has_answer_extracted}
source_sample_size=${DATA_PREP_SOURCE_SAMPLE_SIZE:-40000}
seed=${DATA_PREP_SEED:-42}

# ─── Generation ──────────────────────────────────────────────────
gen_nnodes=${DATA_PREP_GEN_NNODES:-1}
gen_gpus_per_node=${DATA_PREP_GEN_GPUS_PER_NODE:-8}
gen_tp_size=${DATA_PREP_GEN_TP_SIZE:-2}
gen_gpu_memory_utilization=${DATA_PREP_GEN_GPU_MEMORY_UTILIZATION:-0.75}
source_generation_n=${DATA_PREP_SOURCE_GENERATION_N:-1}
source_temperature=${DATA_PREP_SOURCE_TEMPERATURE:-1.0}
source_top_p=${DATA_PREP_SOURCE_TOP_P:-0.95}
max_prompt_length=${DATA_PREP_MAX_PROMPT_LENGTH:-1024}
max_response_length=${DATA_PREP_MAX_RESPONSE_LENGTH:-4096}

# ─── Eval benchmarks ────────────────────────────────────────────
eval_benchmarks=${DATA_PREP_EVAL_BENCHMARKS_BUILD:-"math500 amc minerva olympiad hardverify_math textbook_reasoning"}
hvm_local_path=${DATA_PREP_HVM_LOCAL_PATH:-}
tbr_local_path=${DATA_PREP_TBR_LOCAL_PATH:-}

# ─── Eval settings (shared protocol) ────────────────────────────
eval_n_samples=${EVAL_N_SAMPLES:-8}
eval_temperature=${EVAL_TEMPERATURE:-0.6}
eval_top_p=${EVAL_TOP_P:-0.95}
eval_max_prompt_length=${EVAL_MAX_PROMPT_LENGTH:-1024}
eval_max_response_length=${EVAL_MAX_RESPONSE_LENGTH:-8192}
eval_tp_size=${EVAL_TP_SIZE:-2}
eval_gpu_memory_utilization=${EVAL_GPU_MEMORY_UTILIZATION:-0.85}
eval_gpus_per_node=${EVAL_GPUS_PER_NODE:-8}
eval_nnodes=${EVAL_NNODES:-1}
eval_primary_response_index=${EVAL_PRIMARY_RESPONSE_INDEX:-0}
judge_model=${EVAL_JUDGE_MODEL:-gpt-oss}
judge_base_url=${EVAL_JUDGE_BASE_URL:-https://ellm.nrp-nautilus.io/v1}
judge_concurrency=${EVAL_JUDGE_CONCURRENCY:-32}

# ─── TBR filter settings (all 10 vars) ──────────────────────────
disable_tbr_math_verify_filter=${DATA_PREP_DISABLE_TBR_MATH_VERIFY_FILTER:-0}
tbr_answer_model=${DATA_PREP_TBR_ANSWER_MODEL:-}
tbr_answer_base_url=${DATA_PREP_TBR_ANSWER_BASE_URL:-}
tbr_answer_api_key_env=${DATA_PREP_TBR_ANSWER_API_KEY_ENV:-}
tbr_answer_judge_model=${DATA_PREP_TBR_ANSWER_JUDGE_MODEL:-}
tbr_answer_judge_base_url=${DATA_PREP_TBR_ANSWER_JUDGE_BASE_URL:-}
tbr_answer_judge_api_key_env=${DATA_PREP_TBR_ANSWER_JUDGE_API_KEY_ENV:-}
tbr_suitability_model=${DATA_PREP_TBR_SUITABILITY_MODEL:-}
tbr_suitability_base_url=${DATA_PREP_TBR_SUITABILITY_BASE_URL:-}
tbr_suitability_api_key_env=${DATA_PREP_TBR_SUITABILITY_API_KEY_ENV:-}

# ─── Paths to shared scripts ────────────────────────────────────
cold_start_sft_script="$repo_root/examples/shared/run_cold_start_sft.sh"
eval_script="$repo_root/examples/shared/run_eval.sh"
judge_script="$repo_root/examples/shared/eval_llm_judge.py"
```

**Per-algorithm variables** (training only — in `examples/hero/step_by_step/common.sh`):

```bash
#!/usr/bin/env bash
if [[ -n "${_HERO_STEP_COMMON_SH:-}" ]]; then return 0; fi
_HERO_STEP_COMMON_SH=1

# Source the shared data-prep/eval config
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../shared/step_by_step" && pwd)/common.sh"

# HERO-specific output paths
train_output_dir=${HERO_TRAIN_OUTPUT_DIR:-$HOME/data/hero_train/$run_name/checkpoints/hero_rl}
eval_output_dir=${HERO_EVAL_OUTPUT_DIR:-$HOME/data/hero_train/$run_name/eval_results}

ensure_hero_dirs() { ensure_dirs; mkdir -p "$train_output_dir" "$eval_output_dir"; }

# HERO GPU profile overrides (training only)
case "$gpu_profile" in
    8x24|8x24gb)
        set_default HERO_GPUS_PER_NODE 8
        set_default HERO_NNODES 1
        set_default HERO_TRAIN_BATCH_SIZE 64
        set_default HERO_PPO_MINI_BATCH_SIZE 64
        set_default HERO_PPO_MICRO_BATCH_SIZE_PER_GPU 1
        set_default HERO_PPO_MAX_TOKEN_LEN_PER_GPU 6144
        set_default HERO_ROLLOUT_N 4
        set_default HERO_ROLLOUT_TP_SIZE 2
        set_default HERO_ROLLOUT_GPU_MEMORY_UTILIZATION 0.35
        set_default HERO_ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU 2
        set_default HERO_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU 2
        set_default HERO_ROLLOUT_MAX_NUM_SEQS 128
        set_default HERO_RM_GPUS_PER_NODE 8
        set_default HERO_RM_NNODES 1
        set_default HERO_RM_TP_SIZE 2
        set_default HERO_RM_GPU_MEMORY_UTILIZATION 0.3
        set_default HERO_RM_MAX_NUM_SEQS 128
        ;;
    4x80|4x80gb)
        # ... (4x80 training defaults, identical between HERO and EIF)
        ;;
    2x80|2x80gb)
        # ... (2x80 training defaults)
        ;;
esac
```

EIF's `common.sh` follows the same pattern, with different 8x24gb defaults:

```bash
# 8x24gb differences from HERO:
set_default EIF_PPO_MICRO_BATCH_SIZE_PER_GPU 2    # HERO: 1
set_default EIF_ROLLOUT_GPU_MEMORY_UTILIZATION 0.5 # HERO: 0.35
set_default EIF_RM_GPU_MEMORY_UTILIZATION 0.5      # HERO: 0.3
# (no PPO_MAX_TOKEN_LEN_PER_GPU — HERO has 6144, EIF omits it)
```

### `run_cold_start_sft.sh` — merged from HERO superset

The shared SFT script is based on HERO's version (the superset). Features present in HERO but missing from EIF are added to the unified script:

| Feature | HERO | EIF | Shared |
|---------|------|-----|--------|
| `lr_scheduler_type` | configurable (cosine) | hardcoded default | configurable |
| `lr_warmup_steps_ratio` | configurable (0.1) | absent | configurable |
| `min_lr_ratio` | configurable (0.0) | absent | configurable |
| `attn_implementation` | flash_attention_2 with sdpa fallback | absent | flash_attention_2 with sdpa fallback |
| `data.ignore_input_ids_mismatch` | True | absent | True |
| `+model.override_config.attn_implementation` | set | absent | set |

Env var prefix changes from `HERO_*` to `SFT_*`. This is a functional improvement for EIF users who gain the missing settings.

### `run_eval.sh` — merged from HERO/EIF

The shared eval script is HERO's `run_hero_eval.sh` with `HERO_*` prefix changed to `EVAL_*`. The EIF version was identical except for the prefix and a dynamic path resolution for `eval_llm_judge.py` — the shared version uses the fixed path `$repo_root/examples/shared/eval_llm_judge.py`.

### `eval_llm_judge.py`

Moved from `examples/hero/eval_hero_llm_judge.py` to `examples/shared/eval_llm_judge.py`. No code changes needed — it takes all configuration via CLI arguments.

### Shared step scripts

Each shared step script sources `common.sh` from its own directory. They use `ensure_dirs` and unprefixed variables throughout.

**Step 04** uses unprefixed TBR vars:

```bash
# Before (hero-specific):
if [[ "${HERO_DISABLE_TBR_MATH_VERIFY_FILTER:-0}" == "1" ]]; then
# After (shared):
if [[ "$disable_tbr_math_verify_filter" == "1" ]]; then
```

**Step 06** uses `cold_start_sft_script` variable:

```bash
SFT_INPUT_PATH="$source_generated_path" \
SFT_DATA_DIR="$sft_data_dir" \
SFT_OUTPUT_DIR="$sft_output_dir" \
SFT_MODEL_PATH="$sft_model_path" \
SFT_TRUST_REMOTE_CODE="$trust_remote_code" \
bash "$cold_start_sft_script" "$@"
```

**Step 08** uses `eval_script`:

```bash
EVAL_MODEL_PATH="$eval_model_path" \
EVAL_TRUST_REMOTE_CODE="$trust_remote_code" \
EVAL_EVAL_DIR="$eval_dir" \
EVAL_OUTPUT_DIR="$eval_output_dir" \
bash "$eval_script" "$@"
```

### `repo_root` resolution

The shared `common.sh` computes `repo_root` relative to its own location: `repo_root=$(cd -- "${shared_step_dir}/../../.." && pwd)`. From `examples/shared/step_by_step/`, three levels up reaches the repo root — same depth as the current HERO/EIF `common.sh` files.

When per-algorithm `common.sh` sources the shared one, `repo_root` is set by the shared script (via `shared_step_dir`), so the path is always correct regardless of which file sources it.

## User workflow

```bash
# ═══ Data prep (run once, saves the most GPU hours) ═══
export DATA_PREP_RUN_NAME=my_run
export DATA_PREP_GPU_PROFILE=8x24gb
export DATA_PREP_MODEL_PATH=Qwen/Qwen3-4B-Base

bash examples/shared/step_by_step/01_build_source_prompts.sh
bash examples/shared/step_by_step/02_generate_source_candidates.sh  # slowest step
bash examples/shared/step_by_step/03_build_rl_data.sh
bash examples/shared/step_by_step/05_build_eval_benchmarks.sh

# ═══ Train HERO (reads shared data) ═══
export HERO_DATA_DIR=~/data/rl_data_prep/my_run/openmathreasoning_hero
export HERO_EVAL_DIR=~/data/rl_data_prep/my_run/hero_eval
bash examples/hero/step_by_step/07_run_rl_train.sh

# ═══ Train EIF (reads same shared data) ═══
export EIF_DATA_DIR=~/data/rl_data_prep/my_run/openmathreasoning_hero
export EIF_EVAL_DIR=~/data/rl_data_prep/my_run/hero_eval
bash examples/eif/step_by_step/07_run_rl_train.sh

# ═══ Evaluate (shared script, different models) ═══
EVAL_MODEL_PATH=<hero_checkpoint> \
EVAL_EVAL_DIR=~/data/rl_data_prep/my_run/hero_eval \
  bash examples/shared/step_by_step/08_run_final_eval.sh

EVAL_MODEL_PATH=<eif_checkpoint> \
EVAL_EVAL_DIR=~/data/rl_data_prep/my_run/hero_eval \
  bash examples/shared/step_by_step/08_run_final_eval.sh
```

## Files to create

| File | Source | Changes |
|------|--------|---------|
| `examples/shared/step_by_step/common.sh` | New | Merge shared parts of hero/eif common.sh; use `DATA_PREP_*` / `EVAL_*` / `SFT_*` prefix |
| `examples/shared/step_by_step/01_build_source_prompts.sh` | `hero/step_by_step/01_*` | Source shared common.sh, use `ensure_dirs` |
| `examples/shared/step_by_step/02_generate_source_candidates.sh` | `hero/step_by_step/02_*` | Same |
| `examples/shared/step_by_step/03_build_rl_data.sh` | `hero/step_by_step/03_*` | Same |
| `examples/shared/step_by_step/04_optional_filter_tbr.sh` | `hero/step_by_step/04_*` | Use unprefixed TBR vars (all 10) |
| `examples/shared/step_by_step/05_build_eval_benchmarks.sh` | `hero/step_by_step/05_*` | Same |
| `examples/shared/step_by_step/06_optional_cold_start_sft.sh` | `hero/step_by_step/06_*` | Use `cold_start_sft_script`, `SFT_*` prefix |
| `examples/shared/step_by_step/08_run_final_eval.sh` | `hero/step_by_step/08_*` | Use `eval_script`, `EVAL_*` prefix |
| `examples/shared/step_by_step/README.md` | New | Shared pipeline docs |
| `examples/shared/run_eval.sh` | `hero/run_hero_eval.sh` | `HERO_*` -> `EVAL_*` prefix; judge path -> `shared/eval_llm_judge.py` |
| `examples/shared/run_cold_start_sft.sh` | `hero/run_hero_cold_start_sft.sh` | `HERO_*` -> `SFT_*` prefix; use HERO superset features |
| `examples/shared/eval_llm_judge.py` | `hero/eval_hero_llm_judge.py` | Move, no code changes |

## Files to modify

| File | Changes |
|------|---------|
| `examples/hero/step_by_step/common.sh` | Source shared common.sh, keep only HERO training defaults |
| `examples/eif/step_by_step/common.sh` | Source shared common.sh, keep only EIF training defaults |
| `examples/hero/step_by_step/README.md` | Data-prep via shared scripts, only step 07 is HERO-specific |
| `examples/eif/step_by_step/README.md` | Same |
| `examples/hero/run_hero_pipeline.sh` | Update paths to shared scripts and env var names |
| `examples/eif/run_eif_pipeline.sh` | Update paths to shared scripts and env var names |

## Files to delete

| File | Reason |
|------|--------|
| `examples/hero/step_by_step/01_build_source_prompts.sh` | Replaced by shared |
| `examples/hero/step_by_step/02_generate_source_candidates.sh` | Replaced by shared |
| `examples/hero/step_by_step/03_build_rl_data.sh` | Replaced by shared |
| `examples/hero/step_by_step/04_optional_filter_tbr.sh` | Replaced by shared |
| `examples/hero/step_by_step/05_build_eval_benchmarks.sh` | Replaced by shared |
| `examples/hero/step_by_step/06_optional_cold_start_sft.sh` | Replaced by shared |
| `examples/hero/step_by_step/08_run_final_eval.sh` | Replaced by shared |
| `examples/eif/step_by_step/01_build_source_prompts.sh` | Replaced by shared |
| `examples/eif/step_by_step/02_generate_source_candidates.sh` | Replaced by shared |
| `examples/eif/step_by_step/03_build_rl_data.sh` | Replaced by shared |
| `examples/eif/step_by_step/04_optional_filter_tbr.sh` | Replaced by shared |
| `examples/eif/step_by_step/05_build_eval_benchmarks.sh` | Replaced by shared |
| `examples/eif/step_by_step/06_optional_cold_start_sft.sh` | Replaced by shared |
| `examples/eif/step_by_step/08_run_final_eval.sh` | Replaced by shared |
| `examples/hero/run_hero_eval.sh` | Replaced by `shared/run_eval.sh` |
| `examples/hero/run_hero_cold_start_sft.sh` | Replaced by `shared/run_cold_start_sft.sh` |
| `examples/hero/eval_hero_llm_judge.py` | Moved to `shared/eval_llm_judge.py` |
| `examples/eif/run_eif_eval.sh` | Replaced by `shared/run_eval.sh` |
| `examples/eif/run_eif_cold_start_sft.sh` | Replaced by `shared/run_cold_start_sft.sh` |

## References to update

All files referencing moved/deleted scripts (found via grep):

| File | Reference | Update to |
|------|-----------|-----------|
| `examples/hero/run_hero_pipeline.sh:204` | `bash examples/hero/run_hero_cold_start_sft.sh` | `bash examples/shared/run_cold_start_sft.sh` |
| `examples/hero/run_hero_pipeline.sh:212` | `bash examples/hero/run_hero_eval.sh` | `bash examples/shared/run_eval.sh` |
| `examples/hero/run_hero_eval.sh:168` | `python3 examples/hero/eval_hero_llm_judge.py` | (file is deleted; `shared/run_eval.sh` uses `shared/eval_llm_judge.py`) |
| `examples/eif/run_eif_eval.sh:149` | `examples/hero/eval_hero_llm_judge.py` | (file is deleted; `shared/run_eval.sh` uses `shared/eval_llm_judge.py`) |

HERO baseline scripts (`run_hero_rm_only.sh`, `run_hero_verifier_only.sh`, `run_hero_naive_combine.sh`) do not reference eval or SFT scripts — they are self-contained training configs. No changes needed.

## Backward compatibility

This is a **breaking change** for users who run `bash examples/hero/step_by_step/01_*.sh` directly.

**What breaks:**
- Old step scripts (01-06, 08) in hero/eif directories are deleted
- `HERO_SOURCE_SAMPLE_SIZE` and similar data-prep env vars are replaced by `DATA_PREP_*`
- `HERO_SFT_*` env vars for cold-start SFT are replaced by `SFT_*`
- `HERO_EVAL_*` / `EIF_EVAL_*` env vars are replaced by `EVAL_*`

**What still works:**
- Training env vars (`HERO_MODEL_PATH`, `HERO_ROLLOUT_N`, `EIF_MODEL_PATH`, etc.) for step 07
- `run_hero_train.sh` and `run_eif_train.sh` are unchanged

**Mitigation:**
- READMEs updated with new commands
- Shared `common.sh` includes deprecation warnings for the most common old env vars:

```bash
# Deprecation shims — warn and forward
for _old_var in HERO_SOURCE_SAMPLE_SIZE EIF_SOURCE_SAMPLE_SIZE; do
    _new_var="DATA_PREP_SOURCE_SAMPLE_SIZE"
    if [[ -n "${!_old_var:-}" && -z "${!_new_var:-}" ]]; then
        echo "WARNING: $_old_var is deprecated. Use $_new_var instead." >&2
        export "$_new_var=${!_old_var}"
    fi
done
# (repeat for other high-use vars: MODEL_PATH, GPU_PROFILE, LOCAL_DATASET_PATH, SEED)
```

## Testing

- Run shared steps 01–05 once with default settings
- Run optional step 06 (cold-start SFT)
- Train both HERO and EIF from the shared data
- Evaluate both models with shared eval script (step 08)
- Verify eval metrics match the current separate-pipeline results
- Verify deprecation warnings fire correctly for old env vars

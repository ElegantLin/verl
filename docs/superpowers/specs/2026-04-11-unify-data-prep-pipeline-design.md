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
- Move `run_hero_cold_start_sft.sh` to `examples/shared/run_cold_start_sft.sh`
- Slim down `examples/hero/` and `examples/eif/` to training-only files
- Update READMEs for both algorithms

**Out of scope:**
- Changing the RL training scripts (`run_hero_train.sh`, `run_eif_train.sh`)
- Changing the Python data-preprocessing scripts under `examples/data_preprocess/`
- Changing `verl.trainer.main_generation_server` or `verl.trainer.main_eval`
- HERO baselines (`run_hero_rm_only.sh`, `run_hero_verifier_only.sh`, `run_hero_naive_combine.sh`)

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
│   ├── run_hero_pipeline.sh   (updated to call shared steps)
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
    └── run_eif_pipeline.sh    (updated to call shared steps)
```

### Shared data output (generated once)

```
~/data/rl_data_prep/<run>/
├── source_generation/
│   ├── source_prompts.parquet              # step 01
│   └── source_generated.parquet            # step 02 (most expensive)
├── openmathreasoning_hero/
│   ├── train_verifiable.parquet            # step 03
│   ├── train_hard_to_verify.parquet
│   ├── train_mixed.parquet
│   ├── val_verifiable.parquet
│   ├── val_hard_to_verify.parquet
│   ├── val_mixed.parquet
│   └── meta.json
├── hero_eval/
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

### Shared `common.sh`

Uses algorithm-neutral env var prefix `DATA_PREP_*` for data-related settings and `EVAL_*` for eval settings:

```bash
# Data prep configuration
artifact_root=${DATA_PREP_ARTIFACT_ROOT:-$HOME/data/rl_data_prep}
run_name=${DATA_PREP_RUN_NAME:-step_by_step_run}
gpu_profile=${DATA_PREP_GPU_PROFILE:-8x24gb}
base_model_path=${DATA_PREP_MODEL_PATH:-Qwen/Qwen3-4B-Base}

# Dataset
dataset_name=${DATA_PREP_DATASET:-nvidia/OpenMathReasoning}
dataset_split=${DATA_PREP_DATASET_SPLIT:-cot}
source_sample_size=${DATA_PREP_SOURCE_SAMPLE_SIZE:-40000}
seed=${DATA_PREP_SEED:-42}
# ... (all other vars follow same pattern)

# GPU profile defaults (same values currently shared by HERO and EIF)
case "$gpu_profile" in
    8x24|8x24gb)
        set_default DATA_PREP_MAX_PROMPT_LENGTH 1024
        set_default DATA_PREP_MAX_RESPONSE_LENGTH 2048
        set_default DATA_PREP_GEN_TP_SIZE 2
        set_default DATA_PREP_GEN_GPU_MEMORY_UTILIZATION 0.5
        # ...
        ;;
    4x80|4x80gb)
        # ...
        ;;
esac

# Eval settings (shared protocol)
eval_n_samples=${EVAL_N_SAMPLES:-8}
eval_temperature=${EVAL_TEMPERATURE:-0.6}
eval_top_p=${EVAL_TOP_P:-0.95}
judge_model=${EVAL_JUDGE_MODEL:-gpt-oss}
judge_base_url=${EVAL_JUDGE_BASE_URL:-https://ellm.nrp-nautilus.io/v1}

# TBR filter settings (unprefixed)
disable_tbr_math_verify_filter=${DATA_PREP_DISABLE_TBR_MATH_VERIFY_FILTER:-0}
tbr_answer_model=${DATA_PREP_TBR_ANSWER_MODEL:-}
tbr_answer_base_url=${DATA_PREP_TBR_ANSWER_BASE_URL:-}
# ... (all TBR vars)
```

### Per-algorithm `common.sh`

Each algorithm's `common.sh` sources the shared one and adds training-specific settings:

```bash
# examples/hero/step_by_step/common.sh
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../shared/step_by_step" && pwd)/common.sh"

# HERO-specific training settings
train_output_dir=${HERO_TRAIN_OUTPUT_DIR:-$HOME/data/hero_train/$run_name/checkpoints/hero_rl}
eval_output_dir=${HERO_EVAL_OUTPUT_DIR:-$HOME/data/hero_train/$run_name/eval_results}

# HERO GPU profile overrides for training
case "$gpu_profile" in
    8x24|8x24gb)
        set_default HERO_PPO_MICRO_BATCH_SIZE_PER_GPU 1
        set_default HERO_ROLLOUT_GPU_MEMORY_UTILIZATION 0.35
        set_default HERO_RM_GPU_MEMORY_UTILIZATION 0.3
        # ...
        ;;
esac
```

### Shared step scripts

Each shared step script assumes `common.sh` has already been sourced (either directly or via an algorithm's `common.sh`). They use `ensure_dirs` (generic name) and unprefixed variables throughout.

**Step 04** uses unprefixed TBR vars resolved by `common.sh`:

```bash
# Before (hero-specific):
if [[ "${HERO_DISABLE_TBR_MATH_VERIFY_FILTER:-0}" == "1" ]]; then
# After (shared):
if [[ "$disable_tbr_math_verify_filter" == "1" ]]; then
```

**Step 06** uses `cold_start_sft_script` variable set by `common.sh`:

```bash
# common.sh sets:
cold_start_sft_script="$repo_root/examples/shared/run_cold_start_sft.sh"

# Shared 06 script uses:
bash "$cold_start_sft_script" "$@"
```

**Step 08** uses `eval_script` and `judge_script` variables:

```bash
eval_script="$repo_root/examples/shared/run_eval.sh"
judge_script="$repo_root/examples/shared/eval_llm_judge.py"
```

### `run_eval.sh` and `run_cold_start_sft.sh`

These are the current `run_hero_eval.sh` and `run_hero_cold_start_sft.sh` with env var prefixes changed from `HERO_*` to generic `EVAL_*` / `SFT_*`. The logic is unchanged.

### `eval_llm_judge.py`

Moved from `examples/hero/eval_hero_llm_judge.py` to `examples/shared/eval_llm_judge.py`. No code changes needed — it takes all configuration via CLI arguments.

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
| `examples/shared/step_by_step/common.sh` | New | Merge shared parts of hero/eif common.sh with `DATA_PREP_*` / `EVAL_*` prefix |
| `examples/shared/step_by_step/01_build_source_prompts.sh` | `hero/step_by_step/01_*` | Remove `source common.sh`, use `ensure_dirs`, use unprefixed vars |
| `examples/shared/step_by_step/02_generate_source_candidates.sh` | `hero/step_by_step/02_*` | Same |
| `examples/shared/step_by_step/03_build_rl_data.sh` | `hero/step_by_step/03_*` | Same |
| `examples/shared/step_by_step/04_optional_filter_tbr.sh` | `hero/step_by_step/04_*` | Use unprefixed TBR vars |
| `examples/shared/step_by_step/05_build_eval_benchmarks.sh` | `hero/step_by_step/05_*` | Same |
| `examples/shared/step_by_step/06_optional_cold_start_sft.sh` | `hero/step_by_step/06_*` | Use `cold_start_sft_script` var |
| `examples/shared/step_by_step/08_run_final_eval.sh` | `hero/step_by_step/08_*` | Use `eval_script` var, `EVAL_*` prefix |
| `examples/shared/step_by_step/README.md` | New | Shared pipeline docs |
| `examples/shared/run_eval.sh` | `hero/run_hero_eval.sh` | `HERO_*` -> `EVAL_*` prefix |
| `examples/shared/run_cold_start_sft.sh` | `hero/run_hero_cold_start_sft.sh` | `HERO_*` -> `SFT_*` prefix |
| `examples/shared/eval_llm_judge.py` | `hero/eval_hero_llm_judge.py` | Move, no code changes |

## Files to modify

| File | Changes |
|------|---------|
| `examples/hero/step_by_step/common.sh` | Source shared common.sh, keep only HERO training defaults |
| `examples/eif/step_by_step/common.sh` | Source shared common.sh, keep only EIF training defaults |
| `examples/hero/step_by_step/README.md` | Update: data-prep via shared scripts, only step 07 is HERO-specific |
| `examples/eif/step_by_step/README.md` | Same |
| `examples/hero/run_hero_pipeline.sh` | Point to shared step scripts |
| `examples/eif/run_eif_pipeline.sh` | Point to shared step scripts |

## Files to delete

| File | Reason |
|------|--------|
| `examples/hero/step_by_step/01–06, 08` | Replaced by shared scripts |
| `examples/eif/step_by_step/01–06, 08` | Replaced by shared scripts |
| `examples/hero/run_hero_eval.sh` | Replaced by `shared/run_eval.sh` |
| `examples/hero/run_hero_cold_start_sft.sh` | Replaced by `shared/run_cold_start_sft.sh` |
| `examples/hero/eval_hero_llm_judge.py` | Moved to `shared/eval_llm_judge.py` |
| `examples/eif/run_eif_eval.sh` | Replaced by `shared/run_eval.sh` |
| `examples/eif/run_eif_cold_start_sft.sh` | Replaced by `shared/run_cold_start_sft.sh` |

## Backward compatibility

This is a **breaking change** for existing users who run `bash examples/hero/step_by_step/01_*.sh` directly. Mitigation:

- READMEs updated with new commands
- Old env vars (`HERO_SOURCE_SAMPLE_SIZE`, etc.) no longer work for data prep; users must switch to `DATA_PREP_*` prefix
- Training env vars (`HERO_MODEL_PATH`, `EIF_MODEL_PATH`, etc.) still work for step 07

## Testing

- Run shared steps 01–05 once
- Train both HERO and EIF from the shared data
- Evaluate both models with shared eval script
- Verify results match the current separate-pipeline results

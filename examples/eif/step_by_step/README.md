# EIF Step-by-Step Bash Scripts

These scripts split the EIF pipeline into ordered stages you can run one by one.

Stages 1-6 (data preprocessing) are shared with the HERO pipeline via
`examples/shared/step_by_step/`. The EIF wrappers here map `EIF_*` env vars
to the shared `RL_PIPELINE_*` prefix automatically.

## Running Both HERO and EIF (Recommended)

If you plan to run both algorithms, preprocess once to save GPU hours:

```bash
# 1. Run shared preprocessing (stages 1-6, only once)
export RL_PIPELINE_RUN_NAME=shared_run
export RL_PIPELINE_MODEL_PATH=Qwen/Qwen3-4B-Base
bash examples/shared/run_data_pipeline.sh

# 2. Train HERO using the shared artifacts
HERO_DATA_DIR=$RL_PIPELINE_DATA_DIR \
HERO_EVAL_DIR=$RL_PIPELINE_EVAL_DIR \
HERO_MODEL_PATH=$RL_PIPELINE_TRAIN_MODEL_PATH \
bash examples/hero/run_hero_train.sh

# 3. Train EIF using the same shared artifacts
export NAUTILUS_API_KEY=<your-key>
EIF_DATA_DIR=$RL_PIPELINE_DATA_DIR \
EIF_EVAL_DIR=$RL_PIPELINE_EVAL_DIR \
EIF_MODEL_PATH=$RL_PIPELINE_TRAIN_MODEL_PATH \
bash examples/eif/run_eif_train.sh
```

This avoids running candidate generation (~hours on 8 GPUs) and cold-start
SFT (~30-60 min) twice. See `examples/shared/README.md` for full details.

## Running EIF Only

### One-command pipeline

```bash
export NAUTILUS_API_KEY=<your-key>
bash examples/eif/run_eif_pipeline.sh
```

This runs stages 1-6 (shared preprocessing) then stage 7 (EIF training) and
stage 8 (evaluation) in one go.

### Step-by-step

Setup:

```bash
export EIF_RUN_NAME=my_eif_run
export EIF_GPU_PROFILE=8x24gb   # or 4x80gb or 2x80gb
export EIF_MODEL_PATH=Qwen/Qwen3-4B-Base
export NAUTILUS_API_KEY=<your-key>  # required for tau/m LLM endpoints

# If you already have OpenMathReasoning locally:
# export EIF_LOCAL_DATASET_PATH=/path/to/openmathreasoning
```

Run in order:

```bash
bash examples/eif/step_by_step/01_build_source_prompts.sh
bash examples/eif/step_by_step/02_generate_source_candidates.sh
bash examples/eif/step_by_step/03_build_rl_data.sh
bash examples/eif/step_by_step/05_build_eval_benchmarks.sh
bash examples/eif/step_by_step/07_run_rl_train.sh
bash examples/eif/step_by_step/08_run_final_eval.sh
```

Optional stages:

```bash
bash examples/eif/step_by_step/04_optional_filter_tbr.sh
bash examples/eif/step_by_step/06_optional_cold_start_sft.sh
```

## Speeding Up Stage 2

Stage 2 (candidate generation) is the slowest preprocessing step because
it runs vLLM inference over up to 40k source prompts.

For a quick smoke test:

```bash
export EIF_SOURCE_SAMPLE_SIZE=1000
export EIF_MAX_RESPONSE_LENGTH=1024
```

For better throughput on `Qwen/Qwen3-4B-Base`:

```bash
EIF_SOURCE_SAMPLE_SIZE=5000 \
EIF_MAX_RESPONSE_LENGTH=1024 \
EIF_GEN_TP_SIZE=1 \
EIF_GEN_GPU_MEMORY_UTILIZATION=0.9 \
bash examples/eif/step_by_step/02_generate_source_candidates.sh \
  actor_rollout_ref.rollout.max_model_len=3072 \
  actor_rollout_ref.rollout.max_num_seqs=256 \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384
```

## GPU Profile Presets

| Profile | Description |
|---|---|
| `EIF_GPU_PROFILE=8x24gb` | Conservative defaults for 8 x 24GB GPUs |
| `EIF_GPU_PROFILE=4x80gb` | Larger defaults for 4 x 80GB GPUs |
| `EIF_GPU_PROFILE=2x80gb` | Minimal defaults for 2 x 80GB GPUs |

## Notes

- You can override any underlying env var, such as `EIF_MAX_RESPONSE_LENGTH`,
  `EIF_ROLLOUT_N`, `EIF_RM_TP_SIZE`, or `EIF_REGIME`.
- Set `EIF_DEBUG=1` to use only the `amc` benchmark for validation during
  RL training (faster iteration).
- EIF training requires tau/m LLM endpoints. Configure with `EIF_TAU_MODEL`,
  `EIF_TAU_BASE_URL`, `EIF_M_MODEL`, `EIF_M_BASE_URL`.
- Set `NAUTILUS_API_KEY` before `08_run_final_eval.sh` if you evaluate
  `hardverify_math` or `textbook_reasoning` (requires LLM judge).

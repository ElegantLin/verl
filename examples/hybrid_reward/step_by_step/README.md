# Hybrid Reward Step-by-Step Scripts

These scripts split the HERO/EIF pipeline into ordered stages you can run
one by one. Set `ALGORITHM=hero` (default) or `ALGORITHM=eif` to select the
reward strategy for training (stage 7).

Stages 1-6 (data preprocessing) are identical for both algorithms and only
need to run once.

## One-command pipeline

```bash
ALGORITHM=hero bash examples/hybrid_reward/run_pipeline.sh
ALGORITHM=eif  bash examples/hybrid_reward/run_pipeline.sh
```

## Step-by-step

Setup:

```bash
export RL_PIPELINE_RUN_NAME=my_run
export RL_PIPELINE_GPU_PROFILE=8x24gb   # or 4x80gb or 2x80gb
export RL_PIPELINE_MODEL_PATH=Qwen/Qwen3-4B-Base

# For EIF: set tau/m LLM endpoint credentials
# export NAUTILUS_API_KEY=<your-key>

# If you already have OpenMathReasoning locally:
# export RL_PIPELINE_LOCAL_DATASET_PATH=/path/to/openmathreasoning
```

Run in order:

```bash
bash examples/hybrid_reward/step_by_step/01_build_source_prompts.sh
bash examples/hybrid_reward/step_by_step/02_generate_source_candidates.sh
bash examples/hybrid_reward/step_by_step/03_build_rl_data.sh
bash examples/hybrid_reward/step_by_step/05_build_eval_benchmarks.sh
ALGORITHM=hero bash examples/hybrid_reward/step_by_step/07_run_rl_train.sh
bash examples/hybrid_reward/step_by_step/08_run_final_eval.sh
```

Optional stages:

```bash
bash examples/hybrid_reward/step_by_step/04_optional_filter_tbr.sh
bash examples/hybrid_reward/step_by_step/06_optional_cold_start_sft.sh
```

## Running Both HERO and EIF

Since stages 1-6 are shared, preprocess once then train both:

```bash
# 1. Run stages 1-6 (only once)
bash examples/hybrid_reward/step_by_step/01_build_source_prompts.sh
bash examples/hybrid_reward/step_by_step/02_generate_source_candidates.sh
bash examples/hybrid_reward/step_by_step/03_build_rl_data.sh
bash examples/hybrid_reward/step_by_step/05_build_eval_benchmarks.sh
bash examples/hybrid_reward/step_by_step/06_optional_cold_start_sft.sh

# 2. Train HERO
ALGORITHM=hero bash examples/hybrid_reward/step_by_step/07_run_rl_train.sh

# 3. Train EIF (can run in parallel on a different node)
ALGORITHM=eif bash examples/hybrid_reward/step_by_step/07_run_rl_train.sh
```

## Speeding Up Stage 2

Stage 2 (candidate generation) is the slowest preprocessing step because
it runs vLLM inference over up to 40k source prompts.

For a quick smoke test:

```bash
export RL_PIPELINE_SOURCE_SAMPLE_SIZE=1000
export RL_PIPELINE_MAX_RESPONSE_LENGTH=1024
```

For better throughput on `Qwen/Qwen3-4B-Base`:

```bash
RL_PIPELINE_SOURCE_SAMPLE_SIZE=5000 \
RL_PIPELINE_MAX_RESPONSE_LENGTH=1024 \
RL_PIPELINE_GEN_TP_SIZE=1 \
RL_PIPELINE_GEN_GPU_MEMORY_UTILIZATION=0.9 \
bash examples/hybrid_reward/step_by_step/02_generate_source_candidates.sh \
  actor_rollout_ref.rollout.max_model_len=3072 \
  actor_rollout_ref.rollout.max_num_seqs=256 \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384
```

## GPU Profile Presets

| Profile | Description |
|---|---|
| `RL_PIPELINE_GPU_PROFILE=8x24gb` | Conservative defaults for 8 x 24GB GPUs |
| `RL_PIPELINE_GPU_PROFILE=4x80gb` | Larger defaults for 4 x 80GB GPUs |
| `RL_PIPELINE_GPU_PROFILE=2x80gb` | Minimal defaults for 2 x 80GB GPUs |

## Notes

- Set `DEBUG=1` to use only the `amc` benchmark for validation during
  RL training (faster iteration).
- EIF training requires tau/m LLM endpoints. Configure with `EIF_TAU_MODEL`,
  `EIF_TAU_BASE_URL`, `EIF_M_MODEL`, `EIF_M_BASE_URL`.
- Set `NAUTILUS_API_KEY` before `08_run_final_eval.sh` if you evaluate
  `hardverify_math` or `textbook_reasoning` (requires LLM judge).

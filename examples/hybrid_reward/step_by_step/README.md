# Hybrid Reward Semantic Stage Scripts

These scripts expose the same workflow as `examples/hybrid_reward/run.sh`, but
as direct stage entrypoints when you want to run the pipeline manually.

Available stages:

- `data-train.sh`
- `data-eval.sh`
- `sft.sh`
- `rl.sh`
- `eval.sh`

## Recommended Interface

For most usage, prefer:

```bash
bash examples/hybrid_reward/run.sh pipeline --algorithm hero
bash examples/hybrid_reward/run.sh pipeline --algorithm eif
```

## Stage-by-Stage Usage

Setup:

```bash
export RL_PIPELINE_RUN_NAME=my_run
export RL_PIPELINE_GPU_PROFILE=8x24gb   # or 4x80gb or 2x80gb

# For EIF: set tau/m endpoint credentials
# export NAUTILUS_API_KEY=<your-key>

# If you already have OpenMathReasoning locally:
# export RL_PIPELINE_LOCAL_DATASET_PATH=/path/to/openmathreasoning
```

Run in order:

```bash
bash examples/hybrid_reward/step_by_step/data-train.sh
bash examples/hybrid_reward/step_by_step/data-eval.sh
bash examples/hybrid_reward/step_by_step/sft.sh
bash examples/hybrid_reward/step_by_step/rl.sh --algorithm hero
bash examples/hybrid_reward/step_by_step/eval.sh --algorithm hero
```

To run EIF instead:

```bash
bash examples/hybrid_reward/step_by_step/rl.sh --algorithm eif
bash examples/hybrid_reward/step_by_step/eval.sh --algorithm eif
```

## Shared Preprocess, Separate RL

```bash
# Shared stages
bash examples/hybrid_reward/step_by_step/data-train.sh
bash examples/hybrid_reward/step_by_step/data-eval.sh
bash examples/hybrid_reward/step_by_step/sft.sh

# HERO
bash examples/hybrid_reward/step_by_step/rl.sh --algorithm hero

# EIF
export NAUTILUS_API_KEY=<your-key>
bash examples/hybrid_reward/step_by_step/rl.sh --algorithm eif
```

## Debug Shortcuts

```bash
# Smaller source sample and shorter responses
bash examples/hybrid_reward/step_by_step/data-train.sh --debug

# Build only amc
bash examples/hybrid_reward/step_by_step/data-eval.sh --debug
```

## Optional TBR Filtering

```bash
bash examples/hybrid_reward/step_by_step/data-eval.sh --filter-tbr
```

## GPU Profile Presets

| Profile | Description |
| --- | --- |
| `RL_PIPELINE_GPU_PROFILE=8x24gb` | Conservative defaults for 8 x 24GB GPUs |
| `RL_PIPELINE_GPU_PROFILE=4x80gb` | Larger defaults for 4 x 80GB GPUs |
| `RL_PIPELINE_GPU_PROFILE=2x80gb` | Minimal defaults for 2 x 80GB GPUs |

## Notes

- EIF training requires tau/m LLM endpoints. Configure them with the existing
  `EIF_*` environment variables consumed by `run_train.sh`.
- Use `--model-path` with `eval.sh` if you want to evaluate an explicit
  checkpoint instead of the latest actor for the selected algorithm.
- `hardverify_math` and `textbook_reasoning` still require LLM-as-judge
  credentials during evaluation.

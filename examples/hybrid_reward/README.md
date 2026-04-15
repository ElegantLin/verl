# Hybrid Reward Unified CLI

Canonical entrypoint: `examples/hybrid_reward/run.sh`.

This directory provides one bash CLI for the shared HERO/EIF workflow:

- **HERO** (`--algorithm hero`) — reward shaping with stratified RM normalization
- **EIF** (`--algorithm eif`) — one-step estimator with online tau/m regressors

Data preparation, eval-benchmark preparation, and cold-start SFT are shared.
Only RL training differs by algorithm.

## Canonical Commands

```bash
bash examples/hybrid_reward/run.sh data-train
bash examples/hybrid_reward/run.sh data-train --debug

bash examples/hybrid_reward/run.sh data-eval
bash examples/hybrid_reward/run.sh data-eval --debug

bash examples/hybrid_reward/run.sh sft

bash examples/hybrid_reward/run.sh rl --algorithm hero
bash examples/hybrid_reward/run.sh rl --algorithm eif

bash examples/hybrid_reward/run.sh eval --algorithm hero
bash examples/hybrid_reward/run.sh eval --model-path /path/to/checkpoint

bash examples/hybrid_reward/run.sh pipeline --algorithm hero
```

Stage meanings:

- `data-train` — build source prompts, generate candidates, build RL train/val parquet files
- `data-eval` — build evaluation parquet files, with optional TBR filtering
- `sft` — run the cold-start SFT stage
- `rl` — run algorithm-specific RL training
- `eval` — evaluate a checkpoint or the latest trained actor for the selected algorithm
- `pipeline` — run `data-train -> data-eval -> sft -> rl -> eval`

## Common Flags

- `--algorithm hero|eif`
- `--debug`
- `--run-name`
- `--artifact-root`
- `--work-dir`
- `--model-path`
- `--data-dir`
- `--eval-dir`
- `--train-output-dir`
- `--eval-output-dir`
- `--gpu-profile`
- `--regime`
- `--filter-tbr`
- `--no-sft`

`--debug` is only supported for:

- `data-train --debug`
  Sets smaller defaults for source sampling and response length.
- `data-eval --debug`
  Builds only the `amc` benchmark.

## Quick Start

```bash
# HERO
bash examples/hybrid_reward/run.sh pipeline --algorithm hero

# EIF
export NAUTILUS_API_KEY=<your-key>
bash examples/hybrid_reward/run.sh pipeline --algorithm eif
```

## Preprocess Once, Train Both

```bash
# 1. Build shared train/eval artifacts
bash examples/hybrid_reward/run.sh data-train
bash examples/hybrid_reward/run.sh data-eval
bash examples/hybrid_reward/run.sh sft

# 2. Train HERO
bash examples/hybrid_reward/run.sh rl --algorithm hero

# 3. Train EIF
export NAUTILUS_API_KEY=<your-key>
bash examples/hybrid_reward/run.sh rl --algorithm eif
```

## Directory Layout

Typical shared work directory:

```text
<work_dir>/
├── source_generation/
├── openmathreasoning_hero/
├── hero_eval/
├── checkpoints/
│   ├── cold_start_sft/
│   ├── hero_rl/
│   └── eif_rl/
└── eval_results/
    ├── hero/
    └── eif/
```

## Internals

Implementation backends remain available for advanced/manual use:

- `run_data_pipeline.sh`
- `run_cold_start_sft.sh`
- `run_train.sh`
- `run_eval.sh`
- `run_pipeline.sh`
- `stage_lib.sh`

The step-by-step semantic wrappers live under `step_by_step/`. See
`step_by_step/README.md` for manual stage-by-stage usage.

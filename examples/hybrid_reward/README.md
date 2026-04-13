# Hybrid Reward Training Pipeline

Training pipeline for hybrid-reward GRPO algorithms on math reasoning tasks.
Supports two reward strategies:

- **HERO** (`ALGORITHM=hero`) — Reward shaping with stratified RM normalization (arXiv:2510.07242v1)
- **EIF** (`ALGORITHM=eif`) — One-step estimator with online tau/m LLM regressors

Both algorithms share identical data preprocessing (stages 1-6) and evaluation
protocols, differing only in the reward manager configuration at training time.

## Architecture

```
examples/hybrid_reward/
  run_pipeline.sh              End-to-end: data prep + train + eval
  run_train.sh                 GRPO training (ALGORITHM=hero|eif)
  run_eval.sh                  Evaluation (math_verify + LLM judge)
  run_data_pipeline.sh         Stages 1-6 (data preprocessing)
  run_cold_start_sft.sh        Stage 6 alone (cold-start SFT)
  eval_llm_judge.py            LLM-as-judge scorer
  baseline_reward_fn.py        Verifier scorers for baselines
  run_baseline_rm_only.sh      Baseline: dense RM only
  run_baseline_verifier_only.sh  Baseline: verifier only
  run_baseline_naive_combine.sh  Baseline: linear mix
  step_by_step/                Individual stage scripts
```

## Quick Start

```bash
# HERO (default)
ALGORITHM=hero bash examples/hybrid_reward/run_pipeline.sh

# EIF
export NAUTILUS_API_KEY=<your-key>
ALGORITHM=eif bash examples/hybrid_reward/run_pipeline.sh
```

## Preprocess Once, Train Both

```bash
# 1. Run shared preprocessing
bash examples/hybrid_reward/run_data_pipeline.sh

# 2. Train HERO
ALGORITHM=hero \
DATA_DIR=$RL_PIPELINE_DATA_DIR \
EVAL_DIR=$RL_PIPELINE_EVAL_DIR \
MODEL_PATH=$RL_PIPELINE_TRAIN_MODEL_PATH \
bash examples/hybrid_reward/run_train.sh

# 3. Train EIF (can run in parallel)
export NAUTILUS_API_KEY=<your-key>
ALGORITHM=eif \
DATA_DIR=$RL_PIPELINE_DATA_DIR \
EVAL_DIR=$RL_PIPELINE_EVAL_DIR \
MODEL_PATH=$RL_PIPELINE_TRAIN_MODEL_PATH \
bash examples/hybrid_reward/run_train.sh
```

## Configuration

All data/eval variables use the `RL_PIPELINE_*` prefix.
Training variables use short names (`MODEL_PATH`, `GPUS_PER_NODE`, etc.).
EIF-specific tau/m LLM vars use the `EIF_*` prefix.

See `step_by_step/README.md` for step-by-step usage and GPU profiles.

# HERO: Hybrid Ensemble Reward Optimization

Last updated: 03/03/2026.

This page documents a `verl` implementation of **HERO** from:
- *Hybrid Reinforcement: When Reward Is Sparse, It's Better to Be Dense* (arXiv:2510.07242v1)

## Overview

HERO combines:
- a verifier-side binary reward (`rule score`)
- a dense reward-model score (`RM score`)

and applies two shaping steps per prompt group (`uid`):

1. **Stratified normalization** (paper Eq. 3): normalize RM scores *within* verifier strata (`rule=0` and `rule=1`) and map them to bounded intervals.
2. **Variance-aware weighting** (paper Eq. 4/5): upweight prompt groups with higher RM disagreement.

## Where It Runs In `verl`

- Signal extraction (per sample): `verl/experimental/reward_loop/reward_manager/hero.py`
- Group-level shaping (per batch, grouped by `uid`): `verl/experimental/reward_loop/reward_loop.py`
- Core shaping math: `verl/utils/reward_score/hero.py`

## Configuration

Enable HERO reward manager and a dense reward model:

```bash
reward.reward_manager.name=hero
reward.reward_model.enable=True
reward.reward_model.model_path=<your_dense_rm_path>
```

HERO hyperparameters are under `reward.reward_kwargs.hero`:

```yaml
reward:
  reward_kwargs:
    hero:
      alpha: 0.1
      beta: 0.1
      eps: 1e-6
      w_min: 0.4
      w_max: 3.0
      k: 6.0
      sigma_ema: 0.9
```

Notes:
- `alpha`, `beta` control stratified reward ranges.
- `w_min`, `w_max`, `k` control logistic difficulty reweighting.
- `sigma_ema` smooths the running `sigma_bar` used by the weighting function.

## Dataset Preparation

Use ``examples/data_preprocess/openmathreasoning_hero.py`` to build
verifiable / hard-to-verify / mixed splits from either:
- a Hugging Face dataset repo via `--dataset`
- a local dataset path via `--local_dataset_path`

The script still writes parquet files for `verl` training, and can now also:
- save a processed Hugging Face `DatasetDict` via `--hf_save_dir`
- upload that `DatasetDict` via `--push_to_hub_repo`

Example:

```bash
python examples/data_preprocess/openmathreasoning_hero.py \
  --dataset your-org/your-openmath \
  --dataset_config main \
  --split train \
  --question_col question \
  --answer_col answer \
  --candidate_col candidate_solution \
  --local_save_dir ~/data/openmathreasoning_hero \
  --hf_save_dir ~/data/openmathreasoning_hero_hf
```

End-to-end preprocessing + training:

```bash
bash examples/grpo_trainer/run_qwen3_4b_hero_from_hf.sh \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1
```

Useful environment variables for the wrapper:
- `HERO_DATASET` / `HERO_DATASET_CONFIG`
- `HERO_LOCAL_DATASET_PATH`
- `HERO_QUESTION_COL` / `HERO_ANSWER_COL` / `HERO_CANDIDATE_COL`
- `HERO_LOCAL_SAVE_DIR`
- `HERO_HF_SAVE_DIR`
- `HERO_PUSH_TO_HUB_REPO`
- `HERO_SKIP_PREPROCESS=1` to reuse existing parquet splits

## Output Diagnostics

When HERO is enabled, reward loop emits extra fields:
- `acc` (verifier/rule score)
- `hero_rule_score`
- `hero_rm_score`
- `hero_stratified_score`
- `hero_difficulty_weight`
- `hero_group_sigma`
- `hero_sigma_bar`
- `hero_final_score`

These fields are available through the existing reward extra-info logging path.

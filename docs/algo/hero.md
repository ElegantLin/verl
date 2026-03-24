# HERO: Hybrid Ensemble Reward Optimization

Last updated: 03/23/2026.

This page documents a `verl` reproduction path for **HERO** from:
- *Hybrid Reinforcement: When Reward Is Sparse, It's Better to Be Dense* (arXiv:2510.07242v1)

## Overview

HERO combines:
- a verifier-side binary reward (`rule score`)
- a dense reward-model score (`RM score`)

and applies two shaping steps per prompt group (`uid`):

1. **Stratified normalization** (paper Eq. 3): normalize RM scores within verifier strata (`rule=0` and `rule=1`) and map them to bounded intervals.
2. **Variance-aware weighting** (paper Eq. 4/5): upweight prompt groups with higher RM disagreement.

## Where It Runs In `verl`

- Signal extraction: `verl/experimental/reward_loop/reward_manager/hero.py`
- Group-level shaping: `verl/experimental/reward_loop/reward_loop.py`
- Core shaping math: `verl/utils/reward_score/hero.py`

## Paper-Aligned Defaults

For the Qwen3-4B setting in Table 6:
- prompt length: `1024`
- response length: `8192`
- PPO learning rate: `1e-6`
- rollout samples per prompt: `8`
- rollout temperature: `1.0`
- no KL loss
- no entropy bonus
- dense reward model: `nvidia/AceMath-7B-RM`
- verifier: `math_verify`

Appendix A.1 hyperparameters:
- verifiable-only: `alpha=0.05`, `beta=0.05`
- mixed / hard-to-verify: `alpha=0.1`, `beta=0.1`
- `w_min=0.4`, `w_max=3.0`, `k=6.0`

## Data Preparation

The paper uses OpenMathReasoning with `problem_type == has_answer_extracted`, samples `40k` source problems, generates candidate solutions, then builds:
- `train_verifiable.parquet`
- `train_hard_to_verify.parquet`
- `train_mixed.parquet`
- matching validation splits

Use the following helpers:
- `examples/data_preprocess/openmathreasoning_hero_source.py`: build prompt-only source parquet for candidate generation
- `examples/data_preprocess/openmathreasoning_hero.py`: build HERO RL splits from generated `responses`
- `examples/data_preprocess/openmathreasoning_hero_sft.py`: build the 2k cold-start SFT set from generated responses
- `examples/data_preprocess/hero_eval_benchmarks.py`: build evaluation parquet files
- `examples/data_preprocess/hero_filter_textbook_reasoning.py`: optional paper-style TBR filtering

## Training And Eval Wrappers

Wrappers added for the paper reproduction path:
- `examples/hero/run_hero_cold_start_sft.sh`
- `examples/hero/run_hero_train.sh`
- `examples/hero/run_hero_eval.sh`
- `examples/hero/run_hero_pipeline.sh`

`run_hero_train.sh` now exports Hugging Face checkpoints by default via:
- `actor_rollout_ref.actor.checkpoint.save_contents=['model','optimizer','extra','hf_model']`

This makes the latest actor checkpoint directly usable by evaluation.

## Evaluation Protocol

Paper-aligned evaluation details implemented in this repo:
- decoding uses `N=8`, `temperature=0.6`, `top_p=0.95`
- verifiable benchmarks are scored by `math_verify`
- hard-to-verify benchmarks are scored by LLM-as-judge
- only the first decoded response is scored for pass@1 reproduction

The first-response selection is handled by:
- `verl/trainer/main_eval.py`
- `examples/hero/eval_hero_llm_judge.py`
- `examples/hero/run_hero_eval.sh`

## One-Command Pipeline

End-to-end reproduction from one bash entrypoint:

```bash
bash examples/hero/run_hero_pipeline.sh
```

Useful environment variables:
- `HERO_MODEL_PATH` or `HERO_BASE_MODEL_PATH`
- `HERO_LOCAL_DATASET_PATH` if OpenMathReasoning is already local
- `HERO_HVM_LOCAL_PATH` / `HERO_TBR_LOCAL_PATH` for local hard-to-verify datasets
- `HERO_FILTER_TBR=1` to enable the optional paper-style TBR filtering stage
- `HERO_ENABLE_COLD_START_SFT=0` to skip cold-start SFT
- `HERO_TRAIN_OUTPUT_DIR`, `HERO_SFT_OUTPUT_DIR`, `HERO_EVAL_OUTPUT_DIR` to control artifact locations

## Output Diagnostics

When HERO is enabled, reward loop emits:
- `acc`
- `hero_rule_score`
- `hero_rm_score`
- `hero_stratified_score`
- `hero_difficulty_weight`
- `hero_group_sigma`
- `hero_sigma_bar`
- `hero_final_score`

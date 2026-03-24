# HERO Step-by-Step Bash Scripts

These scripts split the HERO pipeline into ordered stages you can run one by one.

Recommended setup:

```bash
export HERO_RUN_NAME=my_hero_run
export HERO_GPU_PROFILE=8x24gb   # or 4x80gb
export HERO_MODEL_PATH=Qwen/Qwen3-4B-Base
```

If you already have OpenMathReasoning locally, also set:

```bash
export HERO_LOCAL_DATASET_PATH=/path/to/openmathreasoning
```

Run in order:

```bash
bash examples/hero/step_by_step/01_build_source_prompts.sh
bash examples/hero/step_by_step/02_generate_source_candidates.sh
bash examples/hero/step_by_step/03_build_rl_data.sh
bash examples/hero/step_by_step/05_build_eval_benchmarks.sh
bash examples/hero/step_by_step/07_run_rl_train.sh
bash examples/hero/step_by_step/08_run_final_eval.sh
```

Optional stages:

```bash
# Optional: paper-style TBR filtering before building eval benchmarks.
bash examples/hero/step_by_step/04_optional_filter_tbr.sh

# Optional: cold-start SFT before RL training.
bash examples/hero/step_by_step/06_optional_cold_start_sft.sh
```

Notes:

- `HERO_GPU_PROFILE=8x24gb` uses safer defaults for 8 x 24GB GPUs.
- `HERO_GPU_PROFILE=4x80gb` uses larger defaults for 4 x 80GB GPUs.
- You can still override any underlying env var, such as `HERO_MAX_RESPONSE_LENGTH`, `HERO_ROLLOUT_N`, `HERO_RM_TP_SIZE`, or `HERO_REGIME`.
- Hard-to-verify evaluation uses the OpenAI-compatible judge defaults already wired into `examples/hero/run_hero_eval.sh`, so set `OPENAI_API_KEY` before `08_run_final_eval.sh` if you evaluate `hardverify_math` or `textbook_reasoning`.

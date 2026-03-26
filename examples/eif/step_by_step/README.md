# EIF Step-by-Step Bash Scripts

These scripts split the EIF pipeline into ordered stages you can run one by one.

Recommended setup:

```bash
export EIF_RUN_NAME=my_eif_run
export EIF_GPU_PROFILE=8x24gb   # or 4x80gb
export EIF_MODEL_PATH=Qwen/Qwen3-4B-Base
export OPENAI_API_KEY=<your-key>  # required for tau/m LLM endpoints
```

If you already have OpenMathReasoning locally, also set:

```bash
export EIF_LOCAL_DATASET_PATH=/path/to/openmathreasoning
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
# Optional: paper-style TBR filtering before building eval benchmarks.
bash examples/eif/step_by_step/04_optional_filter_tbr.sh

# Optional: cold-start SFT before RL training.
bash examples/eif/step_by_step/06_optional_cold_start_sft.sh
```

Notes:

- `EIF_GPU_PROFILE=8x24gb` uses safer defaults for 8 x 24GB GPUs.
- `EIF_GPU_PROFILE=4x80gb` uses larger defaults for 4 x 80GB GPUs.
- You can still override any underlying env var, such as `EIF_MAX_RESPONSE_LENGTH`, `EIF_ROLLOUT_N`, `EIF_RM_TP_SIZE`, or `EIF_REGIME`.
- EIF training requires tau/m LLM endpoints. Configure with `EIF_TAU_MODEL`, `EIF_TAU_BASE_URL`, `EIF_M_MODEL`, `EIF_M_BASE_URL`.
- Hard-to-verify evaluation reuses the HERO LLM judge (`examples/hero/eval_hero_llm_judge.py`), so set `OPENAI_API_KEY` before `08_run_final_eval.sh` if you evaluate `hardverify_math` or `textbook_reasoning`.

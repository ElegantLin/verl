# Unify HERO/EIF Data Preparation Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract shared data-prep (01–06), eval (08), and supporting scripts into `examples/shared/` so identical GPU work runs only once.

**Architecture:** Shell scripts that are identical between HERO and EIF move to `examples/shared/`. Each algorithm keeps only `common.sh` (training defaults) and `07_run_rl_train.sh`. Shared `common.sh` uses `DATA_PREP_*`/`EVAL_*`/`SFT_*` env var prefixes.

**Tech Stack:** Bash, Python (eval_llm_judge.py — move only, no changes)

**Spec:** `docs/superpowers/specs/2026-04-11-unify-data-prep-pipeline-design.md`

---

### Task 1: Create shared `common.sh`

**Files:**
- Create: `examples/shared/step_by_step/common.sh`

- [ ] **Step 1: Create directory and write shared common.sh**

Write the full `common.sh` as specified in the spec's "Shared variables" section. Include:
- Guard (`_SHARED_STEP_COMMON_SH`), `repo_root` resolution, helper functions (`set_default`, `resolve_latest_hf_dir`, `require_file`, `ensure_dirs`)
- GPU profile defaults for all three profiles (8x24gb, 4x80gb, 2x80gb) — data-prep, SFT, and eval vars only
- Artifact paths, model, dataset, generation, eval, TBR filter vars (all 10)
- Paths to shared scripts (`cold_start_sft_script`, `eval_script`, `judge_script`)

Source: spec lines 115–302 have the complete content.

- [ ] **Step 2: Verify syntax**

Run: `bash -n examples/shared/step_by_step/common.sh`
Expected: no output (clean parse)

- [ ] **Step 3: Commit**

```bash
git add examples/shared/step_by_step/common.sh
git commit -m "feat: add shared common.sh for unified data-prep pipeline"
```

---

### Task 2: Create shared step scripts 01, 02, 03, 05

These four scripts are identical between HERO and EIF — only need `ensure_hero_dirs` → `ensure_dirs` and source path changes.

**Files:**
- Create: `examples/shared/step_by_step/01_build_source_prompts.sh`
- Create: `examples/shared/step_by_step/02_generate_source_candidates.sh`
- Create: `examples/shared/step_by_step/03_build_rl_data.sh`
- Create: `examples/shared/step_by_step/05_build_eval_benchmarks.sh`
- Reference: `examples/hero/step_by_step/01_build_source_prompts.sh` (source to copy from)
- Reference: `examples/hero/step_by_step/02_generate_source_candidates.sh`
- Reference: `examples/hero/step_by_step/03_build_rl_data.sh`
- Reference: `examples/hero/step_by_step/05_build_eval_benchmarks.sh`

- [ ] **Step 1: Create all four scripts**

For each, copy from `examples/hero/step_by_step/` and make these changes:
1. Replace `source "$script_dir/common.sh"` with `source "$script_dir/common.sh"`
   (same line, but `script_dir` now resolves to `examples/shared/step_by_step/`)
2. Replace `# shellcheck source=examples/hero/step_by_step/common.sh` with `# shellcheck source=examples/shared/step_by_step/common.sh`
3. Replace `ensure_hero_dirs` with `ensure_dirs`

No other changes — all variables are already unprefixed.

- [ ] **Step 2: Verify syntax for all four**

Run: `for f in examples/shared/step_by_step/0{1,2,3}_*.sh examples/shared/step_by_step/05_*.sh; do bash -n "$f" && echo "OK: $f"; done`
Expected: four "OK" lines

- [ ] **Step 3: Commit**

```bash
git add examples/shared/step_by_step/0{1,2,3}_*.sh examples/shared/step_by_step/05_*.sh
git commit -m "feat: add shared step scripts 01, 02, 03, 05"
```

---

### Task 3: Move eval_llm_judge.py + create shared run_eval.sh and run_cold_start_sft.sh

**Files:**
- Create: `examples/shared/eval_llm_judge.py` (copy from `examples/hero/eval_hero_llm_judge.py`, no code changes)
- Create: `examples/shared/run_eval.sh` (from `examples/hero/run_hero_eval.sh`, prefix `HERO_*` → `EVAL_*`)
- Create: `examples/shared/run_cold_start_sft.sh` (from `examples/hero/run_hero_cold_start_sft.sh`, prefix `HERO_*` → `SFT_*`)

- [ ] **Step 1: Copy eval_llm_judge.py**

Copy `examples/hero/eval_hero_llm_judge.py` to `examples/shared/eval_llm_judge.py`. No code changes — it uses CLI arguments only.

- [ ] **Step 2: Create shared run_eval.sh**

Copy `examples/hero/run_hero_eval.sh` and apply these changes:
1. All `HERO_*` env var references → `EVAL_*` (e.g., `HERO_MODEL_PATH` → `EVAL_MODEL_PATH`, `HERO_EVAL_BENCHMARKS` → `EVAL_BENCHMARKS`)
2. Replace `HERO_JUDGE_*` → `EVAL_JUDGE_*`
3. Line 168: `python3 examples/hero/eval_hero_llm_judge.py` → resolve path dynamically:
   ```bash
   script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
   judge_script="$script_dir/eval_llm_judge.py"
   ```
   Then use `python3 "$judge_script"` instead of hardcoded path.
4. Update header comments to say "Shared evaluation script" instead of "HERO evaluation script"
5. Default `output_dir` path: `$HOME/data/eval_results/...` (drop `hero_` prefix)
6. `trainer.project_name=rl_eval` (generic)

- [ ] **Step 3: Create shared run_cold_start_sft.sh**

Copy `examples/hero/run_hero_cold_start_sft.sh` (the superset) and apply:
1. All `HERO_*` env var references → `SFT_*` (e.g., `HERO_SFT_INPUT_PATH` → `SFT_INPUT_PATH`, `HERO_MODEL_PATH` → `SFT_MODEL_PATH`)
2. `HERO_SEED` → `SFT_SEED`
3. `HERO_SFT_NODE_RANK` → `SFT_NODE_RANK`, `HERO_SFT_MASTER_ADDR` → `SFT_MASTER_ADDR`, `HERO_SFT_MASTER_PORT` → `SFT_MASTER_PORT`
4. Update header comment
5. `trainer.project_name=cold_start_sft` (generic)

This is the HERO superset — includes `lr_scheduler_type`, `lr_warmup_steps_ratio`, `min_lr_ratio`, `attn_implementation` with fallback, `data.ignore_input_ids_mismatch=True`, `+model.override_config.attn_implementation`.

- [ ] **Step 4: Verify syntax**

Run: `bash -n examples/shared/run_eval.sh && bash -n examples/shared/run_cold_start_sft.sh && echo OK`
Expected: "OK"

- [ ] **Step 5: Commit**

```bash
git add examples/shared/eval_llm_judge.py examples/shared/run_eval.sh examples/shared/run_cold_start_sft.sh
git commit -m "feat: add shared eval and SFT scripts"
```

---

### Task 4: Create shared step 04 (TBR), 06 (SFT wrapper), 08 (eval wrapper)

**Files:**
- Create: `examples/shared/step_by_step/04_optional_filter_tbr.sh`
- Create: `examples/shared/step_by_step/06_optional_cold_start_sft.sh`
- Create: `examples/shared/step_by_step/08_run_final_eval.sh`
- Reference: `examples/hero/step_by_step/04_optional_filter_tbr.sh`
- Reference: `examples/hero/step_by_step/06_optional_cold_start_sft.sh`
- Reference: `examples/hero/step_by_step/08_run_final_eval.sh`

- [ ] **Step 1: Create shared step 04**

Copy from hero version. Changes:
1. `ensure_hero_dirs` → `ensure_dirs`
2. shellcheck source path → shared
3. Replace all 10 `HERO_*` TBR env var checks with unprefixed shell vars from `common.sh`:
   - `"${HERO_DISABLE_TBR_MATH_VERIFY_FILTER:-0}"` → `"$disable_tbr_math_verify_filter"`
   - `"${HERO_TBR_ANSWER_MODEL:-}"` → `"$tbr_answer_model"`
   - `"${HERO_TBR_ANSWER_BASE_URL:-}"` → `"$tbr_answer_base_url"`
   - `"${HERO_TBR_ANSWER_API_KEY_ENV:-}"` → `"$tbr_answer_api_key_env"`
   - `"${HERO_TBR_ANSWER_JUDGE_MODEL:-}"` → `"$tbr_answer_judge_model"`
   - `"${HERO_TBR_ANSWER_JUDGE_BASE_URL:-}"` → `"$tbr_answer_judge_base_url"`
   - `"${HERO_TBR_ANSWER_JUDGE_API_KEY_ENV:-}"` → `"$tbr_answer_judge_api_key_env"`
   - `"${HERO_TBR_SUITABILITY_MODEL:-}"` → `"$tbr_suitability_model"`
   - `"${HERO_TBR_SUITABILITY_BASE_URL:-}"` → `"$tbr_suitability_base_url"`
   - `"${HERO_TBR_SUITABILITY_API_KEY_ENV:-}"` → `"$tbr_suitability_api_key_env"`

- [ ] **Step 2: Create shared step 06**

Copy from hero version. Changes:
1. `ensure_hero_dirs` → `ensure_dirs`
2. shellcheck source path → shared
3. `sft_model_path=${HERO_MODEL_PATH:-$base_model_path}` → `sft_model_path=${SFT_MODEL_PATH:-$base_model_path}`
4. Replace the env-var-prefixed call block:
   ```bash
   SFT_INPUT_PATH="$source_generated_path" \
   SFT_DATA_DIR="$sft_data_dir" \
   SFT_OUTPUT_DIR="$sft_output_dir" \
   SFT_MODEL_PATH="$sft_model_path" \
   SFT_TRUST_REMOTE_CODE="$trust_remote_code" \
   bash "$cold_start_sft_script" "$@"
   ```

- [ ] **Step 3: Create shared step 08**

Copy from hero version. Changes:
1. `ensure_hero_dirs` → `ensure_dirs`
2. shellcheck source path → shared
3. `eval_model_path=${HERO_MODEL_PATH:-}` → `eval_model_path=${EVAL_MODEL_PATH:-}`
4. Error message: generic wording (drop "HERO" / "EIF")
5. Replace the env-var-prefixed call block:
   ```bash
   EVAL_MODEL_PATH="$eval_model_path" \
   EVAL_TRUST_REMOTE_CODE="$trust_remote_code" \
   EVAL_EVAL_DIR="$eval_dir" \
   EVAL_OUTPUT_DIR="$eval_output_dir" \
   bash "$eval_script" "$@"
   ```

- [ ] **Step 4: Verify syntax**

Run: `for f in examples/shared/step_by_step/0{4,6}_*.sh examples/shared/step_by_step/08_*.sh; do bash -n "$f" && echo "OK: $f"; done`
Expected: three "OK" lines

- [ ] **Step 5: Commit**

```bash
git add examples/shared/step_by_step/0{4,6}_*.sh examples/shared/step_by_step/08_*.sh
git commit -m "feat: add shared step scripts 04, 06, 08"
```

---

### Task 5: Update hero/eif common.sh + delete old files

**Files:**
- Modify: `examples/hero/step_by_step/common.sh`
- Modify: `examples/eif/step_by_step/common.sh`
- Delete: `examples/hero/step_by_step/01–06, 08` (7 files)
- Delete: `examples/eif/step_by_step/01–06, 08` (7 files)
- Delete: `examples/hero/run_hero_eval.sh`
- Delete: `examples/hero/run_hero_cold_start_sft.sh`
- Delete: `examples/hero/eval_hero_llm_judge.py`
- Delete: `examples/eif/run_eif_eval.sh`
- Delete: `examples/eif/run_eif_cold_start_sft.sh`

- [ ] **Step 1: Rewrite hero common.sh**

Replace the full content with a slim version that:
1. Sources shared `common.sh`: `source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../shared/step_by_step" && pwd)/common.sh"`
2. Sets HERO-specific training output paths (`train_output_dir`, `eval_output_dir`)
3. Defines `ensure_hero_dirs() { ensure_dirs; mkdir -p "$train_output_dir" "$eval_output_dir"; }`
4. Has GPU profile `case` block with HERO training-only vars (all `HERO_PPO_*`, `HERO_ROLLOUT_*`, `HERO_RM_*`, `HERO_TRAIN_*` etc.) for all three profiles

Source: spec lines 307–348 have the structure. Copy the full training variable set from the current `examples/hero/step_by_step/common.sh` lines 52–163.

- [ ] **Step 2: Rewrite eif common.sh**

Same as step 1 but with `EIF_*` prefix and EIF's 8x24gb defaults:
- `EIF_PPO_MICRO_BATCH_SIZE_PER_GPU 2` (HERO: 1)
- `EIF_ROLLOUT_GPU_MEMORY_UTILIZATION 0.5` (HERO: 0.35)
- `EIF_RM_GPU_MEMORY_UTILIZATION 0.5` (HERO: 0.3)
- No `EIF_PPO_MAX_TOKEN_LEN_PER_GPU` (HERO has 6144)

- [ ] **Step 3: Verify both source the shared common.sh correctly**

Run: `bash -n examples/hero/step_by_step/common.sh && bash -n examples/eif/step_by_step/common.sh && echo OK`
Expected: "OK"

- [ ] **Step 4: Delete old hero step scripts (01–06, 08)**

```bash
git rm examples/hero/step_by_step/0{1,2,3,4,5,6}_*.sh examples/hero/step_by_step/08_*.sh
```

- [ ] **Step 5: Delete old eif step scripts (01–06, 08)**

```bash
git rm examples/eif/step_by_step/0{1,2,3,4,5,6}_*.sh examples/eif/step_by_step/08_*.sh
```

- [ ] **Step 6: Delete old eval/SFT/judge scripts**

```bash
git rm examples/hero/run_hero_eval.sh examples/hero/run_hero_cold_start_sft.sh examples/hero/eval_hero_llm_judge.py
git rm examples/eif/run_eif_eval.sh examples/eif/run_eif_cold_start_sft.sh
```

- [ ] **Step 7: Commit**

```bash
git add examples/hero/step_by_step/common.sh examples/eif/step_by_step/common.sh
git commit -m "refactor: slim hero/eif to training-only, delete duplicated scripts"
```

---

### Task 6: Update pipeline scripts + READMEs

**Files:**
- Modify: `examples/hero/run_hero_pipeline.sh`
- Modify: `examples/eif/run_eif_pipeline.sh`
- Create: `examples/shared/step_by_step/README.md`
- Modify: `examples/hero/step_by_step/README.md`
- Modify: `examples/eif/step_by_step/README.md`

- [ ] **Step 1: Update hero pipeline script**

In `examples/hero/run_hero_pipeline.sh`:
1. Line ~204: `bash examples/hero/run_hero_cold_start_sft.sh` → `bash examples/shared/run_cold_start_sft.sh`
2. Line ~204: Env var prefix for SFT: `HERO_SFT_INPUT_PATH` → `SFT_INPUT_PATH`, etc.
3. Line ~212: `bash examples/hero/run_hero_eval.sh` → `bash examples/shared/run_eval.sh`
4. Line ~212: Env var prefix for eval: `HERO_MODEL_PATH` → `EVAL_MODEL_PATH`, `HERO_EVAL_DIR` → `EVAL_EVAL_DIR`, `HERO_OUTPUT_DIR` → `EVAL_OUTPUT_DIR`

- [ ] **Step 2: Update eif pipeline script**

Same changes as step 1 but for `examples/eif/run_eif_pipeline.sh`.

- [ ] **Step 3: Create shared README**

Create `examples/shared/step_by_step/README.md` with:
- Title: "Shared Data Preparation & Evaluation Pipeline"
- Brief explanation that these scripts are shared by HERO and EIF
- User workflow example (the `DATA_PREP_*` export + bash commands from the spec)
- Stage table (01–06, 08 with descriptions)
- Note about GPU profile tuning tips for step 02
- Note about `NAUTILUS_API_KEY` for hard-to-verify eval

- [ ] **Step 4: Update hero README**

Rewrite `examples/hero/step_by_step/README.md`:
- Data prep: "Run shared data-prep scripts first — see `examples/shared/step_by_step/README.md`"
- Only step 07 documented inline
- Show how to point HERO training at shared data (`export HERO_DATA_DIR=...`)

- [ ] **Step 5: Update eif README**

Same as step 4 but for EIF, plus EIF-specific notes:
- `NAUTILUS_API_KEY` for tau/m LLM endpoints
- `EIF_TAU_MODEL`, `EIF_M_MODEL` configuration

- [ ] **Step 6: Verify all scripts parse cleanly**

Run: `find examples/shared examples/hero/step_by_step examples/eif/step_by_step -name '*.sh' -exec bash -n {} \; -print`
Expected: all files listed with no errors

- [ ] **Step 7: Commit**

```bash
git add examples/hero/run_hero_pipeline.sh examples/eif/run_eif_pipeline.sh
git add examples/shared/step_by_step/README.md examples/hero/step_by_step/README.md examples/eif/step_by_step/README.md
git commit -m "docs: update READMEs and pipeline scripts for shared data-prep"
```

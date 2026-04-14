# Simplify HERO/EIF Hybrid Reward Structure

**Date:** 2026-04-14
**Status:** Draft
**Goal:** Reduce duplicated HERO/EIF reward-loop plumbing and clean up outdated entrypoint references without changing algorithm semantics.

## Background

The repository has already converged on a unified user-facing pipeline under
`examples/hybrid_reward/`, where data preparation and evaluation are shared and
training switches behavior through `ALGORITHM=hero|eif`.

The remaining duplication is now mostly structural:

- rule-side reward managers for `hero`, `hybrid_eif`, and `hybrid_eif_online`
  repeat nearly the same `run_single()` workflow
- `RewardLoopManager.compute_rm_score()` still contains a long algorithm branch
  block that mixes orchestration and algorithm-specific postprocessing
- tests and docs still reference removed paths such as `examples/hero/`,
  `examples/eif/`, and `examples/hybrid_eif/`

The intended semantics are:

- **Training default:** EIF uses `hybrid_eif_online`
- **Evaluation / precomputed data:** EIF uses `hybrid_eif`

This distinction is already reflected in the current config and training script
and should become an explicit design rule for the simplification.

## Scope

**In scope:**

- Extract shared single-sample rule-side execution logic from HERO / EIF reward managers
- Split reward-loop postprocessing into clearer algorithm-specific helpers
- Update stale tests and docs to point at `examples/hybrid_reward/`
- Align docs with current online/offline EIF semantics

**Out of scope:**

- Changing HERO shaping math
- Changing offline EIF math (`m + phi - tau`)
- Changing online EIF math or fail-open behavior
- Renaming public reward manager keys
- Removing legacy launchers under `examples/grpo_trainer/`
- Reworking the unified shell pipeline structure that already landed in
  `examples/hybrid_reward/`

## Definitions

### EIF offline

`hybrid_eif` means the batch already contains precomputed scalar `tau` and `m`
values. The reward loop reads those columns and computes the final score without
making online tau/m requests.

Primary use cases:

- evaluation with fixed responses
- precomputed datasets
- any workflow where `tau_llm_value` and `m_llm_value` are already materialized

### EIF online

`hybrid_eif_online` means the reward loop only starts with verifier-side `phi`
and computes the auxiliary AceMath / tau / m signals online during reward
aggregation.

Primary use case:

- training, where responses are newly generated and auxiliary signals are not
  precomputed

## Design

### 1. Keep algorithm identities, unify plumbing

Do not collapse HERO, offline EIF, and online EIF into one abstract "hybrid"
strategy type. The algorithms have materially different postprocessing,
different state requirements, and different failure modes.

Instead:

- keep `HeroRewardManager`
- keep `HybridEIFRewardManager`
- keep `HybridEIFOnlineRewardManager`
- move their duplicated rule-side execution skeleton into shared reward-manager
  plumbing

This gives the codebase one obvious place for common work while preserving
algorithm-local semantics and diagnostics.

### 2. Shared rule-side execution skeleton

The shared skeleton should cover the existing common workflow:

1. assert the input is a single-item `DataProto`
2. decode the response text from token ids
3. collect `data_source`, `ground_truth`, `extra_info`,
   `tool_extra_fields`, `__num_turns__`, and `reward_scores`
4. invoke `compute_score` with sync/async support
5. normalize the result into
   `{"reward_score": float, "reward_extra_info": dict}`
6. preserve `acc`

The base helper should not know which algorithm-specific diagnostic key to add.
That stays in subclasses.

### 3. Minimal subclass-specific manager behavior

After the shared helper returns the normalized rule-side score, each subclass
adds only the fields that encode its public semantics:

- HERO: `hero_rule_score`
- offline EIF: `hybrid_eif_phi`
- online EIF: `hybrid_eif_online_phi`

This keeps downstream logs and analysis stable while removing duplicated decode
and dispatch code.

### 4. Split reward-loop postprocessing helpers

Refactor `RewardLoopManager.compute_rm_score()` so the top-level method remains
an orchestrator and algorithm-specific postprocessing lives in helpers:

- `_apply_hero_scores(...)`
- `_apply_hybrid_eif_scores(...)`
- `_apply_hybrid_eif_online_scores(...)`

Responsibilities stay unchanged:

- HERO helper:
  - read rule + dense RM scores
  - apply `apply_hero_shaping(...)`
  - emit HERO diagnostics
- offline EIF helper:
  - read `tau_key` / `m_key`
  - compute `m + phi - tau`
  - emit offline EIF diagnostics
- online EIF helper:
  - initialize online scoring components
  - request AceMath / tau / m online
  - preserve existing fail-open-to-phi behavior

### 5. Explicit online/offline product rule

The refactor should make the repository convention explicit:

- `examples/hybrid_reward/run_train.sh` with `ALGORITHM=eif` continues to map to
  `reward.reward_manager.name=hybrid_eif_online`
- docs treat offline EIF as the evaluation / precomputed-data path
- docs treat online EIF as the training path

No user-facing config rename is needed. The change is clarity, not surface-area
expansion.

### 6. Entrypoint and documentation cleanup

Update stale references so the docs and tests reflect the repository as it
exists today:

- tests should validate `examples/hybrid_reward/` instead of removed
  `examples/hero/`, `examples/eif/`, or `examples/hybrid_eif/`
- offline EIF docs should use the current `tau_key` / `m_key` terminology
- docs should state that training defaults to online EIF

Legacy launchers under `examples/grpo_trainer/` remain in place and are treated
as compatibility entrypoints, not primary documentation targets.

### 7. Single-entry CLI structure must stay decomposed

The new shell entrypoint should be implemented as one public script with
subcommands, but it must not become one oversized procedural function.

Shell structure requirements:

- keep one public entrypoint script
- parse global arguments in a dedicated helper
- parse subcommand-specific arguments in dedicated helpers
- implement each subcommand as a small function with one responsibility
- move repeated environment/export setup into shared helpers
- move repeated path resolution into shared helpers
- keep orchestration separate from stage execution

Preferred internal shape:

- `main()`
- `parse_global_args()`
- `cmd_data()`
- `cmd_sft()`
- `cmd_train()`
- `cmd_eval()`
- `cmd_pipeline()`
- small shared helpers such as:
  - `resolve_artifact_paths()`
  - `resolve_train_model_path()`
  - `export_pipeline_env()`
  - `run_shared_data_stage()`
  - `run_rl_train_stage()`
  - `run_eval_stage()`

The goal is not a strict line-count rule, but each function should stay easy to
read in isolation and should avoid mixing:

- CLI parsing
- default resolution
- environment mutation
- external command execution
- summary printing

If a function starts owning more than one of those responsibilities, split it.

## File Plan

### Core code

- Modify: `verl/experimental/reward_loop/reward_manager/base.py`
- Modify: `verl/experimental/reward_loop/reward_manager/hero.py`
- Modify: `verl/experimental/reward_loop/reward_manager/hybrid_eif.py`
- Modify: `verl/experimental/reward_loop/reward_manager/hybrid_eif_online.py`
- Modify: `verl/experimental/reward_loop/reward_loop.py`

### Tests

- Modify: `tests/examples/test_hero_baseline_scripts_on_cpu.py`
- Modify: `tests/examples/test_eif_source_prompt_defaults_on_cpu.py`
- Modify: `tests/examples/test_hybrid_eif_eval_script_on_cpu.py`
- Add or modify targeted reward-loop unit tests covering the extracted shared
  manager skeleton and postprocessing helpers

### Docs

- Modify: `docs/algo/hybrid_reward_eif.md`
- Modify any nearby docs that still describe removed Hero / EIF shell entrypoints
  as primary paths
- Document the new single-entry subcommand CLI and its supported flags

## Compatibility Requirements

- Keep registered reward manager names unchanged:
  - `hero`
  - `hybrid_eif`
  - `hybrid_eif_online`
- Keep existing diagnostics field names unless a test proves they are already
  dead
- Keep `acc` semantics unchanged
- Keep `reward.yaml` public keys unchanged

## Testing Strategy

### Reward manager tests

Add coverage for the extracted shared manager skeleton:

- sync `compute_score`
- async `compute_score`
- propagation of `extra_info`
- propagation of `tool_extra_fields`
- propagation of rollout metadata used today

Then verify each concrete manager still emits its algorithm-specific key and
returns the same primary score as before.

### Reward loop tests

Cover helper behavior without changing semantics:

- HERO helper preserves `apply_hero_shaping(...)` output and diagnostics
- offline EIF helper preserves key-loading and missing-key error behavior
- online EIF helper preserves fail-open fallback behavior

### Entrypoint/docs regression tests

Update the current examples tests to validate the unified shell layout and
current config names.

Add shell-oriented coverage for the new CLI shape where practical:

- subcommand dispatch selects the expected internal stage
- `pipeline` excludes SFT by default
- `train --algorithm eif` still maps to online EIF
- helper-driven defaults preserve the existing artifact layout

## Risks

### Over-abstraction

Trying to unify HERO and EIF postprocessing behind a single generic strategy
interface would hide meaningful algorithm differences and make failures harder
to debug. Avoid this.

### Silent logging regressions

Downstream analysis may depend on current diagnostic keys. Preserve names unless
there is a strong reason and a migration plan.

### False "cleanup-only" change

To remain worthwhile under repository contribution rules, this refactor must
ship both:

- substantive structural deduplication in reward-manager / reward-loop code
- cleanup of stale tests/docs as part of the same change

It should not be submitted as documentation-only or path-fix-only busywork.

## Success Criteria

The simplification is successful if all of the following are true:

- there is one shared implementation path for rule-side single-sample reward
  execution
- reward-loop postprocessing is split into readable algorithm helpers
- `ALGORITHM=eif` still trains with online EIF
- offline EIF still works for evaluation / precomputed data
- no stale tests or docs point at removed primary example directories
- existing reward semantics and diagnostics remain unchanged

# Hybrid Reward EIF Estimator

Last updated: 03/11/2026.

This page documents an implementation of the one-step EIF estimator from:
- *Hybrid Reward* (Feb 28, 2026)

## Goal

Estimate:
- `theta = E[phi(Y, G)]`

with lower variance than the naive sample mean by using auxiliary dense signals.

## Implemented Algorithm

For each instance `i`, with:
- question `x_i`
- primary policy response `y_i`
- binary verifier reward `phi_i`
- one auxiliary dense reward `r_i` from AceMath-7B-RM

the repo computes:
- `tau_i = tau_LLM(x_i, y_i, r_i)`
- `m_i = m_LLM(x_i, y_i)`
- `psi_hat_i = m_i + phi_i - tau_i`

Legacy Monte Carlo compatibility is still available when precomputed `aux_tau_values`
are present, but the primary `hybrid_eif` / `hybrid_eif_online` path now follows the
literal Algorithm 1 from `Hybrid_reward.pdf`.

Implemented in:
- `verl/utils/reward_score/one_step_eif.py`
- `verl/utils/reward_score/hybrid_reward_eif.py`

## Auxiliary Signal Generation

The primary Algorithm 1 path scores each fixed `(x_i, y_i)` once with AceMath-7B-RM
to obtain one dense auxiliary reward `r_i`. Then:

1. query `tau_LLM(question, response, r_i)` to obtain `tau_i`
2. query `m_LLM(question, response)` to obtain `m_i`

AceMath-7B scoring reuses the same discriminative RM request format used by HERO.

The helper / preprocessing entrypoint is:
- `examples/data_preprocess/hybrid_reward_eif_eval.py`

## Offline Evaluation Integration

`verl/trainer/main_eval.py` now supports optional one-step EIF scoring.

Enable in eval config:

```yaml
data:
  tau_key: tau_llm_value
  m_key: m_llm_value

one_step_eif:
  enable: true
  primary_response_index: 0
```

The normalized parquet row should contain:
- standard eval fields (`responses`, `data_source`, `reward_model.ground_truth`)
- `responses`: only the designated primary response `y_i`
- `tau_llm_value`: scalar `tau_LLM(x_i, y_i, r_i)`
- `m_llm_value`: scalar `m_LLM(x_i, y_i)`

Optional diagnostics:
- `aux_reward_value`: scalar AceMath-7B dense reward `r_i`

## Training-Time Reward Integration

The experimental reward loop also supports using the one-step EIF estimator as the
training reward directly.

Enable it with:

```yaml
reward:
  reward_manager:
    name: hybrid_eif
  reward_kwargs:
    hybrid_eif:
      aux_tau_key: aux_tau_values
```

Each training sample may provide precomputed scalar `tau_llm_value` and
`m_llm_value`; otherwise the online path computes them from AceMath-7B-RM plus the
configured tau/m LLM endpoints. The reward loop computes:

- `phi`: verifier score from the configured rule reward
- `hybrid_eif_final_score = m_llm_value + phi - tau_llm_value`

Logged reward diagnostics include:
- `hybrid_eif_phi`
- `hybrid_eif_tau`
- `hybrid_eif_m`
- `hybrid_eif_final_score`

## Preprocessing Modes

`examples/data_preprocess/hybrid_reward_eif_eval.py` supports three modes:

1. Precomputed tau:
   - input already contains `aux_tau_values`
2. Precomputed AceMath rewards:
   - input contains `aux_reward_values`
   - script generates `aux_tau_values`
3. End-to-end generation:
   - default: script repeatedly scores the same response with AceMath-7B-RM
   - compatibility: input may also provide an auxiliary response bundle
   - script computes AceMath rewards and `tau_LLM`

Example:

```bash
python examples/data_preprocess/hybrid_reward_eif_eval.py \
  --input_path /path/to/raw_eval.parquet \
  --output_path /path/to/eif_eval.parquet \
  --prompt_key prompt \
  --response_key responses \
  --num_aux_samples 4 \
  --reward_router_address <host:port> \
  --reward_model_path nvidia/AceMath-7B-RM \
  --reward_temperature 1.0 \
  --tau_model gpt-oss \
  --tau_base_url https://ellm.nrp-nautilus.io/v1
```

Output metrics include:
- naive mean
- one-step EIF mean
- naive / EIF variance and variance reduction

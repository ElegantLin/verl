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
- auxiliary dense rewards `r_{i1..M+1}` from AceMath-7B

the repo computes:
- `tau_{i,j} = tau_LLM(x_i, y_i, r_ij)`
- `m_hat_i = (1/M) * sum_{j=2..M+1} tau_{i,j}`
- `psi_hat_i = m_hat_i + phi_i - tau_{i,1}`

Implemented in:
- `verl/utils/reward_score/one_step_eif.py`
- `verl/utils/reward_score/hybrid_reward_eif.py`

## Auxiliary Signal Generation

The repo now supports two practical ways to construct the auxiliary rewards:

1. Default repeated-sampling mode:
   - fix one `(x_i, y_i)`
   - query AceMath-7B-RM on the same response `M+1` times to obtain `r_{i1..M+1}`
2. Compatibility mode:
   - provide an auxiliary response bundle
   - score each auxiliary response with AceMath-7B-RM to obtain `r_{i1..M+1}`

Then query a fixed LLM (`tau_LLM`) on `(question, response, ace_reward)` to obtain
`tau_{i,j}`.

AceMath-7B scoring reuses the same discriminative RM request format used by HERO, with
optional reward-side sampling kwargs forwarded to the reward endpoint.

The helper / preprocessing entrypoint is:
- `examples/data_preprocess/hybrid_reward_eif_eval.py`

## Offline Evaluation Integration

`verl/trainer/main_eval.py` now supports optional one-step EIF scoring.

Enable in eval config:

```yaml
data:
  aux_tau_key: aux_tau_values

one_step_eif:
  enable: true
  primary_response_index: 0
```

The normalized parquet row should contain:
- standard eval fields (`responses`, `data_source`, `reward_model.ground_truth`)
- `responses`: only the designated primary response `y_i`
- `aux_tau_values`: list of floats with length `M+1` (`>= 2`)

Optional diagnostics:
- `aux_reward_values`: list of AceMath-7B dense rewards aligned with `aux_tau_values`

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

Each training sample must provide `aux_tau_values` either in `DataProto.batch`
as a tensor of shape `[M+1]` or in `DataProto.non_tensor_batch` as a list-like
object. The reward loop computes:

- `phi`: verifier score from the configured rule reward
- `hybrid_eif_final_score = mean(aux_tau_values[1:]) + phi - aux_tau_values[0]`

Logged reward diagnostics include:
- `hybrid_eif_phi`
- `hybrid_eif_tau_control`
- `hybrid_eif_tau_mc_mean`
- `hybrid_eif_num_aux_samples`
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

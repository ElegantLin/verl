# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import numpy as np


def _validate_inputs(
    primary_scores: np.ndarray | list[float], tau_samples: np.ndarray | list[list[float]]
) -> tuple[np.ndarray, np.ndarray]:
    primary = np.asarray(primary_scores, dtype=np.float64)
    tau = np.asarray(tau_samples, dtype=np.float64)

    if primary.ndim != 1:
        raise ValueError(f"primary_scores must be 1D, but got shape={primary.shape}")
    if tau.ndim != 2:
        raise ValueError(f"tau_samples must be 2D with shape [N, M+1], but got shape={tau.shape}")
    if tau.shape[0] != primary.shape[0]:
        raise ValueError(
            f"tau_samples and primary_scores must align on N. Got {tau.shape[0]=} and {primary.shape[0]=}."
        )
    if tau.shape[1] < 2:
        raise ValueError(
            "tau_samples must contain at least M+1=2 values per instance "
            "(one control term + at least one Monte Carlo sample)."
        )
    return primary, tau


def compute_one_step_scores(
    primary_scores: np.ndarray | list[float], tau_samples: np.ndarray | list[list[float]]
) -> np.ndarray:
    """Compute one-step EIF aggregated scores from Algorithm 1.

    For each instance i:
      m_hat_i = mean(tau(x_i, y_i, r_{i,2..M+1}))
      psi_hat_i = m_hat_i + phi_i - tau(x_i, y_i, r_{i,1})
    """
    primary, tau = _validate_inputs(primary_scores, tau_samples)
    m_hat = np.mean(tau[:, 1:], axis=1)
    return (m_hat + primary - tau[:, 0]).astype(np.float64)


def summarize_one_step_estimator(
    primary_scores: np.ndarray | list[float], tau_samples: np.ndarray | list[list[float]]
) -> dict[str, float]:
    """Return naive vs one-step EIF summary statistics."""
    primary, tau = _validate_inputs(primary_scores, tau_samples)
    one_step_scores = compute_one_step_scores(primary, tau)
    n = primary.shape[0]
    ddof = 1 if n > 1 else 0
    naive_var = float(np.var(primary, ddof=ddof))
    one_step_var = float(np.var(one_step_scores, ddof=ddof))
    return {
        "theta_naive": float(np.mean(primary)),
        "theta_one_step_eif": float(np.mean(one_step_scores)),
        "var_naive": naive_var,
        "var_one_step_eif": one_step_var,
        "var_reduction": float(naive_var - one_step_var),
    }

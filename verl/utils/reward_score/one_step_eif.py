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


def _validate_algorithm1_inputs(
    primary_scores: np.ndarray | list[float],
    tau_scores: np.ndarray | list[float],
    m_scores: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    primary = np.asarray(primary_scores, dtype=np.float64)
    tau = np.asarray(tau_scores, dtype=np.float64)
    m = np.asarray(m_scores, dtype=np.float64)

    if primary.ndim != 1:
        raise ValueError(f"primary_scores must be 1D, but got shape={primary.shape}")
    if tau.ndim != 1:
        raise ValueError(f"tau_scores must be 1D, but got shape={tau.shape}")
    if m.ndim != 1:
        raise ValueError(f"m_scores must be 1D, but got shape={m.shape}")
    if tau.shape[0] != primary.shape[0] or m.shape[0] != primary.shape[0]:
        raise ValueError(
            "primary_scores, tau_scores, and m_scores must align on N. "
            f"Got {primary.shape[0]=}, {tau.shape[0]=}, {m.shape[0]=}."
        )
    return primary, tau, m


def compute_algorithm1_scores(
    primary_scores: np.ndarray | list[float],
    tau_scores: np.ndarray | list[float],
    m_scores: np.ndarray | list[float],
) -> np.ndarray:
    """Compute the literal Algorithm 1 score psi_i = m_i + phi_i - tau_i."""
    primary, tau, m = _validate_algorithm1_inputs(primary_scores, tau_scores, m_scores)
    return (m + primary - tau).astype(np.float64)


def compute_one_step_scores(
    primary_scores: np.ndarray | list[float],
    tau_scores: np.ndarray | list[float],
    m_scores: np.ndarray | list[float],
) -> np.ndarray:
    """Compute one-step EIF scores from Algorithm 1."""
    return compute_algorithm1_scores(primary_scores, tau_scores, m_scores)


def summarize_one_step_estimator(
    primary_scores: np.ndarray | list[float],
    tau_scores: np.ndarray | list[float],
    m_scores: np.ndarray | list[float],
) -> dict[str, float]:
    """Return naive vs one-step EIF summary statistics."""
    primary = np.asarray(primary_scores, dtype=np.float64)
    one_step_scores = compute_one_step_scores(primary, tau_scores, m_scores)
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

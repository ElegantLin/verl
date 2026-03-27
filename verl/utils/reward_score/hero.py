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

from collections import OrderedDict
from typing import Any

import numpy as np


def _ordered_unique(values: np.ndarray) -> list[Any]:
    return list(OrderedDict.fromkeys(values.tolist()).keys())


def apply_naive_reward_combine(
    *,
    rule_scores: np.ndarray | list[float],
    rm_scores: np.ndarray | list[float],
    alpha: float = 0.5,
) -> np.ndarray:
    """Linearly combine verifier and RM scores for the naive hybrid baseline.

    This matches the appendix baseline where the final reward is a direct weighted
    average of the sparse verifier score and the dense reward-model score.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], but got {alpha=}.")

    rule = np.asarray(rule_scores, dtype=np.float32)
    rm = np.asarray(rm_scores, dtype=np.float32)

    if rule.shape != rm.shape:
        raise ValueError(f"rule_scores and rm_scores must have the same shape, but got {rule.shape=} and {rm.shape=}.")

    if rule.ndim != 1:
        raise ValueError(f"Only 1D inputs are supported, but got {rule.ndim=}.")

    return (alpha * rule + (1.0 - alpha) * rm).astype(np.float32)


def apply_hero_shaping(
    *,
    rule_scores: np.ndarray | list[float],
    rm_scores: np.ndarray | list[float],
    uids: np.ndarray | list[str],
    alpha: float = 0.1,
    beta: float = 0.1,
    eps: float = 1e-6,
    w_min: float = 0.4,
    w_max: float = 3.0,
    k: float = 6.0,
    sigma_bar: float | None = None,
    sigma_ema: float = 0.9,
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    """Apply HERO reward shaping (Eq. 3-5 in arXiv:2510.07242v1).

    Args:
        rule_scores: Verifier scores per sample (expected in [0, 1], thresholded at 0.5).
        rm_scores: Dense reward model scores per sample.
        uids: Prompt-level group ids aligned with samples.
        alpha: Incorrect-group range controller in Eq. 3.
        beta: Correct-group range controller in Eq. 3.
        eps: Numerical stabilizer for min-max normalization.
        w_min: Lower bound for variance-aware weights.
        w_max: Upper bound for variance-aware weights.
        k: Logistic steepness for variance-aware weighting.
        sigma_bar: Running mean of group stds. If None, current-batch mean is used.
        sigma_ema: EMA momentum used to update running sigma mean.

    Returns:
        final_scores: HERO shaped reward per sample.
        diagnostics: Per-sample diagnostics arrays:
            - stratified_reward
            - difficulty_weight
            - group_sigma
            - group_sigma_bar
        updated_sigma_bar: Updated running sigma mean for next batch.
    """
    rule = np.asarray(rule_scores, dtype=np.float32)
    rm = np.asarray(rm_scores, dtype=np.float32)
    uid_arr = np.asarray(uids, dtype=object)

    if rule.shape != rm.shape or rm.shape != uid_arr.shape:
        raise ValueError(
            "rule_scores, rm_scores, and uids must have the same shape, "
            f"but got {rule.shape=}, {rm.shape=}, {uid_arr.shape=}"
        )

    if rule.ndim != 1:
        raise ValueError(f"Only 1D inputs are supported, but got {rule.ndim=}")

    correct = (rule >= 0.5).astype(np.int64)
    stratified = np.zeros_like(rm, dtype=np.float32)
    difficulty_weight = np.zeros_like(rm, dtype=np.float32)
    group_sigma = np.zeros_like(rm, dtype=np.float32)

    group_ids = _ordered_unique(uid_arr)
    sigmas: list[float] = []
    group_indices: list[np.ndarray] = []

    for group_id in group_ids:
        idx = np.where(uid_arr == group_id)[0]
        if idx.size == 0:
            continue
        sigma = float(np.std(rm[idx]))
        sigmas.append(sigma)
        group_indices.append(idx)

    batch_sigma_mean = float(np.mean(sigmas)) if sigmas else 0.0
    sigma_bar_current = float(batch_sigma_mean if sigma_bar is None else sigma_bar)

    for idx, sigma in zip(group_indices, sigmas, strict=True):
        weight = w_min + (w_max - w_min) / (1.0 + np.exp(-k * (sigma - sigma_bar_current)))

        difficulty_weight[idx] = weight
        group_sigma[idx] = sigma

        for label in (0, 1):
            label_idx = idx[correct[idx] == label]
            if label_idx.size == 0:
                continue

            label_rm = rm[label_idx]
            rm_min = float(np.min(label_rm))
            rm_max = float(np.max(label_rm))
            norm = (label_rm - rm_min) / (rm_max - rm_min + eps)

            if label == 0:
                # Incorrect group: [-alpha, alpha]
                label_reward = -alpha + 2.0 * alpha * norm
            else:
                # Correct group: [1-beta, 1+beta]
                label_reward = (1.0 - beta) + 2.0 * beta * norm

            stratified[label_idx] = label_reward.astype(np.float32)

    final_scores = difficulty_weight * stratified
    updated_sigma_bar = sigma_bar_current if sigma_bar is not None else batch_sigma_mean
    if sigma_bar is not None:
        updated_sigma_bar = sigma_ema * sigma_bar_current + (1.0 - sigma_ema) * batch_sigma_mean

    diagnostics = {
        "stratified_reward": stratified.astype(np.float32),
        "difficulty_weight": difficulty_weight.astype(np.float32),
        "group_sigma": group_sigma.astype(np.float32),
        "group_sigma_bar": np.full_like(group_sigma, fill_value=sigma_bar_current, dtype=np.float32),
    }
    return final_scores.astype(np.float32), diagnostics, float(updated_sigma_bar)

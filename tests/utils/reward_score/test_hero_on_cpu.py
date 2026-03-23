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

import numpy as np
import pytest

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

HERO_MODULE_PATH = Path(__file__).resolve().parents[3] / "verl" / "utils" / "reward_score" / "hero.py"
hero_spec = spec_from_file_location("hero_reward_module", HERO_MODULE_PATH)
assert hero_spec is not None and hero_spec.loader is not None
hero_module = module_from_spec(hero_spec)
hero_spec.loader.exec_module(hero_module)
apply_hero_shaping = hero_module.apply_hero_shaping


def test_hero_stratified_normalization_without_weighting():
    final_scores, diagnostics, sigma_bar = apply_hero_shaping(
        rule_scores=[0, 0, 1, 1],
        rm_scores=[2.0, 4.0, 1.0, 5.0],
        uids=["g1", "g1", "g1", "g1"],
        alpha=0.1,
        beta=0.1,
        w_min=1.0,
        w_max=1.0,
    )

    expected = np.array([-0.1, 0.1, 0.9, 1.1], dtype=np.float32)
    np.testing.assert_allclose(final_scores, expected, atol=1e-6)
    np.testing.assert_allclose(diagnostics["stratified_reward"], expected, atol=1e-6)

    expected_sigma = np.std(np.array([2.0, 4.0, 1.0, 5.0], dtype=np.float32))
    np.testing.assert_allclose(diagnostics["group_sigma"], expected_sigma, atol=1e-6)
    np.testing.assert_allclose(sigma_bar, expected_sigma, atol=1e-6)


def test_hero_variance_aware_weight_and_running_sigma():
    final_scores, diagnostics, sigma_bar = apply_hero_shaping(
        rule_scores=[1, 1, 1, 1],
        rm_scores=[0.0, 1.0, 0.0, 0.0],
        uids=["a", "a", "b", "b"],
        alpha=0.1,
        beta=0.1,
        w_min=0.5,
        w_max=2.0,
        k=5.0,
        sigma_bar=0.1,
        sigma_ema=0.8,
    )

    # Group a has higher RM variance than group b, so it should receive larger weight.
    assert diagnostics["difficulty_weight"][0] > diagnostics["difficulty_weight"][2]
    assert final_scores[1] > final_scores[2]

    expected_sigma_bar = 0.8 * 0.1 + 0.2 * 0.25
    np.testing.assert_allclose(sigma_bar, expected_sigma_bar, atol=1e-6)


def test_hero_shape_validation():
    with pytest.raises(ValueError):
        apply_hero_shaping(rule_scores=[1], rm_scores=[0.2, 0.3], uids=["x"])

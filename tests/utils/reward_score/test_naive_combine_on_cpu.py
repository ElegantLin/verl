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

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest


HELPER_MODULE_PATH = Path(__file__).resolve().parents[3] / "verl" / "utils" / "reward_score" / "hero.py"
helper_spec = spec_from_file_location("hero_reward_module", HELPER_MODULE_PATH)
assert helper_spec is not None and helper_spec.loader is not None
helper_module = module_from_spec(helper_spec)
helper_spec.loader.exec_module(helper_module)

apply_naive_reward_combine = helper_module.apply_naive_reward_combine


def test_naive_reward_combine_interpolates_rule_and_rm_scores():
    scores = apply_naive_reward_combine(
        rule_scores=[1.0, 0.0, 1.0],
        rm_scores=[0.2, 0.8, -0.5],
        alpha=0.5,
    )

    expected = np.array([0.6, 0.4, 0.25], dtype=np.float32)
    np.testing.assert_allclose(scores, expected, atol=1e-6)


def test_naive_reward_combine_validates_alpha_range():
    with pytest.raises(ValueError):
        apply_naive_reward_combine(rule_scores=[1.0], rm_scores=[0.0], alpha=-0.1)

    with pytest.raises(ValueError):
        apply_naive_reward_combine(rule_scores=[1.0], rm_scores=[0.0], alpha=1.1)


def test_naive_reward_combine_validates_shapes():
    with pytest.raises(ValueError):
        apply_naive_reward_combine(rule_scores=[1.0], rm_scores=[0.0, 1.0], alpha=0.5)

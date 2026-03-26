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

EIF_MODULE_PATH = Path(__file__).resolve().parents[3] / "verl" / "utils" / "reward_score" / "one_step_eif.py"
eif_spec = spec_from_file_location("one_step_eif_module", EIF_MODULE_PATH)
assert eif_spec is not None and eif_spec.loader is not None
eif_module = module_from_spec(eif_spec)
eif_spec.loader.exec_module(eif_module)

compute_one_step_scores = eif_module.compute_one_step_scores
compute_algorithm1_scores = eif_module.compute_algorithm1_scores
summarize_one_step_estimator = eif_module.summarize_one_step_estimator


def test_algorithm1_scores_match_paper_formula():
    primary = np.array([1.0, 0.0], dtype=np.float64)
    tau = np.array([0.2, 0.7], dtype=np.float64)
    m = np.array([0.8, 0.3], dtype=np.float64)

    scores = compute_algorithm1_scores(primary, tau, m)
    np.testing.assert_allclose(scores, np.array([1.6, -0.4], dtype=np.float64), atol=1e-6)


def test_one_step_scores_match_algorithm1_formula():
    primary = np.array([1.0, 0.0], dtype=np.float64)
    tau = np.array([0.2, 0.7], dtype=np.float64)
    m = np.array([0.8, 0.3], dtype=np.float64)

    scores = compute_one_step_scores(primary_scores=primary, tau_scores=tau, m_scores=m)
    np.testing.assert_allclose(scores, np.array([1.6, -0.4], dtype=np.float64), atol=1e-6)


def test_one_step_summary_fields():
    summary = summarize_one_step_estimator(
        primary_scores=[1.0, 0.0, 1.0],
        tau_scores=[0.2, 0.8, 0.3],
        m_scores=[0.5, 0.4, 0.6],
    )
    for key in ["theta_naive", "theta_one_step_eif", "var_naive", "var_one_step_eif", "var_reduction"]:
        assert key in summary


def test_one_step_input_validation():
    with pytest.raises(ValueError):
        compute_one_step_scores(primary_scores=[1.0], tau_scores=[[0.2]], m_scores=[0.4])

    with pytest.raises(ValueError):
        compute_one_step_scores(primary_scores=[1.0, 0.0], tau_scores=[0.2, 0.3], m_scores=[[0.4, 0.5]])

    with pytest.raises(ValueError):
        compute_algorithm1_scores(primary_scores=[1.0, 0.0], tau_scores=[0.2], m_scores=[0.4, 0.5])


def test_one_step_summary_requires_explicit_m_scores():
    with pytest.raises(TypeError):
        summarize_one_step_estimator(
            primary_scores=[1.0, 0.0],
            tau_scores=[0.2, 0.7],
        )

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

import asyncio
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import pytest

HELPER_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "verl" / "utils" / "reward_score" / "hybrid_reward_eif.py"
)
helper_spec = spec_from_file_location("hybrid_reward_eif_online_module", HELPER_MODULE_PATH)
assert helper_spec is not None and helper_spec.loader is not None
helper_module = module_from_spec(helper_spec)
sys.modules[helper_spec.name] = helper_module
helper_spec.loader.exec_module(helper_module)

compute_algorithm1_online_scores = helper_module.compute_algorithm1_online_scores
compute_response_algorithm1_eif = helper_module.compute_response_algorithm1_eif
extract_question_text = helper_module.extract_question_text
normalize_chat_prompt = helper_module.normalize_chat_prompt
TauLLMScorer = helper_module.TauLLMScorer


class _FakeAceMathScorer:
    async def score_async(self, question, response_text, *, raw_prompt=None, session=None, semaphore=None):
        assert raw_prompt is not None
        return 0.25


class _FakeTauScorer:
    async def score_async(self, question, primary_response, aux_reward, *, session=None, semaphore=None):
        primary_idx = int(primary_response[-1])
        return primary_idx * 0.1 + aux_reward


class _FakeMarginalScorer:
    async def score_async(self, question, primary_response, *, session=None, semaphore=None):
        primary_idx = int(primary_response[-1])
        return 0.8 - primary_idx * 0.1


class _RecordingTauScorer(TauLLMScorer):
    async def score_async(self, question, primary_response, aux_reward, *, session=None, semaphore=None):
        assert session is _SENTINEL_SESSION
        return aux_reward + 0.1


_SENTINEL_SESSION = object()


def test_legacy_monte_carlo_helpers_removed():
    assert not hasattr(helper_module, "build_aux_index_sets")
    assert not hasattr(helper_module, "compute_online_eif_scores")
    assert not hasattr(helper_module, "compute_response_online_eif")
    assert not hasattr(helper_module, "compute_group_online_eif")


def test_compute_algorithm1_online_scores_returns_diagnostics():
    scores, diagnostics = compute_algorithm1_online_scores(
        phi_scores=[1.0, 0.0],
        tau_scores=[0.2, 0.7],
        m_scores=[0.8, 0.3],
    )
    np.testing.assert_allclose(scores, np.array([1.6, -0.4], dtype=np.float64), atol=1e-6)
    np.testing.assert_allclose(diagnostics["tau"], np.array([0.2, 0.7], dtype=np.float64), atol=1e-6)
    np.testing.assert_allclose(diagnostics["m"], np.array([0.8, 0.3], dtype=np.float64), atol=1e-6)


def test_compute_response_algorithm1_eif_uses_single_aux_reward():
    result = asyncio.run(
        compute_response_algorithm1_eif(
            question="Math question",
            response="resp0",
            phi_score=1.0,
            aux_scorer=_FakeAceMathScorer(),
            tau_scorer=_FakeTauScorer(),
            m_scorer=_FakeMarginalScorer(),
            raw_prompt=[{"role": "user", "content": "Math question"}],
        )
    )

    assert result["aux_reward"] == pytest.approx(0.25)
    assert result["tau_score"] == pytest.approx(0.25)
    assert result["m_score"] == pytest.approx(0.8)
    assert result["score"] == pytest.approx(1.55)


def test_extract_question_text_prefers_extra_info_and_handles_chat():
    assert (
        extract_question_text(
            raw_prompt=[{"role": "user", "content": "fallback"}],
            extra_info={"question": "preferred"},
        )
        == "preferred"
    )
    assert (
        extract_question_text(
            raw_prompt=[
                {"role": "system", "content": "sys"},
                {"role": "user", "content": [{"type": "text", "text": "from raw prompt"}]},
            ]
        )
        == "from raw prompt"
    )


def test_normalize_chat_prompt_flattens_structured_content():
    assert normalize_chat_prompt(
        [{"role": "user", "content": [{"type": "text", "text": "from raw prompt"}]}]
    ) == [{"role": "user", "content": "from raw prompt"}]


def test_tau_score_many_async_reuses_caller_session(monkeypatch):
    import aiohttp

    monkeypatch.setattr(
        aiohttp,
        "ClientSession",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ClientSession should not be constructed")),
    )
    scorer = _RecordingTauScorer(model="mock", base_url="http://localhost")

    values = asyncio.run(
        scorer.score_many_async(
            "Math question",
            "resp0",
            [0.2, 0.4],
            session=_SENTINEL_SESSION,
        )
    )

    np.testing.assert_allclose(values, np.array([0.3, 0.5], dtype=np.float64), atol=1e-6)

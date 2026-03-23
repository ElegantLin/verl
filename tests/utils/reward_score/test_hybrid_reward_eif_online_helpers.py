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

build_aux_index_sets = helper_module.build_aux_index_sets
compute_group_online_eif = helper_module.compute_group_online_eif
compute_online_eif_scores = helper_module.compute_online_eif_scores
compute_response_online_eif = helper_module.compute_response_online_eif
extract_question_text = helper_module.extract_question_text
normalize_chat_prompt = helper_module.normalize_chat_prompt
TauLLMScorer = helper_module.TauLLMScorer


class _FakeAuxScorer:
    """Fake auxiliary reward scorer that returns deterministic scores."""

    async def score_many_async(self, question, responses, session=None, semaphore=None):
        assert "question" in question.lower()
        return [0.1 * (idx + 1) for idx in range(len(responses))]


class _FakeTauScorer:
    async def score_many_async(self, question, primary_response, aux_rewards, session=None, semaphore=None):
        primary_idx = int(primary_response[-1])
        return [primary_idx * 0.1 + reward for reward in aux_rewards]


class _PartiallyFailingTauScorer:
    async def score_many_async(self, question, primary_response, aux_rewards, session=None, semaphore=None):
        if primary_response == "resp1":
            raise RuntimeError("tau service timeout")
        primary_idx = int(primary_response[-1])
        return [primary_idx * 0.1 + reward for reward in aux_rewards]


class _RecordingTauScorer(TauLLMScorer):
    async def score_async(self, question, primary_response, aux_reward, *, session=None, semaphore=None):
        assert session is _SENTINEL_SESSION
        return aux_reward + 0.1


_SENTINEL_SESSION = object()


def test_build_aux_index_sets_cyclic_excludes_primary():
    aux_index_sets = build_aux_index_sets(
        group_size=4,
        num_aux_samples=2,
        include_self_in_aux=False,
        selection_mode="cyclic",
    )
    assert aux_index_sets == [[1, 2], [2, 3], [3, 0], [0, 1]]


def test_build_aux_index_sets_hash_shuffle_is_stable():
    first = build_aux_index_sets(
        group_size=5,
        num_aux_samples=3,
        include_self_in_aux=False,
        selection_mode="hash_shuffle",
        uid="same-uid",
    )
    second = build_aux_index_sets(
        group_size=5,
        num_aux_samples=3,
        include_self_in_aux=False,
        selection_mode="hash_shuffle",
        uid="same-uid",
    )
    assert first == second


def test_compute_online_eif_scores_returns_diagnostics():
    scores, diagnostics = compute_online_eif_scores(
        phi_scores=[1.0, 0.0],
        tau_samples=[[0.2, 0.8], [0.6, 0.4]],
    )
    np.testing.assert_allclose(scores, np.array([1.6, -0.2], dtype=np.float64), atol=1e-6)
    np.testing.assert_allclose(diagnostics["tau_control"], np.array([0.2, 0.6], dtype=np.float64), atol=1e-6)
    np.testing.assert_allclose(diagnostics["tau_mc_mean"], np.array([0.8, 0.4], dtype=np.float64), atol=1e-6)
    np.testing.assert_array_equal(diagnostics["num_aux_samples"], np.array([2, 2], dtype=np.int64))


def test_compute_group_online_eif_uses_group_shared_aux_scores():
    result = asyncio.run(
        compute_group_online_eif(
            uid="group-0",
            question="Math question",
            responses=["resp0", "resp1", "resp2", "resp3"],
            phi_scores=[1.0, 0.0, 1.0, 0.0],
            aux_scorer=_FakeAuxScorer(),
            tau_scorer=_FakeTauScorer(),
            raw_prompt=[{"role": "user", "content": "Math question"}],
            num_aux_samples=2,
            include_self_in_aux=False,
            selection_mode="cyclic",
        )
    )

    np.testing.assert_allclose(result["aux_rewards"], np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64), atol=1e-6)
    assert result["aux_index_sets"] == [[1, 2], [2, 3], [3, 0], [0, 1]]
    np.testing.assert_allclose(result["scores"], np.array([1.1, 0.1, 0.7, 0.1], dtype=np.float64), atol=1e-6)


def test_compute_response_online_eif_repeats_same_response():
    class _RecordingAuxScorer(_FakeAuxScorer):
        def __init__(self):
            self.calls = []

        async def score_many_async(self, question, responses, session=None, semaphore=None):
            self.calls.append(list(responses))
            return await super().score_many_async(
                question, responses, session=session, semaphore=semaphore
            )

    aux_scorer = _RecordingAuxScorer()
    result = asyncio.run(
        compute_response_online_eif(
            question="Math question",
            response="resp0",
            phi_score=1.0,
            aux_scorer=aux_scorer,
            tau_scorer=_FakeTauScorer(),
            raw_prompt=[{"role": "user", "content": "Math question"}],
            num_aux_samples=3,
        )
    )

    assert aux_scorer.calls == [["resp0", "resp0", "resp0"]]
    np.testing.assert_allclose(result["aux_rewards"], np.array([0.1, 0.2, 0.3], dtype=np.float64), atol=1e-6)
    np.testing.assert_allclose(result["tau_samples"], np.array([0.1, 0.2, 0.3], dtype=np.float64), atol=1e-6)
    assert result["score"] == pytest.approx(1.15)
    assert result["diagnostics"]["num_aux_samples"] == 3


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


def test_build_aux_index_sets_requires_two_auxiliaries():
    with pytest.raises(ValueError):
        build_aux_index_sets(group_size=2, num_aux_samples=1, include_self_in_aux=False)

    with pytest.raises(ValueError):
        asyncio.run(
            compute_response_online_eif(
                question="Math question",
                response="resp0",
                phi_score=1.0,
                aux_scorer=_FakeAuxScorer(),
                tau_scorer=_FakeTauScorer(),
                raw_prompt=[{"role": "user", "content": "Math question"}],
                num_aux_samples=1,
            )
        )


def test_compute_group_online_eif_falls_back_per_row():
    result = asyncio.run(
        compute_group_online_eif(
            uid="group-1",
            question="Math question",
            responses=["resp0", "resp1", "resp2"],
            phi_scores=[1.0, 0.0, 1.0],
            aux_scorer=_FakeAuxScorer(),
            tau_scorer=_PartiallyFailingTauScorer(),
            raw_prompt=[{"role": "user", "content": "Math question"}],
            num_aux_samples=2,
            include_self_in_aux=False,
            selection_mode="cyclic",
        )
    )

    np.testing.assert_allclose(result["scores"][[0, 2]], np.array([1.1, 1.1], dtype=np.float64), atol=1e-6)
    assert result["scores"][1] == pytest.approx(0.0)
    np.testing.assert_array_equal(result["diagnostics"]["row_fallback"], np.array([0, 1, 0], dtype=np.int64))
    assert result["row_errors"][1] == "tau service timeout"


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

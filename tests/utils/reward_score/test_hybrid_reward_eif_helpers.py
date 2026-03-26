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
import sys

import pytest

HELPER_MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "verl" / "utils" / "reward_score" / "hybrid_reward_eif.py"
)
helper_spec = spec_from_file_location("hybrid_reward_eif_module", HELPER_MODULE_PATH)
assert helper_spec is not None and helper_spec.loader is not None
helper_module = module_from_spec(helper_spec)
sys.modules[helper_spec.name] = helper_module
helper_spec.loader.exec_module(helper_module)

parse_tau_score = helper_module.parse_tau_score
resolve_auxiliary_response_bundle = helper_module.resolve_auxiliary_response_bundle
select_primary_response = helper_module.select_primary_response
build_aux_reward_messages = helper_module.build_aux_reward_messages
build_m_messages = helper_module.build_m_messages
build_reward_model_messages = helper_module.build_reward_model_messages
build_tau_messages = helper_module.build_tau_messages


def test_select_primary_response_from_scalar_and_list():
    assert select_primary_response("single") == "single"
    assert select_primary_response(["r0", "r1", "r2"], primary_response_index=2) == "r2"


def test_resolve_auxiliary_response_bundle_excludes_primary_when_requested():
    primary, aux = resolve_auxiliary_response_bundle(
        ["r0", "r1", "r2"],
        primary_response_index=1,
        include_primary_response=False,
    )
    assert primary == "r1"
    assert aux == ["r0", "r2"]


def test_parse_tau_score_supports_plain_float_and_percent():
    assert parse_tau_score("0.73") == pytest.approx(0.73)
    assert parse_tau_score("73%") == pytest.approx(0.73)
    assert parse_tau_score("1.7") == pytest.approx(1.0)


def test_build_aux_reward_messages_contains_question_and_response():
    messages = build_aux_reward_messages("What is 2+2?", "The answer is 4.")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "evaluator" in messages[0]["content"].lower()
    assert "What is 2+2?" in messages[1]["content"]
    assert "The answer is 4." in messages[1]["content"]


def test_build_tau_messages_contains_aux_reward():
    messages = build_tau_messages("What is 2+2?", "The answer is 4.", 0.85)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "tau" in messages[0]["content"].lower()
    assert "0.85" in messages[1]["content"]
    assert "What is 2+2?" in messages[1]["content"]


def test_build_m_messages_excludes_auxiliary_reward():
    messages = build_m_messages("What is 2+2?", "The answer is 4.")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "m(x, y)" in messages[1]["content"]
    assert "Auxiliary reward" not in messages[1]["content"]


def test_build_reward_model_messages_reuses_raw_prompt_context():
    messages = build_reward_model_messages(
        "ignored question",
        "assistant answer",
        raw_prompt=[{"role": "system", "content": "sys"}, {"role": "user", "content": "question"}],
    )
    assert messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "assistant answer"},
    ]

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

import asyncio
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from verl.utils.reward_score.one_step_eif import compute_one_step_scores


_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?")


def _try_json_load(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def normalize_string_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist") and not isinstance(value, str):
        value = value.tolist()
    if isinstance(value, str):
        parsed = _try_json_load(value)
        if parsed is not value:
            value = parsed
        else:
            return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def normalize_float_list(value: Any, *, field_name: str) -> list[float]:
    if value is None:
        raise ValueError(f"`{field_name}` is required but missing.")
    if hasattr(value, "tolist") and not isinstance(value, str):
        value = value.tolist()
    if isinstance(value, str):
        parsed = _try_json_load(value)
        if parsed is not value:
            value = parsed
        else:
            raise ValueError(f"`{field_name}` must be list-like, got plain string.")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"`{field_name}` must be list-like, got {type(value).__name__}.")
    values = [float(item) for item in value]
    if len(values) < 2:
        raise ValueError(f"`{field_name}` must contain at least 2 values (M+1 >= 2).")
    return values


def _stringify_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text") is not None:
                    parts.append(str(item["text"]))
                elif item.get("content") is not None:
                    parts.append(str(item["content"]))
                elif item.get("text") is not None:
                    parts.append(str(item["text"]))
            elif item is not None:
                parts.append(str(item))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def extract_question_text(raw_prompt: Any = None, *, extra_info: Any = None, fallback: Any = None) -> str:
    if isinstance(extra_info, dict):
        question = extra_info.get("question")
        if question is not None and str(question).strip():
            return str(question)

    prompt_value = raw_prompt
    if isinstance(prompt_value, str):
        parsed = _try_json_load(prompt_value)
        if parsed is not prompt_value:
            prompt_value = parsed

    if isinstance(prompt_value, dict):
        prompt_value = [prompt_value]

    if isinstance(prompt_value, (list, tuple)):
        user_contents = []
        fallback_contents = []
        for item in prompt_value:
            if not isinstance(item, dict):
                continue
            content = _stringify_message_content(item.get("content"))
            if not content.strip():
                continue
            fallback_contents.append(content)
            if item.get("role") == "user":
                user_contents.append(content)
        if user_contents:
            return user_contents[-1]
        if fallback_contents:
            return fallback_contents[-1]

    if fallback is None:
        return ""
    return str(fallback)


def _stable_hash_int(*parts: Any) -> int:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(str(part).encode("utf-8"))
        hasher.update(b"\0")
    return int.from_bytes(hasher.digest()[:8], byteorder="big", signed=False)


def build_aux_index_sets(
    group_size: int,
    num_aux_samples: int = -1,
    *,
    include_self_in_aux: bool = False,
    selection_mode: str = "cyclic",
    uid: Any = None,
) -> list[list[int]]:
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")

    max_available = group_size if include_self_in_aux else group_size - 1
    if max_available < 2:
        raise ValueError(
            "Online hybrid EIF requires at least 2 auxiliary samples per primary response. "
            f"Got {group_size=} and {include_self_in_aux=}."
        )

    if num_aux_samples in (None, -1):
        num_aux_samples = max_available

    num_aux_samples = int(num_aux_samples)
    if num_aux_samples < 2:
        raise ValueError(f"num_aux_samples must be >= 2, got {num_aux_samples}.")
    if num_aux_samples > max_available:
        raise ValueError(
            f"num_aux_samples={num_aux_samples} exceeds available auxiliary responses {max_available}."
        )

    aux_index_sets: list[list[int]] = []
    for primary_idx in range(group_size):
        if selection_mode == "cyclic":
            ordered = [(primary_idx + offset) % group_size for offset in range(group_size)]
            candidates = [idx for idx in ordered if include_self_in_aux or idx != primary_idx]
        elif selection_mode == "hash_shuffle":
            candidates = list(range(group_size))
            if not include_self_in_aux:
                candidates.remove(primary_idx)
            rng = random.Random(_stable_hash_int(uid, primary_idx, group_size, num_aux_samples, selection_mode))
            rng.shuffle(candidates)
        else:
            raise ValueError(f"Unsupported auxiliary selection mode: {selection_mode}")

        aux_index_sets.append(candidates[:num_aux_samples])

    return aux_index_sets


def compute_online_eif_scores(
    phi_scores: np.ndarray | list[float], tau_samples: np.ndarray | list[list[float]]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    phi = np.asarray(phi_scores, dtype=np.float64)
    tau = np.asarray(tau_samples, dtype=np.float64)
    scores = compute_one_step_scores(phi, tau)
    diagnostics = {
        "phi": phi.astype(np.float64, copy=False),
        "tau_control": tau[:, 0].astype(np.float64, copy=False),
        "tau_mc_mean": np.mean(tau[:, 1:], axis=1).astype(np.float64, copy=False),
        "num_aux_samples": np.full(phi.shape[0], tau.shape[1], dtype=np.int64),
    }
    return scores, diagnostics


def select_primary_response(response_value: Any, primary_response_index: int = 0) -> str:
    responses = normalize_string_list(response_value, field_name="responses")
    if not responses:
        raise ValueError("`responses` must contain at least one response.")
    if primary_response_index < 0 or primary_response_index >= len(responses):
        raise IndexError(
            f"primary_response_index={primary_response_index} is out of range for {len(responses)} responses."
        )
    return responses[primary_response_index]


def resolve_auxiliary_response_bundle(
    response_value: Any,
    *,
    primary_response_index: int = 0,
    aux_response_value: Any = None,
    include_primary_response: bool = True,
    max_aux_samples: int | None = None,
) -> tuple[str, list[str]]:
    responses = normalize_string_list(response_value, field_name="responses")
    primary_response = select_primary_response(responses, primary_response_index=primary_response_index)

    if aux_response_value is not None:
        aux_responses = normalize_string_list(aux_response_value, field_name="aux_responses")
    else:
        aux_responses = list(responses)

    if not include_primary_response and aux_response_value is None:
        aux_responses = [response for idx, response in enumerate(aux_responses) if idx != primary_response_index]

    if max_aux_samples is not None:
        aux_responses = aux_responses[:max_aux_samples]

    return primary_response, aux_responses


def normalize_chat_prompt(raw_prompt: Any) -> list[dict[str, str]] | None:
    if raw_prompt is None:
        return None
    if isinstance(raw_prompt, str):
        parsed = _try_json_load(raw_prompt)
        if parsed is raw_prompt:
            return None
        raw_prompt = parsed
    if isinstance(raw_prompt, dict):
        raw_prompt = [raw_prompt]
    if not isinstance(raw_prompt, (list, tuple)):
        return None

    normalized = []
    for item in raw_prompt:
        if not isinstance(item, dict):
            return None
        role = item.get("role")
        content = item.get("content")
        if role is None or content is None:
            return None
        normalized.append({"role": str(role), "content": _stringify_message_content(content)})
    return normalized


def build_aux_reward_messages(question: str, response: str) -> list[dict[str, str]]:
    """Build prompt messages for the auxiliary reward model (r_ij generator).

    The auxiliary model evaluates (x_i, y_i) and outputs a score in [0, 1].
    Called M+1 times with temperature > 0 to produce diverse auxiliary samples.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are a math solution evaluator. Given a math question and a candidate response, "
                "estimate the probability that the response arrives at the correct final answer. "
                "Return only one number between 0 and 1."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Response:\n{response}\n\n"
                "Estimate the probability that this response is correct. "
                "Return only the number."
            ),
        },
    ]


def build_tau_messages(question: str, primary_response: str, aux_reward: float) -> list[dict[str, str]]:
    """Build prompt messages for tau_LLM, conditioned on the auxiliary reward r_ij."""
    return [
        {
            "role": "system",
            "content": (
                "You estimate tau(x, y, r): the probability that a math response is correct, "
                "given an auxiliary reward signal. Return only one number between 0 and 1."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Primary response:\n{primary_response}\n\n"
                f"Auxiliary reward r:\n{aux_reward:.8f}\n\n"
                "Estimate tau(x, y, r) = P(phi=1 | question, response, auxiliary reward). "
                "Return only the number."
            ),
        },
    ]


def parse_tau_score(text: str) -> float:
    if not isinstance(text, str):
        raise TypeError(f"tau response must be a string, got {type(text).__name__}.")
    match = _FLOAT_RE.search(text.strip())
    if match is None:
        raise ValueError(f"Unable to parse numeric tau score from response: {text!r}")
    value = float(match.group(0))
    if text[match.end() : match.end() + 1] == "%" and 0.0 <= value <= 100.0:
        value /= 100.0
    return min(1.0, max(0.0, value))


def _normalize_http_base(url: str) -> str:
    normalized = url.strip()
    if not normalized.startswith("http://") and not normalized.startswith("https://"):
        normalized = f"http://{normalized}"
    return normalized.rstrip("/")


@dataclass
class AuxRewardScorer:
    """Generative auxiliary reward scorer for r_ij.

    Scores (x_i, y_i) pairs by prompting a generative instruct model to output
    a probability estimate in [0, 1].  Called M+1 times with temperature > 0 to
    produce diverse auxiliary reward samples.

    Default model: Qwen/Qwen2.5-7B-Instruct served via an OpenAI-compatible API.
    """

    model: str = "Qwen/Qwen2.5-7B-Instruct"
    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 1.0
    max_tokens: int = 16
    timeout: float = 300.0
    client: Any = None

    def _get_client(self):
        if self.client is not None:
            return self.client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The `openai` package is required for the auxiliary reward scorer."
            ) from exc
        self.client = OpenAI(api_key=os.environ.get(self.api_key_env), base_url=self.base_url)
        return self.client

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def score(self, question: str, response_text: str) -> float:
        completion = self._get_client().chat.completions.create(
            model=self.model,
            messages=build_aux_reward_messages(question, response_text),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = completion.choices[0].message.content
        return parse_tau_score("" if content is None else str(content))

    def score_many(self, question: str, responses: list[str]) -> list[float]:
        return [self.score(question, response_text) for response_text in responses]

    async def score_async(
        self,
        question: str,
        response_text: str,
        *,
        session=None,
        semaphore=None,
    ) -> float:
        if session is None:
            raise ValueError("session is required for async auxiliary reward scoring.")

        async def _score_once() -> float:
            payload = {
                "model": self.model,
                "messages": build_aux_reward_messages(question, response_text),
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            url = f"{_normalize_http_base(self.base_url)}/chat/completions"
            async with session.post(url, json=payload, headers=self._build_headers()) as resp:
                resp.raise_for_status()
                result = await resp.json()
            content = result["choices"][0]["message"].get("content")
            return parse_tau_score("" if content is None else str(content))

        if semaphore is None:
            return await _score_once()
        async with semaphore:
            return await _score_once()

    async def score_many_async(
        self,
        question: str,
        responses: list[str],
        *,
        session=None,
        semaphore=None,
    ) -> list[float]:
        import aiohttp

        if session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(limit=0)
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as owned_session:
                tasks = [
                    self.score_async(
                        question,
                        response_text,
                        session=owned_session,
                        semaphore=semaphore,
                    )
                    for response_text in responses
                ]
                return list(await asyncio.gather(*tasks))

        tasks = [
            self.score_async(question, response_text, session=session, semaphore=semaphore)
            for response_text in responses
        ]
        return list(await asyncio.gather(*tasks))


async def compute_response_online_eif(
    *,
    question: str,
    response: str,
    phi_score: float,
    aux_scorer: AuxRewardScorer,
    tau_scorer: TauLLMScorer,
    raw_prompt: Any = None,
    num_aux_samples: int = 2,
    aux_session=None,
    tau_session=None,
    aux_semaphore=None,
    tau_semaphore=None,
) -> dict[str, Any]:
    """Compute online EIF for a single (x_i, y_i) pair.

    1. Call aux_scorer M+1 times on (question, response) → r_{i1}...r_{i,M+1}
    2. Call tau_scorer M+1 times with (question, response, r_{ij}) → tau values
    3. Compute one-step EIF aggregation.
    """
    num_aux_samples = int(num_aux_samples)
    if num_aux_samples < 2:
        raise ValueError(f"num_aux_samples must be >= 2, got {num_aux_samples}.")

    repeated_responses = [response] * num_aux_samples
    aux_rewards = await aux_scorer.score_many_async(
        question,
        repeated_responses,
        session=aux_session,
        semaphore=aux_semaphore,
    )
    tau_samples = await tau_scorer.score_many_async(
        question,
        response,
        aux_rewards,
        session=tau_session,
        semaphore=tau_semaphore,
    )
    scores, diagnostics = compute_online_eif_scores([phi_score], [tau_samples])
    return {
        "score": float(scores[0]),
        "aux_rewards": np.asarray(aux_rewards, dtype=np.float64),
        "tau_samples": np.asarray(tau_samples, dtype=np.float64),
        "diagnostics": {
            "phi": float(diagnostics["phi"][0]),
            "tau_control": float(diagnostics["tau_control"][0]),
            "tau_mc_mean": float(diagnostics["tau_mc_mean"][0]),
            "num_aux_samples": int(diagnostics["num_aux_samples"][0]),
        },
    }


@dataclass
class TauLLMScorer:
    model: str = "gpt-oss"
    base_url: str = "https://ellm.nrp-nautilus.io/v1"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 16
    timeout: float = 300.0
    client: Any = None

    def _get_client(self):
        if self.client is not None:
            return self.client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The `openai` package is required to generate tau_LLM scores. "
                "Install it or provide precomputed aux_tau_values."
            ) from exc

        self.client = OpenAI(api_key=os.environ.get(self.api_key_env), base_url=self.base_url)
        return self.client

    def score(self, question: str, primary_response: str, ace_reward: float) -> float:
        completion = self._get_client().chat.completions.create(
            model=self.model,
            messages=build_tau_messages(question, primary_response, ace_reward),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = completion.choices[0].message.content
        return parse_tau_score("" if content is None else str(content))

    def score_many(self, question: str, primary_response: str, ace_rewards: list[float]) -> list[float]:
        return [self.score(question, primary_response, ace_reward) for ace_reward in ace_rewards]

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def score_async(
        self,
        question: str,
        primary_response: str,
        ace_reward: float,
        *,
        session=None,
        semaphore=None,
    ) -> float:
        if session is None:
            raise ValueError("session is required for async tau scoring.")

        async def _score_once() -> float:
            payload = {
                "model": self.model,
                "messages": build_tau_messages(question, primary_response, ace_reward),
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            url = f"{_normalize_http_base(self.base_url)}/chat/completions"
            async with session.post(url, json=payload, headers=self._build_headers()) as resp:
                resp.raise_for_status()
                result = await resp.json()
            content = result["choices"][0]["message"].get("content")
            return parse_tau_score("" if content is None else str(content))

        if semaphore is None:
            return await _score_once()
        async with semaphore:
            return await _score_once()

    async def score_many_async(
        self,
        question: str,
        primary_response: str,
        ace_rewards: list[float],
        *,
        session=None,
        semaphore=None,
    ) -> list[float]:
        import aiohttp

        if session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(limit=0)
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as owned_session:
                tasks = [
                    self.score_async(
                        question,
                        primary_response,
                        ace_reward,
                        session=owned_session,
                        semaphore=semaphore,
                    )
                    for ace_reward in ace_rewards
                ]
                return list(await asyncio.gather(*tasks))

        tasks = [
            self.score_async(
                question,
                primary_response,
                ace_reward,
                session=session,
                semaphore=semaphore,
            )
            for ace_reward in ace_rewards
        ]
        return list(await asyncio.gather(*tasks))


async def compute_group_online_eif(
    *,
    uid: Any,
    question: str,
    responses: list[str],
    phi_scores: list[float] | np.ndarray,
    aux_scorer: AuxRewardScorer,
    tau_scorer: TauLLMScorer,
    raw_prompt: Any = None,
    num_aux_samples: int = -1,
    include_self_in_aux: bool = False,
    selection_mode: str = "cyclic",
    aux_session=None,
    tau_session=None,
    aux_semaphore=None,
    tau_semaphore=None,
) -> dict[str, Any]:
    if len(responses) == 0:
        raise ValueError("responses must be non-empty for online hybrid EIF.")

    phi = np.asarray(phi_scores, dtype=np.float64)
    if phi.shape[0] != len(responses):
        raise ValueError(f"phi_scores length {phi.shape[0]} does not match responses length {len(responses)}.")

    aux_index_sets = build_aux_index_sets(
        len(responses),
        num_aux_samples=num_aux_samples,
        include_self_in_aux=include_self_in_aux,
        selection_mode=selection_mode,
        uid=uid,
    )
    aux_rewards = await aux_scorer.score_many_async(
        question,
        responses,
        session=aux_session,
        semaphore=aux_semaphore,
    )

    tau_tasks = []
    for primary_idx, aux_indices in enumerate(aux_index_sets):
        aux_subset = [aux_rewards[aux_idx] for aux_idx in aux_indices]
        tau_tasks.append(
            tau_scorer.score_many_async(
                question,
                responses[primary_idx],
                aux_subset,
                session=tau_session,
                semaphore=tau_semaphore,
            )
        )
    tau_results = await asyncio.gather(*tau_tasks, return_exceptions=True)

    num_rows = len(responses)
    num_aux_per_row = len(aux_index_sets[0])
    tau_samples = np.full((num_rows, num_aux_per_row), np.nan, dtype=np.float64)
    scores = phi.astype(np.float64, copy=True)
    diagnostics = {
        "phi": phi.astype(np.float64, copy=False),
        "tau_control": np.full(num_rows, np.nan, dtype=np.float64),
        "tau_mc_mean": np.full(num_rows, np.nan, dtype=np.float64),
        "num_aux_samples": np.asarray([len(indices) for indices in aux_index_sets], dtype=np.int64),
        "row_fallback": np.ones(num_rows, dtype=np.int64),
    }

    successful_rows: list[int] = []
    successful_tau_rows: list[np.ndarray] = []
    row_errors: list[str | None] = [None] * num_rows
    for row_idx, tau_result in enumerate(tau_results):
        if isinstance(tau_result, Exception):
            row_errors[row_idx] = str(tau_result)
            continue
        tau_row = np.asarray(tau_result, dtype=np.float64)
        if tau_row.ndim != 1 or tau_row.shape[0] != num_aux_per_row:
            row_errors[row_idx] = (
                f"tau row has invalid shape {tau_row.shape}, expected ({num_aux_per_row},)."
            )
            continue
        tau_samples[row_idx] = tau_row
        successful_rows.append(row_idx)
        successful_tau_rows.append(tau_row)

    if successful_rows:
        successful_indices = np.asarray(successful_rows, dtype=np.int64)
        successful_scores, successful_diag = compute_online_eif_scores(
            phi[successful_indices],
            np.stack(successful_tau_rows, axis=0),
        )
        scores[successful_indices] = successful_scores
        diagnostics["tau_control"][successful_indices] = successful_diag["tau_control"]
        diagnostics["tau_mc_mean"][successful_indices] = successful_diag["tau_mc_mean"]
        diagnostics["num_aux_samples"][successful_indices] = successful_diag["num_aux_samples"]
        diagnostics["row_fallback"][successful_indices] = 0

    return {
        "scores": scores,
        "aux_rewards": np.asarray(aux_rewards, dtype=np.float64),
        "tau_samples": tau_samples,
        "aux_index_sets": aux_index_sets,
        "diagnostics": diagnostics,
        "row_errors": row_errors,
    }

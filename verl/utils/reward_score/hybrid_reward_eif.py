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
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from verl.utils.reward_score.one_step_eif import compute_algorithm1_scores


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





def compute_algorithm1_online_scores(
    phi_scores: np.ndarray | list[float],
    tau_scores: np.ndarray | list[float],
    m_scores: np.ndarray | list[float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    phi = np.asarray(phi_scores, dtype=np.float64)
    tau = np.asarray(tau_scores, dtype=np.float64)
    m = np.asarray(m_scores, dtype=np.float64)
    scores = compute_algorithm1_scores(phi, tau, m)
    diagnostics = {
        "phi": phi.astype(np.float64, copy=False),
        "tau": tau.astype(np.float64, copy=False),
        "m": m.astype(np.float64, copy=False),
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
    """Build prompt messages for the scalar auxiliary reward r_i."""
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


def build_reward_model_messages(question: str, response: str, raw_prompt: Any = None) -> list[dict[str, str]]:
    """Build a discriminative RM conversation for AceMath-style scoring."""
    prompt_messages = normalize_chat_prompt(raw_prompt)
    if prompt_messages:
        return [*prompt_messages, {"role": "assistant", "content": response}]
    if question.strip():
        return [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
        ]
    raise ValueError("Either `question` or `raw_prompt` must provide reward-model context.")


def build_reward_model_prompt(question: str, response: str, reward_model_tokenizer: Any, raw_prompt: Any = None) -> str:
    messages = build_reward_model_messages(question, response, raw_prompt=raw_prompt)
    rm_prompt = reward_model_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=False,
    )
    bos_token = getattr(reward_model_tokenizer, "bos_token", None)
    if bos_token is not None and isinstance(rm_prompt, str) and rm_prompt.startswith(bos_token):
        rm_prompt = rm_prompt[len(bos_token) :]
    return rm_prompt


def build_tau_messages(question: str, primary_response: str, aux_reward: float) -> list[dict[str, str]]:
    """Build prompt messages for tau_LLM, conditioned on the scalar auxiliary reward r_i."""
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


def build_m_messages(question: str, primary_response: str) -> list[dict[str, str]]:
    """Build prompt messages for m_LLM = E[phi | x, y]."""
    return [
        {
            "role": "system",
            "content": (
                "You estimate m(x, y): the probability that a math response is correct before seeing "
                "any auxiliary signal. Return only one number between 0 and 1."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Primary response:\n{primary_response}\n\n"
                "Estimate m(x, y) = P(phi=1 | question, response). Return only the number."
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


def parse_discriminative_rm_score(result: dict[str, Any], engine_name: str) -> float:
    if engine_name == "vllm":
        return float(result["data"][-1]["probs"][-1])
    if engine_name == "sglang":
        return float(result["data"][-1]["embedding"][-1])
    raise NotImplementedError(f"Unsupported discriminative RM backend: {engine_name}")


def _normalize_http_base(url: str) -> str:
    normalized = url.strip()
    if not normalized.startswith("http://") and not normalized.startswith("https://"):
        normalized = f"http://{normalized}"
    return normalized.rstrip("/")


@dataclass
class AceMathRewardScorer:
    """Discriminative AceMath RM scorer that reuses the reward-router API."""

    model: str = 'nvidia/AceMath-7B-RM'
    base_url: str = 'http://localhost:8000'
    engine_name: str = 'vllm'
    reward_model_tokenizer: Any = None
    timeout: float = 300.0

    def _build_payload(self, rm_prompt: str) -> tuple[str, dict[str, Any]]:
        if self.engine_name == 'vllm':
            return 'classify', {
                'model': self.model,
                'input': rm_prompt,
                'activation': False,
            }
        if self.engine_name == 'sglang':
            return 'v1/embeddings', {
                'model': self.model,
                'input': rm_prompt,
            }
        raise NotImplementedError(f'Unsupported discriminative RM backend: {self.engine_name}')

    async def score_async(
        self,
        question: str,
        response_text: str,
        *,
        raw_prompt: Any = None,
        session=None,
        semaphore=None,
    ) -> float:
        if self.reward_model_tokenizer is None:
            raise ValueError('reward_model_tokenizer is required for AceMathRewardScorer.')

        rm_prompt = build_reward_model_prompt(
            question,
            response_text,
            self.reward_model_tokenizer,
            raw_prompt=raw_prompt,
        )
        endpoint, payload = self._build_payload(rm_prompt)

        async def _score_once(active_session) -> float:
            url = f'{_normalize_http_base(self.base_url)}/{endpoint}'
            async with active_session.post(url, json=payload) as resp:
                resp.raise_for_status()
                result = await resp.json()
            return parse_discriminative_rm_score(result, self.engine_name)

        if session is None:
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as owned_session:
                if semaphore is None:
                    return await _score_once(owned_session)
                async with semaphore:
                    return await _score_once(owned_session)

        if semaphore is None:
            return await _score_once(session)
        async with semaphore:
            return await _score_once(session)

    def score(self, question: str, response_text: str, *, raw_prompt: Any = None) -> float:
        return asyncio.run(self.score_async(question, response_text, raw_prompt=raw_prompt))




@dataclass
class TauLLMScorer:
    model: str = "qwen3"
    base_url: str = "https://ellm.nrp-nautilus.io/v1"
    api_key_env: str = "NAUTILUS_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 16
    timeout: float = 300.0
    client: Any = None
    async_client: Any = None

    def _get_client(self):
        if self.client is not None:
            return self.client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The `openai` package is required to generate tau_LLM scores. "
                "Install it or configure a reachable tau_LLM endpoint."
            ) from exc

        self.client = OpenAI(api_key=os.environ.get(self.api_key_env), base_url=self.base_url, timeout=self.timeout)
        return self.client

    def _get_async_client(self):
        if self.async_client is not None:
            return self.async_client
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "The `openai` package is required to generate tau_LLM scores. "
                "Install it or configure a reachable tau_LLM endpoint."
            ) from exc

        self.async_client = AsyncOpenAI(
            api_key=os.environ.get(self.api_key_env),
            base_url=self.base_url,
            timeout=self.timeout,
        )
        return self.async_client

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
        async def _score_once() -> float:
            completion = await self._get_async_client().chat.completions.create(
                model=self.model,
                messages=build_tau_messages(question, primary_response, ace_reward),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = completion.choices[0].message.content
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


@dataclass
class MarginalLLMScorer:
    model: str = 'qwen3'
    base_url: str = 'https://ellm.nrp-nautilus.io/v1'
    api_key_env: str = 'NAUTILUS_API_KEY'
    temperature: float = 0.0
    max_tokens: int = 16
    timeout: float = 300.0
    client: Any = None
    async_client: Any = None

    def _get_client(self):
        if self.client is not None:
            return self.client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                'The `openai` package is required to generate m_LLM scores. '
                'Install it or provide precomputed m_llm_value.'
            ) from exc

        self.client = OpenAI(api_key=os.environ.get(self.api_key_env), base_url=self.base_url, timeout=self.timeout)
        return self.client

    def _get_async_client(self):
        if self.async_client is not None:
            return self.async_client
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                'The `openai` package is required to generate m_LLM scores. '
                'Install it or provide precomputed m_llm_value.'
            ) from exc

        self.async_client = AsyncOpenAI(
            api_key=os.environ.get(self.api_key_env),
            base_url=self.base_url,
            timeout=self.timeout,
        )
        return self.async_client

    def _build_headers(self) -> dict[str, str]:
        headers = {'Content-Type': 'application/json'}
        api_key = os.environ.get(self.api_key_env)
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        return headers

    def score(self, question: str, primary_response: str) -> float:
        completion = self._get_client().chat.completions.create(
            model=self.model,
            messages=build_m_messages(question, primary_response),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = completion.choices[0].message.content
        return parse_tau_score('' if content is None else str(content))

    async def score_async(
        self,
        question: str,
        primary_response: str,
        *,
        session=None,
        semaphore=None,
    ) -> float:
        async def _score_once() -> float:
            completion = await self._get_async_client().chat.completions.create(
                model=self.model,
                messages=build_m_messages(question, primary_response),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = completion.choices[0].message.content
            return parse_tau_score('' if content is None else str(content))

        if semaphore is None:
            return await _score_once()
        async with semaphore:
            return await _score_once()


async def compute_response_algorithm1_eif(
    *,
    question: str,
    response: str,
    phi_score: float,
    aux_scorer: AceMathRewardScorer,
    tau_scorer: TauLLMScorer,
    m_scorer: MarginalLLMScorer,
    raw_prompt: Any = None,
    aux_session=None,
    tau_session=None,
    m_session=None,
    aux_semaphore=None,
    tau_semaphore=None,
    m_semaphore=None,
) -> dict[str, Any]:
    """Compute the literal Algorithm 1 score for one (x_i, y_i, phi_i, r_i)."""
    aux_reward = await aux_scorer.score_async(
        question,
        response,
        raw_prompt=raw_prompt,
        session=aux_session,
        semaphore=aux_semaphore,
    )
    tau_score, m_score = await asyncio.gather(
        tau_scorer.score_async(
            question,
            response,
            aux_reward,
            session=tau_session,
            semaphore=tau_semaphore,
        ),
        m_scorer.score_async(
            question,
            response,
            session=m_session,
            semaphore=m_semaphore,
        ),
    )
    scores, diagnostics = compute_algorithm1_online_scores([phi_score], [tau_score], [m_score])
    return {
        'score': float(scores[0]),
        'aux_reward': float(aux_reward),
        'tau_score': float(tau_score),
        'm_score': float(m_score),
        'diagnostics': {
            'phi': float(diagnostics['phi'][0]),
            'tau': float(diagnostics['tau'][0]),
            'm': float(diagnostics['m'][0]),
        },
    }

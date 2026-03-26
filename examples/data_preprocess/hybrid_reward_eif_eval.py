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
"""Normalize evaluation parquet for one-step EIF estimation (Hybrid_reward.pdf).

NOTE: This is for evaluation with fixed (x_i, y_i) pairs only.
During training, y_i changes with the policy; use hybrid_eif_online mode instead.
"""

import argparse
import os
from typing import Any

import pandas as pd

from verl.utils import hf_tokenizer
from verl.utils.reward_score.hybrid_reward_eif import (
    AceMathRewardScorer,
    MarginalLLMScorer,
    TauLLMScorer,
    extract_question_text,
    select_primary_response,
)


def _normalize_float_scalar(value: Any, *, field_name: str) -> float:
    if value is None:
        raise ValueError(f"`{field_name}` is required but missing.")
    if hasattr(value, "tolist") and not isinstance(value, str):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"`{field_name}` must contain a single scalar value, got length={len(value)}.")
        value = value[0]
    return float(value)


def _row_has_value(row: pd.Series, key: str) -> bool:
    if key not in row:
        return False
    value = row[key]
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except TypeError:
        pass
    return True


def _resolve_reward_model(row: pd.Series, reward_model_key: str, ground_truth_key: str) -> dict[str, Any]:
    reward_model = row.get(reward_model_key, None)
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth")
        return reward_model if ground_truth is not None else {"style": "rule", "ground_truth": ground_truth}

    ground_truth = row.get(ground_truth_key)
    return {"style": "rule", "ground_truth": ground_truth}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Input parquet path.")
    parser.add_argument("--output_path", required=True, help="Output parquet path.")
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--response_key", default="responses")
    parser.add_argument("--data_source_key", default="data_source")
    parser.add_argument("--reward_model_key", default="reward_model")
    parser.add_argument("--ground_truth_key", default="ground_truth")
    parser.add_argument("--primary_response_index", type=int, default=0)
    parser.add_argument("--aux_reward_key", default="aux_reward_value")
    parser.add_argument("--tau_key", default="tau_llm_value")
    parser.add_argument("--m_key", default="m_llm_value")
    # AceMath reward model for r_i.
    parser.add_argument("--reward_model_path", default="nvidia/AceMath-7B-RM")
    parser.add_argument("--reward_base_url", default="http://localhost:8000")
    parser.add_argument("--reward_engine_name", default="vllm")
    parser.add_argument("--reward_model_tokenizer_path", default=None)
    parser.add_argument("--reward_timeout", type=float, default=300.0)
    # tau_LLM
    parser.add_argument("--tau_model", default="qwen3")
    parser.add_argument("--tau_base_url", default="https://ellm.nrp-nautilus.io/v1")
    parser.add_argument("--tau_api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--tau_temperature", type=float, default=0.0)
    parser.add_argument("--tau_max_tokens", type=int, default=16)
    # m_LLM
    parser.add_argument("--m_model", default="qwen3")
    parser.add_argument("--m_base_url", default=None)
    parser.add_argument("--m_api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--m_temperature", type=float, default=0.0)
    parser.add_argument("--m_max_tokens", type=int, default=16)
    args = parser.parse_args()
    if args.m_base_url is None:
        args.m_base_url = args.tau_base_url

    df = pd.read_parquet(os.path.expanduser(args.input_path))

    required = [args.prompt_key, args.response_key, args.data_source_key]
    for key in required:
        if key not in df.columns:
            raise ValueError(f"Missing required column `{key}` in input parquet.")

    if args.reward_model_key not in df.columns and args.ground_truth_key not in df.columns:
        raise ValueError(
            f"Either `{args.reward_model_key}` (dict with ground_truth) or `{args.ground_truth_key}` must exist."
        )

    ace_scorer = None
    tau_scorer = None
    m_scorer = None

    normalized_rows = []
    for _, row in df.iterrows():
        reward_model = _resolve_reward_model(row, args.reward_model_key, args.ground_truth_key)
        raw_prompt = row[args.prompt_key]
        question = extract_question_text(raw_prompt, fallback=row[args.prompt_key])
        primary_response = select_primary_response(row[args.response_key], primary_response_index=args.primary_response_index)

        if _row_has_value(row, args.tau_key) or _row_has_value(row, args.m_key):
            if not _row_has_value(row, args.tau_key) or not _row_has_value(row, args.m_key):
                raise ValueError(
                    f"Rows must provide both `{args.tau_key}` and `{args.m_key}` when either one is present."
                )
            aux_reward_value = (
                _normalize_float_scalar(row[args.aux_reward_key], field_name=args.aux_reward_key)
                if _row_has_value(row, args.aux_reward_key)
                else None
            )
            tau_value = _normalize_float_scalar(row[args.tau_key], field_name=args.tau_key)
            m_value = _normalize_float_scalar(row[args.m_key], field_name=args.m_key)
            normalized_row = {
                "prompt": question,
                "responses": [primary_response],
                "data_source": row[args.data_source_key],
                "reward_model": reward_model,
                args.tau_key: tau_value,
                args.m_key: m_value,
            }
            if aux_reward_value is not None:
                normalized_row[args.aux_reward_key] = aux_reward_value
            normalized_rows.append(normalized_row)
            continue

        if _row_has_value(row, args.aux_reward_key):
            aux_reward_value = _normalize_float_scalar(row[args.aux_reward_key], field_name=args.aux_reward_key)
        else:
            if ace_scorer is None:
                reward_model_tokenizer = hf_tokenizer(
                    args.reward_model_tokenizer_path or args.reward_model_path,
                    trust_remote_code=True,
                )
                ace_scorer = AceMathRewardScorer(
                    model=args.reward_model_path,
                    base_url=args.reward_base_url,
                    engine_name=args.reward_engine_name,
                    reward_model_tokenizer=reward_model_tokenizer,
                    timeout=args.reward_timeout,
                )
            aux_reward_value = ace_scorer.score(question, primary_response, raw_prompt=raw_prompt)

        if tau_scorer is None:
            tau_scorer = TauLLMScorer(
                model=args.tau_model,
                base_url=args.tau_base_url,
                api_key_env=args.tau_api_key_env,
                temperature=args.tau_temperature,
                max_tokens=args.tau_max_tokens,
            )
        if m_scorer is None:
            m_scorer = MarginalLLMScorer(
                model=args.m_model,
                base_url=args.m_base_url,
                api_key_env=args.m_api_key_env,
                temperature=args.m_temperature,
                max_tokens=args.m_max_tokens,
            )

        tau_value = tau_scorer.score(question, primary_response, aux_reward_value)
        m_value = m_scorer.score(question, primary_response)

        normalized_rows.append(
            {
                "prompt": question,
                "responses": [primary_response],
                "data_source": row[args.data_source_key],
                "reward_model": reward_model,
                args.aux_reward_key: float(aux_reward_value),
                args.tau_key: float(tau_value),
                args.m_key: float(m_value),
            }
        )

    out_df = pd.DataFrame(normalized_rows)
    out_df.to_parquet(os.path.expanduser(args.output_path))


if __name__ == "__main__":
    main()

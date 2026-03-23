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
During training, y_i changes with the policy — use hybrid_eif_online mode instead.
"""

import argparse
import os
from typing import Any

import pandas as pd

from verl.utils.reward_score.hybrid_reward_eif import (
    AuxRewardScorer,
    TauLLMScorer,
    normalize_float_list,
    normalize_string_list,
    resolve_auxiliary_response_bundle,
)


def _parse_csv_arg(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    return parsed or None


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


def _resolve_aux_response_value(row: pd.Series, aux_response_key: str | None, aux_response_cols: list[str] | None):
    if aux_response_cols:
        return [row[col] for col in aux_response_cols]
    if aux_response_key and _row_has_value(row, aux_response_key):
        return row[aux_response_key]
    return None


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
    parser.add_argument("--num_aux_samples", type=int, default=None, help="Optional truncation length for M+1 samples.")
    parser.add_argument(
        "--exclude_primary_from_aux",
        action="store_true",
        help="If set, exclude the designated primary response from the fallback auxiliary response bundle.",
    )
    parser.add_argument(
        "--aux_response_key",
        default=None,
        help="Optional list-like column with auxiliary responses. If omitted, a list-valued response column is reused.",
    )
    parser.add_argument(
        "--aux_response_cols",
        default=None,
        help="Optional comma-separated scalar columns to combine into an auxiliary response list.",
    )
    parser.add_argument("--aux_reward_key", default="aux_reward_values")
    parser.add_argument("--aux_tau_key", default="aux_tau_values")
    parser.add_argument(
        "--aux_cols",
        default=None,
        help="Optional comma-separated scalar columns to combine into aux tau list. "
        "If provided, overrides --aux_tau_key.",
    )
    # Auxiliary reward model (r_ij generator)
    parser.add_argument("--aux_model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--aux_base_url", default="http://localhost:8000/v1")
    parser.add_argument("--aux_api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--aux_temperature", type=float, default=1.0)
    parser.add_argument("--aux_max_tokens", type=int, default=16)
    # Tau LLM
    parser.add_argument("--tau_model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--tau_base_url", default="http://localhost:8001/v1")
    parser.add_argument("--tau_api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--tau_temperature", type=float, default=0.0)
    parser.add_argument("--tau_max_tokens", type=int, default=16)
    args = parser.parse_args()

    df = pd.read_parquet(os.path.expanduser(args.input_path))

    required = [args.prompt_key, args.response_key, args.data_source_key]
    for key in required:
        if key not in df.columns:
            raise ValueError(f"Missing required column `{key}` in input parquet.")

    if args.reward_model_key not in df.columns and args.ground_truth_key not in df.columns:
        raise ValueError(
            f"Either `{args.reward_model_key}` (dict with ground_truth) or `{args.ground_truth_key}` must exist."
        )

    aux_cols = _parse_csv_arg(args.aux_cols)
    aux_response_cols = _parse_csv_arg(args.aux_response_cols)

    if aux_cols:
        missing_aux_cols = [col for col in aux_cols if col not in df.columns]
        if missing_aux_cols:
            raise ValueError(f"Missing aux tau columns: {missing_aux_cols}")
    if aux_response_cols:
        missing_aux_response_cols = [col for col in aux_response_cols if col not in df.columns]
        if missing_aux_response_cols:
            raise ValueError(f"Missing auxiliary response columns: {missing_aux_response_cols}")
    if args.aux_response_key and args.aux_response_key not in df.columns:
        raise ValueError(f"Missing auxiliary response column `{args.aux_response_key}`.")

    aux_scorer = None
    tau_scorer = None

    normalized_rows = []
    for _, row in df.iterrows():
        reward_model = _resolve_reward_model(row, args.reward_model_key, args.ground_truth_key)
        question = str(row[args.prompt_key])
        aux_response_value = _resolve_aux_response_value(row, args.aux_response_key, aux_response_cols)
        response_values = normalize_string_list(row[args.response_key], field_name=args.response_key)
        primary_response, aux_responses = resolve_auxiliary_response_bundle(
            row[args.response_key],
            primary_response_index=args.primary_response_index,
            aux_response_value=aux_response_value,
            include_primary_response=not args.exclude_primary_from_aux,
            max_aux_samples=args.num_aux_samples,
        )
        has_explicit_aux_bundle = aux_response_value is not None or len(response_values) > 1
        aux_reward_values: list[float] | None = None

        if aux_cols:
            aux_tau_values = normalize_float_list([row[col] for col in aux_cols], field_name="aux_cols")
        elif _row_has_value(row, args.aux_tau_key):
            aux_tau_values = normalize_float_list(row[args.aux_tau_key], field_name=args.aux_tau_key)
        else:
            if _row_has_value(row, args.aux_reward_key):
                aux_reward_values = normalize_float_list(row[args.aux_reward_key], field_name=args.aux_reward_key)
            else:
                if not aux_responses:
                    raise ValueError(
                        "Missing auxiliary rewards and auxiliary responses for a row. "
                        "Provide aux tau values, aux reward values, or an auxiliary response bundle."
                    )
                if aux_scorer is None:
                    aux_scorer = AuxRewardScorer(
                        model=args.aux_model,
                        base_url=args.aux_base_url,
                        api_key_env=args.aux_api_key_env,
                        temperature=args.aux_temperature,
                        max_tokens=args.aux_max_tokens,
                    )
                if has_explicit_aux_bundle:
                    reward_inputs = aux_responses
                else:
                    if args.num_aux_samples is None:
                        raise ValueError(
                            "Repeated auxiliary reward sampling for a single response requires --num_aux_samples >= 2."
                        )
                    reward_inputs = [primary_response] * args.num_aux_samples
                aux_reward_values = aux_scorer.score_many(question, reward_inputs)
            if tau_scorer is None:
                tau_scorer = TauLLMScorer(
                    model=args.tau_model,
                    base_url=args.tau_base_url,
                    api_key_env=args.tau_api_key_env,
                    temperature=args.tau_temperature,
                    max_tokens=args.tau_max_tokens,
                )
            aux_tau_values = tau_scorer.score_many(question, primary_response, aux_reward_values)

        if aux_reward_values is None and _row_has_value(row, args.aux_reward_key):
            aux_reward_values = normalize_float_list(row[args.aux_reward_key], field_name=args.aux_reward_key)
        elif aux_reward_values is None:
            aux_reward_values = []

        if args.num_aux_samples is not None:
            aux_tau_values = aux_tau_values[: args.num_aux_samples]
            aux_reward_values = aux_reward_values[: args.num_aux_samples]

        aux_tau_values = normalize_float_list(aux_tau_values, field_name="aux_tau_values")
        if aux_reward_values:
            aux_reward_values = normalize_float_list(aux_reward_values, field_name="aux_reward_values")

        normalized_rows.append(
            {
                "prompt": question,
                "responses": [primary_response],
                "data_source": row[args.data_source_key],
                "reward_model": reward_model,
                "aux_reward_values": aux_reward_values,
                "aux_tau_values": aux_tau_values,
            }
        )

    out_df = pd.DataFrame(normalized_rows)
    out_df.to_parquet(os.path.expanduser(args.output_path))


if __name__ == "__main__":
    main()

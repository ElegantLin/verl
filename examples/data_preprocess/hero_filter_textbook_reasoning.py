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
"""Filter TextBookReasoning for HERO-style hard-to-verify evaluation.

This script implements a practical paper-aligned pipeline:
1. keep only questions whose ground-truth answer does not pass `math_verify`
2. optionally drop questions that an answer model can already solve
3. optionally apply the Figure 3 suitability filter with an OpenAI-compatible model
"""

import argparse
import asyncio
import json
import os
import re
from typing import Any

import datasets
from openai import OpenAI
from tqdm.asyncio import tqdm_asyncio


SUITABILITY_PROMPT_TEMPLATE = """I am looking for math questions that are suitable for evaluating a math model. Please help me select questions that meet the following criteria:
1. The question must be clear and unambiguous.
2. The question must have a specific, factual, and answerable solution (not open-ended or subjective).
3. The question must NOT require a proof or explanation of reasoning.
4. The question must NOT be a statement; it should be a direct question.
For each question I provide, please respond with:
- "Conclusion: Suitable" in the end if the question meets all the criteria above.
- "Conclusion: Not Suitable"
If the question does not meet the criteria, briefly explain why.

Question:
{question}
"""

ANSWER_PROMPT_TEMPLATE = """Solve the following question and give your final answer.

Question:
{question}
"""

JUDGE_PROMPT_TEMPLATE = """### Question: {question}

### Ground Truth Answer: {ground_truth}

### Student Answer: {student_answer}

For the above question, please verify if the student's answer is equivalent to the ground truth answer.
Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.
If the student's answer is correct, output "Final Decision: Yes". If the student's answer is incorrect, output "Final Decision: No"."""

DEFAULT_OPENAI_COMPAT_BASE_URL = "https://ellm.nrp-nautilus.io/v1"
DEFAULT_JUDGE_MODEL = "gpt-oss"


def _resolve_dataset_split(dataset_obj: datasets.Dataset | datasets.DatasetDict, split: str) -> datasets.Dataset:
    if isinstance(dataset_obj, datasets.DatasetDict):
        if split in dataset_obj:
            return dataset_obj[split]
        if "train" in dataset_obj:
            return dataset_obj["train"]
        first_split = next(iter(dataset_obj.keys()))
        return dataset_obj[first_split]
    return dataset_obj


def _load_input_dataset(local_dataset_path: str | None, dataset_name: str, split: str) -> datasets.Dataset:
    if local_dataset_path is not None:
        local_dataset_path = os.path.expanduser(local_dataset_path)
        if not os.path.exists(local_dataset_path):
            raise FileNotFoundError(f"Input dataset path does not exist: {local_dataset_path}")
        if os.path.isdir(local_dataset_path):
            try:
                return _resolve_dataset_split(datasets.load_from_disk(local_dataset_path), split)
            except Exception:
                return datasets.load_dataset(local_dataset_path, split=split)
        suffix = os.path.splitext(local_dataset_path)[1].lower()
        if suffix == ".parquet":
            return datasets.load_dataset("parquet", data_files=local_dataset_path, split="train")
        if suffix in {".json", ".jsonl"}:
            return datasets.load_dataset("json", data_files=local_dataset_path, split="train")
        if suffix == ".csv":
            return datasets.load_dataset("csv", data_files=local_dataset_path, split="train")
        return datasets.load_dataset(local_dataset_path, split=split)
    return datasets.load_dataset(dataset_name, split=split)


def _self_math_verify_fails(answer: str) -> bool:
    try:
        from verl.utils.reward_score.math_verify import compute_score as math_verify_score

        return float(math_verify_score(answer, answer)) <= 0.5
    except Exception:
        return True


def _parse_final_decision(response: str) -> bool:
    match = re.search(r"Final Decision:\s*(Yes|No)", response, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "yes"
    return response.strip().lower().endswith("yes")


def _parse_suitability(response: str) -> bool:
    match = re.search(r"Conclusion:\s*(Suitable|Not Suitable)", response, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "suitable"
    return "conclusion: suitable" in response.strip().lower()


async def _chat_once(client: OpenAI, *, model: str, prompt: str, temperature: float, max_tokens: int, semaphore: asyncio.Semaphore) -> str:
    async with semaphore:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return response.choices[0].message.content or ""


async def _run_answerability_filter(rows, args):
    answer_client = OpenAI(
        api_key=os.environ.get(args.answer_api_key_env, ""),
        base_url=args.answer_base_url or DEFAULT_OPENAI_COMPAT_BASE_URL,
    )
    judge_client = OpenAI(
        api_key=os.environ.get(args.answer_judge_api_key_env, ""),
        base_url=args.answer_judge_base_url or DEFAULT_OPENAI_COMPAT_BASE_URL,
    )
    answer_semaphore = asyncio.Semaphore(args.answer_concurrency)
    judge_semaphore = asyncio.Semaphore(args.answer_judge_concurrency)

    async def _process_row(row):
        answer_text = await _chat_once(
            answer_client,
            model=args.answer_model,
            prompt=ANSWER_PROMPT_TEMPLATE.format(question=row["question"]),
            temperature=args.answer_temperature,
            max_tokens=args.answer_max_tokens,
            semaphore=answer_semaphore,
        )
        judge_text = await _chat_once(
            judge_client,
            model=args.answer_judge_model,
            prompt=JUDGE_PROMPT_TEMPLATE.format(
                question=row["question"],
                ground_truth=row["answer"],
                student_answer=answer_text,
            ),
            temperature=0.0,
            max_tokens=512,
            semaphore=judge_semaphore,
        )
        row = dict(row)
        row["answerability_model_response"] = answer_text
        row["answerability_judge_response"] = judge_text
        row["answerability_model_correct"] = _parse_final_decision(judge_text)
        return row

    processed = await tqdm_asyncio.gather(*[_process_row(row) for row in rows])
    return [row for row in processed if not row["answerability_model_correct"]]


async def _run_suitability_filter(rows, args):
    client = OpenAI(
        api_key=os.environ.get(args.suitability_api_key_env, ""),
        base_url=args.suitability_base_url or DEFAULT_OPENAI_COMPAT_BASE_URL,
    )
    semaphore = asyncio.Semaphore(args.suitability_concurrency)

    async def _process_row(row):
        prompt = SUITABILITY_PROMPT_TEMPLATE.format(question=row["question"])
        response = await _chat_once(
            client,
            model=args.suitability_model,
            prompt=prompt,
            temperature=0.0,
            max_tokens=256,
            semaphore=semaphore,
        )
        row = dict(row)
        row["suitability_filter_response"] = response
        row["suitability_filter_keep"] = _parse_suitability(response)
        return row

    processed = await tqdm_asyncio.gather(*[_process_row(row) for row in rows])
    return [row for row in processed if row["suitability_filter_keep"]]


def main():
    parser = argparse.ArgumentParser(description="Filter TextBookReasoning for HERO evaluation")
    parser.add_argument("--dataset", default="MegaScience/TextBookReasoning")
    parser.add_argument("--split", default="train")
    parser.add_argument("--local_dataset_path", default=None)
    parser.add_argument("--question_col", default="question")
    parser.add_argument("--answer_col", default="answer")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--disable_math_verify_filter", action="store_true")
    parser.add_argument("--answer_model", default=None, help="Optional answer model for the Llama-style answerability filter.")
    parser.add_argument("--answer_base_url", default=DEFAULT_OPENAI_COMPAT_BASE_URL)
    parser.add_argument("--answer_api_key_env", default="NAUTILUS_API_KEY")
    parser.add_argument("--answer_temperature", type=float, default=0.0)
    parser.add_argument("--answer_max_tokens", type=int, default=512)
    parser.add_argument("--answer_concurrency", type=int, default=32)
    parser.add_argument("--answer_judge_model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--answer_judge_base_url", default=DEFAULT_OPENAI_COMPAT_BASE_URL)
    parser.add_argument("--answer_judge_api_key_env", default="NAUTILUS_API_KEY")
    parser.add_argument("--answer_judge_concurrency", type=int, default=32)
    parser.add_argument("--suitability_model", default=None, help="Optional GPT-style suitability filter model.")
    parser.add_argument("--suitability_base_url", default=DEFAULT_OPENAI_COMPAT_BASE_URL)
    parser.add_argument("--suitability_api_key_env", default="NAUTILUS_API_KEY")
    parser.add_argument("--suitability_concurrency", type=int, default=32)
    args = parser.parse_args()

    dataset = _load_input_dataset(args.local_dataset_path, args.dataset, args.split)
    if args.question_col not in dataset.column_names or args.answer_col not in dataset.column_names:
        question_col = args.question_col if args.question_col in dataset.column_names else "problem"
        answer_col = args.answer_col if args.answer_col in dataset.column_names else "solution"
    else:
        question_col = args.question_col
        answer_col = args.answer_col

    rows = []
    for idx, example in enumerate(dataset):
        rows.append({
            "index": idx,
            "question": str(example[question_col]).strip(),
            "answer": str(example[answer_col]).strip(),
        })

    summary = {"initial_rows": len(rows)}

    if not args.disable_math_verify_filter:
        rows = [row for row in rows if _self_math_verify_fails(row["answer"])]
    summary["after_math_verify_filter"] = len(rows)

    if args.answer_model:
        rows = asyncio.run(_run_answerability_filter(rows, args))
    summary["after_answerability_filter"] = len(rows)

    if args.suitability_model:
        rows = asyncio.run(_run_suitability_filter(rows, args))
    summary["after_suitability_filter"] = len(rows)

    output_path = os.path.expanduser(args.output_path)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    datasets.Dataset.from_list(rows).to_parquet(output_path)
    summary["output_path"] = output_path
    summary_path = f"{output_path}.meta.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {len(rows)} filtered rows to {output_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()

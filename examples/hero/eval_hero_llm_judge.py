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
"""Evaluate hard-to-verify tasks using an OpenAI-compatible LLM judge.

Implements the evaluation protocol from HERO paper (arXiv:2510.07242v1, Figure 4):
An external judge model compares model outputs against ground-truth answers
without re-solving.

Usage:
  python examples/hero/eval_hero_llm_judge.py \
      --input_parquet /path/to/generated_responses.parquet \
      --judge_model gpt-oss \
      --base_url https://ellm.nrp-nautilus.io/v1 \
      --output_path /path/to/results.json
"""

import argparse
import asyncio
import json
import os
import re
from collections import defaultdict

import pandas as pd
from tqdm.asyncio import tqdm_asyncio

# Paper's hard-to-verify evaluation prompt (Figure 4)
JUDGE_PROMPT_TEMPLATE = """\
### Question: {question}

### Ground Truth Answer: {ground_truth}

### Student Answer: {student_answer}

For the above question, please verify if the student's answer is equivalent to the ground truth answer.
Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.
If the student's answer is correct, output "Final Decision: Yes". If the student's answer is incorrect, output "Final Decision: No"."""

DEFAULT_OPENAI_COMPAT_BASE_URL = "https://ellm.nrp-nautilus.io/v1"
DEFAULT_JUDGE_MODEL = "gpt-oss"


def parse_judge_decision(response: str) -> bool:
    """Extract the final decision from the judge's response."""
    match = re.search(r"Final Decision:\s*(Yes|No)", response, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "yes"
    response_lower = response.strip().lower()
    if response_lower.endswith("yes"):
        return True
    return False


def select_primary_response(responses, primary_response_index: int):
    """Select a single response from a possibly list-valued response column."""
    if not isinstance(responses, list):
        return responses
    if not responses:
        return ""

    index = primary_response_index
    if index < 0:
        index += len(responses)
    if index < 0 or index >= len(responses):
        raise IndexError(
            f"primary_response_index={primary_response_index} is out of range for {len(responses)} responses."
        )
    return responses[index]


async def judge_single(
    client,
    question: str,
    ground_truth: str,
    student_answer: str,
    model: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Judge a single response using the LLM judge."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        ground_truth=ground_truth,
        student_answer=student_answer,
    )

    async with semaphore:
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )
            judge_text = response.choices[0].message.content
            is_correct = parse_judge_decision(judge_text)
            return {
                "is_correct": is_correct,
                "judge_response": judge_text,
                "error": None,
            }
        except Exception as e:
            return {
                "is_correct": False,
                "judge_response": None,
                "error": str(e),
            }


async def run_evaluation(args):
    """Run LLM-as-judge evaluation on the designated primary response."""
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get(args.api_key_env, ""),
        base_url=args.base_url or DEFAULT_OPENAI_COMPAT_BASE_URL,
    )

    df = pd.read_parquet(args.input_parquet)
    print(f"Loaded {len(df)} rows from {args.input_parquet}")

    prompt_col = args.prompt_col
    response_col = args.response_col
    data_source_col = args.data_source_col
    reward_model_col = args.reward_model_col

    semaphore = asyncio.Semaphore(args.concurrency)

    tasks = []
    metadata = []

    for idx, row in df.iterrows():
        prompt_data = row[prompt_col]
        if isinstance(prompt_data, list):
            question = prompt_data[-1]["content"] if prompt_data else ""
        elif isinstance(prompt_data, str):
            question = prompt_data
        else:
            question = str(prompt_data)

        reward_data = row[reward_model_col]
        if isinstance(reward_data, str):
            reward_data = json.loads(reward_data)
        ground_truth = reward_data.get("ground_truth", "")

        response = select_primary_response(row[response_col], args.primary_response_index)
        data_source = row[data_source_col]

        tasks.append(
            judge_single(client, question, ground_truth, response, args.judge_model, semaphore)
        )
        metadata.append({
            "row_idx": idx,
            "resp_idx": args.primary_response_index,
            "data_source": data_source,
            "question": question[:200],
            "ground_truth": ground_truth[:200],
        })

    print(f"Evaluating {len(tasks)} responses with {args.judge_model}...")
    results = await tqdm_asyncio.gather(*tasks)

    source_scores = defaultdict(list)
    source_errors = defaultdict(int)
    detailed_results = []

    for meta, result in zip(metadata, results):
        source = meta["data_source"]
        if result["error"]:
            source_errors[source] += 1
        source_scores[source].append(1.0 if result["is_correct"] else 0.0)
        detailed_results.append({**meta, **result})

    print("\n" + "=" * 60)
    print("HERO Hard-to-Verify Evaluation Results (LLM-as-Judge)")
    print("=" * 60)

    summary = {}
    for source in sorted(source_scores.keys()):
        scores = source_scores[source]
        acc = sum(scores) / len(scores) * 100 if scores else 0.0
        errors = source_errors[source]
        summary[source] = {
            "pass_at_1": round(acc, 1),
            "total": len(scores),
            "correct": int(sum(scores)),
            "errors": errors,
        }
        print(f"  {source:25s}  pass@1={acc:5.1f}%  ({int(sum(scores))}/{len(scores)}, {errors} errors)")

    all_scores = []
    for scores in source_scores.values():
        all_scores.extend(scores)
    if all_scores:
        overall = sum(all_scores) / len(all_scores) * 100
        summary["overall"] = {"pass_at_1": round(overall, 1), "total": len(all_scores)}
        print(f"  {'OVERALL':25s}  pass@1={overall:5.1f}%")

    print("=" * 60)

    if args.output_path:
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump({"summary": summary, "details": detailed_results}, f, indent=2, default=str)
        print(f"\nDetailed results saved to {args.output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="HERO LLM-as-Judge Evaluation")
    parser.add_argument("--input_parquet", required=True, help="Parquet file with generated responses.")
    parser.add_argument("--judge_model", default=DEFAULT_JUDGE_MODEL, help="Model to use as judge.")
    parser.add_argument("--base_url", default=DEFAULT_OPENAI_COMPAT_BASE_URL, help="OpenAI-compatible API base URL.")
    parser.add_argument("--api_key_env", default="OPENAI_API_KEY", help="Env var for API key.")
    parser.add_argument("--concurrency", type=int, default=32, help="Max concurrent API requests.")
    parser.add_argument("--output_path", default=None, help="Path to save detailed results JSON.")
    parser.add_argument(
        "--primary_response_index",
        type=int,
        default=0,
        help="If `responses` is a list, evaluate only this response index. HERO paper uses 0.",
    )
    parser.add_argument("--prompt_col", default="prompt")
    parser.add_argument("--response_col", default="responses")
    parser.add_argument("--data_source_col", default="data_source")
    parser.add_argument("--reward_model_col", default="reward_model")
    args = parser.parse_args()

    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()

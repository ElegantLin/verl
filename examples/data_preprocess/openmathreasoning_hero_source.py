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
"""Build a prompt-only OpenMathReasoning parquet for HERO candidate generation.

The output parquet keeps `question` / `answer` columns so it can be fed into
`main_generation_server` and then reused by:
- `openmathreasoning_hero.py` to build RL splits from generated `responses`
- `openmathreasoning_hero_sft.py` to build cold-start SFT data
"""

import argparse
import json
import os
from typing import Any

import datasets


DEFAULT_INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def _resolve_dataset_split(dataset_obj: datasets.Dataset | datasets.DatasetDict, split: str) -> datasets.Dataset:
    if isinstance(dataset_obj, datasets.DatasetDict):
        if split in dataset_obj:
            return dataset_obj[split]
        if "train" in dataset_obj:
            return dataset_obj["train"]
        first_split = next(iter(dataset_obj.keys()))
        return dataset_obj[first_split]
    return dataset_obj


def _load_local_dataset(local_dataset_path: str, split: str) -> datasets.Dataset:
    local_dataset_path = os.path.expanduser(local_dataset_path)
    if not os.path.exists(local_dataset_path):
        raise FileNotFoundError(f"Local dataset path does not exist: {local_dataset_path}")

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


def _load_source_dataset(args: argparse.Namespace) -> datasets.Dataset:
    if args.local_dataset_path is not None:
        return _load_local_dataset(args.local_dataset_path, args.split)

    load_kwargs: dict[str, Any] = {"split": args.split}
    if args.dataset_config is not None:
        load_kwargs["name"] = args.dataset_config
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    return datasets.load_dataset(args.dataset, **load_kwargs)


def main():
    parser = argparse.ArgumentParser(description="Build prompt-only OpenMathReasoning parquet for HERO")
    parser.add_argument("--dataset", default="OpenMathReasoning")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--local_dataset_path", default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--question_col", default="question")
    parser.add_argument("--answer_col", default="answer")
    parser.add_argument("--problem_type_col", default="problem_type")
    parser.add_argument("--problem_type_value", default="has_answer_extracted")
    parser.add_argument("--source_sample_size", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--output_path", default="~/data/openmathreasoning_hero/source_prompts.parquet")
    args = parser.parse_args()

    dataset = _load_source_dataset(args)

    if args.question_col not in dataset.column_names or args.answer_col not in dataset.column_names:
        raise ValueError(
            f"Expected `{args.question_col}` and `{args.answer_col}` in dataset columns, got {dataset.column_names}"
        )

    if args.problem_type_col in dataset.column_names:
        dataset = dataset.filter(lambda x: x[args.problem_type_col] == args.problem_type_value)

    if args.source_sample_size > 0 and len(dataset) > args.source_sample_size:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.source_sample_size))

    records = []
    for idx, example in enumerate(dataset):
        question = str(example[args.question_col]).strip()
        answer = str(example[args.answer_col]).strip()
        prompt = f"{question} {args.instruction}".strip()
        record = {
            "question": question,
            "answer": answer,
            "data_source": "math_dapo_reasoning",
            "prompt": [{"role": "user", "content": prompt}],
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"index": idx, "split_source": args.split},
        }
        if args.problem_type_col in example:
            record[args.problem_type_col] = example[args.problem_type_col]
        records.append(record)

    output_path = os.path.expanduser(args.output_path)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    ds_out = datasets.Dataset.from_list(records)
    ds_out.to_parquet(output_path)

    meta_path = f"{output_path}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(
            {
                "dataset": args.local_dataset_path if args.local_dataset_path is not None else args.dataset,
                "dataset_config": args.dataset_config,
                "split": args.split,
                "source_sample_size": args.source_sample_size,
                "counts": {"rows": len(ds_out)},
                "question_col": args.question_col,
                "answer_col": args.answer_col,
                "problem_type_filter": {args.problem_type_col: args.problem_type_value},
                "output_path": output_path,
            },
            f,
            indent=2,
        )

    print(f"Saved {len(ds_out)} source prompts to {output_path}")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()

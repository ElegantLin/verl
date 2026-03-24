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
"""Build cold-start SFT data for HERO from a generated OpenMathReasoning parquet.

The paper states that cold-start SFT keeps only model-generated responses that:
- contain the correct final answer
- are entirely in English
- do not show obvious runaway / unstop behavior
"""

import argparse
import json
import os
import re
from typing import Any

import datasets


CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")


def _resolve_dataset_split(dataset_obj: datasets.Dataset | datasets.DatasetDict, split: str) -> datasets.Dataset:
    if isinstance(dataset_obj, datasets.DatasetDict):
        if split in dataset_obj:
            return dataset_obj[split]
        if "train" in dataset_obj:
            return dataset_obj["train"]
        first_split = next(iter(dataset_obj.keys()))
        return dataset_obj[first_split]
    return dataset_obj


def _load_input_dataset(local_path: str, split: str = "train") -> datasets.Dataset:
    local_path = os.path.expanduser(local_path)
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Input dataset path does not exist: {local_path}")

    if os.path.isdir(local_path):
        try:
            return _resolve_dataset_split(datasets.load_from_disk(local_path), split)
        except Exception:
            return datasets.load_dataset(local_path, split=split)

    suffix = os.path.splitext(local_path)[1].lower()
    if suffix == ".parquet":
        return datasets.load_dataset("parquet", data_files=local_path, split="train")
    if suffix in {".json", ".jsonl"}:
        return datasets.load_dataset("json", data_files=local_path, split="train")
    if suffix == ".csv":
        return datasets.load_dataset("csv", data_files=local_path, split="train")
    return datasets.load_dataset(local_path, split=split)


def _verify(solution: str, ground_truth: str) -> float:
    try:
        from verl.utils.reward_score.math_verify import compute_score as math_verify_score

        return float(math_verify_score(solution, ground_truth))
    except Exception:
        from verl.utils.reward_score.math_reward import compute_score as math_reward_score

        return float(math_reward_score(solution, ground_truth))


def _select_primary_response(response_value: Any, primary_response_index: int) -> str:
    if isinstance(response_value, (list, tuple)):
        if not response_value:
            return ""
        index = primary_response_index
        if index < 0:
            index += len(response_value)
        if index < 0 or index >= len(response_value):
            raise IndexError(
                f"primary_response_index={primary_response_index} is out of range for {len(response_value)} responses."
            )
        return str(response_value[index]).strip()
    return str(response_value).strip()


def _extract_ground_truth(example: dict[str, Any], answer_col: str, reward_model_col: str) -> str:
    if answer_col in example:
        return str(example[answer_col]).strip()
    reward_model = example.get(reward_model_col, {})
    if isinstance(reward_model, str):
        reward_model = json.loads(reward_model)
    return str(reward_model.get("ground_truth", "")).strip()


def _extract_prompt_messages(example: dict[str, Any], prompt_col: str, question_col: str) -> list[dict[str, str]]:
    prompt = example.get(prompt_col)
    if isinstance(prompt, list) and prompt:
        return [dict(message) for message in prompt]
    if isinstance(prompt, str) and prompt.strip():
        return [{"role": "user", "content": prompt.strip()}]
    question = str(example.get(question_col, "")).strip()
    return [{"role": "user", "content": question}]


def _contains_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


def _has_unstop_issue(text: str, max_response_chars: int) -> bool:
    if len(text) > max_response_chars:
        return True

    collapsed = re.sub(r"\s+", " ", text[-1024:]).strip()
    for window in (32, 64, 128):
        if len(collapsed) < window * 3:
            continue
        tail = collapsed[-window:]
        if tail and tail * 3 in collapsed:
            return True

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 4 and len(set(lines[-4:])) == 1:
        return True

    return False


def main():
    parser = argparse.ArgumentParser(description="Build cold-start SFT data for HERO")
    parser.add_argument("--input_path", required=True, help="Generated parquet / dataset path with responses.")
    parser.add_argument("--question_col", default="question")
    parser.add_argument("--answer_col", default="answer")
    parser.add_argument("--prompt_col", default="prompt")
    parser.add_argument("--reward_model_col", default="reward_model")
    parser.add_argument("--response_col", default="responses")
    parser.add_argument("--primary_response_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--val_size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_response_chars", type=int, default=12000)
    parser.add_argument("--output_dir", default="~/data/openmathreasoning_hero_sft")
    args = parser.parse_args()

    dataset = _load_input_dataset(args.input_path)
    if args.response_col not in dataset.column_names:
        raise ValueError(f"Missing response column `{args.response_col}` in dataset columns {dataset.column_names}")

    selected = []
    for idx, example in enumerate(dataset):
        response = _select_primary_response(example[args.response_col], args.primary_response_index)
        if not response:
            continue

        ground_truth = _extract_ground_truth(example, args.answer_col, args.reward_model_col)
        if _verify(response, ground_truth) <= 0.5:
            continue
        if _contains_cjk(response):
            continue
        if _has_unstop_issue(response, args.max_response_chars):
            continue

        messages = _extract_prompt_messages(example, args.prompt_col, args.question_col)
        messages.append({"role": "assistant", "content": response})
        selected.append({
            "messages": messages,
            "data_source": example.get("data_source", "math_dapo_reasoning"),
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "question": str(example.get(args.question_col, "")).strip(),
            "answer": ground_truth,
            "extra_info": {
                "index": idx,
                "primary_response_index": args.primary_response_index,
                "source_path": os.path.expanduser(args.input_path),
            },
        })

    required = args.num_samples + args.val_size
    if len(selected) < required:
        raise ValueError(f"Need {required} filtered SFT samples, but only found {len(selected)}")

    ds = datasets.Dataset.from_list(selected).shuffle(seed=args.seed)
    train_ds = ds.select(range(args.num_samples))
    val_ds = ds.select(range(args.num_samples, required)) if args.val_size > 0 else None

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.parquet")
    train_ds.to_parquet(train_path)

    val_path = None
    if val_ds is not None:
        val_path = os.path.join(output_dir, "val.parquet")
        val_ds.to_parquet(val_path)

    summary = {
        "input_path": os.path.expanduser(args.input_path),
        "num_filtered_candidates": len(selected),
        "train_samples": len(train_ds),
        "val_samples": 0 if val_ds is None else len(val_ds),
        "primary_response_index": args.primary_response_index,
        "max_response_chars": args.max_response_chars,
        "train_path": train_path,
        "val_path": val_path,
    }
    summary_path = os.path.join(output_dir, "meta.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved HERO cold-start SFT train parquet to {train_path}")
    if val_path is not None:
        print(f"Saved HERO cold-start SFT val parquet to {val_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()

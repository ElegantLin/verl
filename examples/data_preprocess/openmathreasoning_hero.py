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
"""Prepare OpenMathReasoning-style parquet data for HERO training.

This script follows the paper's split design:
- verifiable subset (verifier pass)
- hard-to-verify subset (verifier fail)
- mixed subset (balanced concat of both)

It expects a candidate solution column for verifier pass/fail filtering.
Following the paper, it first samples a fixed-size candidate pool from the
filtered source dataset before running verifier-based stratification.
"""

import argparse
import json
import os
from typing import Any

import datasets
from tqdm import tqdm

from verl.utils.hdfs_io import copy, makedirs


def _verify(solution: str, ground_truth: str) -> float:
    """Prefer math-verify when available, fall back to math_reward."""
    try:
        from verl.utils.reward_score.math_verify import compute_score as math_verify_score

        return float(math_verify_score(solution, ground_truth))
    except Exception:
        from verl.utils.reward_score.math_reward import compute_score as math_reward_score

        return float(math_reward_score(solution, ground_truth))


def _sample(ds: datasets.Dataset, size: int, seed: int) -> datasets.Dataset:
    if size <= 0:
        return ds.select([])
    if len(ds) < size:
        raise ValueError(f"Not enough samples to draw {size}. Available={len(ds)}")
    return ds.shuffle(seed=seed).select(range(size))


def _load_source_dataset(args: argparse.Namespace) -> datasets.Dataset:
    dataset_path = args.local_dataset_path if args.local_dataset_path is not None else args.dataset
    load_kwargs: dict[str, Any] = {"split": args.split}
    if args.dataset_config is not None:
        load_kwargs["name"] = args.dataset_config
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    return datasets.load_dataset(dataset_path, **load_kwargs)


def _export_hf_dataset(
    outputs: dict[str, datasets.Dataset],
    *,
    hf_save_dir: str | None,
    push_to_hub_repo: str | None,
    hub_private: bool,
) -> None:
    if hf_save_dir is None and push_to_hub_repo is None:
        return

    split_mapping = {
        "train_verifiable.parquet": "train_verifiable",
        "train_hard_to_verify.parquet": "train_hard_to_verify",
        "train_mixed.parquet": "train_mixed",
        "val_verifiable.parquet": "validation_verifiable",
        "val_hard_to_verify.parquet": "validation_hard_to_verify",
        "val_mixed.parquet": "validation_mixed",
    }
    dataset_dict = datasets.DatasetDict({split_mapping[name]: ds for name, ds in outputs.items()})

    if hf_save_dir is not None:
        dataset_dict.save_to_disk(os.path.expanduser(hf_save_dir))

    if push_to_hub_repo is not None:
        dataset_dict.push_to_hub(push_to_hub_repo, private=hub_private)


def _build_processed_example(example: dict[str, Any], idx: int, args: argparse.Namespace) -> dict[str, Any]:
    question = str(example[args.question_col]).strip()
    ground_truth = str(example[args.answer_col]).strip()
    candidate_solution = str(example.get(args.candidate_col, "")).strip()
    verifier_pass = int(_verify(candidate_solution, ground_truth) > 0.5)
    prompt = f"{question} {args.instruction}".strip()
    return {
        "data_source": "math_dapo_reasoning",
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "verifier_pass": verifier_pass,
        "extra_info": {
            "index": idx,
            "raw_question": question,
            "candidate_solution": candidate_solution,
            "split_source": args.split,
        },
    }


def _split_train_val(ds: datasets.Dataset, train_size: int, val_size: int, seed: int) -> tuple[datasets.Dataset, datasets.Dataset]:
    if len(ds) < train_size + val_size:
        raise ValueError(f"Need at least {train_size + val_size} rows, but only found {len(ds)}")
    ds = ds.shuffle(seed=seed)
    train = ds.select(range(train_size))
    val = ds.select(range(train_size, train_size + val_size))
    return train, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="OpenMathReasoning")
    parser.add_argument("--dataset_config", default=None, help="Optional HF dataset config/subset name.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--local_dataset_path", default=None, help="Optional local dataset path for load_dataset.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to datasets.load_dataset when the source dataset requires it.",
    )
    parser.add_argument("--question_col", default="question")
    parser.add_argument("--answer_col", default="answer")
    parser.add_argument("--candidate_col", default="candidate_solution")
    parser.add_argument("--problem_type_col", default="problem_type")
    parser.add_argument("--problem_type_value", default="has_answer_extracted")
    parser.add_argument("--verifiable_train_size", type=int, default=2000)
    parser.add_argument("--hard_train_size", type=int, default=2000)
    parser.add_argument("--verifiable_val_size", type=int, default=250)
    parser.add_argument("--hard_val_size", type=int, default=250)
    parser.add_argument(
        "--source_sample_size",
        type=int,
        default=40000,
        help=(
            "Sample this many rows from the source dataset after the problem_type filter and before verifier "
            "stratification. Set to -1 to disable subsampling."
        ),
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of worker processes for dataset filter/map stages.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dir", default=None, help="Deprecated alias for --local_save_dir.")
    parser.add_argument("--local_save_dir", default="~/data/openmathreasoning_hero")
    parser.add_argument(
        "--hf_save_dir",
        default=None,
        help="Optional output directory for saving the processed splits as a Hugging Face DatasetDict.",
    )
    parser.add_argument(
        "--push_to_hub_repo",
        default=None,
        help="Optional Hugging Face repo id to upload the processed DatasetDict via push_to_hub.",
    )
    parser.add_argument(
        "--hub_private",
        action="store_true",
        help="Create the pushed Hugging Face dataset repo as private when used with --push_to_hub_repo.",
    )
    parser.add_argument(
        "--instruction",
        default="Let's think step by step and output the final answer within \\boxed{}.",
    )
    args = parser.parse_args()

    local_save_dir = args.local_dir if args.local_dir is not None else args.local_save_dir
    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    dataset = _load_source_dataset(args)

    if args.problem_type_col in dataset.column_names:
        dataset = dataset.filter(
            lambda x: x[args.problem_type_col] == args.problem_type_value,
            num_proc=args.num_proc,
        )

    if args.source_sample_size > 0 and len(dataset) > args.source_sample_size:
        dataset = _sample(dataset, args.source_sample_size, args.seed + 100)

    if args.candidate_col not in dataset.column_names:
        raise ValueError(
            f"Missing candidate column `{args.candidate_col}`. "
            "Please provide model-generated solutions for verifier pass/fail splitting."
        )

    if args.question_col not in dataset.column_names or args.answer_col not in dataset.column_names:
        raise ValueError(
            f"Expected `{args.question_col}` and `{args.answer_col}` in dataset columns, got {dataset.column_names}"
        )

    verifiable_target = args.verifiable_train_size + args.verifiable_val_size
    hard_target = args.hard_train_size + args.hard_val_size
    selected_verifiable: list[dict[str, Any]] = []
    selected_hard: list[dict[str, Any]] = []

    for idx, example in enumerate(tqdm(dataset, desc="Verifier stratification", total=len(dataset))):
        processed = _build_processed_example(example, idx, args)
        if processed["verifier_pass"] == 1:
            if len(selected_verifiable) < verifiable_target:
                selected_verifiable.append(processed)
        else:
            if len(selected_hard) < hard_target:
                selected_hard.append(processed)

        if len(selected_verifiable) >= verifiable_target and len(selected_hard) >= hard_target:
            break

    if len(selected_verifiable) < verifiable_target:
        raise ValueError(
            f"Not enough verifier-passing samples after stratification. Need {verifiable_target}, "
            f"found {len(selected_verifiable)}"
        )
    if len(selected_hard) < hard_target:
        raise ValueError(
            f"Not enough hard-to-verify samples after stratification. Need {hard_target}, "
            f"found {len(selected_hard)}"
        )

    verifiable = datasets.Dataset.from_list(selected_verifiable)
    hard = datasets.Dataset.from_list(selected_hard)

    train_verifiable, val_verifiable = _split_train_val(
        verifiable,
        args.verifiable_train_size,
        args.verifiable_val_size,
        args.seed,
    )
    train_hard, val_hard = _split_train_val(
        hard,
        args.hard_train_size,
        args.hard_val_size,
        args.seed + 1,
    )

    train_mixed = datasets.concatenate_datasets([train_verifiable, train_hard]).shuffle(seed=args.seed + 10)
    val_mixed = datasets.concatenate_datasets([val_verifiable, val_hard]).shuffle(seed=args.seed + 11)

    outputs = {
        "train_verifiable.parquet": train_verifiable,
        "train_hard_to_verify.parquet": train_hard,
        "train_mixed.parquet": train_mixed,
        "val_verifiable.parquet": val_verifiable,
        "val_hard_to_verify.parquet": val_hard,
        "val_mixed.parquet": val_mixed,
    }

    for filename, ds in outputs.items():
        ds.to_parquet(os.path.join(local_save_dir, filename))

    _export_hf_dataset(
        outputs,
        hf_save_dir=args.hf_save_dir,
        push_to_hub_repo=args.push_to_hub_repo,
        hub_private=args.hub_private,
    )

    meta = {
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "source_sample_size": args.source_sample_size,
        "num_proc": args.num_proc,
        "question_col": args.question_col,
        "answer_col": args.answer_col,
        "candidate_col": args.candidate_col,
        "problem_type_filter": {args.problem_type_col: args.problem_type_value},
        "counts": {name: len(ds) for name, ds in outputs.items()},
        "hf_save_dir": args.hf_save_dir,
        "push_to_hub_repo": args.push_to_hub_repo,
    }
    with open(os.path.join(local_save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(local_save_dir, "train_mixed_example.json"), "w") as f:
        json.dump(train_mixed[0], f, indent=2)

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)


if __name__ == "__main__":
    main()

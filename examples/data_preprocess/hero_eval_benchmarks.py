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
"""Preprocess evaluation benchmarks for HERO paper (arXiv:2510.07242v1).

Verifiable benchmarks (rule-based, math_verify):
  - MATH500   (HuggingFaceH4/MATH-500, 500 problems)
  - AMC       (AI-MO/aimo-validation-amc, AMC 10/12 problems)
  - Minerva   (math subset of Minerva, from DigitalLearningGmbH/MATH-lighteval filtered)
  - Olympiad  (AI-MO/aimo-validation-olympiad or OlympiadBench)

Hard-to-verify benchmarks (LLM-as-judge, GPT-4o):
  - HardVerify-Math (HVM) — HardVerifyMath benchmark (250 problems)
  - TextBookReasoning (TBR) — TextBookReasoning benchmark (~750 problems)

Usage:
  python examples/data_preprocess/hero_eval_benchmarks.py       --benchmarks math500 amc minerva olympiad hardverify_math textbook_reasoning       --local_save_dir ~/data/hero_eval
"""

import argparse
import json
import os
from typing import Any

import datasets


INSTRUCTION = "Let's think step by step and output the final answer within \\boxed{}."


def _make_prompt(question: str, instruction: str = INSTRUCTION) -> list[dict]:
    return [{"role": "user", "content": f"{question.strip()} {instruction}".strip()}]


def _resolve_dataset_split(dataset_obj: datasets.Dataset | datasets.DatasetDict, preferred_splits: tuple[str, ...]) -> datasets.Dataset:
    if isinstance(dataset_obj, datasets.DatasetDict):
        for split in preferred_splits:
            if split in dataset_obj:
                return dataset_obj[split]
        if "train" in dataset_obj:
            return dataset_obj["train"]
        first_split = next(iter(dataset_obj.keys()))
        return dataset_obj[first_split]
    return dataset_obj


def _load_local_dataset(local_path: str, preferred_splits: tuple[str, ...]) -> datasets.Dataset:
    local_path = os.path.expanduser(local_path)
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local dataset path does not exist: {local_path}")

    if os.path.isdir(local_path):
        try:
            return _resolve_dataset_split(datasets.load_from_disk(local_path), preferred_splits)
        except Exception:
            return datasets.load_dataset(local_path, split=preferred_splits[0])

    suffix = os.path.splitext(local_path)[1].lower()
    if suffix == ".parquet":
        return datasets.load_dataset("parquet", data_files=local_path, split="train")
    if suffix in {".json", ".jsonl"}:
        return datasets.load_dataset("json", data_files=local_path, split="train")
    if suffix == ".csv":
        return datasets.load_dataset("csv", data_files=local_path, split="train")
    return datasets.load_dataset(local_path, split=preferred_splits[0])


def _load_dataset_with_fallback(
    repo_id: str,
    preferred_splits: tuple[str, ...],
    *,
    local_path: str | None = None,
    subset: str | None = None,
) -> datasets.Dataset | None:
    if local_path is not None:
        return _load_local_dataset(local_path, preferred_splits)

    last_error = None
    for split in preferred_splits:
        try:
            load_kwargs: dict[str, Any] = {"split": split}
            if subset is not None:
                load_kwargs["name"] = subset
            return datasets.load_dataset(repo_id, **load_kwargs)
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        print(f"  Failed to load {repo_id}: {last_error}")
    return None


def _normalize_reward_model(reward_model: Any) -> dict[str, Any]:
    if isinstance(reward_model, str):
        return json.loads(reward_model)
    return dict(reward_model)


def _build_records_from_preprocessed_dataset(ds: datasets.Dataset, data_source: str, style: str) -> list[dict[str, Any]]:
    records = []
    for idx, ex in enumerate(ds):
        reward_model = _normalize_reward_model(ex["reward_model"])
        prompt = ex["prompt"]
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        records.append({
            "data_source": data_source,
            "prompt": prompt,
            "ability": ex.get("ability", "math"),
            "reward_model": {"style": style, "ground_truth": reward_model.get("ground_truth", "")},
            "extra_info": {"index": idx},
        })
    return records


def preprocess_math500(args):
    """MATH-500: 500 competition math problems."""
    print("Processing MATH500...")
    ds = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")

    records = []
    for idx, ex in enumerate(ds):
        question = ex["problem"]
        solution = ex["answer"]
        records.append({
            "data_source": "math500",
            "prompt": _make_prompt(question),
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {"index": idx, "level": ex.get("level", ""), "type": ex.get("type", "")},
        })

    ds_out = datasets.Dataset.from_list(records)
    out_path = os.path.join(args.local_save_dir, "math500.parquet")
    ds_out.to_parquet(out_path)
    print(f"  MATH500: {len(records)} problems -> {out_path}")
    return ds_out


def preprocess_amc(args):
    """AMC: American Mathematics Competition problems."""
    print("Processing AMC...")
    try:
        ds = datasets.load_dataset("AI-MO/aimo-validation-amc", split="train")
        question_col, answer_col = "problem", "answer"
    except Exception:
        print("  AI-MO/aimo-validation-amc not found, trying AI-MO/NuminaMath-CoT with AMC filter...")
        ds = datasets.load_dataset("AI-MO/NuminaMath-CoT", split="train")
        ds = ds.filter(lambda x: "AMC" in x.get("source", ""))
        question_col, answer_col = "problem", "solution"

    records = []
    for idx, ex in enumerate(ds):
        question = ex[question_col]
        answer = str(ex[answer_col]).strip()
        if "\\boxed{" in answer:
            from verl.utils.reward_score.math_dapo import last_boxed_only_string, remove_boxed

            boxed = last_boxed_only_string(answer)
            if boxed:
                answer = remove_boxed(boxed)
        records.append({
            "data_source": "amc",
            "prompt": _make_prompt(question),
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"index": idx},
        })

    ds_out = datasets.Dataset.from_list(records)
    out_path = os.path.join(args.local_save_dir, "amc.parquet")
    ds_out.to_parquet(out_path)
    print(f"  AMC: {len(records)} problems -> {out_path}")
    return ds_out


def preprocess_minerva(args):
    """Minerva: math problems from Minerva benchmark (MATH-lighteval subset)."""
    print("Processing Minerva...")
    ds = datasets.load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test")

    from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed

    records = []
    for idx, ex in enumerate(ds):
        question = ex["problem"]
        solution_str = ex["solution"]
        boxed = last_boxed_only_string(solution_str)
        answer = remove_boxed(boxed) if boxed else solution_str.strip()
        records.append({
            "data_source": "minerva",
            "prompt": _make_prompt(question),
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"index": idx, "level": ex.get("level", ""), "type": ex.get("type", "")},
        })

    ds_out = datasets.Dataset.from_list(records)
    out_path = os.path.join(args.local_save_dir, "minerva.parquet")
    ds_out.to_parquet(out_path)
    print(f"  Minerva: {len(records)} problems -> {out_path}")
    return ds_out


def preprocess_olympiad(args):
    """Olympiad: olympiad-level math competition problems."""
    print("Processing Olympiad...")
    try:
        ds = datasets.load_dataset("AI-MO/aimo-validation-olympiad", split="train")
        question_col, answer_col = "problem", "answer"
    except Exception:
        print("  AI-MO/aimo-validation-olympiad not found, trying OlympiadBench...")
        try:
            ds = datasets.load_dataset("GAIR/OlympiadBench", split="test")
            question_col, answer_col = "question", "final_answer"
        except Exception:
            print("  OlympiadBench not found, trying NuminaMath-CoT with Olympiad filter...")
            ds = datasets.load_dataset("AI-MO/NuminaMath-CoT", split="train")
            ds = ds.filter(lambda x: "olympiad" in x.get("source", "").lower())
            question_col, answer_col = "problem", "solution"

    from verl.utils.reward_score.math_dapo import last_boxed_only_string, remove_boxed

    records = []
    for idx, ex in enumerate(ds):
        question = str(ex[question_col]).strip()
        answer = str(ex[answer_col]).strip()
        if "\\boxed{" in answer:
            boxed = last_boxed_only_string(answer)
            if boxed:
                answer = remove_boxed(boxed)
        records.append({
            "data_source": "olympiad",
            "prompt": _make_prompt(question),
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"index": idx},
        })

    ds_out = datasets.Dataset.from_list(records)
    out_path = os.path.join(args.local_save_dir, "olympiad.parquet")
    ds_out.to_parquet(out_path)
    print(f"  Olympiad: {len(records)} problems -> {out_path}")
    return ds_out


def preprocess_hardverify_math(args):
    """HardVerify-Math (HVM): 250 problems where symbolic verifiers struggle."""
    print("Processing HardVerify-Math...")
    ds = _load_dataset_with_fallback(
        "xu-zhangchen/HardVerifyMath",
        ("test", "train"),
        local_path=args.hvm_local_path,
    )
    if ds is None:
        print("  xu-zhangchen/HardVerifyMath not available. Please provide --hvm_local_path.")
        return None

    if {"prompt", "reward_model"}.issubset(set(ds.column_names)):
        records = _build_records_from_preprocessed_dataset(ds, data_source="hardverify_math", style="llm_judge")
    else:
        question_col = "question" if "question" in ds.column_names else "problem"
        answer_col = "answer" if "answer" in ds.column_names else "solution"

        records = []
        for idx, ex in enumerate(ds):
            question = str(ex[question_col]).strip()
            answer = str(ex[answer_col]).strip()
            records.append({
                "data_source": "hardverify_math",
                "prompt": _make_prompt(question),
                "ability": "math",
                "reward_model": {"style": "llm_judge", "ground_truth": answer},
                "extra_info": {"index": idx},
            })

    ds_out = datasets.Dataset.from_list(records)
    out_path = os.path.join(args.local_save_dir, "hardverify_math.parquet")
    ds_out.to_parquet(out_path)
    print(f"  HardVerify-Math: {len(records)} problems -> {out_path}")
    return ds_out


def preprocess_textbook_reasoning(args):
    """TextBookReasoning (TBR): hard-to-verify reasoning problems."""
    print("Processing TextBookReasoning...")
    ds = _load_dataset_with_fallback(
        "MegaScience/TextBookReasoning",
        ("test", "train"),
        local_path=args.tbr_local_path,
    )
    if ds is None:
        print("  MegaScience/TextBookReasoning not available. Please provide --tbr_local_path.")
        return None

    if {"prompt", "reward_model"}.issubset(set(ds.column_names)):
        records = _build_records_from_preprocessed_dataset(ds, data_source="textbook_reasoning", style="llm_judge")
    else:
        question_col = "question" if "question" in ds.column_names else "problem"
        answer_col = "answer" if "answer" in ds.column_names else "solution"

        records = []
        for idx, ex in enumerate(ds):
            question = str(ex[question_col]).strip()
            answer = str(ex[answer_col]).strip()
            records.append({
                "data_source": "textbook_reasoning",
                "prompt": _make_prompt(question),
                "ability": "math",
                "reward_model": {"style": "llm_judge", "ground_truth": answer},
                "extra_info": {"index": idx},
            })

    ds_out = datasets.Dataset.from_list(records)
    out_path = os.path.join(args.local_save_dir, "textbook_reasoning.parquet")
    ds_out.to_parquet(out_path)
    print(f"  TextBookReasoning: {len(records)} problems -> {out_path}")
    return ds_out


BENCHMARK_FNS = {
    "math500": preprocess_math500,
    "amc": preprocess_amc,
    "minerva": preprocess_minerva,
    "olympiad": preprocess_olympiad,
    "hardverify_math": preprocess_hardverify_math,
    "textbook_reasoning": preprocess_textbook_reasoning,
}


def main():
    parser = argparse.ArgumentParser(description="Preprocess HERO paper eval benchmarks")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(BENCHMARK_FNS.keys()),
        choices=list(BENCHMARK_FNS.keys()),
        help="Which benchmarks to preprocess.",
    )
    parser.add_argument("--local_save_dir", default="~/data/hero_eval")
    parser.add_argument("--hvm_local_path", default=None, help="Local path for HardVerifyMath dataset.")
    parser.add_argument(
        "--tbr_local_path",
        default=None,
        help="Local path for TextBookReasoning dataset or a pre-filtered parquet file.",
    )
    args = parser.parse_args()

    args.local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(args.local_save_dir, exist_ok=True)

    results = {}
    for bench in args.benchmarks:
        ds = BENCHMARK_FNS[bench](args)
        if ds is not None:
            results[bench] = len(ds)

    summary_path = os.path.join(args.local_save_dir, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary written to {summary_path}")
    print("Benchmark counts:", results)


if __name__ == "__main__":
    main()

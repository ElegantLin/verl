set -euo pipefail
set -x

# End-to-end HERO pipeline:
# 1. Load a source dataset from Hugging Face or a local load_dataset-compatible path
# 2. Build HERO parquet splits
# 3. Launch the existing Qwen3-4B HERO GRPO training script on train_mixed / val_mixed

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/../.." && pwd)

hero_local_save_dir=${HERO_LOCAL_SAVE_DIR:-$HOME/data/openmathreasoning_hero}
hero_train_path=${HERO_TRAIN_PATH:-${hero_local_save_dir}/train_mixed.parquet}
hero_val_path=${HERO_VAL_PATH:-${hero_local_save_dir}/val_mixed.parquet}

source_dataset=${HERO_DATASET:-OpenMathReasoning}
source_dataset_config=${HERO_DATASET_CONFIG:-}
source_dataset_split=${HERO_DATASET_SPLIT:-train}
source_local_dataset_path=${HERO_LOCAL_DATASET_PATH:-}
source_trust_remote_code=${HERO_TRUST_REMOTE_CODE:-0}

question_col=${HERO_QUESTION_COL:-question}
answer_col=${HERO_ANSWER_COL:-answer}
candidate_col=${HERO_CANDIDATE_COL:-candidate_solution}
problem_type_col=${HERO_PROBLEM_TYPE_COL:-problem_type}
problem_type_value=${HERO_PROBLEM_TYPE_VALUE:-has_answer_extracted}

verifiable_train_size=${HERO_VERIFIABLE_TRAIN_SIZE:-2000}
hard_train_size=${HERO_HARD_TRAIN_SIZE:-2000}
verifiable_val_size=${HERO_VERIFIABLE_VAL_SIZE:-250}
hard_val_size=${HERO_HARD_VAL_SIZE:-250}
seed=${HERO_PREPROCESS_SEED:-42}
instruction=${HERO_INSTRUCTION:-"Let's think step by step and output the final answer within \\boxed{}."}

hf_save_dir=${HERO_HF_SAVE_DIR:-}
push_to_hub_repo=${HERO_PUSH_TO_HUB_REPO:-}
hub_private=${HERO_HUB_PRIVATE:-0}
skip_preprocess=${HERO_SKIP_PREPROCESS:-0}

preprocess_cmd=(
    python
    examples/data_preprocess/openmathreasoning_hero.py
    --split "${source_dataset_split}"
    --question_col "${question_col}"
    --answer_col "${answer_col}"
    --candidate_col "${candidate_col}"
    --problem_type_col "${problem_type_col}"
    --problem_type_value "${problem_type_value}"
    --verifiable_train_size "${verifiable_train_size}"
    --hard_train_size "${hard_train_size}"
    --verifiable_val_size "${verifiable_val_size}"
    --hard_val_size "${hard_val_size}"
    --seed "${seed}"
    --instruction "${instruction}"
    --local_save_dir "${hero_local_save_dir}"
)

if [[ -n "${source_local_dataset_path}" ]]; then
    preprocess_cmd+=(--local_dataset_path "${source_local_dataset_path}")
else
    preprocess_cmd+=(--dataset "${source_dataset}")
fi

if [[ -n "${source_dataset_config}" ]]; then
    preprocess_cmd+=(--dataset_config "${source_dataset_config}")
fi

if [[ "${source_trust_remote_code}" == "1" ]]; then
    preprocess_cmd+=(--trust_remote_code)
fi

if [[ -n "${hf_save_dir}" ]]; then
    preprocess_cmd+=(--hf_save_dir "${hf_save_dir}")
fi

if [[ -n "${push_to_hub_repo}" ]]; then
    preprocess_cmd+=(--push_to_hub_repo "${push_to_hub_repo}")
fi

if [[ "${hub_private}" == "1" ]]; then
    preprocess_cmd+=(--hub_private)
fi

cd "${repo_root}"

if [[ "${skip_preprocess}" != "1" ]]; then
    "${preprocess_cmd[@]}"
fi

export HERO_TRAIN_PATH="${hero_train_path}"
export HERO_VAL_PATH="${hero_val_path}"

bash "${script_dir}/run_qwen3_4b_hero.sh" "$@"

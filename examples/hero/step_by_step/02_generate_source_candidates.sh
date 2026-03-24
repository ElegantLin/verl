#!/usr/bin/env bash
set -euo pipefail
set -x

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=examples/hero/step_by_step/common.sh
source "$script_dir/common.sh"

cd "$repo_root"
ensure_hero_dirs
require_file "$source_prompts_path"

python3 -m verl.trainer.main_generation_server \
    trainer.nnodes="$gen_nnodes" \
    trainer.n_gpus_per_node="$gen_gpus_per_node" \
    data.train_files="['$source_prompts_path']" \
    data.prompt_key=prompt \
    data.output_path="$source_generated_path" \
    actor_rollout_ref.model.path="$base_model_path" \
    actor_rollout_ref.model.trust_remote_code="$trust_remote_code" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n="$source_generation_n" \
    actor_rollout_ref.rollout.temperature="$source_temperature" \
    actor_rollout_ref.rollout.top_p="$source_top_p" \
    actor_rollout_ref.rollout.response_length="$max_response_length" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$gen_tp_size" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$gen_gpu_memory_utilization" \
    "$@"

echo "Generated source parquet saved to: $source_generated_path"

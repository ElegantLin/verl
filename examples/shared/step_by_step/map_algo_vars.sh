#!/usr/bin/env bash
# Shared env-var mapping: translates <PREFIX>_* → RL_PIPELINE_* for any algorithm.
#
# Usage (in algorithm-specific common.sh):
#   _ALGO_PREFIX=HERO
#   source path/to/map_algo_vars.sh
#
# Requires _ALGO_PREFIX to be set before sourcing.

if [[ -z "${_ALGO_PREFIX:-}" ]]; then
    echo "ERROR: _ALGO_PREFIX must be set before sourcing map_algo_vars.sh" >&2
    return 1
fi

_algo_map_var() {
    local src="${_ALGO_PREFIX}_$1"
    local dst="RL_PIPELINE_$1"
    if [[ -n "${!src:-}" ]]; then
        export "$dst=${!src}"
    fi
}

# GPU profile (special: has a default value).
_gp="${_ALGO_PREFIX}_GPU_PROFILE"
export RL_PIPELINE_GPU_PROFILE="${!_gp:-8x24gb}"

# All RL_PIPELINE_* suffixes recognized by the shared pipeline.
# Covers: cluster layout, training, rollout, reward model, generation,
# evaluation, SFT, directories, model, dataset, and eval benchmarks.
for _s in \
    GPUS_PER_NODE NNODES MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH \
    TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE PPO_MICRO_BATCH_SIZE_PER_GPU \
    PPO_MAX_TOKEN_LEN_PER_GPU \
    ROLLOUT_N ROLLOUT_TP_SIZE ROLLOUT_GPU_MEMORY_UTILIZATION \
    ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    ROLLOUT_MAX_NUM_SEQS \
    RM_GPUS_PER_NODE RM_NNODES RM_TP_SIZE RM_GPU_MEMORY_UTILIZATION RM_MAX_NUM_SEQS \
    GEN_GPUS_PER_NODE GEN_NNODES GEN_TP_SIZE GEN_GPU_MEMORY_UTILIZATION \
    SOURCE_GENERATION_N SOURCE_TEMPERATURE SOURCE_TOP_P \
    EVAL_GPUS_PER_NODE EVAL_NNODES EVAL_TP_SIZE EVAL_GPU_MEMORY_UTILIZATION \
    EVAL_N_SAMPLES \
    SFT_GPUS_PER_NODE SFT_NNODES \
    SFT_TRAIN_BATCH_SIZE SFT_MICRO_BATCH_SIZE_PER_GPU SFT_MAX_LENGTH \
    SFT_MAX_TOKEN_LEN_PER_GPU \
    ARTIFACT_ROOT RUN_NAME WORK_DIR SOURCE_DIR DATA_DIR EVAL_DIR \
    SFT_DATA_DIR SFT_OUTPUT_DIR TRAIN_OUTPUT_DIR EVAL_OUTPUT_DIR \
    SOURCE_PROMPTS_PATH SOURCE_GENERATED_PATH FILTERED_TBR_PATH \
    BASE_MODEL_PATH MODEL_PATH TRUST_REMOTE_CODE SOURCE_DATASET_TRUST_REMOTE_CODE \
    DATASET DATASET_CONFIG DATASET_SPLIT LOCAL_DATASET_PATH \
    SOURCE_QUESTION_COL SOURCE_ANSWER_COL QUESTION_COL ANSWER_COL \
    PROBLEM_TYPE_COL PROBLEM_TYPE_VALUE SOURCE_SAMPLE_SIZE SEED \
    EVAL_BENCHMARKS_BUILD HVM_LOCAL_PATH TBR_LOCAL_PATH \
; do
    _algo_map_var "$_s"
done

unset _s _gp

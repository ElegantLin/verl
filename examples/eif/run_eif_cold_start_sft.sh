#!/usr/bin/env bash
# EIF cold-start SFT wrapper.
#
# Same cold-start procedure as HERO:
#   1. Build a 2k-example SFT set from model-generated OpenMathReasoning responses
#   2. Run 2 epochs of SFT
#   3. Export the latest checkpoint in Hugging Face format

set -euo pipefail
set -x

input_path=${EIF_SFT_INPUT_PATH:-$HOME/data/hero_source/source_generated.parquet}
sft_data_dir=${EIF_SFT_DATA_DIR:-$HOME/data/openmathreasoning_hero_sft}
output_dir=${EIF_SFT_OUTPUT_DIR:-checkpoints/eif/eif_cold_start_sft}
model_path=${EIF_MODEL_PATH:-Qwen/Qwen3-4B-Base}
trust_remote_code=${EIF_TRUST_REMOTE_CODE:-True}

question_col=${EIF_QUESTION_COL:-question}
answer_col=${EIF_ANSWER_COL:-answer}
prompt_col=${EIF_PROMPT_COL:-prompt}
reward_model_col=${EIF_REWARD_MODEL_COL:-reward_model}
response_col=${EIF_RESPONSE_COL:-responses}
primary_response_index=${EIF_PRIMARY_RESPONSE_INDEX:-0}
num_samples=${EIF_SFT_NUM_SAMPLES:-2000}
val_size=${EIF_SFT_VAL_SIZE:-0}
seed=${EIF_SEED:-42}
max_response_chars=${EIF_SFT_MAX_RESPONSE_CHARS:-12000}
force_preprocess=${EIF_FORCE_SFT_PREPROCESS:-0}

nnodes=${EIF_SFT_NNODES:-1}
nproc_per_node=${EIF_SFT_GPUS_PER_NODE:-8}
train_batch_size=${EIF_SFT_TRAIN_BATCH_SIZE:-64}
micro_batch_size=${EIF_SFT_MICRO_BATCH_SIZE_PER_GPU:-1}
max_length=${EIF_SFT_MAX_LENGTH:-9216}
max_token_len_per_gpu=${EIF_SFT_MAX_TOKEN_LEN_PER_GPU:-12288}
truncation=${EIF_SFT_TRUNCATION:-error}
learning_rate=${EIF_SFT_LR:-1e-5}
total_epochs=${EIF_SFT_TOTAL_EPOCHS:-2}
save_freq=${EIF_SFT_SAVE_FREQ:-after_each_epoch}
test_freq=${EIF_SFT_TEST_FREQ:--1}
loggers=${EIF_SFT_LOGGERS:-'["console"]'}
experiment_name=${EIF_SFT_EXPERIMENT_NAME:-eif_cold_start_sft_$(date +%Y%m%d_%H%M%S)}
checkpoint_save_contents=${EIF_SFT_CHECKPOINT_SAVE_CONTENTS:-"['model','optimizer','extra','hf_model']"}
max_ckpt_to_keep=${EIF_SFT_MAX_CKPT_TO_KEEP:-1}
resume_mode=${EIF_SFT_RESUME_MODE:-disable}

train_path="$sft_data_dir/train.parquet"
val_path="$sft_data_dir/val.parquet"

if [[ ! -f "$input_path" ]]; then
    echo "Missing generated OMR parquet for cold-start SFT: $input_path" >&2
    exit 1
fi

need_preprocess=0
if [[ "$force_preprocess" == "1" || ! -f "$train_path" ]]; then
    need_preprocess=1
fi
if [[ "$val_size" -gt 0 && ! -f "$val_path" ]]; then
    need_preprocess=1
fi

if [[ "$need_preprocess" == "1" ]]; then
    mkdir -p "$sft_data_dir"
    python3 examples/data_preprocess/openmathreasoning_hero_sft.py \
        --input_path "$input_path" \
        --question_col "$question_col" \
        --answer_col "$answer_col" \
        --prompt_col "$prompt_col" \
        --reward_model_col "$reward_model_col" \
        --response_col "$response_col" \
        --primary_response_index "$primary_response_index" \
        --num_samples "$num_samples" \
        --val_size "$val_size" \
        --seed "$seed" \
        --max_response_chars "$max_response_chars" \
        --output_dir "$sft_data_dir"
fi

mkdir -p "$output_dir"

torchrun_cmd=(torchrun --nnodes="$nnodes" --nproc_per_node="$nproc_per_node")
if [[ "$nnodes" == "1" ]]; then
    torchrun_cmd=(torchrun --standalone --nnodes=1 --nproc_per_node="$nproc_per_node")
else
    torchrun_cmd+=(--node_rank="${EIF_SFT_NODE_RANK:-0}")
    torchrun_cmd+=(--master_addr="${EIF_SFT_MASTER_ADDR:?Set EIF_SFT_MASTER_ADDR for multi-node cold-start SFT}")
    torchrun_cmd+=(--master_port="${EIF_SFT_MASTER_PORT:-29500}")
fi

sft_args=(
    -m verl.trainer.sft_trainer
    data.train_files="$train_path"
    data.messages_key=messages
    data.train_batch_size="$train_batch_size"
    data.micro_batch_size_per_gpu="$micro_batch_size"
    data.max_length="$max_length"
    data.truncation="$truncation"
    data.use_dynamic_bsz=True
    data.max_token_len_per_gpu="$max_token_len_per_gpu"
    model.path="$model_path"
    model.trust_remote_code="$trust_remote_code"
    model.use_remove_padding=True
    model.enable_gradient_checkpointing=True
    optim.lr="$learning_rate"
    checkpoint.save_contents="$checkpoint_save_contents"
    trainer.default_local_dir="$output_dir"
    trainer.project_name=eif_cold_start
    trainer.experiment_name="$experiment_name"
    trainer.total_epochs="$total_epochs"
    trainer.save_freq="$save_freq"
    trainer.test_freq="$test_freq"
    trainer.logger="$loggers"
    trainer.nnodes="$nnodes"
    trainer.n_gpus_per_node="$nproc_per_node"
    trainer.resume_mode="$resume_mode"
    trainer.max_ckpt_to_keep="$max_ckpt_to_keep"
)

if [[ -f "$val_path" ]]; then
    sft_args+=(data.val_files="$val_path")
fi

"${torchrun_cmd[@]}" "${sft_args[@]}" "$@"

tracker_file="$output_dir/latest_checkpointed_iteration.txt"
if [[ -f "$tracker_file" ]]; then
    latest_step=$(<"$tracker_file")
    hf_model_dir="$output_dir/global_step_${latest_step}/huggingface"
    if [[ -d "$hf_model_dir" ]]; then
        echo "Cold-start HF model path: $hf_model_dir"
    fi
fi

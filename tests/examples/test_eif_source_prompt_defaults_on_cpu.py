from pathlib import Path


HYBRID_REWARD_DIR = Path(__file__).resolve().parents[2] / "examples" / "hybrid_reward"
STEP_DIR = HYBRID_REWARD_DIR / "step_by_step"


def test_shared_data_train_contract_uses_openmathreasoning_defaults():
    common_text = (STEP_DIR / "common.sh").read_text()
    stage_text = (STEP_DIR / "data-train.sh").read_text()

    assert "dataset_name=${RL_PIPELINE_DATASET:-nvidia/OpenMathReasoning}" in common_text
    assert "dataset_split=${RL_PIPELINE_DATASET_SPLIT:-cot}" in common_text
    assert "source_question_col=${RL_PIPELINE_SOURCE_QUESTION_COL:-problem}" in common_text
    assert "source_answer_col=${RL_PIPELINE_SOURCE_ANSWER_COL:-expected_answer}" in common_text
    assert "source_dataset_trust_remote_code=${RL_PIPELINE_SOURCE_DATASET_TRUST_REMOTE_CODE:-False}" in common_text
    assert 'source "$script_dir/common.sh"' in stage_text
    assert 'source "$script_dir/../stage_lib.sh"' in stage_text
    assert 'run_stage_data_train "$@"' in stage_text

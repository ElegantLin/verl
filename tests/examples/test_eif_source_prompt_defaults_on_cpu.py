from pathlib import Path


EIF_DIR = Path(__file__).resolve().parents[2] / "examples" / "eif"
EIF_STEP_DIR = EIF_DIR / "step_by_step"


def test_eif_stage1_uses_raw_openmathreasoning_defaults_without_breaking_stage3():
    common_text = (EIF_STEP_DIR / "common.sh").read_text()
    stage1_text = (EIF_STEP_DIR / "01_build_source_prompts.sh").read_text()
    stage3_text = (EIF_STEP_DIR / "03_build_rl_data.sh").read_text()
    pipeline_text = (EIF_DIR / "run_eif_pipeline.sh").read_text()

    assert "dataset_name=${EIF_DATASET:-nvidia/OpenMathReasoning}" in common_text
    assert "dataset_split=${EIF_DATASET_SPLIT:-cot}" in common_text
    assert "source_question_col=${EIF_SOURCE_QUESTION_COL:-problem}" in common_text
    assert "source_answer_col=${EIF_SOURCE_ANSWER_COL:-expected_answer}" in common_text
    assert "source_dataset_trust_remote_code=${EIF_SOURCE_DATASET_TRUST_REMOTE_CODE:-False}" in common_text
    assert '--question_col "$source_question_col"' in stage1_text
    assert '--answer_col "$source_answer_col"' in stage1_text
    assert '[[ "$source_dataset_trust_remote_code" == "True" ]]' in stage1_text
    assert '--question_col "$question_col"' in stage3_text
    assert '--answer_col "$answer_col"' in stage3_text
    assert 'dataset_name=${EIF_DATASET:-nvidia/OpenMathReasoning}' in pipeline_text
    assert 'dataset_split=${EIF_DATASET_SPLIT:-cot}' in pipeline_text
    assert 'source_question_col=${EIF_SOURCE_QUESTION_COL:-problem}' in pipeline_text
    assert 'source_answer_col=${EIF_SOURCE_ANSWER_COL:-expected_answer}' in pipeline_text
    assert 'source_dataset_trust_remote_code=${EIF_SOURCE_DATASET_TRUST_REMOTE_CODE:-False}' in pipeline_text

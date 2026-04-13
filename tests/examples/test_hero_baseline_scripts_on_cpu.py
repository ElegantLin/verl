from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra


EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "hero"


def test_hero_baseline_scripts_exist_and_wire_expected_reward_paths():
    expected_scripts = {
        "run_hero_rm_only.sh": [
            "reward.reward_manager.name=naive",
            "reward.reward_model.enable=True",
            "nvidia/AceMath-7B-RM",
        ],
        "run_hero_verifier_only.sh": [
            "reward.reward_manager.name=naive",
            "reward.reward_model.enable=False",
            "reward.custom_reward_function.path=examples/hybrid_reward/baseline_reward_fn.py",
            "compute_score_math_verify",
        ],
        "run_hero_naive_combine.sh": [
            "reward.reward_manager.name=naive_combine",
            "reward.reward_model.enable=True",
            "reward.custom_reward_function.path=examples/hybrid_reward/baseline_reward_fn.py",
            "reward.reward_kwargs.naive_combine.alpha",
        ],
    }

    for script_name, snippets in expected_scripts.items():
        path = EXAMPLES_DIR / script_name
        assert path.exists(), f"Missing HERO baseline launcher: {path}"

        text = path.read_text()
        for snippet in snippets:
            assert snippet in text, f"{script_name} is missing expected snippet: {snippet}"


STEP_BY_STEP_DIR = EXAMPLES_DIR / "step_by_step"


def test_hero_step1_source_prompt_script_uses_raw_openmathreasoning_defaults():
    common_text = (STEP_BY_STEP_DIR / "common.sh").read_text()
    stage1_text = (STEP_BY_STEP_DIR / "01_build_source_prompts.sh").read_text()
    stage3_text = (STEP_BY_STEP_DIR / "03_build_rl_data.sh").read_text()

    assert "dataset_name=${HERO_DATASET:-nvidia/OpenMathReasoning}" in common_text
    assert "dataset_split=${HERO_DATASET_SPLIT:-cot}" in common_text
    assert "source_question_col=${HERO_SOURCE_QUESTION_COL:-problem}" in common_text
    assert "source_answer_col=${HERO_SOURCE_ANSWER_COL:-expected_answer}" in common_text
    assert "source_dataset_trust_remote_code=${HERO_SOURCE_DATASET_TRUST_REMOTE_CODE:-False}" in common_text
    assert '--question_col "$source_question_col"' in stage1_text
    assert '--answer_col "$source_answer_col"' in stage1_text
    assert '[[ "$source_dataset_trust_remote_code" == "True" ]]' in stage1_text
    assert '--question_col "$question_col"' in stage3_text
    assert '--answer_col "$answer_col"' in stage3_text



def test_generation_server_config_accepts_output_path_override_for_step2_scripts():
    config_dir = Path(__file__).resolve().parents[2] / "verl" / "trainer" / "config"
    overrides = [
        "data.train_files=['/tmp/source_prompts.parquet']",
        "data.prompt_key=prompt",
        "data.output_path=/tmp/source_generated.parquet",
    ]

    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            config = compose(config_name="ppo_trainer", overrides=overrides)
    finally:
        GlobalHydra.instance().clear()

    assert config.data.output_path == "/tmp/source_generated.parquet"


def test_cold_start_sft_script_falls_back_when_flash_attn_is_unavailable():
    text = (EXAMPLES_DIR / "run_hero_cold_start_sft.sh").read_text()

    assert "HERO_SFT_ATTN_IMPLEMENTATION" in text
    assert 'python3 -c "import flash_attn"' in text
    assert 'falling back to "sdpa"' in text
    assert '+model.override_config.attn_implementation="$attn_implementation"' in text

from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "examples" / "hybrid_eif" / "run_hybrid_eif_eval.sh"


def test_hybrid_eif_eval_script_exists_and_matches_hero_style_flow():
    assert SCRIPT_PATH.exists()

    text = SCRIPT_PATH.read_text()
    assert "main_generation_server" in text
    assert "examples/data_preprocess/hybrid_reward_eif_eval.py" in text
    assert "python3 -m verl.trainer.main_eval" in text
    assert "one_step_eif.enable=true" in text
    assert "tau_llm_value" in text
    assert "m_llm_value" in text

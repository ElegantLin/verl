from pathlib import Path
import re


HYBRID_REWARD_DIR = Path(__file__).resolve().parents[2] / "examples" / "hybrid_reward"
CLI_PATH = HYBRID_REWARD_DIR / "run.sh"
STEP_EVAL_PATH = HYBRID_REWARD_DIR / "step_by_step" / "eval.sh"
BACKEND_EVAL_PATH = HYBRID_REWARD_DIR / "run_eval.sh"


def read_text(path: Path) -> str:
    assert path.exists(), f"Missing script under test: {path}"
    return path.read_text()


def test_hybrid_reward_eval_entrypoints_route_to_shared_eval_backend():
    backend_text = read_text(BACKEND_EVAL_PATH)
    cli_text = read_text(CLI_PATH)
    step_text = read_text(STEP_EVAL_PATH)

    assert "main_generation_server" in backend_text
    assert "python3 -m verl.trainer.main_eval" in backend_text
    assert "eval_llm_judge.py" in backend_text
    assert re.search(r"^cmd_eval\(\)\s*\{", cli_text, re.M)
    assert "run_stage_eval" in cli_text
    assert "run_stage_eval" in step_text

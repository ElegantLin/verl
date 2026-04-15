from pathlib import Path
import re


HYBRID_REWARD_DIR = Path(__file__).resolve().parents[2] / "examples" / "hybrid_reward"
CLI_PATH = HYBRID_REWARD_DIR / "run.sh"
STAGE_LIB_PATH = HYBRID_REWARD_DIR / "stage_lib.sh"


def read_text(path: Path) -> str:
    assert path.exists(), f"Missing file under test: {path}"
    return path.read_text()


def test_cli_declares_semantic_command_handlers_and_uses_stage_library():
    text = read_text(CLI_PATH)

    assert re.search(r"^cmd_data_train\(\)\s*\{", text, re.M)
    assert re.search(r"^cmd_data_eval\(\)\s*\{", text, re.M)
    assert re.search(r"^cmd_sft\(\)\s*\{", text, re.M)
    assert re.search(r"^cmd_rl\(\)\s*\{", text, re.M)
    assert re.search(r"^cmd_eval\(\)\s*\{", text, re.M)
    assert re.search(r"^cmd_pipeline\(\)\s*\{", text, re.M)
    assert "stage_lib.sh" in text


def test_stage_library_declares_semantic_stage_functions():
    text = read_text(STAGE_LIB_PATH)

    assert re.search(r"^run_stage_data_train\(\)\s*\{", text, re.M)
    assert re.search(r"^run_stage_data_eval\(\)\s*\{", text, re.M)
    assert re.search(r"^run_stage_sft\(\)\s*\{", text, re.M)
    assert re.search(r"^run_stage_rl\(\)\s*\{", text, re.M)
    assert re.search(r"^run_stage_eval\(\)\s*\{", text, re.M)

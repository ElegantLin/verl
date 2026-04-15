from pathlib import Path
import subprocess


CLI_PATH = Path(__file__).resolve().parents[2] / "examples" / "hybrid_reward" / "run.sh"


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(CLI_PATH), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_help_mentions_semantic_commands_and_key_flags():
    result = run_cli("--help")
    output = result.stdout + result.stderr

    assert result.returncode == 0
    assert "data-train" in output
    assert "data-eval" in output
    assert "sft" in output
    assert "rl" in output
    assert "eval" in output
    assert "pipeline" in output
    assert "--algorithm" in output
    assert "--debug" in output
    assert "--model-path" in output


def test_cli_rejects_eval_without_algorithm_or_model_path():
    result = run_cli("eval")
    output = result.stdout + result.stderr

    assert result.returncode != 0
    assert "--algorithm" in output
    assert "--model-path" in output

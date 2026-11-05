import json
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "he_rhf.json"


@pytest.mark.parametrize(("cycles", "exit_code"), [(80, 0), (1, 1)])
def test_cli_outputs(tmp_path, cycles, exit_code):
    prefix = tmp_path / "output"
    result = subprocess.run([sys.executable, "-m", "opendft.cli", str(EXAMPLE),
                             "-o", str(prefix), "--max-cycles", str(cycles)],
                            capture_output=True, text=True)
    assert result.returncode == exit_code, result.stderr
    output = json.loads(prefix.with_suffix(".json").read_text())
    assert output["converged"] == (exit_code == 0)
    assert prefix.with_suffix(".log").read_text() == result.stdout
    assert "Nuclear repulsion" in result.stdout


def test_cli_preserves_input_on_output_collision(tmp_path):
    source = tmp_path / "same.json"
    source.write_bytes(EXAMPLE.read_bytes())
    result = subprocess.run([sys.executable, "-m", "opendft.cli", str(source),
                             "-o", str(source.with_suffix(""))], capture_output=True)
    assert result.returncode == 2
    assert source.read_bytes() == EXAMPLE.read_bytes()

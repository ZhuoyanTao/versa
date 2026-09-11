"""Check function coverage semantics independently of model dependencies."""

import subprocess
import sys
from pathlib import Path

from ci.check_function_docstrings import function_inventory


CHECKER = Path(__file__).resolve().parents[1] / "ci/check_function_docstrings.py"


def test_inventory_includes_nested_async_private_and_init(tmp_path):
    """Class docs must not cover methods; nested and async functions count separately."""
    source = tmp_path / "sample.py"
    source.write_text(
        '"""Module docs."""\n'
        "class Example:\n"
        '    """Class docs."""\n'
        "    def __init__(self):\n"
        "        pass\n"
        "    async def _run(self):\n"
        '        """Run the operation."""\n'
        "        def nested():\n"
        '            """"""\n'
        "            return 1\n"
        "        return nested()\n"
    )
    assert list(function_inventory(source)) == [
        (4, "Example.__init__", False),
        (6, "Example._run", True),
        (8, "Example._run.nested", False),
    ]


def test_cli_enforces_exact_threshold(tmp_path):
    """Three documented functions out of four fail at 80% and pass at 75%."""
    source = tmp_path / "sample.py"
    source.write_text(
        "\n".join(
            f'def f{i}():\n    """Return {i}."""\n    return {i}' for i in range(3)
        )
        + "\ndef missing():\n    pass\n"
    )
    for threshold, expected in [(80, 1), (75, 0)]:
        result = subprocess.run(
            [sys.executable, str(CHECKER), str(source), "--fail-under", str(threshold)],
            text=True,
            capture_output=True,
        )
        assert result.returncode == expected
        assert "3/4 (75.00%)" in result.stdout
        assert "missing missing" in result.stdout


def test_cli_rejects_empty_inventory(tmp_path):
    """An empty directory cannot produce a misleading passing coverage result."""
    result = subprocess.run(
        [sys.executable, str(CHECKER), str(tmp_path)], text=True, capture_output=True
    )
    assert result.returncode == 2
    assert "no Python functions found" in result.stderr

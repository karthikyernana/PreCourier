"""Developer setup creates a test merchant without enabling a simulated provider."""

import os
import subprocess
import sys


def test_local_provision_requires_a_pepper(tmp_path):
    env = dict(os.environ)
    env.pop("PRECOURIER_PEPPER", None)
    env["PRECOURIER_DB"] = str(tmp_path / "local.db")
    result = subprocess.run(
        [sys.executable, "tools/provision_local.py"],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "PRECOURIER_PEPPER is required" in result.stderr

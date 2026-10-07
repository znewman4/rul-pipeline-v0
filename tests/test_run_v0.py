"""Smoke test: the end-to-end V0 script runs and writes all five plots."""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_v0.py"


def test_run_v0_end_to_end(tmp_path):
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--n-elements", "8", "--snr-db", "0", "--n-samples", "2000",
         "--outdir", str(tmp_path)],
        capture_output=True, text=True, check=True,
    )
    for step in ("[1]", "[4-5]", "[9]", "[10]", "[11]"):
        assert step in out.stdout
    assert len(list(tmp_path.glob("*.png"))) == 5

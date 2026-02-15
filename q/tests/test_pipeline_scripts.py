import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _make_csv(path: Path, n: int = 360):
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0002, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2022-01-03", periods=n, freq="B"),
            "Close": close,
        }
    )
    df.to_csv(path, index=False)


def test_run_pipeline_smoke(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir(parents=True, exist_ok=True)
    _make_csv(data_dir / "A.csv", n=320)

    cmd = [
        sys.executable,
        str(root / "run_pipeline.py"),
        "--data",
        str(data_dir),
        "--asset",
        "A.csv",
        "--out",
        str(out_dir),
        "--dream_frames",
        "12",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    cp = subprocess.run(cmd, cwd=str(tmp_path), env=env, capture_output=True, text=True)
    assert cp.returncode == 0, cp.stderr

    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert "sharpe" in summary
    assert (out_dir / "dream.png").exists()
    assert (out_dir / "dream.gif").exists()


def test_run_pipeline_cv_plus_writes_non_placeholder_alarms(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out_cv_plus"
    data_dir.mkdir(parents=True, exist_ok=True)
    _make_csv(data_dir / "B.csv", n=360)

    cmd = [
        sys.executable,
        str(root / "run_pipeline_cv_plus.py"),
        "--data",
        str(data_dir),
        "--asset",
        "B.csv",
        "--out",
        str(out_dir),
        "--frames",
        "10",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    cp = subprocess.run(cmd, cwd=str(tmp_path), env=env, capture_output=True, text=True)
    assert cp.returncode == 0, cp.stderr

    alarms = json.loads((out_dir / "alarms.json").read_text(encoding="utf-8"))
    assert isinstance(alarms, list)

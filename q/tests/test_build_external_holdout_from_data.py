from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools import build_external_holdout_from_data as beh


def _write_price(path: Path, n: int = 520) -> None:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.linspace(100.0, 150.0, n)
    df = pd.DataFrame({"DATE": dates.strftime("%Y-%m-%d"), "Close": close})
    path.write_text(df.to_csv(index=False), encoding="utf-8")


def test_build_external_holdout_success(tmp_path: Path, monkeypatch):
    root = tmp_path
    runs = root / "runs_plus"
    data = root / "data"
    holdout = root / "data_holdout"
    runs.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    (runs / "asset_names.csv").write_text("asset\nSPY\nQQQ\n", encoding="utf-8")
    _write_price(data / "SPY.csv", n=520)
    _write_price(data / "QQQ.csv", n=520)

    monkeypatch.setattr(beh, "ROOT", root)
    monkeypatch.setattr(beh, "RUNS", runs)
    monkeypatch.setenv("Q_EXTERNAL_HOLDOUT_SOURCE_DIR", str(data))
    monkeypatch.setenv("Q_EXTERNAL_HOLDOUT_DIR", str(holdout))
    monkeypatch.setenv("Q_EXTERNAL_HOLDOUT_BUILD_ROWS", "200")
    monkeypatch.setenv("Q_EXTERNAL_HOLDOUT_BUILD_MIN_SYMBOLS", "2")
    monkeypatch.setenv("Q_EXTERNAL_HOLDOUT_REQUIRED", "1")

    rc = beh.main()
    assert rc == 0
    assert (holdout / "SPY.csv").exists()
    assert (holdout / "QQQ.csv").exists()

    info = json.loads((runs / "external_holdout_build_info.json").read_text(encoding="utf-8"))
    assert info["ok"] is True
    assert info["built_count"] == 2


def test_build_external_holdout_required_failure(tmp_path: Path, monkeypatch):
    root = tmp_path
    runs = root / "runs_plus"
    data = root / "data"
    holdout = root / "data_holdout"
    runs.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    (runs / "asset_names.csv").write_text("asset\nSPY\nQQQ\n", encoding="utf-8")
    _write_price(data / "SPY.csv", n=520)

    monkeypatch.setattr(beh, "ROOT", root)
    monkeypatch.setattr(beh, "RUNS", runs)
    monkeypatch.setenv("Q_EXTERNAL_HOLDOUT_SOURCE_DIR", str(data))
    monkeypatch.setenv("Q_EXTERNAL_HOLDOUT_DIR", str(holdout))
    monkeypatch.setenv("Q_EXTERNAL_HOLDOUT_BUILD_ROWS", "252")
    monkeypatch.setenv("Q_EXTERNAL_HOLDOUT_BUILD_MIN_SYMBOLS", "2")
    monkeypatch.setenv("Q_EXTERNAL_HOLDOUT_REQUIRED", "1")

    rc = beh.main()
    assert rc == 2
    info = json.loads((runs / "external_holdout_build_info.json").read_text(encoding="utf-8"))
    assert info["ok"] is False
    assert info["built_count"] == 1

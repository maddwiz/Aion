import json
from pathlib import Path

import numpy as np
import pandas as pd

import tools.run_credit_leadlag_signal as cls


def _write_series_csv(path: Path, values: np.ndarray) -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2021-01-01", periods=len(values), freq="B"),
            "Close": values.astype(float),
        }
    )
    df.to_csv(path, index=False)


def test_run_credit_leadlag_writes_outputs_and_detects_bearish_divergence(tmp_path: Path, monkeypatch):
    root = tmp_path
    runs = root / "runs_plus"
    data = root / "data"
    runs.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    t = 280
    x = np.linspace(0.0, 20.0, t)
    hyg = 90.0 + np.cumsum(0.08 * np.sin(x))
    lqd = 110.0 + np.cumsum(0.03 * np.sin(0.9 * x))
    spy = 300.0 + np.cumsum(0.10 * np.sin(0.8 * x))
    # Inject a final-window divergence: credit weakens while equities remain strong.
    tail = 45
    hyg[-tail:] = hyg[-tail:] - np.linspace(0.0, 7.0, tail)
    spy[-tail:] = spy[-tail:] + np.linspace(0.0, 5.0, tail)

    _write_series_csv(data / "HYG.csv", hyg)
    _write_series_csv(data / "LQD.csv", lqd)
    _write_series_csv(data / "SPY.csv", spy)
    np.savetxt(runs / "asset_returns.csv", np.zeros((220, 5), dtype=float), delimiter=",")

    monkeypatch.setattr(cls, "ROOT", root)
    monkeypatch.setattr(cls, "RUNS", runs)
    monkeypatch.setattr(cls, "DATA", data)
    monkeypatch.setenv("Q_DISABLE_REPORT_CARDS", "1")
    monkeypatch.setenv("Q_CREDIT_LEADLAG_EQ_SYMBOL", "SPY")

    rc = cls.main()
    assert rc == 0

    sig = np.loadtxt(runs / "credit_leadlag_signal.csv", delimiter=",").ravel()
    ov = np.loadtxt(runs / "credit_leadlag_overlay.csv", delimiter=",").ravel()
    info = json.loads((runs / "credit_leadlag_info.json").read_text(encoding="utf-8"))

    assert len(sig) == 220
    assert len(ov) == 220
    assert float(np.min(sig)) >= -1.0 - 1e-9
    assert float(np.max(sig)) <= 1.0 + 1e-9
    assert float(np.min(ov)) >= float(info["params"]["floor"]) - 1e-9
    assert float(np.max(ov)) <= float(info["params"]["ceil"]) + 1e-9
    assert ov[-1] < 1.0
    assert info["ok"] is True


def test_run_credit_leadlag_missing_inputs_writes_neutral_series(tmp_path: Path, monkeypatch):
    root = tmp_path
    runs = root / "runs_plus"
    data = root / "data"
    runs.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    np.savetxt(runs / "asset_returns.csv", np.zeros((60, 3), dtype=float), delimiter=",")

    monkeypatch.setattr(cls, "ROOT", root)
    monkeypatch.setattr(cls, "RUNS", runs)
    monkeypatch.setattr(cls, "DATA", data)
    monkeypatch.setenv("Q_DISABLE_REPORT_CARDS", "1")

    rc = cls.main()
    assert rc == 0

    sig = np.loadtxt(runs / "credit_leadlag_signal.csv", delimiter=",").ravel()
    ov = np.loadtxt(runs / "credit_leadlag_overlay.csv", delimiter=",").ravel()
    info = json.loads((runs / "credit_leadlag_info.json").read_text(encoding="utf-8"))

    assert len(sig) == 60
    assert len(ov) == 60
    assert float(np.max(np.abs(sig))) == 0.0
    assert float(np.min(ov)) == 1.0
    assert float(np.max(ov)) == 1.0
    assert info["ok"] is False

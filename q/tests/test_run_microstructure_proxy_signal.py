import json
from pathlib import Path

import numpy as np
import pandas as pd

import tools.run_microstructure_proxy_signal as ms


def _write_asset(path: Path, close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray) -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2022-01-03", periods=len(close), freq="B"),
            "Close": close.astype(float),
            "High": high.astype(float),
            "Low": low.astype(float),
            "Volume": volume.astype(float),
        }
    )
    df.to_csv(path, index=False)


def test_run_microstructure_proxy_writes_outputs(tmp_path: Path, monkeypatch):
    root = tmp_path
    runs = root / "runs_plus"
    data = root / "data"
    runs.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)

    t = 260
    x = np.linspace(0.0, 24.0, t)
    for i in range(10):
        base = 100.0 + i * 2.0
        close = base + np.cumsum(0.15 * np.sin(0.9 * x + i * 0.2))
        # Inject late stress: larger swings + lower volume and closes near lows.
        close[-45:] = close[-45:] - np.linspace(0.0, 8.0, 45)
        high = close + 0.7 + 0.06 * np.cos(0.5 * x + i)
        low = close - 0.9 - 0.04 * np.sin(0.7 * x + i)
        vol = 1_400_000 + (np.sin(x + i) * 180_000)
        vol[-45:] = np.maximum(60_000, vol[-45:] * 0.28)
        _write_asset(data / f"SYM{i}.csv", close, high, low, vol)

    np.savetxt(runs / "asset_returns.csv", np.zeros((180, 7), dtype=float), delimiter=",")

    monkeypatch.setattr(ms, "ROOT", root)
    monkeypatch.setattr(ms, "RUNS", runs)
    monkeypatch.setattr(ms, "DATA", data)
    monkeypatch.setenv("Q_DISABLE_REPORT_CARDS", "1")
    monkeypatch.setenv("Q_MICROSTRUCTURE_MIN_ASSETS", "8")

    rc = ms.main()
    assert rc == 0

    sig = np.loadtxt(runs / "microstructure_signal.csv", delimiter=",").ravel()
    ov = np.loadtxt(runs / "microstructure_overlay.csv", delimiter=",").ravel()
    info = json.loads((runs / "microstructure_info.json").read_text(encoding="utf-8"))

    assert len(sig) == 180
    assert len(ov) == 180
    assert float(np.min(sig)) >= -1.0 - 1e-9
    assert float(np.max(sig)) <= 1.0 + 1e-9
    assert float(np.min(ov)) >= float(info["params"]["floor"]) - 1e-9
    assert float(np.max(ov)) <= float(info["params"]["ceil"]) + 1e-9
    assert ov[-1] < 1.0
    assert info["ok"] is True


def test_run_microstructure_proxy_missing_inputs_writes_neutral(tmp_path: Path, monkeypatch):
    root = tmp_path
    runs = root / "runs_plus"
    data = root / "data"
    runs.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    np.savetxt(runs / "asset_returns.csv", np.zeros((50, 3), dtype=float), delimiter=",")

    monkeypatch.setattr(ms, "ROOT", root)
    monkeypatch.setattr(ms, "RUNS", runs)
    monkeypatch.setattr(ms, "DATA", data)
    monkeypatch.setenv("Q_DISABLE_REPORT_CARDS", "1")
    monkeypatch.setenv("Q_MICROSTRUCTURE_MIN_ASSETS", "8")

    rc = ms.main()
    assert rc == 0

    sig = np.loadtxt(runs / "microstructure_signal.csv", delimiter=",").ravel()
    ov = np.loadtxt(runs / "microstructure_overlay.csv", delimiter=",").ravel()
    info = json.loads((runs / "microstructure_info.json").read_text(encoding="utf-8"))

    assert len(sig) == 50
    assert len(ov) == 50
    assert float(np.max(np.abs(sig))) == 0.0
    assert float(np.min(ov)) == 1.0
    assert float(np.max(ov)) == 1.0
    assert info["ok"] is False

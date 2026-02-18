from pathlib import Path

import numpy as np

import tools.run_vol_forecast as rvf


def test_run_vol_forecast_tool_writes_overlay(tmp_path: Path, monkeypatch):
    root = tmp_path
    runs = root / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(19)
    r = rng.normal(0.0, 0.01, size=320)
    np.savetxt(runs / "daily_returns.csv", r, delimiter=",")

    monkeypatch.setattr(rvf, "ROOT", root)
    monkeypatch.setattr(rvf, "RUNS", runs)
    monkeypatch.setenv("Q_DISABLE_REPORT_CARDS", "1")

    rc = rvf.main()
    assert rc == 0

    forecast = np.loadtxt(runs / "vol_forecast.csv", delimiter=",").ravel()
    overlay = np.loadtxt(runs / "vol_forecast_overlay.csv", delimiter=",").ravel()
    assert len(forecast) == len(r)
    assert len(overlay) == len(r)
    assert np.isfinite(forecast).all()
    assert float(np.min(overlay)) >= 0.75 - 1e-9
    assert float(np.max(overlay)) <= 1.25 + 1e-9

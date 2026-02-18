from pathlib import Path

import numpy as np

import tools.run_synthetic_vol_overlay as rsv


def test_run_synthetic_vol_overlay_writes_scalar(tmp_path: Path, monkeypatch):
    root = tmp_path
    runs = root / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)

    t = 240
    rng = np.random.default_rng(9)
    daily = rng.normal(0.0, 0.01, size=t)
    np.savetxt(runs / "daily_returns.csv", daily, delimiter=",")

    # Create alternating forecast spread so signal is non-trivial.
    vf = np.full(t, 0.18, dtype=float)
    vf[: t // 2] = 0.30
    np.savetxt(runs / "vol_forecast.csv", vf, delimiter=",")

    monkeypatch.setattr(rsv, "ROOT", root)
    monkeypatch.setattr(rsv, "RUNS", runs)
    monkeypatch.setenv("Q_DISABLE_REPORT_CARDS", "1")

    rc = rsv.main()
    assert rc == 0

    out = np.loadtxt(runs / "synthetic_vol_overlay.csv", delimiter=",").ravel()
    assert len(out) == t
    assert float(np.min(out)) >= 0.75 - 1e-9
    assert float(np.max(out)) <= 1.25 + 1e-9

from pathlib import Path

import numpy as np

import tools.run_adaptive_signal_decomposition as rasd


def test_run_adaptive_signal_decomposition_writes_outputs(tmp_path: Path, monkeypatch):
    root = tmp_path
    runs = root / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)

    t = 140
    x = np.linspace(0.0, 12.0, t)
    r = np.column_stack(
        [
            0.002 * np.sin(x),
            0.0015 * np.cos(1.7 * x),
            0.001 * np.sin(0.5 * x + 0.2),
        ]
    )
    np.savetxt(runs / "asset_returns.csv", r, delimiter=",")

    monkeypatch.setattr(rasd, "ROOT", root)
    monkeypatch.setattr(rasd, "RUNS", runs)
    monkeypatch.setenv("Q_DISABLE_REPORT_CARDS", "1")

    rc = rasd.main()
    assert rc == 0

    trend = np.loadtxt(runs / "council_adaptive_trend.csv", delimiter=",")
    cycle = np.loadtxt(runs / "council_adaptive_cycle.csv", delimiter=",")
    comp = np.loadtxt(runs / "adaptive_signal_composite.csv", delimiter=",")

    assert trend.shape == r.shape
    assert cycle.shape == r.shape
    assert comp.shape == r.shape

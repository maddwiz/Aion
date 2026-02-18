from pathlib import Path

import numpy as np

import tools.run_fractal_efficiency_signal as rfes


def test_run_fractal_efficiency_signal_writes_outputs(tmp_path: Path, monkeypatch):
    root = tmp_path
    runs = root / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)

    t = 160
    x = np.linspace(0.0, 10.0, t)
    arr = np.column_stack([0.002 * np.sin(x), 0.0015 * np.cos(1.2 * x)])
    np.savetxt(runs / "asset_returns.csv", arr, delimiter=",")

    monkeypatch.setattr(rfes, "ROOT", root)
    monkeypatch.setattr(rfes, "RUNS", runs)
    monkeypatch.setenv("Q_DISABLE_REPORT_CARDS", "1")

    rc = rfes.main()
    assert rc == 0

    out = np.loadtxt(runs / "council_fractal_efficiency.csv", delimiter=",")
    assert out.shape == arr.shape
    assert float(np.max(np.abs(out))) <= 1.0 + 1e-12

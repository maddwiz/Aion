from pathlib import Path

import numpy as np

import tools.run_dream_coherence as rdc


def test_run_dream_coherence_writes_bandit_confidence_widths(tmp_path: Path, monkeypatch):
    root = tmp_path
    runs = root / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)

    t = 220
    x = np.linspace(0.0, 10.0, t)
    np.savetxt(runs / "daily_returns.csv", 0.002 * np.sin(x), delimiter=",")
    np.savetxt(runs / "reflex_latent.csv", np.sin(x - 0.1), delimiter=",")
    np.savetxt(runs / "symbolic_latent.csv", np.sin(x - 0.12), delimiter=",")
    np.savetxt(runs / "meta_mix.csv", np.sin(x - 0.08), delimiter=",")

    monkeypatch.setattr(rdc, "ROOT", root)
    monkeypatch.setattr(rdc, "RUNS", runs)
    monkeypatch.setenv("Q_DREAM_USE_BANDIT_CERTAINTY", "1")
    monkeypatch.setenv("Q_DISABLE_REPORT_CARDS", "1")
    rc = rdc.main()
    assert rc == 0

    bw = np.loadtxt(runs / "bandit_confidence_widths.csv", delimiter=",").ravel()
    assert len(bw) >= 3
    assert np.isfinite(bw).all()

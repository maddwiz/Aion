from pathlib import Path

import numpy as np
import pandas as pd

import tools.run_regime_transition_predictor as rtp


def test_run_regime_transition_predictor_writes_outputs(tmp_path: Path, monkeypatch):
    root = tmp_path
    runs = root / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)

    t = 260
    rng = np.random.default_rng(5)
    daily = rng.normal(0.0, 0.009, size=t)
    np.savetxt(runs / "daily_returns.csv", daily, delimiter=",")
    labels = np.array(["calm_trend"] * 120 + ["calm_chop"] * 70 + ["crisis"] * 70, dtype=object)
    pd.DataFrame({"regime": labels}).to_csv(runs / "regime_series.csv", index=False)

    monkeypatch.setattr(rtp, "ROOT", root)
    monkeypatch.setattr(rtp, "RUNS", runs)
    monkeypatch.setenv("Q_DISABLE_REPORT_CARDS", "1")

    rc = rtp.main()
    assert rc == 0

    scalars = np.loadtxt(runs / "regime_transition_scalar.csv", delimiter=",").ravel()
    assert len(scalars) == t
    assert np.isfinite(scalars).all()
    assert float(np.min(scalars)) >= 0.0

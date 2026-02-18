from pathlib import Path

import numpy as np
import pandas as pd

import tools.run_leadlag_network as rln


def test_run_leadlag_network_writes_signal_and_info(tmp_path: Path, monkeypatch):
    root = tmp_path
    runs = root / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)

    t = 180
    rng = np.random.default_rng(11)
    lead = rng.normal(0.0, 0.01, size=t)
    follower = np.roll(lead, 1)
    follower[0] = 0.0
    follower += rng.normal(0.0, 0.004, size=t)
    third = rng.normal(0.0, 0.01, size=t)
    arr = np.column_stack([lead, follower, third])
    np.savetxt(runs / "asset_returns.csv", arr, delimiter=",")
    pd.DataFrame({"symbol": ["AAA", "BBB", "CCC"]}).to_csv(runs / "asset_names.csv", index=False)

    monkeypatch.setattr(rln, "ROOT", root)
    monkeypatch.setattr(rln, "RUNS", runs)
    monkeypatch.setenv("Q_DISABLE_REPORT_CARDS", "1")

    rc = rln.main()
    assert rc == 0

    out = np.loadtxt(runs / "council_leadlag_network.csv", delimiter=",")
    assert out.shape == arr.shape
    assert np.isfinite(out).all()

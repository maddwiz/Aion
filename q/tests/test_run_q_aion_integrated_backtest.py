import json
from pathlib import Path

import numpy as np

import tools.run_q_aion_integrated_backtest as iqab


def test_run_q_aion_integrated_backtest_writes_outputs(tmp_path: Path, monkeypatch):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)

    t = 320
    n = 4
    rng = np.random.default_rng(13)
    rets = rng.normal(0.0004, 0.01, size=(t, n))
    w = np.full((t, n), 1.0 / n, dtype=float)

    np.savetxt(runs / "asset_returns.csv", rets, delimiter=",")
    np.savetxt(runs / "portfolio_weights_final.csv", w, delimiter=",")
    np.savetxt(runs / "daily_returns.csv", np.mean(rets, axis=1), delimiter=",")

    monkeypatch.setattr(iqab, "ROOT", tmp_path)
    monkeypatch.setattr(iqab, "RUNS", runs)
    monkeypatch.setenv("Q_STRICT_OOS_MIN_TRAIN", "200")
    monkeypatch.setenv("Q_STRICT_OOS_MIN_TEST", "80")

    rc = iqab.main()
    assert rc == 0

    out = json.loads((runs / "q_aion_integrated_backtest.json").read_text(encoding="utf-8"))
    assert out["ok"] is True
    assert (runs / "daily_returns_aion_integrated.csv").exists()
    assert int(out["rows"]) == t

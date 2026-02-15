import json

import numpy as np
import pandas as pd

import tools.run_regime_fracture as rrf


def _w(path, vals):
    np.savetxt(path, np.asarray(vals, float), delimiter=",")


def test_run_regime_fracture_writes_outputs(tmp_path, monkeypatch):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)

    # Base weights define target length.
    np.savetxt(runs / "portfolio_weights.csv", np.full((100, 4), 0.25, float), delimiter=",")
    _w(runs / "meta_mix_disagreement.csv", np.linspace(0.1, 0.9, 100))
    _w(runs / "meta_mix_quality.csv", np.linspace(0.55, 0.85, 100))
    _w(runs / "shock_mask.csv", np.linspace(0.0, 1.0, 100))
    _w(runs / "daily_returns.csv", np.concatenate([np.full(70, 0.001), np.full(30, -0.01)]))
    _w(runs / "heartbeat_stress.csv", np.linspace(0.1, 0.8, 100))
    _w(runs / "hive_persistence_governor.csv", np.linspace(1.02, 0.85, 100))
    df = pd.DataFrame(
        {
            "DATE": pd.date_range("2025-01-01", periods=100, freq="D").strftime("%Y-%m-%d"),
            "EQ": np.linspace(0.6, 0.8, 100),
            "FX": np.linspace(0.2, 0.1, 100),
            "RATES": np.linspace(0.2, 0.1, 100),
            "arb_alpha": np.linspace(1.0, 2.0, 100),
        }
    )
    df.to_csv(runs / "cross_hive_weights.csv", index=False)

    monkeypatch.setattr(rrf, "ROOT", tmp_path)
    monkeypatch.setattr(rrf, "RUNS", runs)
    rc = rrf.main(root=tmp_path, runs=runs)
    assert rc == 0

    sig = pd.read_csv(runs / "regime_fracture_signal.csv")
    gov = pd.read_csv(runs / "regime_fracture_governor.csv")
    info = json.loads((runs / "regime_fracture_info.json").read_text())

    assert len(sig) == 100
    assert len(gov) == 100
    assert set(["regime_fracture_score", "disagreement_stress", "volatility_convexity"]).issubset(sig.columns)
    assert float(sig["regime_fracture_score"].iloc[-1]) >= 0.0
    assert 0.70 <= float(gov["regime_fracture_governor"].min()) <= 1.04
    assert info["ok"] is True
    assert info["state"] in {"calm", "watch", "fracture_warn", "fracture_alert"}

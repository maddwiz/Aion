import json
from pathlib import Path

import numpy as np
import pandas as pd

from qmods.council import MeanRevRep, MomentumRep, _member_stats, run_council


def _frames(seed: int = 7, t: int = 160, n: int = 12):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-02", periods=t, freq="B")
    cols = [f"A{i:02d}" for i in range(n)]
    base = pd.DataFrame(rng.normal(0.0, 1.0, size=(t, n)), index=idx, columns=cols)
    return base


def test_member_stats_detects_anti_predictive_signal():
    fwd = _frames()
    sig_pos = fwd.copy()
    sig_neg = -fwd
    sig_noise = _frames(seed=99)

    pos = _member_stats(sig_pos, fwd, lookback=120)
    neg = _member_stats(sig_neg, fwd, lookback=120)
    noise = _member_stats(sig_noise, fwd, lookback=120)

    assert pos["quality"] > 0.5
    assert pos["polarity"] > 0.5
    assert pos["mean_ic"] > 0.2

    assert neg["quality"] > 0.5
    assert neg["polarity"] < -0.5
    assert neg["mean_ic"] < -0.2

    assert abs(noise["polarity"]) <= 0.5
    assert noise["quality"] >= 0.10


def test_run_council_emits_polarity_metadata(tmp_path: Path):
    rng = np.random.default_rng(11)
    idx = pd.date_range("2023-01-03", periods=280, freq="B")
    cols = [f"S{i:02d}" for i in range(18)]
    rets = rng.normal(0.0004, 0.012, size=(len(idx), len(cols)))
    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=cols)

    out_json = tmp_path / "council.json"
    out = run_council(prices, out_json=str(out_json))

    assert "member_quality" in out
    assert "member_polarity" in out
    assert "member_mean_ic" in out
    assert "member_ic_samples" in out
    assert set(out["member_quality"].keys()) == set(out["member_polarity"].keys())

    payload = json.loads(out_json.read_text())
    assert "member_polarity" in payload


def test_momentum_rep_single_lookback_backward_compatible():
    rng = np.random.default_rng(23)
    idx = pd.date_range("2024-01-03", periods=80, freq="B")
    prices = pd.DataFrame({"A": 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(idx))))}, index=idx)
    rep = MomentumRep(lookback=10)
    mat = rep.signal_matrix(prices)
    expected = (prices.pct_change(10) / 0.15).clip(-1.0, 1.0)
    pd.testing.assert_frame_equal(mat, expected)


def test_meanrev_rep_multi_horizon_outputs_finite_bounded_signal():
    rng = np.random.default_rng(29)
    idx = pd.date_range("2024-01-03", periods=120, freq="B")
    prices = pd.DataFrame({"A": 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, len(idx))))}, index=idx)
    rep = MeanRevRep(lookbacks=(3, 5, 8), weights=(0.5, 0.3, 0.2))
    mat = rep.signal_matrix(prices).dropna(how="all")
    assert len(mat) > 0
    assert np.isfinite(mat.to_numpy(dtype=float)).all()
    assert float(mat.max().max()) <= 1.0 + 1e-12
    assert float(mat.min().min()) >= -1.0 - 1e-12

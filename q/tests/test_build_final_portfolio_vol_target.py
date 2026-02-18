import numpy as np

import tools.build_final_portfolio as bfp


def test_compute_vol_target_scalars_are_finite_and_clipped():
    w = np.full((120, 3), 1.0 / 3.0, dtype=float)
    rng = np.random.default_rng(123)
    a = rng.normal(0.0, 0.01, size=(120, 3))
    s = bfp.compute_vol_target_scalars(
        w,
        a,
        target_annual_vol=0.10,
        lookback=20,
        min_scalar=0.4,
        max_scalar=1.6,
        smooth_alpha=0.25,
    )
    assert len(s) == 120
    assert np.isfinite(s).all()
    assert float(np.min(s)) >= 0.4 - 1e-9
    assert float(np.max(s)) <= 1.6 + 1e-9


def test_compute_vol_target_scalars_lower_in_high_vol():
    w = np.ones((80, 1), dtype=float)
    low = np.full((40, 1), 0.001, dtype=float)
    high = np.array(([0.02], [-0.02]) * 20, dtype=float)
    a = np.vstack([low, high])
    s = bfp.compute_vol_target_scalars(
        w,
        a,
        target_annual_vol=0.10,
        lookback=10,
        min_scalar=0.2,
        max_scalar=2.0,
        smooth_alpha=0.0,
    )
    assert float(np.mean(s[-20:])) < float(np.mean(s[:20]))


def test_compute_vol_target_scalars_uses_forecast_blend_when_available():
    w = np.ones((60, 1), dtype=float)
    a = np.full((60, 1), 0.001, dtype=float)
    # Forecast implies much higher future vol, so scalar should compress.
    forecast = np.full(60, 0.40, dtype=float)
    s = bfp.compute_vol_target_scalars(
        w,
        a,
        target_annual_vol=0.10,
        lookback=10,
        min_scalar=0.2,
        max_scalar=2.0,
        smooth_alpha=0.0,
        forecast_annual_vol=forecast,
        forecast_blend=1.0,
        forecast_lag=0,
    )
    assert float(np.mean(s[-20:])) < 0.5


def test_base_weight_candidates_include_regime_council_when_enabled(monkeypatch):
    monkeypatch.setenv("Q_REGIME_COUNCIL_ENABLED", "1")
    cands = bfp._base_weight_candidates()
    assert cands[0] == "runs_plus/weights_regime_council.csv"

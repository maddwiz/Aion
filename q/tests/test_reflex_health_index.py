import numpy as np

from qmods.reflex_health_index import reflex_health, reflex_health_governor


def test_reflex_health_governor_bounds_and_monotonic_mean():
    T = 260
    low = np.full(T, 0.1, dtype=float)
    high = np.full(T, 2.5, dtype=float)
    g_low = reflex_health_governor(low, lo=0.72, hi=1.10, smooth=0.88)
    g_high = reflex_health_governor(high, lo=0.72, hi=1.10, smooth=0.88)

    assert g_low.shape == (T,)
    assert g_high.shape == (T,)
    assert np.isfinite(g_low).all() and np.isfinite(g_high).all()
    assert float(np.min(g_low)) >= 0.72 - 1e-9
    assert float(np.max(g_high)) <= 1.10 + 1e-9
    assert float(np.mean(g_high)) > float(np.mean(g_low))


def test_reflex_health_penalizes_deep_drawdown():
    T = 320
    rng = np.random.default_rng(9)
    steady = 0.0008 + 0.002 * rng.standard_normal(T)
    stressed = steady.copy()
    stressed[150:170] -= 0.045  # concentrated drawdown shock

    h_steady = reflex_health(steady, lookback=126)
    h_stress = reflex_health(stressed, lookback=126)

    assert h_steady.shape == h_stress.shape == (T,)
    assert float(np.mean(h_stress[190:260])) < float(np.mean(h_steady[190:260]))

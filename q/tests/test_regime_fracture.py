import numpy as np

from qmods.regime_fracture import (
    breadth_stress_from_weights,
    fracture_governor,
    realized_vol_convexity,
    rolling_percentile_stress,
    smooth_ema,
)


def test_rolling_percentile_stress_increases_on_ramp():
    x = np.concatenate([np.linspace(0.0, 1.0, 80), np.linspace(1.0, 0.1, 39), [2.0]])
    s = rolling_percentile_stress(x, window=30, min_periods=8)
    assert len(s) == len(x)
    assert float(s[-1]) > float(s[-2])
    assert float(np.min(s)) >= 0.0
    assert float(np.max(s)) <= 1.0


def test_realized_vol_convexity_detects_late_spike():
    rng = np.random.default_rng(7)
    calm = 0.002 * rng.standard_normal(100)
    stress = 0.020 * rng.standard_normal(40)
    r = np.concatenate([calm, stress])
    c = realized_vol_convexity(r, short_w=10, long_w=63)
    assert len(c) == len(r)
    assert float(np.mean(c[-25:])) > float(np.mean(c[20:60]))
    assert float(np.min(c)) >= 0.0
    assert float(np.max(c)) <= 1.0


def test_breadth_stress_rises_with_concentration_and_turnover():
    t = 120
    w = np.full((t, 3), 1.0 / 3.0, float)
    # Late phase: concentrate + rotate.
    for i in range(80, t):
        if i % 2 == 0:
            w[i] = np.array([0.85, 0.10, 0.05], float)
        else:
            w[i] = np.array([0.10, 0.85, 0.05], float)
    b = breadth_stress_from_weights(w)
    assert len(b) == t
    assert float(np.mean(b[-20:])) > float(np.mean(b[20:60]))
    assert float(np.min(b)) >= 0.0
    assert float(np.max(b)) <= 1.0


def test_fracture_governor_cuts_more_on_higher_stress():
    f = np.array([0.10, 0.40, 0.75, 0.90], float)
    g = fracture_governor(f, alpha=0.32, min_gov=0.72, max_gov=1.04)
    assert len(g) == len(f)
    assert g[0] > g[1] > g[2] >= g[3]
    assert float(np.min(g)) >= 0.72 - 1e-9
    assert float(np.max(g)) <= 1.04 + 1e-9


def test_smooth_ema_reduces_jumps():
    x = np.array([0.0, 1.0, 0.0, 1.0, 0.0], float)
    z = smooth_ema(x, inertia=0.85)
    raw_jump = float(np.max(np.abs(np.diff(x))))
    sm_jump = float(np.max(np.abs(np.diff(z))))
    assert sm_jump < raw_jump

import numpy as np

from qmods.guardrails_bundle import (
    apply_turnover_budget_governor,
    disagreement_gate_series,
    regime_governor_from_returns,
    stability_governor,
)


def test_regime_governor_bounds_and_shape():
    rng = np.random.default_rng(7)
    r = rng.normal(0.0002, 0.01, 400)
    dna = np.zeros(400, dtype=float)
    dna[-50:] = 1.0
    g = regime_governor_from_returns(r, lookback=63, dna_state=dna)
    assert g.shape == (400,)
    assert np.isfinite(g).all()
    assert float(np.min(g)) >= 0.45 - 1e-9
    assert float(np.max(g)) <= 1.10 + 1e-9


def test_stability_governor_penalizes_high_churn():
    # Low churn weights
    w1 = np.zeros((300, 5), dtype=float)
    w1[:, 0] = 0.2
    # High churn weights
    w2 = np.zeros((300, 5), dtype=float)
    for t in range(300):
        w2[t, t % 5] = 0.2
        w2[t, (t + 1) % 5] = -0.2
    g1 = stability_governor(w1, lookback=21)
    g2 = stability_governor(w2, lookback=21)
    assert g1.shape == g2.shape == (300,)
    assert np.isfinite(g1).all() and np.isfinite(g2).all()
    assert float(np.mean(g2)) < float(np.mean(g1))
    assert float(np.min(g2)) >= 0.55 - 1e-9


def test_turnover_budget_governor_enforces_rolling_limit():
    T = 120
    N = 4
    w = np.zeros((T, N), dtype=float)
    # High-churn rotating target.
    for t in range(T):
        k = t % N
        w[t, k] = 0.45
        w[t, (k + 1) % N] = -0.45

    window = 5
    limit = 0.9
    res = apply_turnover_budget_governor(w, max_step_turnover=0.35, budget_window=window, budget_limit=limit)
    assert res.weights.shape == w.shape
    assert res.turnover_after.shape == (T - 1,)
    assert res.rolling_turnover_after.shape == (T - 1,)
    assert np.isfinite(res.weights).all()
    assert np.max(res.turnover_after) <= 0.35 + 1e-6
    assert np.max(res.rolling_turnover_after) <= limit + 1e-6
    assert float(np.mean(res.turnover_after)) <= float(np.mean(res.turnover_before))


def test_disagreement_gate_series_penalizes_high_dispersion():
    T = 260
    low = np.zeros((T, 4), dtype=float)
    low[:, 0] = 0.4
    low[:, 1] = 0.42
    low[:, 2] = 0.39
    low[:, 3] = 0.41

    high = np.zeros((T, 4), dtype=float)
    for t in range(T):
        s = 1.0 if (t % 2 == 0) else -1.0
        high[t] = np.array([0.9 * s, -0.9 * s, 0.8 * s, -0.8 * s], dtype=float)

    g_low = disagreement_gate_series(low, clamp=(0.45, 1.0), lookback=63, smooth=0.85)
    g_high = disagreement_gate_series(high, clamp=(0.45, 1.0), lookback=63, smooth=0.85)

    assert g_low.shape == g_high.shape == (T,)
    assert np.isfinite(g_low).all() and np.isfinite(g_high).all()
    assert float(np.min(g_low)) >= 0.45 - 1e-9
    assert float(np.max(g_low)) <= 1.0 + 1e-9
    assert float(np.mean(g_high)) < float(np.mean(g_low))

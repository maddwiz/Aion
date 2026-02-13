import numpy as np

from qmods.guardrails_bundle import regime_governor_from_returns, stability_governor


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

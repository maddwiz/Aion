import numpy as np

from qmods.cross_hive_arb_v1 import arb_weights


def test_arb_weights_sum_and_bounds():
    T = 120
    scores = {
        "EQ": np.linspace(-0.2, 0.6, T),
        "FX": np.linspace(0.3, -0.1, T),
        "RATES": np.sin(np.linspace(0, 6, T)) * 0.2,
    }
    names, W = arb_weights(
        scores,
        alpha=2.0,
        inertia=0.85,
        min_weight=0.02,
        max_weight=0.70,
    )
    assert names == ["EQ", "FX", "RATES"]
    assert W.shape == (T, 3)
    assert np.isfinite(W).all()
    rs = W.sum(axis=1)
    assert np.allclose(rs, 1.0, atol=1e-6)
    assert float(np.min(W)) >= 0.0
    assert float(np.max(W)) <= 1.0 + 1e-9


def test_arb_weights_inertia_reduces_turnover():
    T = 200
    rng = np.random.default_rng(3)
    scores = {
        "A": rng.normal(0, 1, T),
        "B": rng.normal(0, 1, T),
        "C": rng.normal(0, 1, T),
    }
    _, w_fast = arb_weights(scores, inertia=0.0, alpha=2.0)
    _, w_slow = arb_weights(scores, inertia=0.9, alpha=2.0)
    t_fast = float(np.mean(np.sum(np.abs(np.diff(w_fast, axis=0)), axis=1)))
    t_slow = float(np.mean(np.sum(np.abs(np.diff(w_slow, axis=0)), axis=1)))
    assert t_slow < t_fast

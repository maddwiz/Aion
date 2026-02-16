import numpy as np
from pathlib import Path

import tools.run_meta_execution_gate as rmeg


def test_prob_to_gate_respects_bounds_and_warmup():
    p = np.linspace(0.1, 0.9, 20)
    g = rmeg._prob_to_gate(
        p,
        threshold=0.55,
        floor=0.4,
        ceiling=1.1,
        slope=8.0,
        warmup=5,
    )
    assert g.shape[0] == p.shape[0]
    assert np.all(g >= 0.4 - 1e-12)
    assert np.all(g <= 1.1 + 1e-12)
    assert np.allclose(g[:5], np.ones(5))


def test_ridge_walkforward_prob_outputs_finite_probabilities():
    n = 260
    x0 = np.linspace(-1.0, 1.0, n)
    x1 = np.sin(np.linspace(0.0, 6.0, n))
    X = np.column_stack([x0, x1])
    y = (x0 + 0.2 * x1 > 0.05).astype(float)
    p = rmeg._ridge_walkforward_prob(X, y, min_train=80, l2=2.0, min_prob=0.1, max_prob=0.9)
    assert p.shape[0] == n
    assert np.all(np.isfinite(p))
    assert np.all(p >= 0.1 - 1e-12)
    assert np.all(p <= 0.9 + 1e-12)
    # Model should have signal in later region.
    assert float(np.mean(p[-30:])) > float(np.mean(p[80:110]))


def test_assemble_features_falls_back_to_return_state_when_files_missing(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(rmeg, "RUNS", tmp_path)
    r = np.linspace(-0.01, 0.01, 120)
    X, names = rmeg._assemble_features(r)
    assert X.shape[0] == r.shape[0]
    assert X.shape[1] >= 2
    assert "ret_lag1" in names
    assert "absret_roll21" in names

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


def test_adaptive_threshold_tightens_on_losing_streak():
    r = np.array([-0.01] * 40 + [0.005] * 5, dtype=float)
    th = rmeg._adaptive_threshold_series(
        r,
        base_threshold=0.53,
        enabled=True,
        hit_window=20,
        tighten_step=0.04,
        loosen_step=0.02,
        threshold_min=0.42,
        threshold_max=0.68,
        ema_alpha=1.0,
    )
    assert float(th[-1]) > 0.53


def test_adaptive_threshold_loosens_on_winning_streak():
    r = np.array([0.01] * 45 + [-0.005] * 3, dtype=float)
    th = rmeg._adaptive_threshold_series(
        r,
        base_threshold=0.53,
        enabled=True,
        hit_window=20,
        tighten_step=0.04,
        loosen_step=0.02,
        threshold_min=0.42,
        threshold_max=0.68,
        ema_alpha=1.0,
    )
    assert float(th[-1]) < 0.53


def test_adaptive_threshold_respects_bounds():
    r = np.array([-0.01, 0.01] * 120, dtype=float)
    th = rmeg._adaptive_threshold_series(
        r,
        base_threshold=0.53,
        enabled=True,
        hit_window=20,
        tighten_step=0.20,
        loosen_step=0.20,
        threshold_min=0.42,
        threshold_max=0.68,
        ema_alpha=1.0,
    )
    assert float(np.min(th)) >= 0.42 - 1e-12
    assert float(np.max(th)) <= 0.68 + 1e-12


def test_adaptive_threshold_disabled_keeps_base_value():
    r = np.array([-0.01, 0.01, -0.02, 0.03], dtype=float)
    th = rmeg._adaptive_threshold_series(
        r,
        base_threshold=0.53,
        enabled=False,
        hit_window=20,
        tighten_step=0.04,
        loosen_step=0.02,
        threshold_min=0.42,
        threshold_max=0.68,
        ema_alpha=0.15,
    )
    assert np.allclose(th, np.full_like(th, 0.53))

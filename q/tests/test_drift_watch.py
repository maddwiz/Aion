import numpy as np

from qmods.drift_watch import compute_weight_drift


def test_compute_weight_drift_basic_metrics():
    prev = np.array([[0.1, -0.1], [0.2, -0.2], [0.3, -0.3]], dtype=float)
    cur = np.array([[0.1, -0.1], [0.15, -0.15], [0.4, -0.4]], dtype=float)
    out = compute_weight_drift(cur, prev)
    assert out["rows_overlap"] == 3
    assert out["cols_overlap"] == 2
    assert out["latest_l1"] > 0.0
    assert out["mean_l1"] > 0.0
    assert out["p95_l1"] >= out["mean_l1"]


def test_compute_weight_drift_handles_empty_overlap():
    cur = np.zeros((0, 2), dtype=float)
    prev = np.zeros((0, 2), dtype=float)
    out = compute_weight_drift(cur, prev)
    assert out["rows_overlap"] == 0
    assert out["latest_l1"] == 0.0

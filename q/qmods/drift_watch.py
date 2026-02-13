from __future__ import annotations

import numpy as np


def _as_mat(a):
    x = np.asarray(a, float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x


def compute_weight_drift(current, previous):
    cur = _as_mat(current)
    prev = _as_mat(previous)
    T = min(cur.shape[0], prev.shape[0])
    N = min(cur.shape[1], prev.shape[1])
    if T <= 0 or N <= 0:
        return {
            "rows_overlap": 0,
            "cols_overlap": 0,
            "latest_l1": 0.0,
            "latest_l2": 0.0,
            "mean_l1": 0.0,
            "p95_l1": 0.0,
        }
    c = cur[-T:, :N]
    p = prev[-T:, :N]
    d = c - p
    l1 = np.sum(np.abs(d), axis=1)
    l2 = np.sqrt(np.sum(d * d, axis=1))
    return {
        "rows_overlap": int(T),
        "cols_overlap": int(N),
        "latest_l1": float(l1[-1]),
        "latest_l2": float(l2[-1]),
        "mean_l1": float(np.mean(l1)),
        "p95_l1": float(np.percentile(l1, 95)),
    }

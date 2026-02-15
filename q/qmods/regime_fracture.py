from __future__ import annotations

import numpy as np


def _clip01(x):
    return np.clip(np.asarray(x, float), 0.0, 1.0)


def rolling_percentile_stress(x, window: int = 63, min_periods: int = 12) -> np.ndarray:
    a = np.asarray(x, float).ravel()
    n = len(a)
    if n == 0:
        return np.asarray([], float)
    w = max(2, int(window))
    mp = max(2, int(min_periods))
    out = np.zeros(n, float)
    for i in range(n):
        lo = max(0, i - w + 1)
        seg = a[lo : i + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) < mp:
            out[i] = 0.0
            continue
        v = a[i]
        if not np.isfinite(v):
            out[i] = 0.0
            continue
        out[i] = float(np.mean(seg <= v))
    return _clip01(out)


def realized_vol_convexity(returns: np.ndarray, short_w: int = 10, long_w: int = 63) -> np.ndarray:
    r = np.asarray(returns, float).ravel()
    n = len(r)
    if n == 0:
        return np.asarray([], float)
    sw = max(2, int(short_w))
    lw = max(sw + 1, int(long_w))
    out = np.zeros(n, float)
    for i in range(n):
        lo_s = max(0, i - sw + 1)
        lo_l = max(0, i - lw + 1)
        s = r[lo_s : i + 1]
        l = r[lo_l : i + 1]
        s = s[np.isfinite(s)]
        l = l[np.isfinite(l)]
        if len(s) < 2 or len(l) < 6:
            continue
        vol_s = float(np.std(s, ddof=1))
        vol_l = float(np.std(l, ddof=1))
        ratio = vol_s / max(1e-9, vol_l)
        out[i] = float(np.clip((ratio - 1.0) / 1.0, 0.0, 1.0))
    return _clip01(out)


def breadth_stress_from_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, float)
    if w.ndim != 2 or w.shape[0] == 0:
        return np.asarray([], float)
    t, n = w.shape
    n = max(1, int(n))
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, 0.0, None)
    rs = w.sum(axis=1, keepdims=True)
    rs = np.where(rs <= 0.0, 1.0, rs)
    w = w / rs

    hhi = np.sum(w * w, axis=1)
    base = 1.0 / float(n)
    conc = np.clip((hhi - base) / max(1e-9, 1.0 - base), 0.0, 1.0)

    turn = np.zeros(t, float)
    if t > 1:
        turn[1:] = np.sum(np.abs(np.diff(w, axis=0)), axis=1)
    turn_p = rolling_percentile_stress(turn, window=63, min_periods=12)
    return _clip01(0.60 * conc + 0.40 * turn_p)


def smooth_ema(x: np.ndarray, inertia: float = 0.85) -> np.ndarray:
    a = np.asarray(x, float).ravel()
    if len(a) == 0:
        return a
    z = a.copy()
    k = float(np.clip(inertia, 0.0, 0.99))
    for i in range(1, len(z)):
        z[i] = k * z[i - 1] + (1.0 - k) * z[i]
    return z


def fracture_governor(
    fracture_score: np.ndarray,
    alpha: float = 0.32,
    min_gov: float = 0.72,
    max_gov: float = 1.04,
) -> np.ndarray:
    f = _clip01(fracture_score).ravel()
    if len(f) == 0:
        return f
    g = float(max_gov) - float(alpha) * f
    g = np.where(f >= 0.85, g - 0.07, g)
    g = np.where((f >= 0.72) & (f < 0.85), g - 0.04, g)
    return np.clip(g, float(min_gov), float(max_gov))


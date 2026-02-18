from __future__ import annotations

import numpy as np


def _safe_1d(x) -> np.ndarray:
    a = np.asarray(x, float).ravel()
    if a.size == 0:
        return np.zeros(0, dtype=float)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def _smooth_ewm(x: np.ndarray, beta: float = 0.88) -> np.ndarray:
    a = _safe_1d(x)
    if len(a) <= 1:
        return a
    b = float(np.clip(beta, 0.0, 0.99))
    out = a.copy()
    for t in range(1, len(out)):
        out[t] = b * out[t - 1] + (1.0 - b) * out[t]
    return out


def _rolling_quality_from_pnl(
    pnl: np.ndarray,
    win: int = 63,
    min_periods: int = 15,
    scale: float = 1.5,
) -> np.ndarray:
    p = _safe_1d(pnl)
    T = len(p)
    if T == 0:
        return np.zeros(0, dtype=float)
    w = int(max(8, win))
    mp = int(max(6, min_periods))
    q = np.full(T, 0.5, dtype=float)
    for t in range(T):
        lo = max(0, t - w + 1)
        seg = p[lo : t + 1]
        if len(seg) < mp:
            continue
        mu = float(np.mean(seg))
        sd = float(np.std(seg, ddof=1)) + 1e-12
        sh = mu / sd
        q[t] = float(np.clip(0.5 + 0.5 * np.tanh(sh / max(1e-6, scale)), 0.0, 1.0))
    return q


def rolling_component_quality(
    signal: np.ndarray,
    forward_returns: np.ndarray,
    gross: float = 0.24,
    win: int = 63,
    min_periods: int = 15,
    smooth: float = 0.85,
) -> np.ndarray:
    """
    Rolling directional quality of one component signal.
    signal[t] predicts returns[t+1] (forward_returns is returns[1:]).
    Returns quality at signal granularity (length == len(signal), values in [0,1]).
    """
    s = _safe_1d(signal)
    y = _safe_1d(forward_returns)
    T = min(len(s), len(y) + 1)
    if T <= 1:
        return np.full(T, 0.5, dtype=float)
    s = s[:T]
    y = y[: T - 1]

    g = float(np.clip(gross, 0.05, 1.50))
    pos = np.tanh(g * s)
    pnl = pos[:-1] * y
    q = _rolling_quality_from_pnl(pnl, win=win, min_periods=min_periods, scale=1.5)
    out = np.full(T, 0.5, dtype=float)
    out[1:] = q
    return np.clip(_smooth_ewm(out, beta=float(np.clip(smooth, 0.0, 0.98))), 0.0, 1.0)


def adaptive_blend_series(
    meta_signal: np.ndarray,
    syn_signal: np.ndarray,
    meta_conf: np.ndarray,
    syn_conf: np.ndarray,
    forward_returns: np.ndarray,
    base_alpha: float,
    base_gross: float,
    quality_sensitivity: float = 0.55,
    conf_sensitivity: float = 0.25,
    alpha_smooth: float = 0.90,
    gross_smooth: float = 0.88,
    alpha_bounds: tuple[float, float] = (0.05, 0.95),
    gross_bounds: tuple[float, float] = (0.12, 0.45),
) -> dict:
    """
    Build adaptive alpha/gross schedules using:
    - relative rolling efficacy (meta vs synapses)
    - confidence differential
    - disagreement penalty
    """
    m = _safe_1d(meta_signal)
    s = _safe_1d(syn_signal)
    mc = np.clip(_safe_1d(meta_conf), 0.0, 1.0)
    sc = np.clip(_safe_1d(syn_conf), 0.0, 1.0)
    y = _safe_1d(forward_returns)

    T = min(len(m), len(s), len(mc), len(sc), len(y) + 1)
    if T <= 1:
        z = np.zeros(T, dtype=float)
        return {
            "alpha": np.full(T, float(np.clip(base_alpha, 0.0, 1.0)), dtype=float),
            "gross": np.full(T, float(np.clip(base_gross, gross_bounds[0], gross_bounds[1])), dtype=float),
            "quality_meta": np.full(T, 0.5, dtype=float),
            "quality_syn": np.full(T, 0.5, dtype=float),
            "quality_mix": np.full(T, 0.5, dtype=float),
            "disagreement_norm": z,
        }

    m = m[:T]
    s = s[:T]
    mc = mc[:T]
    sc = sc[:T]
    y = y[: T - 1]

    q_m = rolling_component_quality(m, y, gross=base_gross, win=63, min_periods=15, smooth=0.85)
    q_s = rolling_component_quality(s, y, gross=base_gross, win=63, min_periods=15, smooth=0.85)
    q_delta = np.clip(q_m - q_s, -1.0, 1.0)
    c_delta = np.clip(mc - sc, -1.0, 1.0)

    lo_a, hi_a = float(min(alpha_bounds)), float(max(alpha_bounds))
    lo_g, hi_g = float(min(gross_bounds)), float(max(gross_bounds))
    a0 = float(np.clip(base_alpha, lo_a, hi_a))
    g0 = float(np.clip(base_gross, lo_g, hi_g))

    alpha_raw = a0 + float(quality_sensitivity) * q_delta + float(conf_sensitivity) * c_delta
    alpha = np.clip(_smooth_ewm(alpha_raw, beta=float(np.clip(alpha_smooth, 0.0, 0.99))), lo_a, hi_a)

    disagree = np.abs(m - s)
    p90 = float(np.percentile(disagree, 90)) if len(disagree) else 0.0
    disagree_n = np.clip(disagree / (p90 + 1e-9), 0.0, 1.0)
    q_mix = np.clip(alpha * q_m + (1.0 - alpha) * q_s, 0.0, 1.0)

    gross_raw = g0 * (0.82 + 0.60 * q_mix) * (1.0 - 0.20 * disagree_n)
    gross = np.clip(_smooth_ewm(gross_raw, beta=float(np.clip(gross_smooth, 0.0, 0.99))), lo_g, hi_g)

    return {
        "alpha": alpha,
        "gross": gross,
        "quality_meta": q_m,
        "quality_syn": q_s,
        "quality_mix": q_mix,
        "disagreement_norm": disagree_n,
    }

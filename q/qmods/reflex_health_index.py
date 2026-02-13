#!/usr/bin/env python3
import numpy as np

def reflex_health(latent_returns: np.ndarray, lookback=126):
    r = np.asarray(latent_returns, float)
    out = np.zeros_like(r, float)
    for i in range(len(r)):
        j = max(0, i - lookback + 1)
        w = r[j:i+1]
        mu = np.nanmean(w)
        sd = np.nanstd(w)
        if not np.isfinite(sd) or sd < 1e-6:
            out[i] = 0.0
            continue
        raw = (mu / (sd + 1e-12)) * np.sqrt(252.0)
        # Clamp to avoid exploding values when window std is tiny.
        out[i] = float(np.clip(raw, 0.0, 5.0))
    return out

def gate_reflex(reflex_signal: np.ndarray, health: np.ndarray, min_h=0.5):
    s = np.asarray(reflex_signal, float)
    h = np.asarray(health, float)
    scale = np.clip(h / (min_h + 1e-12), 0.0, 1.0)
    L = min(len(s), len(scale))
    return s[:L] * scale[:L]


def reflex_health_governor(health: np.ndarray, lo: float = 0.72, hi: float = 1.10, smooth: float = 0.88):
    """
    Map reflex health (annualized Sharpe-like in [0,5]) to exposure governor.
    """
    h = np.asarray(health, float).ravel()
    if h.size == 0:
        return h
    q = np.clip(np.tanh(np.clip(h, 0.0, 5.0) / 2.0), 0.0, 1.0)
    g = float(lo) + (float(hi) - float(lo)) * q
    g = np.clip(g, min(lo, hi), max(lo, hi))
    a = float(np.clip(smooth, 0.0, 0.99))
    if a > 0.0 and len(g) > 1:
        out = g.copy()
        for t in range(1, len(out)):
            out[t] = a * out[t - 1] + (1.0 - a) * out[t]
        g = np.clip(out, min(lo, hi), max(lo, hi))
    return g

#!/usr/bin/env python3
import numpy as np


def _rolling_dd_penalty(window_rets: np.ndarray, dd_ref: float = 0.35) -> float:
    w = np.asarray(window_rets, float).ravel()
    if len(w) == 0:
        return 0.0
    eq = np.cumprod(1.0 + np.clip(w, -0.95, 0.95))
    peak = np.maximum.accumulate(eq)
    dd = eq / (peak + 1e-12) - 1.0
    mdd = float(np.abs(np.min(dd))) if len(dd) else 0.0
    return float(np.clip(mdd / max(1e-6, dd_ref), 0.0, 1.0))


def reflex_health(latent_returns: np.ndarray, lookback=126, return_components: bool = False):
    r = np.asarray(latent_returns, float)
    out = np.zeros_like(r, float)
    base_sharpe = np.zeros_like(r, float)
    dd_pen = np.zeros_like(r, float)
    instab_pen = np.zeros_like(r, float)
    downside_pen = np.zeros_like(r, float)
    for i in range(len(r)):
        j = max(0, i - lookback + 1)
        w = r[j:i+1]
        mu = np.nanmean(w)
        sd = np.nanstd(w)
        if not np.isfinite(sd) or sd < 1e-6:
            out[i] = 0.0
            continue
        raw = float((mu / (sd + 1e-12)) * np.sqrt(252.0))
        base = float(np.clip(raw, 0.0, 5.0))
        ddp = _rolling_dd_penalty(w, dd_ref=0.35)
        # Volatility-of-returns proxy for reflex instability.
        inst = float(np.nanstd(np.diff(w))) if len(w) > 1 else 0.0
        inst_ref = float(np.nanstd(w) + 1e-6)
        ip = float(np.clip(inst / (2.5 * inst_ref + 1e-12), 0.0, 1.0))
        # Downside pressure from persistent losses + left-tail severity.
        neg_freq = float(np.mean(w < 0.0)) if len(w) else 0.0
        q10 = float(np.nanpercentile(w, 10)) if len(w) else 0.0
        tail_mag = float(np.clip(abs(min(q10, 0.0)) / (2.2 * (inst_ref + 1e-12)), 0.0, 1.0))
        dsp = float(np.clip(0.55 * neg_freq + 0.45 * tail_mag, 0.0, 1.0))

        adj = base * (1.0 - 0.35 * ddp) * (1.0 - 0.12 * ip) * (1.0 - 0.18 * dsp)
        # Clamp to avoid exploding values when window std is tiny.
        out[i] = float(np.clip(adj, 0.0, 5.0))
        base_sharpe[i] = base
        dd_pen[i] = ddp
        instab_pen[i] = ip
        downside_pen[i] = dsp
    if return_components:
        return out, {
            "base_sharpe": base_sharpe,
            "drawdown_penalty": dd_pen,
            "instability_penalty": instab_pen,
            "downside_penalty": downside_pen,
        }
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

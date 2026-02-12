from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_weights(raw: pd.Series, min_w: float, max_w: float) -> pd.Series:
    if raw.empty:
        return raw
    raw = raw.clip(lower=0.0)
    total = float(raw.sum())
    if total <= 0:
        return raw
    w = raw / total
    w = w.clip(lower=min_w)
    w = w / max(w.sum(), 1e-9)
    w = w.clip(upper=max_w)
    w = w / max(w.sum(), 1e-9)
    return w


def allocate_candidates(candidates: list[dict], equity: float, cfg) -> dict:
    if not candidates or equity <= 0:
        return {}

    top = sorted(candidates, key=lambda c: c.get("confidence", 0.0), reverse=True)[: cfg.PORTFOLIO_MAX_CANDIDATES]

    names = [c["symbol"] for c in top]
    exp = pd.Series([max(0.0, float(c.get("expected_edge", c.get("confidence", 0.0)))) for c in top], index=names)
    vol = pd.Series([max(1e-6, float(c.get("volatility", 0.01))) for c in top], index=names)

    ret_df = pd.DataFrame({c["symbol"]: c.get("returns", pd.Series(dtype=float)) for c in top}).dropna(how="all")
    if not ret_df.empty and len(ret_df.columns) > 1:
        corr = ret_df.corr().fillna(0.0)
    else:
        corr = pd.DataFrame(np.eye(len(names)), index=names, columns=names)

    avg_corr = corr.abs().mean(axis=1)
    corr_penalty = (1.0 - cfg.PORTFOLIO_CORR_PENALTY * avg_corr).clip(lower=0.25)

    base = (exp / vol).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    adjusted = base * corr_penalty

    weights = _normalize_weights(adjusted, cfg.PORTFOLIO_MIN_WEIGHT, cfg.PORTFOLIO_MAX_WEIGHT)
    if weights.empty:
        return {}

    max_gross_notional = equity * cfg.MAX_GROSS_LEVERAGE
    alloc_notional = {}
    for sym, w in weights.items():
        side = next((c["side"] for c in top if c["symbol"] == sym), "LONG")
        notional = float(max_gross_notional * w)
        alloc_notional[sym] = {
            "target_notional": notional,
            "weight": float(w),
            "side": side,
        }

    return alloc_notional

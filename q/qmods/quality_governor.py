from __future__ import annotations

import numpy as np


def _clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def sharpe_quality(sh: float | None) -> float | None:
    if sh is None:
        return None
    if not np.isfinite(sh):
        return None
    # -1 -> ~0.12, 0 -> 0.50, +1 -> ~0.88
    return _clip01(0.5 + 0.5 * np.tanh(float(sh)))


def hit_quality(hit: float | None, center: float = 0.5, span: float = 0.20) -> float | None:
    if hit is None:
        return None
    if not np.isfinite(hit):
        return None
    span = max(1e-6, float(span))
    return _clip01((float(hit) - (float(center) - span)) / (2.0 * span))


def drawdown_quality(max_dd: float | None, bad_dd: float = 0.35) -> float | None:
    if max_dd is None:
        return None
    if not np.isfinite(max_dd):
        return None
    # max_dd is usually negative; magnitude near bad_dd => low quality.
    dd = abs(float(max_dd))
    bad = max(1e-6, float(bad_dd))
    return _clip01(1.0 - dd / bad)


def blend_quality(components: dict[str, tuple[float | None, float]]) -> tuple[float, dict]:
    """
    components:
      name -> (value_in_0_1_or_None, weight)
    Returns:
      (blended_quality_in_0_1, details)
    """
    kept = {}
    wsum = 0.0
    acc = 0.0
    for name, (val, w) in components.items():
        ww = max(0.0, float(w))
        if val is None or not np.isfinite(val) or ww <= 0.0:
            continue
        vv = _clip01(val)
        kept[name] = {"value": vv, "weight": ww}
        acc += ww * vv
        wsum += ww
    if wsum <= 0.0:
        return 0.50, {"used_components": kept, "weight_sum": 0.0}
    out = _clip01(acc / wsum)
    return out, {"used_components": kept, "weight_sum": float(wsum)}


def build_governor_series(
    length: int,
    base_quality: float,
    disagreement_gate: np.ndarray | None = None,
    global_governor: np.ndarray | None = None,
    lo: float = 0.55,
    hi: float = 1.15,
    smooth: float = 0.85,
) -> np.ndarray:
    """
    Build a bounded exposure scaler from a quality score with optional time-varying modifiers.
    """
    T = int(max(0, length))
    if T <= 0:
        return np.zeros(0, dtype=float)

    q = _clip01(base_quality)
    base = float(lo) + (float(hi) - float(lo)) * q
    out = np.full(T, base, dtype=float)

    if disagreement_gate is not None:
        dg = np.asarray(disagreement_gate, float).ravel()
        L = min(T, len(dg))
        if L > 0:
            # lower disagreement gate should reduce exposure; keep effect modest.
            out[:L] *= np.clip(0.80 + 0.20 * np.clip(dg[:L], 0.0, 1.0), 0.70, 1.05)

    if global_governor is not None:
        gg = np.asarray(global_governor, float).ravel()
        L = min(T, len(gg))
        if L > 0:
            # couple to global governor but keep quality governor primary.
            out[:L] *= np.clip(0.90 + 0.10 * np.clip(gg[:L], 0.30, 1.20), 0.85, 1.08)

    out = np.clip(out, float(lo), float(hi))

    a = float(np.clip(smooth, 0.0, 0.98))
    if a > 0.0 and T > 1:
        for t in range(1, T):
            out[t] = a * out[t - 1] + (1.0 - a) * out[t]
    return np.clip(out, float(lo), float(hi))

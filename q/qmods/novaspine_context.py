from __future__ import annotations

import numpy as np


def clip01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def context_resonance(
    count: int | float | None,
    top_k: int | float | None,
    scores: list[float] | None = None,
) -> float:
    c = max(0.0, float(count or 0.0))
    k = max(1.0, float(top_k or 1.0))
    cov = clip01(c / k)
    if scores:
        s = [clip01(float(x)) for x in scores if np.isfinite(float(x))]
        avg = float(np.mean(s)) if s else 0.0
    else:
        avg = 0.0
    # Coverage matters slightly more than score.
    return clip01(0.55 * cov + 0.45 * avg)


def context_boost(resonance: float, status_ok: bool = True) -> float:
    """
    Convert memory resonance into a gentle exposure scaler.
    - Neutral is 1.00 when unavailable.
    - Active range is bounded for safety.
    """
    r = clip01(resonance)
    if not bool(status_ok):
        return 1.0
    return float(np.clip(0.94 + 0.14 * r, 0.90, 1.10))


def turnover_pressure(
    mean_turnover: float | None,
    max_turnover: float | None,
    rolling_turnover_max: float | None,
    mean_limit: float = 0.45,
    max_limit: float = 1.00,
    rolling_limit: float = 1.25,
) -> float:
    """
    Normalize cross-hive turnover stress into a bounded [0, 1] pressure score.
    0 means stable churn; 1 means severe churn beyond configured limits.
    """

    def _ratio(v, lim):
        if v is None:
            return None
        try:
            x = float(v)
            l = float(lim)
        except Exception:
            return None
        if not np.isfinite(x) or not np.isfinite(l) or l <= 0.0:
            return None
        return max(0.0, x) / l

    r_mean = _ratio(mean_turnover, mean_limit)
    r_max = _ratio(max_turnover, max_limit)
    r_roll = _ratio(rolling_turnover_max, rolling_limit)
    active = [r for r in (r_mean, r_max, r_roll) if r is not None]
    if not active:
        return 0.0
    # No pressure until roughly 60% of threshold usage; full pressure by 160%.
    shifted = [np.clip((r - 0.60) / 1.00, 0.0, 1.0) for r in active]
    return clip01(float(np.mean(shifted)))


def apply_turnover_dampener(boost: float, pressure: float, max_cut: float = 0.12) -> float:
    """
    Reduce a memory-derived boost under churn stress.
    - pressure in [0, 1]
    - max_cut is the largest absolute downward cut applied at full pressure.
    """
    b = float(np.clip(float(boost), 0.85, 1.10))
    p = clip01(pressure)
    cut = float(np.clip(float(max_cut), 0.0, 0.30)) * p
    return float(np.clip(b - cut, 0.85, 1.10))


def build_context_query(
    top_hives: list[str] | None,
    quality_score: float | None,
    alerts: list[str] | None,
) -> str:
    hives = [str(h).upper() for h in (top_hives or []) if str(h).strip()]
    hs = ", ".join(hives[:4]) if hives else "mixed market hives"
    q = float(quality_score) if quality_score is not None else 0.5
    a = ", ".join((alerts or [])[:3]) if (alerts or []) else "none"
    return (
        f"Recall prior decisions and outcomes for regimes similar to current {hs}. "
        f"Current system quality score is {q:.3f}. Active alerts: {a}. "
        "Focus on high-confidence failure/success patterns that should adjust risk now."
    )

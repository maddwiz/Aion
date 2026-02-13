from __future__ import annotations

import numpy as np


def _safe_1d(x):
    a = np.asarray(x, float).ravel()
    if a.size == 0:
        return np.zeros(0, dtype=float)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def _smooth(x: np.ndarray, alpha: float = 0.88) -> np.ndarray:
    a = _safe_1d(x)
    if len(a) <= 1:
        return a
    k = float(np.clip(alpha, 0.0, 0.99))
    out = a.copy()
    for t in range(1, len(out)):
        out[t] = k * out[t - 1] + (1.0 - k) * out[t]
    return out


def build_symbolic_governor(
    sym_signal: np.ndarray,
    sym_affect: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    events_n: np.ndarray | None = None,
    sym_regime: np.ndarray | None = None,
    confidence_uncertainty: np.ndarray | None = None,
    affect_uncertainty: np.ndarray | None = None,
    lo: float = 0.72,
    hi: float = 1.12,
    smooth: float = 0.88,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Convert symbolic/affective state into:
      - symbolic stress [0,1]
      - symbolic governor [lo,hi]
    """
    s = _safe_1d(sym_signal)
    T = len(s)
    if T == 0:
        return np.zeros(0, float), np.zeros(0, float), {"status": "empty"}

    a = _safe_1d(sym_affect) if sym_affect is not None else np.zeros(T, float)
    c = _safe_1d(confidence) if confidence is not None else np.ones(T, float) * 0.5
    n = _safe_1d(events_n) if events_n is not None else np.zeros(T, float)
    r = _safe_1d(sym_regime) if sym_regime is not None else np.zeros(T, float)
    cu = _safe_1d(confidence_uncertainty) if confidence_uncertainty is not None else np.zeros(T, float)
    au = _safe_1d(affect_uncertainty) if affect_uncertainty is not None else np.zeros(T, float)
    if len(a) < T:
        x = np.zeros(T, float)
        x[: len(a)] = a
        a = x
    else:
        a = a[:T]
    if len(c) < T:
        x = np.zeros(T, float)
        x[: len(c)] = c
        c = x
    else:
        c = c[:T]
    if len(n) < T:
        x = np.zeros(T, float)
        x[: len(n)] = n
        n = x
    else:
        n = n[:T]
    if len(r) < T:
        x = np.zeros(T, float)
        x[: len(r)] = r
        r = x
    else:
        r = r[:T]
    if len(cu) < T:
        x = np.zeros(T, float)
        x[: len(cu)] = cu
        cu = x
    else:
        cu = cu[:T]
    if len(au) < T:
        x = np.zeros(T, float)
        x[: len(au)] = au
        au = x
    else:
        au = au[:T]

    c = np.clip(c, 0.0, 1.0)
    cu = np.clip(cu, 0.0, 1.0)
    au = np.clip(au, 0.0, 1.0)
    neg_bias = np.clip(-s, 0.0, 1.0)
    affect = np.clip(a, 0.0, 1.0)
    regime = np.clip(r, 0.0, 1.0)
    # Event intensity anomaly proxy.
    den = float(np.percentile(n, 90)) + 1e-9 if len(n) else 1.0
    inten = np.clip(n / den, 0.0, 2.0)
    inten = np.clip(inten / 1.25, 0.0, 1.0)
    burst = np.clip(np.r_[0.0, np.diff(inten)], 0.0, 1.0)
    uncertainty = np.clip(0.65 * cu + 0.35 * au, 0.0, 1.0)
    trust = np.clip(c * (1.0 - 0.65 * uncertainty), 0.0, 1.0)

    directional = np.clip(0.38 * neg_bias + 0.26 * affect + 0.16 * inten + 0.10 * regime + 0.10 * burst, 0.0, 1.0)
    stress = np.clip(directional * (0.45 + 0.55 * trust) + 0.22 * uncertainty, 0.0, 1.0)
    stress = _smooth(stress, alpha=smooth)

    lo_f = float(min(lo, hi))
    hi_f = float(max(lo, hi))
    gov = hi_f - (hi_f - lo_f) * stress
    gov = np.clip(_smooth(gov, alpha=smooth), lo_f, hi_f)

    info = {
        "status": "ok",
        "length": int(T),
        "mean_stress": float(np.mean(stress)),
        "max_stress": float(np.max(stress)),
        "mean_governor": float(np.mean(gov)),
        "min_governor": float(np.min(gov)),
        "max_governor": float(np.max(gov)),
        "mean_trust": float(np.mean(trust)),
        "mean_uncertainty": float(np.mean(uncertainty)),
        "mean_regime": float(np.mean(regime)),
    }
    return stress, gov, info

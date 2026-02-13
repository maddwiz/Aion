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


def build_dna_stress_governor(
    drift: np.ndarray,
    velocity: np.ndarray | None = None,
    drift_z: np.ndarray | None = None,
    regime_state: np.ndarray | None = None,
    lo: float = 0.72,
    hi: float = 1.12,
    smooth: float = 0.88,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Convert DNA drift diagnostics into:
      - stress score in [0,1]
      - exposure governor in [lo,hi]
    """
    d = _safe_1d(drift)
    T = len(d)
    if T == 0:
        return np.zeros(0, float), np.zeros(0, float), {"status": "empty"}

    v = _safe_1d(velocity) if velocity is not None else np.zeros(T, float)
    z = _safe_1d(drift_z) if drift_z is not None else np.zeros(T, float)
    s = _safe_1d(regime_state) if regime_state is not None else np.zeros(T, float)
    if len(v) < T:
        vv = np.zeros(T, float)
        vv[: len(v)] = v
        v = vv
    else:
        v = v[:T]
    if len(z) < T:
        zz = np.zeros(T, float)
        zz[: len(z)] = z
        z = zz
    else:
        z = z[:T]
    if len(s) < T:
        ss = np.zeros(T, float)
        ss[: len(s)] = s
        s = ss
    else:
        s = s[:T]

    # Drift level percentile.
    den_d = float(np.percentile(np.abs(d), 90)) + 1e-9
    lvl = np.clip(np.abs(d) / den_d, 0.0, 2.0)
    lvl = np.clip(lvl / 1.25, 0.0, 1.0)

    # Positive velocity means drift accelerating.
    vp = np.clip(v, 0.0, None)
    den_v = float(np.percentile(vp, 90)) + 1e-9
    vel = np.clip(vp / den_v, 0.0, 2.0)
    vel = np.clip(vel / 1.25, 0.0, 1.0)

    # Z-normalized drift stress.
    zz = np.clip((z + 2.0) / 4.0, 0.0, 1.0)

    # Regime stress: -1 calm, 0 neutral, +1 stressed.
    rs = np.clip((s + 1.0) / 2.0, 0.0, 1.0)

    stress = np.clip(0.40 * lvl + 0.25 * vel + 0.20 * zz + 0.15 * rs, 0.0, 1.0)
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
    }
    return stress, gov, info

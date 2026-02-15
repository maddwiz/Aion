#!/usr/bin/env python3
# Regime Fracture Engine (novel early-instability detector).
#
# Reads (best-effort):
#   runs_plus/meta_mix_disagreement.csv
#   runs_plus/meta_mix_quality.csv
#   runs_plus/shock_mask.csv
#   runs_plus/daily_returns.csv
#   runs_plus/cross_hive_weights.csv
#   runs_plus/heartbeat_stress.csv
#   runs_plus/hive_persistence_governor.csv
#
# Writes:
#   runs_plus/regime_fracture_signal.csv
#   runs_plus/regime_fracture_governor.csv
#   runs_plus/regime_fracture_info.json

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)

from qmods.regime_fracture import (  # noqa: E402
    breadth_stress_from_weights,
    fracture_governor,
    realized_vol_convexity,
    rolling_percentile_stress,
    smooth_ema,
)


def _load_series(path: Path):
    if not path.exists() or path.is_dir():
        return None
    try:
        a = np.loadtxt(path, delimiter=",")
    except Exception:
        try:
            a = np.loadtxt(path, delimiter=",", skiprows=1)
        except Exception:
            return None
    a = np.asarray(a, float)
    if a.ndim == 2 and a.shape[1] >= 1:
        a = a[:, -1]
    a = np.asarray(a, float).ravel()
    if len(a) == 0:
        return None
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def _load_cross_hive_weights(path: Path):
    if not path.exists() or path.is_dir():
        return None, None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None, None
    if df.empty:
        return None, None
    dates = None
    if "DATE" in df.columns:
        dates = pd.to_datetime(df["DATE"], errors="coerce")
    cols = [c for c in df.columns if str(c).upper() not in {"DATE"} and not str(c).startswith("arb_")]
    if not cols:
        return None, dates
    mat = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(float)
    return mat, dates


def _infer_target_len() -> int:
    candidates = [
        RUNS / "portfolio_weights_final.csv",
        RUNS / "weights_regime.csv",
        RUNS / "weights_tail_blend.csv",
        RUNS / "portfolio_weights.csv",
        RUNS / "cross_hive_weights.csv",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            a = np.loadtxt(p, delimiter=",")
        except Exception:
            try:
                a = np.loadtxt(p, delimiter=",", skiprows=1)
            except Exception:
                continue
        a = np.asarray(a, float)
        if a.ndim == 1:
            return int(len(a))
        if a.ndim == 2:
            return int(a.shape[0])
    return 0


def _tail_align(x, n: int, fill: float = 0.0):
    if n <= 0:
        return np.asarray([], float)
    if x is None:
        return np.full(n, float(fill), dtype=float)
    a = np.asarray(x, float).ravel()
    if len(a) >= n:
        return a[-n:]
    out = np.full(n, float(fill), dtype=float)
    out[-len(a) :] = a
    return out


def _derive_disagreement(n: int):
    d = _load_series(RUNS / "meta_mix_disagreement.csv")
    if d is not None:
        return _tail_align(d, n, fill=0.0)
    m = _load_series(RUNS / "meta_stack_pred.csv")
    s = _load_series(RUNS / "synapses_pred.csv")
    if m is None or s is None:
        return np.zeros(n, float)
    L = min(len(m), len(s))
    if L <= 0:
        return np.zeros(n, float)
    d = np.abs(np.asarray(m[-L:], float) - np.asarray(s[-L:], float))
    p90 = float(np.percentile(d, 90)) if len(d) else 0.0
    d = np.clip(d / max(1e-9, p90), 0.0, 1.0)
    return _tail_align(d, n, fill=0.0)


def main(root: Path | None = None, runs: Path | None = None) -> int:
    global ROOT, RUNS
    if root is not None:
        ROOT = Path(root)
    if runs is not None:
        RUNS = Path(runs)
    else:
        RUNS = ROOT / "runs_plus"
    RUNS.mkdir(parents=True, exist_ok=True)

    T = int(_infer_target_len())
    if T <= 0:
        raise SystemExit("Need base weights/cross-hive artifacts before running regime fracture engine.")

    w_cross, d_cross = _load_cross_hive_weights(RUNS / "cross_hive_weights.csv")

    disagreement = _derive_disagreement(T)
    quality = _tail_align(_load_series(RUNS / "meta_mix_quality.csv"), T, fill=0.5)
    shock = _tail_align(_load_series(RUNS / "shock_mask.csv"), T, fill=0.0)
    ret = _load_series(RUNS / "daily_returns.csv")
    if ret is None:
        ret = _load_series(RUNS / "wf_oos_returns.csv")
    ret = _tail_align(ret, T, fill=0.0)
    hb_stress = _tail_align(_load_series(RUNS / "heartbeat_stress.csv"), T, fill=0.0)
    hive_pg = _tail_align(_load_series(RUNS / "hive_persistence_governor.csv"), T, fill=1.0)

    d_stress = rolling_percentile_stress(disagreement, window=63, min_periods=12)
    contradiction = np.clip(d_stress * np.clip((quality - 0.55) / 0.35, 0.0, 1.0), 0.0, 1.0)
    v_convex = realized_vol_convexity(ret, short_w=10, long_w=63)
    b_stress = (
        _tail_align(breadth_stress_from_weights(w_cross), T, fill=0.0)
        if w_cross is not None
        else np.zeros(T, float)
    )
    hb_component = np.clip(hb_stress, 0.0, 1.0)
    persistence_component = np.clip((1.02 - hive_pg) / 0.24, 0.0, 1.0)
    shock_component = np.clip(shock, 0.0, 1.0)

    raw = (
        0.27 * d_stress
        + 0.18 * contradiction
        + 0.20 * v_convex
        + 0.15 * b_stress
        + 0.10 * hb_component
        + 0.05 * persistence_component
        + 0.05 * shock_component
    )
    raw = np.clip(raw, 0.0, 1.0)
    inertia = float(np.clip(float(os.getenv("Q_FRACTURE_SMOOTH_INERTIA", "0.85")), 0.0, 0.99))
    score = np.clip(smooth_ema(raw, inertia=inertia), 0.0, 1.0)

    alpha = float(np.clip(float(os.getenv("Q_FRACTURE_ALPHA", "0.32")), 0.0, 1.50))
    gmin = float(np.clip(float(os.getenv("Q_FRACTURE_MIN_GOV", "0.72")), 0.10, 1.10))
    gmax = float(np.clip(float(os.getenv("Q_FRACTURE_MAX_GOV", "1.04")), 0.20, 1.20))
    gov = fracture_governor(score, alpha=alpha, min_gov=gmin, max_gov=gmax)

    latest = float(score[-1]) if len(score) else 0.0
    if latest >= 0.85:
        state = "fracture_alert"
    elif latest >= 0.72:
        state = "fracture_warn"
    elif latest >= 0.55:
        state = "watch"
    else:
        state = "calm"

    risk_flags = []
    if state == "fracture_warn":
        risk_flags.append("fracture_warn")
    elif state == "fracture_alert":
        risk_flags.append("fracture_alert")

    if d_cross is not None and len(d_cross) >= T:
        dates = pd.to_datetime(d_cross, errors="coerce")
        dates = dates[-T:]
    else:
        dates = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=T, freq="D")
    dates_s = [pd.Timestamp(x).strftime("%Y-%m-%d") for x in dates]

    sig = pd.DataFrame(
        {
            "DATE": dates_s,
            "regime_fracture_score": score,
            "disagreement_stress": d_stress,
            "contradiction_stress": contradiction,
            "volatility_convexity": v_convex,
            "breadth_stress": b_stress,
            "heartbeat_stress": hb_component,
            "persistence_stress": persistence_component,
            "shock_stress": shock_component,
        }
    )
    sig.to_csv(RUNS / "regime_fracture_signal.csv", index=False)

    gov_df = pd.DataFrame({"DATE": dates_s, "regime_fracture_governor": gov})
    gov_df.to_csv(RUNS / "regime_fracture_governor.csv", index=False)

    info = {
        "ok": True,
        "state": state,
        "risk_flags": risk_flags,
        "latest_score": latest,
        "latest_governor": float(gov[-1]) if len(gov) else 1.0,
        "mean_score": float(np.mean(score)) if len(score) else 0.0,
        "max_score": float(np.max(score)) if len(score) else 0.0,
        "parameters": {
            "fracture_alpha": alpha,
            "fracture_min_governor": gmin,
            "fracture_max_governor": gmax,
            "smooth_inertia": inertia,
        },
        "components_latest": {
            "disagreement_stress": float(d_stress[-1]) if len(d_stress) else 0.0,
            "contradiction_stress": float(contradiction[-1]) if len(contradiction) else 0.0,
            "volatility_convexity": float(v_convex[-1]) if len(v_convex) else 0.0,
            "breadth_stress": float(b_stress[-1]) if len(b_stress) else 0.0,
            "heartbeat_stress": float(hb_component[-1]) if len(hb_component) else 0.0,
            "persistence_stress": float(persistence_component[-1]) if len(persistence_component) else 0.0,
            "shock_stress": float(shock_component[-1]) if len(shock_component) else 0.0,
        },
    }
    (RUNS / "regime_fracture_info.json").write_text(json.dumps(info, indent=2))

    print(f"✅ Wrote {RUNS/'regime_fracture_signal.csv'}")
    print(f"✅ Wrote {RUNS/'regime_fracture_governor.csv'}")
    print(f"✅ Wrote {RUNS/'regime_fracture_info.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

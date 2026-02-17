#!/usr/bin/env python3
# Builds runs_plus/daily_returns.csv by multiplying weights × asset_returns
# Chooses best available weights automatically.

import numpy as np
from pathlib import Path
import os
import json

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"; RUNS.mkdir(exist_ok=True)

def load_mat(rel):
    p = ROOT/rel
    if not p.exists(): return None
    try:
        a = np.loadtxt(p, delimiter=",")
    except:
        a = np.loadtxt(p, delimiter=",", skiprows=1)
    if a.ndim == 1: a = a.reshape(-1,1)
    return a

def first_mat(paths):
    for rel in paths:
        a = load_mat(rel)
        if a is not None: return a, rel
    return None, None


def build_costed_daily_returns(
    W: np.ndarray,
    A: np.ndarray,
    *,
    base_bps: float,
    vol_scaled_bps: float = 0.0,
    vol_lookback: int = 20,
    vol_ref_daily: float = 0.0063,
    half_turnover: bool = True,
    fixed_daily_fee: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    w = np.asarray(W, float)
    a = np.asarray(A, float)
    T = min(w.shape[0], a.shape[0])
    w = w[:T]
    a = a[:T]
    gross = np.sum(w * a, axis=1)
    turnover = np.zeros(T, dtype=float)
    if T > 1:
        turnover[1:] = np.sum(np.abs(np.diff(w, axis=0)), axis=1)
    if bool(half_turnover):
        turnover = 0.5 * turnover

    # Rolling realized volatility (daily) for volatility-scaled slippage.
    vlook = int(np.clip(int(vol_lookback), 2, max(2, T)))
    vol = np.zeros(T, dtype=float)
    for t in range(T):
        lo = max(0, t - vlook + 1)
        seg = gross[lo : t + 1]
        vol[t] = float(np.std(seg)) if seg.size else 0.0
    ref = float(max(1e-6, vol_ref_daily))
    vol_ratio = np.clip(vol / ref, 0.0, 6.0)
    eff_bps = np.clip(float(base_bps), 0.0, 10_000.0) + np.clip(float(vol_scaled_bps), 0.0, 10_000.0) * vol_ratio

    var_cost = (eff_bps / 10000.0) * turnover
    fee = np.full(T, float(max(0.0, fixed_daily_fee)), dtype=float)
    cost = var_cost + fee
    net = gross - cost
    return net, gross, cost, turnover, eff_bps

if __name__ == "__main__":
    A = load_mat("runs_plus/asset_returns.csv")
    if A is None:
        print("(!) runs_plus/asset_returns.csv missing. Run tools/rebuild_asset_matrix.py first.")
        raise SystemExit(0)
    clip_abs = float(np.clip(float(os.getenv("Q_ASSET_RET_CLIP", "0.35")), 0.01, 5.0))
    A_clean = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A_clean = np.clip(A_clean, -0.95, clip_abs)
    clip_events = int(np.sum(np.abs(A_clean - A) > 1e-12))
    A = A_clean

    W, src = first_mat([
        "runs_plus/portfolio_weights_final.csv",
        "runs_plus/tune_best_weights.csv",
        "runs_plus/weights_regime.csv",
        "runs_plus/weights_tail_blend.csv",
        "runs_plus/portfolio_weights.csv",
        "portfolio_weights.csv",
    ])
    if W is None:
        print("(!) No weights found."); raise SystemExit(0)

    # Align T and N
    T = min(A.shape[0], W.shape[0])
    if A.shape[1] != W.shape[1]:
        print(f"(!) Col mismatch: asset_returns N={A.shape[1]} vs weights N={W.shape[1]}.")
        raise SystemExit(0)

    # Backward-compatible legacy knob.
    legacy_bps = str(os.getenv("Q_COST_BPS", "")).strip()
    if legacy_bps and (not str(os.getenv("Q_COST_BASE_BPS", "")).strip()):
        base_bps = float(np.clip(float(legacy_bps), 0.0, 100.0))
    else:
        base_bps = float(np.clip(float(os.getenv("Q_COST_BASE_BPS", "10.0")), 0.0, 100.0))
    vol_scaled_bps = float(np.clip(float(os.getenv("Q_COST_VOL_SCALED_BPS", "0.0")), 0.0, 100.0))
    vol_lookback = int(np.clip(int(float(os.getenv("Q_COST_VOL_LOOKBACK", "20"))), 2, 252))
    vol_ref_daily = float(np.clip(float(os.getenv("Q_COST_VOL_REF_DAILY", "0.0063")), 1e-5, 0.25))
    half_turnover = str(os.getenv("Q_COST_HALF_TURNOVER", "1")).strip().lower() in {"1", "true", "yes", "on"}
    fixed_daily_fee = float(np.clip(float(os.getenv("Q_FIXED_DAILY_FEE", "0.0")), 0.0, 1.0))
    net, gross, cost, turnover, eff_bps = build_costed_daily_returns(
        W[:T],
        A[:T],
        base_bps=base_bps,
        vol_scaled_bps=vol_scaled_bps,
        vol_lookback=vol_lookback,
        vol_ref_daily=vol_ref_daily,
        half_turnover=half_turnover,
        fixed_daily_fee=fixed_daily_fee,
    )
    np.savetxt(RUNS/"daily_returns.csv", net, delimiter=",")
    np.savetxt(RUNS/"daily_returns_gross.csv", gross, delimiter=",")
    np.savetxt(RUNS/"daily_costs.csv", cost, delimiter=",")
    np.savetxt(RUNS/"daily_turnover.csv", turnover, delimiter=",")
    np.savetxt(RUNS/"daily_effective_cost_bps.csv", eff_bps, delimiter=",")
    (RUNS / "daily_costs_info.json").write_text(
        json.dumps(
            {
                "cost_base_bps": float(base_bps),
                "cost_vol_scaled_bps": float(vol_scaled_bps),
                "cost_vol_lookback": int(vol_lookback),
                "cost_vol_ref_daily": float(vol_ref_daily),
                "cost_half_turnover": bool(half_turnover),
                "fixed_daily_fee": float(fixed_daily_fee),
                "rows": int(T),
                "mean_cost_daily": float(np.mean(cost)) if T else 0.0,
                "ann_cost_estimate": float(np.mean(cost) * 252.0) if T else 0.0,
                "mean_gross_daily": float(np.mean(gross)) if T else 0.0,
                "mean_net_daily": float(np.mean(net)) if T else 0.0,
                "mean_turnover": float(np.mean(turnover)) if T else 0.0,
                "mean_effective_cost_bps": float(np.mean(eff_bps)) if T else 0.0,
                "max_effective_cost_bps": float(np.max(eff_bps)) if T else 0.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if clip_events > 0:
        print(f"(!) Clipped {clip_events} extreme asset-return values with |r|>{clip_abs:.3f}")
    print(
        f"✅ Wrote runs_plus/daily_returns.csv (T={T}) from weights='{src}' "
        f"[base_bps={base_bps:.2f}, vol_scaled_bps={vol_scaled_bps:.2f}, fixed_daily_fee={fixed_daily_fee:.5f}]"
    )

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
    cost_bps: float,
    fixed_daily_fee: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = np.asarray(W, float)
    a = np.asarray(A, float)
    T = min(w.shape[0], a.shape[0])
    w = w[:T]
    a = a[:T]
    gross = np.sum(w * a, axis=1)
    turnover = np.zeros(T, dtype=float)
    if T > 1:
        turnover[1:] = np.sum(np.abs(np.diff(w, axis=0)), axis=1)
    var_cost = np.clip(float(cost_bps), 0.0, 10_000.0) / 10000.0 * turnover
    fee = np.full(T, float(max(0.0, fixed_daily_fee)), dtype=float)
    cost = var_cost + fee
    net = gross - cost
    return net, gross, cost

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

    cost_bps = float(np.clip(float(os.getenv("Q_COST_BPS", "10.0")), 0.0, 100.0))
    fixed_daily_fee = float(np.clip(float(os.getenv("Q_FIXED_DAILY_FEE", "0.0")), 0.0, 1.0))
    net, gross, cost = build_costed_daily_returns(
        W[:T],
        A[:T],
        cost_bps=cost_bps,
        fixed_daily_fee=fixed_daily_fee,
    )
    np.savetxt(RUNS/"daily_returns.csv", net, delimiter=",")
    np.savetxt(RUNS/"daily_returns_gross.csv", gross, delimiter=",")
    np.savetxt(RUNS/"daily_costs.csv", cost, delimiter=",")
    (RUNS / "daily_costs_info.json").write_text(
        json.dumps(
            {
                "cost_bps": float(cost_bps),
                "fixed_daily_fee": float(fixed_daily_fee),
                "rows": int(T),
                "mean_cost_daily": float(np.mean(cost)) if T else 0.0,
                "ann_cost_estimate": float(np.mean(cost) * 252.0) if T else 0.0,
                "mean_gross_daily": float(np.mean(gross)) if T else 0.0,
                "mean_net_daily": float(np.mean(net)) if T else 0.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if clip_events > 0:
        print(f"(!) Clipped {clip_events} extreme asset-return values with |r|>{clip_abs:.3f}")
    print(
        f"✅ Wrote runs_plus/daily_returns.csv (T={T}) from weights='{src}' "
        f"[cost_bps={cost_bps:.2f}, fixed_daily_fee={fixed_daily_fee:.5f}]"
    )

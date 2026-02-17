#!/usr/bin/env python3
"""
Integrated Q -> AION execution backtest.

Applies AION execution simulator (slippage + partial fills) to Q portfolio
weights so backtests reflect the same execution realism used in AION runtime.

Outputs:
  - runs_plus/daily_returns_aion_integrated.csv
  - runs_plus/q_aion_integrated_backtest.json
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)

AION_ROOT = ROOT.parent / "aion"
if str(AION_ROOT) not in sys.path:
    sys.path.insert(0, str(AION_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aion.execution.simulator import ExecutionSimulator  # noqa: E402
from tools import run_strict_oos_validation as so  # noqa: E402


@dataclass
class _ExecCfg:
    SPREAD_BPS_BASE: float
    SPREAD_BPS_VOL_MULT: float
    EXEC_QUEUE_IMPACT_BPS: float
    EXEC_LATENCY_MS: int
    SLIPPAGE_BPS: float
    EXEC_PARTIAL_FILL_MIN: float
    EXEC_PARTIAL_FILL_MAX: float


def _load_mat(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        arr = np.loadtxt(path, delimiter=",")
    except Exception:
        try:
            arr = np.loadtxt(path, delimiter=",", skiprows=1)
        except Exception:
            return None
    arr = np.asarray(arr, float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _first(paths: list[Path]) -> tuple[np.ndarray | None, str]:
    for p in paths:
        arr = _load_mat(p)
        if arr is not None:
            return arr, str(p.relative_to(ROOT))
    return None, ""


def _build_price_matrix(returns_mat: np.ndarray, start_price: float = 100.0) -> np.ndarray:
    t, n = returns_mat.shape
    p = np.full((t, n), float(start_price), dtype=float)
    for i in range(1, t):
        p[i] = p[i - 1] * (1.0 + returns_mat[i])
    return np.clip(p, 1e-6, np.inf)


def _robust_metrics(r: np.ndarray) -> tuple[dict, dict]:
    v = np.asarray(r, float).reshape(-1)
    train_frac = float(np.clip(float(os.getenv("Q_STRICT_OOS_TRAIN_FRAC", "0.75")), 0.50, 0.95))
    min_train = int(np.clip(int(float(os.getenv("Q_STRICT_OOS_MIN_TRAIN", "756"))), 100, 100000))
    min_test = int(np.clip(int(float(os.getenv("Q_STRICT_OOS_MIN_TEST", "252"))), 50, 100000))
    robust_splits = int(np.clip(int(float(os.getenv("Q_STRICT_OOS_ROBUST_SPLITS", "5"))), 1, 16))
    split = so._build_split_index(len(v), train_frac, min_train, min_test)
    net = so._metrics(v[split:])
    split_ix = so._robust_splits(len(v), min_train=min_train, min_test=min_test, n_splits=robust_splits)
    agg = so._aggregate_robust([so._metrics(v[s:]) for s in split_ix])
    return net, agg


def main() -> int:
    returns_mat, ret_src = _first([RUNS / "asset_returns.csv"])
    weights, w_src = _first(
        [
            RUNS / "portfolio_weights_final.csv",
            RUNS / "tune_best_weights.csv",
            RUNS / "weights_regime.csv",
            RUNS / "weights_tail_blend.csv",
            RUNS / "portfolio_weights.csv",
            ROOT / "portfolio_weights.csv",
        ]
    )
    if returns_mat is None or weights is None:
        out = {
            "ok": False,
            "reason": "missing_inputs",
            "returns_found": bool(returns_mat is not None),
            "weights_found": bool(weights is not None),
        }
        (RUNS / "q_aion_integrated_backtest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"✅ Wrote {RUNS/'q_aion_integrated_backtest.json'}")
        print("(!) Integrated backtest skipped: missing inputs")
        return 0

    if returns_mat.shape[1] != weights.shape[1]:
        out = {
            "ok": False,
            "reason": "shape_mismatch",
            "returns_shape": list(returns_mat.shape),
            "weights_shape": list(weights.shape),
        }
        (RUNS / "q_aion_integrated_backtest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"✅ Wrote {RUNS/'q_aion_integrated_backtest.json'}")
        print("(!) Integrated backtest skipped: shape mismatch")
        return 0

    t = int(min(returns_mat.shape[0], weights.shape[0]))
    returns_mat = returns_mat[:t]
    weights = weights[:t]

    exec_cfg = _ExecCfg(
        SPREAD_BPS_BASE=float(np.clip(float(os.getenv("AION_SPREAD_BPS_BASE", "2.5")), 0.0, 100.0)),
        SPREAD_BPS_VOL_MULT=float(np.clip(float(os.getenv("AION_SPREAD_BPS_VOL_MULT", "18.0")), 0.0, 500.0)),
        EXEC_QUEUE_IMPACT_BPS=float(np.clip(float(os.getenv("AION_EXEC_QUEUE_IMPACT_BPS", "3.0")), 0.0, 100.0)),
        EXEC_LATENCY_MS=int(np.clip(int(float(os.getenv("AION_EXEC_LATENCY_MS", "250"))), 0, 10000)),
        SLIPPAGE_BPS=float(np.clip(float(os.getenv("AION_SLIPPAGE_BPS", "5")), 0.0, 100.0)),
        EXEC_PARTIAL_FILL_MIN=float(np.clip(float(os.getenv("AION_EXEC_PARTIAL_FILL_MIN", "0.35")), 0.01, 1.0)),
        EXEC_PARTIAL_FILL_MAX=float(np.clip(float(os.getenv("AION_EXEC_PARTIAL_FILL_MAX", "1.00")), 0.01, 1.0)),
    )
    exe = ExecutionSimulator(exec_cfg)

    prices = _build_price_matrix(returns_mat, start_price=float(os.getenv("Q_AION_INTEGRATED_START_PRICE", "100.0")))
    atr_proxy = np.clip(np.vstack([np.abs(returns_mat[0]), np.abs(returns_mat)]).mean(axis=0), 1e-4, 0.25)
    start_equity = float(np.clip(float(os.getenv("Q_AION_INTEGRATED_START_EQUITY", "1000000")), 1000.0, 1e12))

    n = int(returns_mat.shape[1])
    qty = np.zeros(n, dtype=int)
    cash = float(start_equity)
    prev_value = float(start_equity)
    out_r = np.zeros(t, dtype=float)
    fill_ratios = []
    slippages = []

    for i in range(t):
        px = prices[i]
        port_value = float(cash + np.sum(qty * px))
        if not np.isfinite(port_value) or port_value <= 0:
            port_value = float(max(1.0, prev_value))
        target_notional = weights[i] * port_value
        desired_qty = np.round(target_notional / np.clip(px, 1e-6, np.inf)).astype(int)
        delta = desired_qty - qty
        for j in range(n):
            d = int(delta[j])
            if d == 0:
                continue
            side = "BUY" if d > 0 else "SELL"
            conf = float(np.clip(abs(weights[i, j]) * 2.5, 0.20, 1.0))
            fill = exe.execute(side=side, qty=abs(d), ref_price=float(px[j]), atr_pct=float(atr_proxy[j]), confidence=conf, allow_partial=True)
            if fill.filled_qty <= 0:
                continue
            q_exec = int(fill.filled_qty if d > 0 else -fill.filled_qty)
            qty[j] += q_exec
            if q_exec > 0:
                cash -= float(fill.avg_fill) * float(q_exec)
            else:
                cash += float(fill.avg_fill) * float(-q_exec)
            fill_ratios.append(float(fill.fill_ratio))
            slippages.append(float(fill.est_slippage_bps))

        new_value = float(cash + np.sum(qty * px))
        out_r[i] = float((new_value / (prev_value + 1e-12)) - 1.0)
        prev_value = float(new_value)

    np.savetxt(RUNS / "daily_returns_aion_integrated.csv", out_r, delimiter=",")
    net, robust = _robust_metrics(out_r)
    baseline = _load_mat(RUNS / "daily_returns.csv")
    baseline_metrics = so._metrics(baseline.reshape(-1)) if baseline is not None else {}

    out = {
        "ok": True,
        "returns_source": ret_src,
        "weights_source": w_src,
        "rows": int(t),
        "assets": int(n),
        "execution_model": {
            "spread_bps_base": float(exec_cfg.SPREAD_BPS_BASE),
            "spread_bps_vol_mult": float(exec_cfg.SPREAD_BPS_VOL_MULT),
            "queue_impact_bps": float(exec_cfg.EXEC_QUEUE_IMPACT_BPS),
            "latency_ms": int(exec_cfg.EXEC_LATENCY_MS),
            "slippage_bps_base": float(exec_cfg.SLIPPAGE_BPS),
            "partial_fill_min": float(exec_cfg.EXEC_PARTIAL_FILL_MIN),
            "partial_fill_max": float(exec_cfg.EXEC_PARTIAL_FILL_MAX),
        },
        "fill_stats": {
            "fills": int(len(fill_ratios)),
            "avg_fill_ratio": float(np.mean(fill_ratios)) if fill_ratios else 0.0,
            "avg_slippage_bps": float(np.mean(slippages)) if slippages else 0.0,
            "p90_slippage_bps": float(np.percentile(slippages, 90)) if slippages else 0.0,
        },
        "metrics_oos_net": net,
        "metrics_oos_robust": robust,
        "baseline_daily_returns_metrics": baseline_metrics,
        "daily_returns_file": str((RUNS / "daily_returns_aion_integrated.csv").relative_to(ROOT)),
    }
    (RUNS / "q_aion_integrated_backtest.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"✅ Wrote {RUNS/'daily_returns_aion_integrated.csv'}")
    print(f"✅ Wrote {RUNS/'q_aion_integrated_backtest.json'}")
    print(
        "Integrated OOS:",
        f"Sharpe={float(net.get('sharpe', 0.0)):.3f}",
        f"Hit={float(net.get('hit_rate', 0.0)):.3f}",
        f"MaxDD={float(net.get('max_drawdown', 0.0)):.3f}",
        f"N={int(net.get('n', 0))}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

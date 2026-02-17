#!/usr/bin/env python3
"""
Strict OOS validation on costed returns.

Reads:
  - runs_plus/daily_returns.csv (net)
  - runs_plus/daily_returns_gross.csv (optional)
  - runs_plus/daily_costs.csv (optional)

Writes:
  - runs_plus/wf_oos_returns.csv
  - runs_plus/strict_oos_validation.json
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def _load_series(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        x = np.loadtxt(path, delimiter=",")
    except Exception:
        try:
            x = np.loadtxt(path, delimiter=",", skiprows=1)
        except Exception:
            return None
    x = np.asarray(x, float).ravel()
    if x.size == 0:
        return None
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _metrics(r: np.ndarray) -> dict:
    v = np.asarray(r, float).ravel()
    if v.size == 0:
        return {
            "n": 0,
            "mean_daily": 0.0,
            "vol_daily": 0.0,
            "sharpe": 0.0,
            "hit_rate": 0.0,
            "max_drawdown": 0.0,
            "ann_return_arith": 0.0,
            "ann_vol": 0.0,
        }
    mu = float(np.mean(v))
    sd = float(np.std(v) + 1e-12)
    sh = float((mu / sd) * math.sqrt(252.0))
    hit = float(np.mean(v > 0.0))
    eq = np.cumsum(v)
    peak = np.maximum.accumulate(eq)
    mdd = float(np.min(eq - peak))
    return {
        "n": int(v.size),
        "mean_daily": mu,
        "vol_daily": sd,
        "sharpe": sh,
        "hit_rate": hit,
        "max_drawdown": mdd,
        "ann_return_arith": float(mu * 252.0),
        "ann_vol": float(sd * math.sqrt(252.0)),
    }


def _append_card(title: str, html: str) -> None:
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html", "report.html"]:
        p = ROOT / name
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        txt = txt.replace("</body>", card + "</body>") if "</body>" in txt else txt + card
        p.write_text(txt, encoding="utf-8")


def main() -> int:
    r = _load_series(RUNS / "daily_returns.csv")
    if r is None:
        print("(!) Missing runs_plus/daily_returns.csv; run make_daily_from_weights.py first.")
        return 0

    T = len(r)
    train_frac = float(np.clip(float(os.getenv("Q_STRICT_OOS_TRAIN_FRAC", "0.75")), 0.50, 0.95))
    min_train = int(np.clip(int(float(os.getenv("Q_STRICT_OOS_MIN_TRAIN", "756"))), 100, 100000))
    min_test = int(np.clip(int(float(os.getenv("Q_STRICT_OOS_MIN_TEST", "252"))), 50, 100000))

    split = max(min_train, int(T * train_frac))
    if (T - split) < min_test:
        split = max(min_train, T - min_test)
    split = int(np.clip(split, 1, max(1, T - 1)))

    train = r[:split]
    oos = r[split:]
    np.savetxt(RUNS / "wf_oos_returns.csv", oos, delimiter=",")

    gross = _load_series(RUNS / "daily_returns_gross.csv")
    costs = _load_series(RUNS / "daily_costs.csv")
    cost_info = {}
    if gross is not None and len(gross) >= T:
        gv = gross[:T]
        cost_info["gross_full"] = _metrics(gv)
    if costs is not None and len(costs) >= T:
        cv = costs[:T]
        cost_info["cost"] = {
            "mean_daily": float(np.mean(cv)),
            "ann_cost_estimate": float(np.mean(cv) * 252.0),
            "sum": float(np.sum(cv)),
        }

    out = {
        "method": "strict_holdout_costed",
        "source": "runs_plus/daily_returns.csv",
        "rows_total": int(T),
        "split_index": int(split),
        "train_frac_requested": float(train_frac),
        "train_rows": int(len(train)),
        "oos_rows": int(len(oos)),
        "metrics_full_net": _metrics(r),
        "metrics_train_net": _metrics(train),
        "metrics_oos_net": _metrics(oos),
        "cost_context": cost_info,
    }
    (RUNS / "strict_oos_validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    m = out["metrics_oos_net"]
    html = (
        f"<p>Strict OOS net validation: rows={out['oos_rows']}, "
        f"Sharpe={m['sharpe']:.3f}, Hit={m['hit_rate']:.3f}, MaxDD={m['max_drawdown']:.3f}.</p>"
        f"<p>Split: train={out['train_rows']} / oos={out['oos_rows']}.</p>"
    )
    _append_card("Strict OOS Validation ✔", html)

    print(f"✅ Wrote {RUNS/'wf_oos_returns.csv'}")
    print(f"✅ Wrote {RUNS/'strict_oos_validation.json'}")
    print(
        f"OOS net: Sharpe={m['sharpe']:.3f} Hit={m['hit_rate']:.3f} MaxDD={m['max_drawdown']:.3f} N={m['n']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

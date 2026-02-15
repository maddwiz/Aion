#!/usr/bin/env python3
# Portfolio drift watchdog between runs.
#
# Reads:
#   runs_plus/portfolio_weights_final.csv
#   runs_plus/portfolio_weights_prev.csv (if exists)
# Writes:
#   runs_plus/portfolio_drift_watch.json
#   runs_plus/portfolio_weights_prev.csv (updated snapshot)

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmods.drift_watch import compute_weight_drift  # noqa: E402

RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def _load_mat(path: Path):
    if not path.exists():
        return None
    try:
        a = np.loadtxt(path, delimiter=",")
    except Exception:
        try:
            a = np.loadtxt(path, delimiter=",", skiprows=1)
        except Exception:
            return None
    a = np.asarray(a, float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a


def _classify_drift(drift: dict, max_l1_alert: float, ratio_alert: float, ratio_warn: float):
    d = dict(drift or {})
    latest = float(d.get("latest_l1", 0.0) or 0.0)
    p95 = float(d.get("p95_l1", 0.0) or 0.0)
    ratio = float(latest / (p95 + 1e-9)) if p95 > 0.0 else 0.0
    d["latest_over_p95"] = ratio

    if latest > float(max_l1_alert) or ratio > float(ratio_alert):
        d["status"] = "alert"
    elif ratio > float(ratio_warn):
        d["status"] = "warn"
    else:
        d["status"] = "ok"
    return d


if __name__ == "__main__":
    cur_path = RUNS / "portfolio_weights_final.csv"
    prev_path = RUNS / "portfolio_weights_prev.csv"
    cur = _load_mat(cur_path)
    prev = _load_mat(prev_path)

    if cur is None:
        raise SystemExit("Missing runs_plus/portfolio_weights_final.csv")

    if prev is None:
        drift = {
            "rows_overlap": 0,
            "cols_overlap": 0,
            "latest_l1": 0.0,
            "latest_l2": 0.0,
            "mean_l1": 0.0,
            "p95_l1": 0.0,
            "status": "bootstrap",
        }
    else:
        drift = compute_weight_drift(cur, prev)
        drift = _classify_drift(
            drift,
            max_l1_alert=float(np.clip(float(os.getenv("Q_DRIFT_ALERT_L1", "1.20")), 0.10, 10.0)),
            ratio_alert=float(np.clip(float(os.getenv("Q_DRIFT_ALERT_RATIO", "3.0")), 1.05, 20.0)),
            ratio_warn=float(np.clip(float(os.getenv("Q_DRIFT_WARN_RATIO", "1.8")), 1.01, 10.0)),
        )

    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "current_rows": int(cur.shape[0]),
        "current_cols": int(cur.shape[1]),
        "drift": drift,
        "thresholds": {
            "drift_alert_l1": float(np.clip(float(os.getenv("Q_DRIFT_ALERT_L1", "1.20")), 0.10, 10.0)),
            "drift_alert_ratio": float(np.clip(float(os.getenv("Q_DRIFT_ALERT_RATIO", "3.0")), 1.05, 20.0)),
            "drift_warn_ratio": float(np.clip(float(os.getenv("Q_DRIFT_WARN_RATIO", "1.8")), 1.01, 10.0)),
        },
    }
    (RUNS / "portfolio_drift_watch.json").write_text(json.dumps(out, indent=2))

    # Update baseline for next run.
    np.savetxt(prev_path, cur, delimiter=",")

    print(f"✅ Wrote {RUNS/'portfolio_drift_watch.json'}")
    print(f"✅ Updated {prev_path}")

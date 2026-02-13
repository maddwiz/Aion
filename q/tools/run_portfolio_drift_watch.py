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
        drift["status"] = "ok"

    out = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "current_rows": int(cur.shape[0]),
        "current_cols": int(cur.shape[1]),
        "drift": drift,
    }
    (RUNS / "portfolio_drift_watch.json").write_text(json.dumps(out, indent=2))

    # Update baseline for next run.
    np.savetxt(prev_path, cur, delimiter=",")

    print(f"✅ Wrote {RUNS/'portfolio_drift_watch.json'}")
    print(f"✅ Updated {prev_path}")

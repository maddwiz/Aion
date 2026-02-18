#!/usr/bin/env python3
"""
Hive conviction gate.

Builds a per-asset, per-time scalar matrix based on intra-hive sign agreement.

Writes:
  - runs_plus/hive_conviction_scalar.csv
  - runs_plus/hive_conviction_info.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def _append_card(title: str, html: str) -> None:
    if str(os.getenv("Q_DISABLE_REPORT_CARDS", "0")).strip().lower() in {"1", "true", "yes", "on"}:
        return
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html", "report.html"]:
        p = ROOT / name
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        txt = txt.replace("</body>", card + "</body>") if "</body>" in txt else txt + card
        p.write_text(txt, encoding="utf-8")


def _load_matrix(path: Path) -> np.ndarray | None:
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
    if a.size == 0:
        return None
    return a


def _load_asset_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if df.empty:
        return []
    col = None
    for c in df.columns:
        if str(c).strip().lower() in {"symbol", "asset", "ticker", "name"}:
            col = c
            break
    if col is None:
        col = df.columns[0]
    out = [str(x).strip().upper() for x in df[col].tolist() if str(x).strip()]
    return out


def _load_hives(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(obj, dict):
        return {}
    h = obj.get("hives", obj)
    if not isinstance(h, dict):
        return {}

    out = {}
    for k, vals in h.items():
        if not isinstance(vals, list):
            continue
        key = str(k).strip().upper()
        members = sorted({str(x).strip().upper() for x in vals if str(x).strip()})
        if key and members:
            out[key] = members
    return out


def _asset_to_hive_members(hives: dict[str, list[str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for _hive, members in hives.items():
        for m in members:
            out[m] = list(members)
    return out


def compute_hive_conviction_scalar(
    weights: np.ndarray,
    asset_names: list[str],
    hives: dict[str, list[str]],
    *,
    threshold: float,
    floor: float,
    ceil: float,
    high_conviction: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    w = np.asarray(weights, float)
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    t, n = w.shape
    if t <= 0 or n <= 0:
        return np.ones((0, 0), dtype=float), {
            "rows": 0,
            "cols": 0,
            "mean_scalar": 1.0,
            "low_conviction_cells": 0,
            "high_conviction_cells": 0,
            "hive_assets_covered": 0,
        }

    threshold = float(np.clip(float(threshold), 0.05, 0.95))
    high_conviction = float(np.clip(float(high_conviction), threshold + 1e-6, 0.99))
    floor = float(np.clip(float(floor), 0.05, 1.0))
    ceil = float(np.clip(float(ceil), floor, 2.0))

    names = [str(x).strip().upper() for x in asset_names]
    if len(names) != n:
        names = [f"ASSET_{i+1}" for i in range(n)]
    idx_of = {s: i for i, s in enumerate(names)}
    a2m = _asset_to_hive_members(hives)

    out = np.ones((t, n), dtype=float)
    low_cells = 0
    high_cells = 0

    for j, sym in enumerate(names):
        members = [m for m in a2m.get(sym, []) if m in idx_of and m != sym]
        if not members:
            continue
        peer_idx = [idx_of[m] for m in members]

        for i in range(t):
            sgn = np.sign(float(w[i, j]))
            if sgn == 0:
                out[i, j] = 1.0
                continue

            peer_signs = np.sign(w[i, peer_idx])
            active = peer_signs != 0
            if int(np.count_nonzero(active)) <= 0:
                out[i, j] = 1.0
                continue

            agree = float(np.mean((peer_signs[active] == sgn).astype(float)))
            if agree <= threshold:
                val = floor
                low_cells += 1
            elif agree >= high_conviction:
                val = ceil
                high_cells += 1
            else:
                alpha = (agree - threshold) / max(1e-9, (high_conviction - threshold))
                val = floor + alpha * (ceil - floor)
            out[i, j] = float(np.clip(val, floor, ceil))

    info = {
        "rows": int(t),
        "cols": int(n),
        "mean_scalar": float(np.mean(out)),
        "low_conviction_cells": int(low_cells),
        "high_conviction_cells": int(high_cells),
        "hive_assets_covered": int(sum(1 for s in names if s in a2m)),
    }
    return out, info


def _first_weights() -> tuple[np.ndarray | None, str | None]:
    cands = [
        RUNS / "portfolio_weights_final.csv",
        RUNS / "weights_regime.csv",
        RUNS / "portfolio_weights.csv",
        ROOT / "portfolio_weights.csv",
    ]
    for p in cands:
        m = _load_matrix(p)
        if m is not None:
            return m, str(p.relative_to(ROOT))
    return None, None


def main() -> int:
    threshold = float(np.clip(float(os.getenv("Q_HIVE_CONVICTION_THRESHOLD", "0.40")), 0.05, 0.95))
    floor = float(np.clip(float(os.getenv("Q_HIVE_CONVICTION_FLOOR", "0.30")), 0.05, 1.0))
    ceil = float(np.clip(float(os.getenv("Q_HIVE_CONVICTION_CEIL", "1.12")), floor, 2.0))
    high_conviction = float(np.clip(float(os.getenv("Q_HIVE_CONVICTION_HIGH", "0.70")), threshold + 1e-6, 0.99))

    w, source = _first_weights()
    if w is None:
        print("(!) Missing base weights for hive conviction gate; skipping.")
        return 0

    asset_names = _load_asset_names(RUNS / "asset_names.csv")
    hives = _load_hives(RUNS / "hive.json")

    if (not hives) or (not asset_names):
        scalar = np.ones_like(w, dtype=float)
        info = {
            "ok": False,
            "source_weights": source,
            "reason": "missing_hive_or_asset_map",
            "rows": int(w.shape[0]),
            "cols": int(w.shape[1]),
            "mean_scalar": 1.0,
        }
    else:
        scalar, stats = compute_hive_conviction_scalar(
            w,
            asset_names,
            hives,
            threshold=threshold,
            floor=floor,
            ceil=ceil,
            high_conviction=high_conviction,
        )
        info = {
            "ok": True,
            "source_weights": source,
            "params": {
                "threshold": float(threshold),
                "high_conviction": float(high_conviction),
                "floor": float(floor),
                "ceil": float(ceil),
            },
            "hive_count": int(len(hives)),
            **stats,
        }

    np.savetxt(RUNS / "hive_conviction_scalar.csv", np.asarray(scalar, float), delimiter=",")
    (RUNS / "hive_conviction_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    _append_card(
        "Hive Conviction Gate ✔",
        (
            f"<p>rows={int(scalar.shape[0])}, cols={int(scalar.shape[1])}, source={source}.</p>"
            f"<p>mean_scalar={float(np.mean(scalar)):.3f}, "
            f"range=[{float(np.min(scalar)):.3f}, {float(np.max(scalar)):.3f}]</p>"
        ),
    )

    print(f"✅ Wrote {RUNS/'hive_conviction_scalar.csv'}")
    print(f"✅ Wrote {RUNS/'hive_conviction_info.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

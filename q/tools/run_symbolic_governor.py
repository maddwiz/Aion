#!/usr/bin/env python3
# Build symbolic risk governor from symbolic_signal.csv.
#
# Writes:
#   runs_plus/symbolic_governor.csv
#   runs_plus/symbolic_governor_info.json

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmods.symbolic_governor import build_symbolic_governor

RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def _append_card(title, html):
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html", "report.html"]:
        p = ROOT / name
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        txt = txt.replace("</body>", card + "</body>") if "</body>" in txt else txt + card
        p.write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    p = RUNS / "symbolic_signal.csv"
    if not p.exists():
        print("(!) symbolic_signal.csv missing; skipping symbolic governor.")
        raise SystemExit(0)
    try:
        df = pd.read_csv(p)
    except Exception:
        print("(!) Failed to read symbolic_signal.csv; skipping.")
        raise SystemExit(0)
    need = {"DATE", "sym_signal"}
    if not need.issubset(df.columns):
        print("(!) symbolic_signal.csv missing required columns; skipping.")
        raise SystemExit(0)

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"]).sort_values("DATE")
    g = df.groupby("DATE", as_index=False)
    agg = g.agg(sym_signal=("sym_signal", "mean"))
    if "sym_affect" in df.columns:
        agg = agg.merge(g.agg(sym_affect=("sym_affect", "mean")), on="DATE", how="left")
        agg = agg.merge(g.agg(sym_affect_std=("sym_affect", "std")), on="DATE", how="left")
    else:
        agg["sym_affect"] = agg["sym_signal"]
        agg["sym_affect_std"] = 0.0
    if "confidence" in df.columns:
        agg = agg.merge(g.agg(confidence=("confidence", "mean")), on="DATE", how="left")
        agg = agg.merge(g.agg(confidence_std=("confidence", "std")), on="DATE", how="left")
    else:
        agg["confidence"] = 0.5
        agg["confidence_std"] = 0.0
    if "events_n" in df.columns:
        agg = agg.merge(g.agg(events_n=("events_n", "sum")), on="DATE", how="left")
    else:
        agg = agg.merge(g.agg(events_n=("sym_signal", "size")), on="DATE", how="left")
    if "sym_regime" in df.columns:
        agg = agg.merge(g.agg(sym_regime=("sym_regime", "mean")), on="DATE", how="left")
    else:
        agg["sym_regime"] = 0.0

    if "confidence" not in agg.columns:
        agg["confidence"] = 0.5
    if "events_n" not in agg.columns:
        agg["events_n"] = 1.0
    agg["confidence"] = pd.to_numeric(agg["confidence"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    agg["events_n"] = pd.to_numeric(agg["events_n"], errors="coerce").fillna(1.0).clip(lower=0.0)
    agg["sym_affect"] = pd.to_numeric(agg["sym_affect"], errors="coerce").fillna(0.0)
    agg["sym_signal"] = pd.to_numeric(agg["sym_signal"], errors="coerce").fillna(0.0)
    agg["sym_regime"] = pd.to_numeric(agg.get("sym_regime", 0.0), errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    agg["confidence_std"] = pd.to_numeric(agg.get("confidence_std", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    agg["sym_affect_std"] = pd.to_numeric(agg.get("sym_affect_std", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    cden = float(np.percentile(agg["confidence_std"].values, 90)) + 1e-9 if len(agg) else 1.0
    aden = float(np.percentile(agg["sym_affect_std"].values, 90)) + 1e-9 if len(agg) else 1.0
    agg["confidence_uncertainty"] = np.clip(agg["confidence_std"] / cden, 0.0, 1.0)
    agg["affect_uncertainty"] = np.clip(agg["sym_affect_std"] / aden, 0.0, 1.0)

    stress, gov, info = build_symbolic_governor(
        agg["sym_signal"].values,
        sym_affect=agg["sym_affect"].values,
        confidence=agg["confidence"].values,
        events_n=agg["events_n"].values,
        sym_regime=np.clip(agg["sym_regime"].values, 0.0, 1.0),
        confidence_uncertainty=agg["confidence_uncertainty"].values,
        affect_uncertainty=agg["affect_uncertainty"].values,
        lo=0.72,
        hi=1.12,
        smooth=0.88,
    )
    out = pd.DataFrame(
        {
            "DATE": agg["DATE"].dt.strftime("%Y-%m-%d"),
            "symbolic_stress": stress,
            "symbolic_governor": gov,
        }
    )
    out.to_csv(RUNS / "symbolic_governor.csv", index=False)

    meta = {
        **info,
        "rows": int(len(out)),
        "source_rows": int(len(df)),
        "source_file": str(p),
    }
    (RUNS / "symbolic_governor_info.json").write_text(json.dumps(meta, indent=2))

    _append_card(
        "Symbolic Governor ✔",
        (
            f"<p>rows={meta['rows']}, mean stress={meta['mean_stress']:.3f}, max stress={meta['max_stress']:.3f}</p>"
            f"<p>governor mean={meta['mean_governor']:.3f}, "
            f"min={meta['min_governor']:.3f}, max={meta['max_governor']:.3f}</p>"
        ),
    )

    print(f"✅ Wrote {RUNS/'symbolic_governor.csv'}")
    print(f"✅ Wrote {RUNS/'symbolic_governor_info.json'}")

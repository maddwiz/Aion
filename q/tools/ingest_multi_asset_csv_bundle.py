#!/usr/bin/env python3
"""
Ingest a local multi-asset CSV bundle into q/data_new.

Use this when you have a folder of historical CSVs and want Q to absorb them
without manual per-file cleanup.

Env:
  - Q_MULTI_ASSET_SOURCE_DIR (required unless --source is passed)
  - Q_MULTI_ASSET_COPY_LIMIT (default: 5000)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_NEW = ROOT / "data_new"
RUNS = ROOT / "runs_plus"
DATA_NEW.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)


def _infer_class(sym: str) -> str:
    s = str(sym or "").upper().replace("-", "").replace("_", "").replace("/", "")
    if not s:
        return "EQ"
    if s in {"VIX", "VIX9D", "VIX3M", "VIXCLS", "UVXY", "VXX"}:
        return "VOL"
    if s in {"LQD", "HYG", "JNK", "BND", "AGG", "MBB", "HYGTR", "LQDTR"} or s.endswith("_TR"):
        return "CREDIT"
    if s in {"TLT", "IEF", "SHY", "VGSH", "DGS2", "DGS3MO", "DGS5", "DGS10", "DGS30", "TY", "ZN", "ZB"}:
        return "RATES"
    if s in {"GLD", "SLV", "USO", "UNG", "CORN", "WEAT", "CPER", "DBC", "DBA", "XLE", "XOP"}:
        return "COMMOD"
    if len(s) == 6 and s[:3] in {"USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "NZD"} and s[3:] in {
        "USD",
        "EUR",
        "JPY",
        "GBP",
        "CHF",
        "CAD",
        "AUD",
        "NZD",
    }:
        return "FX"
    if s.startswith("BTC") or s.startswith("ETH") or s in {"IBIT", "FBTC", "BITO"}:
        return "CRYPTO"
    return "EQ"


def _sanitize_symbol(name: str) -> str:
    base = Path(str(name)).stem.upper().strip()
    base = re.sub(r"[^A-Z0-9_]", "", base)
    return base[:32] if base else ""


def _usable_csv(path: Path) -> tuple[bool, str]:
    try:
        df = pd.read_csv(path, nrows=20)
    except Exception:
        return False, "unreadable"
    if df.empty:
        return False, "empty"
    cols = {str(c).strip().lower(): c for c in df.columns}
    has_date = any(x in cols for x in ["date", "timestamp", "time"])
    has_close = any(x in cols for x in ["close", "adj close", "adj_close", "value", "price"])
    if not has_date:
        return False, "missing_date"
    if not has_close:
        return False, "missing_close"
    return True, "ok"


def _merge_cluster_map(pairs: list[tuple[str, str]]) -> None:
    p = RUNS / "cluster_map.csv"
    existing = {}
    if p.exists():
        try:
            df = pd.read_csv(p)
            if {"asset", "cluster"}.issubset({str(c).strip().lower() for c in df.columns}):
                aset = [str(x).strip().upper() for x in df.iloc[:, 0].tolist()]
                clus = [str(x).strip().upper() for x in df.iloc[:, 1].tolist()]
                for a, c in zip(aset, clus):
                    if a and c:
                        existing[a] = c
        except Exception:
            existing = {}
    for a, c in pairs:
        if a and c:
            existing[a] = c
    out = pd.DataFrame({"asset": sorted(existing.keys()), "cluster": [existing[k] for k in sorted(existing.keys())]})
    out.to_csv(p, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="")
    ap.add_argument("--limit", type=int, default=int(np.clip(int(float(os.getenv("Q_MULTI_ASSET_COPY_LIMIT", "5000"))), 1, 50000)))
    args = ap.parse_args()

    src_raw = str(args.source).strip() or str(os.getenv("Q_MULTI_ASSET_SOURCE_DIR", "")).strip()
    if not src_raw:
        print("(!) Missing source directory. Set Q_MULTI_ASSET_SOURCE_DIR or pass --source.")
        return 0
    src = Path(src_raw).expanduser().resolve()
    if (not src.exists()) or (not src.is_dir()):
        print(f"(!) Source directory not found: {src}")
        return 0

    files = sorted([p for p in src.rglob("*.csv") if p.is_file()])[: max(1, int(args.limit))]
    copied = []
    skipped = []
    cluster_pairs = []
    used_names = set()

    for f in files:
        ok, reason = _usable_csv(f)
        if not ok:
            skipped.append({"file": str(f), "reason": reason})
            continue
        sym = _sanitize_symbol(f.name)
        if not sym:
            skipped.append({"file": str(f), "reason": "bad_symbol"})
            continue
        out_name = sym
        suffix = 1
        while out_name in used_names:
            suffix += 1
            out_name = f"{sym}_{suffix}"
        used_names.add(out_name)
        dst = DATA_NEW / f"{out_name}.csv"
        try:
            shutil.copy2(f, dst)
        except Exception:
            skipped.append({"file": str(f), "reason": "copy_failed"})
            continue
        copied.append(str(dst))
        cluster_pairs.append((out_name, _infer_class(out_name)))

    if cluster_pairs:
        _merge_cluster_map(cluster_pairs)

    out = {
        "ok": True,
        "source_dir": str(src),
        "files_scanned": int(len(files)),
        "files_copied": int(len(copied)),
        "files_skipped": int(len(skipped)),
        "copied_examples": copied[:20],
        "skipped_examples": skipped[:40],
        "cluster_map_updated": bool(len(cluster_pairs) > 0),
        "cluster_classes_added": sorted(set(c for _a, c in cluster_pairs)),
    }
    (RUNS / "multi_asset_ingest_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"✅ Wrote {RUNS/'multi_asset_ingest_report.json'}")
    print(f"Ingested {len(copied)} CSVs into {DATA_NEW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

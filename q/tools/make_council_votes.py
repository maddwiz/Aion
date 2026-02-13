#!/usr/bin/env python3
# Build runs_plus/council_votes.csv from strongest available sources.
# Priority:
# 1) runs_plus/council_preds*.csv (K columns)
# 2) runs_plus/sleeve_*_signal.csv (merge columns)
# 3) runs_plus/hive_signals.csv (pivot HIVE into K columns)
# 4) Fallback: deterministic engineered votes from returns
# 5) Last resort: deterministic synthetic waves (no randomness)

import csv, glob
from pathlib import Path
import numpy as np
import json

try:
    import pandas as pd
except Exception:
    pd = None

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT/"runs_plus"; RUNS.mkdir(exist_ok=True)

def save_votes(V, path, source):
    V = np.asarray(V, float)
    if V.ndim == 1:
        V = V.reshape(-1, 1)
    # Robust squashing for stable downstream meta learners.
    V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
    V = np.tanh(zscore(V))
    np.savetxt(path, V, delimiter=",")
    info = {"source": source, "rows": int(V.shape[0]), "cols": int(V.shape[1])}
    (RUNS / "council_votes_info.json").write_text(json.dumps(info, indent=2))
    print(f"✅ Wrote {path}  shape={V.shape}  source={source}")

def load_first_match(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(str(RUNS/pat)))
        if hits:
            return hits[0]
    return None

def load_csv_matrix(path):
    # accepts csv with/without header; returns np.ndarray [T,K]
    try:
        arr = np.loadtxt(path, delimiter=",")
        if arr.ndim == 1: arr = arr.reshape(-1,1)
        return arr
    except Exception:
        # try skip header
        with open(path) as f:
            r = csv.reader(f)
            rows = []
            header = next(r, None)
            for row in r:
                try:
                    rows.append([float(x) for x in row])
                except:
                    pass
        arr = np.array(rows, float)
        if arr.ndim == 1: arr = arr.reshape(-1,1)
        return arr

def load_series(path):
    try:
        a = np.loadtxt(path, delimiter=",").ravel()
    except Exception:
        try:
            a = np.loadtxt(path, delimiter=",", skiprows=1).ravel()
        except Exception:
            return None
    a = np.asarray(a, float).ravel()
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

def zscore(x, axis=0):
    x = np.asarray(x, float)
    mu = np.nanmean(x, axis=axis, keepdims=True)
    sd = np.nanstd(x, axis=axis, keepdims=True) + 1e-9
    return (x - mu) / sd

def rolling_mean(x, w):
    w = int(max(1, w))
    if w <= 1:
        return np.asarray(x, float).copy()
    k = np.ones(w, float) / float(w)
    return np.convolve(np.asarray(x, float), k, mode="same")

def rolling_std(x, w):
    w = int(max(2, w))
    x = np.asarray(x, float)
    mu = rolling_mean(x, w)
    mu2 = rolling_mean(x * x, w)
    v = np.maximum(mu2 - mu * mu, 0.0)
    return np.sqrt(v + 1e-12)

def build_votes_from_returns(r):
    r = np.asarray(r, float).ravel()
    if len(r) < 32:
        return None

    m5 = rolling_mean(r, 5)
    m21 = rolling_mean(r, 21)
    m63 = rolling_mean(r, 63)
    vol21 = rolling_std(r, 21)
    vol63 = rolling_std(r, 63)

    # Core council-style channels.
    trend_fast = m5
    trend_mid = m21
    mean_rev = -(r - m5)
    breakout = r / (vol21 + 1e-9)
    regime = -(vol21 - vol63)  # calmer regime => positive vote

    eq = np.cumprod(1.0 + np.clip(r, -0.95, 0.95))
    peak = np.maximum.accumulate(eq)
    dd = eq / (peak + 1e-12) - 1.0
    dd_rebound = -dd

    V = np.column_stack([trend_fast, trend_mid, mean_rev, breakout, regime, dd_rebound])
    return np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)

def build_votes_from_hive():
    p = RUNS / "hive_signals.csv"
    if pd is None or not p.exists():
        return None
    try:
        h = pd.read_csv(p)
    except Exception:
        return None
    need = {"DATE", "HIVE", "hive_signal"}
    if not need.issubset(h.columns):
        return None
    h["DATE"] = pd.to_datetime(h["DATE"], errors="coerce")
    h = h.dropna(subset=["DATE"]).sort_values(["DATE", "HIVE"])
    piv = h.pivot(index="DATE", columns="HIVE", values="hive_signal").fillna(0.0)
    if piv.empty or piv.shape[0] < 20:
        return None
    return piv.values

def main():
    out = RUNS/"council_votes.csv"

    # 1) Real council predictions?
    real = load_first_match(["council_preds.csv", "council_predictions.csv"])
    if real:
        V = load_csv_matrix(real)
        save_votes(V, out, source=Path(real).name)
        return

    # 2) Merge sleeve signals if present
    sleeves = sorted(glob.glob(str(RUNS/"sleeve_*_signal.csv")))
    if sleeves:
        mats = [load_csv_matrix(p) for p in sleeves]
        T = min(m.shape[0] for m in mats)
        mats = [m[:T] for m in mats]
        V = np.column_stack(mats)
        save_votes(V, out, source="sleeve_signals")
        return

    # 3) Candidate from hive signals
    Vh = build_votes_from_hive()

    # 4) Candidate from returns
    ret_file = load_first_match(["daily_returns.csv","portfolio_daily_returns.csv","returns.csv"])
    Vr = None
    if ret_file:
        r = load_series(ret_file)
        Vr = build_votes_from_returns(r) if r is not None else None

    # Prefer richer/longer matrix when both exist.
    if Vh is not None and Vr is not None:
        if Vh.shape[0] >= int(0.60 * Vr.shape[0]):
            save_votes(Vh, out, source="hive_signals")
        else:
            save_votes(Vr, out, source=Path(ret_file).name if ret_file else "daily_returns")
        return
    if Vh is not None:
        save_votes(Vh, out, source="hive_signals")
        return
    if Vr is not None:
        save_votes(Vr, out, source=Path(ret_file).name if ret_file else "daily_returns")
        return

    # 5) Last resort deterministic synthetic wave channels
    T = 600
    t = np.linspace(0.0, 20.0, T)
    v1 = np.sin(t)
    v2 = np.cos(0.7 * t)
    v3 = np.sin(0.17 * t + 1.3)
    V = np.vstack([v1, v2, v3]).T
    save_votes(V, out, source="deterministic_synthetic")

if __name__ == "__main__":
    main()

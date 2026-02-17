#!/usr/bin/env python3
"""
Microstructure proxy overlay from daily OHLCV.

Builds a market-wide scalar from:
  - Amihud illiquidity stress (|ret| / volume) across assets
  - Close-location value pressure (where closes print within daily range)

Writes:
  - runs_plus/microstructure_signal.csv   (signed signal in [-1, 1], + risk-on)
  - runs_plus/microstructure_overlay.csv  (exposure scalar around 1.0)
  - runs_plus/microstructure_info.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
DATA = ROOT / "data"
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


def _target_len() -> int:
    p = RUNS / "asset_returns.csv"
    if not p.exists():
        return 0
    try:
        a = np.loadtxt(p, delimiter=",")
    except Exception:
        try:
            a = np.loadtxt(p, delimiter=",", skiprows=1)
        except Exception:
            return 0
    a = np.asarray(a, float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return int(a.shape[0])


def _align_tail(v: np.ndarray, t: int, fill: float) -> np.ndarray:
    x = np.asarray(v, float).ravel()
    if t <= 0:
        return x
    if x.size >= t:
        return x[-t:]
    out = np.full(t, float(fill), dtype=float)
    if x.size > 0:
        out[-x.size :] = x
        out[: t - x.size] = float(x[0])
    return out


def _roll_z(s: pd.Series, lookback: int, minp: int) -> pd.Series:
    if s.empty:
        return s
    mu = s.rolling(lookback, min_periods=minp).mean()
    sd = s.rolling(lookback, min_periods=minp).std(ddof=1).replace(0.0, np.nan)
    z = (s - mu) / (sd + 1e-12)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _read_ohlcv(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None

    dcol = None
    for c in ["DATE", "Date", "date", "timestamp", "Timestamp"]:
        if c in df.columns:
            dcol = c
            break
    if dcol is None:
        return None

    cols = {str(c).lower(): c for c in df.columns}
    c_close = cols.get("close") or cols.get("adj close") or cols.get("adj_close")
    c_high = cols.get("high")
    c_low = cols.get("low")
    c_vol = cols.get("volume")
    if c_close is None or c_high is None or c_low is None or c_vol is None:
        return None

    idx = pd.to_datetime(df[dcol], errors="coerce")
    out = pd.DataFrame(
        {
            "close": pd.to_numeric(df[c_close], errors="coerce").to_numpy(),
            "high": pd.to_numeric(df[c_high], errors="coerce").to_numpy(),
            "low": pd.to_numeric(df[c_low], errors="coerce").to_numpy(),
            "volume": pd.to_numeric(df[c_vol], errors="coerce").to_numpy(),
        },
        index=idx,
    ).dropna()
    if out.empty:
        return None
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if len(out) < 40:
        return None
    return out


def main() -> int:
    files = sorted(DATA.glob("*.csv"))
    max_assets = int(np.clip(int(float(os.getenv("Q_MICROSTRUCTURE_MAX_ASSETS", "120"))), 10, 5000))
    lookback = int(np.clip(int(float(os.getenv("Q_MICROSTRUCTURE_LOOKBACK", "126"))), 40, 756))
    min_assets = int(np.clip(int(float(os.getenv("Q_MICROSTRUCTURE_MIN_ASSETS", "8"))), 2, 500))
    smooth_alpha = float(np.clip(float(os.getenv("Q_MICROSTRUCTURE_SMOOTH_ALPHA", "0.16")), 0.01, 0.90))
    beta = float(np.clip(float(os.getenv("Q_MICROSTRUCTURE_BETA", "0.16")), 0.0, 1.2))
    floor = float(np.clip(float(os.getenv("Q_MICROSTRUCTURE_FLOOR", "0.82")), 0.2, 1.2))
    ceil = float(np.clip(float(os.getenv("Q_MICROSTRUCTURE_CEIL", "1.18")), floor, 1.8))

    am_parts = []
    clv_parts = []
    used = 0
    for p in files:
        if used >= max_assets:
            break
        df = _read_ohlcv(p)
        if df is None:
            continue
        c = df["close"]
        h = df["high"]
        l = df["low"]
        v = df["volume"].replace(0.0, np.nan)
        ret = c.pct_change()
        amihud = (ret.abs() / (v + 1e-9)).replace([np.inf, -np.inf], np.nan)
        am_log = np.log1p((amihud * 1e6).clip(lower=0.0))
        am_z = _roll_z(am_log, lookback=lookback, minp=max(20, lookback // 4))

        clv = (((c - l) - (h - c)) / ((h - l).abs() + 1e-9)).clip(-1.0, 1.0)
        clv_s = clv.rolling(3, min_periods=1).mean()

        am_parts.append(am_z.rename(p.stem))
        clv_parts.append(clv_s.rename(p.stem))
        used += 1

    t = _target_len()
    if (used < min_assets) or (not am_parts) or (not clv_parts):
        if t <= 0:
            print("(!) Microstructure inputs insufficient; skipping.")
            return 0
        sig = np.zeros(t, dtype=float)
        ov = np.ones(t, dtype=float)
        np.savetxt(RUNS / "microstructure_signal.csv", sig, delimiter=",")
        np.savetxt(RUNS / "microstructure_overlay.csv", ov, delimiter=",")
        (RUNS / "microstructure_info.json").write_text(
            json.dumps(
                {
                    "ok": False,
                    "reason": "insufficient_assets",
                    "assets_used": int(used),
                    "assets_min_required": int(min_assets),
                    "rows": int(t),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return 0

    am_df = pd.concat(am_parts, axis=1).sort_index()
    clv_df = pd.concat(clv_parts, axis=1).sort_index()
    idx = am_df.index.union(clv_df.index).sort_values()
    am_df = am_df.reindex(idx)
    clv_df = clv_df.reindex(idx)

    # Cross-sectional medians avoid single-asset outlier domination.
    am_cross = am_df.median(axis=1, skipna=True).fillna(0.0)
    clv_cross = clv_df.median(axis=1, skipna=True).fillna(0.0)

    am_z = _roll_z(am_cross, lookback=lookback, minp=max(20, lookback // 4))
    clv_z = _roll_z(clv_cross, lookback=lookback, minp=max(20, lookback // 4))

    raw = np.clip(-0.65 * am_z.values + 0.35 * clv_z.values, -6.0, 6.0)
    sig = np.tanh(raw / 2.0)
    sig = pd.Series(sig, index=idx).ewm(alpha=smooth_alpha, adjust=False).mean().clip(-1.0, 1.0).values
    ov = np.clip(1.0 + beta * sig, floor, ceil)

    if t > 0:
        sig = _align_tail(sig, t, 0.0)
        ov = _align_tail(ov, t, 1.0)

    np.savetxt(RUNS / "microstructure_signal.csv", np.asarray(sig, float), delimiter=",")
    np.savetxt(RUNS / "microstructure_overlay.csv", np.asarray(ov, float), delimiter=",")

    info = {
        "ok": True,
        "rows": int(len(sig)),
        "assets_used": int(used),
        "params": {
            "lookback": int(lookback),
            "smooth_alpha": float(smooth_alpha),
            "beta": float(beta),
            "floor": float(floor),
            "ceil": float(ceil),
            "max_assets": int(max_assets),
            "min_assets": int(min_assets),
        },
        "signal_mean": float(np.mean(sig)),
        "signal_min": float(np.min(sig)),
        "signal_max": float(np.max(sig)),
        "overlay_mean": float(np.mean(ov)),
        "overlay_min": float(np.min(ov)),
        "overlay_max": float(np.max(ov)),
    }
    (RUNS / "microstructure_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    _append_card(
        "Microstructure Overlay ✔",
        (
            f"<p>rows={len(sig)}, assets={used}, signal_mean={info['signal_mean']:.3f}, "
            f"overlay_range=[{info['overlay_min']:.3f},{info['overlay_max']:.3f}].</p>"
        ),
    )
    print(f"✅ Wrote {RUNS/'microstructure_signal.csv'}")
    print(f"✅ Wrote {RUNS/'microstructure_overlay.csv'}")
    print(f"✅ Wrote {RUNS/'microstructure_info.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

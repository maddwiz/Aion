#!/usr/bin/env python3
"""
Credit-to-equity lead/lag overlay.

Detects divergence between credit risk appetite (HYG/LQD) and an equity
benchmark (SPY default, with fallback to QQQ). Produces:
  - runs_plus/credit_leadlag_signal.csv   (signed signal in [-1, 1], + risk-on)
  - runs_plus/credit_leadlag_overlay.csv  (exposure scalar around 1.0)
  - runs_plus/credit_leadlag_info.json
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


def _read_level(sym: str) -> pd.Series:
    p = DATA / f"{sym}.csv"
    if not p.exists():
        return pd.Series(dtype=float)
    try:
        df = pd.read_csv(p)
    except Exception:
        return pd.Series(dtype=float)
    if df.empty:
        return pd.Series(dtype=float)
    dcol = None
    for c in ["DATE", "Date", "date", "timestamp", "Timestamp"]:
        if c in df.columns:
            dcol = c
            break
    if dcol is None:
        return pd.Series(dtype=float)
    vcol = None
    for c in ["Close", "Adj Close", "close", "adj_close", "value", "Value", "PRICE", "price"]:
        if c in df.columns:
            vcol = c
            break
    if vcol is None:
        for c in df.columns:
            if c == dcol:
                continue
            try:
                pd.to_numeric(df[c], errors="raise")
                vcol = c
                break
            except Exception:
                continue
    if vcol is None:
        return pd.Series(dtype=float)
    d = pd.to_datetime(df[dcol], errors="coerce")
    v = pd.to_numeric(df[vcol], errors="coerce")
    s = pd.Series(v.values, index=d).dropna()
    if s.empty:
        return pd.Series(dtype=float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.astype(float)


def _roll_z(s: pd.Series, lookback: int, minp: int) -> pd.Series:
    if s.empty:
        return s
    mu = s.rolling(lookback, min_periods=minp).mean()
    sd = s.rolling(lookback, min_periods=minp).std(ddof=1).replace(0.0, np.nan)
    z = (s - mu) / (sd + 1e-12)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


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


def build_credit_leadlag_signal(
    hyg: pd.Series,
    lqd: pd.Series,
    equity: pd.Series,
    *,
    lookback: int = 126,
    fast_horizon: int = 5,
    corr_window: int = 20,
    smooth_alpha: float = 0.15,
) -> pd.Series:
    idx = hyg.index.union(lqd.index).union(equity.index).sort_values()
    h = hyg.reindex(idx).ffill()
    l = lqd.reindex(idx).ffill()
    e = equity.reindex(idx).ffill()

    # Credit risk appetite proxy.
    credit_ratio = np.log((h + 1e-6) / (l + 1e-6))
    credit_fast = credit_ratio.diff(max(1, int(fast_horizon)))
    equity_fast = np.log(e + 1e-6).diff(max(1, int(fast_horizon)))

    # Divergence: when credit weakens while equities stay elevated -> risk-off lead.
    divergence = credit_fast - equity_fast
    z_div = _roll_z(divergence, lookback=max(20, int(lookback)), minp=max(20, int(lookback // 4)))

    # Correlation breakdown between daily credit and equity returns.
    c_ret = credit_ratio.diff()
    e_ret = np.log(e + 1e-6).diff()
    short_w = max(10, int(corr_window))
    long_w = max(60, int(corr_window) * 4)
    c_short = c_ret.rolling(short_w, min_periods=max(8, short_w // 2)).corr(e_ret)
    c_long = c_ret.rolling(long_w, min_periods=max(24, long_w // 3)).corr(e_ret)
    corr_break = (c_long - c_short).clip(lower=0.0)
    z_break = _roll_z(corr_break, lookback=max(30, int(lookback)), minp=max(20, int(lookback // 4)))

    stress = np.clip(np.maximum(0.0, -z_div.values), 0.0, 4.0) + 0.6 * np.clip(
        np.maximum(0.0, z_break.values),
        0.0,
        4.0,
    )
    relief = np.clip(np.maximum(0.0, z_div.values), 0.0, 4.0)

    raw = np.clip(relief - stress, -6.0, 6.0)
    signed = np.tanh(raw / 2.0)
    sig = pd.Series(signed, index=idx).ewm(alpha=float(np.clip(smooth_alpha, 0.01, 0.9)), adjust=False).mean()
    return sig.clip(-1.0, 1.0)


def main() -> int:
    hyg = _read_level("HYG")
    lqd = _read_level("LQD")
    pref = str(os.getenv("Q_CREDIT_LEADLAG_EQ_SYMBOL", "SPY")).strip().upper() or "SPY"
    eq_candidates = [pref]
    for s in ["SPY", "QQQ"]:
        if s not in eq_candidates:
            eq_candidates.append(s)
    equity = pd.Series(dtype=float)
    eq_used = None
    for sym in eq_candidates:
        s = _read_level(sym)
        if not s.empty:
            equity = s
            eq_used = sym
            break

    t = _target_len()
    if hyg.empty or lqd.empty or equity.empty:
        if t <= 0:
            print("(!) Missing credit/equity inputs and no target length; skipping.")
            return 0
        sig = np.zeros(t, dtype=float)
        overlay = np.ones(t, dtype=float)
        np.savetxt(RUNS / "credit_leadlag_signal.csv", sig, delimiter=",")
        np.savetxt(RUNS / "credit_leadlag_overlay.csv", overlay, delimiter=",")
        (RUNS / "credit_leadlag_info.json").write_text(
            json.dumps(
                {
                    "ok": False,
                    "reason": "missing_inputs",
                    "rows": int(t),
                    "inputs_present": {
                        "HYG": bool(not hyg.empty),
                        "LQD": bool(not lqd.empty),
                        "equity": bool(not equity.empty),
                        "equity_symbol": eq_used,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return 0

    lookback = int(np.clip(int(float(os.getenv("Q_CREDIT_LEADLAG_LOOKBACK", "126"))), 40, 504))
    fast_horizon = int(np.clip(int(float(os.getenv("Q_CREDIT_LEADLAG_FAST_HORIZON", "5"))), 1, 40))
    corr_window = int(np.clip(int(float(os.getenv("Q_CREDIT_LEADLAG_CORR_WINDOW", "20"))), 8, 80))
    smooth_alpha = float(np.clip(float(os.getenv("Q_CREDIT_LEADLAG_SMOOTH_ALPHA", "0.15")), 0.01, 0.90))

    sig_s = build_credit_leadlag_signal(
        hyg,
        lqd,
        equity,
        lookback=lookback,
        fast_horizon=fast_horizon,
        corr_window=corr_window,
        smooth_alpha=smooth_alpha,
    )

    beta = float(np.clip(float(os.getenv("Q_CREDIT_LEADLAG_BETA", "0.18")), 0.0, 1.2))
    floor = float(np.clip(float(os.getenv("Q_CREDIT_LEADLAG_FLOOR", "0.80")), 0.2, 1.2))
    ceil = float(np.clip(float(os.getenv("Q_CREDIT_LEADLAG_CEIL", "1.18")), floor, 1.8))

    sig = np.asarray(sig_s.values, float)
    overlay = np.clip(1.0 + beta * sig, floor, ceil)

    if t > 0:
        sig = _align_tail(sig, t, 0.0)
        overlay = _align_tail(overlay, t, 1.0)

    np.savetxt(RUNS / "credit_leadlag_signal.csv", sig, delimiter=",")
    np.savetxt(RUNS / "credit_leadlag_overlay.csv", overlay, delimiter=",")

    info = {
        "ok": True,
        "rows": int(len(sig)),
        "equity_symbol": str(eq_used or pref),
        "params": {
            "lookback": int(lookback),
            "fast_horizon": int(fast_horizon),
            "corr_window": int(corr_window),
            "smooth_alpha": float(smooth_alpha),
            "beta": float(beta),
            "floor": float(floor),
            "ceil": float(ceil),
        },
        "signal_mean": float(np.mean(sig)),
        "signal_min": float(np.min(sig)),
        "signal_max": float(np.max(sig)),
        "overlay_mean": float(np.mean(overlay)),
        "overlay_min": float(np.min(overlay)),
        "overlay_max": float(np.max(overlay)),
    }
    (RUNS / "credit_leadlag_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    _append_card(
        "Credit Lead-Lag Overlay ✔",
        (
            f"<p>rows={len(sig)}, eq={info['equity_symbol']}, "
            f"signal_mean={info['signal_mean']:.3f}, "
            f"overlay_range=[{info['overlay_min']:.3f},{info['overlay_max']:.3f}].</p>"
        ),
    )
    print(f"✅ Wrote {RUNS/'credit_leadlag_signal.csv'}")
    print(f"✅ Wrote {RUNS/'credit_leadlag_overlay.csv'}")
    print(f"✅ Wrote {RUNS/'credit_leadlag_info.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

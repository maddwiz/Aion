#!/usr/bin/env python3
# Per-hive walk-forward evaluator.
#
# Reads:
#   runs_plus/hive_signals.csv  (DATE,HIVE,hive_signal)
#   runs_plus/daily_returns.csv OR daily_returns.csv OR runs_plus/portfolio_plus.csv
#
# Writes:
#   runs_plus/hive_wf_metrics.csv
#   runs_plus/hive_wf_oos_returns.csv
#   runs_plus/hive_wf_summary.json

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def _annualized_sharpe(r):
    r = np.asarray(r, float).ravel()
    r = r[np.isfinite(r)]
    if r.size < 4:
        return 0.0
    mu = float(np.mean(r))
    sd = float(np.std(r) + 1e-12)
    return float((mu / sd) * np.sqrt(252.0))


def _max_dd(r):
    r = np.asarray(r, float).ravel()
    if r.size == 0:
        return 0.0
    eq = np.cumprod(1.0 + np.clip(r, -0.95, 0.95))
    peak = np.maximum.accumulate(eq)
    dd = eq / (peak + 1e-12) - 1.0
    return float(np.min(dd))


def _load_portfolio_returns():
    p = RUNS / "portfolio_plus.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
    except Exception:
        return None
    lowers = {c.lower(): c for c in df.columns}
    dcol = lowers.get("date") or lowers.get("timestamp")
    if dcol is None or dcol not in df.columns:
        return None
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).rename(columns={dcol: "DATE"}).sort_values("DATE")
    for c in ["ret_net", "ret", "ret_plus", "daily_ret", "portfolio_ret", "port_ret", "return"]:
        if c in df.columns:
            y = pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(-0.5, 0.5)
            return pd.DataFrame({"DATE": df["DATE"].dt.normalize(), "ret": y.values})
    return None


def _load_daily_returns():
    for rel in ["runs_plus/daily_returns.csv", "daily_returns.csv"]:
        p = ROOT / rel
        if not p.exists():
            continue
        try:
            a = np.loadtxt(p, delimiter=",").ravel()
        except Exception:
            try:
                a = np.loadtxt(p, delimiter=",", skiprows=1).ravel()
            except Exception:
                continue
        a = np.asarray(a, float).ravel()
        if len(a):
            return a
    return None

def _load_asset_returns_from_data():
    data_dir = ROOT / "data"
    if not data_dir.exists():
        return None
    frames = []
    for p in sorted(data_dir.glob("*.csv")):
        sym = p.stem.replace("_prices", "").upper().strip()
        if not sym:
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        dcol = None
        for c in ["date", "Date", "DATE", "timestamp", "Timestamp"]:
            if c in df.columns:
                dcol = c
                break
        if dcol is None:
            continue
        pcol = None
        for c in ["Adj Close", "adj_close", "AdjClose", "Close", "close", "price", "Price"]:
            if c in df.columns:
                pcol = c
                break
        if pcol is None:
            continue
        dd = pd.DataFrame({"DATE": pd.to_datetime(df[dcol], errors="coerce"), sym: pd.to_numeric(df[pcol], errors="coerce")})
        dd = dd.dropna(subset=["DATE"]).sort_values("DATE")
        dd[sym] = dd[sym].pct_change()
        dd["DATE"] = dd["DATE"].dt.normalize()
        frames.append(dd[["DATE", sym]])
    if not frames:
        return None
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="DATE", how="outer")
    out = out.sort_values("DATE").reset_index(drop=True)
    return out


def _append_card(title, html):
    for name in ["report_all.html", "report_best_plus.html", "report_plus.html", "report.html"]:
        p = ROOT / name
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        card = f'\n<div style="border:1px solid #ccc;padding:10px;margin:10px 0;"><h3>{title}</h3>{html}</div>\n'
        txt = txt.replace("</body>", card + "</body>") if "</body>" in txt else txt + card
        p.write_text(txt, encoding="utf-8")


def _wf_eval(signal, ret, train=126, test=21, step=21):
    s = np.asarray(signal, float).ravel()
    y = np.asarray(ret, float).ravel()
    T = min(len(s), len(y))
    s = s[:T]
    y = y[:T]
    if T < (train + test + 5):
        pos = np.tanh(0.6 * s)
        pnl = np.roll(pos, 1) * y
        pnl[0] = 0.0
        return pnl

    gross_grid = [0.4, 0.7, 1.0, 1.3]
    out = np.zeros(T, float)
    for start in range(train, T - test + 1, step):
        tr0 = 0
        tr1 = start
        te0 = start
        te1 = start + test

        s_tr = s[tr0:tr1]
        y_tr = y[tr0:tr1]
        if len(s_tr) < 20:
            continue

        best = (-1e9, 0.7)
        for g in gross_grid:
            pos_tr = np.tanh(g * s_tr)
            pnl_tr = np.roll(pos_tr, 1) * y_tr
            pnl_tr[0] = 0.0
            sh = _annualized_sharpe(pnl_tr)
            if sh > best[0]:
                best = (sh, g)

        g = best[1]
        s_te = s[te0:te1]
        y_te = y[te0:te1]
        pos_te = np.tanh(g * s_te)
        pnl_te = np.roll(pos_te, 1) * y_te
        if len(pnl_te):
            pnl_te[0] = 0.0
        out[te0:te1] = pnl_te
    return out


if __name__ == "__main__":
    p = RUNS / "hive_signals.csv"
    if not p.exists():
        raise SystemExit("Need runs_plus/hive_signals.csv (run tools/make_hive.py first)")

    h = pd.read_csv(p)
    need = {"DATE", "HIVE", "hive_signal"}
    if not need.issubset(h.columns):
        raise SystemExit("hive_signals.csv missing required columns: DATE,HIVE,hive_signal")
    h["DATE"] = pd.to_datetime(h["DATE"], errors="coerce")
    h = h.dropna(subset=["DATE"]).sort_values(["DATE", "HIVE"])
    dates = pd.Series(sorted(h["DATE"].dt.normalize().unique()))
    asset_ret = _load_asset_returns_from_data()
    hive_assets = None
    ha = RUNS / "hive_assets.csv"
    if ha.exists():
        try:
            tmp = pd.read_csv(ha)
            if {"ASSET", "HIVE"}.issubset(tmp.columns):
                hive_assets = tmp.copy()
                hive_assets["ASSET"] = hive_assets["ASSET"].astype(str).str.upper()
                hive_assets["HIVE"] = hive_assets["HIVE"].astype(str)
        except Exception:
            hive_assets = None

    pret = _load_portfolio_returns()
    if pret is not None:
        m = pd.DataFrame({"DATE": dates}).merge(pret, on="DATE", how="left").fillna(0.0)
        ret = m["ret"].values.astype(float)
    else:
        dr = _load_daily_returns()
        if dr is None:
            raise SystemExit("Need daily returns or portfolio_plus returns for hive walk-forward")
        # align by tail to hive timeline
        if len(dr) >= len(dates):
            ret = dr[-len(dates) :]
        else:
            ret = np.zeros(len(dates), float)
            ret[-len(dr) :] = dr

    rows = []
    oos_frames = []
    for hive, g in h.groupby("HIVE"):
        gg = g.copy()
        gg["DATE_DAY"] = gg["DATE"].dt.normalize()
        s = gg.groupby("DATE_DAY", as_index=False)["hive_signal"].mean().rename(columns={"DATE_DAY": "DATE"})
        s["DATE"] = pd.to_datetime(s["DATE"], errors="coerce")
        s = pd.DataFrame({"DATE": dates}).merge(s, on="DATE", how="left").fillna(0.0)
        sig = s["hive_signal"].values.astype(float)
        ret_h = ret
        ret_source = "portfolio_returns"
        if asset_ret is not None and hive_assets is not None:
            aset = hive_assets.loc[hive_assets["HIVE"] == str(hive), "ASSET"].astype(str).str.upper().tolist()
            aset = [a for a in aset if a in asset_ret.columns]
            if aset:
                rr = asset_ret[["DATE"] + aset].copy()
                rr["ret_h"] = rr[aset].mean(axis=1)
                rr = pd.DataFrame({"DATE": dates}).merge(rr[["DATE", "ret_h"]], on="DATE", how="left").fillna(0.0)
                ret_h = rr["ret_h"].values.astype(float)
                ret_source = f"asset_mean[{len(aset)}]"
        pnl = _wf_eval(sig, ret_h, train=126, test=21, step=21)
        lag_pos = np.roll(np.tanh(0.7 * sig), 1)
        lag_pos[0] = 0.0

        sh = _annualized_sharpe(pnl)
        hit = float(np.mean(np.sign(lag_pos) == np.sign(ret_h))) if len(ret_h) else 0.0
        mdd = _max_dd(pnl)
        ntr = int(np.sum(np.abs(np.diff(np.sign(lag_pos))) > 0.0))

        rows.append(
            {
                "HIVE": str(hive),
                "sharpe_oos": sh,
                "hit_rate": hit,
                "max_dd": mdd,
                "trades_proxy": ntr,
                "mean_pnl": float(np.mean(pnl)) if len(pnl) else 0.0,
                "ret_source": ret_source,
            }
        )
        oos_frames.append(pd.DataFrame({"DATE": dates, "HIVE": str(hive), "hive_oos_ret": pnl}))

    met = pd.DataFrame(rows).sort_values("sharpe_oos", ascending=False)
    met.to_csv(RUNS / "hive_wf_metrics.csv", index=False)
    if oos_frames:
        pd.concat(oos_frames, ignore_index=True).to_csv(RUNS / "hive_wf_oos_returns.csv", index=False)
    else:
        pd.DataFrame(columns=["DATE", "HIVE", "hive_oos_ret"]).to_csv(RUNS / "hive_wf_oos_returns.csv", index=False)

    summary = {
        "rows": int(len(met)),
        "best_hive": str(met.iloc[0]["HIVE"]) if len(met) else None,
        "best_sharpe": float(met.iloc[0]["sharpe_oos"]) if len(met) else None,
        "mean_sharpe": float(met["sharpe_oos"].mean()) if len(met) else None,
        "hives": met["HIVE"].tolist() if len(met) else [],
    }
    (RUNS / "hive_wf_summary.json").write_text(json.dumps(summary, indent=2))

    if len(met):
        top = met.head(5)
        txt = "<br>".join(
            f"{r.HIVE}: Sharpe {r.sharpe_oos:.3f}, Hit {r.hit_rate:.3f}, MaxDD {r.max_dd:.3f}"
            for _, r in top.iterrows()
        )
        _append_card("Hive Walk-Forward ✔", f"<p>{txt}</p>")

    print(f"✅ Wrote {RUNS/'hive_wf_metrics.csv'}")
    print(f"✅ Wrote {RUNS/'hive_wf_oos_returns.csv'}")
    print(f"✅ Wrote {RUNS/'hive_wf_summary.json'}")

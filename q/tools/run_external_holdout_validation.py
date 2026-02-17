#!/usr/bin/env python3
"""
External untouched holdout validation.

Evaluates frozen portfolio weights on an external holdout dataset that is not
used by the core training/tuning pipeline.

Inputs (priority):
  1) Q_EXTERNAL_HOLDOUT_RETURNS_FILE (CSV matrix of returns)
  2) Q_EXTERNAL_HOLDOUT_DIR (default: q/data_holdout/*.csv with Date+Close)

Writes:
  - runs_plus/external_holdout_returns.csv
  - runs_plus/external_holdout_validation.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from tools import make_daily_from_weights as mdw
from tools import run_strict_oos_validation as so

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


def _load_mat(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        arr = np.loadtxt(path, delimiter=",")
    except Exception:
        try:
            arr = np.loadtxt(path, delimiter=",", skiprows=1)
        except Exception:
            return None
    arr = np.asarray(arr, float)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0:
        return None
    return arr


def _read_price_series(path: Path) -> pd.Series | None:
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
    vcol = None
    for c in ["Close", "Adj Close", "close", "adj_close", "value", "Value", "PRICE", "price"]:
        if c in df.columns:
            vcol = c
            break
    if vcol is None:
        return None
    idx = pd.to_datetime(df[dcol], errors="coerce")
    vals = pd.to_numeric(df[vcol], errors="coerce")
    s = pd.Series(vals.values, index=idx).dropna()
    if s.empty:
        return None
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.astype(float)


def _load_asset_names(n: int) -> list[str]:
    p = RUNS / "asset_names.csv"
    if p.exists():
        try:
            df = pd.read_csv(p)
            if not df.empty:
                col = df.columns[0]
                vals = [str(x).strip().upper() for x in df[col].tolist() if str(x).strip()]
                if len(vals) >= n:
                    return vals[:n]
        except Exception:
            pass
    return [f"A{i:03d}" for i in range(n)]


def _load_frozen_weights() -> tuple[np.ndarray | None, str]:
    cands = [
        RUNS / "portfolio_weights_final.csv",
        RUNS / "weights_exec.csv",
        RUNS / "weights_asset_class_diversified.csv",
        RUNS / "portfolio_weights.csv",
    ]
    for p in cands:
        a = _load_mat(p)
        if a is None:
            continue
        row = np.asarray(a[-1], float).ravel()
        if row.size:
            return row, str(p)
    return None, ""


def _load_holdout_returns_from_dir(holdout_dir: Path, symbols: list[str]) -> tuple[np.ndarray | None, list[str], str]:
    if (not holdout_dir.exists()) or (not holdout_dir.is_dir()):
        return None, [], "missing_holdout_dir"
    frames = []
    used = []
    for sym in symbols:
        p = holdout_dir / f"{sym}.csv"
        if not p.exists():
            continue
        s = _read_price_series(p)
        if s is None:
            continue
        r = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if len(r) < 40:
            continue
        frames.append(r.rename(sym))
        used.append(sym)
    if not frames:
        return None, [], "no_symbol_overlap"
    df = pd.concat(frames, axis=1, join="inner").dropna(how="any")
    if df.empty:
        return None, [], "no_aligned_rows"
    return df.to_numpy(dtype=float), used, "ok"


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


def main() -> int:
    min_rows = int(np.clip(int(float(os.getenv("Q_EXTERNAL_HOLDOUT_MIN_ROWS", "126"))), 30, 100000))
    returns_file = str(os.getenv("Q_EXTERNAL_HOLDOUT_RETURNS_FILE", "")).strip()
    holdout_dir = Path(str(os.getenv("Q_EXTERNAL_HOLDOUT_DIR", str(ROOT / "data_holdout"))).strip())

    w_full, w_source = _load_frozen_weights()
    if w_full is None:
        out = {"ok": False, "reason": "missing_frozen_weights", "weights_source": w_source}
        (RUNS / "external_holdout_validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"✅ Wrote {RUNS/'external_holdout_validation.json'}")
        return 0

    names = _load_asset_names(len(w_full))
    ret = None
    used_symbols: list[str] = []
    source_kind = ""
    source_detail = ""

    if returns_file:
        p = Path(returns_file)
        mat = _load_mat(p)
        if mat is not None:
            ret = mat
            k = min(mat.shape[1], len(names))
            used_symbols = names[:k]
            source_kind = "returns_file"
            source_detail = str(p)
    if ret is None:
        mat, used_symbols, status = _load_holdout_returns_from_dir(holdout_dir, names)
        if mat is not None:
            ret = mat
            source_kind = "holdout_dir_prices"
            source_detail = str(holdout_dir)
        else:
            out = {
                "ok": False,
                "reason": status,
                "weights_source": w_source,
                "holdout_dir": str(holdout_dir),
                "returns_file": returns_file,
            }
            (RUNS / "external_holdout_validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(f"✅ Wrote {RUNS/'external_holdout_validation.json'}")
            return 0

    ret = np.asarray(ret, float)
    ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
    ret = np.clip(ret, -0.95, 0.95)
    t, n = ret.shape
    if t < min_rows:
        out = {
            "ok": False,
            "reason": f"rows<{min_rows}",
            "rows": int(t),
            "assets": int(n),
            "weights_source": w_source,
            "data_source": {"kind": source_kind, "detail": source_detail},
        }
        (RUNS / "external_holdout_validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"✅ Wrote {RUNS/'external_holdout_validation.json'}")
        return 0

    k = min(n, len(w_full), len(used_symbols))
    if k <= 0:
        out = {"ok": False, "reason": "no_asset_overlap", "weights_source": w_source}
        (RUNS / "external_holdout_validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"✅ Wrote {RUNS/'external_holdout_validation.json'}")
        return 0
    ret = ret[:, :k]
    used_symbols = used_symbols[:k]
    w = np.asarray(w_full[:k], float).ravel()
    full_gross = float(np.sum(np.abs(w_full)))
    sub_gross = float(np.sum(np.abs(w)))
    if sub_gross > 1e-9 and full_gross > 0.0:
        w = w * (full_gross / sub_gross)
    w_static = np.tile(w.reshape(1, -1), (t, 1))

    mdw.RUNS = RUNS
    cost_cfg = mdw.resolve_cost_params(runs_dir=RUNS)
    class_mult, class_info = mdw._class_cost_multipliers(k)
    net, gross, cost, turnover, eff_bps, cash_carry, cash_frac = mdw.build_costed_daily_returns(
        w_static,
        ret,
        base_bps=float(cost_cfg["base_bps"]),
        vol_scaled_bps=float(cost_cfg["vol_scaled_bps"]),
        vol_lookback=int(np.clip(int(float(os.getenv("Q_EXTERNAL_HOLDOUT_COST_VOL_LOOKBACK", "20"))), 2, 252)),
        vol_ref_daily=float(np.clip(float(os.getenv("Q_EXTERNAL_HOLDOUT_COST_VOL_REF_DAILY", "0.0063")), 1e-5, 0.25)),
        half_turnover=True,
        fixed_daily_fee=float(np.clip(float(os.getenv("Q_EXTERNAL_HOLDOUT_FIXED_DAILY_FEE", "0.0")), 0.0, 1.0)),
        cash_yield_annual=float(np.clip(float(os.getenv("Q_CASH_YIELD_ANNUAL", "0.0")), 0.0, 0.20)),
        cash_exposure_target=float(np.clip(float(os.getenv("Q_CASH_EXPOSURE_TARGET", "1.0")), 0.25, 5.0)),
        asset_cost_multipliers=class_mult,
    )

    np.savetxt(RUNS / "external_holdout_returns.csv", net, delimiter=",")
    np.savetxt(RUNS / "external_holdout_returns_gross.csv", gross, delimiter=",")
    np.savetxt(RUNS / "external_holdout_costs.csv", cost, delimiter=",")

    m = so._metrics(net)
    out = {
        "ok": True,
        "method": "external_untouched_frozen_weights",
        "weights_source": w_source,
        "rows": int(t),
        "assets": int(k),
        "symbols_used": used_symbols,
        "data_source": {"kind": source_kind, "detail": source_detail},
        "metrics_external_holdout_net": m,
        "cost_context": {
            "base_bps": float(cost_cfg["base_bps"]),
            "vol_scaled_bps": float(cost_cfg["vol_scaled_bps"]),
            "mean_cost_daily": float(np.mean(cost)),
            "ann_cost_estimate": float(np.mean(cost) * 252.0),
            "mean_turnover": float(np.mean(turnover)),
            "mean_effective_cost_bps": float(np.mean(eff_bps)),
            "mean_cash_fraction": float(np.mean(cash_frac)),
            "class_cost_model_enabled": bool(class_info.get("enabled", False)),
            "class_cost_classes": list(class_info.get("classes", [])),
        },
        "files": {
            "returns_net": str(RUNS / "external_holdout_returns.csv"),
            "returns_gross": str(RUNS / "external_holdout_returns_gross.csv"),
            "costs": str(RUNS / "external_holdout_costs.csv"),
        },
    }
    (RUNS / "external_holdout_validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    _append_card(
        "External Holdout Validation ✔",
        (
            f"<p>method=external_untouched_frozen_weights, rows={t}, assets={k}, "
            f"Sharpe={m['sharpe']:.3f}, Hit={m['hit_rate']:.3f}, MaxDD={m['max_drawdown']:.3f}.</p>"
        ),
    )
    print(f"✅ Wrote {RUNS/'external_holdout_returns.csv'}")
    print(f"✅ Wrote {RUNS/'external_holdout_validation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

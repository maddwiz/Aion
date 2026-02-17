#!/usr/bin/env python3
"""
Cross-asset diversification overlay.

Reads:
  - runs_plus/asset_returns.csv
  - runs_plus/asset_names.csv (preferred)
  - base weights: weights_tail_blend / weights_regime / portfolio_weights
  - optional runs_plus/cluster_map.csv (asset,cluster)
  - optional runs_plus/shock_mask.csv

Writes:
  - runs_plus/weights_asset_class_diversified.csv
  - runs_plus/asset_class_map_used.csv
  - runs_plus/asset_class_diversification_info.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(parents=True, exist_ok=True)


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
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _first_weights() -> tuple[np.ndarray | None, str]:
    cands = [
        RUNS / "weights_tail_blend.csv",
        RUNS / "weights_regime.csv",
        RUNS / "portfolio_weights.csv",
        ROOT / "portfolio_weights.csv",
    ]
    for p in cands:
        arr = _load_mat(p)
        if arr is not None:
            return arr, str(p.name)
    return None, ""


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


def _load_asset_names(n: int) -> list[str]:
    p = RUNS / "asset_names.csv"
    if p.exists():
        try:
            df = pd.read_csv(p)
            col = df.columns[0]
            vals = [str(x).strip().upper() for x in df[col].tolist()]
            vals = [x if x else f"A{i:03d}" for i, x in enumerate(vals)]
            if len(vals) >= n:
                return vals[:n]
        except Exception:
            pass
    return [f"A{i:03d}" for i in range(n)]


def _load_class_map(names: list[str]) -> list[str]:
    p = RUNS / "cluster_map.csv"
    mapping = {}
    if p.exists():
        try:
            df = pd.read_csv(p)
            lower = {str(c).strip().lower(): c for c in df.columns}
            a = lower.get("asset")
            c = lower.get("cluster")
            if a and c:
                for _, r in df[[a, c]].dropna().iterrows():
                    mapping[str(r[a]).strip().upper()] = str(r[c]).strip().upper()
        except Exception:
            mapping = {}
    out = []
    for nm in names:
        if nm in mapping and mapping[nm]:
            out.append(mapping[nm])
        else:
            out.append(_infer_class(nm))
    return out


def _load_shock(t: int) -> np.ndarray:
    p = RUNS / "shock_mask.csv"
    if not p.exists():
        return np.zeros(t, dtype=float)
    arr = _load_mat(p)
    if arr is None:
        return np.zeros(t, dtype=float)
    s = np.asarray(arr, float).ravel()
    if s.size < t:
        fill = float(s[-1]) if s.size else 0.0
        s = np.concatenate([s, np.full(t - s.size, fill, dtype=float)], axis=0)
    return np.clip(s[:t], 0.0, 1.0)


def _class_budgets_for_day(
    class_order: list[str],
    class_vol: dict[str, float],
    *,
    invvol_blend: float,
    shock: float,
    shock_tilt: float,
) -> dict[str, float]:
    k = max(1, len(class_order))
    eq = {c: 1.0 / k for c in class_order}
    inv = {}
    for c in class_order:
        v = max(1e-6, float(class_vol.get(c, 0.0)))
        inv[c] = 1.0 / v
    s_inv = max(1e-9, float(sum(inv.values())))
    inv_norm = {c: inv[c] / s_inv for c in class_order}
    b = {}
    a = float(np.clip(invvol_blend, 0.0, 1.0))
    for c in class_order:
        b[c] = (1.0 - a) * eq[c] + a * inv_norm[c]

    sh = float(np.clip(shock, 0.0, 1.0))
    tilt = float(np.clip(shock_tilt, 0.0, 1.0)) * sh
    if tilt > 0.0:
        risk_off = {"RATES", "COMMOD", "FX", "VOL"}
        risk_on = {"EQ", "CREDIT", "CRYPTO"}
        for c in class_order:
            if c in risk_off:
                b[c] *= 1.0 + 0.60 * tilt
            elif c in risk_on:
                b[c] *= max(0.60, 1.0 - 0.75 * tilt)
    s = max(1e-9, float(sum(max(0.0, x) for x in b.values())))
    for c in class_order:
        b[c] = max(0.0, b[c]) / s
    return b


def diversify_weights(
    w_in: np.ndarray,
    asset_returns: np.ndarray,
    classes: list[str],
    *,
    lookback: int,
    invvol_blend: float,
    class_mult_min: float,
    class_mult_max: float,
    shock_tilt: float,
) -> tuple[np.ndarray, dict]:
    w = np.asarray(w_in, float).copy()
    r = np.asarray(asset_returns, float)
    t = min(w.shape[0], r.shape[0])
    n = w.shape[1]
    w = w[:t, :n]
    r = r[:t, :n]
    if len(classes) != n:
        classes = (classes + ["EQ"] * n)[:n]
    class_order = sorted(set(str(c).upper() for c in classes))
    idx = {c: np.where(np.asarray(classes) == c)[0] for c in class_order}
    shock = _load_shock(t)

    w_out = w.copy()
    avg_mult = np.ones(t, dtype=float)
    class_gross_before = {c: [] for c in class_order}
    class_gross_after = {c: [] for c in class_order}
    for i in range(t):
        gross0 = float(np.sum(np.abs(w_out[i])))
        if gross0 <= 1e-9:
            continue
        lo = max(0, i - lookback + 1)
        seg = r[lo : i + 1]
        av = np.std(seg, axis=0)
        av = np.clip(np.nan_to_num(av, nan=0.0, posinf=0.0, neginf=0.0), 1e-6, 10.0)
        class_vol = {}
        class_now = {}
        for c in class_order:
            j = idx[c]
            if j.size == 0:
                continue
            class_vol[c] = float(np.mean(av[j]))
            class_now[c] = float(np.sum(np.abs(w_out[i, j])) / gross0)
            class_gross_before[c].append(class_now[c])

        target = _class_budgets_for_day(
            class_order,
            class_vol,
            invvol_blend=invvol_blend,
            shock=float(shock[i]),
            shock_tilt=shock_tilt,
        )

        mult = np.ones(n, dtype=float)
        for c in class_order:
            j = idx[c]
            cur = float(class_now.get(c, 0.0))
            tgt = float(target.get(c, 0.0))
            m = 1.0
            if cur > 1e-9 and tgt > 0.0:
                m = float(np.clip(tgt / cur, class_mult_min, class_mult_max))
            mult[j] = m
        w_i = w_out[i] * mult
        gross1 = float(np.sum(np.abs(w_i)))
        if gross1 > 1e-9:
            w_i *= gross0 / gross1
        w_out[i] = w_i
        avg_mult[i] = float(np.mean(np.abs(mult)))
        for c in class_order:
            j = idx[c]
            g = float(np.sum(np.abs(w_i[j])) / max(1e-9, float(np.sum(np.abs(w_i)))))
            class_gross_after[c].append(g)

    info = {
        "rows": int(t),
        "assets": int(n),
        "classes": class_order,
        "avg_abs_multiplier": float(np.mean(avg_mult)),
        "class_gross_before_mean": {c: float(np.mean(class_gross_before.get(c, [0.0]))) for c in class_order},
        "class_gross_after_mean": {c: float(np.mean(class_gross_after.get(c, [0.0]))) for c in class_order},
    }
    return w_out, info


def main() -> int:
    w, wsrc = _first_weights()
    a = _load_mat(RUNS / "asset_returns.csv")
    if w is None or a is None:
        print("(!) Missing base weights or asset_returns for asset-class diversification.")
        return 0
    if w.shape[1] != a.shape[1]:
        print(f"(!) Column mismatch weights N={w.shape[1]} vs returns N={a.shape[1]}; skipping.")
        return 0
    names = _load_asset_names(w.shape[1])
    classes = _load_class_map(names)

    lookback = int(np.clip(int(float(os.getenv("Q_CLASS_VOL_LOOKBACK", "63"))), 10, 252))
    invvol_blend = float(np.clip(float(os.getenv("Q_CLASS_INVVOL_BLEND", "0.65")), 0.0, 1.0))
    class_mult_min = float(np.clip(float(os.getenv("Q_CLASS_MULT_MIN", "0.55")), 0.1, 1.0))
    class_mult_max = float(np.clip(float(os.getenv("Q_CLASS_MULT_MAX", "1.80")), 1.0, 5.0))
    shock_tilt = float(np.clip(float(os.getenv("Q_CLASS_SHOCK_TILT", "0.45")), 0.0, 1.0))

    w_out, info = diversify_weights(
        w,
        a,
        classes,
        lookback=lookback,
        invvol_blend=invvol_blend,
        class_mult_min=class_mult_min,
        class_mult_max=class_mult_max,
        shock_tilt=shock_tilt,
    )

    np.savetxt(RUNS / "weights_asset_class_diversified.csv", w_out, delimiter=",")
    pd.DataFrame({"asset": names, "asset_class": classes}).to_csv(RUNS / "asset_class_map_used.csv", index=False)
    payload = {
        "ok": True,
        "weights_source": wsrc,
        "params": {
            "class_vol_lookback": lookback,
            "class_invvol_blend": invvol_blend,
            "class_mult_min": class_mult_min,
            "class_mult_max": class_mult_max,
            "class_shock_tilt": shock_tilt,
        },
        **info,
    }
    (RUNS / "asset_class_diversification_info.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _append_card(
        "Asset-Class Diversification ✔",
        (
            f"<p>Built <b>weights_asset_class_diversified.csv</b> from {wsrc}. "
            f"classes={len(info.get('classes', []))}, avg_abs_multiplier={float(info.get('avg_abs_multiplier', 1.0)):.3f}</p>"
        ),
    )
    print(f"✅ Wrote {RUNS/'weights_asset_class_diversified.csv'}")
    print(f"✅ Wrote {RUNS/'asset_class_diversification_info.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

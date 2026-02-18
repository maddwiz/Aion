#!/usr/bin/env python3
"""
Calendar mask governor.

Builds a time-varying scalar from validated calendar effects using expanding
walk-forward calibration on historical hit-rate behavior.

Writes:
  - runs_plus/calendar_mask_scalar.csv
  - runs_plus/calendar_mask_info.json
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

STATIC_SCALARS = {
    "turn_of_month": 1.10,
    "fomc_day_after": 1.08,
    "opex_preweek": 1.05,
    "pre_3day_weekend": 0.85,
    "jan_first_day": 0.80,
    "fomc_announcement_day": 0.90,
    "quad_witching": 0.88,
}


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


def _load_series(path: Path) -> np.ndarray | None:
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
    if a.ndim == 2:
        a = a[:, -1]
    a = a.ravel()
    if a.size == 0:
        return None
    return a


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


def _align_tail(v: np.ndarray, t: int, fill: float = 0.0) -> np.ndarray:
    x = np.asarray(v, float).ravel()
    if t <= 0:
        return np.zeros(0, dtype=float)
    if x.size >= t:
        return x[-t:]
    out = np.full(t, float(fill), dtype=float)
    if x.size > 0:
        out[-x.size :] = x
        out[: t - x.size] = float(x[0])
    return out


def _load_calendar_index(target_len: int) -> pd.DatetimeIndex:
    preferred = [DATA / "SPY.csv", DATA / "QQQ.csv"]
    files = [p for p in preferred if p.exists()]
    files.extend([p for p in sorted(DATA.glob("*.csv")) if p not in files])

    for p in files:
        try:
            df = pd.read_csv(p, usecols=lambda c: str(c).lower() in {"date", "timestamp"})
        except Exception:
            continue
        if df.empty:
            continue
        dcol = None
        for c in df.columns:
            if str(c).lower() in {"date", "timestamp"}:
                dcol = c
                break
        if dcol is None:
            continue
        idx = pd.to_datetime(df[dcol], errors="coerce").dropna()
        if len(idx) >= min(40, max(20, target_len)):
            idx = pd.DatetimeIndex(idx).sort_values().unique()
            if target_len > 0 and len(idx) >= target_len:
                return idx[-target_len:]
            return idx

    if target_len <= 0:
        return pd.DatetimeIndex([])
    end = pd.Timestamp.today().normalize()
    return pd.bdate_range(end=end, periods=target_len)


def _load_fomc_announcement_dates(path: Path) -> set[pd.Timestamp]:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
    except Exception:
        return set()
    if df.empty:
        return set()

    if "date" not in df.columns:
        return set()
    dt = pd.to_datetime(df["date"], errors="coerce").dropna().dt.normalize()
    if "type" in df.columns:
        types = df["type"].astype(str).str.strip().str.lower()
        mask = types.str.contains("announcement", na=False)
        if bool(mask.any()):
            dt = pd.to_datetime(df.loc[mask, "date"], errors="coerce").dropna().dt.normalize()
    return set(pd.Timestamp(x) for x in dt.tolist())


def _turn_of_month_mask(idx: pd.DatetimeIndex) -> np.ndarray:
    if len(idx) == 0:
        return np.zeros(0, dtype=bool)
    out = np.zeros(len(idx), dtype=bool)
    periods = idx.to_period("M")
    for m in periods.unique():
        loc = np.where(periods == m)[0]
        if loc.size == 0:
            continue
        n = int(loc.size)
        ranks = np.arange(1, n + 1)
        mask = (ranks <= 3) | (ranks >= max(1, n - 1))
        out[loc[mask]] = True
    return out


def _opex_dates(idx: pd.DatetimeIndex) -> dict[tuple[int, int], pd.Timestamp]:
    out: dict[tuple[int, int], pd.Timestamp] = {}
    if len(idx) == 0:
        return out
    months = sorted({(int(d.year), int(d.month)) for d in idx})
    for y, m in months:
        start = pd.Timestamp(year=y, month=m, day=1)
        end = start + pd.offsets.MonthEnd(0)
        fridays = pd.date_range(start, end, freq="W-FRI")
        if len(fridays) >= 3:
            out[(y, m)] = pd.Timestamp(fridays[2]).normalize()
    return out


def _calendar_feature_matrix(idx: pd.DatetimeIndex, fomc_ann_dates: set[pd.Timestamp]) -> dict[str, np.ndarray]:
    n = len(idx)
    feats = {k: np.zeros(n, dtype=bool) for k in STATIC_SCALARS.keys()}
    if n == 0:
        return feats

    # Turn-of-month.
    feats["turn_of_month"] = _turn_of_month_mask(idx)

    # First trading day of January.
    for y in sorted({int(d.year) for d in idx}):
        loc = np.where((idx.year == y) & (idx.month == 1))[0]
        if loc.size > 0:
            feats["jan_first_day"][int(loc[0])] = True

    # OPEX and quad witching markers.
    opex = _opex_dates(idx)
    idx_norm = pd.DatetimeIndex(idx.normalize())
    for i, d in enumerate(idx_norm):
        k = (int(d.year), int(d.month))
        ox = opex.get(k)
        if ox is None:
            continue
        days_to_opex = int((ox - d).days)
        if 1 <= days_to_opex <= 7:
            feats["opex_preweek"][i] = True
        if d == ox and int(d.month) in {3, 6, 9, 12}:
            feats["quad_witching"][i] = True

    # Thursday/Friday before a >=3 calendar-day market gap.
    for i in range(n - 1):
        d0 = idx[i]
        d1 = idx[i + 1]
        gap_days = int((d1.normalize() - d0.normalize()).days)
        if gap_days >= 3 and int(d0.weekday()) in {3, 4}:
            feats["pre_3day_weekend"][i] = True

    # FOMC announcement and day-after markers.
    if fomc_ann_dates:
        for i, d in enumerate(idx_norm):
            if d in fomc_ann_dates:
                feats["fomc_announcement_day"][i] = True
        for i, d in enumerate(idx_norm):
            if i == 0:
                continue
            if idx_norm[i - 1] in fomc_ann_dates:
                feats["fomc_day_after"][i] = True

    return feats


def compute_calendar_mask_scalar(
    hit: np.ndarray,
    features: dict[str, np.ndarray],
    *,
    beta: float,
    floor: float,
    ceil: float,
    min_feature_days: int,
) -> tuple[np.ndarray, dict[str, dict[str, float | int]]]:
    h = np.asarray(hit, float).ravel()
    t = len(h)
    s = np.ones(t, dtype=float)

    beta = float(np.clip(beta, 0.0, 2.0))
    floor = float(np.clip(floor, 0.4, 1.5))
    ceil = float(np.clip(ceil, floor, 1.8))
    min_feature_days = int(max(5, min_feature_days))

    feat_names = list(STATIC_SCALARS.keys())
    feat_mat = {}
    for name in feat_names:
        feat_mat[name] = np.asarray(features.get(name, np.zeros(t, dtype=bool)), bool)
        if feat_mat[name].size < t:
            padded = np.zeros(t, dtype=bool)
            padded[-feat_mat[name].size :] = feat_mat[name]
            feat_mat[name] = padded

    for i in range(t):
        active_vals = []
        for name in feat_names:
            if not bool(feat_mat[name][i]):
                continue
            static_v = float(STATIC_SCALARS[name])
            dynamic_v = static_v
            if i >= 10:
                hist_hits = h[:i]
                overall = float(np.mean(hist_hits)) if hist_hits.size else 0.5
                mask_hist = feat_mat[name][:i]
                if int(mask_hist.sum()) >= min_feature_days:
                    cond = float(np.mean(hist_hits[mask_hist]))
                    empirical = 1.0 + beta * (cond - overall)
                    dynamic_v = 0.5 * static_v + 0.5 * float(empirical)
            active_vals.append(float(np.clip(dynamic_v, floor, ceil)))

        if active_vals:
            s[i] = float(np.exp(np.mean(np.log(np.maximum(1e-9, np.asarray(active_vals, float))))))
        else:
            s[i] = 1.0

    s = np.clip(s, floor, ceil)

    stats = {}
    overall_full = float(np.mean(h)) if h.size else 0.5
    for name in feat_names:
        mask = feat_mat[name]
        n = int(mask.sum())
        cond = float(np.mean(h[mask])) if n > 0 else overall_full
        emp = float(1.0 + beta * (cond - overall_full))
        stats[name] = {
            "active_days": n,
            "overall_hit_rate": overall_full,
            "conditional_hit_rate": cond,
            "empirical_scalar": float(np.clip(emp, floor, ceil)),
            "static_scalar": float(STATIC_SCALARS[name]),
        }

    return s, stats


def _infer_hit_series(target_len: int) -> np.ndarray:
    dr = _load_series(RUNS / "daily_returns.csv")
    if dr is not None:
        dr = _align_tail(dr, target_len, 0.0)
        return (dr > 0.0).astype(float)

    w = _load_matrix(RUNS / "portfolio_weights_final.csv")
    ar = _load_matrix(RUNS / "asset_returns.csv")
    if (w is not None) and (ar is not None) and (w.shape[1] == ar.shape[1]):
        t = min(w.shape[0], ar.shape[0], target_len)
        pr = np.sum(w[:t] * ar[:t], axis=1)
        pr = _align_tail(pr, target_len, 0.0)
        return (pr > 0.0).astype(float)

    return np.full(target_len, 0.5, dtype=float)


def main() -> int:
    enabled = str(os.getenv("Q_CALENDAR_MASK_ENABLED", "1")).strip().lower() in {"1", "true", "yes", "on"}
    beta = float(np.clip(float(os.getenv("Q_CALENDAR_MASK_BETA", "0.6")), 0.0, 2.0))
    floor = float(np.clip(float(os.getenv("Q_CALENDAR_MASK_FLOOR", "0.75")), 0.4, 1.5))
    ceil = float(np.clip(float(os.getenv("Q_CALENDAR_MASK_CEIL", "1.15")), floor, 1.8))
    min_feature_days = int(np.clip(int(float(os.getenv("Q_CALENDAR_MASK_MIN_FEATURE_DAYS", "20"))), 5, 260))

    target_len = 0
    ar = _load_matrix(RUNS / "asset_returns.csv")
    if ar is not None:
        target_len = max(target_len, int(ar.shape[0]))
    dr = _load_series(RUNS / "daily_returns.csv")
    if dr is not None:
        target_len = max(target_len, int(dr.size))

    if target_len <= 0:
        print("(!) Missing target rows for calendar mask; skipping.")
        return 0

    idx = _load_calendar_index(target_len)
    if len(idx) != target_len:
        idx = _load_calendar_index(target_len)
        if len(idx) >= target_len:
            idx = idx[-target_len:]
        else:
            idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=target_len)

    if not enabled:
        scalar = np.ones(target_len, dtype=float)
        info = {
            "ok": True,
            "enabled": False,
            "rows": int(target_len),
            "params": {
                "beta": float(beta),
                "floor": float(floor),
                "ceil": float(ceil),
                "min_feature_days": int(min_feature_days),
            },
            "features": {},
            "mean_scalar": 1.0,
            "min_scalar": 1.0,
            "max_scalar": 1.0,
        }
    else:
        hit = _infer_hit_series(target_len)
        fomc_path = Path(os.getenv("Q_FOMC_DATES_FILE", str(DATA / "fomc_dates.csv")))
        fomc_dates = _load_fomc_announcement_dates(fomc_path)
        feats = _calendar_feature_matrix(idx, fomc_dates)
        scalar, feat_stats = compute_calendar_mask_scalar(
            hit,
            feats,
            beta=beta,
            floor=floor,
            ceil=ceil,
            min_feature_days=min_feature_days,
        )

        info = {
            "ok": True,
            "enabled": True,
            "rows": int(target_len),
            "fomc_file": str(fomc_path),
            "fomc_file_exists": bool(fomc_path.exists()),
            "fomc_announcement_count": int(len(fomc_dates)),
            "params": {
                "beta": float(beta),
                "floor": float(floor),
                "ceil": float(ceil),
                "min_feature_days": int(min_feature_days),
            },
            "features": feat_stats,
            "mean_scalar": float(np.mean(scalar)),
            "min_scalar": float(np.min(scalar)),
            "max_scalar": float(np.max(scalar)),
        }

    np.savetxt(RUNS / "calendar_mask_scalar.csv", np.asarray(scalar, float), delimiter=",")
    (RUNS / "calendar_mask_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    _append_card(
        "Calendar Mask Governor ✔",
        (
            f"<p>rows={int(target_len)}, beta={beta:.2f}, floor={floor:.2f}, ceil={ceil:.2f}.</p>"
            f"<p>mean={float(np.mean(scalar)):.3f}, range=[{float(np.min(scalar)):.3f}, {float(np.max(scalar)):.3f}], "
            f"fomc_file_exists={bool(info.get('fomc_file_exists', False))}.</p>"
        ),
    )

    print(f"✅ Wrote {RUNS/'calendar_mask_scalar.csv'}")
    print(f"✅ Wrote {RUNS/'calendar_mask_info.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

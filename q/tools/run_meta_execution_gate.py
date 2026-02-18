#!/usr/bin/env python3
"""
Meta-label execution gate (walk-forward ridge classifier).

Builds a probability that today's portfolio return will be positive, then maps
that probability to a smooth exposure gate.

Writes:
  - runs_plus/meta_execution_prob.csv
  - runs_plus/meta_execution_gate.csv
  - runs_plus/meta_execution_info.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_plus"
RUNS.mkdir(exist_ok=True)


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


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, float).ravel()
    y = np.asarray(b, float).ravel()
    n = min(len(x), len(y))
    if n < 5:
        return 0.0
    x = x[:n]
    y = y[:n]
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 5:
        return 0.0
    try:
        c = float(np.corrcoef(x[m], y[m])[0, 1])
        return c if np.isfinite(c) else 0.0
    except Exception:
        return 0.0


def _ann_sharpe(r: np.ndarray) -> float:
    x = np.asarray(r, float).ravel()
    m = np.isfinite(x)
    x = x[m]
    if x.size < 10:
        return 0.0
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=1) + 1e-12) if x.size >= 2 else 0.0
    return float((mu / sd) * np.sqrt(252.0))


def _max_dd(r: np.ndarray) -> float:
    x = np.asarray(r, float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    eq = np.cumsum(x)
    peak = np.maximum.accumulate(eq)
    return float(np.min(eq - peak))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, float)
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def _adaptive_threshold_series(
    returns: np.ndarray,
    *,
    base_threshold: float,
    enabled: bool,
    hit_window: int = 20,
    tighten_step: float = 0.04,
    loosen_step: float = 0.02,
    threshold_min: float = 0.42,
    threshold_max: float = 0.68,
    ema_alpha: float = 0.15,
) -> np.ndarray:
    r = np.asarray(returns, float).ravel()
    t = int(r.size)
    base = float(np.clip(base_threshold, threshold_min, threshold_max))
    out = np.full(t, base, dtype=float)
    if (not enabled) or t <= 0:
        return out

    w = int(max(5, hit_window))
    tstep = float(max(0.0, tighten_step))
    lstep = float(max(0.0, loosen_step))
    lo = float(min(threshold_min, threshold_max))
    hi = float(max(threshold_min, threshold_max))
    alpha = float(np.clip(ema_alpha, 0.01, 1.0))

    cur = base
    for i in range(t):
        i0 = max(0, i - w + 1)
        seg = r[i0 : i + 1]
        hit = float(np.mean(seg > 0.0)) if seg.size else 0.5
        raw = cur
        if hit < 0.45:
            raw = min(hi, raw + tstep)
        elif hit > 0.55:
            raw = max(lo, raw - lstep)
        cur = (alpha * raw) + ((1.0 - alpha) * cur)
        cur = float(np.clip(cur, lo, hi))
        out[i] = cur
    return out


def _assemble_features(returns: np.ndarray) -> tuple[np.ndarray, list[str]]:
    r = np.asarray(returns, float).ravel()
    T = len(r)
    cols = []
    names = []

    def add_feature(arr: np.ndarray | None, name: str, transform=None):
        if arr is None:
            return
        a = np.asarray(arr, float).ravel()
        if transform is not None:
            a = transform(a)
        if a.size == 0:
            return
        a = a[:T]
        if a.size < T:
            pad = np.full(T - a.size, a[-1] if a.size else 0.0, float)
            a = np.concatenate([a, pad], axis=0)
        cols.append(a.astype(float))
        names.append(name)

    add_feature(_load_series(RUNS / "meta_mix_confidence_calibrated.csv"), "meta_conf_cal")
    add_feature(_load_series(RUNS / "meta_mix_confidence_raw.csv"), "meta_conf_raw")
    add_feature(_load_series(RUNS / "meta_mix_reliability_governor.csv"), "meta_reliability")
    add_feature(_load_series(RUNS / "disagreement_gate.csv"), "disagreement_gate")
    add_feature(_load_series(RUNS / "global_governor.csv"), "global_governor")
    add_feature(_load_series(RUNS / "quality_governor.csv"), "quality_governor")
    add_feature(_load_series(RUNS / "regime_fracture_governor.csv"), "regime_fracture")
    add_feature(_load_series(RUNS / "heartbeat_exposure_scaler.csv"), "heartbeat_exposure")
    add_feature(_load_series(RUNS / "shock_mask.csv"), "shock_inverse", transform=lambda x: 1.0 - np.clip(x, 0.0, 1.0))

    # Always include basic return-state features.
    lag1 = np.zeros(T, float)
    lag1[1:] = r[:-1]
    cols.append(lag1)
    names.append("ret_lag1")

    absret = np.abs(r)
    roll = np.full(T, np.nan, float)
    w = 21
    for i in range(T):
        lo = max(0, i - w + 1)
        seg = absret[lo : i + 1]
        roll[i] = float(np.mean(seg)) if seg.size else 0.0
    roll = np.nan_to_num(roll, nan=float(np.nanmean(absret) if np.isfinite(np.nanmean(absret)) else 0.0))
    cols.append(roll)
    names.append("absret_roll21")

    if not cols:
        return np.zeros((T, 1), float), ["bias"]
    X = np.column_stack(cols).astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names


def _ridge_walkforward_prob(
    X: np.ndarray,
    y_bin: np.ndarray,
    *,
    min_train: int = 252,
    l2: float = 5.0,
    min_prob: float = 0.05,
    max_prob: float = 0.95,
) -> np.ndarray:
    X = np.asarray(X, float)
    y = np.asarray(y_bin, float).ravel()
    T = min(X.shape[0], y.size)
    X = X[:T]
    y = y[:T]
    K = X.shape[1]
    p = np.full(T, 0.5, float)
    mt = int(max(30, min_train))
    lam = float(max(1e-8, l2))
    eye = np.eye(K + 1, dtype=float)
    eye[0, 0] = 0.0  # don't regularize intercept

    for t in range(mt, T):
        Xt = X[:t]
        yt = y[:t]
        mu = np.mean(Xt, axis=0)
        sd = np.std(Xt, axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Z = (Xt - mu) / sd
        D = np.column_stack([np.ones(t, float), Z])
        yy = 2.0 * yt - 1.0
        lhs = D.T @ D + lam * eye
        rhs = D.T @ yy
        try:
            beta = np.linalg.solve(lhs, rhs)
        except Exception:
            beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        zt = (X[t] - mu) / sd
        st = float(np.dot(np.concatenate([[1.0], zt]), beta))
        train_scores = D @ beta
        scale = float(np.std(train_scores))
        if not np.isfinite(scale) or scale < 0.20:
            scale = 0.20
        pt = float(_sigmoid(np.array([st / scale]))[0])
        p[t] = float(np.clip(pt, min_prob, max_prob))
    return p


def _prob_to_gate(
    prob: np.ndarray,
    *,
    threshold: float | np.ndarray = 0.53,
    floor: float = 0.35,
    ceiling: float = 1.10,
    slope: float = 10.0,
    warmup: int = 252,
) -> np.ndarray:
    p = np.asarray(prob, float).ravel()
    thr = np.asarray(threshold, float)
    if thr.ndim == 0:
        thr = np.full(p.size, float(thr), dtype=float)
    else:
        thr = thr.ravel()
        if thr.size < p.size:
            pad = np.full(p.size - thr.size, float(thr[-1]) if thr.size else 0.53, dtype=float)
            thr = np.concatenate([thr, pad], axis=0)
        thr = thr[: p.size]
    thr = np.clip(thr, 0.30, 0.90)
    lo = float(np.clip(floor, 0.0, 1.2))
    hi = float(np.clip(ceiling, max(lo, 0.2), 1.5))
    k = float(np.clip(slope, 1.0, 40.0))
    raw = _sigmoid((p - thr) * k)
    g = lo + (hi - lo) * raw
    g = np.clip(g, lo, hi)
    w = int(max(0, warmup))
    if w > 0:
        g[: min(w, g.size)] = 1.0
    return g.astype(float)


def append_card(title: str, html: str) -> None:
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
    r = _load_series(RUNS / "daily_returns.csv")
    if r is None:
        print("(!) Missing runs_plus/daily_returns.csv; skipping meta execution gate.")
        return 0
    r = np.asarray(r, float).ravel()
    if r.size < 300:
        print("(!) Too few rows for meta execution gate; skipping.")
        return 0

    min_train = int(max(60, int(float(os.getenv("Q_META_EXEC_MIN_TRAIN", "252")))))
    l2 = float(max(1e-6, float(os.getenv("Q_META_EXEC_L2", "5.0"))))
    pmin = float(os.getenv("Q_META_EXEC_MIN_PROB_CLIP", "0.05"))
    pmax = float(os.getenv("Q_META_EXEC_MAX_PROB_CLIP", "0.95"))
    thr = float(os.getenv("Q_META_EXEC_MIN_PROB", "0.53"))
    floor = float(os.getenv("Q_META_EXEC_FLOOR", "0.35"))
    ceil = float(os.getenv("Q_META_EXEC_CEIL", "1.10"))
    slope = float(os.getenv("Q_META_EXEC_SLOPE", "10.0"))
    adaptive_enabled = str(os.getenv("Q_META_GATE_ADAPTIVE", "1")).strip().lower() in {"1", "true", "yes", "on"}
    adaptive_window = int(max(5, int(float(os.getenv("Q_META_GATE_HIT_RATE_WINDOW", "20")))))
    adaptive_tighten = float(max(0.0, float(os.getenv("Q_META_GATE_ADAPTIVE_TIGHTEN_STEP", "0.04"))))
    adaptive_loosen = float(max(0.0, float(os.getenv("Q_META_GATE_ADAPTIVE_LOOSEN_STEP", "0.02"))))
    adaptive_min = float(os.getenv("Q_META_GATE_ADAPTIVE_THRESHOLD_MIN", "0.42"))
    adaptive_max = float(os.getenv("Q_META_GATE_ADAPTIVE_THRESHOLD_MAX", "0.68"))
    adaptive_ema_alpha = float(np.clip(float(os.getenv("Q_META_GATE_ADAPTIVE_EMA_ALPHA", "0.15")), 0.01, 1.0))

    X, names = _assemble_features(r)
    T = min(len(r), X.shape[0])
    r = r[:T]
    X = X[:T]
    y = (r > 0.0).astype(float)
    prob = _ridge_walkforward_prob(X, y, min_train=min_train, l2=l2, min_prob=pmin, max_prob=pmax)
    threshold_series = _adaptive_threshold_series(
        r,
        base_threshold=thr,
        enabled=adaptive_enabled,
        hit_window=adaptive_window,
        tighten_step=adaptive_tighten,
        loosen_step=adaptive_loosen,
        threshold_min=adaptive_min,
        threshold_max=adaptive_max,
        ema_alpha=adaptive_ema_alpha,
    )
    gate = _prob_to_gate(prob, threshold=threshold_series, floor=floor, ceiling=ceil, slope=slope, warmup=min_train)

    np.savetxt(RUNS / "meta_execution_prob.csv", prob, delimiter=",")
    np.savetxt(RUNS / "meta_execution_gate.csv", gate, delimiter=",")
    np.savetxt(RUNS / "meta_execution_threshold.csv", threshold_series, delimiter=",")

    base_sh = _ann_sharpe(r)
    gate_sh = _ann_sharpe(r * gate)
    info = {
        "rows": int(T),
        "features": int(X.shape[1]),
        "feature_names": names,
        "min_train": int(min_train),
        "ridge_l2": float(l2),
        "prob_clip": [float(pmin), float(pmax)],
        "gate_threshold": float(thr),
        "adaptive_enabled": bool(adaptive_enabled),
        "adaptive_hit_rate_window": int(adaptive_window),
        "adaptive_tighten_step": float(adaptive_tighten),
        "adaptive_loosen_step": float(adaptive_loosen),
        "adaptive_threshold_min": float(min(adaptive_min, adaptive_max)),
        "adaptive_threshold_max": float(max(adaptive_min, adaptive_max)),
        "adaptive_ema_alpha": float(adaptive_ema_alpha),
        "adaptive_threshold_mean": float(np.mean(threshold_series)),
        "adaptive_threshold_min_realized": float(np.min(threshold_series)),
        "adaptive_threshold_max_realized": float(np.max(threshold_series)),
        "adaptive_threshold_series": threshold_series.tolist(),
        "adaptive_threshold_file": str(RUNS / "meta_execution_threshold.csv"),
        "gate_floor": float(floor),
        "gate_ceiling": float(ceil),
        "gate_slope": float(slope),
        "prob_mean": float(np.mean(prob)),
        "prob_std": float(np.std(prob)),
        "gate_mean": float(np.mean(gate)),
        "gate_min": float(np.min(gate)),
        "gate_max": float(np.max(gate)),
        "gate_corr_ret": float(_safe_corr(gate, r)),
        "base_sharpe": float(base_sh),
        "gated_sharpe": float(gate_sh),
        "base_max_dd": float(_max_dd(r)),
        "gated_max_dd": float(_max_dd(r * gate)),
    }
    (RUNS / "meta_execution_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    html = (
        f"<p>Meta execution gate rows={T}, features={X.shape[1]}, min_train={min_train}, l2={l2:.2f}.</p>"
        f"<p>Gate mean={info['gate_mean']:.3f} [{info['gate_min']:.3f}, {info['gate_max']:.3f}], "
        f"prob mean={info['prob_mean']:.3f}.</p>"
        f"<p>Base Sharpe={info['base_sharpe']:.3f} → Gated Sharpe={info['gated_sharpe']:.3f}; "
        f"MaxDD {info['base_max_dd']:.3f} → {info['gated_max_dd']:.3f}.</p>"
    )
    append_card("Meta Execution Gate ✔", html)

    print(f"✅ Wrote {RUNS/'meta_execution_prob.csv'}")
    print(f"✅ Wrote {RUNS/'meta_execution_gate.csv'}")
    print(f"✅ Wrote {RUNS/'meta_execution_threshold.csv'}")
    print(f"✅ Wrote {RUNS/'meta_execution_info.json'}")
    print(
        "Meta execution:",
        f"base_sh={base_sh:.3f}",
        f"gated_sh={gate_sh:.3f}",
        f"gate_mean={info['gate_mean']:.3f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

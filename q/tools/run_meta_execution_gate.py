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
    sd = float(np.std(x) + 1e-12)
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
    threshold: float = 0.53,
    floor: float = 0.35,
    ceiling: float = 1.10,
    slope: float = 10.0,
    warmup: int = 252,
) -> np.ndarray:
    p = np.asarray(prob, float).ravel()
    thr = float(np.clip(threshold, 0.45, 0.90))
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

    X, names = _assemble_features(r)
    T = min(len(r), X.shape[0])
    r = r[:T]
    X = X[:T]
    y = (r > 0.0).astype(float)
    prob = _ridge_walkforward_prob(X, y, min_train=min_train, l2=l2, min_prob=pmin, max_prob=pmax)
    gate = _prob_to_gate(prob, threshold=thr, floor=floor, ceiling=ceil, slope=slope, warmup=min_train)

    np.savetxt(RUNS / "meta_execution_prob.csv", prob, delimiter=",")
    np.savetxt(RUNS / "meta_execution_gate.csv", gate, delimiter=",")

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

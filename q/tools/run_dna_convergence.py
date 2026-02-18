#!/usr/bin/env python3
"""
Cross-sectional DNA convergence signal.

Builds per-asset signals from pairwise changes in spectral fingerprint similarity.
Also writes a council-member matrix so this alpha can vote directionally.

Writes:
  - runs_plus/dna_convergence_signal.csv     (T x N)
  - runs_plus/council_dna_convergence.csv    (T x N)
  - runs_plus/dna_convergence_info.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qengine.dna import dna_from_window

RUNS = ROOT / "runs_plus"
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
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def _load_asset_returns() -> np.ndarray | None:
    return _load_matrix(RUNS / "asset_returns.csv")


def _target_shape() -> tuple[int, int]:
    for p in [
        RUNS / "asset_returns.csv",
        RUNS / "portfolio_weights_final.csv",
        RUNS / "portfolio_weights.csv",
        ROOT / "portfolio_weights.csv",
    ]:
        m = _load_matrix(p)
        if m is not None:
            return int(m.shape[0]), int(m.shape[1])
    return 0, 0


def _load_asset_names(n: int) -> list[str]:
    p = RUNS / "asset_names.csv"
    names: list[str] = []
    if p.exists():
        try:
            df = pd.read_csv(p)
            if not df.empty:
                col = None
                for c in df.columns:
                    if str(c).strip().lower() in {"asset", "symbol", "ticker", "name"}:
                        col = c
                        break
                if col is None:
                    col = df.columns[0]
                names = [str(x).strip().upper() for x in df[col].tolist() if str(x).strip()]
        except Exception:
            names = []
    if n > 0:
        if len(names) < n:
            names = names + [f"ASSET_{i+1}" for i in range(len(names), n)]
        if len(names) > n:
            names = names[:n]
    return names


def _ema_2d(x: np.ndarray, alpha: float) -> np.ndarray:
    a = np.asarray(x, float)
    if a.ndim != 2 or a.size == 0:
        return np.asarray(a, float)
    alpha = float(np.clip(float(alpha), 0.0, 1.0))
    if alpha <= 0.0:
        return a
    out = a.copy()
    for j in range(out.shape[1]):
        for i in range(1, out.shape[0]):
            out[i, j] = alpha * out[i, j] + (1.0 - alpha) * out[i - 1, j]
    return out


def _fingerprint_tensor(returns: np.ndarray, window: int, topk: int) -> np.ndarray:
    r = np.asarray(returns, float)
    t, n = r.shape
    bins = int(window // 2 + 1)
    fp = np.zeros((t, n, bins), dtype=float)

    for j in range(n):
        x = np.nan_to_num(r[:, j], nan=0.0, posinf=0.0, neginf=0.0)
        for i in range(window - 1, t):
            idx, vals = dna_from_window(x[i - window + 1 : i + 1], topk=topk)
            if len(idx) == 0:
                continue
            vec = np.zeros(bins, dtype=float)
            for k, v in zip(idx, vals):
                kk = int(k)
                if 0 <= kk < bins:
                    vec[kk] = float(v)
            nrm = float(np.linalg.norm(vec))
            if nrm > 1e-12:
                vec /= nrm
            fp[i, j, :] = vec

    return fp


def compute_dna_convergence_signal(
    asset_returns: np.ndarray,
    *,
    window: int = 64,
    topk: int = 16,
    delta_lookback: int = 21,
    smooth_alpha: float = 0.18,
    divergence_threshold: float = 0.05,
) -> tuple[np.ndarray, dict]:
    r = np.asarray(asset_returns, float)
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

    t, n = r.shape
    info = {
        "rows": int(t),
        "cols": int(n),
        "window": int(window),
        "topk": int(topk),
        "delta_lookback": int(delta_lookback),
        "smooth_alpha": float(smooth_alpha),
    }
    if t <= 0 or n <= 0:
        info.update({"ok": False, "reason": "empty_input"})
        return np.zeros((max(0, t), max(0, n)), dtype=float), info

    window = int(np.clip(int(window), 16, max(16, t)))
    topk = int(np.clip(int(topk), 4, 128))
    delta_lookback = int(np.clip(int(delta_lookback), 3, max(3, t - 1)))
    smooth_alpha = float(np.clip(float(smooth_alpha), 0.0, 0.95))
    divergence_threshold = float(np.clip(float(divergence_threshold), 1e-6, 1.0))

    if n < 2 or t < max(window + 2, delta_lookback + 2):
        info.update(
            {
                "ok": False,
                "reason": "insufficient_shape",
                "valid_days": 0,
                "divergence_rate": 0.0,
                "mean_signal": 0.0,
                "std_signal": 0.0,
                "max_abs_signal": 0.0,
                "top_pairs": [],
            }
        )
        return np.zeros((t, n), dtype=float), info

    fp = _fingerprint_tensor(r, window=window, topk=topk)
    sims = np.full((t, n, n), np.nan, dtype=float)
    for i in range(window - 1, t):
        f = fp[i, :, :]
        if not np.isfinite(f).any():
            continue
        sims[i, :, :] = np.clip(f @ f.T, -1.0, 1.0)

    rel = np.zeros((t, n), dtype=float)
    for i in range(delta_lookback, t):
        rel[i, :] = np.sum(r[i - delta_lookback + 1 : i + 1, :], axis=0)

    raw = np.zeros((t, n), dtype=float)
    sum_abs_ds = np.zeros((n, n), dtype=float)
    cnt_ds = np.zeros((n, n), dtype=float)
    divergence_hits = 0
    divergence_total = 0
    valid_days = 0

    for i in range(max(window - 1 + delta_lookback, delta_lookback), t):
        s_now = sims[i, :, :]
        s_prev = sims[i - delta_lookback, :, :]
        if not np.isfinite(s_now).all() or not np.isfinite(s_prev).all():
            continue

        ds = s_now - s_prev
        sign_diff = np.sign(rel[i, :].reshape(1, -1) - rel[i, :].reshape(-1, 1))
        m = ds * sign_diff
        np.fill_diagonal(m, np.nan)

        day = np.nanmean(m, axis=1)
        day = np.nan_to_num(day, nan=0.0, posinf=0.0, neginf=0.0)

        mu = float(np.mean(day))
        sd = float(np.std(day, ddof=1)) if n > 1 else 0.0
        z = (day - mu) / (sd + 1e-12)
        raw[i, :] = np.tanh(z)

        abs_ds = np.abs(ds)
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        sum_abs_ds[mask] += abs_ds[mask]
        cnt_ds[mask] += 1.0

        divergence_hits += int(np.count_nonzero((ds < -divergence_threshold) & mask))
        divergence_total += int(np.count_nonzero(mask))
        valid_days += 1

    sig = _ema_2d(raw, alpha=smooth_alpha)
    sig = np.clip(sig, -1.0, 1.0)

    mean_abs_ds = np.divide(
        sum_abs_ds,
        np.maximum(cnt_ds, 1.0),
        out=np.zeros_like(sum_abs_ds),
        where=cnt_ds > 0,
    )
    top_pairs = []
    if n >= 2:
        tri = []
        for a in range(n):
            for b in range(a + 1, n):
                tri.append((float(mean_abs_ds[a, b]), a, b))
        tri.sort(key=lambda x: x[0], reverse=True)
        for val, a, b in tri[:10]:
            top_pairs.append({"i": int(a), "j": int(b), "mean_abs_delta_similarity": float(val)})

    info.update(
        {
            "ok": True,
            "valid_days": int(valid_days),
            "divergence_rate": float(divergence_hits / max(1, divergence_total)),
            "mean_signal": float(np.mean(sig)),
            "std_signal": float(np.std(sig, ddof=1)) if sig.size > 1 else 0.0,
            "max_abs_signal": float(np.max(np.abs(sig))) if sig.size else 0.0,
            "top_pairs": top_pairs,
        }
    )
    return sig, info


def main() -> int:
    arr = _load_asset_returns()
    t, n = _target_shape()

    if arr is None and t <= 0:
        print("(!) Missing inputs for DNA convergence; skipping.")
        return 0

    if arr is None:
        sig = np.zeros((t, n), dtype=float)
        info = {
            "ok": False,
            "reason": "missing_asset_returns",
            "rows": int(t),
            "cols": int(n),
            "valid_days": 0,
            "divergence_rate": 0.0,
            "mean_signal": 0.0,
            "std_signal": 0.0,
            "max_abs_signal": 0.0,
            "top_pairs": [],
        }
    else:
        window = int(np.clip(int(float(os.getenv("Q_DNA_CONVERGENCE_WINDOW", "64"))), 16, 252))
        topk = int(np.clip(int(float(os.getenv("Q_DNA_CONVERGENCE_TOPK", "16"))), 4, 128))
        delta = int(np.clip(int(float(os.getenv("Q_DNA_CONVERGENCE_LOOKBACK", "21"))), 3, 126))
        smooth_alpha = float(np.clip(float(os.getenv("Q_DNA_CONVERGENCE_SMOOTH_ALPHA", "0.18")), 0.0, 0.95))
        div_thr = float(np.clip(float(os.getenv("Q_DNA_CONVERGENCE_DIVERGENCE_THRESHOLD", "0.05")), 1e-6, 1.0))

        sig, info = compute_dna_convergence_signal(
            arr,
            window=window,
            topk=topk,
            delta_lookback=delta,
            smooth_alpha=smooth_alpha,
            divergence_threshold=div_thr,
        )

    # Align to discovered target shape if needed.
    if sig.shape != (t, n) and t > 0 and n > 0:
        out = np.zeros((t, n), dtype=float)
        tt = min(t, sig.shape[0])
        nn = min(n, sig.shape[1])
        out[-tt:, :nn] = sig[-tt:, :nn]
        sig = out

    names = _load_asset_names(sig.shape[1] if sig.ndim == 2 else 0)
    if names and info.get("top_pairs"):
        remapped = []
        for row in info["top_pairs"]:
            a = int(row.get("i", -1))
            b = int(row.get("j", -1))
            if 0 <= a < len(names) and 0 <= b < len(names):
                remapped.append(
                    {
                        "pair": f"{names[a]}|{names[b]}",
                        "mean_abs_delta_similarity": float(row.get("mean_abs_delta_similarity", 0.0)),
                    }
                )
        info["top_pairs"] = remapped

    np.savetxt(RUNS / "dna_convergence_signal.csv", np.asarray(sig, float), delimiter=",")
    np.savetxt(RUNS / "council_dna_convergence.csv", np.asarray(sig, float), delimiter=",")
    (RUNS / "dna_convergence_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    _append_card(
        "DNA Convergence Signal ✔",
        (
            f"<p>rows={sig.shape[0]}, cols={sig.shape[1]}, valid_days={info.get('valid_days', 0)}.</p>"
            f"<p>mean={info.get('mean_signal', 0.0):.3f}, std={info.get('std_signal', 0.0):.3f}, "
            f"divergence_rate={info.get('divergence_rate', 0.0):.3f}.</p>"
        ),
    )

    print(f"✅ Wrote {RUNS/'dna_convergence_signal.csv'}")
    print(f"✅ Wrote {RUNS/'council_dna_convergence.csv'}")
    print(f"✅ Wrote {RUNS/'dna_convergence_info.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

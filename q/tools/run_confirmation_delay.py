#!/usr/bin/env python3
"""
Confirmation-delay governor.

Delays exposure after directional signal flips, then ramps back toward 1.0.
Optionally fast-confirms when a portfolio proxy breaks above/below a trailing
range in the new direction.

Writes:
  - runs_plus/confirmation_delay_scalar.csv
  - runs_plus/confirmation_delay_info.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
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
    return a


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


def _align_tail(v: np.ndarray, t: int, fill: float) -> np.ndarray:
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


def _params_from_env() -> dict[str, float | int | bool]:
    floor = float(np.clip(float(os.getenv("Q_CONFIRMATION_FLOOR", "0.15")), 0.0, 0.50))
    ramp_days = float(np.clip(float(os.getenv("Q_CONFIRMATION_RAMP_DAYS", "2.0")), 0.5, 10.0))
    fast_confirm = str(os.getenv("Q_CONFIRMATION_FAST_CONFIRM", "1")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    lookback = int(np.clip(int(float(os.getenv("Q_CONFIRMATION_LOOKBACK", "5"))), 2, 30))
    mag_change = float(np.clip(float(os.getenv("Q_CONFIRMATION_MAG_CHANGE", "0.20")), 0.01, 2.0))
    min_exposure = float(np.clip(float(os.getenv("Q_CONFIRMATION_MIN_EXPOSURE", "0.02")), 0.0, 1.0))
    return {
        "floor": floor,
        "ramp_days": ramp_days,
        "fast_confirm": fast_confirm,
        "lookback": lookback,
        "mag_change": mag_change,
        "min_exposure": min_exposure,
    }


def _build_fast_confirm_flags(proxy_returns: np.ndarray | None, lookback: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    if proxy_returns is None:
        return None, None
    r = np.asarray(proxy_returns, float).ravel()
    if r.size == 0:
        return None, None

    eq = np.cumprod(1.0 + np.clip(r, -0.95, 10.0))
    s = pd.Series(eq, dtype=float)
    hi = s.rolling(int(max(2, lookback)), min_periods=int(max(2, lookback))).max().shift(1)
    lo = s.rolling(int(max(2, lookback)), min_periods=int(max(2, lookback))).min().shift(1)

    fast_long = (s > hi).fillna(False).to_numpy(dtype=bool)
    fast_short = (s < lo).fillna(False).to_numpy(dtype=bool)
    return fast_long, fast_short


def compute_confirmation_delay_scalar(
    weights: np.ndarray,
    *,
    floor: float,
    ramp_days: float,
    fast_confirm: bool,
    lookback: int,
    mag_change: float,
    min_exposure: float,
    proxy_returns: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    w = np.asarray(weights, float)
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    t = int(w.shape[0])
    if t <= 0:
        return np.zeros(0, dtype=float), {
            "rows": 0,
            "signal_reset_days": 0,
            "fast_confirm_days": 0,
            "flat_direction_days": 0,
            "mean_scalar": 1.0,
            "pct_days_at_floor": 0.0,
            "pct_fast_confirmed": 0.0,
            "fast_confirm_available": False,
        }

    floor = float(np.clip(float(floor), 0.0, 0.50))
    ramp_days = float(np.clip(float(ramp_days), 0.5, 10.0))
    lookback = int(np.clip(int(lookback), 2, 30))
    mag_change = float(np.clip(float(mag_change), 0.01, 2.0))
    min_exposure = float(np.clip(float(min_exposure), 0.0, 1.0))

    net = np.sum(w, axis=1)
    direction = np.sign(net)

    resets = np.zeros(t, dtype=bool)
    for i in range(1, t):
        sign_flip = (direction[i] != direction[i - 1]) and (direction[i] != 0 or direction[i - 1] != 0)
        mag_reset = (
            abs(float(net[i] - net[i - 1])) >= mag_change
            and abs(float(net[i])) >= min_exposure
            and direction[i] != 0
        )
        if sign_flip or mag_reset:
            resets[i] = True

    fast_long, fast_short = (None, None)
    if fast_confirm:
        fast_long, fast_short = _build_fast_confirm_flags(proxy_returns, lookback)
        if fast_long is not None:
            fast_long = _align_tail(fast_long.astype(float), t, 0.0).astype(bool)
            fast_short = _align_tail(fast_short.astype(float), t, 0.0).astype(bool)

    scalar = np.ones(t, dtype=float)
    active = False
    active_dir = 0.0
    day_count = 0
    fast_hits = 0

    for i in range(t):
        d = float(direction[i])

        if resets[i]:
            scalar[i] = floor
            active = d != 0.0
            active_dir = d
            day_count = 1
            continue

        if not active:
            scalar[i] = 1.0
            continue

        if d == 0.0 or d != active_dir:
            active = False
            active_dir = 0.0
            day_count = 0
            scalar[i] = 1.0
            continue

        if fast_confirm and fast_long is not None:
            if (active_dir > 0 and bool(fast_long[i])) or (active_dir < 0 and bool(fast_short[i])):
                scalar[i] = 1.0
                active = False
                active_dir = 0.0
                day_count = 0
                fast_hits += 1
                continue

        ramp = floor + (1.0 - floor) * (1.0 - np.exp(-float(day_count) / ramp_days))
        scalar[i] = float(np.clip(ramp, floor, 1.0))
        day_count += 1

    floor_mask = np.isclose(scalar, floor, atol=1e-12)
    n_resets = int(np.count_nonzero(resets))
    info = {
        "rows": int(t),
        "signal_reset_days": n_resets,
        "fast_confirm_days": int(fast_hits),
        "flat_direction_days": int(np.count_nonzero(direction == 0.0)),
        "mean_scalar": float(np.mean(scalar)) if scalar.size else 1.0,
        "pct_days_at_floor": float(np.mean(floor_mask.astype(float))) if scalar.size else 0.0,
        "pct_fast_confirmed": float(fast_hits / max(1, n_resets)),
        "fast_confirm_available": bool(fast_long is not None),
        "net_exposure_mean": float(np.mean(net)) if net.size else 0.0,
        "net_exposure_std": float(np.std(net)) if net.size else 0.0,
    }
    return np.clip(scalar, 0.0, 1.0), info


def _first_existing_matrix(paths: list[Path]) -> tuple[np.ndarray | None, str | None]:
    for p in paths:
        m = _load_matrix(p)
        if m is not None:
            return m, str(p.relative_to(ROOT))
    return None, None


def main() -> int:
    params = _params_from_env()

    weights, source = _first_existing_matrix(
        [
            RUNS / "portfolio_weights_final.csv",
            RUNS / "weights_turnover_budget_governed.csv",
            RUNS / "weights_turnover_governed.csv",
            RUNS / "weights_regime.csv",
            RUNS / "portfolio_weights.csv",
            ROOT / "portfolio_weights.csv",
        ]
    )

    # Best effort proxy returns for fast-confirm detection.
    daily_ret = _load_series(RUNS / "daily_returns.csv")
    proxy_ret = daily_ret

    asset_ret = _load_matrix(RUNS / "asset_returns.csv")
    if (proxy_ret is None) and (weights is not None) and (asset_ret is not None) and (asset_ret.shape[1] == weights.shape[1]):
        t = min(weights.shape[0], asset_ret.shape[0])
        proxy_ret = np.sum(weights[:t] * asset_ret[:t], axis=1)

    target_len = 0
    if weights is not None:
        target_len = max(target_len, int(weights.shape[0]))
    if asset_ret is not None:
        target_len = max(target_len, int(asset_ret.shape[0]))
    if proxy_ret is not None:
        target_len = max(target_len, int(np.asarray(proxy_ret).size))

    info = {
        "ok": True,
        "source_weights": source,
        "fallback_reason": None,
        "params": params,
    }

    if (weights is None) or target_len <= 0:
        if target_len <= 0:
            print("(!) No usable inputs for confirmation delay; skipping.")
            return 0
        scalar = np.ones(target_len, dtype=float)
        info.update(
            {
                "ok": False,
                "fallback_reason": "missing_weights",
                "rows": int(target_len),
                "mean_scalar": 1.0,
                "pct_days_at_floor": 0.0,
                "pct_fast_confirmed": 0.0,
            }
        )
    else:
        if proxy_ret is not None:
            proxy_ret = _align_tail(proxy_ret, int(weights.shape[0]), 0.0)
        scalar, stats = compute_confirmation_delay_scalar(
            weights,
            floor=float(params["floor"]),
            ramp_days=float(params["ramp_days"]),
            fast_confirm=bool(params["fast_confirm"]),
            lookback=int(params["lookback"]),
            mag_change=float(params["mag_change"]),
            min_exposure=float(params["min_exposure"]),
            proxy_returns=proxy_ret,
        )
        if scalar.size < target_len:
            scalar = _align_tail(scalar, target_len, 1.0)
        info.update(stats)

    np.savetxt(RUNS / "confirmation_delay_scalar.csv", np.asarray(scalar, float), delimiter=",")
    (RUNS / "confirmation_delay_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    _append_card(
        "Confirmation Delay Governor ✔",
        (
            f"<p>rows={int(np.asarray(scalar).size)}, floor={float(params['floor']):.2f}, "
            f"ramp_days={float(params['ramp_days']):.2f}, fast_confirm={bool(params['fast_confirm'])}.</p>"
            f"<p>mean_scalar={float(np.mean(scalar)):.3f}, "
            f"pct_floor={float(info.get('pct_days_at_floor', 0.0)):.2%}, "
            f"pct_fast_confirmed={float(info.get('pct_fast_confirmed', 0.0)):.2%}.</p>"
        ),
    )

    print(f"✅ Wrote {RUNS/'confirmation_delay_scalar.csv'}")
    print(f"✅ Wrote {RUNS/'confirmation_delay_info.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Microstructure council member.

Converts the signed market microstructure signal into a T x N matrix so it can
vote directionally inside the council stack.

Writes:
  - runs_plus/council_microstructure.csv
  - runs_plus/council_microstructure_info.json
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
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def _load_series(path: Path) -> np.ndarray | None:
    m = _load_matrix(path)
    if m is None:
        return None
    if m.shape[1] == 1:
        return m.ravel()
    return np.nanmean(m, axis=1)


def _split_syms(raw: str) -> set[str]:
    out = set()
    for token in str(raw).replace(";", ",").split(","):
        s = str(token).strip().upper()
        if s:
            out.add(s)
    return out


def _env_syms(key: str, default: set[str]) -> set[str]:
    raw = str(os.getenv(key, "")).strip()
    return _split_syms(raw) if raw else set(default)


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


def _load_asset_names(path: Path, n: int) -> list[str]:
    names: list[str] = []
    if path.exists():
        try:
            df = pd.read_csv(path)
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
    if n <= 0:
        return names
    if len(names) < n:
        names = names + [f"ASSET_{i+1}" for i in range(len(names), n)]
    if len(names) > n:
        names = names[:n]
    return names


def _align_tail(v: np.ndarray, t: int, fill: float = 0.0) -> np.ndarray:
    x = np.asarray(v, float).ravel()
    if t <= 0:
        return x
    if x.size >= t:
        return x[-t:]
    out = np.full(t, float(fill), dtype=float)
    if x.size > 0:
        out[-x.size :] = x
    return out


def build_microstructure_council_member(
    signal: np.ndarray,
    asset_names: list[str],
    *,
    direct_symbols: set[str] | None = None,
    inverse_symbols: set[str] | None = None,
    attenuated_symbols: set[str] | None = None,
    attenuation: float = 0.30,
    default_gain: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    sig = np.asarray(signal, float).ravel()
    names = [str(x).strip().upper() for x in asset_names]

    direct = set(direct_symbols or set())
    inverse = set(inverse_symbols or set())
    atten = set(attenuated_symbols or set())

    attenuation = float(np.clip(float(attenuation), -1.0, 1.0))
    default_gain = float(np.clip(float(default_gain), -1.0, 1.0))

    gains = np.full(len(names), default_gain, dtype=float)
    for i, sym in enumerate(names):
        if sym in inverse:
            gains[i] = -1.0
        elif sym in atten:
            gains[i] = attenuation
        elif sym in direct:
            gains[i] = 1.0

    mat = np.clip(sig.reshape(-1, 1) * gains.reshape(1, -1), -1.0, 1.0)
    return mat, gains


def main() -> int:
    t, n = _target_shape()
    sig_raw = _load_series(RUNS / "microstructure_signal.csv")

    if t <= 0 and sig_raw is not None:
        t = int(len(sig_raw))

    names = _load_asset_names(RUNS / "asset_names.csv", n)
    if n <= 0:
        n = len(names)

    if t <= 0 or n <= 0:
        print("(!) Missing target shape for council microstructure member; skipping.")
        return 0

    if not names:
        names = [f"ASSET_{i+1}" for i in range(n)]

    if sig_raw is None:
        signal = np.zeros(t, dtype=float)
        missing_signal = True
    else:
        signal = np.clip(_align_tail(sig_raw, t, fill=0.0), -1.0, 1.0)
        missing_signal = False

    direct = _env_syms(
        "Q_MICRO_MEMBER_DIRECT_SYMBOLS",
        {
            "SPY",
            "QQQ",
            "DIA",
            "IWM",
            "AAPL",
            "MSFT",
            "NVDA",
            "AMZN",
            "GOOGL",
            "META",
            "TSLA",
            "AMD",
            "NFLX",
            "HYG",
            "LQD",
            "JNK",
        },
    )
    inverse = _env_syms(
        "Q_MICRO_MEMBER_INVERSE_SYMBOLS",
        {"TLT", "IEF", "IEI", "VGSH", "SHY", "BIL", "GOVT"},
    )
    attenuated = _env_syms(
        "Q_MICRO_MEMBER_ATTENUATED_SYMBOLS",
        {"GLD", "SLV", "UUP", "FXY"},
    )
    attenuation = float(np.clip(float(os.getenv("Q_MICRO_MEMBER_ATTENUATION", "0.30")), -1.0, 1.0))
    default_gain = float(np.clip(float(os.getenv("Q_MICRO_MEMBER_DEFAULT_GAIN", "1.0")), -1.0, 1.0))

    mat, gains = build_microstructure_council_member(
        signal,
        names,
        direct_symbols=direct,
        inverse_symbols=inverse,
        attenuated_symbols=attenuated,
        attenuation=attenuation,
        default_gain=default_gain,
    )

    np.savetxt(RUNS / "council_microstructure.csv", np.asarray(mat, float), delimiter=",")

    info = {
        "ok": True,
        "rows": int(mat.shape[0]),
        "cols": int(mat.shape[1]),
        "signal_missing": bool(missing_signal),
        "signal_mean": float(np.mean(signal)),
        "signal_min": float(np.min(signal)),
        "signal_max": float(np.max(signal)),
        "matrix_mean": float(np.mean(mat)),
        "matrix_min": float(np.min(mat)),
        "matrix_max": float(np.max(mat)),
        "params": {
            "attenuation": float(attenuation),
            "default_gain": float(default_gain),
            "direct_symbols": int(len(direct)),
            "inverse_symbols": int(len(inverse)),
            "attenuated_symbols": int(len(attenuated)),
        },
        "class_counts": {
            "inverse": int(np.sum(np.isclose(gains, -1.0))),
            "attenuated": int(np.sum(np.isclose(gains, attenuation))),
            "direct_or_default": int(np.sum(~np.isclose(gains, -1.0) & ~np.isclose(gains, attenuation))),
        },
    }
    (RUNS / "council_microstructure_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    _append_card(
        "Council Member: Microstructure ✔",
        (
            f"<p>rows={mat.shape[0]}, cols={mat.shape[1]}, signal_missing={missing_signal}.</p>"
            f"<p>matrix_mean={info['matrix_mean']:.3f}, range=[{info['matrix_min']:.3f},{info['matrix_max']:.3f}]</p>"
        ),
    )

    print(f"✅ Wrote {RUNS/'council_microstructure.csv'}")
    print(f"✅ Wrote {RUNS/'council_microstructure_info.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

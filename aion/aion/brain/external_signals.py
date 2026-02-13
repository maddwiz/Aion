from __future__ import annotations

import json
import math
from pathlib import Path


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _normalize_signal(payload: dict, min_confidence: float, max_bias: float):
    bias = _safe_float(payload.get("bias"), 0.0)
    conf = _safe_float(payload.get("confidence"), 0.0)
    conf = _clamp(conf, 0.0, 1.0)
    if conf < float(min_confidence):
        return None
    bias = _clamp(bias, -abs(float(max_bias)), abs(float(max_bias)))
    return {"bias": bias, "confidence": conf}


def blend_external_signals(primary: dict | None, secondary: dict | None, max_bias: float = 0.90) -> dict | None:
    if not primary and not secondary:
        return None
    if not primary:
        return _normalize_signal(secondary or {}, min_confidence=0.0, max_bias=max_bias)
    if not secondary:
        return _normalize_signal(primary or {}, min_confidence=0.0, max_bias=max_bias)

    p = _normalize_signal(primary, min_confidence=0.0, max_bias=max_bias) or {"bias": 0.0, "confidence": 0.0}
    s = _normalize_signal(secondary, min_confidence=0.0, max_bias=max_bias) or {"bias": 0.0, "confidence": 0.0}

    wp = max(0.0, _safe_float(p.get("confidence"), 0.0))
    ws = max(0.0, _safe_float(s.get("confidence"), 0.0))
    wsum = wp + ws
    if wsum <= 1e-12:
        wp = 1.0
        ws = 1.0
        wsum = 2.0

    bias = (wp * _safe_float(p.get("bias"), 0.0) + ws * _safe_float(s.get("bias"), 0.0)) / wsum
    conf = max(_safe_float(p.get("confidence"), 0.0), _safe_float(s.get("confidence"), 0.0))
    return {"bias": _clamp(bias, -abs(float(max_bias)), abs(float(max_bias))), "confidence": _clamp(conf, 0.0, 1.0)}


def load_external_signal_map(path: Path, min_confidence: float = 0.55, max_bias: float = 0.90) -> dict[str, dict]:
    try:
        p = Path(path)
    except Exception:
        return {}
    if not p.exists():
        return {}

    try:
        payload = json.loads(p.read_text())
    except Exception:
        return {}

    out: dict[str, dict] = {}

    # Preferred format:
    # {"signals":{"SPY":{"bias":0.2,"confidence":0.7},...}}
    if isinstance(payload, dict):
        global_sig = payload.get("global")
        if isinstance(global_sig, dict):
            g = _normalize_signal(global_sig, min_confidence=min_confidence, max_bias=max_bias)
            if g is not None:
                out["__GLOBAL__"] = g

        signals = payload.get("signals")
        if isinstance(signals, dict):
            for sym, sig in signals.items():
                if not isinstance(sig, dict):
                    continue
                n = _normalize_signal(sig, min_confidence=min_confidence, max_bias=max_bias)
                if n is None:
                    continue
                key = str(sym).strip().upper()
                if key:
                    out[key] = n

    # Fallback format: list of rows
    # [{"symbol":"SPY","bias":0.2,"confidence":0.7}, ...]
    if not out and isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            key = str(row.get("symbol", "")).strip().upper()
            if not key:
                continue
            n = _normalize_signal(row, min_confidence=min_confidence, max_bias=max_bias)
            if n is None:
                continue
            out[key] = n

    return out

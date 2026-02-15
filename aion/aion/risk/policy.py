from __future__ import annotations

import json
import math
from pathlib import Path


def _safe_float(x, default=None):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default


def _clean_symbol_list(values) -> list[str]:
    if not isinstance(values, list):
        return []
    out = []
    seen = set()
    for v in values:
        s = str(v).strip().upper()
        if not s:
            continue
        if s in seen:
            continue
        out.append(s)
        seen.add(s)
    return out


def default_policy() -> dict:
    return {
        "enabled": True,
        "block_new_entries": False,
        "max_trades_per_day": None,
        "max_open_positions": None,
        "risk_per_trade": None,
        "max_position_notional_pct": None,
        "max_gross_leverage": None,
        "daily_loss_limit_abs": None,
        "daily_loss_limit_pct": None,
        "blocked_symbols": [],
        "allowed_symbols": [],
    }


def load_policy(path: Path) -> dict:
    p = default_policy()
    try:
        f = Path(path)
    except Exception:
        return p
    if not f.exists():
        return p
    try:
        raw = json.loads(f.read_text())
    except Exception:
        return p
    if not isinstance(raw, dict):
        return p

    p["enabled"] = bool(raw.get("enabled", p["enabled"]))
    p["block_new_entries"] = bool(raw.get("block_new_entries", p["block_new_entries"]))

    mt = _safe_int(raw.get("max_trades_per_day"), None)
    if mt is not None and mt > 0:
        p["max_trades_per_day"] = mt
    mo = _safe_int(raw.get("max_open_positions"), None)
    if mo is not None and mo > 0:
        p["max_open_positions"] = mo
    rpt = _safe_float(raw.get("risk_per_trade"), None)
    if rpt is not None and 0.0 < rpt <= 0.25:
        p["risk_per_trade"] = rpt
    npct = _safe_float(raw.get("max_position_notional_pct"), None)
    if npct is not None and 0.0 < npct <= 1.0:
        p["max_position_notional_pct"] = npct
    gl = _safe_float(raw.get("max_gross_leverage"), None)
    if gl is not None and 0.1 <= gl <= 5.0:
        p["max_gross_leverage"] = gl
    dabs = _safe_float(raw.get("daily_loss_limit_abs"), None)
    if dabs is not None and dabs > 0.0:
        p["daily_loss_limit_abs"] = dabs
    dpct = _safe_float(raw.get("daily_loss_limit_pct"), None)
    if dpct is not None and 0.0 < dpct <= 1.0:
        p["daily_loss_limit_pct"] = dpct

    p["blocked_symbols"] = _clean_symbol_list(raw.get("blocked_symbols", []))
    p["allowed_symbols"] = _clean_symbol_list(raw.get("allowed_symbols", []))
    return p


def apply_policy_caps(
    policy: dict,
    *,
    max_trades_per_day: int,
    max_open_positions: int,
    risk_per_trade: float,
    max_position_notional_pct: float,
    max_gross_leverage: float,
) -> dict:
    out = {
        "max_trades_per_day": int(max(1, int(max_trades_per_day))),
        "max_open_positions": int(max(1, int(max_open_positions))),
        "risk_per_trade": float(max(1e-5, float(risk_per_trade))),
        "max_position_notional_pct": float(max(1e-5, float(max_position_notional_pct))),
        "max_gross_leverage": float(max(0.05, float(max_gross_leverage))),
        "block_new_entries": False,
        "blocked_symbols": set(),
        "allowed_symbols": set(),
        "daily_loss_limit_abs": None,
        "daily_loss_limit_pct": None,
    }
    if not isinstance(policy, dict) or not bool(policy.get("enabled", True)):
        return out

    mt = _safe_int(policy.get("max_trades_per_day"), None)
    if mt is not None and mt > 0:
        out["max_trades_per_day"] = min(out["max_trades_per_day"], int(mt))

    mo = _safe_int(policy.get("max_open_positions"), None)
    if mo is not None and mo > 0:
        out["max_open_positions"] = min(out["max_open_positions"], int(mo))

    rpt = _safe_float(policy.get("risk_per_trade"), None)
    if rpt is not None and rpt > 0:
        out["risk_per_trade"] = min(out["risk_per_trade"], float(rpt))

    npct = _safe_float(policy.get("max_position_notional_pct"), None)
    if npct is not None and npct > 0:
        out["max_position_notional_pct"] = min(out["max_position_notional_pct"], float(npct))

    gl = _safe_float(policy.get("max_gross_leverage"), None)
    if gl is not None and gl > 0:
        out["max_gross_leverage"] = min(out["max_gross_leverage"], float(gl))

    out["block_new_entries"] = bool(policy.get("block_new_entries", False))
    out["blocked_symbols"] = set(_clean_symbol_list(policy.get("blocked_symbols", [])))
    out["allowed_symbols"] = set(_clean_symbol_list(policy.get("allowed_symbols", [])))
    out["daily_loss_limit_abs"] = _safe_float(policy.get("daily_loss_limit_abs"), None)
    out["daily_loss_limit_pct"] = _safe_float(policy.get("daily_loss_limit_pct"), None)
    return out


def symbol_allowed(symbol: str, caps: dict) -> bool:
    s = str(symbol).strip().upper()
    if not s:
        return False
    blocked = set(caps.get("blocked_symbols", set()) or set())
    if s in blocked:
        return False
    allowed = set(caps.get("allowed_symbols", set()) or set())
    if allowed and s not in allowed:
        return False
    return True

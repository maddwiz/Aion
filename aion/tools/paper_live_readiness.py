#!/usr/bin/env python3
"""
Paper-live readiness precheck for AION.

This verifies that Q promotion artifacts, external holdout validation,
overlay integrity, and local safety conditions are in place before
starting trade loops.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
Q_RUNS = REPO_ROOT / "q" / "runs_plus"
STATE_DIR = ROOT / "state"
LOG_DIR = ROOT / "logs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from aion.brain.external_signals import validate_overlay  # type: ignore
except Exception:
    validate_overlay = None  # type: ignore


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def evaluate(
    *,
    q_runs: Path = Q_RUNS,
    state_dir: Path = STATE_DIR,
    check_ib: bool = False,
) -> dict:
    hard: list[str] = []
    soft: list[str] = []
    checks: dict[str, object] = {}

    strict = _read_json(q_runs / "strict_oos_validation.json")
    promo = _read_json(q_runs / "q_promotion_gate.json")
    stress = _read_json(q_runs / "cost_stress_validation.json")
    alerts = _read_json(q_runs / "health_alerts.json")
    holdout = _read_json(q_runs / "external_holdout_validation.json")

    net = strict.get("metrics_oos_net", {}) if isinstance(strict.get("metrics_oos_net"), dict) else {}
    n = int(net.get("n", 0))
    sharpe = float(net.get("sharpe", 0.0))
    hit_rate = float(net.get("hit_rate", 0.0))
    max_dd = float(net.get("max_drawdown", 0.0))
    checks["strict_oos_net"] = {"n": n, "sharpe": sharpe, "hit_rate": hit_rate, "max_drawdown": max_dd}
    if n < 252:
        hard.append(f"strict_oos_n<252 ({n})")
    if sharpe <= 0.0:
        hard.append(f"strict_oos_sharpe<=0 ({sharpe:.3f})")

    promo_ok = bool(promo.get("ok", False))
    checks["promotion_gate_ok"] = promo_ok
    if not promo_ok:
        hard.append("q_promotion_gate_fail")

    stress_ok = bool(stress.get("ok", False))
    checks["cost_stress_ok"] = stress_ok
    if not stress_ok:
        hard.append("cost_stress_fail")

    alerts_ok = bool(alerts.get("ok", False))
    hard_alerts = int(alerts.get("alerts_hard", 0))
    soft_alerts = int(alerts.get("alerts_soft", 0))
    checks["health_alerts"] = {
        "ok": alerts_ok,
        "hard": hard_alerts,
        "soft": soft_alerts,
        "alerts": list(alerts.get("alerts", [])) if isinstance(alerts.get("alerts", []), list) else [],
    }
    if (not alerts_ok) or hard_alerts > 0:
        hard.append(f"health_alerts_hard>0 ({hard_alerts})")
    if soft_alerts > 0:
        soft.append(f"health_alerts_soft={soft_alerts}")

    holdout_ok = bool(holdout.get("ok", False))
    holdout_skipped = bool(holdout.get("skipped", False))
    checks["external_holdout"] = {"ok": holdout_ok, "skipped": holdout_skipped, "reason": holdout.get("reason", "")}
    if not holdout_ok:
        hard.append("external_holdout_fail")
    if holdout_skipped:
        hard.append("external_holdout_skipped")

    overlay_path = Path(os.getenv("AION_EXT_SIGNAL_FILE", str(state_dir / "q_signal_overlay.json")))
    overlay_exists = overlay_path.exists()
    checks["overlay_file"] = str(overlay_path)
    checks["overlay_exists"] = bool(overlay_exists)
    if not overlay_exists:
        hard.append(f"overlay_missing:{overlay_path}")
    else:
        payload = _read_json(overlay_path)
        if not payload:
            hard.append("overlay_unreadable")
        elif callable(validate_overlay):
            valid, reason = validate_overlay(payload)
            checks["overlay_valid"] = bool(valid)
            checks["overlay_reason"] = str(reason)
            if not valid:
                hard.append(f"overlay_invalid:{reason}")
        else:
            soft.append("overlay_validate_function_unavailable")

    kill_file = state_dir / "KILL_SWITCH"
    checks["kill_switch_active"] = bool(kill_file.exists())
    if kill_file.exists():
        hard.append("kill_switch_active")

    paper_mode = _env_bool("AION_PAPER_MODE", True)
    block_live = _env_bool("AION_BLOCK_LIVE_ORDERS", True)
    checks["paper_mode"] = bool(paper_mode)
    checks["block_live_orders"] = bool(block_live)
    if not paper_mode:
        hard.append("AION_PAPER_MODE!=1")
    if not block_live:
        soft.append("AION_BLOCK_LIVE_ORDERS=0 (live-route path)")

    checks["state_dir_exists"] = state_dir.exists()
    checks["log_dir_exists"] = LOG_DIR.exists()
    if not state_dir.exists():
        hard.append("state_dir_missing")
    if not LOG_DIR.exists():
        hard.append("log_dir_missing")

    if check_ib:
        ib_ok = False
        ib_err = ""
        try:
            from aion.data.ib_client import disconnect, ib  # type: ignore

            c = ib()
            ib_ok = bool(c and c.isConnected())
            disconnect()
        except Exception as exc:
            ib_err = str(exc)
        checks["ib_connectivity_ok"] = bool(ib_ok)
        checks["ib_connectivity_error"] = ib_err
        if not ib_ok:
            hard.append(f"ib_connectivity_fail:{ib_err or 'unknown'}")

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "ok": len(hard) == 0,
        "hard_blockers": hard,
        "soft_warnings": soft,
        "checks": checks,
    }
    return payload


def main() -> int:
    check_ib = _env_bool("AION_READINESS_CHECK_IB", False)
    out = evaluate(check_ib=check_ib)
    state_dir = Path(os.getenv("AION_STATE_DIR", str(STATE_DIR)))
    state_dir.mkdir(parents=True, exist_ok=True)
    out_path = state_dir / "paper_live_readiness.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"✅ Wrote {out_path}")
    if not bool(out.get("ok", False)):
        print("READINESS FAIL:", "; ".join([str(x) for x in out.get("hard_blockers", [])]))
        return 2
    warnings = out.get("soft_warnings", [])
    if isinstance(warnings, list) and warnings:
        print("READINESS WARN:", "; ".join([str(x) for x in warnings]))
    print("✅ Paper-live readiness passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

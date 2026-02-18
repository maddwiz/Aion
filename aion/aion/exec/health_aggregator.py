"""Aggregate operational health snapshots into one system_health.json."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _load_json_safe(path: Path):
    p = Path(path)
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, (dict, list)) else {}
    except Exception:
        return {}


def _write_json_atomic(path: Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(p)


def write_system_health(state_dir: Path, log_dir: Path | None = None) -> Path:
    sdir = Path(state_dir)
    ldir = Path(log_dir) if log_dir is not None else sdir.parent / "logs"

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "doctor": _load_json_safe(ldir / "doctor_report.json"),
        "runtime_monitor": _load_json_safe(ldir / "runtime_monitor.json"),
        "telemetry_summary": _load_json_safe(sdir / "telemetry_summary.json"),
        "reconciliation": _load_json_safe(sdir / "reconciliation_result.json"),
        "kill_switch_active": (sdir / "KILL_SWITCH").exists(),
        "paper_mode": str(os.getenv("AION_PAPER_MODE", "1")).strip().lower() not in {"0", "false", "no", "off"},
    }

    out = sdir / "system_health.json"
    _write_json_atomic(out, payload)
    return out

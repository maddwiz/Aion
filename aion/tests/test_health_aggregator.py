import json
from pathlib import Path

from aion.exec.health_aggregator import write_system_health


def test_write_system_health_creates_unified_file(tmp_path: Path):
    state_dir = tmp_path / "state"
    log_dir = tmp_path / "logs"
    state_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    (log_dir / "doctor_report.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (log_dir / "runtime_monitor.json").write_text(json.dumps({"alerts": []}), encoding="utf-8")
    (state_dir / "telemetry_summary.json").write_text(json.dumps({"rolling_hit_rate": 0.5}), encoding="utf-8")

    out = write_system_health(state_dir=state_dir, log_dir=log_dir)
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert "doctor" in payload
    assert "runtime_monitor" in payload
    assert "telemetry_summary" in payload
    assert "kill_switch_active" in payload

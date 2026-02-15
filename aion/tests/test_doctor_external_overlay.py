import json
from pathlib import Path

from aion.exec.doctor import check_external_overlay


def _write(path: Path, payload: dict):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_check_external_overlay_ok(tmp_path: Path):
    p = tmp_path / "overlay.json"
    _write(
        p,
        {
            "signals": {"AAPL": {"bias": 0.2, "confidence": 0.7}},
            "quality_gate": {"ok": True},
            "runtime_context": {"runtime_multiplier": 0.9, "risk_flags": []},
            "degraded_safe_mode": False,
        },
    )
    ok, msg, details = check_external_overlay(p, max_age_hours=24.0, require_runtime_context=True)
    assert ok is True
    assert "healthy" in msg.lower()
    assert details["signals"] == 1
    assert details["runtime_context_present"] is True


def test_check_external_overlay_flags_degraded_and_qgate(tmp_path: Path):
    p = tmp_path / "overlay.json"
    _write(
        p,
        {
            "signals": {"AAPL": {"bias": 0.2, "confidence": 0.7}},
            "quality_gate": {"ok": False},
            "runtime_context": {"runtime_multiplier": 0.7, "risk_flags": ["drift_alert"]},
            "degraded_safe_mode": True,
        },
    )
    ok, msg, details = check_external_overlay(p, max_age_hours=24.0, require_runtime_context=True)
    assert ok is False
    assert "degraded_safe_mode=true" in msg
    assert "quality_gate_not_ok" in msg
    assert details["quality_gate_ok"] is False


def test_check_external_overlay_requires_runtime_context(tmp_path: Path):
    p = tmp_path / "overlay.json"
    _write(
        p,
        {
            "signals": {"AAPL": {"bias": 0.2, "confidence": 0.7}},
            "quality_gate": {"ok": True},
            "degraded_safe_mode": False,
        },
    )
    ok, msg, _details = check_external_overlay(p, max_age_hours=24.0, require_runtime_context=True)
    assert ok is False
    assert "runtime_context_missing" in msg

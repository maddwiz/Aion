from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    mod_path = root / "tools" / "paper_live_readiness.py"
    spec = importlib.util.spec_from_file_location("paper_live_readiness_mod", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _valid_overlay_payload() -> dict:
    signals = {"__GLOBAL__": {"bias": 0.2, "confidence": 0.6}, "SPY": {"bias": 0.3, "confidence": 0.7}}
    sig_bytes = json.dumps(signals, sort_keys=True, separators=(",", ":")).encode()
    checksum = hashlib.sha256(sig_bytes).hexdigest()
    return {
        "version": "2026.02",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checksum": checksum,
        "signals": signals,
    }


def test_readiness_passes_with_valid_inputs(tmp_path: Path, monkeypatch):
    mod = _load_module()
    q_runs = tmp_path / "q" / "runs_plus"
    state = tmp_path / "aion" / "state"
    logs = tmp_path / "aion" / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    _write_json(q_runs / "strict_oos_validation.json", {"metrics_oos_net": {"n": 300, "sharpe": 1.2, "hit_rate": 0.5, "max_drawdown": -0.1}})
    _write_json(q_runs / "q_promotion_gate.json", {"ok": True})
    _write_json(q_runs / "cost_stress_validation.json", {"ok": True})
    _write_json(q_runs / "health_alerts.json", {"ok": True, "alerts_hard": 0, "alerts_soft": 0, "alerts": []})
    _write_json(q_runs / "external_holdout_validation.json", {"ok": True, "skipped": False})
    _write_json(state / "q_signal_overlay.json", _valid_overlay_payload())

    monkeypatch.setattr(mod, "LOG_DIR", logs)
    monkeypatch.setenv("AION_PAPER_MODE", "1")
    monkeypatch.setenv("AION_BLOCK_LIVE_ORDERS", "1")
    monkeypatch.setenv("AION_EXT_SIGNAL_FILE", str(state / "q_signal_overlay.json"))

    out = mod.evaluate(q_runs=q_runs, state_dir=state, check_ib=False)
    assert out["ok"] is True
    assert out["hard_blockers"] == []


def test_readiness_fails_when_external_holdout_skipped(tmp_path: Path, monkeypatch):
    mod = _load_module()
    q_runs = tmp_path / "q" / "runs_plus"
    state = tmp_path / "aion" / "state"
    logs = tmp_path / "aion" / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    _write_json(q_runs / "strict_oos_validation.json", {"metrics_oos_net": {"n": 300, "sharpe": 1.2, "hit_rate": 0.5, "max_drawdown": -0.1}})
    _write_json(q_runs / "q_promotion_gate.json", {"ok": True})
    _write_json(q_runs / "cost_stress_validation.json", {"ok": True})
    _write_json(q_runs / "health_alerts.json", {"ok": True, "alerts_hard": 0, "alerts_soft": 0, "alerts": []})
    _write_json(q_runs / "external_holdout_validation.json", {"ok": True, "skipped": True, "reason": "missing_holdout_dir"})
    _write_json(state / "q_signal_overlay.json", _valid_overlay_payload())

    monkeypatch.setattr(mod, "LOG_DIR", logs)
    monkeypatch.setenv("AION_PAPER_MODE", "1")
    monkeypatch.setenv("AION_BLOCK_LIVE_ORDERS", "1")
    monkeypatch.setenv("AION_EXT_SIGNAL_FILE", str(state / "q_signal_overlay.json"))

    out = mod.evaluate(q_runs=q_runs, state_dir=state, check_ib=False)
    assert out["ok"] is False
    assert "external_holdout_skipped" in out["hard_blockers"]

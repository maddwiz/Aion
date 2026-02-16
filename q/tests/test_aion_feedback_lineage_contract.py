import json

import pandas as pd

import tools.run_health_alerts as rha
import tools.run_quality_governor as rqg
import tools.run_system_health as rsh
import tools.sync_novaspine_memory as sm


def _thresholds():
    return {
        "min_health_score": 70,
        "min_global_governor_mean": 0.45,
        "min_quality_gov_mean": 0.60,
        "min_quality_score": 0.45,
        "require_immune_pass": False,
        "max_health_issues": 2,
        "min_nested_sharpe": 0.2,
        "min_nested_assets": 3,
        "max_shock_rate": 0.25,
        "max_concentration_hhi_after": 0.18,
        "max_concentration_top1_after": 0.30,
        "max_portfolio_l1_drift": 1.2,
        "min_aion_feedback_risk_scale": 0.80,
        "min_aion_feedback_closed_trades": 8,
        "min_aion_feedback_hit_rate": 0.38,
        "min_aion_feedback_profit_factor": 0.78,
        "max_aion_feedback_age_hours": 24.0,
    }


def _write_overlay(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_shadow(path):
    pd.DataFrame(
        {
            "timestamp": ["2026-02-16 10:00:00", "2026-02-16 10:05:00", "2026-02-16 10:10:00"],
            "side": ["EXIT_BUY", "EXIT_SELL", "PARTIAL_BUY"],
            "pnl": [4.0, -1.0, 2.5],
        }
    ).to_csv(path, index=False)


def test_lineage_contract_overlay_preference(tmp_path, monkeypatch):
    monkeypatch.setattr(rqg, "RUNS", tmp_path)
    monkeypatch.setattr(sm, "RUNS", tmp_path)
    shadow = tmp_path / "shadow_trades.csv"
    _write_shadow(shadow)
    monkeypatch.setenv("Q_AION_SHADOW_TRADES", str(shadow))
    monkeypatch.setenv("Q_AION_FEEDBACK_SOURCE", "overlay")

    overlay = {
        "runtime_context": {
            "aion_feedback": {
                "active": True,
                "status": "ok",
                "source": "overlay",
                "source_selected": "shadow_trades",
                "risk_scale": 0.98,
                "closed_trades": 20,
                "age_hours": 1.0,
                "max_age_hours": 24.0,
                "stale": False,
            }
        }
    }
    _write_overlay(tmp_path / "q_signal_overlay.json", overlay)

    fb, src = rqg._load_aion_feedback()
    assert src["source"] == "overlay"
    assert src["source_selected"] == "overlay"
    assert src["source_preference"] == "overlay"
    assert fb.get("status") == "ok"

    fallback = {
        "active": True,
        "status": "alert",
        "risk_scale": 0.70,
        "closed_trades": 20,
        "age_hours": 2.0,
        "max_age_hours": 24.0,
        "stale": False,
    }
    metrics, _issues = rsh._overlay_aion_feedback_metrics_with_fallback(
        overlay, fallback_feedback=fallback, source_pref="overlay"
    )
    assert metrics["aion_feedback_source"] == "overlay"
    assert metrics["aion_feedback_source_selected"] == "overlay"
    assert metrics["aion_feedback_source_preference"] == "overlay"

    payload = rha.build_alert_payload(
        health={"health_score": 95, "issues": [], "shape": {}},
        guards={"global_governor": {"mean": 0.85}},
        nested={"assets": 4, "avg_oos_sharpe": 0.8},
        quality={"quality_governor_mean": 0.88, "quality_score": 0.72},
        immune={"ok": True, "pass": True},
        pipeline={"failed_count": 0},
        shock={"shock_rate": 0.05},
        concentration={"stats": {"hhi_after": 0.12, "top1_after": 0.18}},
        drift_watch={"drift": {"status": "ok", "latest_l1": 0.5}},
        fracture={"state": "stable", "latest_score": 0.22},
        overlay=overlay,
        aion_feedback_fallback=fallback,
        aion_feedback_source_pref="overlay",
        thresholds=_thresholds(),
    )
    assert payload["observed"]["aion_feedback_source"] == "overlay"
    assert payload["observed"]["aion_feedback_source_selected"] == "overlay"
    assert payload["observed"]["aion_feedback_source_preference"] == "overlay"

    events = sm.build_events()
    rc = [e for e in events if e.get("event_type") == "governance.risk_controls"][0]
    af = rc.get("payload", {}).get("aion_feedback", {})
    assert af["source"] == "overlay"
    assert af["source_selected"] == "overlay"
    assert af["source_preference"] == "overlay"


def test_lineage_contract_shadow_preference(tmp_path, monkeypatch):
    monkeypatch.setattr(rqg, "RUNS", tmp_path)
    monkeypatch.setattr(sm, "RUNS", tmp_path)
    shadow = tmp_path / "shadow_trades.csv"
    _write_shadow(shadow)
    monkeypatch.setenv("Q_AION_SHADOW_TRADES", str(shadow))
    monkeypatch.setenv("Q_AION_FEEDBACK_SOURCE", "shadow")

    overlay = {
        "runtime_context": {
            "aion_feedback": {
                "active": True,
                "status": "ok",
                "source": "overlay",
                "risk_scale": 0.99,
                "closed_trades": 20,
                "age_hours": 1.0,
                "max_age_hours": 24.0,
                "stale": False,
            }
        }
    }
    _write_overlay(tmp_path / "q_signal_overlay.json", overlay)

    fb, src = rqg._load_aion_feedback()
    assert src["source"] == "shadow_trades"
    assert src["source_selected"] == "shadow_trades"
    assert src["source_preference"] == "shadow"
    assert int(fb.get("closed_trades", 0)) == 3

    fallback = {
        "active": True,
        "status": "alert",
        "risk_scale": 0.70,
        "closed_trades": 20,
        "age_hours": 2.0,
        "max_age_hours": 24.0,
        "stale": False,
    }
    metrics, _issues = rsh._overlay_aion_feedback_metrics_with_fallback(
        overlay, fallback_feedback=fallback, source_pref="shadow"
    )
    assert metrics["aion_feedback_source"] == "shadow_trades"
    assert metrics["aion_feedback_source_selected"] == "shadow_trades"
    assert metrics["aion_feedback_source_preference"] == "shadow"

    payload = rha.build_alert_payload(
        health={"health_score": 95, "issues": [], "shape": {}},
        guards={"global_governor": {"mean": 0.85}},
        nested={"assets": 4, "avg_oos_sharpe": 0.8},
        quality={"quality_governor_mean": 0.88, "quality_score": 0.72},
        immune={"ok": True, "pass": True},
        pipeline={"failed_count": 0},
        shock={"shock_rate": 0.05},
        concentration={"stats": {"hhi_after": 0.12, "top1_after": 0.18}},
        drift_watch={"drift": {"status": "ok", "latest_l1": 0.5}},
        fracture={"state": "stable", "latest_score": 0.22},
        overlay=overlay,
        aion_feedback_fallback=fallback,
        aion_feedback_source_pref="shadow",
        thresholds=_thresholds(),
    )
    assert payload["observed"]["aion_feedback_source"] == "shadow_trades"
    assert payload["observed"]["aion_feedback_source_selected"] == "shadow_trades"
    assert payload["observed"]["aion_feedback_source_preference"] == "shadow"

    events = sm.build_events()
    rc = [e for e in events if e.get("event_type") == "governance.risk_controls"][0]
    af = rc.get("payload", {}).get("aion_feedback", {})
    assert af["source"] == "shadow_trades"
    assert af["source_selected"] == "shadow_trades"
    assert af["source_preference"] == "shadow"

from aion.brain import novaspine_bridge as nsb
from aion.exec import paper_loop as pl


def test_aion_feedback_lineage_flows_into_novaspine_metadata(monkeypatch):
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_AION_FEEDBACK_ENABLED", True)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_AION_FEEDBACK_ALERT_THRESHOLD", 0.82)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_AION_FEEDBACK_BLOCK_ON_ALERT", False)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_AION_FEEDBACK_MIN_CLOSED_TRADES", 8)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_AION_FEEDBACK_MAX_AGE_HOURS", 72.0)
    monkeypatch.setattr(pl.cfg, "EXT_SIGNAL_AION_FEEDBACK_IGNORE_STALE", True)

    ctl = pl._aion_feedback_controls(
        {
            "active": True,
            "status": "warn",
            "source": "overlay",
            "source_selected": "shadow_trades",
            "source_preference": "shadow",
            "risk_scale": 0.91,
            "closed_trades": 14,
            "stale": False,
            "reasons": ["negative_expectancy_warn"],
        }
    )
    assert ctl["active"] is True
    assert ctl["source"] == "overlay"
    assert ctl["source_selected"] == "shadow_trades"
    assert ctl["source_preference"] == "shadow"

    runtime_ctx = pl._compact_memory_runtime_context(
        ext_runtime_scale=0.93,
        ext_position_risk_scale=0.89,
        ext_runtime_diag={"regime": "balanced", "overlay_stale": False, "flags": ["aion_outcome_warn"]},
        ext_overlay_age_hours=0.4,
        ext_overlay_age_source="payload",
        memory_feedback_status="ok",
        memory_feedback_risk_scale=1.0,
        memory_feedback_turnover_pressure=0.42,
        memory_feedback_turnover_dampener=0.03,
        memory_feedback_block_new_entries=False,
        aion_feedback_status=str(ctl["status"]),
        aion_feedback_source=str(ctl["source"]),
        aion_feedback_source_selected=str(ctl["source_selected"]),
        aion_feedback_source_preference=str(ctl["source_preference"]),
        aion_feedback_risk_scale=float(ctl["risk_scale"]),
        aion_feedback_stale=bool(ctl["stale"]),
        aion_feedback_block_new_entries=bool(ctl["block_new_entries"]),
        policy_block_new_entries=False,
        killswitch_block_new_entries=False,
        exec_governor_state="warn",
        exec_governor_block_new_entries=False,
    )

    ev = nsb.build_trade_event(
        event_type="trade.entry",
        symbol="AAPL",
        side="LONG",
        qty=5,
        entry=190.25,
        exit=0.0,
        pnl=0.0,
        reason="test",
        confidence=0.82,
        regime="balanced",
        extra={"runtime_context": runtime_ctx},
    )
    ingest = nsb._event_to_novaspine_ingest(ev, namespace="private/nova/actions", source_prefix="aion")
    rc = ingest.get("metadata", {}).get("runtime_context_summary", {})

    assert rc.get("aion_feedback_status") == "warn"
    assert rc.get("aion_feedback_source") == "overlay"
    assert rc.get("aion_feedback_source_selected") == "shadow_trades"
    assert rc.get("aion_feedback_source_preference") == "shadow"
    assert rc.get("external_regime") == "balanced"
    assert rc.get("memory_feedback_status") == "ok"
    assert rc.get("memory_feedback_turnover_pressure") == 0.42
    assert rc.get("memory_feedback_turnover_dampener") == 0.03
    assert rc.get("exec_governor_state") == "warn"

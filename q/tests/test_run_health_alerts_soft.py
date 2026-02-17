import tools.run_health_alerts as rha


def test_soft_alert_prefix_matching():
    prefs = ["aion_feedback_", "stale_optional_file>"]
    assert rha._is_soft_alert("aion_feedback_status=alert", prefs) is True
    assert rha._is_soft_alert("stale_optional_file>72.0h (x:90h)", prefs) is True
    assert rha._is_soft_alert("runtime_total_scalar_min<0.04 (0.01)", prefs) is False


def test_health_issue_threshold_counts_only_hard_issues():
    health = {
        "health_score": 90.0,
        "issues": [
            "aion_feedback_status=alert",
            "stale_optional_file>72.0h (x:90h)",
        ],
        "shape": {},
    }
    payload = rha.build_alert_payload(
        health=health,
        guards={},
        nested={},
        quality={},
        immune={},
        pipeline={},
        shock={},
        concentration={},
        drift_watch={},
        thresholds={"max_health_issues": 0},
    )
    assert not any(str(a).startswith("health_issues>") for a in payload.get("alerts", []))

    health_hard = {
        "health_score": 90.0,
        "issues": ["runtime_total_scalar mean below threshold"],
        "shape": {},
    }
    payload_hard = rha.build_alert_payload(
        health=health_hard,
        guards={},
        nested={},
        quality={},
        immune={},
        pipeline={},
        shock={},
        concentration={},
        drift_watch={},
        thresholds={"max_health_issues": 0},
    )
    assert any(str(a).startswith("health_issues>") for a in payload_hard.get("alerts", []))

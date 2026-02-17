import tools.run_health_alerts as rha


def test_soft_alert_prefix_matching():
    prefs = ["aion_feedback_", "stale_optional_file>"]
    assert rha._is_soft_alert("aion_feedback_status=alert", prefs) is True
    assert rha._is_soft_alert("stale_optional_file>72.0h (x:90h)", prefs) is True
    assert rha._is_soft_alert("runtime_total_scalar_min<0.04 (0.01)", prefs) is False

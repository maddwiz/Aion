import tools.run_portfolio_drift_watch as rpdw


def test_classify_drift_alert_on_large_latest_l1():
    out = rpdw._classify_drift(
        {"latest_l1": 1.5, "p95_l1": 0.5},
        max_l1_alert=1.2,
        ratio_alert=3.0,
        ratio_warn=1.8,
    )
    assert out["status"] == "alert"
    assert out["latest_over_p95"] > 2.9


def test_classify_drift_warn_on_ratio_only():
    out = rpdw._classify_drift(
        {"latest_l1": 0.9, "p95_l1": 0.4},
        max_l1_alert=1.2,
        ratio_alert=3.0,
        ratio_warn=1.8,
    )
    assert out["status"] == "warn"


def test_classify_drift_ok_when_stable():
    out = rpdw._classify_drift(
        {"latest_l1": 0.3, "p95_l1": 0.35},
        max_l1_alert=1.2,
        ratio_alert=3.0,
        ratio_warn=1.8,
    )
    assert out["status"] == "ok"

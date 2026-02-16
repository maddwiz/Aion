import tools.run_quality_governor as rqg
import numpy as np
import pandas as pd


def test_execution_constraint_quality_high_for_reasonable_retention():
    q, detail = rqg._execution_constraint_quality(
        {
            "gross_before_mean": 0.80,
            "gross_after_mean": 0.62,
            "turnover_before_mean": 0.20,
            "turnover_after_mean": 0.13,
            "turnover_after_max": 0.30,
            "max_step_turnover": 0.35,
        },
        [],
    )
    assert q is not None
    assert 0.60 <= float(q) <= 1.0
    assert float(detail["gross_retention"]) > 0.7


def test_execution_constraint_quality_drops_when_over_throttled():
    q, _detail = rqg._execution_constraint_quality(
        {
            "gross_before_mean": 0.40,
            "gross_after_mean": 0.01,
            "turnover_before_mean": 0.22,
            "turnover_after_mean": 0.001,
        },
        ["execution constraints may be over-throttling turnover"],
    )
    assert q is not None
    assert float(q) < 0.20


def test_execution_constraint_quality_drops_when_turnover_increases():
    q, detail = rqg._execution_constraint_quality(
        {
            "gross_before_mean": 0.30,
            "gross_after_mean": 0.28,
            "turnover_before_mean": 0.10,
            "turnover_after_mean": 0.18,
            "turnover_after_max": 0.40,
            "max_step_turnover": 0.30,
        },
        [],
    )
    assert q is not None
    assert float(detail["turnover_retention"]) > 1.0
    assert float(q) < 0.60


def test_cap_step_change_limits_adjacent_moves():
    x = np.array([0.60, 0.90, 0.30, 1.10], dtype=float)
    y = rqg._cap_step_change(x, max_step=0.10, lo=0.55, hi=1.15)
    d = np.diff(y)
    assert float(np.max(np.abs(d))) <= 0.10 + 1e-12
    assert float(np.min(y)) >= 0.55 - 1e-12
    assert float(np.max(y)) <= 1.15 + 1e-12


def test_nested_leakage_quality_high_when_utilization_is_healthy():
    q, detail = rqg._nested_leakage_quality(
        {
            "assets": 20,
            "avg_outer_fold_utilization": 0.82,
            "low_utilization_assets": 1,
            "avg_train_ratio_mean": 0.88,
            "params": {"purge_embargo_ratio": 0.22},
        }
    )
    assert q is not None
    assert float(q) > 0.75
    assert float(detail["avg_outer_fold_utilization"]) == 0.82


def test_nested_leakage_quality_drops_with_low_utilization_and_heavy_purge():
    q, detail = rqg._nested_leakage_quality(
        {
            "assets": 16,
            "avg_outer_fold_utilization": 0.38,
            "low_utilization_assets": 10,
            "avg_train_ratio_mean": 0.50,
            "params": {"purge_embargo_ratio": 0.95},
        }
    )
    assert q is not None
    assert float(q) < 0.45
    assert float(detail["purge_embargo_ratio"]) == 0.95


def test_aion_outcome_quality_high_on_strong_feedback():
    q, detail = rqg._aion_outcome_quality(
        {
            "active": True,
            "status": "ok",
            "closed_trades": 22,
            "risk_scale": 1.02,
            "hit_rate": 0.56,
            "profit_factor": 1.42,
            "expectancy_norm": 0.22,
            "drawdown_norm": 0.45,
        },
        min_closed_trades=8,
    )
    assert q is not None
    assert float(q) > 0.75
    assert detail["mature_window"] is True


def test_aion_outcome_quality_low_on_alert_feedback():
    q, detail = rqg._aion_outcome_quality(
        {
            "active": True,
            "status": "alert",
            "closed_trades": 18,
            "risk_scale": 0.72,
            "hit_rate": 0.31,
            "profit_factor": 0.62,
            "expectancy_norm": -0.24,
            "drawdown_norm": 3.20,
        },
        min_closed_trades=8,
    )
    assert q is not None
    assert float(q) < 0.45
    assert detail["status"] == "alert"


def test_load_aion_feedback_falls_back_to_shadow_trades(monkeypatch, tmp_path):
    monkeypatch.setattr(rqg, "RUNS", tmp_path)
    shadow = tmp_path / "shadow_trades.csv"
    pd.DataFrame(
        {
            "timestamp": ["2026-02-16 09:31:00", "2026-02-16 09:45:00", "2026-02-16 10:10:00"],
            "side": ["EXIT_BUY", "EXIT_SELL", "PARTIAL_BUY"],
            "pnl": [10.0, -4.0, 7.0],
        }
    ).to_csv(shadow, index=False)

    monkeypatch.setenv("Q_AION_SHADOW_TRADES", str(shadow))
    fb, src = rqg._load_aion_feedback()
    assert src["source"] == "shadow_trades"
    assert fb["active"] is True
    assert int(fb["closed_trades"]) == 3
    assert "risk_scale" in fb

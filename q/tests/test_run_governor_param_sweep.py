import tools.run_governor_param_sweep as rps


def test_objective_penalizes_drawdown_and_can_veto():
    base = {
        "sharpe": 0.75,
        "hit_rate": 0.52,
        "max_drawdown": -0.05,
        "turnover_mean": 0.01,
    }
    bad = {
        "sharpe": 0.90,
        "hit_rate": 0.48,
        "max_drawdown": -0.12,
        "turnover_mean": 0.03,
    }
    score, detail = rps._objective(bad, base)
    assert detail["dd_ratio"] > 2.0
    assert detail["veto"] is True
    assert score < 0.5


def test_profile_from_row_casts_types():
    row = {
        "runtime_total_floor": 0.1,
        "shock_alpha": 0.35,
        "rank_sleeve_blend": 0.06,
        "low_vol_sleeve_blend": 0.04,
        "meta_execution_gate_strength": 0.95,
        "council_gate_strength": 0.9,
        "meta_mix_leverage_strength": 1.15,
        "meta_reliability_strength": 1.1,
        "global_governor_strength": 0.95,
        "heartbeat_scaler_strength": 0.4,
        "quality_governor_strength": 1.05,
        "regime_moe_strength": 1.2,
        "uncertainty_sizing_strength": 0.85,
        "vol_target_strength": 0.7,
        "hit_gate_strength": 0.6,
        "hit_gate_threshold": 0.53,
        "signal_deadzone": 0.0012,
        "use_concentration_governor": 1,
        "concentration_top1_cap": 0.18,
        "concentration_top3_cap": 0.42,
        "concentration_max_hhi": 0.14,
    }
    out = rps._profile_from_row(row)
    assert out["runtime_total_floor"] == 0.1
    assert out["shock_alpha"] == 0.35
    assert out["rank_sleeve_blend"] == 0.06
    assert out["low_vol_sleeve_blend"] == 0.04
    assert out["meta_execution_gate_strength"] == 0.95
    assert out["council_gate_strength"] == 0.9
    assert out["meta_mix_leverage_strength"] == 1.15
    assert out["meta_reliability_strength"] == 1.1
    assert out["global_governor_strength"] == 0.95
    assert out["heartbeat_scaler_strength"] == 0.4
    assert out["quality_governor_strength"] == 1.05
    assert out["regime_moe_strength"] == 1.2
    assert out["uncertainty_sizing_strength"] == 0.85
    assert out["vol_target_strength"] == 0.7
    assert out["hit_gate_strength"] == 0.6
    assert out["hit_gate_threshold"] == 0.53
    assert out["signal_deadzone"] == 0.0012
    assert out["use_concentration_governor"] is True
    assert out["concentration_top1_cap"] == 0.18


def test_objective_uses_oos_metrics_when_available():
    base = {
        "sharpe": 1.6,
        "hit_rate": 0.55,
        "max_drawdown": -0.04,
        "turnover_mean": 0.02,
        "oos_sharpe": 1.10,
        "oos_hit_rate": 0.50,
        "oos_max_drawdown": -0.05,
        "oos_n": 300,
    }
    cur = {
        "sharpe": 2.1,  # better in-sample
        "hit_rate": 0.58,
        "max_drawdown": -0.03,
        "turnover_mean": 0.02,
        "oos_sharpe": 0.85,  # worse OOS
        "oos_hit_rate": 0.47,
        "oos_max_drawdown": -0.07,
        "oos_n": 300,
    }
    score, detail = rps._objective(cur, base)
    assert detail["objective_sharpe"] == cur["oos_sharpe"]
    assert detail["objective_hit_rate"] == cur["oos_hit_rate"]
    assert score < cur["sharpe"]

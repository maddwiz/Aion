from qmods.novaspine_context import (
    apply_turnover_dampener,
    build_context_query,
    context_boost,
    context_resonance,
    turnover_pressure,
)


def test_context_resonance_bounds_and_order():
    r0 = context_resonance(0, 6, [])
    r1 = context_resonance(3, 6, [0.2, 0.3, 0.4])
    r2 = context_resonance(6, 6, [0.8, 0.9, 1.0])
    assert 0.0 <= r0 <= 1.0
    assert 0.0 <= r1 <= 1.0
    assert 0.0 <= r2 <= 1.0
    assert r2 > r1 > r0


def test_context_boost_behavior():
    b_disabled = context_boost(1.0, status_ok=False)
    b_low = context_boost(0.1, status_ok=True)
    b_high = context_boost(0.9, status_ok=True)
    assert abs(b_disabled - 1.0) < 1e-9
    assert 0.90 <= b_low <= 1.10
    assert 0.90 <= b_high <= 1.10
    assert b_high > b_low


def test_turnover_pressure_tracks_cross_hive_stress():
    p_low = turnover_pressure(0.15, 0.35, 0.40)
    p_high = turnover_pressure(0.60, 1.20, 1.70)
    assert 0.0 <= p_low <= 1.0
    assert 0.0 <= p_high <= 1.0
    assert p_high > p_low


def test_apply_turnover_dampener_only_reduces_boost_under_pressure():
    base = 1.06
    low = apply_turnover_dampener(base, pressure=0.10, max_cut=0.12)
    high = apply_turnover_dampener(base, pressure=1.00, max_cut=0.12)
    assert 0.85 <= low <= 1.10
    assert 0.85 <= high <= 1.10
    assert high < low < base


def test_build_context_query_contains_signals():
    q = build_context_query(["RATES", "COMMOD"], 0.62, ["quality_governor_mean<0.6"])
    assert "RATES" in q
    assert "COMMOD" in q
    assert "0.620" in q

from qmods.novaspine_hive import build_hive_query, hive_boost, hive_resonance


def test_hive_resonance_increases_with_coverage_and_scores():
    r0 = hive_resonance(0, 4, [])
    r1 = hive_resonance(2, 4, [0.4, 0.5])
    r2 = hive_resonance(4, 4, [0.8, 0.9, 1.0])
    assert 0.0 <= r0 <= 1.0
    assert 0.0 <= r1 <= 1.0
    assert 0.0 <= r2 <= 1.0
    assert r2 > r1 > r0


def test_hive_boost_is_neutral_when_unavailable():
    assert abs(hive_boost(0.9, status_ok=False) - 1.0) < 1e-9


def test_hive_boost_bounds_and_order():
    low = hive_boost(0.1, status_ok=True)
    high = hive_boost(0.9, status_ok=True)
    assert 0.85 <= low <= 1.10
    assert 0.85 <= high <= 1.10
    assert high > low


def test_build_hive_query_contains_key_metrics():
    q = build_hive_query("eq", sharpe=1.24, hit=0.57, max_dd=0.18)
    assert "EQ" in q
    assert "1.240" in q
    assert "0.570" in q
    assert "0.180" in q

import json

import pandas as pd

import tools.run_cross_hive as rch


def test_novaspine_hive_multipliers_default_when_missing(tmp_path):
    old_runs = rch.RUNS
    try:
        rch.RUNS = tmp_path
        out = rch.novaspine_hive_multipliers(["EQ", "FX"])
    finally:
        rch.RUNS = old_runs

    assert out == {"EQ": 1.0, "FX": 1.0}


def test_novaspine_hive_multipliers_reads_and_clips(tmp_path):
    payload = {
        "per_hive": {
            "EQ": {"boost": 1.5},
            "FX": {"boost": 0.6},
            "RATES": {"boost": 1.03},
        }
    }
    (tmp_path / "novaspine_hive_feedback.json").write_text(json.dumps(payload), encoding="utf-8")

    old_runs = rch.RUNS
    try:
        rch.RUNS = tmp_path
        out = rch.novaspine_hive_multipliers(["EQ", "FX", "RATES", "COMMOD"])
    finally:
        rch.RUNS = old_runs

    assert out["EQ"] == 1.2
    assert out["FX"] == 0.8
    assert out["RATES"] == 1.03
    assert out["COMMOD"] == 1.0


def test_adaptive_arb_schedules_respond_to_disagreement():
    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    stab = pd.DataFrame(
        {
            "EQ": [1.0, 0.5, 0.1],
            "FX": [1.0, 0.5, 0.1],
        },
        index=idx,
    )
    alpha_t, inertia_t, diag = rch.adaptive_arb_schedules(2.2, 0.8, stab)
    assert len(alpha_t) == 3
    assert len(inertia_t) == 3
    assert alpha_t[0] > alpha_t[-1]
    assert inertia_t[0] < inertia_t[-1]
    assert diag["alpha_min"] <= diag["alpha_max"]

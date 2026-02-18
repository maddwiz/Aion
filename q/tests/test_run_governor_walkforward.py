import json
from pathlib import Path

import numpy as np

import tools.run_governor_walkforward as rgw


def test_build_folds_expanding_window():
    folds = rgw._build_folds(total_rows=260, train_min=120, test_size=40, max_folds=10)
    assert len(folds) == 3
    assert folds[0]["train_end"] == 120
    assert folds[0]["test_start"] == 120
    assert folds[0]["test_end"] == 160
    assert folds[-1]["train_end"] == 200
    assert folds[-1]["test_end"] == 240


def test_run_governor_walkforward_selects_by_train_score(monkeypatch):
    base_ret = np.full(240, 0.0005, dtype=float)
    monkeypatch.setattr(rgw.rcs, "_load_daily_returns_for_governor_eval", lambda: base_ret)
    monkeypatch.setattr(rgw.rcs, "_base_runtime_env", lambda: {})
    monkeypatch.setattr(rgw.rcs, "_refresh_friction_calibration", lambda _env: None)

    candidates = [
        {
            "runtime_total_floor": 0.18,
            "disable_governors": [],
            "search_flag_count": 0,
            "enable_asset_class_diversification": False,
            "macro_proxy_strength": 0.0,
            "capacity_impact_strength": 0.0,
            "uncertainty_macro_shock_blend": 0.0,
        },
        {
            "runtime_total_floor": 0.22,
            "disable_governors": [],
            "search_flag_count": 0,
            "enable_asset_class_diversification": False,
            "macro_proxy_strength": 0.0,
            "capacity_impact_strength": 0.0,
            "uncertainty_macro_shock_blend": 0.0,
        },
    ]
    monkeypatch.setattr(
        rgw,
        "_build_candidate_grid",
        lambda: (
            candidates,
            {
                "floors": [0.18, 0.22],
                "flags": [],
                "class_enables": [0],
                "macro_strengths": [0.0],
                "capacity_strengths": [0.0],
                "macro_blends": [0.0],
                "full_grid_total": 2,
                "sampled_grid_total": 2,
                "max_combos": 2,
            },
        ),
    )

    monkeypatch.setenv("Q_GOVERNOR_WF_TRAIN_MIN", "120")
    monkeypatch.setenv("Q_GOVERNOR_WF_TEST_SIZE", "40")
    monkeypatch.setenv("Q_GOVERNOR_WF_MAX_FOLDS", "2")

    state = {"floor": 0.0}

    def _fake_eval(env):
        state["floor"] = float(env.get("Q_RUNTIME_TOTAL_FLOOR", 0.18))
        return {
            "promotion_ok": True,
            "cost_stress_ok": True,
            "health_ok": True,
            "health_alerts_hard": 0,
            "rc": [{"step": "x", "code": 0}],
        }

    def _fake_returns():
        # Candidate 0.18 has stronger train metrics and should be selected.
        if abs(state["floor"] - 0.18) < 1e-9:
            lead = np.tile(np.array([0.0010, 0.0002], dtype=float), 100)
            tail = np.full(40, -0.0008, dtype=float)
            return np.concatenate([lead, tail])
        lead = np.tile(np.array([0.0004, -0.0001], dtype=float), 100)
        tail = np.full(40, 0.0002, dtype=float)
        return np.concatenate([lead, tail])

    out = rgw.run_governor_walkforward(eval_combo_fn=_fake_eval, returns_loader_fn=_fake_returns)
    assert out["ok"] is True
    assert int(out["folds_completed"]) >= 1
    assert int(out["oos_rows"]) > 0
    assert float(out["folds"][0]["best_params"]["runtime_total_floor"]) == 0.18


def test_main_writes_metrics_file_when_enabled(monkeypatch, tmp_path: Path):
    root = tmp_path
    runs = root / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rgw, "ROOT", root)
    monkeypatch.setattr(rgw, "RUNS", runs)
    monkeypatch.setenv("Q_DISABLE_REPORT_CARDS", "1")
    monkeypatch.setenv("Q_GOVERNOR_WALKFORWARD_ENABLED", "1")

    monkeypatch.setattr(
        rgw,
        "run_governor_walkforward",
        lambda **_kwargs: {
            "ok": True,
            "reason": "ok",
            "rows_total": 100,
            "folds_requested": 1,
            "folds_completed": 1,
            "folds_skipped": 0,
            "grid": {"sampled_grid_total": 1},
            "settings": {"train_min": 50, "test_size": 25, "max_folds": 1},
            "folds": [],
            "oos_metrics": {"sharpe": 1.1, "hit_rate": 0.51, "max_drawdown": -0.03, "n": 25},
            "oos_rows": 25,
        },
    )
    rc = rgw.main()
    assert rc == 0
    out = json.loads((runs / "governor_walkforward_metrics.json").read_text(encoding="utf-8"))
    assert out["enabled"] is True
    assert out["ok"] is True

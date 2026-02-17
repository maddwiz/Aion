import json
from pathlib import Path

import numpy as np

import tools.run_q_promotion_gate as pg
import tools.run_strict_oos_validation as so


def test_strict_oos_metrics_basic():
    r = np.array([0.01, -0.01, 0.02, -0.02], dtype=float)
    m = so._metrics(r)
    assert m["n"] == 4
    assert abs(m["hit_rate"] - 0.5) < 1e-12
    assert m["vol_daily"] > 0.0


def test_q_promotion_gate_pass(tmp_path: Path, monkeypatch):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics_oos_net": {
            "sharpe": 1.2,
            "hit_rate": 0.51,
            "max_drawdown": -0.04,
            "n": 300,
        }
    }
    (runs / "strict_oos_validation.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(pg, "ROOT", tmp_path)
    monkeypatch.setattr(pg, "RUNS", runs)
    rc = pg.main()
    assert rc == 0
    out = json.loads((runs / "q_promotion_gate.json").read_text(encoding="utf-8"))
    assert out["ok"] is True
    assert out["reasons"] == []


def test_q_promotion_gate_fail(tmp_path: Path, monkeypatch):
    runs = tmp_path / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics_oos_net": {
            "sharpe": 0.6,
            "hit_rate": 0.45,
            "max_drawdown": -0.20,
            "n": 100,
        }
    }
    (runs / "strict_oos_validation.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(pg, "ROOT", tmp_path)
    monkeypatch.setattr(pg, "RUNS", runs)
    rc = pg.main()
    assert rc == 0
    out = json.loads((runs / "q_promotion_gate.json").read_text(encoding="utf-8"))
    assert out["ok"] is False
    assert len(out["reasons"]) >= 1

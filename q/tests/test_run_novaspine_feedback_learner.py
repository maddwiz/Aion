import json
from pathlib import Path

import tools.run_novaspine_feedback_learner as rnfl


def test_run_novaspine_feedback_learner_builds_signal_priors(tmp_path: Path, monkeypatch):
    root = tmp_path / "q"
    runs = root / "runs_plus"
    runs.mkdir(parents=True, exist_ok=True)
    aion_state = tmp_path / "aion" / "state"
    aion_state.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "symbol": "AAPL",
            "category_scores": {"q_overlay": 0.82, "multi_timeframe": 0.75},
            "regime": "calm_trend",
            "session_phase": "opening_drive",
            "pnl_realized": 120.0,
            "reasons": ["Q overlay supports long"],
        },
        {
            "symbol": "MSFT",
            "category_scores": {"q_overlay": 0.70},
            "regime": "calm_trend",
            "session_phase": "range_extension",
            "pnl_realized": -50.0,
            "reasons": ["Q overlay supports long"],
        },
        {
            "symbol": "NVDA",
            "category_scores": {"q_overlay": 0.90},
            "regime": "calm_trend",
            "session_phase": "opening_drive",
            "pnl_realized": 80.0,
            "reasons": ["Q overlay supports long"],
        },
    ]
    with (aion_state / "trade_decisions.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    monkeypatch.setattr(rnfl, "ROOT", root)
    monkeypatch.setattr(rnfl, "RUNS", runs)
    monkeypatch.setattr(rnfl, "AION_STATE", aion_state)
    monkeypatch.setenv("Q_DISABLE_REPORT_CARDS", "1")

    rc = rnfl.main()
    assert rc == 0

    payload = json.loads((runs / "novaspine_signal_priors.json").read_text(encoding="utf-8"))
    assert payload["records_usable"] == 3
    assert "q_overlay" in payload["signal_priors"]
    q = payload["signal_priors"]["q_overlay"]
    assert float(q["alpha"]) > float(q["beta"])

import json
from pathlib import Path

import aion.exec.promotion_gate as pg


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_trades(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = "timestamp,symbol,side,qty,entry,exit,pnl,reason,confidence,regime,stop,target,trail,fill_ratio,slippage_bps"
    lines = [hdr]
    lines.extend(rows)
    path.write_text("\n".join(lines), encoding="utf-8")


def test_promotion_gate_approves_with_strong_execution_quality(tmp_path, monkeypatch):
    perf = tmp_path / "performance_report.json"
    wf = tmp_path / "walkforward_results.json"
    mon = tmp_path / "runtime_monitor.json"
    trades = tmp_path / "shadow_trades.csv"
    out = tmp_path / "live_promotion.json"

    _write_json(
        perf,
        {
            "trade_metrics": {"closed_trades": 120, "winrate": 0.59, "profit_factor": 1.62},
            "equity_metrics": {"max_drawdown": 0.08},
        },
    )
    _write_json(wf, {"summary": {"avg_symbol_test_pnl": 0.12}})
    _write_json(mon, {"slippage_points": [10.0 + (i % 3) for i in range(60)]})
    _write_trades(
        trades,
        [
            "2026-02-10 10:00:00,SPY,EXIT_BUY,1,500,501,1.0,ok,0.8,balanced,0,0,0,1.0,9.0",
            "2026-02-10 13:00:00,QQQ,EXIT_SELL,1,400,399,1.0,ok,0.8,balanced,0,0,0,1.0,11.0",
            "2026-02-11 10:00:00,IWM,EXIT_BUY,1,200,201,1.0,ok,0.8,balanced,0,0,0,1.0,10.0",
            "2026-02-11 13:00:00,DIA,EXIT_SELL,1,300,299,1.0,ok,0.8,balanced,0,0,0,1.0,10.0",
            "2026-02-12 10:00:00,XLK,EXIT_BUY,1,150,151,1.0,ok,0.8,balanced,0,0,0,1.0,9.0",
            "2026-02-12 13:00:00,XLF,EXIT_SELL,1,120,119,1.0,ok,0.8,balanced,0,0,0,1.0,12.0",
        ],
    )

    monkeypatch.setattr(pg, "PERF", perf)
    monkeypatch.setattr(pg, "WF", wf)
    monkeypatch.setattr(pg, "MON", mon)
    monkeypatch.setattr(pg, "TRADES", trades)
    monkeypatch.setattr(pg, "OUT", out)
    monkeypatch.setattr(pg.cfg, "PROMOTION_MIN_TRADES", 60)
    monkeypatch.setattr(pg.cfg, "PROMOTION_MIN_WINRATE", 0.52)
    monkeypatch.setattr(pg.cfg, "PROMOTION_MIN_PROFIT_FACTOR", 1.25)
    monkeypatch.setattr(pg.cfg, "PROMOTION_MAX_DRAWDOWN", 0.12)
    monkeypatch.setattr(pg.cfg, "PROMOTION_MIN_WF_AVG_PNL", 0.0)
    monkeypatch.setattr(pg.cfg, "PROMOTION_MIN_SLIPPAGE_SAMPLES", 30)
    monkeypatch.setattr(pg.cfg, "PROMOTION_MAX_AVG_SLIPPAGE_BPS", 18.0)
    monkeypatch.setattr(pg.cfg, "PROMOTION_MAX_P90_SLIPPAGE_BPS", 28.0)
    monkeypatch.setattr(pg.cfg, "PROMOTION_MAX_CLOSED_TRADES_PER_DAY", 18.0)

    rc = pg.main()
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["approved"] is True
    assert payload["reasons"] == []


def test_promotion_gate_rejects_high_slippage_and_churn(tmp_path, monkeypatch):
    perf = tmp_path / "performance_report.json"
    wf = tmp_path / "walkforward_results.json"
    mon = tmp_path / "runtime_monitor.json"
    trades = tmp_path / "shadow_trades.csv"
    out = tmp_path / "live_promotion.json"

    _write_json(
        perf,
        {
            "trade_metrics": {"closed_trades": 200, "winrate": 0.61, "profit_factor": 1.70},
            "equity_metrics": {"max_drawdown": 0.09},
        },
    )
    _write_json(wf, {"summary": {"avg_symbol_test_pnl": 0.15}})
    _write_json(mon, {"slippage_points": [34.0, 36.0, 40.0, 38.0, 35.0, 37.0] * 8})

    rows = []
    for d in ["2026-02-10", "2026-02-11", "2026-02-12"]:
        for h in range(9, 21):
            rows.append(f"{d} {h:02d}:00:00,SPY,EXIT_BUY,1,500,501,1.0,ok,0.8,balanced,0,0,0,1.0,35.0")
    _write_trades(trades, rows)

    monkeypatch.setattr(pg, "PERF", perf)
    monkeypatch.setattr(pg, "WF", wf)
    monkeypatch.setattr(pg, "MON", mon)
    monkeypatch.setattr(pg, "TRADES", trades)
    monkeypatch.setattr(pg, "OUT", out)
    monkeypatch.setattr(pg.cfg, "PROMOTION_MIN_TRADES", 60)
    monkeypatch.setattr(pg.cfg, "PROMOTION_MIN_WINRATE", 0.52)
    monkeypatch.setattr(pg.cfg, "PROMOTION_MIN_PROFIT_FACTOR", 1.25)
    monkeypatch.setattr(pg.cfg, "PROMOTION_MAX_DRAWDOWN", 0.12)
    monkeypatch.setattr(pg.cfg, "PROMOTION_MIN_WF_AVG_PNL", 0.0)
    monkeypatch.setattr(pg.cfg, "PROMOTION_MIN_SLIPPAGE_SAMPLES", 30)
    monkeypatch.setattr(pg.cfg, "PROMOTION_MAX_AVG_SLIPPAGE_BPS", 18.0)
    monkeypatch.setattr(pg.cfg, "PROMOTION_MAX_P90_SLIPPAGE_BPS", 28.0)
    monkeypatch.setattr(pg.cfg, "PROMOTION_MAX_CLOSED_TRADES_PER_DAY", 10.0)

    rc = pg.main()
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["approved"] is False
    reasons = " | ".join(payload["reasons"])
    assert "Avg slippage too high" in reasons
    assert "Slippage p90 too high" in reasons
    assert "Closed-trade cadence too high" in reasons

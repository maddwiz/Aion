from pathlib import Path

from aion.exec.paper_loop import _daily_loss_limits_hit
from aion.risk.policy import apply_policy_caps, load_policy, symbol_allowed


def test_load_policy_and_symbol_filters(tmp_path: Path):
    p = tmp_path / "risk_policy.json"
    p.write_text(
        """
        {
          "enabled": true,
          "max_trades_per_day": 9,
          "max_open_positions": 4,
          "risk_per_trade": 0.015,
          "max_position_notional_pct": 0.12,
          "max_gross_leverage": 1.1,
          "daily_loss_limit_abs": 120,
          "daily_loss_limit_pct": 0.03,
          "blocked_symbols": ["gme", " amc "],
          "allowed_symbols": ["aapl", "msft"]
        }
        """,
        encoding="utf-8",
    )
    policy = load_policy(p)
    caps = apply_policy_caps(
        policy,
        max_trades_per_day=15,
        max_open_positions=7,
        risk_per_trade=0.02,
        max_position_notional_pct=0.25,
        max_gross_leverage=1.7,
    )

    assert caps["max_trades_per_day"] == 9
    assert caps["max_open_positions"] == 4
    assert caps["risk_per_trade"] == 0.015
    assert caps["max_position_notional_pct"] == 0.12
    assert caps["max_gross_leverage"] == 1.1
    assert symbol_allowed("AAPL", caps) is True
    assert symbol_allowed("TSLA", caps) is False
    assert symbol_allowed("GME", caps) is False


def test_daily_loss_limit_hit_by_abs_and_pct():
    caps_abs = {"daily_loss_limit_abs": 100.0, "daily_loss_limit_pct": None}
    hit_abs, loss_abs, loss_pct = _daily_loss_limits_hit(caps_abs, 5000.0, 4885.0)
    assert hit_abs is True
    assert round(loss_abs, 2) == 115.0
    assert loss_pct > 0.02

    caps_pct = {"daily_loss_limit_abs": None, "daily_loss_limit_pct": 0.02}
    hit_pct, *_ = _daily_loss_limits_hit(caps_pct, 5000.0, 4920.0)
    assert hit_pct is False

    hit_pct2, *_ = _daily_loss_limits_hit(caps_pct, 5000.0, 4890.0)
    assert hit_pct2 is True

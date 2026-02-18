import json
from pathlib import Path

from aion.exec.reconciliation import reconcile_on_startup


class _Contract:
    def __init__(self, symbol: str):
        self.symbol = symbol


class _Position:
    def __init__(self, symbol: str, qty: int):
        self.contract = _Contract(symbol)
        self.position = qty


class _FakeIB:
    def __init__(self, positions):
        self._positions = positions

    def positions(self):
        return list(self._positions)


def test_reconciliation_passes_when_shadow_matches(tmp_path: Path):
    shadow_path = tmp_path / "shadow_trades.json"
    shadow_path.write_text(
        json.dumps({"AAPL": {"qty": 10, "avg_price": 100.0, "last_updated": "x"}}),
        encoding="utf-8",
    )
    ib = _FakeIB([_Position("AAPL", 10)])
    res = reconcile_on_startup(ib, shadow_path, auto_fix=True)
    assert res.passed is True
    assert res.mismatches == []
    assert res.action_taken == "none"


def test_reconciliation_auto_fix_updates_shadow(tmp_path: Path):
    shadow_path = tmp_path / "shadow_trades.json"
    shadow_path.write_text(
        json.dumps({"AAPL": {"qty": 5, "avg_price": 100.0, "last_updated": "x"}}),
        encoding="utf-8",
    )
    ib = _FakeIB([_Position("AAPL", 10), _Position("MSFT", -3)])
    res = reconcile_on_startup(ib, shadow_path, auto_fix=True)
    assert res.passed is False
    assert res.action_taken == "shadow_updated"

    fixed = json.loads(shadow_path.read_text(encoding="utf-8"))
    assert fixed["AAPL"]["qty"] == 10
    assert fixed["MSFT"]["qty"] == -3

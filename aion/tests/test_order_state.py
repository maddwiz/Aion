from pathlib import Path

from aion.exec.order_state import load_order_state, merge_safe_req_id, save_order_state


def test_order_state_roundtrip(tmp_path: Path):
    save_order_state(
        state_dir=tmp_path,
        next_valid_id=123,
        open_orders=[{"order_id": 1, "symbol": "AAPL", "qty": 10}],
    )
    data = load_order_state(tmp_path)
    assert isinstance(data, dict)
    assert data["next_valid_id"] == 123
    assert data["open_orders"][0]["symbol"] == "AAPL"


def test_order_state_atomic_write_no_tmp_leftover(tmp_path: Path):
    save_order_state(state_dir=tmp_path, next_valid_id=45, open_orders=[])
    assert (tmp_path / "order_state.json").exists()
    assert not (tmp_path / "order_state.json.tmp").exists()


def test_merge_safe_req_id_uses_max():
    assert merge_safe_req_id(100, 90) == 100
    assert merge_safe_req_id(100, 110) == 110
    assert merge_safe_req_id(None, 7) == 7

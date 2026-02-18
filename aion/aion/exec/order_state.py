"""Persistent order state management for IBKR reconnect safety."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


STATE_FILE = "order_state.json"


def _state_path(state_dir: Path) -> Path:
    return Path(state_dir) / STATE_FILE


def save_order_state(
    state_dir: Path,
    next_valid_id: int,
    open_orders: list[dict],
) -> Path:
    """Persist order state via atomic tmp->rename write."""
    d = Path(state_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = _state_path(d)
    payload = {
        "next_valid_id": int(max(0, int(next_valid_id))),
        "open_orders": list(open_orders or []),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def load_order_state(state_dir: Path) -> dict | None:
    """Load persisted order state if available and valid."""
    path = _state_path(Path(state_dir))
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {
            "next_valid_id": int(max(0, int(raw.get("next_valid_id", 0)))),
            "open_orders": list(raw.get("open_orders", [])),
            "saved_at": str(raw.get("saved_at", "")),
        }
    except Exception:
        return None


def merge_safe_req_id(saved_next_id: int | None, ibkr_next_id: int | None) -> int:
    """Return safe order id floor across saved and live values."""
    s = int(saved_next_id or 0)
    i = int(ibkr_next_id or 0)
    return max(s, i)

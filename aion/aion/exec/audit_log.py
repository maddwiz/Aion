"""Thread-safe append-only JSONL audit log for order decisions and lifecycle events."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path


_lock = threading.Lock()


def audit_log(record: dict, log_dir: Path | None = None) -> Path:
    """Append one record to state/audit_orders.jsonl."""
    d = Path(log_dir) if log_dir is not None else Path("state")
    d.mkdir(parents=True, exist_ok=True)
    path = d / "audit_orders.jsonl"

    rec = dict(record or {})
    rec["ts"] = datetime.now(timezone.utc).isoformat()
    line = json.dumps(rec, separators=(",", ":"), ensure_ascii=True) + "\n"

    with _lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    return path

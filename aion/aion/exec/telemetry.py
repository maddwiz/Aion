"""
Structured decision telemetry writer.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _to_utc(ts: dt.datetime | None) -> dt.datetime:
    if not isinstance(ts, dt.datetime):
        return _utc_now()
    if ts.tzinfo is None:
        return ts.replace(tzinfo=dt.timezone.utc)
    return ts.astimezone(dt.timezone.utc)


def _ts_str(ts: dt.datetime | None = None) -> str:
    return _to_utc(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm_ts_from_record(rec: dict | None) -> dt.datetime:
    if not isinstance(rec, dict):
        return _utc_now()
    for key in ("timestamp", "ts", "time"):
        raw = rec.get(key)
        if raw is None:
            continue
        s = str(raw).strip()
        if not s:
            continue
        try:
            if s.endswith("Z"):
                return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(dt.timezone.utc)
            return dt.datetime.fromisoformat(s).astimezone(dt.timezone.utc)
        except Exception:
            continue
    return _utc_now()


class DecisionTelemetry:
    def __init__(
        self,
        cfg_mod,
        *,
        filename: str = "trade_decisions.jsonl",
        state_dir: Path | None = None,
    ):
        base = Path(state_dir) if state_dir is not None else Path(getattr(cfg_mod, "STATE_DIR", Path.cwd()))
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / str(filename)
        self._active_day: str | None = None

    def _rotate_if_needed(self, ts_utc: dt.datetime):
        day = ts_utc.strftime("%Y%m%d")
        if self._active_day is None:
            self._active_day = day
            return
        if day == self._active_day:
            return
        if self.path.exists() and self.path.stat().st_size > 0:
            archived = self.path.with_name(f"{self.path.stem}.{self._active_day}{self.path.suffix}")
            suffix = 1
            while archived.exists():
                archived = self.path.with_name(f"{self.path.stem}.{self._active_day}.{suffix}{self.path.suffix}")
                suffix += 1
            self.path.replace(archived)
        self._active_day = day

    def write(self, record: dict, *, timestamp: dt.datetime | None = None) -> dict:
        rec = dict(record or {})
        ts = _to_utc(timestamp if isinstance(timestamp, dt.datetime) else _norm_ts_from_record(rec))
        self._rotate_if_needed(ts)
        if "timestamp" not in rec and "ts" not in rec:
            rec["timestamp"] = _ts_str(ts)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=True, separators=(",", ":"), default=str) + "\n")
        return rec

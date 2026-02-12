from __future__ import annotations

import datetime as dt
import json
from pathlib import Path


def _parse_ts(value: str):
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value)
    except Exception:
        return None


class EventRiskFilter:
    def __init__(self, cfg):
        self.cfg = cfg
        self.file = Path(cfg.EVENT_BLOCK_FILE)

    def _load_manual_windows(self) -> list[dict]:
        if not self.file.exists():
            return []
        try:
            data = json.loads(self.file.read_text())
        except Exception:
            return []
        if isinstance(data, dict):
            data = data.get("windows", [])
        if not isinstance(data, list):
            return []

        out = []
        for w in data:
            if not isinstance(w, dict):
                continue
            start = _parse_ts(str(w.get("start", "")))
            end = _parse_ts(str(w.get("end", "")))
            if start and end and end > start:
                out.append({"start": start, "end": end, "reason": str(w.get("reason", "manual_blackout"))})
        return out

    def _daily_session_windows(self, now: dt.datetime) -> list[dict]:
        # US cash session defaults
        day = now.date()
        open_ts = dt.datetime.combine(day, dt.time(hour=9, minute=30))
        close_ts = dt.datetime.combine(day, dt.time(hour=16, minute=0))

        windows = [
            {
                "start": open_ts - dt.timedelta(minutes=self.cfg.EVENT_BLOCK_OPEN_MIN),
                "end": open_ts + dt.timedelta(minutes=self.cfg.EVENT_BLOCK_OPEN_MIN),
                "reason": "market_open_volatility",
            },
            {
                "start": close_ts - dt.timedelta(minutes=self.cfg.EVENT_BLOCK_CLOSE_MIN),
                "end": close_ts + dt.timedelta(minutes=self.cfg.EVENT_BLOCK_CLOSE_MIN),
                "reason": "market_close_volatility",
            },
        ]
        return windows

    def blocked(self, symbol: str, now: dt.datetime | None = None) -> tuple[bool, str]:
        if not self.cfg.EVENT_FILTER_ENABLED:
            return False, ""

        now = now or dt.datetime.now()
        windows = self._daily_session_windows(now) + self._load_manual_windows()
        for w in windows:
            if w["start"] <= now <= w["end"]:
                return True, w["reason"]

        return False, ""

"""Filesystem kill switch watcher for immediate flatten+shutdown."""

from __future__ import annotations

import logging
import time
from pathlib import Path


log = logging.getLogger(__name__)


class KillSwitchWatcher:
    def __init__(self, state_dir: Path, poll_seconds: float = 5.0):
        self.kill_file = Path(state_dir) / "KILL_SWITCH"
        self.poll_seconds = max(0.2, float(poll_seconds))
        self._last_check = 0.0

    def check(self) -> bool:
        """Return True when kill-switch is triggered."""
        now = time.monotonic()
        if (now - float(self._last_check)) < self.poll_seconds:
            return False
        self._last_check = now

        if not self.kill_file.exists():
            return False

        content = ""
        try:
            content = self.kill_file.read_text(encoding="utf-8").strip().upper()
        except Exception:
            content = ""

        # File presence is enough to trigger.
        if self.kill_file.exists():
            log.critical("KILL SWITCH TRIGGERED: %s content=%s", self.kill_file, content[:120])
            return True
        return False

    def acknowledge(self) -> None:
        """Rename kill file to avoid repeated trigger on restart."""
        if not self.kill_file.exists():
            return
        ack = self.kill_file.with_suffix(".acknowledged")
        try:
            self.kill_file.rename(ack)
        except Exception:
            pass

"""Best-effort webhook alerting for critical runtime events."""

from __future__ import annotations

import json
import os
from urllib import request


WEBHOOK_URL = os.getenv("AION_ALERT_WEBHOOK", "")


def send_alert(message: str, level: str = "WARNING") -> None:
    """Send alert payload to configured webhook (best-effort, non-fatal)."""
    url = str(os.getenv("AION_ALERT_WEBHOOK", WEBHOOK_URL)).strip()
    if not url:
        return

    payload = {"content": f"**[AION {level}]** {message}"}
    try:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        request.urlopen(req, timeout=5)
    except Exception:
        return

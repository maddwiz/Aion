from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib import request


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_slug(x: str) -> str:
    out = []
    for ch in str(x):
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        elif ch in (" ", "/", "\\", ":"):
            out.append("_")
    return "".join(out) or "default"


@dataclass
class PublishResult:
    backend: str
    published: int
    queued: int
    failed: int
    outbox_file: str | None = None
    error: str | None = None


def write_jsonl_outbox(events: Iterable[dict], outbox_dir: Path, prefix: str = "novaspine") -> Path:
    outbox_dir.mkdir(parents=True, exist_ok=True)
    stamp = _utc_now_iso().replace(":", "").replace("-", "")
    p = outbox_dir / f"{_safe_slug(prefix)}_{stamp}.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, separators=(",", ":"), ensure_ascii=True) + "\n")
    return p


def publish_events(
    events: list[dict],
    backend: str = "filesystem",
    namespace: str = "private/nova/actions",
    outbox_dir: Path | None = None,
    http_url: str | None = None,
    http_token: str | None = None,
    timeout_sec: float = 6.0,
) -> PublishResult:
    backend = str(backend or "filesystem").strip().lower()
    outbox_dir = outbox_dir or Path("runs_plus") / "novaspine_outbox"
    if not events:
        return PublishResult(backend=backend, published=0, queued=0, failed=0)

    # Ensure every event is namespaced + timestamped.
    normalized = []
    for ev in events:
        x = dict(ev or {})
        x.setdefault("namespace", namespace)
        x.setdefault("ts_utc", _utc_now_iso())
        normalized.append(x)

    if backend in ("none", "off", "disabled"):
        return PublishResult(backend=backend, published=0, queued=len(normalized), failed=0)

    if backend in ("filesystem", "file", "local"):
        p = write_jsonl_outbox(normalized, outbox_dir=outbox_dir, prefix="novaspine_batch")
        return PublishResult(backend="filesystem", published=0, queued=len(normalized), failed=0, outbox_file=str(p))

    if backend in ("http", "https"):
        if not http_url:
            p = write_jsonl_outbox(normalized, outbox_dir=outbox_dir, prefix="novaspine_failed")
            return PublishResult(
                backend="http",
                published=0,
                queued=len(normalized),
                failed=0,
                outbox_file=str(p),
                error="missing_http_url",
            )

        payload = {"namespace": namespace, "events": normalized}
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        req = request.Request(
            str(http_url),
            method="POST",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                **({"Authorization": f"Bearer {http_token}"} if http_token else {}),
            },
        )
        try:
            with request.urlopen(req, timeout=float(timeout_sec)) as resp:
                code = int(getattr(resp, "status", 200))
                if 200 <= code < 300:
                    return PublishResult(backend="http", published=len(normalized), queued=0, failed=0)
                p = write_jsonl_outbox(normalized, outbox_dir=outbox_dir, prefix="novaspine_failed")
                return PublishResult(
                    backend="http",
                    published=0,
                    queued=len(normalized),
                    failed=len(normalized),
                    outbox_file=str(p),
                    error=f"http_status_{code}",
                )
        except Exception as e:
            p = write_jsonl_outbox(normalized, outbox_dir=outbox_dir, prefix="novaspine_failed")
            return PublishResult(
                backend="http",
                published=0,
                queued=len(normalized),
                failed=len(normalized),
                outbox_file=str(p),
                error=str(e),
            )

    # Unknown backend: safe fallback to local queue.
    p = write_jsonl_outbox(normalized, outbox_dir=outbox_dir, prefix="novaspine_unknown_backend")
    return PublishResult(
        backend=backend,
        published=0,
        queued=len(normalized),
        failed=0,
        outbox_file=str(p),
        error="unknown_backend",
    )

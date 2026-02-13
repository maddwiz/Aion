from pathlib import Path

from qmods.novaspine_adapter import publish_events, write_jsonl_outbox


def test_write_jsonl_outbox(tmp_path: Path):
    events = [{"event_type": "x", "payload": {"a": 1}}]
    p = write_jsonl_outbox(events, outbox_dir=tmp_path, prefix="test_batch")
    assert p.exists()
    txt = p.read_text(encoding="utf-8").strip()
    assert '"event_type":"x"' in txt


def test_publish_filesystem(tmp_path: Path):
    events = [{"event_type": "decision.signal_export", "payload": {"signals_count": 7}}]
    res = publish_events(
        events=events,
        backend="filesystem",
        namespace="private/nova/actions",
        outbox_dir=tmp_path,
    )
    assert res.failed == 0
    assert res.queued == 1
    assert res.published == 0
    assert res.outbox_file is not None
    assert Path(res.outbox_file).exists()


def test_publish_unknown_backend_fallback(tmp_path: Path):
    events = [{"event_type": "governance.health_gate", "payload": {"ok": True}}]
    res = publish_events(
        events=events,
        backend="made_up_backend",
        namespace="private/nova/actions",
        outbox_dir=tmp_path,
    )
    assert res.failed == 0
    assert res.queued == 1
    assert res.outbox_file is not None

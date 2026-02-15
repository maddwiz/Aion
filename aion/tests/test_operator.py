import aion.exec.operator as op


def test_operator_defaults_to_status(monkeypatch):
    seen = {"status": 0}

    def fake_status():
        seen["status"] += 1
        return 0

    monkeypatch.setattr(op, "_status", fake_status)
    rc = op.main([])
    assert rc == 0
    assert seen["status"] == 1


def test_operator_start_dispatch(monkeypatch):
    seen = {"tasks": None}

    def fake_start(tasks):
        seen["tasks"] = tasks
        return 0

    monkeypatch.setattr(op, "_start", fake_start)
    rc = op.main(["start", "--task", "trade"])
    assert rc == 0
    assert seen["tasks"] == ["trade"]

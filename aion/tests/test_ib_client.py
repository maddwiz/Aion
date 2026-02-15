import aion.data.ib_client as ibc


def test_candidate_client_ids_dedupes_and_includes_base(monkeypatch):
    monkeypatch.setattr(ibc.cfg, "IB_CLIENT_ID", 1731)
    monkeypatch.setattr(ibc.cfg, "IB_CLIENT_ID_CANDIDATES", [1732, 1731, -1, 1733])
    ids = ibc._candidate_client_ids()
    assert ids[0] == 1731
    assert 1732 in ids
    assert 1733 in ids
    assert len(ids) == len(set(ids))
    assert all(i > 0 for i in ids)


def test_connect_ib_with_retries_uses_fallback_client_id(monkeypatch):
    monkeypatch.setattr(ibc.cfg, "IB_HOST", "127.0.0.1")
    monkeypatch.setattr(ibc.cfg, "IB_PORT", 4002)
    monkeypatch.setattr(ibc.cfg, "IB_HOST_CANDIDATES", ["127.0.0.1"])
    monkeypatch.setattr(ibc.cfg, "IB_PORT_CANDIDATES", [4002])
    monkeypatch.setattr(ibc.cfg, "IB_CLIENT_ID", 100)
    monkeypatch.setattr(ibc.cfg, "IB_CLIENT_ID_CANDIDATES", [101, 102])
    monkeypatch.setattr(ibc.time, "sleep", lambda _s: None)

    class FakeClient:
        def __init__(self):
            self.connected = False
            self.calls = []

        def connect(self, host, port, clientId, timeout):
            self.calls.append((host, port, clientId, timeout))
            if clientId == 102:
                self.connected = True
                return True
            raise RuntimeError("Error 326 client id already in use")

        def isConnected(self):
            return self.connected

        def disconnect(self):
            self.connected = False

    c = FakeClient()
    host, port, cid = ibc._connect_ib_with_retries(c)
    assert host == "127.0.0.1"
    assert port == 4002
    assert cid == 102
    tried = [x[2] for x in c.calls]
    assert tried[:3] == [100, 101, 102]

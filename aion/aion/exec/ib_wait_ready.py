import json
import time

from .. import config as cfg
from .doctor import _candidate_hosts, _candidate_ports, check_handshake_candidates, check_port_candidates


def main() -> int:
    hosts = _candidate_hosts()
    ports = _candidate_ports()
    max_wait = max(0, int(cfg.IB_WARMUP_SECONDS))
    poll = max(1, int(cfg.IB_WARMUP_POLL_SECONDS))
    deadline = time.time() + max_wait

    last = {
        "port_ok": False,
        "handshake_ok": False,
        "selected_port": None,
        "selected_handshake": None,
    }

    while True:
        port_ok, selected_port, port_rows = check_port_candidates(hosts, ports)
        hs_ok, selected_hs, hs_rows = check_handshake_candidates(hosts, ports, int(cfg.IB_CLIENT_ID))
        last = {
            "port_ok": bool(port_ok),
            "handshake_ok": bool(hs_ok),
            "selected_port": selected_port,
            "selected_handshake": selected_hs,
            "port_rows": port_rows,
            "handshake_rows": hs_rows,
        }
        if hs_ok:
            print(
                json.dumps(
                    {
                        "ok": True,
                        "message": f"IB handshake ready at {selected_hs['host']}:{selected_hs['port']}",
                        "selected": selected_hs,
                        "waited_seconds": max_wait - max(0, int(deadline - time.time())),
                    },
                    indent=2,
                )
            )
            return 0

        if time.time() >= deadline:
            break
        time.sleep(poll)

    print(
        json.dumps(
            {
                "ok": False,
                "message": "IB handshake not ready within warmup window",
                "hosts": hosts,
                "ports": ports,
                "state": last,
                "waited_seconds": max_wait,
            },
            indent=2,
        )
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

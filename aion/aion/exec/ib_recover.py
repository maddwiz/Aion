import datetime as dt
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from .. import config as cfg
from .doctor import check_handshake_candidates, check_port_candidates


def _run(cmd: str) -> str:
    try:
        return subprocess.run(["sh", "-lc", cmd], capture_output=True, text=True, timeout=6).stdout
    except Exception:
        return ""


def _candidate_ports() -> list[int]:
    ports = [int(cfg.IB_PORT)]
    for p in getattr(cfg, "IB_PORT_CANDIDATES", []):
        try:
            ports.append(int(p))
        except Exception:
            continue
    uniq = []
    seen = set()
    for p in ports:
        if p <= 0 or p in seen:
            continue
        uniq.append(p)
        seen.add(p)
    return uniq


def _candidate_hosts() -> list[str]:
    hosts = [str(cfg.IB_HOST)]
    for h in getattr(cfg, "IB_HOST_CANDIDATES", []):
        hs = str(h).strip()
        if hs:
            hosts.append(hs)
    uniq = []
    seen = set()
    for h in hosts:
        k = h.lower()
        if k in seen:
            continue
        uniq.append(h)
        seen.add(k)
    return uniq


def _ib_processes(listener_pids: set[int] | None = None):
    listener_pids = listener_pids or set()
    out = _run("ps -axo pid,ppid,command")
    rows = []
    for line in out.splitlines()[1:]:
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid_raw, ppid_raw, cmd = parts
        low = cmd.lower()
        try:
            pid = int(pid_raw)
            ppid = int(ppid_raw)
        except Exception:
            continue
        matched = (
            "ib gateway" in low
            or "ibgateway" in low
            or "trader workstation" in low
            or "tws" in low
            or "javaapplicationstub" in low
            or pid in listener_pids
        )
        if not matched:
            continue
        rows.append({"pid": pid, "ppid": ppid, "command": cmd})
    return rows


def _listener_pids(ports: list[int]):
    out = _run("lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null")
    pset = set()
    lines = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[1])
        except Exception:
            continue
        for p in ports:
            if f":{p}" in line:
                pset.add(pid)
                lines.append(line.strip())
                break
    return pset, lines


def _choose_keep_pid(processes: list[dict], listener_pids: set[int]):
    by_pid = {int(r["pid"]): r for r in processes}
    listener_matches = [by_pid[pid] for pid in listener_pids if pid in by_pid]
    if len(listener_matches) == 1:
        return int(listener_matches[0]["pid"])
    if len(listener_matches) > 1:
        return int(sorted(listener_matches, key=lambda r: int(r["pid"]), reverse=True)[0]["pid"])
    if not processes:
        return None
    newest = sorted(processes, key=lambda r: int(r["pid"]), reverse=True)[0]
    return int(newest["pid"])


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _terminate_pid(pid: int):
    try:
        os.kill(int(pid), signal.SIGTERM)
    except Exception:
        return "term_failed"
    for _ in range(12):
        time.sleep(0.25)
        if not _pid_alive(pid):
            return "terminated"
    try:
        os.kill(int(pid), signal.SIGKILL)
    except Exception:
        return "kill_failed"
    for _ in range(8):
        time.sleep(0.2)
        if not _pid_alive(pid):
            return "killed"
    return "still_alive"


def _extract_app_path(cmd: str):
    raw = str(cmd or "")
    if raw.lower().endswith(".app"):
        return raw
    marker = ".app/Contents/MacOS/"
    idx = raw.lower().find(marker.lower())
    if idx < 0:
        return None
    return raw[: idx + 4]


def _restart_app_candidates(active_cmd: str):
    ordered = []
    preferred = _extract_app_path(getattr(cfg, "IB_APP_PREFERRED", ""))
    if preferred:
        ordered.append(preferred)
    active = _extract_app_path(active_cmd)
    if active:
        ordered.append(active)
    for item in getattr(cfg, "IB_APP_CANDIDATES", []):
        candidate = _extract_app_path(str(item))
        if candidate:
            ordered.append(candidate)

    uniq = []
    seen = set()
    for path in ordered:
        key = path.lower()
        if key in seen:
            continue
        seen.add(key)
        if Path(path).exists():
            uniq.append(path)
    return uniq


def _start_app(app_path: str):
    try:
        subprocess.run(["open", "-a", app_path], capture_output=True, text=True, timeout=8)
        return "started"
    except Exception:
        return "start_failed"


def main() -> int:
    hosts = _candidate_hosts()
    ports = _candidate_ports()
    listener_pids, listener_lines = _listener_pids(ports)
    processes = _ib_processes(listener_pids=listener_pids)
    keep_pid = _choose_keep_pid(processes, listener_pids)

    payload = {
        "ts": dt.datetime.now().isoformat(),
        "auto_resolve_enabled": bool(cfg.AUTO_RESOLVE_IB_CONFLICT),
        "auto_restart_on_timeout": bool(cfg.AUTO_RESTART_IB_ON_TIMEOUT),
        "host": cfg.IB_HOST,
        "candidate_hosts": hosts,
        "candidate_ports": ports,
        "listener_lines": listener_lines,
        "processes_before": processes,
        "selected_keep_pid": keep_pid,
        "actions": [],
    }

    if len(processes) > 1:
        extras = [int(r["pid"]) for r in processes if int(r["pid"]) != int(keep_pid or -1)]
        payload["duplicate_count"] = len(extras)
        if cfg.AUTO_RESOLVE_IB_CONFLICT:
            for pid in extras:
                status = _terminate_pid(pid)
                payload["actions"].append({"pid": pid, "action": "terminate_extra", "result": status})
        else:
            payload["actions"].append(
                {
                    "action": "dry_run",
                    "note": "Duplicate IB processes detected. Set AION_AUTO_RESOLVE_IB_CONFLICT=1 to auto-terminate extras.",
                    "target_pids": extras,
                }
            )
    else:
        payload["duplicate_count"] = 0

    time.sleep(0.6)
    listener_pids_after, _ = _listener_pids(ports)
    processes_after = _ib_processes(listener_pids=listener_pids_after)
    payload["processes_after"] = processes_after

    ok_port, selected_port, port_rows = check_port_candidates(hosts, ports)
    ok_hs, selected_hs_port, hs_rows = check_handshake_candidates(hosts, ports, int(cfg.IB_CLIENT_ID))
    payload["port_check"] = {"ok": ok_port, "selected_port": selected_port, "details": port_rows}
    payload["handshake_check"] = {"ok": ok_hs, "selected_port": selected_hs_port, "details": hs_rows}

    if (not ok_hs) and cfg.AUTO_RESOLVE_IB_CONFLICT and cfg.AUTO_RESTART_IB_ON_TIMEOUT:
        keep_proc = next((r for r in processes_after if int(r.get("pid", -1)) == int(keep_pid)), None)
        app_candidates = _restart_app_candidates(str((keep_proc or {}).get("command", "")))
        payload["restart_app_candidates"] = app_candidates
        app_path = app_candidates[0] if app_candidates else None
        if app_path:
            if keep_pid:
                term_status = _terminate_pid(int(keep_pid))
                payload["actions"].append({"pid": int(keep_pid), "action": "restart_gateway_kill_keep", "result": term_status})
            start_status = _start_app(app_path)
            payload["actions"].append({"app_path": app_path, "action": "restart_gateway_open_app", "result": start_status})
            time.sleep(12.0)
            listener_pids_after2, listener_lines_after2 = _listener_pids(ports)
            processes_after2 = _ib_processes(listener_pids=listener_pids_after2)
            ok_port, selected_port, port_rows = check_port_candidates(hosts, ports)
            ok_hs, selected_hs_port, hs_rows = check_handshake_candidates(hosts, ports, int(cfg.IB_CLIENT_ID))
            payload["listener_lines_after_restart"] = listener_lines_after2
            payload["processes_after_restart"] = processes_after2
            payload["port_check"] = {"ok": ok_port, "selected_port": selected_port, "details": port_rows}
            payload["handshake_check"] = {"ok": ok_hs, "selected_port": selected_hs_port, "details": hs_rows}
        else:
            payload["actions"].append(
                {
                    "action": "restart_gateway_open_app",
                    "result": "no_candidate_app",
                    "note": "Set AION_IB_APP_PREFERRED or AION_IB_APP_CANDIDATES to known IB Gateway/TWS .app paths.",
                }
            )

    payload["ok"] = bool(payload["handshake_check"].get("ok", False))

    if not payload["ok"]:
        payload["next_steps"] = [
            "Verify only one IB Gateway/TWS process is active.",
            "Ensure API socket is enabled and not read-only.",
            f"Confirm active API port matches one of {ports}.",
        ]

    out = cfg.LOG_DIR / "ib_recover_report.json"
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

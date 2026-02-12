import datetime as dt
import json

from .. import config as cfg
from .adaptive_tuner import main as tune_main
from .backtest_walkforward import main as wf_main
from .performance_report import main as report_main
from .promotion_gate import main as gate_main

STATE = cfg.STATE_DIR / "recalibration_state.json"


def _load():
    if not STATE.exists():
        return {}
    try:
        data = json.loads(STATE.read_text())
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _save(payload):
    STATE.write_text(json.dumps(payload, indent=2))


def main() -> int:
    prior = _load()
    start = dt.datetime.now()
    force = False

    last_finished = prior.get("finished_at")
    if isinstance(last_finished, str):
        try:
            last_dt = dt.datetime.fromisoformat(last_finished)
            days_since = (start - last_dt).days
            if days_since < cfg.RECALIBRATION_MIN_DAYS_BETWEEN and not force:
                payload = {
                    "started_at": start.isoformat(),
                    "finished_at": start.isoformat(),
                    "ok": True,
                    "skipped": True,
                    "reason": f"Only {days_since} day(s) since last recalibration (min {cfg.RECALIBRATION_MIN_DAYS_BETWEEN}).",
                    "steps": [],
                }
                _save(payload)
                print(json.dumps(payload, indent=2))
                return 0
        except Exception:
            pass

    payload = {
        "started_at": start.isoformat(),
        "steps": [],
        "skipped": False,
    }

    code = wf_main()
    payload["steps"].append({"step": "walkforward", "exit_code": code})

    code = report_main()
    payload["steps"].append({"step": "performance_report", "exit_code": code})

    code = tune_main()
    payload["steps"].append({"step": "adaptive_tuner", "exit_code": code})

    code = gate_main()
    payload["steps"].append({"step": "promotion_gate", "exit_code": code})

    end = dt.datetime.now()
    payload["finished_at"] = end.isoformat()
    payload["duration_seconds"] = (end - start).total_seconds()
    payload["ok"] = all(step["exit_code"] == 0 for step in payload["steps"])

    _save(payload)
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

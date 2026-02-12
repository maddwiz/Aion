from __future__ import annotations

import datetime as dt
import json
from pathlib import Path


class RuntimeMonitor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.path = Path(cfg.LOG_DIR) / "runtime_monitor.json"
        self.state = {
            "ts": None,
            "equity_points": [],
            "confidence_points": [],
            "slippage_points": [],
            "system_events": [],
            "alerts": [],
        }
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            if isinstance(data, dict):
                self.state.update(data)
        except Exception:
            return

    def _save(self):
        self.path.write_text(json.dumps(self.state, indent=2, default=str))

    def _trim(self):
        w = self.cfg.MONITOR_SIGNAL_DRIFT_WINDOW
        for key in ["equity_points", "confidence_points", "slippage_points"]:
            arr = self.state.get(key, [])
            if len(arr) > w:
                self.state[key] = arr[-w:]
        events = self.state.get("system_events", [])
        if len(events) > 400:
            self.state["system_events"] = events[-400:]

    def record_cycle(self, equity: float, avg_conf: float):
        now = dt.datetime.now().isoformat()
        self.state["ts"] = now
        self.state.setdefault("equity_points", []).append(float(equity))
        self.state.setdefault("confidence_points", []).append(float(avg_conf))
        self._trim()
        self._save()

    def record_execution(self, slippage_bps: float):
        self.state.setdefault("slippage_points", []).append(float(slippage_bps))
        self._trim()
        self._save()

    def record_system_event(self, event_type: str, detail: str = ""):
        self.state.setdefault("system_events", []).append(
            {
                "ts": dt.datetime.now().isoformat(),
                "type": str(event_type),
                "detail": str(detail)[:220],
            }
        )
        self._trim()
        self._save()

    def _count_recent_events(self, event_type: str, window_minutes: int) -> int:
        events = self.state.get("system_events", [])
        if not events:
            return 0
        now = dt.datetime.now(dt.timezone.utc)
        count = 0
        for evt in events:
            if str(evt.get("type", "")) != event_type:
                continue
            ts_raw = evt.get("ts")
            try:
                ts = dt.datetime.fromisoformat(ts_raw)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=dt.timezone.utc)
                else:
                    ts = ts.astimezone(dt.timezone.utc)
            except Exception:
                continue
            delta = now - ts
            if delta.total_seconds() <= max(0, window_minutes) * 60:
                count += 1
        return count

    def check_alerts(self) -> list[str]:
        alerts = []
        eq = self.state.get("equity_points", [])
        cf = self.state.get("confidence_points", [])
        sl = self.state.get("slippage_points", [])

        if cf:
            avg_conf = sum(cf) / len(cf)
            if avg_conf < self.cfg.MONITOR_MIN_AVG_CONF:
                alerts.append(f"Signal confidence drift low ({avg_conf:.3f})")

        if sl:
            avg_slip = sum(sl) / len(sl)
            if avg_slip > self.cfg.MONITOR_MAX_SLIPPAGE_BPS:
                alerts.append(f"Execution slippage elevated ({avg_slip:.2f} bps)")

        if len(eq) > 12:
            recent_peak = max(eq[:-1]) if len(eq) > 1 else eq[-1]
            dd = (recent_peak - eq[-1]) / max(recent_peak, 1e-9)
            if dd > self.cfg.MONITOR_MAX_HOURLY_DD:
                alerts.append(f"Rapid drawdown alert ({dd:.2%})")

        ib_fails = self._count_recent_events("ib_connect_fail", self.cfg.MONITOR_EVENT_WINDOW_MIN)
        if ib_fails >= self.cfg.MONITOR_IB_FAIL_ALERT_COUNT:
            alerts.append(
                f"IB reconnect instability ({ib_fails} failures/{self.cfg.MONITOR_EVENT_WINDOW_MIN}m). "
                f"Verify TWS/Gateway API enabled, trusted IP, and clientId conflicts."
            )

        if alerts:
            now = dt.datetime.now(dt.timezone.utc)
            prior = self.state.get("alerts", [])
            if prior:
                last = prior[-1]
                last_msgs = last.get("messages", [])
                last_ts_raw = last.get("ts")
                try:
                    last_ts = dt.datetime.fromisoformat(last_ts_raw)
                    if last_ts.tzinfo is None:
                        last_ts = last_ts.replace(tzinfo=dt.timezone.utc)
                    else:
                        last_ts = last_ts.astimezone(dt.timezone.utc)
                except Exception:
                    last_ts = None

                same_msgs = sorted(str(m) for m in last_msgs) == sorted(str(m) for m in alerts)
                if same_msgs and last_ts is not None and (now - last_ts).total_seconds() <= 300:
                    return alerts

            self.state.setdefault("alerts", []).append(
                {
                    "ts": dt.datetime.now().isoformat(),
                    "messages": alerts,
                }
            )
            self._save()

        return alerts

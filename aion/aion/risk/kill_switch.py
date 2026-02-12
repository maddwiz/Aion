from pathlib import Path
import json


class KillSwitch:
    def __init__(
        self,
        state_path: Path,
        daily_limit: float,
        total_limit: float = None,
        max_consecutive_losses: int = 5,
    ):
        self.path = state_path
        self.daily_limit = daily_limit
        self.total_limit = total_limit
        self.max_consecutive_losses = max_consecutive_losses
        self.last_reason = ""
        self.state = {
            "day": None,
            "start_equity_day": None,
            "start_equity_total": None,
            "tripped_day": False,
            "tripped_total": False,
            "consecutive_losses": 0,
            "day_realized_pnl": 0.0,
        }

    def load(self):
        try:
            raw = json.loads(self.path.read_text())
            if isinstance(raw, dict):
                self.state.update(raw)
        except Exception:
            pass

    def save(self):
        self.path.write_text(json.dumps(self.state, indent=2))

    def reset_day(self, today: str, equity: float):
        if self.state.get("start_equity_total") is None:
            self.state["start_equity_total"] = equity
        self.state.update(
            {
                "day": today,
                "start_equity_day": equity,
                "tripped_day": False,
                "consecutive_losses": 0,
                "day_realized_pnl": 0.0,
            }
        )
        self.save()

    def hard_reset(self, today: str, equity: float):
        self.state = {
            "day": today,
            "start_equity_day": equity,
            "start_equity_total": equity,
            "tripped_day": False,
            "tripped_total": False,
            "consecutive_losses": 0,
            "day_realized_pnl": 0.0,
        }
        self.save()

    def register_trade(self, pnl: float, today: str):
        if self.state.get("day") != today:
            return
        self.state["day_realized_pnl"] = float(self.state.get("day_realized_pnl", 0.0) + pnl)
        if pnl < 0:
            self.state["consecutive_losses"] = int(self.state.get("consecutive_losses", 0) + 1)
        else:
            self.state["consecutive_losses"] = 0
        self.save()

    def check(self, today: str, equity: float) -> bool:
        self.last_reason = ""

        if self.state.get("day") != today or self.state.get("start_equity_day") is None:
            self.reset_day(today, equity)

        if self.state.get("start_equity_total") is None:
            self.state["start_equity_total"] = equity

        start_day = float(self.state.get("start_equity_day", equity))
        dd_day = (start_day - equity) / max(1e-9, start_day)

        start_total = float(self.state.get("start_equity_total", equity))
        dd_total = (start_total - equity) / max(1e-9, start_total)

        if dd_day >= self.daily_limit:
            self.state["tripped_day"] = True
            self.last_reason = f"Daily drawdown limit hit ({dd_day:.2%})"

        if self.total_limit is not None and dd_total >= self.total_limit:
            self.state["tripped_total"] = True
            self.last_reason = f"Total drawdown limit hit ({dd_total:.2%})"

        if int(self.state.get("consecutive_losses", 0)) >= self.max_consecutive_losses:
            self.state["tripped_day"] = True
            self.last_reason = f"Consecutive-loss lock ({self.state['consecutive_losses']})"

        self.save()
        return not bool(self.state.get("tripped_day") or self.state.get("tripped_total"))

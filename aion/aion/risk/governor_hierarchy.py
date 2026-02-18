"""Governor hierarchy with graduated actions."""

from __future__ import annotations

from enum import IntEnum


class GovernorAction(IntEnum):
    PASS = 0
    WARN = 1
    SCALE_DOWN = 2
    VETO = 3
    FLATTEN = 4


GOVERNOR_PRECEDENCE = {
    "kill_switch": GovernorAction.FLATTEN,
    "daily_loss_limit": GovernorAction.FLATTEN,
    "exposure_gate": GovernorAction.VETO,
    "crisis_sentinel": GovernorAction.VETO,
    "shock_mask_guard": GovernorAction.VETO,
    "regime_fracture_governor": GovernorAction.SCALE_DOWN,
    "dream_coherence": GovernorAction.SCALE_DOWN,
    "dna_stress_governor": GovernorAction.SCALE_DOWN,
    "quality_governor": GovernorAction.SCALE_DOWN,
    "confirmation_delay": GovernorAction.SCALE_DOWN,
    "calendar_mask": GovernorAction.SCALE_DOWN,
    "hive_conviction_gate": GovernorAction.SCALE_DOWN,
    "heartbeat_scaler": GovernorAction.WARN,
    "turnover_governor": GovernorAction.SCALE_DOWN,
}


def resolve_governor_action(governor_results: list[dict]) -> GovernorAction:
    """Resolve highest-priority action among failing governors."""
    action = GovernorAction.PASS
    for res in governor_results or []:
        name = str(res.get("name", "")).strip().lower()
        score = float(res.get("score", 1.0))
        threshold = float(res.get("threshold", 0.0))
        if score <= threshold:
            action = max(action, GOVERNOR_PRECEDENCE.get(name, GovernorAction.SCALE_DOWN))
    return GovernorAction(action)

from aion.risk.governor_hierarchy import GovernorAction, resolve_governor_action


def test_resolve_governor_action_returns_flatten_for_kill_switch():
    action = resolve_governor_action(
        [
            {"name": "turnover_governor", "score": 0.1, "threshold": 0.2},
            {"name": "kill_switch", "score": 0.0, "threshold": 0.1},
        ]
    )
    assert action == GovernorAction.FLATTEN


def test_resolve_governor_action_returns_pass_when_all_above_threshold():
    action = resolve_governor_action(
        [
            {"name": "quality_governor", "score": 0.9, "threshold": 0.5},
            {"name": "calendar_mask", "score": 1.0, "threshold": 0.8},
        ]
    )
    assert action == GovernorAction.PASS

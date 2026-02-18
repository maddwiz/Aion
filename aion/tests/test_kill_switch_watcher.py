from pathlib import Path

from aion.exec.kill_switch import KillSwitchWatcher


def test_kill_switch_triggers_when_file_exists(tmp_path: Path):
    watcher = KillSwitchWatcher(tmp_path, poll_seconds=0.0)
    (tmp_path / "KILL_SWITCH").write_text("KILL", encoding="utf-8")
    assert watcher.check() is True


def test_kill_switch_acknowledge_renames_file(tmp_path: Path):
    watcher = KillSwitchWatcher(tmp_path, poll_seconds=0.0)
    kill_file = tmp_path / "KILL_SWITCH"
    kill_file.write_text("", encoding="utf-8")
    watcher.acknowledge()
    assert not kill_file.exists()
    assert (tmp_path / "KILL_SWITCH.acknowledged").exists()

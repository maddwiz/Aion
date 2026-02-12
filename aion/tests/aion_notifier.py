import os
import subprocess
import time
from pathlib import Path


AION_HOME = Path(os.getenv("AION_HOME", Path(__file__).resolve().parents[1]))
LOG = Path(os.getenv("AION_LOG_DIR", AION_HOME / "logs")) / "shadow_trades.csv"


def notify(title, msg):
    script = f'display notification "{msg}" with title "{title}"'
    subprocess.run(["osascript", "-e", script], check=False)


def tail_new_lines(path, poll=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    with path.open("r") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if line:
                yield line.rstrip("\n")
            else:
                time.sleep(poll)


def main():
    for line in tail_new_lines(LOG):
        if ",BUY," in line:
            notify("Aion", f"Entry: {line}")
        elif ",EXIT," in line:
            notify("Aion", f"Exit: {line}")


if __name__ == "__main__":
    main()

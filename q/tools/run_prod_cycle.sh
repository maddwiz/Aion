#!/usr/bin/env bash
set -euo pipefail

# One-command production cycle with strict failure handling.
# Usage:
#   ./tools/run_prod_cycle.sh
# Optional env:
#   Q_MIN_HEALTH_SCORE=75 Q_MIN_GLOBAL_GOV_MEAN=0.45 ./tools/run_prod_cycle.sh

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY_BIN="${PY_BIN:-python3}"
if [[ -x ".venv/bin/python" ]]; then
  PY_BIN=".venv/bin/python"
fi

export Q_STRICT=1

echo "[Q] Running strict all-in-one cycle..."
"$PY_BIN" tools/run_all_in_one_plus.py

echo "[Q] Re-running alert gate..."
"$PY_BIN" tools/run_health_alerts.py

echo "[Q] Production cycle complete."

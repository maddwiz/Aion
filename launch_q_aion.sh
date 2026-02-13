#!/usr/bin/env bash
set -euo pipefail

# One-button launcher:
# 1) Refresh Q walk-forward summary
# 2) Export Q overlay and mirror into AION state
# 3) Start AION trader with fallback if wrapper startup is flaky

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
Q_ROOT="${Q_ROOT:-/Users/desmondpottle/Desktop/Qs Big Folder /q_v2_5_foundations}"
AION_ROOT="${AION_ROOT:-$ROOT/aion}"

Q_PY="${Q_PY:-$Q_ROOT/.venv/bin/python}"
AION_PY="${AION_PY:-/Users/desmondpottle/aion-venv/bin/python}"
IB_CLIENT_ID_LAUNCH="${IB_CLIENT_ID:-$((2000 + (RANDOM % 5000)))}"

if [[ ! -x "$Q_PY" ]]; then
  echo "ERROR: Q python not found: $Q_PY"
  exit 1
fi
if [[ ! -x "$AION_PY" ]]; then
  echo "ERROR: AION python not found: $AION_PY"
  exit 1
fi
if [[ ! -d "$AION_ROOT" ]]; then
  echo "ERROR: AION path not found: $AION_ROOT"
  exit 1
fi

echo "[1/3] Refreshing Q walk-forward table..."
cd "$Q_ROOT"
"$Q_PY" tools/walk_forward_plus.py >/dev/null

echo "[2/3] Exporting Q overlay to AION..."
"$Q_PY" tools/export_aion_signal_pack.py \
  --mirror-json "$AION_ROOT/state/q_signal_overlay.json"

echo "[3/3] Starting AION trader..."
cd "$AION_ROOT"

if pgrep -f 'aion.exec.paper_loop' >/dev/null; then
  pkill -f 'aion.exec.paper_loop' || true
  sleep 1
fi

# Primary path: normal launcher, but skip slow warmup/doctor wrappers.
nohup env \
  AION_TASK=trade \
  AION_SKIP_DOCTOR=1 \
  AION_AUTO_IB_WARMUP=0 \
  IB_CLIENT_ID="$IB_CLIENT_ID_LAUNCH" \
  ./run_aion.sh > logs/live_trade.out 2>&1 &

sleep 6
if pgrep -f 'aion.exec.paper_loop' >/dev/null; then
  echo "AION started via run_aion.sh"
  pgrep -fl 'aion.exec.paper_loop'
  exit 0
fi

echo "run_aion.sh path did not attach paper_loop quickly; using direct fallback..."
nohup env \
  AION_HOME="$AION_ROOT" \
  AION_STATE_DIR="$AION_ROOT/state" \
  AION_LOG_DIR="$AION_ROOT/logs" \
  IB_HOST="${IB_HOST:-127.0.0.1}" \
  IB_PORT="${IB_PORT:-4002}" \
  IB_CLIENT_ID="$IB_CLIENT_ID_LAUNCH" \
  "$AION_PY" -m aion.exec.paper_loop > logs/live_trade.out 2>&1 &

sleep 3
if pgrep -f 'aion.exec.paper_loop' >/dev/null; then
  echo "AION started via direct paper_loop fallback"
  pgrep -fl 'aion.exec.paper_loop'
  exit 0
fi

echo "ERROR: AION paper_loop is not running. Check:"
echo "  $AION_ROOT/logs/live_trade.out"
echo "  $AION_ROOT/logs/aion_run.log"
exit 1

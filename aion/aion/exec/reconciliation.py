"""Startup reconciliation between shadow positions and IBKR positions."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path


log = logging.getLogger(__name__)


@dataclass
class ReconciliationResult:
    passed: bool
    mismatches: list[dict]
    ibkr_positions: dict[str, int]
    shadow_positions: dict[str, int]
    action_taken: str



def _load_shadow_positions(shadow_path: Path) -> dict[str, int]:
    if not Path(shadow_path).exists():
        return {}
    try:
        raw = json.loads(Path(shadow_path).read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("Failed loading shadow state %s: %s", shadow_path, exc)
        return {}

    out: dict[str, int] = {}
    if isinstance(raw, dict):
        for sym, info in raw.items():
            try:
                qty = int((info or {}).get("qty", 0))
            except Exception:
                qty = 0
            if qty != 0:
                out[str(sym).upper()] = qty
    return out


def _load_ibkr_positions(ib_client) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in (ib_client.positions() or []):
        try:
            sym = str(p.contract.symbol).upper()
            qty = int(p.position)
        except Exception:
            continue
        if qty != 0:
            out[sym] = qty
    return out


def _save_shadow_positions(shadow_path: Path, positions: dict[str, int]) -> None:
    payload = {
        sym: {
            "qty": int(qty),
            "avg_price": 0.0,
            "last_updated": "reconciled",
        }
        for sym, qty in sorted(positions.items())
        if int(qty) != 0
    }
    p = Path(shadow_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(p)


def reconcile_on_startup(
    ib_client,
    shadow_path: Path,
    auto_fix: bool = True,
    max_auto_fix_value: float = 5000.0,
) -> ReconciliationResult:
    """Reconcile local shadow with IBKR and optionally auto-fix."""
    del max_auto_fix_value  # reserved for future notional-based cap checks

    ibkr_pos = _load_ibkr_positions(ib_client)
    shadow_pos = _load_shadow_positions(shadow_path)

    symbols = sorted(set(ibkr_pos.keys()) | set(shadow_pos.keys()))
    mismatches: list[dict] = []
    for sym in symbols:
        ib_qty = int(ibkr_pos.get(sym, 0))
        sh_qty = int(shadow_pos.get(sym, 0))
        if ib_qty == sh_qty:
            continue
        mismatch = {
            "symbol": sym,
            "ibkr_qty": ib_qty,
            "shadow_qty": sh_qty,
            "delta": ib_qty - sh_qty,
        }
        mismatches.append(mismatch)
        log.warning(
            "RECONCILIATION MISMATCH: %s ibkr=%d shadow=%d delta=%d",
            sym,
            ib_qty,
            sh_qty,
            ib_qty - sh_qty,
        )

    if not mismatches:
        log.info("Reconciliation passed: shadow matches IBKR (%d positions)", len(ibkr_pos))
        return ReconciliationResult(
            passed=True,
            mismatches=[],
            ibkr_positions=ibkr_pos,
            shadow_positions=shadow_pos,
            action_taken="none",
        )

    if auto_fix:
        _save_shadow_positions(Path(shadow_path), ibkr_pos)
        log.warning(
            "RECONCILIATION: Auto-fixed %d mismatch(es). Shadow updated to match IBKR.",
            len(mismatches),
        )
        action = "shadow_updated"
    else:
        log.critical("RECONCILIATION FAILED: %d mismatch(es) require manual review", len(mismatches))
        action = "manual_review_required"

    return ReconciliationResult(
        passed=False,
        mismatches=mismatches,
        ibkr_positions=ibkr_pos,
        shadow_positions=shadow_pos,
        action_taken=action,
    )

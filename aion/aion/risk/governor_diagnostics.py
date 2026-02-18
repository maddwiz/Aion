"""Utilities for structured governor diagnostics on the AION runtime side."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GovernorDiagnostic:
    name: str
    score: float
    min: float
    max: float
    threshold: float | None
    action: str
    reason: str
    pct_below_floor: float



def build_diagnostic(
    name: str,
    values,
    *,
    threshold: float | None = None,
    action: str = "scale",
    reason: str = "runtime",
    floor: float = 0.0,
) -> dict:
    arr = np.asarray(values, float).ravel()
    if arr.size == 0:
        arr = np.asarray([1.0], float)
    return {
        "name": str(name),
        "score": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "threshold": (None if threshold is None else float(threshold)),
        "action": str(action),
        "reason": str(reason),
        "pct_below_floor": float(np.mean(arr < float(floor))),
    }


def write_governor_diagnostics(path: Path, rows: list[dict]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(list(rows or []), indent=2), encoding="utf-8")
    tmp.replace(p)
    return p

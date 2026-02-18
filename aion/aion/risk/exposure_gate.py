"""Hard pre-trade exposure gate using live account exposure."""

from __future__ import annotations

import logging
from dataclasses import dataclass


log = logging.getLogger(__name__)


@dataclass
class ExposureCheckResult:
    allowed: bool
    reason: str
    current_exposure_pct: float
    proposed_exposure_pct: float
    limit_pct: float


def _safe_nlv(nlv: float) -> float:
    try:
        v = float(nlv)
    except Exception:
        return 0.0
    return v if v > 0.0 else 0.0


def check_exposure(
    current_positions: dict[str, float],
    proposed_symbol: str,
    proposed_value: float,
    net_liquidation: float,
    max_gross_exposure_pct: float = 0.95,
    max_single_position_pct: float = 0.20,
    max_correlated_exposure_pct: float = 0.40,
    correlated_symbols: dict[str, list[str]] | None = None,
) -> ExposureCheckResult:
    """Validate live account exposure before order submission."""
    nlv = _safe_nlv(net_liquidation)
    if nlv <= 0.0:
        return ExposureCheckResult(
            allowed=False,
            reason="net_liquidation <= 0",
            current_exposure_pct=0.0,
            proposed_exposure_pct=0.0,
            limit_pct=float(max_gross_exposure_pct),
        )

    positions = dict(current_positions or {})
    current_gross = sum(abs(float(v)) for v in positions.values())
    current_pct = current_gross / nlv

    prop_value = float(proposed_value)
    proposed_gross = current_gross + abs(prop_value)
    proposed_pct = proposed_gross / nlv

    if proposed_pct > float(max_gross_exposure_pct):
        log.warning(
            "EXPOSURE GATE VETO: gross %.1f%% > limit %.1f%% | symbol=%s value=%.2f",
            proposed_pct * 100.0,
            float(max_gross_exposure_pct) * 100.0,
            proposed_symbol,
            prop_value,
        )
        return ExposureCheckResult(
            allowed=False,
            reason=f"gross_exposure {proposed_pct:.1%} > {float(max_gross_exposure_pct):.1%}",
            current_exposure_pct=current_pct,
            proposed_exposure_pct=proposed_pct,
            limit_pct=float(max_gross_exposure_pct),
        )

    existing = abs(float(positions.get(str(proposed_symbol), 0.0)))
    single_pct = (existing + abs(prop_value)) / nlv
    if single_pct > float(max_single_position_pct):
        log.warning(
            "EXPOSURE GATE VETO: single %.1f%% > limit %.1f%% | symbol=%s",
            single_pct * 100.0,
            float(max_single_position_pct) * 100.0,
            proposed_symbol,
        )
        return ExposureCheckResult(
            allowed=False,
            reason=f"single_position {single_pct:.1%} > {float(max_single_position_pct):.1%}",
            current_exposure_pct=current_pct,
            proposed_exposure_pct=proposed_pct,
            limit_pct=float(max_single_position_pct),
        )

    if correlated_symbols:
        symbol_u = str(proposed_symbol).upper()
        positions_u = {str(k).upper(): float(v) for k, v in positions.items()}
        for group_name, group_syms in correlated_symbols.items():
            group = [str(s).upper() for s in (group_syms or [])]
            if symbol_u not in group:
                continue
            corr_exposure = sum(abs(float(positions_u.get(s, 0.0))) for s in group) + abs(prop_value)
            corr_pct = corr_exposure / nlv
            if corr_pct > float(max_correlated_exposure_pct):
                log.warning(
                    "EXPOSURE GATE VETO: correlated '%s' %.1f%% > limit %.1f%%",
                    group_name,
                    corr_pct * 100.0,
                    float(max_correlated_exposure_pct) * 100.0,
                )
                return ExposureCheckResult(
                    allowed=False,
                    reason=(
                        f"correlated_group '{group_name}' {corr_pct:.1%} > "
                        f"{float(max_correlated_exposure_pct):.1%}"
                    ),
                    current_exposure_pct=current_pct,
                    proposed_exposure_pct=proposed_pct,
                    limit_pct=float(max_correlated_exposure_pct),
                )

    return ExposureCheckResult(
        allowed=True,
        reason="passed",
        current_exposure_pct=current_pct,
        proposed_exposure_pct=proposed_pct,
        limit_pct=float(max_gross_exposure_pct),
    )

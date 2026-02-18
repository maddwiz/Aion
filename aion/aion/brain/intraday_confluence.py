"""
Confluence scoring for day_skimmer entry decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bar_engine import TimeframeData
from .session_analyzer import SessionPhase, SessionState, SessionType


@dataclass
class IntradaySignalBundle:
    symbol: str
    side: str  # LONG or SHORT
    session: SessionState | None
    patterns: dict
    bars: dict[str, TimeframeData]
    q_overlay_bias: float
    q_overlay_confidence: float


@dataclass
class ConfluenceResult:
    score: float
    entry_allowed: bool
    reasons: list[str]
    category_scores: dict[str, float]


def _latest_close(bars: dict[str, TimeframeData], tf: str) -> float | None:
    tfd = bars.get(tf)
    if tfd is None or tfd.bars is None or tfd.bars.empty:
        return None
    try:
        return float(tfd.bars["close"].iloc[-1])
    except Exception:
        return None


def score_intraday_entry(bundle: IntradaySignalBundle, cfg) -> ConfluenceResult:
    scores: dict[str, float] = {}
    reasons: list[str] = []
    side = str(bundle.side or "").strip().upper()
    if side not in {"LONG", "SHORT"}:
        return ConfluenceResult(score=0.0, entry_allowed=False, reasons=["Invalid side"], category_scores={})

    # Category 1: session structure.
    s = bundle.session
    session_score = 0.0
    if s is not None:
        session_score = float(np.clip(s.aggression_scalar, 0.0, 1.0))
        if s.phase == SessionPhase.OPENING_DRIVE:
            reasons.append("Opening drive phase (high edge)")
        elif s.phase == SessionPhase.RANGE_EXTENSION:
            reasons.append("Range extension phase")
        if s.session_type == SessionType.TREND_DAY:
            trend_dir = 1 if (s.levels.session_high - s.levels.open_price) > (s.levels.open_price - s.levels.session_low) else -1
            aligned = (side == "LONG" and trend_dir == 1) or (side == "SHORT" and trend_dir == -1)
            if aligned:
                session_score = min(1.0, session_score + 0.15)
                reasons.append("Aligned with trend day direction")
            else:
                session_score = max(0.0, session_score - 0.30)
                reasons.append("Against trend day direction (penalized)")
    scores["session_structure"] = float(np.clip(session_score, 0.0, 1.0))

    # Category 2: pattern confluence.
    pattern_score = 0.0
    pattern_count = 0
    for name, pat in (bundle.patterns or {}).items():
        if not isinstance(pat, dict) or not bool(pat.get("detected", False)):
            continue
        pat_dir = int(pat.get("direction", 0))
        pat_str = float(np.clip(float(pat.get("strength", 0.5)), 0.0, 1.0))
        if (side == "LONG" and pat_dir > 0) or (side == "SHORT" and pat_dir < 0):
            pattern_score += pat_str * 0.25
            pattern_count += 1
            reasons.append(f"Pattern: {name} ({pat_str:.0%} strength)")
        elif pat_dir != 0:
            pattern_score -= 0.10
            reasons.append(f"Conflicting pattern: {name}")
    pattern_score = float(np.clip(pattern_score, 0.0, 1.0))
    if pattern_count >= 2:
        pattern_score = min(1.0, pattern_score + 0.12)
        reasons.append(f"Multi-pattern confluence ({pattern_count} patterns)")
    scores["pattern_confluence"] = pattern_score

    # Category 3: multi-timeframe alignment.
    mtf_agrees = 0
    for tf in ["5m", "15m", "1H"]:
        tfd = bundle.bars.get(tf)
        if tfd is None or tfd.ema_fast is None or tfd.ema_slow is None or len(tfd.ema_fast) == 0:
            continue
        ema_bull = float(tfd.ema_fast.iloc[-1]) > float(tfd.ema_slow.iloc[-1])
        rsi_v = float(tfd.rsi.iloc[-1]) if (tfd.rsi is not None and len(tfd.rsi) > 0) else 50.0
        if side == "LONG":
            if ema_bull and rsi_v > 45:
                mtf_agrees += 1
        else:
            if (not ema_bull) and rsi_v < 55:
                mtf_agrees += 1
    if mtf_agrees >= 3:
        mtf_score = 0.95
        reasons.append("All timeframes aligned")
    elif mtf_agrees == 2:
        mtf_score = 0.65
        reasons.append("2/3 timeframes aligned")
    elif mtf_agrees == 1:
        mtf_score = 0.30
        reasons.append("Only 1/3 timeframes aligned")
    else:
        mtf_score = 0.05
        reasons.append("No timeframe alignment")
    scores["multi_timeframe"] = float(mtf_score)

    # Category 4: key level interaction.
    level_score = 0.0
    if s is not None:
        current = _latest_close(bundle.bars, "1m")
        if current is None:
            current = 0.0
        tfd5 = bundle.bars.get("5m")
        atr = (
            float(tfd5.atr.iloc[-1])
            if (tfd5 is not None and tfd5.atr is not None and len(tfd5.atr) > 0)
            else 0.01
        )
        near = 0.3 * max(atr, 1e-9)
        lvl = s.levels

        if side == "LONG":
            near_vwap = abs(current - lvl.vwap) < near and current >= lvl.vwap
            near_poc = abs(current - lvl.poc) < near and current >= lvl.poc
            near_pd_high = abs(current - lvl.prior_day_high) < near and current >= lvl.prior_day_high
            above_va = current > lvl.value_area_high
            if near_vwap:
                level_score += 0.30
                reasons.append("Near VWAP support (long)")
            if near_poc:
                level_score += 0.25
                reasons.append("Near POC support (long)")
            if above_va:
                level_score += 0.20
                reasons.append("Above value area (long)")
            if near_pd_high:
                level_score += 0.20
                reasons.append("Breaking prior day high")
        else:
            near_vwap = abs(current - lvl.vwap) < near and current <= lvl.vwap
            near_poc = abs(current - lvl.poc) < near and current <= lvl.poc
            near_pd_low = abs(current - lvl.prior_day_low) < near and current <= lvl.prior_day_low
            below_va = current < lvl.value_area_low
            if near_vwap:
                level_score += 0.30
                reasons.append("Near VWAP resistance (short)")
            if near_poc:
                level_score += 0.25
                reasons.append("Near POC resistance (short)")
            if below_va:
                level_score += 0.20
                reasons.append("Below value area (short)")
            if near_pd_low:
                level_score += 0.20
                reasons.append("Breaking prior day low")
    scores["key_levels"] = float(np.clip(level_score, 0.0, 1.0))

    # Category 5: Q overlay alignment.
    q_score = 0.5
    bias = float(bundle.q_overlay_bias)
    conf = float(np.clip(bundle.q_overlay_confidence, 0.0, 1.0))
    if side == "LONG" and bias > 0.1:
        q_score = 0.5 + 0.5 * min(1.0, bias * conf)
        reasons.append(f"Q overlay supports long (bias={bias:.2f})")
    elif side == "SHORT" and bias < -0.1:
        q_score = 0.5 + 0.5 * min(1.0, abs(bias) * conf)
        reasons.append(f"Q overlay supports short (bias={bias:.2f})")
    elif (side == "LONG" and bias < -0.3) or (side == "SHORT" and bias > 0.3):
        q_score = max(0.0, 0.5 - 0.4 * abs(bias) * conf)
        reasons.append(f"Q overlay opposes entry (bias={bias:.2f})")
    scores["q_overlay"] = float(np.clip(q_score, 0.0, 1.0))

    # Category 6: volume + momentum.
    vm_score = 0.0
    tfd5 = bundle.bars.get("5m")
    if tfd5 is not None and tfd5.bars is not None and not tfd5.bars.empty and tfd5.volume_ma is not None and len(tfd5.volume_ma) > 0:
        vol_now = float(tfd5.bars["volume"].iloc[-1]) if "volume" in tfd5.bars.columns else 0.0
        vol_avg = float(tfd5.volume_ma.iloc[-1])
        if vol_avg > 0:
            vol_ratio = vol_now / vol_avg
            if vol_ratio > 1.3:
                vm_score += 0.40
                reasons.append("Above-average volume (5m)")
            elif vol_ratio < 0.5:
                vm_score -= 0.20
                reasons.append("Low volume warning")

        close_5m = tfd5.bars["close"]
        if len(close_5m) >= 3:
            mom = float(close_5m.iloc[-1] - close_5m.iloc[-3])
            if (side == "LONG" and mom > 0) or (side == "SHORT" and mom < 0):
                vm_score += 0.35
                reasons.append("Short-term momentum confirms")
            elif (side == "LONG" and mom < 0) or (side == "SHORT" and mom > 0):
                vm_score -= 0.15
                reasons.append("Momentum opposes entry")
    scores["volume_momentum"] = float(np.clip(vm_score, 0.0, 1.0))

    weights = {
        "session_structure": 0.20,
        "pattern_confluence": 0.25,
        "multi_timeframe": 0.20,
        "key_levels": 0.15,
        "q_overlay": 0.10,
        "volume_momentum": 0.10,
    }
    total = float(np.clip(sum(scores.get(k, 0.0) * w for k, w in weights.items()), 0.0, 1.0))
    threshold = float(getattr(cfg, "SKIMMER_ENTRY_THRESHOLD", 0.58))
    entry_allowed = bool(total >= threshold and (s is None or s.trade_allowed))

    return ConfluenceResult(
        score=total,
        entry_allowed=entry_allowed,
        reasons=reasons,
        category_scores={k: float(v) for k, v in scores.items()},
    )


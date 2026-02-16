from __future__ import annotations


def _to_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "on"}


def _to_float(x, default: float | None = None):
    try:
        v = float(x)
    except Exception:
        return default
    if v != v:  # NaN
        return default
    return v


def _uniq(items):
    out = []
    seen = set()
    for raw in items:
        s = str(raw).strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


def runtime_decision_summary(
    runtime_controls: dict | None,
    external_overlay_runtime: dict | None = None,
    external_overlay_risk_flags: list[str] | None = None,
) -> dict:
    rc = runtime_controls if isinstance(runtime_controls, dict) else {}
    ext_rt = external_overlay_runtime if isinstance(external_overlay_runtime, dict) else {}
    ext_flags = external_overlay_risk_flags if isinstance(external_overlay_risk_flags, list) else []
    ext_flags = [str(x).strip().lower() for x in ext_flags if str(x).strip()]

    blocked_reasons = []
    if _to_bool(rc.get("killswitch_block_new_entries", False)):
        blocked_reasons.append("killswitch")
    if _to_bool(rc.get("policy_block_new_entries", False)):
        blocked_reasons.append("risk_policy")
    if _to_bool(rc.get("overlay_block_new_entries", False)):
        blocked_reasons.append("external_overlay")
        for r in rc.get("overlay_block_reasons", []) if isinstance(rc.get("overlay_block_reasons", []), list) else []:
            blocked_reasons.append(f"external_overlay:{str(r).strip().lower()}")
    if _to_bool(rc.get("memory_feedback_block_new_entries", False)):
        blocked_reasons.append("memory_feedback")
        for r in rc.get("memory_feedback_reasons", []) if isinstance(rc.get("memory_feedback_reasons", []), list) else []:
            blocked_reasons.append(f"memory_feedback:{str(r).strip().lower()}")
    if _to_bool(rc.get("exec_governor_block_new_entries", False)):
        blocked_reasons.append("execution_governor")
    if _to_bool(ext_rt.get("stale", False)):
        blocked_reasons.append("external_overlay:stale")

    throttle_reasons = []
    score = 0
    pos_scale = _to_float(rc.get("external_position_risk_scale"), None)
    if pos_scale is not None:
        if pos_scale <= 0.70:
            score += 2
            throttle_reasons.append("position_risk_scale_critical")
        elif pos_scale <= 0.90:
            score += 1
            throttle_reasons.append("position_risk_scale_tight")

    rt_scale = _to_float(rc.get("external_runtime_scale"), None)
    if rt_scale is not None:
        if rt_scale <= 0.75:
            score += 2
            throttle_reasons.append("overlay_runtime_scale_critical")
        elif rt_scale <= 0.90:
            score += 1
            throttle_reasons.append("overlay_runtime_scale_tight")

    exec_state = str(rc.get("exec_governor_state", "unknown")).strip().lower()
    if exec_state == "alert":
        score += 2
        throttle_reasons.append("execution_governor_alert")
    elif exec_state == "warn":
        score += 1
        throttle_reasons.append("execution_governor_warn")

    mem_state = str(rc.get("memory_feedback_status", "unknown")).strip().lower()
    if mem_state == "alert":
        score += 2
        throttle_reasons.append("memory_feedback_alert")
    elif mem_state == "warn":
        score += 1
        throttle_reasons.append("memory_feedback_warn")

    if "fracture_alert" in ext_flags or "drift_alert" in ext_flags:
        score += 1
        throttle_reasons.append("overlay_risk_flag_alert")

    if score >= 3:
        throttle_state = "alert"
    elif score >= 1:
        throttle_state = "warn"
    else:
        throttle_state = "normal"

    blocked_reasons = _uniq(blocked_reasons)
    throttle_reasons = _uniq(throttle_reasons)
    return {
        "entry_blocked": bool(len(blocked_reasons) > 0),
        "entry_block_reasons": blocked_reasons,
        "throttle_state": throttle_state,
        "throttle_reasons": throttle_reasons,
    }

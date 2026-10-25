"""Diagnostic execution event building, full ledger formatting, and evidence frame selection."""

from __future__ import annotations
import json
import math
from typing import Any
from PIL import Image

try:
    from .route_progress import _turn_blocks, instruction_turn_witness, turn_directions, turn_witness
except ImportError:
    from route_progress import _turn_blocks, instruction_turn_witness, turn_directions, turn_witness

_VALID_VIEWS = ("front", "left", "right", "back")
_VALID_PHASES = ("BEFORE", "AFTER")
_VALID_ACTIONS = {"turn", "move_forward", "stop", "wait", "look_up", "look_down", "recovery"}
_TURN_MAP = {"left": "LEFT", "right": "RIGHT", "around": "BACK", "back": "BACK"}
_MAX_INT = (1 << 63) - 1


def _valid_int(val: Any) -> int | None:
    """Return non-negative 64-bit integer or None for bool, negative, overflow, or non-int."""
    if isinstance(val, bool) or not isinstance(val, int) or val < 0 or val > _MAX_INT:
        return None
    return val


def _finite_float(val: Any) -> float | None:
    """Return finite float or None when missing, boolean, non-numeric, non-finite, or overflowing."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    try:
        f = float(val)
    except (ValueError, TypeError, OverflowError):
        return None
    return f if math.isfinite(f) else None


def _finite_yaw_delta(val: Any) -> float | None:
    """Return finite measured yaw_delta in [-pi, pi] (+0.001 rounding margin) or None."""
    f = _finite_float(val)
    return f if f is not None and abs(f) <= math.pi + 0.001 else None


def _valid_str(val: Any) -> str | None:
    """Return stripped non-empty string or None."""
    if not isinstance(val, str):
        return None
    s = val.strip()
    return s if s else None


def _valid_id(val: Any) -> int | str | None:
    """Return valid non-negative int or non-empty string identifier, else None."""
    i = _valid_int(val)
    return i if i is not None else _valid_str(val)


def _clean_pose(val: Any) -> list[float] | None:
    """Return 2D/3D pose as list of finite floats, or None if missing or malformed."""
    if not isinstance(val, (list, tuple)) or not (2 <= len(val) <= 3):
        return None
    coords = [_finite_float(x) for x in val]
    return [round(c, 4) for c in coords if c is not None] if all(c is not None for c in coords) else None


def _check_limit(val: Any, name: str) -> int:
    """Validate non-negative integer limit; reject bool, None, negative, or overflow values."""
    i = _valid_int(val)
    if i is None:
        raise ValueError(f"{name} must be a non-negative integer")
    return i


def _fmt_num(val: float | None) -> str:
    """Format finite float to 3 decimal places or 'null' when unknown."""
    return f"{val:.3f}" if val is not None else "null"


def _fmt_pose(val: list[float] | None) -> str:
    """Format finite pose list as compact JSON or 'null' when unknown."""
    return json.dumps(val, separators=(",", ":")) if val is not None else "null"


def _validate_instruction_turn(row: dict[str, Any], yaw: float | None) -> dict[str, Any] | None:
    """Validate controller instruction_turn against request, verb direction, and measured yaw."""
    cmd, wp = row.get("instruction_turn"), row.get("waypoint_execution")
    if (not isinstance(cmd, dict) or row.get("action_type") != "turn" or yaw is None
            or (isinstance(wp, dict) and wp.get("phase") == "aligning")):
        return None
    sub, cond_id, text = row.get("subtask_id"), cmd.get("condition_id"), cmd.get("text")
    if (_valid_str(sub) is None or cmd.get("subtask_id") != sub
            or _valid_str(cond_id) is None or _valid_str(text) is None):
        return None
    dirs = turn_directions(text)
    if len(dirs) != 1:
        return None
    norm_dir = _TURN_MAP.get(dirs[0])
    if not norm_dir or cmd.get("direction") != norm_dir:
        return None
    if row.get("requested_action") != {"type": "turn", "direction": norm_dir}:
        return None
    probe = {"step": 0, "action_type": "turn", "yaw_delta": yaw, "recovery": False, "rejected_by_rollback": False}
    if not turn_witness(text, [probe], [probe])[0]:
        return None
    le = cmd.get("landmark_evidence")
    view = le.get("view") if isinstance(le, dict) else None
    return {
        "id": cond_id, "condition_id": cond_id, "subtask_id": sub, "text": text,
        "direction": norm_dir, "landmark_view": view if view in _VALID_VIEWS else None,
    }


def _build_summary_text(ev: dict[str, Any]) -> str:
    """Build concise measured summary string without VLM prose or request fabrication."""
    st = ev.get("step")
    parts = [
        f"step={st}" if st is not None else f"ctrl_step={ev.get('controller_step')}",
        f"type={ev.get('event_type', 'unknown')}",
        f"status={ev.get('status', 'unknown')}",
    ]
    if ev.get("condition_id"):
        parts.append(f"cond={ev['condition_id']}")
    if ev.get("turn_direction"):
        parts.append(f"dir={ev['turn_direction']}")
    if ev.get("waypoint_id") is not None:
        parts.append(f"wp={ev['waypoint_id']}")
    if ev.get("motion_distance") is not None:
        parts.append(f"dist={_fmt_num(ev['motion_distance'])}m")
    if ev.get("yaw_delta") is not None:
        parts.append(f"yaw={_fmt_num(ev['yaw_delta'])}rad")
    if ev.get("height_change"):
        parts.append(f"height={ev['height_change']}({_fmt_num(ev.get('height_delta'))}m)")
    if ev.get("platform_supported"):
        parts.append("platform=true")
    if ev.get("recovery"):
        parts.append(f"recovery_source={ev.get('recovery_source') or 'unknown'}")
    if ev.get("rejected_by_rollback"):
        parts.append("rejected_by_rollback=true")
    return " ".join(parts)


def _refresh_event(ev: dict[str, Any]) -> dict[str, Any]:
    """Sync dynamic rollback/recovery flags from source row and linked prior turn."""
    prior = ev.get("_prior_turn")
    if isinstance(prior, dict):
        _refresh_event(prior)
    row = ev.get("_source_row")
    if isinstance(row, dict):
        rej = row.get("rejected_by_rollback") is True
        rec = (row.get("recovery") is True) or bool(ev.get("recovery"))
        if rej != bool(ev.get("rejected_by_rollback")) or rec != bool(ev.get("recovery")):
            ev["rejected_by_rollback"] = rej
            ev["recovery"] = rec
            if rec and not ev.get("recovery_source"):
                ev["recovery_source"] = _valid_str(row.get("recovery_source")) or "unknown"
            if rej or rec:
                ev["diagnostic_only"] = True
                ev["route_evidence"] = False
                ev["owned_condition_id"] = None
                if ev.get("departure_accepted") is True:
                    ev["departure_accepted"] = False
                cls = list(ev.get("classifications") or [])
                tag = "rejected_by_rollback" if rej else "recovery"
                if tag not in cls:
                    cls.insert(0, tag)
                ev["classifications"] = cls
                ev["status"] = tag
                ev["summary"] = _build_summary_text(ev)
    if isinstance(prior, dict) and any(prior.get(k) for k in ("rejected_by_rollback", "recovery", "undone_in_place")):
        if ev.get("departure_accepted") is not False or ev.get("route_evidence"):
            ev["departure_accepted"] = False
            ev["diagnostic_only"] = True
            ev["route_evidence"] = False
            ev["owned_condition_id"] = None
            cls = list(ev.get("classifications") or [])
            if "rejected_turn_departure" not in cls:
                cls.append("rejected_turn_departure")
            ev["classifications"] = cls
            if not ev.get("rejected_by_rollback") and not ev.get("recovery"):
                ev["status"] = "rejected_departure"
            ev["summary"] = _build_summary_text(ev)
    return ev


def build_execution_events(
    ledger: Any, recovery_events: Any = (), scope_key: str | None = None
) -> list[dict[str, Any]]:
    """Build transient diagnostic execution events from ledger and recovery events.

    Returned events are transient per verifier call and should be rebuilt on each
    verifier invocation; select_event_summaries and select_evidence_frames also
    refresh dynamic rollback/recovery flags from source rows.
    """
    raw_rows = [r for r in (ledger or ()) if isinstance(r, dict)]
    valid_sorted = sorted(
        ((st, i, r) for i, r in enumerate(raw_rows) if (st := _valid_int(r.get("step"))) is not None),
        key=lambda x: (x[0], x[1]),
    )
    it_valid = iter(valid_sorted)
    ordered_rows = [next(it_valid)[2] if _valid_int(r.get("step")) is not None else r for r in raw_rows]

    clean_rows = [
        {
            **r, "step": st,
            "action_type": r.get("action_type") if isinstance(r.get("action_type"), str) and r.get("action_type") in _VALID_ACTIONS else None,
            "yaw_delta": _finite_yaw_delta(r.get("yaw_delta")),
            "motion_distance": d if (d := _finite_float(r.get("motion_distance"))) is not None and d >= 0.0 else None,
            "recovery": r.get("recovery") is True,
            "rejected_by_rollback": r.get("rejected_by_rollback") is True,
        }
        for st, _, r in valid_sorted
    ]
    in_place_undo_map: dict[int, int] = {}
    undone_steps: set[int] = set()
    undo_cond_map: dict[int, str] = {}
    for blk in _turn_blocks(clean_rows, clean_rows):
        blk_valid_turns: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for idx, cr in enumerate(blk):
            st, y_cur = cr["step"], cr["yaw_delta"]
            for prev_r, v_prev in blk_valid_turns:
                y_prev = prev_r["yaw_delta"]
                if (prev_r["step"] in undone_steps and y_cur is not None and y_prev is not None
                        and abs(y_cur) >= 0.05 and (y_cur * y_prev < 0)
                        and not instruction_turn_witness(v_prev, [prev_r], blk[:idx + 1], v_prev["subtask_id"])):
                    in_place_undo_map[st] = prev_r["step"]
                    undo_cond_map[st] = v_prev["condition_id"]
                    break
            v_it = _validate_instruction_turn(cr, y_cur)
            if v_it is not None:
                if not instruction_turn_witness(v_it, [cr], clean_rows, v_it["subtask_id"]):
                    undone_steps.add(st)
                blk_valid_turns.append((cr, v_it))

    rec_by_ctrl: dict[int | None, list[dict[str, Any]]] = {}
    for rec in (recovery_events or ()):
        if not isinstance(rec, dict):
            continue
        r_sub = _valid_str(rec.get("subtask_id"))
        if scope_key is not None and r_sub is not None and r_sub != scope_key:
            continue
        c_step = _valid_int(rec.get("controller_step"))
        if c_step is None:
            c_step = _valid_int(rec.get("step"))
        src = _valid_str(rec.get("recovery_source")) or "unknown"
        ev_rec: dict[str, Any] = {
            "step": None, "controller_step": c_step, "subtask_id": r_sub,
            "event_type": "recovery", "classifications": ["recovery"],
            "status": _valid_str(rec.get("status")) or "recovery",
            "condition_id": None, "owned_condition_id": None,
            "turn_direction": None, "turn_step": None, "departure_accepted": None,
            "in_place_undo": False, "undone_in_place": False, "undo_of_step": None,
            "stale_instruction_turn": False, "waypoint_id": None, "action_type": "recovery",
            "pose_before": None, "pose": None, "yaw_before": None, "yaw_after": None,
            "motion_distance": None, "yaw_delta": None, "height_delta": None, "height_change": None,
            "platform_supported": False, "blocked": False, "recovery": True, "recovery_source": src,
            "rejected_by_rollback": False, "diagnostic_only": True, "route_evidence": False,
            "observation_id_before": None, "observation_id_after": None, "landmark_view": None,
        }
        ev_rec["summary"] = _build_summary_text(ev_rec)
        rec_by_ctrl.setdefault(c_step, []).append(ev_rec)

    valid_turns: dict[int, dict[str, Any]] = {}
    consumed_turns: set[int] = set()
    out: list[dict[str, Any]] = []
    for r in ordered_rows:
        st = _valid_int(r.get("step"))
        c_step = _valid_int(r.get("controller_step"))
        sub = _valid_str(r.get("subtask_id"))
        act = r.get("action_type") if isinstance(r.get("action_type"), str) and r.get("action_type") in _VALID_ACTIONS else None
        obs_b, obs_a = _valid_int(r.get("observation_id_before")), _valid_int(r.get("observation_id_after"))
        pose_b = _clean_pose(r.get("pose_before"))
        pose_a = _clean_pose(r.get("pose") if r.get("pose") is not None else r.get("pose_after"))
        yaw_b, yaw_a = _finite_float(r.get("yaw_before")), _finite_float(r.get("yaw_after"))
        dist = d if (d := _finite_float(r.get("motion_distance"))) is not None and d >= 0.0 else None
        req_dist = _finite_float(r.get("requested_distance"))
        yaw, h_delta = _finite_yaw_delta(r.get("yaw_delta")), _finite_float(r.get("height_delta"))
        is_rec, is_rej = r.get("recovery") is True, r.get("rejected_by_rollback") is True

        if st is None:
            is_blk = (not is_rec) and (r.get("blocked") is True)
            et = "recovery" if is_rec else ("blocked" if is_blk else "unknown")
            status = "rejected_by_rollback" if is_rej else ("recovery" if is_rec else ("blocked" if is_blk else "unknown"))
            cls = [n for f, n in ((is_rej, "rejected_by_rollback"), (is_rec, "recovery"), (is_blk, "blocked")) if f] or ["unknown"]
            ev_inv: dict[str, Any] = {
                "step": None, "controller_step": c_step, "subtask_id": sub, "event_type": et,
                "classifications": cls, "status": status, "condition_id": None, "owned_condition_id": None,
                "turn_direction": None, "turn_step": None, "departure_accepted": None,
                "in_place_undo": False, "undone_in_place": False, "undo_of_step": None,
                "stale_instruction_turn": r.get("instruction_turn") is not None, "waypoint_id": None,
                "action_type": act, "pose_before": pose_b, "pose": pose_a, "yaw_before": yaw_b, "yaw_after": yaw_a,
                "motion_distance": dist, "yaw_delta": yaw, "height_delta": h_delta, "height_change": None,
                "platform_supported": False, "blocked": is_blk, "recovery": is_rec,
                "recovery_source": (_valid_str(r.get("recovery_source")) or "unknown") if is_rec else None,
                "rejected_by_rollback": is_rej, "diagnostic_only": True, "route_evidence": False,
                "observation_id_before": obs_b, "observation_id_after": obs_a, "landmark_view": None,
                "_source_row": r,
            }
            ev_inv["summary"] = _build_summary_text(ev_inv)
            if scope_key is None or sub == scope_key or sub is None:
                out.append(ev_inv)
            continue

        is_blk = (not is_rec) and (
            r.get("blocked") is True
            or (act == "move_forward" and dist is not None and dist < 0.02 and req_dist is not None and req_dist >= 0.08)
        )
        has_trans = (not is_blk) and (not is_rec) and (dist is not None and dist >= 0.02)
        h_chg = None if (is_blk or is_rec or h_delta is None) else ("ascent" if h_delta > 0.04 else ("descent" if h_delta < -0.04 else None))
        platform = bool(r.get("landing_supported") is True and not is_blk and not is_rec and not is_rej)
        v_it = _validate_instruction_turn(r, yaw) if act == "turn" else None
        stale_it = r.get("instruction_turn") is not None and v_it is None
        undone, undo_of = st in undone_steps, in_place_undo_map.get(st)
        in_undo = undo_of is not None
        wp = r.get("waypoint_execution")
        is_wp = bool(v_it is None and isinstance(wp, dict) and wp.get("phase") == "aligning" and act == "turn" and yaw is not None and abs(yaw) >= 0.05)
        wp_id = (_valid_id(wp.get("waypoint_id")) if _valid_id(wp.get("waypoint_id")) is not None else _valid_id(wp.get("action_id"))) if is_wp else None

        td = r.get("turn_departure")
        has_td = isinstance(td, dict)
        t_step = _valid_int(td.get("turn_step")) if has_td else None
        td_cond = _valid_str(td.get("condition_id")) if has_td else None
        prior_ev = valid_turns.get(t_step) if (t_step is not None and t_step < st) else None
        prior_ok = bool(
            prior_ev is not None and td_cond is not None and prior_ev.get("owned_condition_id") == td_cond
            and prior_ev.get("subtask_id") == sub and t_step not in consumed_turns
            and not any(prior_ev.get(k) for k in ("undone_in_place", "recovery", "rejected_by_rollback"))
        )
        dep_acc = bool(has_td and td.get("accepted") is True and has_trans and prior_ok and not is_rec and not is_rej)
        if dep_acc and t_step is not None:
            consumed_turns.add(t_step)

        cls = []
        for flag, name in (
            (is_rej, "rejected_by_rollback"), (is_rec, "recovery"), (v_it is not None, "instruction_turn"),
            (undone, "undone_in_place"), (in_undo, "in_place_undo"), (is_wp, "waypoint_alignment"),
            (has_td, "turn_departure"), (has_td and not dep_acc, "rejected_turn_departure"),
            (is_blk, "blocked"), (platform, "platform"), (h_chg is not None, h_chg or ""), (has_trans, "translation"),
        ):
            if flag and name:
                cls.append(name)
        if act == "turn" and v_it is None and not is_wp and not in_undo:
            cls.append("unattributed_turn")

        unknown_meas = (act == "move_forward" and dist is None) or (act == "turn" and yaw is None) or (act is None and dist is None and yaw is None and h_delta is None)
        if is_rec:
            et = "recovery"
        elif v_it is not None:
            et = "instruction_turn"
        elif in_undo:
            et = "in_place_undo"
        elif is_wp:
            et = "waypoint_alignment"
        elif has_td:
            et = "turn_departure"
        elif is_blk:
            et = "blocked"
        elif platform:
            et = "platform"
        elif h_chg is not None:
            et = h_chg
        elif has_trans:
            et = "translation"
        elif act == "turn":
            et = "unattributed_turn"
        elif unknown_meas:
            et = "unknown"
        else:
            et = "noop"
        if not cls and et == "unknown":
            cls.append("unknown")

        diag = bool(
            is_rec or is_rej or is_blk or undone or in_undo or (has_td and not dep_acc)
            or stale_it or unknown_meas or (et in ("unattributed_turn", "noop", "unknown") and not has_trans and not platform and not h_chg)
        )
        route_ev = not diag
        owned_c = v_it["condition_id"] if (v_it and route_ev) else (td_cond if dep_acc else None)
        cond_id = owned_c or (v_it["condition_id"] if v_it else (td_cond if (has_td and prior_ev is not None) else undo_cond_map.get(st)))
        t_dir = v_it["direction"] if v_it else (
            ("BACK" if abs(yaw) >= math.radians(150) else ("LEFT" if yaw < 0 else "RIGHT"))
            if (yaw is not None and abs(yaw) >= 0.05) else None
        )
        status = (
            "rejected_by_rollback" if is_rej else "recovery" if is_rec else "undone_in_place" if undone
            else "in_place_undo" if in_undo else "rejected_departure" if (has_td and not dep_acc)
            else "blocked" if is_blk else "accepted" if dep_acc else ("unknown" if unknown_meas else "ok")
        )
        ev_data: dict[str, Any] = {
            "step": st, "controller_step": c_step, "subtask_id": sub, "event_type": et, "classifications": cls,
            "status": status, "condition_id": cond_id, "owned_condition_id": owned_c,
            "turn_direction": t_dir, "turn_step": t_step, "departure_accepted": dep_acc if has_td else None,
            "in_place_undo": in_undo, "undone_in_place": undone, "undo_of_step": undo_of,
            "stale_instruction_turn": stale_it, "waypoint_id": wp_id, "action_type": act,
            "pose_before": pose_b, "pose": pose_a, "yaw_before": yaw_b, "yaw_after": yaw_a,
            "motion_distance": dist, "yaw_delta": yaw, "height_delta": h_delta, "height_change": h_chg,
            "platform_supported": platform, "blocked": is_blk, "recovery": is_rec,
            "recovery_source": (_valid_str(r.get("recovery_source")) or "unknown") if is_rec else None,
            "rejected_by_rollback": is_rej, "diagnostic_only": diag, "route_evidence": route_ev,
            "observation_id_before": obs_b, "observation_id_after": obs_a,
            "landmark_view": v_it["landmark_view"] if v_it else None,
            "_source_row": r, "_prior_turn": prior_ev if has_td else None,
        }
        ev_data["summary"] = _build_summary_text(ev_data)
        if v_it is not None:
            valid_turns[st] = ev_data
        if scope_key is None or sub == scope_key or (sub is None and (is_rec or is_rej)):
            out.append(ev_data)
        if c_step is not None and c_step in rec_by_ctrl:
            out.extend(rec_by_ctrl.pop(c_step))
    for rem in rec_by_ctrl.values():
        out.extend(rem)
    return out


def select_event_summaries(events: Any, current_condition_ids: Any = (), limit: int = 12) -> list[dict[str, Any]]:
    """Select prioritized metadata summaries in chronological order without images or private refs."""
    cap = _check_limit(limit, "limit")
    if cap == 0:
        return []
    conds = {c for c in (current_condition_ids or ()) if _valid_str(c) is not None}
    synced = [_refresh_event(e) for e in (events or ()) if isinstance(e, dict)]

    def _tier(ev: dict[str, Any]) -> int:
        et = ev.get("event_type")
        if ev.get("condition_id") in conds:
            return 0
        if et in ("instruction_turn", "turn_departure", "in_place_undo"):
            return 1
        if et in ("blocked", "recovery", "platform", "ascent", "descent") or ev.get("rejected_by_rollback"):
            return 2
        return 3 if et == "waypoint_alignment" else (4 if et == "translation" else 5)

    if len(synced) > cap:
        chosen: set[int] = set()
        seen_types: set[str] = set()
        ranked = sorted(range(len(synced)), key=lambda i: (_tier(synced[i]), -i))
        for i in ranked:
            et = str(synced[i].get("event_type"))
            if _tier(synced[i]) <= 3 and et not in seen_types and len(chosen) < cap:
                chosen.add(i)
                seen_types.add(et)
        for i in ranked:
            if len(chosen) >= cap:
                break
            chosen.add(i)
        synced = [synced[i] for i in sorted(chosen)]
    return [
        {k: (list(v) if isinstance(v, list) else v) for k, v in ev.items() if not k.startswith("_")}
        for ev in synced
    ]


def format_execution_ledger(ledger: Any) -> str:
    """Format full execution ledger preserving every step, route geometry, and unknown nulls."""
    events = build_execution_events(ledger)
    if not events:
        return "Execution Ledger: empty"
    lines = []
    for ev in events:
        _refresh_event(ev)
        st_s = str(ev["step"]) if ev["step"] is not None else "null"
        cs_s = str(ev["controller_step"]) if ev["controller_step"] is not None else "null"
        ts_s = str(ev["turn_step"]) if ev["turn_step"] is not None else "null"
        td_s = str(ev["departure_accepted"]).lower() if ev["departure_accepted"] is not None else "null"
        wp_s = str(ev["waypoint_id"]) if ev["waypoint_id"] is not None else "null"
        ob_s = str(ev["observation_id_before"]) if ev["observation_id_before"] is not None else "null"
        oa_s = str(ev["observation_id_after"]) if ev["observation_id_after"] is not None else "null"
        lines.append(
            f"step={st_s} ctrl_step={cs_s} subtask={ev['subtask_id'] or 'null'} "
            f"action={ev['action_type'] or 'unknown'} event={ev['event_type']} status={ev['status']} "
            f"pose_before={_fmt_pose(ev['pose_before'])} pose={_fmt_pose(ev['pose'])} "
            f"yaw_before={_fmt_num(ev['yaw_before'])} yaw_after={_fmt_num(ev['yaw_after'])} "
            f"motion_m={_fmt_num(ev['motion_distance'])} yaw_rad={_fmt_num(ev['yaw_delta'])} "
            f"height_m={_fmt_num(ev['height_delta'])} height_change={ev['height_change'] or 'null'} "
            f"platform={str(bool(ev['platform_supported'])).lower()} blocked={str(bool(ev['blocked'])).lower()} "
            f"recovery={str(bool(ev['recovery'])).lower()} recovery_source={ev['recovery_source'] or 'null'} "
            f"rejected_by_rollback={str(bool(ev['rejected_by_rollback'])).lower()} "
            f"route_evidence={str(bool(ev['route_evidence'])).lower()} "
            f"diagnostic_only={str(bool(ev['diagnostic_only'])).lower()} "
            f"cond={ev['condition_id'] or 'null'} turn_cond={ev['owned_condition_id'] or 'null'} "
            f"dir={ev['turn_direction'] or 'null'} turn_step={ts_s} turn_departure={td_s} "
            f"wp={wp_s} obs_before={ob_s} obs_after={oa_s}"
        )
    return "\n".join(lines)


def select_evidence_frames(
    frame_records: Any, events: Any, current_condition_ids: Any = (), max_frames: int = 6
) -> list[dict[str, Any]]:
    """Select bounded historical evidence frames matched by exact step, phase, and observation_id."""
    cap = _check_limit(max_frames, "max_frames")
    if cap == 0:
        return []
    conds = {c for c in (current_condition_ids or ()) if _valid_str(c) is not None}
    synced = [_refresh_event(e) for e in (events or ()) if isinstance(e, dict)]
    valid_frames = [
        fr for fr in (frame_records or ())
        if isinstance(fr, dict) and _valid_int(fr.get("step")) is not None
        and _valid_int(fr.get("observation_id")) is not None
        and fr.get("phase") in _VALID_PHASES and fr.get("view") in _VALID_VIEWS
        and isinstance(fr.get("image"), Image.Image)
    ]
    candidates: dict[tuple[int, str, int, str], tuple[tuple[int, int, int, int, int], dict[str, Any]]] = {}
    for ev in synced:
        st = _valid_int(ev.get("step"))
        if st is None:
            continue
        ev_sub, ev_cond = _valid_str(ev.get("subtask_id")), _valid_str(ev.get("condition_id"))
        in_c = bool(ev_cond and ev_cond in conds)
        is_it = "instruction_turn" in (ev.get("classifications") or ())
        lm_view = ev.get("landmark_view") if is_it and ev.get("landmark_view") in _VALID_VIEWS else None
        et = ev.get("event_type")
        ev_tier = (0 if in_c else 1) if is_it else (
            (1 if in_c else 2) if et == "turn_departure" else (
                (2 if in_c else 3) if (et in ("blocked", "recovery", "platform", "ascent", "descent", "in_place_undo") or ev.get("rejected_by_rollback")) else (3 if in_c else 4)
            )
        )
        phase_matches = []
        for phase_idx, (phase, obs_key) in enumerate((("BEFORE", "observation_id_before"), ("AFTER", "observation_id_after"))):
            exp_obs = _valid_int(ev.get(obs_key))
            if exp_obs is None:
                continue
            matched = [
                fr for fr in valid_frames
                if fr["step"] == st and fr["phase"] == phase and fr["observation_id"] == exp_obs
                and (fr.get("subtask_id") is None or (_valid_str(fr.get("subtask_id")) is not None and fr.get("subtask_id") == ev_sub))
                and (fr.get("scope_key") is None or (_valid_str(fr.get("scope_key")) is not None and fr.get("scope_key") == ev_sub))
                and (fr.get("condition_id") is None or (_valid_str(fr.get("condition_id")) is not None and fr.get("condition_id") == ev_cond))
            ]
            if not matched:
                continue
            has_lm = bool(lm_view and any(fr["view"] == lm_view for fr in matched))
            has_front = any(fr["view"] == "front" for fr in matched)
            if is_it and has_lm:
                chosen_frs = [fr for fr in matched if fr["view"] == lm_view]
            elif has_front:
                chosen_frs = [fr for fr in matched if fr["view"] == "front"]
            else:
                chosen_frs = matched[:1]
            phase_matches.append((phase_idx, phase, exp_obs, has_lm, chosen_frs))
        is_paired = len(phase_matches) == 2
        not_paired = 0 if is_paired else 1
        for phase_idx, phase, exp_obs, has_lm, chosen_frs in phase_matches:
            for fr in chosen_frs:
                vw = fr["view"]
                if is_it:
                    is_prim = bool(lm_view and has_lm and vw == lm_view and not ev.get("diagnostic_only"))
                    lm_miss = not (lm_view and has_lm)
                    is_sec = not (lm_view and has_lm and vw == lm_view)
                else:
                    is_prim, lm_miss, is_sec = False, False, vw != "front"
                role = "primary_landmark" if is_prim else ("secondary_diagnostic" if is_sec else ("diagnostic" if ev.get("diagnostic_only") else "route_evidence"))
                annotated = dict(fr)
                annotated.update({
                    "historical": True, "current_geometry_authority": False, "fresh_view_authority": False,
                    "depth_authority": False, "independent_view_witness": False,
                    "event_type": et, "status": ev.get("status"), "paired_before_after": is_paired,
                    "diagnostic_only": bool(ev.get("diagnostic_only")), "route_evidence": bool(ev.get("route_evidence")),
                    "recovery": bool(ev.get("recovery")), "rejected_by_rollback": bool(ev.get("rejected_by_rollback")),
                    "is_primary_landmark_view": is_prim, "landmark_match": is_prim, "secondary_context": is_sec,
                    "primary_landmark_view_missing": lm_miss, "requested_landmark_view": lm_view, "evidence_role": role,
                    "caption": (
                        f"historical step={st} phase={phase} obs={exp_obs} view={vw} "
                        f"event={et} status={ev.get('status')} role={role}"
                        + (" [REJECTED_BY_ROLLBACK]" if ev.get("rejected_by_rollback") else "")
                        + (" [RECOVERY]" if ev.get("recovery") else "")
                    ),
                })
                key = (st, phase, exp_obs, vw)
                prio = (ev_tier, not_paired, 0 if is_prim else (2 if is_sec else 1), st, phase_idx)
                if key not in candidates or prio < candidates[key][0]:
                    candidates[key] = (prio, annotated)
    ordered = sorted(candidates.values(), key=lambda item: item[0])[:cap]
    return [item[1] for item in ordered]


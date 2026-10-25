from __future__ import annotations

import base64
import io
import json
from typing import Any

import numpy as np
from PIL import Image

import navigation_policy
import verification_bev
import verification_evidence

_VALID_VIEWS = ("front", "left", "right", "back")
_RESAMPLE = getattr(getattr(Image, "Resampling", Image), "BILINEAR", Image.BILINEAR)


def _valid_int(val: Any) -> int | None:
    if isinstance(val, bool) or val is None:
        return None
    if isinstance(val, (int, np.integer)):
        iv = int(val)
        return iv if iv >= 0 else None
    return None


def _valid_str(val: Any) -> str | None:
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


_VALID_RECOVERY_LIFECYCLE_EVENTS: dict[str, str] = {
    "physical_recovery_started": "started",
    "physical_recovery_completed": "completed",
    "physical_recovery_failed": "failed",
    "physical_recovery_aborted": "aborted",
    "wrong_path_rollback_started": "started",
    "wrong_path_rollback_completed": "completed",
    "wrong_path_rollback_failed": "failed",
    "wrong_path_rollback_aborted": "aborted",
}

_BEV_AXES_KEYS = ("x_positive", "y_positive", "pixel_y_positive", "yaw_convention")
_BEV_COUNT_SCALAR_KEYS = (
    "total_ledger_rows",
    "retained_rows",
    "omitted_history_rows",
    "invalid_rows",
    "rendered_observed_points",
    "omitted_observed_points",
    "invalid_observed_points",
    "clipped_points_count",
    "omitted_map_labels",
    "move_arrow_count",
)
_BEV_CLIP_KEYS = ("active", "clipped_row_count", "clipped_point_count", "outlier_radius_m")
_BEV_Z_KEYS = ("unit", "min_z", "max_z", "current_z")
_BEV_TRANSFORM_KEYS = ("scale_px_per_m", "x_min", "x_max", "y_min", "y_max")


def _filter_recovery_trace_events(trace: Any) -> list[dict[str, Any]]:
    if not isinstance(trace, (list, tuple)):
        return []
    filtered: list[dict[str, Any]] = []
    for item in trace:
        if not isinstance(item, dict):
            continue
        role = _valid_str(item.get("role"))
        if role is not None and role != "controller":
            continue
        ev_name = _valid_str(item.get("event"))
        if ev_name not in _VALID_RECOVERY_LIFECYCLE_EVENTS:
            continue
        expected_status = _VALID_RECOVERY_LIFECYCLE_EVENTS[ev_name]
        status = _valid_str(item.get("status"))
        if status != expected_status:
            continue
        subtask_id = _valid_str(item.get("subtask_id"))
        if subtask_id is None:
            continue
        src = _valid_str(item.get("recovery_source")) or _valid_str(item.get("source")) or "unknown"
        ctrl_step = _valid_int(item.get("controller_step"))
        if ctrl_step is None:
            ctrl_step = _valid_int(item.get("step"))
        exec_step = _valid_int(item.get("execution_step"))
        norm = dict(item)
        norm["role"] = "controller"
        norm["event"] = ev_name
        norm["status"] = status
        norm["subtask_id"] = subtask_id
        norm["recovery_source"] = src
        norm["source"] = src
        norm["controller_step"] = ctrl_step
        norm["step"] = exec_step
        if norm.get("pose") is None and item.get("actual_ground_pose") is not None:
            norm["pose"] = item.get("actual_ground_pose")
        filtered.append(norm)
    return filtered


def _summarize_bev_meta(bev_meta: Any, yaw: float | None = None, foot_pose: Any = None) -> dict[str, Any]:
    if not isinstance(bev_meta, dict):
        out: dict[str, Any] = {}
        if yaw is not None:
            out["current_yaw"] = _json_safe(float(yaw))
        if foot_pose is not None:
            out["current_foot_pose"] = _json_safe(foot_pose)
        return out
    summary: dict[str, Any] = {}
    if "image_size" in bev_meta:
        summary["image_size"] = _json_safe(bev_meta.get("image_size"))
    if isinstance(bev_meta.get("axes"), dict):
        axes_src = bev_meta["axes"]
        summary["axes"] = {k: _json_safe(axes_src[k]) for k in _BEV_AXES_KEYS if k in axes_src}
    if "current_pose_valid" in bev_meta:
        summary["current_pose_valid"] = bool(bev_meta.get("current_pose_valid"))
    if "current_yaw_valid" in bev_meta:
        summary["current_yaw_valid"] = bool(bev_meta.get("current_yaw_valid"))
    raw_yaw = bev_meta.get("current_yaw", float(yaw) if yaw is not None else None)
    if raw_yaw is not None:
        summary["current_yaw"] = _json_safe(raw_yaw)
    raw_pose = bev_meta.get("current_foot_pose", foot_pose)
    if raw_pose is not None:
        summary["current_foot_pose"] = _json_safe(raw_pose)
    if isinstance(bev_meta.get("counts"), dict):
        cnt_src = bev_meta["counts"]
        summary["counts"] = {
            k: _json_safe(cnt_src[k])
            for k in _BEV_COUNT_SCALAR_KEYS
            if k in cnt_src and not isinstance(cnt_src[k], (list, tuple, dict))
        }
    if isinstance(bev_meta.get("coordinate_clipping"), dict):
        clip_src = bev_meta["coordinate_clipping"]
        summary["coordinate_clipping"] = {k: _json_safe(clip_src[k]) for k in _BEV_CLIP_KEYS if k in clip_src}
    if isinstance(bev_meta.get("z_profile"), dict):
        z_src = bev_meta["z_profile"]
        summary["z_profile"] = {k: _json_safe(z_src[k]) for k in _BEV_Z_KEYS if k in z_src}
    if isinstance(bev_meta.get("transform"), dict):
        tr_src = bev_meta["transform"]
        summary["transform"] = {k: _json_safe(tr_src[k]) for k in _BEV_TRANSFORM_KEYS if k in tr_src}
    if "legend" in bev_meta:
        leg_src = bev_meta.get("legend")
        if isinstance(leg_src, dict):
            summary["legend"] = sorted(str(k) for k in leg_src.keys())
        elif isinstance(leg_src, (list, tuple)):
            summary["legend"] = [str(k) for k in leg_src if isinstance(k, str)]
    return summary


def _extract_raw_view(obs: dict[str, Any], view: str) -> Any:
    if view == "front":
        if "color_sensor" in obs and obs.get("color_sensor") is not None:
            return obs.get("color_sensor")
        front = obs.get("front")
        return front.get("color_sensor") if isinstance(front, dict) else front
    sub = obs.get(view)
    return sub.get("color_sensor") if isinstance(sub, dict) else None


def _to_bounded_rgb_copy(raw: Any, max_side: int = 512) -> Image.Image | None:
    img: Image.Image | None = None
    if isinstance(raw, np.ndarray):
        if raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[0] <= 0 or raw.shape[1] <= 0 or raw.shape[2] not in (3, 4):
            return None
        rgb = np.array(raw[:, :, :3], dtype=np.uint8, copy=True)
        img = Image.fromarray(rgb, mode="RGB").copy()
    elif isinstance(raw, Image.Image):
        if raw.width <= 0 or raw.height <= 0:
            return None
        img = raw.convert("RGB").copy()
    if img is None:
        return None
    hi = max(img.width, img.height)
    if hi > max_side:
        scale = float(max_side) / float(hi)
        w = max(1, int(round(img.width * scale)))
        h = max(1, int(round(img.height * scale)))
        img = img.resize((w, h), resample=_RESAMPLE)
    return img


def _extract_condition_ids(ctx: Any) -> list[str]:
    if not isinstance(ctx, dict):
        return []
    direct = ctx.get("current_condition_ids")
    if isinstance(direct, (list, tuple, set)):
        ids = [s.strip() for s in direct if isinstance(s, str) and s.strip()]
        if ids:
            return list(dict.fromkeys(ids))
    checks = ctx.get("checks")
    if not isinstance(checks, (list, tuple)):
        return []
    unresolved: list[str] = []
    all_ids: list[str] = []
    for chk in checks:
        if isinstance(chk, str) and chk.strip():
            all_ids.append(chk.strip())
        elif isinstance(chk, dict):
            cid = _valid_str(chk.get("condition_id")) or _valid_str(chk.get("id"))
            if cid:
                all_ids.append(cid)
                st = str(chk.get("status") or "unknown").strip().lower()
                if st != "satisfied" and chk.get("satisfied") is not True:
                    unresolved.append(cid)
    chosen = unresolved if unresolved else all_ids
    return list(dict.fromkeys(chosen))


def _record_landmark_view(record: dict[str, Any]) -> str | None:
    it = record.get("instruction_turn")
    if isinstance(it, dict):
        lm = it.get("landmark_evidence")
        if isinstance(lm, dict) and lm.get("view") in _VALID_VIEWS:
            return str(lm["view"])
        if it.get("landmark_view") in _VALID_VIEWS:
            return str(it["landmark_view"])
    if record.get("landmark_view") in _VALID_VIEWS:
        return str(record["landmark_view"])
    return None


def _json_safe(val: Any) -> Any:
    if val is None or isinstance(val, (bool, str, int)):
        return val
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        return float(val) if np.isfinite(val) else None
    if isinstance(val, Image.Image) or isinstance(val, (bytes, bytearray, np.ndarray)):
        return None
    if isinstance(val, dict):
        out: dict[str, Any] = {}
        for k, v in val.items():
            ks = str(k)
            if ks.startswith("_") or ks in ("image", "base64", "observed_points", "points"):
                continue
            out[ks] = _json_safe(v)
        return out
    if isinstance(val, (list, tuple)):
        return [_json_safe(x) for x in val if not isinstance(x, (Image.Image, bytes, bytearray, np.ndarray))]
    return str(val)


class VerificationInputs:
    """Mixin providing historical evidence caching, prompt formatting, and fixed BEV assembly."""

    def _verif_cfg(self) -> dict[str, Any]:
        cfg = getattr(self, "navigation_cfg", None)
        return cfg if isinstance(cfg, dict) else {}

    def _evidence_enabled(self) -> bool:
        return bool(self._verif_cfg().get("verification_evidence_enabled", False))

    def _fixed_bev_enabled(self) -> bool:
        return bool(self._verif_cfg().get("verifier_fixed_bev", False))

    def _encode_image_b64(self, img: Image.Image) -> str:
        enc = getattr(self, "_pil_to_base64", None)
        if callable(enc):
            res = enc(img)
            if isinstance(res, str) and res:
                return res
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _snapshot_verification_views(self, obs: Any) -> dict[str, Any] | None:
        if not self._evidence_enabled() or not isinstance(obs, dict):
            return None
        obs_id = _valid_int(getattr(getattr(self, "mapper", None), "observation_id", None))
        if obs_id is None:
            return None
        views: dict[str, Image.Image] = {}
        for vw in _VALID_VIEWS:
            img = _to_bounded_rgb_copy(_extract_raw_view(obs, vw), max_side=512)
            if img is not None:
                views[vw] = img
        if not views:
            return None
        return {"observation_id": obs_id, "views": views}

    def _cache_verification_frames(self, record: Any, before_snapshot: Any, obs: Any) -> None:
        if not self._evidence_enabled() or not isinstance(record, dict):
            return
        if not hasattr(self, "_verification_event_frames") or not isinstance(self._verification_event_frames, list):
            self._verification_event_frames = []
        ledger = getattr(self, "_execution_ledger", None)
        step = _valid_int(record.get("step"))
        if step is None or not isinstance(ledger, (list, tuple)) or not ledger:
            return
        if not any(isinstance(r, dict) and _valid_int(r.get("step")) == step for r in ledger):
            return
        after_snapshot = self._snapshot_verification_views(obs)
        subtask_id = _valid_str(record.get("subtask_id"))
        for phase, exp_obs, snap in (
            ("BEFORE", _valid_int(record.get("observation_id_before")), before_snapshot),
            ("AFTER", _valid_int(record.get("observation_id_after")), after_snapshot),
        ):
            if exp_obs is None or not isinstance(snap, dict) or _valid_int(snap.get("observation_id")) != exp_obs:
                continue
            s_views = snap.get("views")
            if not isinstance(s_views, dict):
                continue
            for vw in _VALID_VIEWS:
                img = s_views.get(vw)
                if isinstance(img, Image.Image):
                    self._verification_event_frames.append({
                        "step": step,
                        "phase": phase,
                        "observation_id": exp_obs,
                        "view": vw,
                        "subtask_id": subtask_id,
                        "image": img.copy(),
                    })
        dedup_map: dict[tuple[int, str, int, str], dict[str, Any]] = {}
        for fr in self._verification_event_frames:
            if isinstance(fr, dict) and isinstance(fr.get("image"), Image.Image):
                k = (int(fr["step"]), str(fr["phase"]), int(fr["observation_id"]), str(fr["view"]))
                dedup_map.pop(k, None)
                dedup_map[k] = fr
        cache = list(dedup_map.values())
        raw_cap = self._verif_cfg().get("verification_frame_cache_size", 48)
        try:
            cap = max(16, min(96, int(raw_cap)))
        except (TypeError, ValueError):
            cap = 48
        if len(cache) > cap:
            nav_trace = _filter_recovery_trace_events(getattr(self, "navigation_trace", ()) or ())
            cond_ids = _extract_condition_ids(getattr(self, "_route_verification_context", None))
            cond_set = set(cond_ids)
            live_events = verification_evidence.build_execution_events(ledger, recovery_events=nav_trace)
            pri_events = [
                ev for ev in live_events
                if isinstance(ev, dict) and (ev.get("event_type") != "translation" or ev.get("condition_id") in cond_set)
            ]
            pri_sel = verification_evidence.select_evidence_frames(
                cache, pri_events, current_condition_ids=cond_ids, max_frames=max(0, cap - 8)
            )
            kept: set[tuple[int, str, int, str]] = {
                (int(f["step"]), str(f["phase"]), int(f["observation_id"]), str(f["view"])) for f in pri_sel
            }
            step_lm: dict[int, str] = {}
            for r in ledger:
                if isinstance(r, dict) and _valid_int(r.get("step")) is not None:
                    lm_v = _record_landmark_view(r)
                    if lm_v:
                        step_lm[int(r["step"])] = lm_v
            for fr in reversed(cache):
                if len(kept) >= cap:
                    break
                st_i = int(fr["step"])
                vw_s = str(fr["view"])
                if vw_s == step_lm.get(st_i, "front") or vw_s == "front":
                    kept.add((st_i, str(fr["phase"]), int(fr["observation_id"]), vw_s))
            for fr in reversed(cache):
                if len(kept) >= cap:
                    break
                kept.add((int(fr["step"]), str(fr["phase"]), int(fr["observation_id"]), str(fr["view"])))
            cache = [fr for fr in cache if (int(fr["step"]), str(fr["phase"]), int(fr["observation_id"]), str(fr["view"])) in kept][:cap]
        self._verification_event_frames = cache

    def _execution_ledger_text(self, records: Any) -> str:
        return verification_evidence.format_execution_ledger(records if records is not None else ())

    def _verification_execution_prompt(self, ctx: Any) -> str:
        if not self._evidence_enabled():
            return ""
        ledger = getattr(self, "_execution_ledger", ()) or ()
        nav_trace = _filter_recovery_trace_events(getattr(self, "navigation_trace", ()) or ())
        cond_ids = _extract_condition_ids(ctx if isinstance(ctx, dict) else getattr(self, "_route_verification_context", None))
        events = verification_evidence.build_execution_events(ledger, recovery_events=nav_trace)
        summaries = verification_evidence.select_event_summaries(events, current_condition_ids=cond_ids, limit=12)
        diag_steps = {
            _valid_int(ev.get("step"))
            for ev in events
            if isinstance(ev, dict) and (ev.get("recovery") or ev.get("rejected_by_rollback") or ev.get("diagnostic_only"))
        }
        diag_rows = [
            r for r in ledger
            if isinstance(r, dict) and (
                r.get("rejected_by_rollback") or r.get("recovery")
                or r.get("status") in ("rejected", "rolled_back", "recovery")
                or _valid_int(r.get("step")) in diag_steps
            )
        ]
        diag_text = verification_evidence.format_execution_ledger(diag_rows) if diag_rows else "none"
        return (
            "VERIFICATION EXECUTION EVIDENCE SUMMARY (does NOT replace full valid scoped execution ledger rendered separately):\n"
            "- Identifier rule: execution_step != controller_step != observation_id (never conflate these IDs).\n"
            "- Authority rule: historical frames, fixed BEV, and counter-turn/in-place undo facts are historical/diagnostic context only; "
            "they do NOT automatically satisfy route clauses, override current 4-view geometry/depth authority, or authorize a wrong path.\n"
            f"- Prioritized key event summaries (max 12, chronological): {json.dumps(_json_safe(summaries), separators=(',', ':'))}\n"
            "- Recovery / rejected diagnostic rows (diagnostic_only; CANNOT prove completion):\n"
            f"{diag_text}"
        )

    def _verifier_evidence_content(self) -> list[dict[str, Any]]:
        ev_on = self._evidence_enabled()
        bev_on = self._fixed_bev_enabled()
        meta: dict[str, Any] = {
            "evidence_enabled": ev_on,
            "bev_enabled": bev_on,
            "selected_frames": [],
            "bev": None,
            "errors": [],
        }
        if not ev_on and not bev_on:
            self._last_verification_input_metadata = meta
            return []
        content: list[dict[str, Any]] = []
        ledger = getattr(self, "_execution_ledger", ()) or ()
        nav_trace = getattr(self, "_navigation_trace", None) or getattr(self, "navigation_trace", ()) or ()
        ctx = getattr(self, "_route_verification_context", None)
        cond_ids = _extract_condition_ids(ctx)
        if ev_on:
            raw_max = self._verif_cfg().get("verifier_event_max_frames", 4)
            try:
                max_frames = max(0, min(4, int(raw_max)))
            except (TypeError, ValueError):
                max_frames = 4
            try:
                events = verification_evidence.build_execution_events(
                    ledger,
                    recovery_events=_filter_recovery_trace_events(nav_trace),
                )
                cache = getattr(self, "_verification_event_frames", ()) or ()
                selected = (
                    verification_evidence.select_evidence_frames(cache, events, current_condition_ids=cond_ids, max_frames=max_frames)
                    if max_frames > 0 else []
                )
                for fr in selected:
                    img = fr.get("image")
                    if not isinstance(img, Image.Image):
                        continue
                    b64 = self._encode_image_b64(img)
                    url = b64 if b64.startswith("data:image/") else f"data:image/jpeg;base64,{b64}"
                    caption = str(fr.get("caption") or "")
                    content.append({
                        "type": "text",
                        "text": f"HISTORICAL EVIDENCE FRAME ({caption}) [historical=True, current_geometry_authority=False, depth_authority=False]",
                    })
                    content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
                    meta["selected_frames"].append(_json_safe({
                        "step": fr.get("step"),
                        "phase": fr.get("phase"),
                        "observation_id": fr.get("observation_id"),
                        "view": fr.get("view"),
                        "subtask_id": fr.get("subtask_id"),
                        "event_type": fr.get("event_type"),
                        "status": fr.get("status"),
                        "evidence_role": fr.get("evidence_role"),
                        "paired_before_after": bool(fr.get("paired_before_after")),
                        "diagnostic_only": bool(fr.get("diagnostic_only")),
                        "route_evidence": bool(fr.get("route_evidence")),
                        "recovery": bool(fr.get("recovery")),
                        "rejected_by_rollback": bool(fr.get("rejected_by_rollback")),
                        "historical": True,
                        "current_geometry_authority": False,
                        "depth_authority": False,
                        "independent_view_witness": False,
                        "caption": caption,
                    }))
            except Exception as exc:
                err = f"evidence_frame_selection_error:{type(exc).__name__}:{exc}"
                meta["errors"].append(err)
                content.append({"type": "text", "text": f"VERIFICATION EVIDENCE DIAGNOSTIC ERROR: {err}"})
        if bev_on:
            mapper = getattr(self, "mapper", None)
            try:
                if mapper is None:
                    raise ValueError("mapper_missing")
                raw_pos = getattr(mapper, "current_position", None)
                arr_pos = np.asarray(raw_pos, dtype=float).reshape(-1) if raw_pos is not None else np.array([])
                if arr_pos.size != 3 or not np.all(np.isfinite(arr_pos)):
                    raise ValueError("invalid_current_position")
                floor_h = float(getattr(mapper, "initial_floor_height", 0.0) or 0.0)
                foot_pose = [float(arr_pos[0]), float(arr_pos[1]), float(arr_pos[2] + floor_h)]
                raw_rot = getattr(mapper, "current_rotation", None)
                yaw = navigation_policy.checkpoint_yaw(raw_rot) if raw_rot is not None else None
                if yaw is None:
                    fb_yaw = getattr(mapper, "current_yaw", None)
                    if isinstance(fb_yaw, (int, float, np.floating)) and np.isfinite(fb_yaw):
                        yaw = float(fb_yaw)
                if yaw is None or not np.isfinite(yaw):
                    raise ValueError("invalid_current_yaw")
                obs_pts = None
                get_pts = getattr(mapper, "get_bev_height_points", None)
                if callable(get_pts):
                    obs_pts = get_pts()
                bev_img, bev_meta = verification_bev.render_verification_bev(
                    ledger, foot_pose, float(yaw), observed_points=obs_pts, max_side=672
                )
                if not isinstance(bev_img, Image.Image):
                    raise ValueError("invalid_bev_image")
                b64 = self._encode_image_b64(bev_img)
                url = b64 if b64.startswith("data:image/") else f"data:image/jpeg;base64,{b64}"
                content.append({
                    "type": "text",
                    "text": (
                        "FIXED MAPPER AXES VERIFICATION BEV (NOT heading-up; fixed +X right / +Y up in mapper coordinates; "
                        "arrows=real executed steps, ticks=in-place turns, see in-image legend for recovery/rejected styles, "
                        "bottom strip=actual foot height profile; display-only observed points, no goals/reference paths, "
                        "not current geometry authority)."
                    ),
                })
                content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
                meta["bev"] = _summarize_bev_meta(bev_meta, yaw=float(yaw), foot_pose=foot_pose)
            except Exception as exc:
                err = f"fixed_bev_render_error:{type(exc).__name__}:{exc}"
                meta["errors"].append(err)
                content.append({"type": "text", "text": f"FIXED VERIFICATION BEV DIAGNOSTIC ERROR: {err}"})
        self._last_verification_input_metadata = _json_safe(meta)
        return content


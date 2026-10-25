"""Visual candidate prompt/crop helpers, candidate resolution, and frozen-frame spatial verification."""
import base64
import io
import numpy as np
from PIL import Image

from spatial_relation import evaluate_spatial_relation, parse_endpoint_relation
from target_identity import detector_matches_endpoint, endpoint_object_roles, is_between_endpoint
from visual_candidate_binding import resolve_candidate_region, select_candidate_crops

_VIEWS = ('front', 'left', 'right', 'back')


def _clean_serializable(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        num = float(value)
        return num if np.isfinite(num) else None
    if isinstance(value, np.generic):
        return _clean_serializable(value.item())
    if isinstance(value, np.ndarray):
        return [_clean_serializable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_clean_serializable(v) for v in value]
    if isinstance(value, dict):
        return {
            str(k): _clean_serializable(v)
            for k, v in value.items()
            if k not in ('image', 'mask') and not isinstance(v, Image.Image)
        }
    return str(value)


class VisualGroundingMixin:
    def _visual_enabled(self):
        return bool((getattr(self, 'navigation_cfg', None) or {}).get('task_visual_enabled', False))

    def _maybe_refresh_verifier_perception(self, ctx_or_text=None):
        cfg = getattr(self, 'navigation_cfg', None) or {}
        import os as _os
        raw_env = str(_os.environ.get('HSGM_VERIFIER_DINO_SAM3', '') or '').strip().lower()
        enabled = True if raw_env in ('1', 'true', 'yes', 'on') else (
            False if raw_env in ('0', 'false', 'no', 'off') else bool(cfg.get('verifier_dino_sam3_enabled', False))
        )
        if not enabled:
            return {'mode': 'yoloe', 'enabled': False, 'refreshed': False}
        from segmentation.verifier_dino_sam3 import refresh_verifier_candidates_with_dino_sam3
        res = refresh_verifier_candidates_with_dino_sam3(self, ctx_or_text)
        return {'mode': 'dino_sam3', **(res if isinstance(res, dict) else {})}

    def _pil_to_base64(self, image):
        buf = io.BytesIO()
        image.convert('RGB').save(buf, format='JPEG', quality=85)
        return base64.b64encode(buf.getvalue()).decode('ascii')

    def _extract_clause_text(self, source):
        if isinstance(source, str):
            return source.strip()
        if isinstance(source, dict):
            if (source.get('active') is False or source.get('satisfied') is True
                    or source.get('completed') is True
                    or source.get('status') in ('satisfied', 'completed', 'done', 'inactive')):
                return ''
            for key in ('landmark_text', 'target_text', 'object_text', 'landmark', 'text', 'clause_text', 'clause'):
                if source.get(key):
                    return str(source[key]).strip()
        return ''

    def _endpoint_clause_text(self, ctx_or_text=None):
        if isinstance(ctx_or_text, dict) and ctx_or_text.get('endpoint_clause'):
            direct = self._extract_clause_text(ctx_or_text.get('endpoint_clause'))
            if direct:
                return direct
        rctx = getattr(self, '_route_verification_context', None)
        if isinstance(rctx, dict) and rctx.get('endpoint_clause'):
            direct = self._extract_clause_text(rctx.get('endpoint_clause'))
            if direct:
                return direct
        ep = getattr(self, '_endpoint_clause', None)
        if not callable(ep):
            direct = self._extract_clause_text(ep)
            if direct:
                return direct
        raw_text = ''
        if isinstance(ctx_or_text, str) and ctx_or_text.strip():
            raw_text = ctx_or_text.strip()
        elif isinstance(ctx_or_text, dict):
            for key in ('target_text', 'text', 'instruction'):
                if ctx_or_text.get(key):
                    raw_text = str(ctx_or_text[key]).strip()
                    if raw_text:
                        break
        if not raw_text and isinstance(rctx, dict):
            for key in ('target_text', 'text', 'instruction'):
                if rctx.get(key):
                    raw_text = str(rctx[key]).strip()
                    if raw_text:
                        break
        if not raw_text:
            for attr in ('target_text', 'current_instruction', 'instruction'):
                val = getattr(self, attr, None)
                if isinstance(val, str) and val.strip():
                    raw_text = val.strip()
                    break
        if raw_text and callable(ep):
            try:
                extracted = self._extract_clause_text(ep(raw_text))
                if extracted:
                    return extracted
            except TypeError:
                pass
        return raw_text

    def _active_turn_clause_info(self, ctx_or_text=None):
        sources = []
        if isinstance(ctx_or_text, dict):
            sources.extend(ctx_or_text.get(k) for k in ('turn_clause', 'active_turn_clause', 'named_turn_clause', 'turn_landmark_text', 'turn_text'))
        for attr in ('_turn_clause', '_active_turn_clause', 'active_turn_clause', '_named_turn_clause', '_turn_landmark_text'):
            sources.append(getattr(self, attr, None))
        rctx = getattr(self, '_route_verification_context', None)
        if isinstance(rctx, dict):
            sources.extend(rctx.get(k) for k in ('turn_clause', 'active_turn_clause', 'named_turn_clause', 'turn_landmark_text', 'turn_text'))
        explicitly_inactive = False
        for src in sources:
            if callable(src):
                try:
                    val = src()
                except TypeError:
                    continue
            else:
                val = src
            if val is None:
                continue
            if isinstance(val, dict) and (
                val.get('active') is False or val.get('satisfied') is True
                or val.get('completed') is True or val.get('status') in ('satisfied', 'completed', 'done', 'inactive')
            ):
                explicitly_inactive = True
                continue
            text = self._extract_clause_text(val)
            if text:
                return True, text, False
        return False, '', explicitly_inactive

    def _candidate_roles_for_context(self, record, endpoint_text, turn_text, turn_inactive=False):
        labels = []
        for key in ('detector_class_name', 'canonical_label', 'class_name', 'label'):
            val = str(record.get(key) or '').strip()
            if val and val not in labels:
                labels.append(val)
        roles = []
        for r in (record.get('roles') or ()):
            r_norm = 'spatial' if r == 'spatial_landmark' else str(r)
            if r_norm in ('primary', 'reference', 'spatial', 'turn') and r_norm not in roles:
                if r_norm == 'turn' and turn_inactive:
                    continue
                roles.append(r_norm)
        if labels and endpoint_text:
            spec = parse_endpoint_relation(endpoint_text)
            primary_text = (spec.get('target_text') or endpoint_text) if spec.get('required') else endpoint_text
            if primary_text and any(detector_matches_endpoint(lbl, primary_text, 'primary') for lbl in labels) and 'primary' not in roles:
                roles.append('primary')
            ref_text = primary_text if spec.get('required') else endpoint_text
            if ref_text and (is_between_endpoint(ref_text) or endpoint_object_roles(ref_text).get('reference')):
                if any(detector_matches_endpoint(lbl, ref_text, 'reference') for lbl in labels) and 'reference' not in roles:
                    roles.append('reference')
            if spec.get('required') and spec.get('landmark_text'):
                if any(detector_matches_endpoint(lbl, spec['landmark_text'], 'primary') for lbl in labels) and 'spatial' not in roles:
                    roles.append('spatial')
        if labels and turn_text and not turn_inactive:
            if any(detector_matches_endpoint(lbl, turn_text, 'primary') for lbl in labels) and 'turn' not in roles:
                roles.append('turn')
        return roles

    def _enriched_current_candidates(self, ctx_or_text=None):
        mapper = getattr(self, 'mapper', None)
        obs_id = getattr(mapper, 'observation_id', None)
        raw = getattr(mapper, 'current_visual_candidates', None) or {}
        endpoint_text = self._endpoint_clause_text(ctx_or_text)
        _, turn_text, turn_inactive = self._active_turn_clause_info(ctx_or_text)

        def _build_enriched(raw_source):
            items = (
                raw_source.items()
                if isinstance(raw_source, dict)
                else [(r.get('candidate_id'), r) for r in raw_source if isinstance(r, dict)]
            )
            out = {}
            for key, rec in items:
                if not isinstance(rec, dict) or not isinstance(key, str) or rec.get('candidate_id') != key:
                    continue
                if obs_id is not None and rec.get('observation_id') != obs_id:
                    continue
                raw_conf = rec.get('confidence', 0.0)
                if isinstance(raw_conf, bool):
                    continue
                try:
                    conf = 0.0 if raw_conf is None else float(raw_conf)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(conf):
                    continue
                roles = self._candidate_roles_for_context(rec, endpoint_text, turn_text, turn_inactive=turn_inactive)
                out[key] = {**rec, 'confidence': conf, 'roles': roles}
            return out

        enriched = _build_enriched(raw)
        if (
            self._visual_enabled()
            and turn_text
            and not turn_inactive
            and obs_id is not None
            and int(obs_id) >= 1
            and not any('turn' in (r.get('roles') or ()) for r in enriched.values())
        ):
            from segmentation.verifier_dino_sam3 import refresh_verifier_candidates_with_dino_sam3
            refresh_info = refresh_verifier_candidates_with_dino_sam3(
                self, ctx_or_text, only_non_structural=True
            )
            if refresh_info.get('refreshed'):
                raw = getattr(mapper, 'current_visual_candidates', None) or {}
                enriched = _build_enriched(raw)
        return enriched, obs_id, endpoint_text, turn_text, turn_inactive

    def _visual_candidate_prompt(self, ctx_or_text=None):
        if not self._visual_enabled():
            return ''
        enriched, obs_id, _, _, _ = self._enriched_current_candidates(ctx_or_text)
        role_rank = {'primary': 0, 'reference': 1, 'spatial': 2, 'turn': 3}
        rows = [r for r in enriched.values() if r.get('roles')]
        rows.sort(key=lambda r: (
            min((role_rank.get(role, 9) for role in (r.get('roles') or ())), default=9),
            -float(r.get('confidence') or 0.0),
            str(r.get('candidate_id') or ''),
        ))
        lines = [
            'VISUAL CANDIDATE GROUNDING CONTRACT:',
            '- Detector candidates are class hypotheses only, not endpoint instance truth.',
            '- All bboxes MUST use normalized [0, 1] full-view xyxy coordinates with bbox_space="full_view" (target_bbox_space="full_view"); never output pixel or crop-relative coordinates.',
            '- candidate_id (string "obs:view:index") is distinct from optional integer entity_id. Explicit invalid/stale/wrong-view/wrong-class candidate_id fails closed and never falls back.',
            '- Set target_candidate_id and target_entity_id ONLY from candidates whose roles include "primary"; never put a "spatial", "reference", or "turn" candidate into target_candidate_id or target_entity_id.',
            '- Null candidate_id (target_candidate_id=null, target_entity_id=null) is required when no candidate has role="primary"; a fresh VLM full_view bbox requires independent changed-view confirmation.',
            '- spatial_landmark_region (role="spatial_landmark", candidate_id, view, bbox, bbox_space="full_view") is independent of reference_region (door_jamb / between).',
            '- Named turn landmark candidate (role="turn") is independent of the endpoint candidate.',
            f'Current visual candidates (observation_id={obs_id}, showing <=16):',
        ]
        for rec in rows[:16]:
            cid = rec.get('candidate_id')
            cand_roles = rec.get('roles') or ()
            eid = rec.get('entity_id') if 'primary' in cand_roles else None
            cls_name = (
                rec.get('detector_class_name')
                or rec.get('canonical_label')
                or rec.get('class_name')
                or rec.get('label')
                or 'unknown'
            )
            view = rec.get('view')
            bbox = rec.get('original_bbox') if rec.get('original_bbox') is not None else rec.get('bbox')
            g_status = rec.get('geometry_status') or ('3d_measured' if eid is not None else '2d_only')
            raw_dist = rec.get('horizontal_distance_m')
            if raw_dist is None and 'horizontal_distance_m' not in rec and g_status != '2d_only':
                raw_dist = rec.get('distance_m')
            dist_str = 'null'
            if not isinstance(raw_dist, bool) and isinstance(raw_dist, (int, float, np.floating)) and np.isfinite(float(raw_dist)):
                dist_str = f'{float(raw_dist):.2f}m'
            roles_str = ','.join(rec.get('roles') or ()) or 'none'
            lines.append(
                f'- candidate_id={cid} | entity_id={eid if eid is not None else "null"} | class={cls_name} '
                f'| view={view} | bbox={bbox} (bbox_space=full_view) | obs_id={rec.get("observation_id")} '
                f'| geometry_status={g_status} | horizontal_distance_m={dist_str} | roles={roles_str}'
            )
        if not rows:
            lines.append('- none (use target_candidate_id=null with target_bbox_space="full_view" if localizing from raw view)')
        return '\n'.join(lines)

    def _visual_candidate_content(self, for_verifier=True):
        attr = '_last_visual_candidate_crops_for_verifier' if for_verifier else '_last_visual_candidate_crops_for_planner'
        if not self._visual_enabled():
            setattr(self, attr, [])
            return []
        obs = getattr(self, 'curr_obs', None)
        if not obs and for_verifier:
            obs = getattr(self, '_verification_obs', None)
        if not isinstance(obs, dict):
            setattr(self, attr, [])
            return []
        rgb_views = {}
        for view in _VIEWS:
            v_obs = obs if view == 'front' else (obs.get(view) if isinstance(obs.get(view), dict) else {})
            img = v_obs.get('color_sensor') if isinstance(v_obs, dict) else None
            if isinstance(img, np.ndarray) and img.ndim in (2, 3) and img.shape[0] > 0 and img.shape[1] > 0:
                rgb_views[view] = img
            elif isinstance(img, Image.Image) and img.width > 0 and img.height > 0:
                rgb_views[view] = img
        if not rgb_views:
            setattr(self, attr, [])
            return []
        enriched, obs_id, _, _, turn_inactive = self._enriched_current_candidates()
        if not enriched:
            setattr(self, attr, [])
            return []
        roles_map = {cid: list(rec.get('roles') or ()) for cid, rec in enriched.items()}
        if for_verifier:
            limit = 4
            selected, seen = [], set()
            for target_role in ('primary', 'reference', 'spatial'):
                role_subset = {cid: rec for cid, rec in enriched.items() if target_role in (rec.get('roles') or ())}
                if not role_subset:
                    continue
                role_crops = select_candidate_crops(
                    role_subset, rgb_views, observation_id=obs_id,
                    roles={cid: [target_role] for cid in role_subset}, limit=limit, max_side=640,
                )
                for row in role_crops:
                    cid = row.get('candidate_id')
                    if cid not in seen and len(selected) < limit:
                        row['roles'] = list(roles_map.get(cid, row.get('roles') or ()))
                        selected.append(row)
                        seen.add(cid)
                        break
            if len(selected) < limit:
                pool = select_candidate_crops(
                    enriched, rgb_views, observation_id=obs_id,
                    roles=roles_map, limit=6, max_side=640,
                )
                for row in pool:
                    if len(selected) >= limit:
                        break
                    cid = row.get('candidate_id')
                    if cid not in seen:
                        row['roles'] = list(roles_map.get(cid, row.get('roles') or ()))
                        selected.append(row)
                        seen.add(cid)
        else:
            limit = 2
            selected, seen = [], set()
            if not turn_inactive:
                turn_subset = {cid: rec for cid, rec in enriched.items() if 'turn' in (rec.get('roles') or ())}
                if turn_subset:
                    turn_crops = select_candidate_crops(
                        turn_subset, rgb_views, observation_id=obs_id,
                        roles={cid: ['turn'] for cid in turn_subset}, limit=limit, max_side=640,
                    )
                    non_turn_exists = any(
                        cid not in turn_subset and any(r in ('primary', 'reference', 'spatial') for r in (rec.get('roles') or ()))
                        for cid, rec in enriched.items()
                    )
                    max_turn_first = 1 if (non_turn_exists and limit > 1) else limit
                    for row in turn_crops[:max_turn_first]:
                        cid = row.get('candidate_id')
                        if cid not in seen and len(selected) < limit:
                            row['roles'] = list(roles_map.get(cid, row.get('roles') or ()))
                            selected.append(row)
                            seen.add(cid)
            if len(selected) < limit:
                rest = select_candidate_crops(
                    enriched, rgb_views, observation_id=obs_id,
                    roles=roles_map, limit=6, max_side=640,
                )
                for row in rest:
                    if len(selected) >= limit:
                        break
                    cid = row.get('candidate_id')
                    if cid not in seen:
                        row['roles'] = list(roles_map.get(cid, row.get('roles') or ()))
                        selected.append(row)
                        seen.add(cid)
        metadata = [_clean_serializable(row) for row in selected]
        setattr(self, attr, metadata)
        blocks = []
        for row in selected:
            pil_img = row.get('image')
            if pil_img is None:
                continue
            b64 = self._pil_to_base64(pil_img)
            url = b64 if str(b64).startswith('data:image/') else f'data:image/jpeg;base64,{b64}'
            caption = row.get('caption') or (
                f"candidate_id={row.get('candidate_id')} view={row.get('view')} "
                f"original_bbox={row.get('original_bbox')} bbox_space=full_view roles={row.get('roles')}"
            )
            blocks.append({'type': 'text', 'text': str(caption)})
            blocks.append({'type': 'image_url', 'image_url': {'url': url}})
        return blocks

    def _resolve_visual_region(self, region):
        if not isinstance(region, dict):
            return None
        if 'candidate_id' not in region or region.get('candidate_id') is None:
            return None
        candidate_id = region.get('candidate_id')
        if type(candidate_id) is not str or not candidate_id.strip():
            return {'valid': False, 'reason': 'Malformed candidate_id; must be a non-empty string.', 'candidate_id': candidate_id}
        mapper = getattr(self, 'mapper', None)
        observation_id = getattr(mapper, 'observation_id', None)
        candidates = getattr(mapper, 'current_visual_candidates', None) or {}
        if isinstance(candidates, (list, tuple)):
            candidates = {
                r.get('candidate_id'): r
                for r in candidates
                if isinstance(r, dict) and isinstance(r.get('candidate_id'), str)
            }
        masks = getattr(mapper, 'current_observation_masks', None) or {}
        endpoint_text = region.get('endpoint_text')
        if endpoint_text is None:
            endpoint_text = self._endpoint_clause_text()
        evidence_role = region.get('evidence_role') or 'primary'
        match_role = region.get('_match_role') or ('primary' if evidence_role == 'spatial_landmark' else evidence_role)
        resolved = resolve_candidate_region(
            region, candidates=candidates, masks=masks, observation_id=observation_id,
            endpoint_text=endpoint_text, evidence_role=match_role,
        )
        if not resolved.get('valid'):
            return resolved
        if region.get('view') and str(region.get('view')).lower() != str(resolved.get('view') or '').lower():
            return {'valid': False, 'reason': 'candidate_view_mismatch', 'candidate_id': candidate_id}
        if region.get('entity_id') is not None:
            if type(region.get('entity_id')) is not int or region.get('entity_id') != resolved.get('entity_id'):
                return {'valid': False, 'reason': 'candidate_entity_mismatch', 'candidate_id': candidate_id}
        return resolved

    def _overlapping_same_candidate_jambs(self, first, second):
        if first.get('candidate_id') is None or first.get('candidate_id') != second.get('candidate_id'):
            return False
        try:
            a = np.asarray(first.get('bbox'), dtype=float)
            b = np.asarray(second.get('bbox'), dtype=float)
            if a.shape != (4,) or b.shape != (4,):
                return True
            inter = np.maximum(0.0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
            return bool(float(np.prod(inter)) > 0.0)
        except (TypeError, ValueError):
            return True

    def _is_rolling_back(self, ctx=None):
        if isinstance(ctx, dict) and ctx.get('rolling_back'):
            return True
        if getattr(self, '_rolling_back', False) or getattr(self, 'rolling_back', False):
            return True
        for attr in ('_rollback_state', 'rollback_state', '_wrong_path_rollback_state'):
            st = getattr(self, attr, None)
            if isinstance(st, dict) and (st.get('active') or st.get('rolling_back')):
                return True
        return False

    def _spatial_branch_revision(self):
        trace_markers = []
        trace = getattr(self, 'navigation_trace', None)
        if not isinstance(trace, (list, tuple)):
            trace = getattr(self, '_navigation_trace', None)
        if isinstance(trace, (list, tuple)):
            for idx, entry in enumerate(trace):
                if not isinstance(entry, dict):
                    continue
                ev = str(entry.get('event') or entry.get('type') or entry.get('status') or '')
                if (
                    ev.startswith('wrong_path_rollback')
                    or ev in ('rollback_triggered', 'rollback_completed', 'physical_recovery', 'stuck_recovery', 'branch_invalidated')
                    or bool(entry.get('wrong_path_rollback') or entry.get('rejected_by_rollback') or entry.get('physical_recovery'))
                ):
                    trace_markers.append((
                        idx, ev,
                        entry.get('observation_id') if entry.get('observation_id') is not None else entry.get('step'),
                        entry.get('condition_id') or entry.get('subtask_id') or entry.get('scope'),
                    ))
        ledger_markers = []
        for attr in (
            'subtask_execution_ledger', '_subtask_execution_ledger',
            'execution_ledger', '_execution_ledger',
            'route_condition_ledger', '_route_condition_ledger',
            'condition_ledger', '_condition_ledger',
        ):
            ledger = getattr(self, attr, None)
            items = list(ledger.items()) if isinstance(ledger, dict) else (list(enumerate(ledger)) if isinstance(ledger, (list, tuple)) else [])
            for key, item in items:
                if not isinstance(item, dict):
                    continue
                status = str(item.get('status') or '')
                if (
                    item.get('rejected_by_rollback') or item.get('invalidated_by_rollback')
                    or item.get('physical_recovery') or item.get('rollback_revision') is not None
                    or item.get('rejected_revision') is not None or item.get('rollback_count')
                    or item.get('recovery_count') or status in ('rejected_by_rollback', 'rolled_back', 'invalidated', 'recovered')
                ):
                    ledger_markers.append((
                        attr, str(key),
                        str(item.get('condition_id') or item.get('subtask_id') or item.get('scope') or ''),
                        bool(item.get('rejected_by_rollback') or item.get('invalidated_by_rollback')),
                        status, item.get('rollback_revision'), item.get('rejected_revision'),
                        item.get('rollback_count'), item.get('recovery_count'), bool(item.get('physical_recovery')),
                    ))
        counters = (
            getattr(self, '_wrong_path_rollback_count', None),
            getattr(self, '_rollback_revision', None),
            getattr(self, '_branch_epoch', None),
            getattr(self, '_physical_recovery_count', None),
        )
        return (tuple(trace_markers), tuple(ledger_markers), counters)

    def _sync_spatial_relation_frames(self):
        frames = getattr(self, '_spatial_relation_frames', None)
        if not isinstance(frames, dict):
            frames = {}
            self._spatial_relation_frames = frames
        curr_rev = self._spatial_branch_revision()
        last_rev = getattr(self, '_spatial_relation_frame_revision', None)
        if last_rev is not None and curr_rev != last_rev:
            frames.clear()
        self._spatial_relation_frame_revision = curr_rev
        return frames, curr_rev

    def _maybe_freeze_observer_frame(self, frame_key, target_packet, landmark_packet, ctx=None):
        frames, _ = self._sync_spatial_relation_frames()
        if self._is_rolling_back(ctx):
            return None
        if frame_key in frames:
            return frames[frame_key]
        mapper = getattr(self, 'mapper', None)
        obs_id = getattr(mapper, 'observation_id', None)
        if type(obs_id) is not int or not isinstance(target_packet, dict) or not isinstance(landmark_packet, dict):
            return None
        if not (target_packet.get('valid') and landmark_packet.get('valid')):
            return None
        if target_packet.get('observation_id') != obs_id or landmark_packet.get('observation_id') != obs_id:
            return None
        t_view = str(target_packet.get('view') or target_packet.get('target_view') or '').lower()
        l_view = str(landmark_packet.get('view') or landmark_packet.get('target_view') or '').lower()
        is_doorway_target = target_packet.get('endpoint_kind') == 'doorway'
        if not t_view or t_view not in _VIEWS or not l_view or l_view not in _VIEWS:
            return None
        if t_view != l_view and not is_doorway_target:
            return None
        t_cid, l_cid = target_packet.get('candidate_id'), landmark_packet.get('candidate_id')
        t_eid, l_eid = target_packet.get('target_id'), landmark_packet.get('target_id')
        if ((t_cid is None and t_eid is None) and not is_doorway_target) or (l_cid is None and l_eid is None):
            return None
        if (t_cid is not None and t_cid == l_cid) or (t_eid is not None and t_eid == l_eid):
            return None
        obs = getattr(self, '_verification_obs', None) or getattr(self, 'curr_obs', None) or {}
        ref_view = ('front' if l_view == 'back' else l_view) if is_doorway_target else t_view
        view_obs = obs if ref_view == 'front' else (obs.get(ref_view) if isinstance(obs.get(ref_view), dict) else {})
        state = view_obs.get('agent_state') if isinstance(view_obs, dict) else None
        try:
            l_center = np.asarray(landmark_packet.get('world_center'), dtype=float)
            if l_center.shape != (3,) or not np.isfinite(l_center).all() or not np.all(np.abs(l_center) < 1e6):
                return None
            sensor_rot = state.sensor_states['color_sensor'].rotation
            rot = np.asarray(mapper.rotation_func(sensor_rot), dtype=float)
            if rot.shape != (3, 3) or not np.isfinite(rot).all() or not np.all(np.abs(rot) <= 10.0):
                return None
            if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-2) or abs(abs(float(np.linalg.det(rot))) - 1.0) > 5e-2:
                return None
            right_3d = rot @ np.array([1.0, 0.0, 0.0], dtype=float)
            right_xy = np.array([right_3d[0], right_3d[1], 0.0], dtype=float)
            norm = float(np.linalg.norm(right_xy[:2]))
            if not np.isfinite(norm) or norm < 1e-6:
                return None
            right_axis = [float(right_xy[0] / norm), float(right_xy[1] / norm), 0.0]
            origin_arr = np.asarray(mapper.translation_func(state.position) - mapper.initial_position, dtype=float)
            if origin_arr.shape != (3,) or not np.isfinite(origin_arr).all() or not np.all(np.abs(origin_arr) < 1e6):
                return None
            ref_eid = l_eid if type(l_eid) is int else (
                landmark_packet.get('entity_id') if type(landmark_packet.get('entity_id')) is int else None
            )
            ref_cls = str(landmark_packet.get('selected_detector_class_name') or '').strip().lower() or None
            frame = {
                'kind': 'frozen_observer',
                'source': 'measured_camera',
                'observation_id': int(obs_id),
                'view': t_view,
                'right_axis': right_axis,
                'origin': [float(origin_arr[0]), float(origin_arr[1]), float(origin_arr[2])],
                'reference_class_name': ref_cls,
                'reference_target_id': ref_eid,
                'reference_world_center': [float(l_center[0]), float(l_center[1]), float(l_center[2])],
            }
            while len(frames) >= 16:
                frames.pop(next(iter(frames)))
            frames[frame_key] = frame
            return frame
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            return None

    def _collect_spatial_relation_observation(self, parsed, ctx, result):
        if not self._visual_enabled() or not isinstance(result, dict):
            return None
        parsed = parsed if isinstance(parsed, dict) else {}
        ctx = ctx if isinstance(ctx, dict) else {}
        if not ctx.get('final') or self._is_rolling_back(ctx):
            return None
        clause_text = self._endpoint_clause_text(ctx)
        spec = parse_endpoint_relation(clause_text)
        if not spec.get('required'):
            return None
        mapper = getattr(self, 'mapper', None)
        obs_id = getattr(mapper, 'observation_id', None)
        raw_cid = parsed.get('target_candidate_id')
        raw_eid = parsed.get('target_entity_id')
        cand_map = getattr(mapper, 'current_visual_candidates', None)
        has_valid_cid = (
            isinstance(raw_cid, str) and bool(raw_cid.strip())
            and isinstance(cand_map, dict) and raw_cid in cand_map
        )
        has_valid_eid = raw_cid is None and type(raw_eid) is int
        if not (has_valid_cid or has_valid_eid):
            target_packet = {
                'valid': False,
                'accepted': False,
                'reason': 'spatial_target_requires_detector_mask',
                'evidence_role': 'spatial_observation_target',
                'expected_object_class': spec.get('target_class'),
                'selected_detector_class_name': None,
                'candidate_id': raw_cid if has_valid_cid else None,
            }
        else:
            target_region = {
                'role': 'target',
                'view': parsed.get('target_view'),
                'bbox': parsed.get('target_bbox'),
                'entity_id': raw_eid if type(raw_eid) is int else None,
                'candidate_id': raw_cid if has_valid_cid else None,
                'bbox_space': parsed.get('target_bbox_space'),
                'endpoint_text': spec.get('target_text', ''),
                'evidence_role': 'primary',
            }
            target_packet = self._ground_region(target_region, 'near_object')
            if isinstance(target_packet, dict):
                target_packet = {**target_packet, 'accepted': False, 'evidence_role': 'spatial_observation_target'}
            if isinstance(target_packet, dict) and target_packet.get('valid'):
                t_cid = target_packet.get('candidate_id')
                t_eid = target_packet.get('target_id') if type(target_packet.get('target_id')) is int else (
                    target_packet.get('entity_id') if type(target_packet.get('entity_id')) is int else None
                )
                det_cls = target_packet.get('selected_detector_class_name')
                min_conf = float((getattr(self, 'navigation_cfg', None) or {}).get('endpoint_mask_min_confidence', 0.48))
                cand_entry = cand_map.get(t_cid) if isinstance(cand_map, dict) and isinstance(t_cid, str) else None
                conf_vals = []
                for src_d in (target_packet, cand_entry):
                    if isinstance(src_d, dict) and 'confidence' in src_d:
                        conf_vals.append(src_d.get('confidence'))
                conf_ok = True
                if isinstance(t_cid, str) and t_cid.strip():
                    if not isinstance(cand_entry, dict) or not conf_vals:
                        conf_ok = False
                for cv in conf_vals:
                    try:
                        cv_f = float(cv)
                        if isinstance(cv, bool) or not np.isfinite(cv_f) or cv_f < min_conf:
                            conf_ok = False
                    except (TypeError, ValueError, OverflowError):
                        conf_ok = False
                depth_val = target_packet.get('depth_m') if target_packet.get('depth_m') is not None else target_packet.get('distance_m')
                try:
                    depth_ok = not isinstance(depth_val, bool) and np.isfinite(float(depth_val)) and float(depth_val) > 0.0
                except (TypeError, ValueError, OverflowError):
                    depth_ok = False
                if target_packet.get('source') != 'detector_mask' or (
                    not (isinstance(t_cid, str) and t_cid.strip()) and type(t_eid) is not int
                ):
                    target_packet = {**target_packet, 'valid': False, 'reason': 'spatial_target_requires_detector_mask'}
                elif type(obs_id) is not int or target_packet.get('observation_id') != obs_id:
                    target_packet = {**target_packet, 'valid': False, 'reason': 'stale_spatial_target_observation'}
                elif not isinstance(det_cls, str) or not det_cls.strip() or not detector_matches_endpoint(det_cls, spec.get('target_text', '')):
                    target_packet = {**target_packet, 'valid': False, 'reason': 'spatial_target_class_mismatch'}
                elif not conf_ok:
                    target_packet = {**target_packet, 'valid': False, 'reason': 'low_confidence_spatial_target'}
                elif not depth_ok:
                    target_packet = {**target_packet, 'valid': False, 'reason': 'invalid_spatial_target_depth'}
        temp_diag = {'endpoint_grounding': target_packet}
        self._verify_spatial_relation(parsed, ctx, temp_diag)
        result['spatial_observation_grounding'] = _clean_serializable(target_packet)
        if 'spatial_landmark_grounding' in temp_diag:
            result['spatial_landmark_grounding'] = temp_diag['spatial_landmark_grounding']
        if 'spatial_relation' in temp_diag:
            result['spatial_relation'] = temp_diag['spatial_relation']
        return None

    def _verify_spatial_relation(self, parsed, ctx, result):
        if not self._visual_enabled():
            return None
        parsed = parsed if isinstance(parsed, dict) else {}
        ctx = ctx if isinstance(ctx, dict) else {}
        clause_text = self._endpoint_clause_text(ctx)
        spec = parse_endpoint_relation(clause_text)
        mapper = getattr(self, 'mapper', None)
        obs_id = getattr(mapper, 'observation_id', None)
        target_packet = result.get('endpoint_grounding')
        if not spec.get('required'):
            result['spatial_relation'] = _clean_serializable(evaluate_spatial_relation(
                spec, target_packet, None, None, observation_id=obs_id,
            ))
            return None
        raw_lm = parsed.get('spatial_landmark_region')
        if not isinstance(raw_lm, dict):
            landmark_packet = {
                'valid': False,
                'reason': 'missing_spatial_landmark_region' if spec.get('landmark_text') else (spec.get('reason') or 'unsupported_spatial_relation'),
                'evidence_role': 'spatial_landmark',
                'expected_object_class': spec.get('landmark_class'),
                'selected_detector_class_name': None,
                'candidate_id': None,
            }
        elif raw_lm.get('role') != 'spatial_landmark':
            landmark_packet = {
                'valid': False,
                'reason': 'invalid_spatial_landmark_role',
                'evidence_role': 'spatial_landmark',
                'expected_object_class': spec.get('landmark_class'),
                'selected_detector_class_name': None,
                'candidate_id': raw_lm.get('candidate_id'),
            }
        elif not spec.get('landmark_text'):
            landmark_packet = {
                'valid': False,
                'reason': spec.get('reason') or 'unsupported_spatial_relation',
                'evidence_role': 'spatial_landmark',
                'expected_object_class': spec.get('landmark_class'),
                'selected_detector_class_name': None,
                'candidate_id': raw_lm.get('candidate_id'),
            }
        else:
            lm_region = {
                'role': 'spatial_landmark',
                'view': raw_lm.get('view'),
                'bbox': raw_lm.get('bbox'),
                'entity_id': raw_lm.get('entity_id'),
                'candidate_id': raw_lm.get('candidate_id'),
                'bbox_space': raw_lm.get('bbox_space'),
                'endpoint_text': spec.get('landmark_text', ''),
                'evidence_role': 'spatial_landmark',
            }
            landmark_packet = self._ground_region(lm_region, 'near_object')
            raw_has_id = (
                (isinstance(raw_lm.get('candidate_id'), str) and bool(raw_lm.get('candidate_id').strip()))
                or type(raw_lm.get('entity_id')) is int
            )
            if isinstance(landmark_packet, dict) and landmark_packet.get('valid'):
                l_cid = landmark_packet.get('candidate_id')
                l_eid = landmark_packet.get('target_id') if type(landmark_packet.get('target_id')) is int else (
                    landmark_packet.get('entity_id') if type(landmark_packet.get('entity_id')) is int else None
                )
                det_cls = landmark_packet.get('selected_detector_class_name')
                min_conf = float((getattr(self, 'navigation_cfg', None) or {}).get('endpoint_mask_min_confidence', 0.48))
                cand_map = getattr(mapper, 'current_visual_candidates', None)
                cand_entry = cand_map.get(l_cid) if isinstance(cand_map, dict) and isinstance(l_cid, str) else None
                conf_vals = []
                for src_d in (landmark_packet, cand_entry):
                    if isinstance(src_d, dict) and 'confidence' in src_d:
                        conf_vals.append(src_d.get('confidence'))
                conf_ok = True
                if isinstance(l_cid, str) and l_cid.strip():
                    if not isinstance(cand_entry, dict) or not conf_vals:
                        conf_ok = False
                for cv in conf_vals:
                    try:
                        cv_f = float(cv)
                        if isinstance(cv, bool) or not np.isfinite(cv_f) or cv_f < min_conf:
                            conf_ok = False
                    except (TypeError, ValueError, OverflowError):
                        conf_ok = False
                if not raw_has_id or landmark_packet.get('source') != 'detector_mask' or (
                    not (isinstance(l_cid, str) and l_cid.strip()) and type(l_eid) is not int
                ):
                    landmark_packet = {**landmark_packet, 'valid': False, 'reason': 'spatial_landmark_requires_detector_mask'}
                elif type(obs_id) is not int or landmark_packet.get('observation_id') != obs_id:
                    landmark_packet = {**landmark_packet, 'valid': False, 'reason': 'stale_spatial_landmark_observation'}
                elif not isinstance(det_cls, str) or not det_cls.strip() or not detector_matches_endpoint(det_cls, spec.get('landmark_text', '')):
                    landmark_packet = {**landmark_packet, 'valid': False, 'reason': 'spatial_landmark_class_mismatch'}
                elif not conf_ok:
                    landmark_packet = {**landmark_packet, 'valid': False, 'reason': 'low_confidence_spatial_landmark'}
            if isinstance(target_packet, dict) and isinstance(landmark_packet, dict) and landmark_packet.get('valid'):
                t_cid, l_cid = target_packet.get('candidate_id'), landmark_packet.get('candidate_id')
                t_eid = target_packet.get('target_id') if target_packet.get('target_id') is not None else target_packet.get('entity_id')
                l_eid = landmark_packet.get('target_id') if landmark_packet.get('target_id') is not None else landmark_packet.get('entity_id')
                if (t_cid is not None and t_cid == l_cid) or (t_eid is not None and t_eid == l_eid):
                    landmark_packet = {
                        **landmark_packet,
                        'valid': False,
                        'reason': 'spatial_landmark_same_instance',
                    }
        checks = ctx.get('checks')
        last_check = checks[-1] if isinstance(checks, (list, tuple)) and checks and isinstance(checks[-1], dict) else {}
        scope = ctx.get('scope') or last_check.get('scope') or ctx.get('key') or last_check.get('key') or 'endpoint'
        cond_id = (
            ctx.get('condition_id')
            or ctx.get('subtask_id')
            or last_check.get('condition_id')
            or last_check.get('subtask_id')
            or last_check.get('id')
            or last_check.get('key')
            or ctx.get('key')
        )
        frames, branch_rev = self._sync_spatial_relation_frames()
        frame_key = (
            scope,
            cond_id,
            clause_text.lower(),
            spec.get('relation'),
            spec.get('target_text'),
            spec.get('landmark_text'),
            branch_rev,
        )
        if frame_key in frames and isinstance(landmark_packet, dict) and landmark_packet.get('valid'):
            existing_frame = frames[frame_key]
            ref_eid = existing_frame.get('reference_target_id')
            cur_eid = landmark_packet.get('target_id') if type(landmark_packet.get('target_id')) is int else (
                landmark_packet.get('entity_id') if type(landmark_packet.get('entity_id')) is int else None
            )
            try:
                cur_center = np.asarray(landmark_packet.get('world_center'), dtype=float)
                ref_center = np.asarray(existing_frame.get('reference_world_center'), dtype=float)
                centers_valid = (
                    cur_center.shape == (3,) and ref_center.shape == (3,)
                    and np.isfinite(cur_center).all() and np.isfinite(ref_center).all()
                )
            except (TypeError, ValueError, OverflowError):
                centers_valid = False
            if not centers_valid:
                landmark_packet = {**landmark_packet, 'valid': False, 'reason': 'current_reference_instance_changed'}
            elif type(ref_eid) is int and type(cur_eid) is int:
                if cur_eid != ref_eid:
                    landmark_packet = {**landmark_packet, 'valid': False, 'reason': 'current_reference_instance_changed'}
            elif float(np.linalg.norm(cur_center - ref_center)) > 0.65:
                landmark_packet = {**landmark_packet, 'valid': False, 'reason': 'current_reference_instance_changed'}
        result['spatial_landmark_grounding'] = _clean_serializable(landmark_packet)
        frame = self._maybe_freeze_observer_frame(frame_key, target_packet, landmark_packet, ctx=ctx)
        min_sep = float((getattr(self, 'navigation_cfg', None) or {}).get('spatial_relation_min_separation_m', 0.15))
        diag = evaluate_spatial_relation(
            spec, target_packet, landmark_packet, frame,
            observation_id=obs_id, minimum_separation_m=min_sep,
        )
        if isinstance(landmark_packet, dict) and landmark_packet.get('reason') == 'current_reference_instance_changed':
            diag = {**diag, 'status': 'unknown', 'reason': 'current_reference_instance_changed'}
        result['spatial_relation'] = _clean_serializable(diag)
        if diag.get('status') == 'satisfied':
            return None
        return str(diag.get('reason') or f"spatial_relation_{diag.get('status', 'unknown')}")


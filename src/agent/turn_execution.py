"""Bind an instructed turn to its landmark and first measured departure.

This does not satisfy route conditions or infer landmark identity from depth.
Unknown localisation permits observation/approach, never an automatic turn.
"""
import json
import math
import re

import numpy as np

from endpoint_grounding import consistent_target_observations, ground_target_depth


def _visual_turn_enabled(agent):
    cfg = getattr(agent, 'navigation_cfg', None)
    return bool(isinstance(cfg, dict) and cfg.get('task_visual_enabled') is True)


def _mask_min_confidence(agent):
    cfg = getattr(agent, 'navigation_cfg', None)
    val = cfg.get('endpoint_mask_min_confidence', 0.48) if isinstance(cfg, dict) else 0.48
    return float(val) if isinstance(val, (int, float)) and math.isfinite(float(val)) else 0.48


def _wrap_yaw(angle):
    return float(math.atan2(math.sin(angle), math.cos(angle)))


def _valid_rot3(mat):
    try:
        arr = np.asarray(mat, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.shape != (3, 3) or not np.all(np.isfinite(arr)):
        return None
    if not (np.allclose(arr @ arr.T, np.eye(3), atol=1e-3) and np.isclose(abs(np.linalg.det(arr)), 1.0, atol=1e-3)):
        return None
    return arr


def _valid_vec3(vec):
    try:
        arr = np.asarray(vec, dtype=float)
    except (TypeError, ValueError):
        return None
    return arr if arr.shape == (3,) and np.all(np.isfinite(arr)) else None


def _current_base_heading(agent):
    mapper = getattr(agent, 'mapper', None)
    rot = _valid_rot3(getattr(mapper, 'current_rotation', None))
    if rot is None:
        return None
    forward = -rot[:2, 2]
    norm = float(np.linalg.norm(forward))
    if not math.isfinite(norm) or norm <= 1e-6:
        return None
    return float(math.atan2(float(forward[1]), float(forward[0])))


def _extract_sensor_pose(agent, view, packet=None):
    obs = getattr(agent, 'curr_obs', {}) or {}
    packet_obs = obs if view == 'front' else (obs.get(view, {}) if isinstance(obs, dict) else {})
    state = packet_obs.get('agent_state') if isinstance(packet_obs, dict) else None
    mapper = getattr(agent, 'mapper', None)
    if state is not None and mapper is not None:
        try:
            rot = _valid_rot3(mapper.rotation_func(state.sensor_states['color_sensor'].rotation))
            pos = _valid_vec3(mapper.translation_func(state.position) - mapper.initial_position)
            if pos is not None and rot is not None:
                return pos, rot
        except (AttributeError, TypeError, ValueError, KeyError):
            pass
    if isinstance(packet, dict):
        pos = _valid_vec3(packet.get('camera_position'))
        rot = _valid_rot3(packet.get('camera_rotation'))
        if pos is not None and rot is not None:
            return pos, rot
    return None, None


def _clause_state_entry(agent, clause):
    inst = getattr(agent, 'instruction_obj', None)
    subtask_key = inst.get_current_subtask_key() if hasattr(inst, 'get_current_subtask_key') else 'P1'
    cond_id = clause['condition_id']
    landmark = (clause.get('landmark') or '').strip().lower()
    ledger = getattr(agent, '_execution_ledger', None) or []
    trace = getattr(agent, 'navigation_trace', None) or []
    rolling_back = bool(getattr(agent, 'rolling_back', False))

    trace_markers = []
    max_rollback_step = None
    for tidx, item in enumerate(trace):
        if not isinstance(item, dict):
            continue
        ev = str(item.get('event') or item.get('type') or '')
        reason = str(item.get('reason') or '')
        if (ev.startswith('wrong_path_rollback')
                or 'rollback' in ev.lower()
                or 'blocked_path' in ev.lower()
                or 'rollback' in reason.lower()
                or bool(item.get('recovery'))
                or (ev == 'instruction_turn_departure' and item.get('accepted') is False)):
            t_step = item.get('step')
            trace_markers.append((tidx, ev, t_step, item.get('subtask_id')))
            if isinstance(t_step, (int, np.integer)) and not isinstance(t_step, bool):
                max_rollback_step = int(t_step) if max_rollback_step is None else max(max_rollback_step, int(t_step))

    clause_boundary_idx = -1
    reset_idx = -1
    revision_markers = []
    for idx, row in enumerate(ledger):
        if not isinstance(row, dict):
            continue
        row_subtask = row.get('subtask_id') if row.get('subtask_id') is not None else row.get('subtask_key')
        iturn = row.get('instruction_turn') if isinstance(row.get('instruction_turn'), dict) else {}
        iturn_subtask = iturn.get('subtask_id') if iturn.get('subtask_id') is not None else iturn.get('subtask_key')
        is_rejected = bool(
            row.get('rejected_by_rollback')
            or row.get('rejected')
            or row.get('rejected_branch')
            or row.get('branch_rejected')
        )
        is_recovery = bool(row.get('recovery'))
        r_step = row.get('step')
        step_rolled_back = (
            max_rollback_step is not None
            and isinstance(r_step, (int, np.integer))
            and not isinstance(r_step, bool)
            and int(r_step) <= max_rollback_step
        )
        other_scope = (
            row_subtask not in (None, subtask_key)
            or iturn.get('condition_id') not in (None, cond_id)
            or iturn_subtask not in (None, subtask_key)
        )
        if other_scope:
            clause_boundary_idx = idx
        if is_recovery or is_rejected or step_rolled_back or other_scope:
            reset_idx = idx
            revision_markers.append((
                idx,
                r_step,
                is_recovery,
                is_rejected,
                step_rolled_back,
                row_subtask,
                iturn.get('condition_id'),
            ))

    recovery_epoch = (
        int(getattr(agent, '_recovery_epoch', 0) or 0),
        int(getattr(agent, '_rollback_epoch', 0) or 0),
        int(getattr(agent, '_rollback_count', 0) or 0),
        int(getattr(agent, 'rollback_count', 0) or 0),
        int(getattr(agent, 'wrong_path_rollback_count', 0) or 0),
    )
    epoch_token = (
        reset_idx,
        tuple(revision_markers),
        tuple(trace_markers),
        recovery_epoch,
        rolling_back,
    )
    cache = getattr(agent, '_visual_turn_state', None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(agent, '_visual_turn_state', cache)
    key = (subtask_key, cond_id, landmark)
    prev_entry = cache.get(key)
    if prev_entry is not None and len(ledger) < int(prev_entry.get('seen_ledger_len', 0)):
        prev_entry = None
    entry = prev_entry
    if entry is None or entry.get('epoch_token') != epoch_token:
        min_ledger_idx = clause_boundary_idx + 1
        if prev_entry is not None and (
            prev_entry.get('trace_markers') != tuple(trace_markers)
            or prev_entry.get('recovery_epoch') != recovery_epoch
            or rolling_back
        ):
            min_ledger_idx = max(min_ledger_idx, int(prev_entry.get('seen_ledger_len', 0)))
        entry = {
            'subtask_id': subtask_key,
            'condition_id': cond_id,
            'landmark': landmark,
            'epoch_token': epoch_token,
            'trace_markers': tuple(trace_markers),
            'recovery_epoch': recovery_epoch,
            'min_ledger_idx': min_ledger_idx,
            'seen_ledger_len': len(ledger),
            'incoming_heading_rad': None,
            'incoming_heading_source': None,
            'incoming_heading_step': None,
            'samples': [],
            'confirmed_pair': None,
            'reorientation_count': 0,
        }
        cache[key] = entry
        while len(cache) > 8:
            cache.pop(next(iter(cache)))

    start_idx = max(clause_boundary_idx + 1, int(entry.get('min_ledger_idx', 0)))
    move_heading = move_step = None
    reorient_count = 0
    has_turn_since_reset = False
    for idx_in_slice, row in enumerate(ledger[start_idx:], start=start_idx):
        if not isinstance(row, dict):
            continue
        if (row.get('recovery')
                or row.get('rejected_by_rollback')
                or row.get('rejected')
                or row.get('rejected_branch')
                or row.get('branch_rejected')):
            continue
        r_step = row.get('step')
        if (max_rollback_step is not None
                and isinstance(r_step, (int, np.integer))
                and not isinstance(r_step, bool)
                and int(r_step) <= max_rollback_step):
            continue
        row_subtask = row.get('subtask_id') if row.get('subtask_id') is not None else row.get('subtask_key')
        if row_subtask not in (None, subtask_key):
            continue
        req = row.get('requested_action') if isinstance(row.get('requested_action'), dict) else {}
        req_subtask = req.get('subtask_id') if req.get('subtask_id') is not None else req.get('subtask_key')
        if req_subtask not in (None, subtask_key):
            continue
        atype = row.get('action_type') or req.get('type')
        if req.get('type') == 'landmark_reorientation' and req.get('condition_id') == cond_id:
            if idx_in_slice > reset_idx and atype in (None, 'turn', 'landmark_reorientation'):
                reorient_count += 1
                has_turn_since_reset = True
            continue
        if atype == 'turn' or req.get('type') == 'turn':
            if idx_in_slice > reset_idx:
                has_turn_since_reset = True
            continue
        if atype in ('move_forward', 'waypoint', 'navigate') or req.get('type') in ('move_forward', 'waypoint', 'navigate'):
            p0, p1 = row.get('pose_before'), row.get('pose')
            if isinstance(p0, (list, tuple, np.ndarray)) and isinstance(p1, (list, tuple, np.ndarray)) and len(p0) >= 2 and len(p1) >= 2:
                try:
                    dx, dy = float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(dx) and math.isfinite(dy) and math.hypot(dx, dy) >= 0.10:
                    move_heading = float(math.atan2(dy, dx))
                    move_step = row.get('step')

    entry['reorientation_count'] = reorient_count
    entry['seen_ledger_len'] = len(ledger)
    if rolling_back:
        entry['samples'] = []
        entry['confirmed_pair'] = None
        return entry
    if move_heading is not None:
        entry['incoming_heading_rad'] = move_heading
        entry['incoming_heading_source'] = 'move_forward'
        entry['incoming_heading_step'] = move_step
    elif entry['incoming_heading_rad'] is None and not has_turn_since_reset:
        base_heading = _current_base_heading(agent)
        if base_heading is not None:
            entry['incoming_heading_rad'] = base_heading
            entry['incoming_heading_source'] = 'initial_base'
            entry['incoming_heading_step'] = getattr(getattr(agent, 'mapper', None), 'observation_step', None)
    curr_obs_id = getattr(getattr(agent, 'mapper', None), 'observation_id', None)
    if isinstance(curr_obs_id, (int, np.integer)) and not isinstance(curr_obs_id, bool):
        oid = int(curr_obs_id)
        entry['samples'] = [s for s in entry['samples'] if 0 <= oid - s['observation_id'] <= 12]
        pair = entry.get('confirmed_pair')
        if pair is not None and not (0 <= oid - pair[0]['observation_id'] <= 12 and 0 <= oid - pair[1]['observation_id'] <= 12):
            entry['confirmed_pair'] = None
    return entry


def _update_landmark_confirmation(agent, entry, context, evidence):
    if entry is None or not isinstance(entry, dict):
        return False
    if getattr(agent, 'rolling_back', False):
        entry['samples'] = []
        entry['confirmed_pair'] = None
        if isinstance(evidence, dict):
            evidence['confirmed_changed_view'] = False
        return False
    if not isinstance(evidence, dict) or not evidence.get('valid'):
        entry['confirmed_pair'] = None
        if isinstance(evidence, dict):
            evidence['confirmed_changed_view'] = False
        return False
    dist = evidence.get('median_distance_m')
    obs_id = evidence.get('observation_id')
    mapper = getattr(agent, 'mapper', None)
    mapper_obs_id = getattr(mapper, 'observation_id', None)
    if obs_id is None:
        obs_id = mapper_obs_id
    world_center = _valid_vec3(evidence.get('world_center'))
    view = (
        evidence.get('view')
        or evidence.get('target_view')
        or (context.get('landmark_view') if isinstance(context, dict) else None)
    )
    sensor_pos, sensor_rot = _extract_sensor_pose(agent, view, evidence)
    base_pos = _valid_vec3(getattr(mapper, 'current_position', None))
    base_rot = _valid_rot3(getattr(mapper, 'current_rotation', None))
    max_dist = 2.2 if (isinstance(context, dict) and context.get('status') in ('approaching', 'observe')) else 1.85
    if (dist is None or isinstance(dist, bool) or not isinstance(dist, (int, float, np.integer, np.floating))
            or not math.isfinite(float(dist)) or not (0.0 < float(dist) <= max_dist)
            or not (isinstance(obs_id, (int, np.integer)) and not isinstance(obs_id, bool) and int(obs_id) > 0)
            or (mapper_obs_id is not None and int(obs_id) != int(mapper_obs_id))
            or world_center is None or sensor_pos is None or sensor_rot is None or base_pos is None or base_rot is None):
        entry['confirmed_pair'] = None
        evidence['confirmed_changed_view'] = False
        return False
    oid = int(obs_id)
    curr = {
        'observation_id': oid,
        'packet': dict(evidence, observation_id=oid, endpoint_kind=evidence.get('endpoint_kind', 'near_object')),
        'sensor_position': sensor_pos.copy(),
        'sensor_rotation': sensor_rot.copy(),
        'base_position': base_pos.copy(),
        'base_rotation': base_rot.copy(),
        'world_center': world_center.copy(),
    }
    entry['samples'] = [s for s in entry['samples'] if 0 <= oid - s['observation_id'] <= 12]
    matched_pair = None
    for prior in entry['samples']:
        if not (1 <= oid - prior['observation_id'] <= 12):
            continue
        r1, r2 = prior['base_rotation'], curr['base_rotation']
        if np.linalg.det(r1) * np.linalg.det(r2) <= 0:
            continue
        base_moved = float(np.linalg.norm(curr['base_position'] - prior['base_position']))
        base_turn = float(np.degrees(np.arccos(np.clip((np.trace(r2 @ r1.T) - 1.0) / 2.0, -1.0, 1.0))))
        if base_moved < 0.10 and base_turn < 8.0:
            continue
        corr = consistent_target_observations(
            prior['packet'], curr['packet'],
            first_position=prior['sensor_position'], second_position=curr['sensor_position'],
            first_rotation=prior['sensor_rotation'], second_rotation=curr['sensor_rotation'],
            max_center_shift_m=0.65, min_translation_m=0.10, min_turn_degrees=8.0)
        if corr.get('consistent') is True:
            matched_pair = (prior, curr)
            break
    if matched_pair is not None:
        entry['confirmed_pair'] = matched_pair
    elif entry.get('confirmed_pair') is not None:
        s1, s2 = entry['confirmed_pair']
        tid = curr['packet'].get('target_id')
        valid_existing = (1 <= s2['observation_id'] - s1['observation_id'] <= 12
                          and 0 <= oid - s1['observation_id'] <= 12
                          and 0 <= oid - s2['observation_id'] <= 12)
        for ref in (s1, s2):
            rtid = ref['packet'].get('target_id')
            if tid is not None and rtid is not None and str(tid) != str(rtid):
                valid_existing = False
            if float(np.linalg.norm(curr['world_center'] - ref['world_center'])) > 0.65:
                valid_existing = False
        if not valid_existing:
            entry['confirmed_pair'] = None
    replaced = False
    for idx, s in enumerate(entry['samples']):
        if s['observation_id'] == oid:
            entry['samples'][idx] = curr
            replaced = True
            break
    if not replaced:
        entry['samples'].append(curr)
        if len(entry['samples']) > 8:
            entry['samples'] = entry['samples'][-8:]
    confirmed = entry.get('confirmed_pair') is not None
    evidence['confirmed_changed_view'] = bool(confirmed)
    if confirmed:
        pair = entry['confirmed_pair']
        evidence['confirmation_observation_ids'] = [int(pair[0]['observation_id']), int(pair[1]['observation_id'])]
    return bool(confirmed)


def _turn_prompt_state(agent, clause):
    entry = _clause_state_entry(agent, clause)
    pair = entry.get('confirmed_pair')
    samples = entry['samples']
    confirmed = pair is not None
    seed_obs_id = pair[0]['observation_id'] if pair else (samples[0]['observation_id'] if samples else None)
    mapper = getattr(agent, 'mapper', None)
    curr_oid = getattr(mapper, 'observation_id', None)
    base_pos = _valid_vec3(getattr(mapper, 'current_position', None))
    has_distinct_seed_pose = False
    if not confirmed and samples and base_pos is not None and isinstance(curr_oid, (int, np.integer)) and not isinstance(curr_oid, bool):
        for prior in samples:
            if 1 <= int(curr_oid) - int(prior['observation_id']) <= 12:
                if float(np.linalg.norm(base_pos - prior['base_position'])) >= 0.10:
                    has_distinct_seed_pose = True
                    break
    inc = entry.get('incoming_heading_rad')
    curr = _current_base_heading(agent)
    drift = abs(math.degrees(_wrap_yaw(inc - curr))) if inc is not None and curr is not None else None
    return {
        'condition_id': clause['condition_id'],
        'seed_observation_id': seed_obs_id,
        'confirmed_changed_view': bool(confirmed),
        'second_changed_base_pose_needed': not bool(confirmed or has_distinct_seed_pose),
        'incoming_heading_rad': round(inc, 4) if inc is not None else None,
        'incoming_heading_source': entry.get('incoming_heading_source'),
        'incoming_heading_step': entry.get('incoming_heading_step'),
        'current_heading_drift_deg': round(drift, 2) if drift is not None else None,
        'pending_heading_restore': bool(drift is not None and drift > 8.0),
        'reorientation_count': int(entry.get('reorientation_count', 0)),
    }
from navigation_contracts import completion_conditions, validate_condition_checks
from navigation_control import ActionProtocolError
from route_progress import requires_straight, straight_witness, turn_directions, turn_witness


def turn_landmark(text):
    match = re.search(r'\bat\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:[.;,]|$)', text, re.I)
    return match.group(1).strip() if match else None


def landmark_identity_matches(expected, claimed):
    # Identity is still a current-image semantic claim. Require its specific
    # source nouns, so a bathroom-door ROI cannot stand in for an eye chart.
    words = lambda s: set(re.findall(r'[a-z0-9]+', str(s).lower())) - {'a', 'an', 'the'}
    required = words(expected)
    return bool(required and required == words(claimed))


def first_segment_heading(path, origin):
    try:
        origin = np.asarray(origin, dtype=float)[:2]
        for point in path:
            delta = np.asarray(point, dtype=float)[:2] - origin
            if np.isfinite(delta).all() and np.linalg.norm(delta) >= .03:
                return math.atan2(delta[1], delta[0])
    except (TypeError, ValueError, IndexError):
        pass
    return None


def heading_matches(expected, actual):
    return actual is not None and abs(math.atan2(math.sin(actual-expected), math.cos(actual-expected))) <= math.pi/4


class TurnExecutionGuard:
    def _active_turn_clause(self):
        key, st = self.instruction_obj.get_current_subtask()
        st = st or {}
        checks = completion_conditions(st.get('instruction', ''), key or 'P1')
        records = [r for r in st.get('record', []) if isinstance(r, dict)
                   and not r.get('recovery') and not r.get('rejected_by_rollback')]
        saved = {c['id']: c for c in st.get('route_progress', {}).get('conditions', [])}
        if saved:
            checks, _ = validate_condition_checks(
                checks, checks, records, previous=saved,
                require_instruction_turn=bool(getattr(self, 'navigation_cfg', {}).get('habitat_control_enabled', False)),
                instruction_subtask_id=key)
        after_step = -1
        for condition in checks:
            if condition.get('status') == 'satisfied' and condition.get('execution_checked'):
                after_step = max([after_step] + condition.get('steps', []))
                continue
            directions = turn_directions(condition['text'])
            if len(directions) != 1:
                return None
            direction = {'left': 'LEFT', 'right': 'RIGHT', 'back': 'BACK', 'around': 'BACK'}[directions[0]]
            current_records = [r for r in records if r.get('step', -1) > after_step]
            rotations = [r for r in current_records if r.get('action_type') == 'turn']
            physical, _ = turn_witness(condition['text'], rotations, current_records)
            return {'subtask_id': key, 'condition_id': condition['id'], 'text': condition['text'],
                    'direction': direction, 'landmark': turn_landmark(condition['text']),
                    'measured_turn_witness': bool(physical),
                    'recent_turns': [{'step': r.get('step'), 'degrees': round(math.degrees(r.get('yaw_delta', 0)), 1),
                                      'requested_action': r.get('requested_action')} for r in rotations[-6:]]}
        return None

    def _turn_execution_prompt(self):
        clause = self._active_turn_clause()
        pending = self._current_turn_departure()
        if not clause and not pending:
            return ''
        if clause:
            executed = self._executed_instruction_turn(clause)
            clause['instruction_turn_step'] = executed['step'] if executed is not None else None
            clause['instruction_turn_executed'] = executed is not None
        extra = ''
        if _visual_turn_enabled(self) and clause and clause.get('landmark'):
            extra = (
                '\nTURN LANDMARK LOCALIZATION STATE (advisory telemetry, not geometric proof): '
                + json.dumps(_turn_prompt_state(self, clause))
                + '\nWith visual turn grounding enabled, every named landmark requires TWO real distinct '
                'changed-camera-pose (base translation >=0.10m or base rotation >=8deg) observations within '
                '12 observations before at_landmark can arm an instruction turn; same-observation retries or '
                'switching among the four simultaneous camera views without base motion do not count. '
                'While second_changed_base_pose_needed is true, use observe or approaching with the shortest legal '
                'FRONT waypoint (0.25m-0.80m ahead in FRONT view, preserving your current forward heading; NEVER pick '
                'a BACK waypoint and do not pick a >1.0m waypoint that overshoots a nearby landmark) rather than repeating '
                'at_landmark, and if the landmark is visible '
                'now, ALWAYS populate landmark_identity, landmark_view, and landmark_candidate_id/landmark_bbox '
                '(never leave both candidate_id and bbox null when visible!) in turn_context so this frame is '
                'recorded as the first pose seed. Once at the second pose '
                '(second_changed_base_pose_needed is false) and within 1.5m of the landmark, use status="at_landmark" '
                'with the explicit turn action. If current base heading drifts '
                '>8deg from the measured incoming heading when at_landmark is requested, an internal '
                'landmark_reorientation turn restores incoming heading before you select the explicit turn.'
            )
        status_enum = (
            'approaching|observe|at_landmark'
            if (clause and clause.get('landmark'))
            else 'approaching|observe|at_landmark|execute_turn'
        )
        cid_example = (clause or {}).get('condition_id') or 'P1.C1'
        return ('\nACTIVE INSTRUCTION TURN (measured angles are RIGHT positive, LEFT negative): '
                + json.dumps(clause) + '\nPENDING FIRST DEPARTURE: ' + json.dumps(pending)
                + extra
                + '\nSmall waypoint alignment is not the instructed turn (measured_turn_witness=true with '
                'instruction_turn_step=null means ONLY a waypoint alignment occurred and the explicit instruction '
                'turn has NOT been executed yet!). Only when instruction_turn_step is an integer '
                '(instruction_turn_executed=true) was the explicit turn already executed; if instruction_turn_step '
                'is null, do NOT skip the turn—you MUST still localize the landmark and execute the explicit turn. '
                'If instruction_turn_step is already an integer, do not repeat it merely '
                'because route verification is pending. Before departure, choose a currently legal waypoint whose '
                'first path segment preserves the instructed heading; ordinary observation turns do not erase it. '
                'For the current turn clause, REQUIRED top-level turn_context: null when the landmark is not visible, or '
                '{"condition_id":"' + str(cid_example) + '","status":"' + status_enum + '",'
                '"landmark_identity":null,"landmark_view":null,"landmark_bbox":null,'
                '"landmark_candidate_id":null,"landmark_bbox_space":"full_view"}. '
                'Use at_landmark only when the turn location is reached in the CURRENT images, then select '
                'the explicit turn action in the clause direction (for a named landmark, never use status="execute_turn"). For a named landmark supply its own exact '
                'identity, its own landmark_candidate_id (e.g. eye chart ID separate from the bathroom door '
                'endpoint), view(front/left/right/back), and normalized [x1,y1,x2,y2] box in original '
                'full_view coordinates with landmark_bbox_space="full_view" (or omit landmark_bbox to use the '
                'candidate default full_view box); the endpoint object '
                'cannot substitute for this landmark. Visibility on the RIGHT alone does not establish arrival. '
                'For an immediate turn with no named landmark, use execute_turn to distinguish it from a look-around. '
                'When uncertain or still approaching, use observe/approaching (with landmark_candidate_id/bbox if visible, or null if not yet visible) and a legal action to '
                'localize/approach, not a claimed completed turn. All existing completion checks still apply.')

    def _measure_turn_landmark(self, context):
        if not isinstance(context, dict):
            return {'valid': False, 'reason': 'invalid_turn_context'}
        obs = getattr(self, 'curr_obs', {}) or {}
        visual = _visual_turn_enabled(self)
        view = context.get('landmark_view')
        bbox = context.get('landmark_bbox')
        bbox_space = context.get('landmark_bbox_space')
        candidate_id = context.get('landmark_candidate_id')
        resolved = None
        if visual:
            if bbox is not None and bbox_space != 'full_view':
                return {
                    'valid': False,
                    'reason': 'full_view_landmark_bbox_required',
                    'candidate_id': candidate_id,
                    'source': 'detector_mask' if candidate_id is not None else 'vlm_bbox',
                    'binding_source': 'candidate' if candidate_id is not None else 'free_bbox',
                    'selected_detector_class_name': None,
                }
            if bbox_space not in (None, 'full_view'):
                return {
                    'valid': False,
                    'reason': 'full_view_landmark_bbox_required',
                    'candidate_id': candidate_id,
                    'source': 'detector_mask' if candidate_id is not None else 'vlm_bbox',
                    'binding_source': 'candidate' if candidate_id is not None else 'free_bbox',
                    'selected_detector_class_name': None,
                }
            if candidate_id is not None:
                if not isinstance(candidate_id, str) or not candidate_id.strip():
                    return {
                        'valid': False,
                        'reason': 'invalid_landmark_candidate_id',
                        'candidate_id': candidate_id,
                        'source': 'detector_mask',
                        'binding_source': 'candidate',
                        'selected_detector_class_name': None,
                    }
                resolver = getattr(self, '_resolve_visual_region', None)
                if not callable(resolver):
                    return {
                        'valid': False,
                        'reason': 'visual_region_resolver_unavailable',
                        'candidate_id': candidate_id,
                        'source': 'detector_mask',
                        'binding_source': 'candidate',
                        'selected_detector_class_name': None,
                    }
                clause = self._active_turn_clause()
                expected_landmark = (clause or {}).get('landmark') or context.get('landmark_identity')
                resolved = resolver({
                    'candidate_id': candidate_id,
                    'endpoint_text': expected_landmark,
                    'evidence_role': 'primary',
                    'view': view,
                    'bbox': bbox,
                    'bbox_space': 'full_view',
                })
                if not isinstance(resolved, dict) or not resolved.get('valid'):
                    return {
                        'valid': False,
                        'reason': (resolved or {}).get('reason') or 'invalid_landmark_candidate',
                        'candidate_id': candidate_id,
                        'source': 'detector_mask',
                        'binding_source': 'candidate',
                        'selected_detector_class_name': (resolved or {}).get('selected_detector_class_name') if isinstance(resolved, dict) else None,
                    }
                if view is not None and resolved.get('view') not in (None, view):
                    return {
                        'valid': False,
                        'reason': 'candidate_view_mismatch',
                        'candidate_id': candidate_id,
                        'source': 'detector_mask',
                        'binding_source': 'candidate',
                        'selected_detector_class_name': resolved.get('selected_detector_class_name'),
                    }
                eff_view = resolved.get('view') or view
                if resolved.get('mask_view') not in (None, eff_view):
                    return {
                        'valid': False,
                        'reason': 'candidate_view_mismatch',
                        'candidate_id': candidate_id,
                        'source': 'detector_mask',
                        'binding_source': 'candidate',
                        'selected_detector_class_name': resolved.get('selected_detector_class_name'),
                    }
                curr_oid = getattr(getattr(self, 'mapper', None), 'observation_id', None)
                if curr_oid is not None and resolved.get('mask_observation_id') not in (None, curr_oid):
                    return {
                        'valid': False,
                        'reason': 'stale_candidate_id',
                        'candidate_id': candidate_id,
                        'source': 'detector_mask',
                        'binding_source': 'candidate',
                        'selected_detector_class_name': resolved.get('selected_detector_class_name'),
                    }
                if resolved.get('mask') is None:
                    return {
                        'valid': False,
                        'reason': 'candidate_mask_unavailable',
                        'candidate_id': candidate_id,
                        'source': 'detector_mask',
                        'binding_source': 'candidate',
                        'selected_detector_class_name': resolved.get('selected_detector_class_name'),
                    }
                conf = resolved.get('confidence')
                conf_valid = (
                    conf is not None
                    and not isinstance(conf, bool)
                    and isinstance(conf, (int, float, np.integer, np.floating))
                    and math.isfinite(float(conf))
                )
                if not conf_valid or float(conf) < _mask_min_confidence(self):
                    return {
                        'valid': False,
                        'reason': 'landmark_confidence_below_threshold',
                        'candidate_id': resolved.get('candidate_id', candidate_id),
                        'confidence': float(conf) if conf_valid else None,
                        'source': 'detector_mask',
                        'binding_source': 'candidate',
                        'selected_detector_class_name': resolved.get('selected_detector_class_name'),
                    }
                view = eff_view
                bbox = resolved.get('bbox') if resolved.get('bbox') is not None else bbox
            elif bbox is None:
                return {
                    'valid': False,
                    'reason': 'current_landmark_bbox_required',
                    'candidate_id': None,
                    'source': 'vlm_bbox',
                    'binding_source': 'free_bbox',
                    'selected_detector_class_name': None,
                }
        if view not in ('front', 'left', 'right', 'back'):
            return {'valid': False, 'reason': 'current_landmark_view_required'}
        packet_obs = obs if view == 'front' else obs.get(view, {})
        state = packet_obs.get('agent_state') if isinstance(packet_obs, dict) else None
        mapper = self.mapper
        if state is None or view not in ('front', 'left', 'right', 'back'):
            return {'valid': False, 'reason': 'current_landmark_view_required'}
        try:
            extra = {}
            if visual and resolved is not None:
                extra = {
                    'target_mask': resolved['mask'],
                    'mask_view': resolved.get('mask_view') or view,
                    'mask_observation_id': resolved.get('mask_observation_id'),
                    'mask_confidence': float(resolved['confidence']),
                    'target_id': resolved.get('entity_id'),
                }
            elif visual:
                extra = {'foreground_jamb': True}
            pkt = ground_target_depth(
                {name: (obs if name == 'front' else obs.get(name, {})).get('depth_sensor')
                 for name in ('front', 'left', 'right', 'back')},
                target_view=view,
                target_bbox=bbox,
                endpoint_kind='near_object',
                observation_id=getattr(mapper, 'observation_id', None),
                observation_step=getattr(mapper, 'observation_step', None),
                camera_intrinsic=mapper.camera_intrinsic,
                camera_rotation=mapper.rotation_func(state.sensor_states['color_sensor'].rotation),
                camera_position=mapper.translation_func(state.position) - mapper.initial_position,
                **extra,
            )
        except (AttributeError, TypeError, ValueError, KeyError):
            return {'valid': False, 'reason': 'current_landmark_depth_unavailable'}
        if visual and isinstance(pkt, dict):
            pkt = dict(pkt)
            for k in list(pkt):
                if isinstance(pkt[k], np.ndarray):
                    pkt.pop(k, None)
            dist = pkt.get('median_distance_m')
            if (dist is None
                    or isinstance(dist, bool)
                    or not isinstance(dist, (int, float, np.integer, np.floating))
                    or not math.isfinite(float(dist))
                    or float(dist) <= 0):
                pkt['valid'] = False
                pkt.setdefault('reason', 'current_landmark_depth_unavailable')
            pkt['candidate_id'] = resolved.get('candidate_id', candidate_id) if resolved is not None else None
            pkt['entity_id'] = resolved.get('entity_id') if resolved is not None else None
            if resolved is not None and resolved.get('entity_id') is not None:
                pkt['target_id'] = resolved.get('entity_id')
            pkt['selected_detector_class_name'] = resolved.get('selected_detector_class_name') if resolved is not None else None
            pkt.setdefault('source', 'detector_mask' if resolved is not None else 'vlm_bbox')
            pkt['binding_source'] = 'candidate' if resolved is not None else 'free_bbox'
            pkt['bbox_space'] = 'full_view'
        return pkt

    def _validate_turn_selection(self, parsed, selected):
        clause = self._active_turn_clause()
        if not clause:
            return selected
        visual = _visual_turn_enabled(self)
        entry = _clause_state_entry(self, clause) if visual and clause.get('landmark') else None
        if 'turn_context' not in parsed:
            raise ActionProtocolError('active turn clause requires turn_context (null if localisation is unknown); waypoint alignment is not the instructed turn')
        context = parsed.get('turn_context')
        if context is not None and (not isinstance(context, dict)
                or context.get('condition_id') != clause['condition_id']
                or context.get('status') not in ('approaching', 'observe', 'at_landmark', 'execute_turn')):
            raise ActionProtocolError('turn_context must be null or identify the active condition and localisation status')
        at_landmark = bool(context and context.get('status') in ('at_landmark', 'execute_turn'))
        evidence = None
        confirmed = False
        if visual and clause['landmark'] and isinstance(context, dict):
            if landmark_identity_matches(clause['landmark'], context.get('landmark_identity')):
                cid_raw = context.get('landmark_candidate_id')
                bare_cid = isinstance(cid_raw, (int, str)) and ':' not in str(cid_raw)
                if context.get('landmark_bbox_space') is None and context.get('landmark_bbox') is not None:
                    context = {**context, 'landmark_bbox_space': 'full_view'}
                if ((cid_raw is None and context.get('landmark_bbox') is None) or bare_cid) and hasattr(self, '_enriched_current_candidates'):
                    try:
                        enriched, _, _, _, _ = self._enriched_current_candidates()
                        req_view = context.get('landmark_view')
                        all_turn_cands = [
                            r for r in (enriched or {}).values()
                            if isinstance(r, dict) and 'turn' in (r.get('roles') or ())
                        ]
                        scored_cands = []
                        prior_samples = list((entry or {}).get('samples') or [])
                        max_cand_dist = 2.2 if context.get('status') in ('approaching', 'observe') else 1.85
                        for cand in all_turn_cands:
                            test_ctx = {
                                **context,
                                'landmark_candidate_id': cand.get('candidate_id'),
                                'landmark_view': cand.get('view') or req_view,
                                'landmark_bbox': cand.get('bbox'),
                                'landmark_bbox_space': 'full_view',
                            }
                            ev = self._measure_turn_landmark(test_ctx)
                            dist = ev.get('median_distance_m')
                            valid_near = bool(ev.get('valid') and dist is not None and 0 < float(dist) <= max_cand_dist)
                            matches_prior = False
                            ev_center = _valid_vec3(ev.get('world_center'))
                            if ev.get('valid') and ev_center is not None and prior_samples:
                                for prev in prior_samples:
                                    prev_center = _valid_vec3(prev.get('world_center'))
                                    if prev_center is not None and float(np.linalg.norm(ev_center - prev_center)) <= 0.65:
                                        matches_prior = True
                                        break
                            scored_cands.append((
                                (1 if valid_near else 0,
                                 1 if matches_prior else 0,
                                 1 if cand.get('view') == req_view else 0,
                                 1 if ev.get('valid') else 0,
                                 float(cand.get('confidence') or 0.0)),
                                test_ctx,
                                ev,
                            ))
                        if scored_cands:
                            scored_cands.sort(key=lambda item: item[0], reverse=True)
                            best_score, best_ctx, _ = scored_cands[0]
                            if best_score[3] > 0 or not (bare_cid and context.get('landmark_bbox') is not None):
                                context = best_ctx
                            else:
                                context = {
                                    **context,
                                    'landmark_candidate_id': None,
                                    'landmark_bbox_space': 'full_view',
                                }
                    except Exception:
                        pass
                if bare_cid and context.get('landmark_candidate_id') == cid_raw and context.get('landmark_bbox') is not None:
                    context = {
                        **context,
                        'landmark_candidate_id': None,
                        'landmark_bbox_space': 'full_view',
                    }
        if visual and clause['landmark'] and isinstance(context, dict) and not at_landmark:
            if any(context.get(k) is not None for k in ('landmark_identity', 'landmark_candidate_id', 'landmark_bbox')):
                if landmark_identity_matches(clause['landmark'], context.get('landmark_identity')):
                    if context.get('landmark_candidate_id') is not None or context.get('landmark_bbox') is not None:
                        ev = self._measure_turn_landmark(context)
                        _update_landmark_confirmation(self, entry, context, ev)
                        ev_dist = ev.get('median_distance_m') if isinstance(ev, dict) else None
                        if (isinstance(ev, dict) and ev.get('valid')
                                and ev_dist is not None and 0 < float(ev_dist) <= 1.85
                                and selected.get('type') == 'waypoint'):
                            wp_view = selected.get('view')
                            wp_len = float(selected.get('route_length_m') or 0.0)
                            cur_actions = getattr(self, 'last_actions', None) or {}
                            has_short_front_wp = (not cur_actions) or any(
                                isinstance(v, dict)
                                and v.get('type') == 'waypoint'
                                and not v.get('rejected')
                                and v.get('view') == 'front'
                                and 0 < float(v.get('route_length_m') or 0.0) <= 1.0
                                for v in cur_actions.values()
                            )
                            if has_short_front_wp and wp_view in ('left', 'right', 'back') and context.get('landmark_view') in ('front', 'left', 'right', 'back'):
                                raise ActionProtocolError(
                                    'when turn landmark is already localized within 1.85m, '
                                    'choose a short FRONT waypoint (0.25m-0.80m, such as WP_01) to preserve '
                                    'your incoming corridor heading rather than pre-rotating to a LEFT/RIGHT/BACK waypoint'
                                )
                            if has_short_front_wp and wp_view == 'front' and wp_len > 1.0:
                                raise ActionProtocolError(
                                    'when turn landmark is already localized within 1.85m, choose a shorter '
                                    'FRONT waypoint (<= 1.0m, such as WP_01) to obtain the second camera pose '
                                    'without overshooting the landmark'
                                )
                    elif context.get('landmark_view') in ('front', 'left', 'right', 'back'):
                        raise ActionProtocolError(
                            'when landmark_identity and landmark_view are provided in turn_context, '
                            'you must also provide a non-null landmark_bbox [x1,y1,x2,y2] (with '
                            'landmark_bbox_space="full_view") or landmark_candidate_id so the pose seed can be measured'
                        )
                else:
                    entry['confirmed_pair'] = None
        if at_landmark and clause['landmark']:
            if not landmark_identity_matches(clause['landmark'], context.get('landmark_identity')):
                if entry is not None:
                    entry['confirmed_pair'] = None
                raise ActionProtocolError('turn landmark identity must match the active clause, not the endpoint object')
            evidence = self._measure_turn_landmark(context)
            # Median is deliberately used rather than a single close pixel.
            distance = evidence.get('median_distance_m')
            if not evidence.get('valid') or distance is None or not 0 < distance <= 1.85:
                if entry is not None:
                    entry['confirmed_pair'] = None
                raise ActionProtocolError('turn landmark is not localized nearby in this frame; observe/approach before claiming at_landmark')
            if visual:
                confirmed = _update_landmark_confirmation(self, entry, context, evidence)
        if at_landmark and getattr(self, 'rolling_back', False):
            raise ActionProtocolError('cannot arm or execute instruction turn while rolling_back is active')
        expected = {'type': 'turn', 'direction': clause['direction']}
        matches = selected.get('requested_action') == expected
        performed = self._executed_instruction_turn(clause)
        if at_landmark and performed is not None:
            if selected.get('type') == 'turn':
                raise ActionProtocolError('instruction turn already executed; choose a legal departure waypoint or use observe for a look-around, do not repeat the turn')
            if selected.get('type') == 'waypoint':
                route = self._validated_waypoint_route(selected['target_point_world'])
                if route is None or not self._turn_route_allowed(route['path']):
                    raise ActionProtocolError('instruction turn already executed; departure must use a legal route preserving its first heading')
            # A completion proposal still goes through the unchanged verifier;
            # a turn-and-stand clause need not invent an extra forward move.
            return selected
        if visual and at_landmark and clause['landmark'] and not confirmed:
            raise ActionProtocolError('named turn landmark requires two distinct changed-pose observations before at_landmark can arm the turn; use observe or approaching first')
        if at_landmark and not matches:
            raise ActionProtocolError('at_landmark requires the current explicit ' + clause['direction'] + ' turn action, not waypoint alignment')
        if visual and at_landmark and clause['landmark'] and matches:
            inc = entry.get('incoming_heading_rad')
            curr = _current_base_heading(self)
            if inc is None or curr is None:
                raise ActionProtocolError('incoming heading is unknown; observe or approach with a valid base heading before executing instruction turn')
            delta = _wrap_yaw(inc - curr)
            drift_deg = abs(math.degrees(delta))
            if drift_deg > 8.0:
                if int(entry.get('reorientation_count', 0)) >= 2:
                    raise ActionProtocolError('incoming heading restore bound exceeded (max 2); use observe or approaching rather than executing an unaligned turn')
                return {
                    'type': 'turn',
                    'angle': float(-delta),
                    'requested_action': {
                        'type': 'landmark_reorientation',
                        'condition_id': clause['condition_id'],
                        'subtask_id': entry['subtask_id'],
                        'purpose': 'restore_incoming_heading',
                    },
                }
        # Observation/correction turns are not silently promoted to route intent.
        instruction_turn = matches and at_landmark and not getattr(self, 'rolling_back', False)
        if instruction_turn and not self._current_turn_departure():
            selected = dict(selected, instruction_turn={**clause, 'landmark_evidence': evidence})
        return selected

    def _executed_instruction_turn(self, clause):
        """Find this clause's explicit physical turn, not an arbitrary look.

        Before departure, observation rotations may temporarily change heading;
        the pending route guard restores the instructed heading. Afterwards,
        require the accepted physical departure and the complete rotation block.
        A recovery invalidates this action-selection memory, without changing
        any saved verifier condition or promoting it to semantic route proof.
        """
        rows = getattr(self, '_execution_ledger', [])
        pending = self._current_turn_departure()
        for row in reversed(rows):
            command = row.get('instruction_turn') or {}
            if (any(command.get(k) != clause[k] for k in ('subtask_id', 'condition_id', 'text', 'direction'))
                    or row.get('recovery') or row.get('rejected_by_rollback')
                    or row.get('requested_action') != {'type': 'turn', 'direction': clause['direction']}):
                continue
            tail = [r for r in rows if r.get('step', -1) > row.get('step', -1)]
            if any(r.get('recovery') for r in tail):
                continue
            if not turn_witness(clause['text'], [row], [row])[0]:
                continue
            if (pending and pending['step'] == row['step']
                    and pending['condition_id'] == clause['condition_id']
                    and pending['subtask_id'] == clause['subtask_id']):
                return row
            departed = any((r.get('turn_departure') or {}).get('accepted') is True
                          and r['turn_departure'].get('condition_id') == clause['condition_id']
                          and r['turn_departure'].get('turn_step') == row['step']
                          and r.get('action_type') == 'move_forward'
                          and heading_matches(row['yaw_after'], first_segment_heading([r.get('pose')], r.get('pose_before')))
                          and not r.get('rejected_by_rollback') for r in tail)
            if departed and turn_witness(clause['text'], [row], rows)[0]:
                return row
        return None

    def _current_turn_departure(self):
        pending = getattr(self, '_turn_departure', None)
        if pending:
            key = self.instruction_obj.get_current_subtask_key()
            rows = getattr(self, '_execution_ledger', [])
            source = next((r for r in rows if r.get('step') == pending['step']), None)
            if (getattr(self, 'rolling_back', False) or key != pending['subtask_id']
                    or source is None or source.get('rejected_by_rollback') or source.get('recovery')
                    or any(r.get('recovery') for r in rows if r.get('step', -1) > pending['step'])):
                self._turn_departure = pending = None
        return pending

    def _turn_route_allowed(self, path):
        pending = self._current_turn_departure()
        if not pending:
            return True
        return heading_matches(pending['heading'], first_segment_heading(path, self.mapper.current_position))

    def _active_unwitnessed_straight_clause(self):
        if not hasattr(self, 'instruction_obj') or not self.instruction_obj:
            return None
        try:
            key, st = self.instruction_obj.get_current_subtask()
            st = st or {}
            checks = completion_conditions(st.get('instruction', ''), key or 'P1')
            records = [r for r in st.get('record', []) if isinstance(r, dict)
                       and not r.get('recovery') and not r.get('rejected_by_rollback')]
            saved = {c['id']: c for c in st.get('route_progress', {}).get('conditions', [])}
            if saved:
                checks, _ = validate_condition_checks(
                    checks, checks, records, previous=saved,
                    require_instruction_turn=bool(getattr(self, 'navigation_cfg', {}).get('habitat_control_enabled', False)),
                    instruction_subtask_id=key)
            after_step = -1
            for condition in checks:
                if condition.get('status') == 'satisfied' and condition.get('execution_checked'):
                    after_step = max([after_step] + condition.get('steps', []))
                    continue
                if requires_straight(condition.get('text', '')) and not condition.get('requires_rotation'):
                    current_records = [r for r in records if r.get('step', -1) > after_step]
                    if not current_records:
                        return {'subtask_id': key, 'condition_id': condition['id'], 'after_step': after_step}
                return None
        except Exception:
            return None
        return None

    def _filter_turn_departure_actions(self, actions):
        if self._current_turn_departure():
            return {name: action for name, action in actions.items()
                    if action.get('type') != 'waypoint'
                    or self._validated_waypoint_route(action['target_point_world']) is not None}
        straight_clause = self._active_unwitnessed_straight_clause()
        if straight_clause and not getattr(self, 'rolling_back', False):
            curr_heading = _current_base_heading(self)
            origin = getattr(getattr(self, 'mapper', None), 'current_position', None)
            if curr_heading is not None and origin is not None:
                origin_xy = np.asarray(origin, dtype=float)[:2]
                straight_wps = {}
                max_dev = math.radians(18.0)
                for name, action in actions.items():
                    if not isinstance(action, dict) or action.get('type') != 'waypoint':
                        continue
                    target = action.get('target_point_world')
                    if target is None:
                        continue
                    t_xy = np.asarray(target, dtype=float)[:2]
                    delta = t_xy - origin_xy
                    dist = float(np.linalg.norm(delta))
                    if not math.isfinite(dist) or dist < 0.35:
                        continue
                    overall_h = float(math.atan2(delta[1], delta[0]))
                    if abs(_wrap_yaw(overall_h - curr_heading)) > max_dev:
                        continue
                    route = self._validated_waypoint_route(target)
                    if route is None:
                        continue
                    first_h = first_segment_heading(route['path'], origin)
                    if first_h is not None and abs(_wrap_yaw(first_h - curr_heading)) <= max_dev:
                        straight_wps[name] = action
                if straight_wps:
                    return {
                        name: action for name, action in actions.items()
                        if action.get('type') != 'waypoint' or name in straight_wps
                    }
        return actions

    def _observe_turn_execution(self, record):
        command = getattr(self, '_instruction_turn_command', None)
        self._instruction_turn_command = None
        if record.get('recovery'):
            self._turn_departure = None
            return
        if (command and record.get('action_type') == 'turn'
                and record.get('requested_action') == {'type': 'turn', 'direction': command['direction']}):
            matched, _ = turn_witness(command['text'], [record], [record])
            if matched:
                self._turn_departure = {'subtask_id': command['subtask_id'], 'condition_id': command['condition_id'],
                                        'direction': command['direction'], 'heading': float(record['yaw_after']),
                                        'step': record['step']}
                record['instruction_turn'] = command
                self.navigation_trace.append({'role': 'controller', 'event': 'instruction_turn_executed',
                                               **self._turn_departure})
        pending = self._current_turn_departure()
        if pending and record.get('action_type') == 'move_forward':
            heading = first_segment_heading([record.get('pose')], record.get('pose_before'))
            if heading is not None:
                accepted = heading_matches(pending['heading'], heading)
                record['turn_departure'] = {'accepted': accepted, 'condition_id': pending['condition_id'],
                                            'turn_step': pending['step']}
                self.navigation_trace.append({'role': 'controller', 'event': 'instruction_turn_departure',
                                               'step': record['step'], **record['turn_departure']})
                self._turn_departure = None
                if not accepted:
                    self._invalidate_blocked_path('actual_departure_opposes_instructed_turn')


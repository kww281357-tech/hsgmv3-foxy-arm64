"""Typed actions and measured waypoint/stair progress, without simulator or GT.

Planner prose is advisory. These contracts consume only legal candidates,
source instruction clauses, and the executor's measured poses.
"""
import json
import re

import numpy as np


class ActionProtocolError(ValueError):
    pass


TURN_IDS = {'LEFT': 'TURN_LEFT', 'RIGHT': 'TURN_RIGHT', 'BACK': 'TURN_BACK'}


def legal_action_objects(actions, completion=True):
    result = []
    for name, candidate in actions.items():
        if candidate.get('rejected'):
            continue
        if candidate.get('type') == 'waypoint' and re.fullmatch(r'WP_\d{2,}', str(name)):
            result.append({'type': 'navigate', 'waypoint_id': name})
        elif candidate.get('type') == 'turn' and name in TURN_IDS.values():
            result.append({'type': 'turn', 'direction': (name[5:] if name.startswith('TURN_') else name)})
    if completion:
        result.append({'type': 'propose_completion'})
    return result


def decode_action(value, actions, completion=True):
    """No bare IDs, fuzzy parsing, arbitrary angles, coordinates or STOP aliases."""
    if not isinstance(value, dict):
        raise ActionProtocolError('action must be an object; bare B/L/R/numbers are invalid')
    kind = value.get('type')
    if kind == 'propose_completion' and completion and set(value) == {'type'}:
        return {'type': 'completion_proposal'}
    if kind == 'navigate' and set(value) == {'type', 'waypoint_id'}:
        name = value['waypoint_id']
        if not isinstance(name, str) or not re.fullmatch(r'WP_\d{2,}', name):
            raise ActionProtocolError('navigate requires an exact WP_01-style waypoint_id')
        expected = 'waypoint'
    elif kind == 'turn' and set(value) == {'type', 'direction'}:
        name = TURN_IDS.get(value['direction']) if isinstance(value['direction'], str) else None
        expected = 'turn'
    else:
        raise ActionProtocolError('use exactly one legal navigate/turn/propose_completion object')
    candidate = actions.get(name)
    if not candidate or candidate.get('type') != expected or candidate.get('rejected'):
        raise ActionProtocolError('action is absent from the current candidate list')
    return dict(candidate, action_id=name, requested_action=dict(value))


def normalize_action_object(parsed_json):
    """Normalize unambiguous shorthand action objects without accepting legacy B/L/R/STOP."""
    if not isinstance(parsed_json, dict):
        return parsed_json
    if 'action' not in parsed_json:
        out = parsed_json.get('output')
        if isinstance(out, dict) and 'action' in out:
            parsed_json['action'] = out['action']
        elif isinstance(parsed_json.get('navigate'), str) and re.fullmatch(r'WP_\d{2,}', parsed_json['navigate'].strip()):
            parsed_json['action'] = {'type': 'navigate', 'waypoint_id': parsed_json['navigate'].strip()}
        elif isinstance(parsed_json.get('waypoint_id'), str) and re.fullmatch(r'WP_\d{2,}', parsed_json['waypoint_id'].strip()):
            parsed_json['action'] = {'type': 'navigate', 'waypoint_id': parsed_json['waypoint_id'].strip()}
        elif isinstance(parsed_json.get('turn'), str) and (parsed_json['turn'].strip().upper()[5:] if parsed_json['turn'].strip().upper().startswith('TURN_') else parsed_json['turn'].strip().upper()) in TURN_IDS:
            parsed_json['action'] = {'type': 'turn', 'direction': (parsed_json['turn'].strip().upper()[5:] if parsed_json['turn'].strip().upper().startswith('TURN_') else parsed_json['turn'].strip().upper())}
    act = parsed_json.get('action')
    if isinstance(act, str):
        s = act.strip()
        if re.fullmatch(r'WP_\d{2,}', s):
            parsed_json['action'] = {'type': 'navigate', 'waypoint_id': s}
        elif s.upper() in TURN_IDS.values():
            parsed_json['action'] = {'type': 'turn', 'direction': (s.upper()[5:] if s.upper().startswith('TURN_') else s.upper())}
        elif s.lower() == 'propose_completion':
            parsed_json['action'] = {'type': 'propose_completion'}
    elif isinstance(act, dict):
        kind = act.get('type')
        if kind is None:
            wp = act.get('navigate') or act.get('waypoint_id') or act.get('waypoint')
            turn_dir = act.get('turn') or act.get('direction')
            if isinstance(wp, str) and re.fullmatch(r'WP_\d{2,}', wp.strip()):
                parsed_json['action'] = {'type': 'navigate', 'waypoint_id': wp.strip()}
            elif isinstance(turn_dir, str) and (turn_dir.strip().upper()[5:] if turn_dir.strip().upper().startswith('TURN_') else turn_dir.strip().upper()) in TURN_IDS:
                parsed_json['action'] = {'type': 'turn', 'direction': (turn_dir.strip().upper()[5:] if turn_dir.strip().upper().startswith('TURN_') else turn_dir.strip().upper())}
            elif act.get('propose_completion') is True:
                parsed_json['action'] = {'type': 'propose_completion'}
        elif kind == 'navigate':
            wp = act.get('waypoint_id') or act.get('waypoint') or act.get('navigate')
            if isinstance(wp, str) and re.fullmatch(r'WP_\d{2,}', wp.strip()) and 'target' not in act:
                parsed_json['action'] = {'type': 'navigate', 'waypoint_id': wp.strip()}
        elif kind == 'turn':
            d = act.get('direction') or act.get('turn')
            if isinstance(d, str) and (d.strip().upper()[5:] if d.strip().upper().startswith('TURN_') else d.strip().upper()) in TURN_IDS and 'angle' not in act:
                parsed_json['action'] = {'type': 'turn', 'direction': (d.strip().upper()[5:] if d.strip().upper().startswith('TURN_') else d.strip().upper())}
        elif kind in ('propose_completion', 'completion_proposal') and 'completed' not in act:
            parsed_json['action'] = {'type': 'propose_completion'}
    return parsed_json


def action_prompt(actions, completion=True):
    return (
        'Action protocol v3: return one JSON object containing "action" as one of the objects below. '
        'WP_ IDs are movement destinations. TURN_LEFT/RIGHT/BACK are in-place rotations only, '
        'never forward motion or doorway names. Reversing a previous turn is legal. '
        'navigate lets the controller align and follow the route without further model decisions. '
        'propose_completion requests verification; it does not complete the task. '
        'Do not output bare numbers, letters, STOP, angles or target coordinates. '
        'Do not report imagined actions as executed. "completed" and explanatory prose have no control authority. '
        'Current legal action objects: ' + json.dumps(legal_action_objects(actions, completion), ensure_ascii=False)
    )


class WaypointExecution:
    def __init__(self, target, action_id='', max_ticks=48, max_alignment_turns=8):
        self.target = np.asarray(target, dtype=float).tolist()
        self.action_id = action_id
        self.phase = 'aligning'
        self.ticks = self.blocked_forward = self.alignment_turns = 0
        self.distance = 0.0
        self.max_ticks = max_ticks
        self.max_alignment_turns = max_alignment_turns
        self.reason = ''

    def observe(self, record):
        self.ticks += 1
        distance = float(record['motion_distance'])
        self.distance += distance
        pose = np.asarray(record.get('pose', []), dtype=float)
        if (pose.shape == (3,) and np.linalg.norm(pose[:2]-np.asarray(self.target)[:2]) <= .08
                and abs(pose[2]-self.target[2]) <= .25):
            self.phase = 'arrived'
        if distance >= .02:
            self.alignment_turns = 0
        if record['action_type'] == 'move_forward':
            is_blocked = (distance < .02 and (float(record.get('requested_distance', 0)) >= .04 or record.get('action_type') == 'move_forward'))
            self.blocked_forward = self.blocked_forward + 1 if is_blocked else 0
        elif record['action_type'] == 'turn':
            self.alignment_turns += 1
        reason = None
        if self.phase == 'arrived':
            return None
        if self.blocked_forward >= 2:
            reason = 'two_forward_commands_without_motion'
        elif self.alignment_turns >= self.max_alignment_turns:
            reason = 'waypoint_alignment_without_motion'
        elif self.ticks >= self.max_ticks and self.phase != 'arrived':
            reason = 'waypoint_execution_budget_exhausted'
        if reason:
            self.phase, self.reason = 'blocked', reason
        return reason

    def packet(self):
        return {'action_id': self.action_id, 'phase': self.phase, 'target': self.target,
                'ticks': self.ticks, 'measured_path_m': round(self.distance, 3), 'reason': self.reason}


_DETERMINERS = r'(?:(?:the|a|an|one|two|three|first|second|third|next|another|single|set|flight|of|this|these|those)\s+){0,7}'
_UP = re.compile(r'\b(?:(?:go|walk|head|move|continue|proceed|climb)\s+upstairs|ascend(?:ing)?|up\s+' + _DETERMINERS + r'(?:stairs?|steps?|staircase|flight))\b', re.I)
_DOWN = re.compile(r'\b(?:(?:go|walk|head|move|continue|proceed|climb)\s+downstairs|descend(?:ing)?|down\s+' + _DETERMINERS + r'(?:stairs?|steps?|staircase|flight))\b', re.I)


def stair_intent(text):
    text = str(text or '')
    up, down = bool(_UP.search(text)), bool(_DOWN.search(text))
    if not up and not down and re.search(r'\bclimb\s+(?:the\s+)?(?:stairs|steps|staircase)\b', text, re.I):
        up = True
    direction = 1 if up and not down else -1 if down and not up else 0
    counted = bool(re.search(r'\b(?:up|down|ascend(?:ing)?|descend(?:ing)?|climb(?:ing)?)\s+(?:the\s+)?(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+steps?\b', text, re.I))
    return {'direction': direction, 'ambiguous': up and down, 'landing_required': bool(direction and not counted)}


def stair_condition_intents(conditions):
    """Bind a directionless stair traversal to its stated landing endpoint.

    'Take the stairs ... stop at the bottom' names the descent through two
    clauses. A bare arrival at a landing does not request a new stair flight.
    Explicit directions keep their own meaning; inference never crosses the
    next stair traversal or resolves conflicting top/bottom endpoints.
    """
    traversal = re.compile(r'\b(?:take|use|traverse)\s+' + _DETERMINERS + r'(?:stairs?|steps?|staircase)\b', re.I)
    endpoint = re.compile(r'\b(?:stop|wait|halt|stand)\b.{0,70}?\b(?:at|on|near)\s+(?:the\s+)?'
                          r'(bottom|foot|base|top)\s+of\s+(?:the\s+)?(?:stairs?|steps?|staircase)\b', re.I)
    result = []
    for index, condition in enumerate(conditions):
        intent = stair_intent(condition.get('text', ''))
        if not intent['direction'] and not intent['ambiguous'] and traversal.search(condition.get('text', '')):
            hints = []
            for following in conditions[index:]:
                text = following.get('text', '')
                other = stair_intent(text)
                if following is not condition and (traversal.search(text) or other['direction'] or other['ambiguous']):
                    break
                for match in endpoint.finditer(text):
                    hints.append((1 if match.group(1).lower() == 'top' else -1, following['id']))
            directions = {direction for direction, _ in hints}
            if directions:
                intent = {'direction': next(iter(directions)) if len(directions) == 1 else 0,
                          'ambiguous': len(directions) != 1, 'landing_required': True,
                          'direction_source': 'stated_landing_endpoint',
                          'endpoint_condition_ids': [cid for _, cid in hints]}
        if intent['direction'] or intent['ambiguous']:
            result.append((condition, intent))
    return result


def pending_stair_condition(conditions):
    stairs = stair_condition_intents(conditions)
    if not stairs:
        return None
    return next(((c, intent) for c, intent in stairs
                 if c.get('status') != 'satisfied' or not c.get('execution_checked')), stairs[-1])


def check_stair_witnesses(conditions, ledger):
    """A forward action on the wrong vertical direction cannot prove stairs."""
    by_step = {r.get('step'): r for r in ledger if not r.get('recovery') and not r.get('rejected_by_rollback')}
    checked = []
    intents = {condition['id']: intent for condition, intent in stair_condition_intents(conditions)}
    for original in conditions:
        row = dict(original)
        intent = intents.get(row.get('id'), stair_intent(row.get('text', '')))
        if row.get('status') == 'satisfied' and (intent['direction'] or intent['ambiguous']):
            moves = [by_step[s] for s in row.get('steps', []) if s in by_step
                     and by_step[s].get('action_type') == 'move_forward']
            signed_height = sum(intent['direction']*float(r.get('height_delta', 0.)) for r in moves)
            if signed_height < .08 and not row.get('steps') and ledger:
                all_forward = [r for r in ledger if isinstance(r, dict) and r.get('action_type') == 'move_forward'
                               and not r.get('recovery') and not r.get('rejected_by_rollback')]
                ledger_signed = sum(intent['direction']*float(r.get('height_delta', 0.)) for r in all_forward)
                if ledger_signed >= .08:
                    signed_height = ledger_signed
            if intent['ambiguous'] or signed_height < .08:
                row.update(status='unknown', execution_checked=False,
                           evidence='Cited actions do not establish the source stair direction; separate opposing flights.')
        checked.append(row)
    return checked


def doorway_crossing_required(text):
    modifiers = r'(?:(?:the|a|an|this|that|first|second|third|left|right|open|double|single|wooden)\s+){0,6}'
    return bool(re.search(r'\b(?:through|into|out(?:\s+of)?)\s+' + modifiers + r'(?:doors?|doorway|entrance)\b|'
                         r'\b(?:enter|exit|cross)\s+' + modifiers + r'(?:doors?|doorway|entrance)\b', str(text), re.I))


def doorway_witnesses(conditions, reports, ledger):
    """Require measured crossing-action references plus explicit visual sides.

    This validates action/pose provenance, not a geometric doorway detector.
    The verifier must still identify the correct doorway and side in RGB.
    """
    by_id = {r.get('condition_id'): r for r in reports if isinstance(r, dict)} if isinstance(reports, list) else {}
    by_step = {r.get('step'): r for r in ledger if not r.get('recovery') and not r.get('rejected_by_rollback')}
    result = []
    for condition in conditions:
        if not doorway_crossing_required(condition.get('text', '')):
            continue
        report = by_id.get(condition['id'], {})
        steps = report.get('steps') or []
        valid = bool(isinstance(steps, list) and steps and len(steps) <= len(by_step)
                     and all(type(s) is int and s in by_step for s in steps)
                     and len(set(steps)) == len(steps) and steps == sorted(steps)
                     and str(report.get('entry_side') or '').strip()
                     and str(report.get('exit_side') or '').strip()
                     and str(report.get('visual_evidence') or '').strip())
        distance = 0.
        if valid:
            for step in steps:
                row = by_step[step]
                if row.get('action_type') != 'move_forward':
                    continue
                before, after = np.asarray(row.get('pose_before', [])), np.asarray(row.get('pose', []))
                if before.shape == (3,) and after.shape == (3,) and np.isfinite([before, after]).all():
                    distance += float(np.linalg.norm(after[:2]-before[:2]))
        result.append({'condition_id': condition['id'], 'execution_witnessed': valid and distance >= .10,
                       'steps': steps if valid else [], 'measured_translation_m': round(distance, 3)})
    return result


def landing_support(points, ground, radius=.6, minimum_area=.30):
    """Occupied cells near foot height, not a convex hull over unknown gaps."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    ground = np.asarray(ground, dtype=float)
    points = points[np.isfinite(points).all(axis=1)]
    delta = points - ground
    near = points[(np.linalg.norm(delta[:, :2], axis=1) <= radius) & (np.abs(delta[:, 2]) <= .06)]
    if not len(near):
        return False
    cells = np.floor((near[:, :2]-ground[:2])/.1 + 1e-6).astype(int)
    if len(np.unique(cells, axis=0))*.01 < minimum_area:
        return False
    design = np.column_stack((near[:, :2]-ground[:2], np.ones(len(near))))
    plane, _, rank, _ = np.linalg.lstsq(design, near[:, 2]-ground[2], rcond=None)
    return bool(rank == 3 and np.linalg.norm(plane[:2]) <= np.tan(np.deg2rad(5)))


class StairProgress:
    def __init__(self, subtask_id, condition_id, intent, height, config=None):
        self.subtask_id, self.condition_id = subtask_id, condition_id
        self.direction = intent['direction']
        self.ambiguous = intent['ambiguous']
        self.landing_required = intent['landing_required']
        self.direction_source = intent.get('direction_source', 'source_clause')
        self.endpoint_condition_ids = list(intent.get('endpoint_condition_ids', []))
        self.start_height = self.last_height = float(height)
        self.phase = 'approach' if self.direction else 'ambiguous' if self.ambiguous else 'not_applicable'
        self.stable_samples = self.opposite_moves = 0
        self.violation = False
        self.handoff_steps = []
        self.cfg = config or {}

    def route_allowed(self, path):
        points = np.asarray(path, dtype=float)
        if len(points) < 2:
            return False
        if not self.direction:
            if self.ambiguous:
                return True
            flat_max_dz = float(self.cfg.get('flat_route_max_dz_m', .45))
            return bool(np.abs(points[:, 2] - points[0, 2]).max() <= flat_max_dz)
        signed = self.direction*(points[:, 2]-points[0, 2])
        # Flat approach and switchback XY turns remain legal. Only a measured
        # route excursion in the wrong vertical direction is filtered.
        return bool(signed.min() >= -float(self.cfg.get('stair_reverse_tolerance_m', .15)))

    def observe(self, record, platform_supported):
        if not self.direction:
            return None
        height = float(record['pose'][2])
        dz = float(record['height_delta'])
        signed = self.direction*dz
        progress = self.direction*(height-self.start_height)
        incident = None
        # Once the stair climb/descent has already reached the landing with substantial
        # elevation change (>=0.40m), minor floor/carpet height variations (|dz| <= 0.12m)
        # while walking across the upper/lower room are not stair reversal violations.
        opp_thresh = -0.12 if (self.phase == 'landing' and progress >= 0.40) else -0.03
        if record['action_type'] == 'move_forward' and signed < opp_thresh:
            self.opposite_moves += 1
            if self.opposite_moves >= 2:
                self.violation, self.phase = True, 'reobserve'
                incident = 'stair_motion_opposes_current_clause'
        elif signed >= .03:
            self.opposite_moves = 0
            self.violation = False
        elif record['action_type'] == 'move_forward':
            self.opposite_moves = 0
        if (self.phase != 'landing' and progress >= float(self.cfg.get('stair_enter_height_m', .12)) and not self.violation):
            self.phase = 'traverse'
        stable = abs(height-self.last_height) <= .04
        self.stable_samples = self.stable_samples + 1 if stable and platform_supported else 0
        # Landing is current support, not a permanent milestone. A new height
        # change or lost support must earn a fresh stable landing sequence.
        if self.phase == 'landing' and not (stable and platform_supported):
            self.phase = ('traverse' if progress >= float(self.cfg.get('stair_enter_height_m', .12))
                          else 'approach')
        if (progress >= float(self.cfg.get('stair_min_progress_m', .25))
                and self.stable_samples >= int(self.cfg.get('stair_landing_observations', 3))):
            self.violation = False
            self.phase = 'landing'
        self.last_height = height
        return incident

    def completion_ready(self):
        if self.ambiguous:
            return False
        if not self.direction:
            return True
        if self.violation:
            return False
        if self.landing_required:
            if self.phase == 'landing':
                return True
            return False
        return self.direction*(self.last_height-self.start_height) >= .08

    def packet(self):
        return {'condition_id': self.condition_id, 'phase': self.phase,
                'direction': 'UP' if self.direction == 1 else 'DOWN' if self.direction == -1 else 'UNRESOLVED' if self.ambiguous else 'NONE',
                'measured_height_change_m': round(self.last_height-self.start_height, 3),
                'stable_supported_observations': self.stable_samples,
                'landing_required': self.landing_required, 'direction_violation': self.violation,
                'direction_source': self.direction_source, 'endpoint_condition_ids': self.endpoint_condition_ids,
                'handoff_steps': list(self.handoff_steps)}


"""Small, source-grounded contracts for route execution (no evaluator inputs)."""

import re

import numpy as np

from route_progress import instruction_turn_witness, measured_condition_witness, turn_witness, turn_witness_steps


STATES = ('satisfied', 'contradicted', 'unknown')
_ACTION = re.compile(
    r"\b(?:take|make)\s+(?:(?:the|a|another)\s+)?(?:(?:first|second|third|next)\s+)?"
    r"(?:(?:left|right)(?:\s+turn)?|turn(?:\s+on\s+(?:your|the)\s+(?:left|right))?)\b|"
    r"\b(?:turn|walk|go(?!\s+(?:stand|wait|stop|halt)\b)|proceed|enter|exit|pass|climb|continue|cross|leave|"
    r"move|approach|follow|travel|navigate|reach|run|advance|stroll|stride|"
    r"stop|wait|halt|stand|rotate|pivot)\b|"
    r"\bhead(?=\s+(?:towards?|for|down|up|straight|left|right|out|into)\b)|"
    r"\bstep(?=\s+(?:forward|back|backwards?|inside|outside|into|through|out|down|up)\b)|"
    r"\bface(?=\s+(?:the|a|left|right|towards?)\b)",
    re.I,
)
_MOVE = re.compile(r'\b(walk|go(?!\s+(?:stand|wait|stop|halt)\b)|proceed|enter|exit|pass|climb|continue|cross|leave|move|approach|follow|travel|navigate|reach|run|advance|stroll|stride)\b|'
                   r'\b(?:take|use|traverse)\s+(?:(?:the|a|an|next|another|flight|of)\s+){0,5}(?:stairs?|steps?|staircase)\b|'
                   r'\bhead(?=\s+(?:towards?|for|down|up|straight|left|right|out|into)\b)|'
                   r'\bstep(?=\s+(?:forward|back|backwards?|inside|outside|into|through|out|down|up)\b)', re.I)
_TURN = re.compile(r'\b(turn|rotate|pivot)\b|\bface(?=\s+(?:the|a|left|right|towards?)\b)|'
                   r'\b(?:take|make)\s+.{0,20}\b(left|right)\b', re.I)
_ARRIVE = re.compile(r'\b(stop|wait|halt|stand)\b', re.I)


def completion_conditions(text, prefix='P1', max_conditions=6):
    """Partition the original text into ordered coarse conditions, without paraphrasing.

    Colors and object attributes stay attached to the action/arrival they qualify.
    Long tails are merged, never dropped. Offsets refer to the original segment.
    """
    text = str(text or '')
    starts = [0]
    for number, match in enumerate(_ACTION.finditer(text)):
        if number == 0:
            # Introductory context/auxiliaries ("You'll stop ...") qualify
            # the first action; they are not a separate checklist item.
            continue
        start = match.start()
        if start <= 0 or not text[starts[-1]:start].strip():
            continue
        # Keep linking words with the following action where possible.
        link = re.search(r'\b(?:(?:and|then)\s+(?:go\s+)?|go\s+)$', text[starts[-1]:start], re.I)
        if link:
            start = starts[-1] + link.start()
        if start > starts[-1]:
            starts.append(start)
    starts = sorted(set(starts))[:max(1, int(max_conditions))]
    ends = starts[1:] + [len(text)]
    checks = []
    for start, end in zip(starts, ends):
        span = text[start:end].strip()
        if not span:
            continue
        requires_rotation = bool(_TURN.search(span))
        translation_with_turn = bool(requires_rotation and re.search(r'\b(through|into|out of)\b', span, re.I)
                                     and not re.search(r'\b(see|look|face)\b', span, re.I))
        requires_translation = bool(_MOVE.search(span) or translation_with_turn)
        if requires_translation:
            kind = 'movement'
        elif _TURN.search(span):
            kind = 'turn'
        elif _ARRIVE.search(span):
            kind = 'arrival'
        else:
            kind = 'relation'
        checks.append({'id': f'{prefix}.C{len(checks)+1}', 'text': span,
                       'source_start': start, 'source_end': end, 'kind': kind,
                       'requires_translation': requires_translation,
                       'requires_rotation': requires_rotation,
                       'requires_arrival': bool(_ARRIVE.search(span)),
                       'status': 'unknown', 'evidence': '', 'steps': []})
    return checks


def measured_movement(records):
    """Blocked forward proposals and net displacement are not executed progress."""
    return sum(max(0.0, float(r.get('motion_distance', 0.0) or 0.0))
               for r in records if isinstance(r, dict) and r.get('action_type') == 'move_forward')


def validate_condition_checks(expected, reported, ledger, previous=None,
                             require_current_arrival=False, current_pose=None,
                             arrival_pose_tolerance=0.35,
                             current_arrival_ids=None, witness_after_step=-1,
                             revoked_condition_ids=(), current_observation_id=None,
                             require_instruction_turn=False, instruction_subtask_id=None):
    """Check coverage, execution references and order; retain witnessed route milestones.

    Configured endpoint-arrival conditions are rechecked at the current pose.
    Earlier route milestones retain their original step bindings after live
    revalidation. A later model citation is not authority to rewrite history.
    Callers may revoke a prefix at a positively verified contradiction.
    Live Habitat control requires controller-issued directional turn intent;
    the default preserves offline interpretation of older unmarked ledgers.
    """
    previous = previous or {}
    arrival_ids = [c['id'] for c in expected
                   if c.get('requires_arrival', c.get('kind') == 'arrival')]
    endpoint_ids = (set(current_arrival_ids) if current_arrival_ids is not None
                    else set(arrival_ids[-1:]))
    retained = {}
    if previous:
        # Revalidate the entire old prefix together, preserving chronology and
        # distinct turn ownership. Rechecking rows individually would allow an
        # early clause to steal a later clause's turn when the model recites it.
        historical_rows = []
        for condition in expected:
            old = previous.get(condition['id'], {})
            historical_rows.append(old if old.get('execution_checked') and condition['id'] not in endpoint_ids
                                   else {**condition, 'status': 'unknown', 'steps': [], 'evidence': ''})
        historical, _ = validate_condition_checks(
            expected, historical_rows, ledger, witness_after_step=witness_after_step,
            require_instruction_turn=require_instruction_turn,
            instruction_subtask_id=instruction_subtask_id)
        for condition in historical:
            if condition['id'] in revoked_condition_ids or condition['status'] != 'satisfied':
                break
            retained[condition['id']] = condition
    by_step = {int(r['step']): r for r in ledger if isinstance(r, dict) and 'step' in r
               and not r.get('recovery') and not r.get('rejected_by_rollback')}
    expected_ids = [c['id'] for c in expected]
    rows = reported if isinstance(reported, list) else []
    expected_set = set(expected_ids)
    norm_rows = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        rc = dict(r)
        raw_id = str(rc.get('id') or '').strip()
        if raw_id and raw_id not in expected_set:
            upper_id = raw_id.upper()
            if upper_id in expected_set:
                rc['id'] = upper_id
            elif re.fullmatch(r'C\d+', upper_id) and expected_ids:
                prefix = expected_ids[0].split('.')[0]
                cand_id = f"{prefix}.{upper_id}"
                if cand_id in expected_set:
                    rc['id'] = cand_id
        raw_st = str(rc.get('status') or '').strip().lower()
        if raw_st in ('failed', 'not_satisfied', 'unmet', 'false', 'no', 'violated', 'rejected'):
            rc['status'] = 'contradicted'
        elif raw_st in ('passed', 'true', 'yes', 'completed', 'done', 'verified', 'executed'):
            rc['status'] = 'satisfied'
        elif raw_st in ('pending', 'incomplete', 'uncertain', 'unverified'):
            rc['status'] = 'unknown'
        norm_rows.append(rc)
    rows = norm_rows
    ids = [r.get('id') for r in rows if isinstance(r, dict)]
    conflict = len(ids) != len(expected_ids) or set(ids) != set(expected_ids)
    reported_by_id = {r.get('id'): r for r in rows if isinstance(r, dict)}
    checks = []
    last_witness_step = -1
    used_turn_steps = set()
    for condition in expected:
        row = retained.get(condition['id'], reported_by_id.get(condition['id'], {}))
        status = row.get('status', 'unknown')
        if status not in STATES:
            status = 'unknown'
            conflict = True
        evidence = str(row.get('evidence', '') or '')[:300]
        steps = row.get('steps', [])
        parsed_steps = []
        malformed_steps = not isinstance(steps, list)
        if isinstance(steps, list):
            for s in steps:
                if type(s) is int:
                    parsed_steps.append(s)
                elif isinstance(s, str):
                    s_str = s.strip()
                    rng = re.fullmatch(r'(\d+)\s*(?:-|to|\.\.)\s*(\d+)', s_str)
                    if re.fullmatch(r'\d+', s_str):
                        parsed_steps.append(int(s_str))
                    elif rng:
                        a_s, b_s = int(rng.group(1)), int(rng.group(2))
                        if a_s <= b_s and (b_s - a_s) <= 32:
                            parsed_steps.extend(range(a_s, b_s + 1))
                        else:
                            malformed_steps = True
                    else:
                        malformed_steps = True
                else:
                    malformed_steps = True
        clean_steps = sorted(set(parsed_steps))
        valid_steps = not malformed_steps and all(s in by_step for s in clean_steps)
        observation_id = row.get('observation_id')
        observation_checked = False
        # A fresh image ID may accompany an executed witness as metadata. It
        # cannot repair bad step references, motion, or an old arrival pose.
        observation_fresh = bool(
            type(observation_id) is int and observation_id >= 0
            and type(current_observation_id) is int
            and observation_id == current_observation_id)
        if observation_id is not None and not observation_fresh:
            conflict = True
            status = 'unknown'
            evidence = 'Observation metadata must reference the current controller observation.'
        # Only a pure live terminal arrival with no action citations may use
        # the observation itself as its witness. Motion stays ledger-bound.
        observation_allowed = bool(
            require_current_arrival and condition['id'] in endpoint_ids
            and condition.get('kind') == 'arrival' and condition.get('requires_arrival')
            and not condition.get('requires_translation') and not condition.get('requires_rotation')
            and observation_fresh and not clean_steps
            and valid_steps)
        if not valid_steps:
            conflict = True
            status, steps = 'unknown', []
        else:
            steps = clean_steps
        progress_boundary = max([witness_after_step] + [s for c in checks if c['status'] == 'satisfied' for s in c['steps']])
        motion_ok, motion_reason = measured_condition_witness(
            condition, [by_step[s] for s in steps], ledger, after_step=progress_boundary)
        if status == 'satisfied':
            if not evidence.strip():
                status = 'unknown'
            elif condition.get('requires_translation', condition['kind'] == 'movement') and not any(
                    by_step[s].get('action_type') == 'move_forward'
                    and float(by_step[s].get('motion_distance', 0.0)) >= 0.03 for s in steps):
                status = 'unknown'
                evidence = 'No referenced forward action produced measured movement.'
            elif not motion_ok:
                status = 'unknown'
                evidence = motion_reason
            elif (condition.get('requires_arrival', condition['kind'] == 'arrival') and require_current_arrival
                  and (current_arrival_ids is None or condition['id'] in current_arrival_ids)):
                pose = np.asarray(current_pose, dtype=float) if current_pose is not None else None
                current_pose_witnessed = bool(observation_allowed and pose is not None
                                             and pose.ndim == 1 and pose.size >= 2 and np.isfinite(pose).all())
                observation_checked = current_pose_witnessed
                if pose is not None and pose.ndim == 1 and pose.size >= 2 and np.isfinite(pose).all():
                    for step in steps:
                        recorded_pose = by_step.get(step, {}).get('pose')
                        if recorded_pose is None:
                            continue
                        recorded_pose = np.asarray(recorded_pose, dtype=float)
                        if (recorded_pose.ndim == 1 and recorded_pose.size >= 2
                                and recorded_pose.size == pose.size
                                and np.isfinite(recorded_pose).all()
                                and np.linalg.norm(recorded_pose - pose)
                                <= float(arrival_pose_tolerance)):
                            current_pose_witnessed = True
                            break
                if not current_pose_witnessed:
                    status = 'unknown'
                    evidence = 'No cited executed pose or fresh observation matches the current arrival position.'
        if status == 'contradicted':
            # The first unfinished condition is current. Later clauses are
            # future work, not violations, even if the endpoint is not visible.
            prefix_complete = all(c['status'] == 'satisfied' for c in checks)
            acted = any((by_step[s].get('action_type') == 'move_forward'
                         and float(by_step[s].get('motion_distance', 0.0)) >= .03)
                        or (by_step[s].get('action_type') == 'turn'
                            and abs(float(by_step[s].get('yaw_delta', 0.0))) >= .05)
                        for s in steps)
            if not prefix_complete or not steps or not evidence.strip() or not acted:
                status = 'unknown'
                evidence = 'Unfinished or future route condition has no executed contradiction witness.'
        # A terminal request does not make an arbitrary historical action
        # evidence for this particular clause.  The verifier must identify
        # the matching action and observed route relation itself.
        if status == 'satisfied' and condition['kind'] in ('movement', 'turn') and steps:
            # A single action may cover linked conditions; earlier events may
            # not be invented after a later event has already been witnessed.
            # Compare against the earliest step of the preceding clause so broad
            # step ranges (e.g. [1..40] followed by [15..18]) are not rejected.
            witness_max = max(steps)
            witness_min = min(steps)
            if witness_max < last_witness_step:
                status = 'unknown'
                evidence = 'Execution references contradict the source condition order.'
            else:
                last_witness_step = max(last_witness_step, witness_min)
        if status == 'satisfied' and condition.get('requires_rotation', condition['kind'] == 'turn'):
            fresh_records = [by_step[s] for s in steps if s not in used_turn_steps]
            turn_steps = turn_witness_steps(fresh_records, ledger)
            fresh_ok, _ = turn_witness(condition.get('text', ''), fresh_records, ledger)
            fresh_ok = fresh_ok and not (turn_steps & used_turn_steps)
            if not fresh_ok:
                status = 'unknown'
                evidence = 'Sequential turn clauses require distinct direction-matching executed turns.'
            elif require_instruction_turn and not instruction_turn_witness(
                    condition, fresh_records, ledger, instruction_subtask_id):
                status = 'unknown'
                evidence = ('This turn condition needs its own landmark-bound explicit instruction turn; '
                            'waypoint alignment or observation yaw is not instruction execution.')
            else:
                used_turn_steps.update(turn_steps)
        checks.append({**condition, 'status': status, 'evidence': evidence, 'steps': steps,
                       'execution_checked': status == 'satisfied' and not observation_checked,
                       'observation_checked': status == 'satisfied' and observation_checked,
                       'observation_id': observation_id if observation_checked else None})
    return checks, conflict


def verification_schema(conditions, entity_ids, candidate_ids=None):
    """The same schema is used for current-segment and remaining-route checks."""
    bbox = {'type': ['array', 'null'], 'items': {'type': 'number', 'minimum': 0, 'maximum': 1},
            'minItems': 4, 'maxItems': 4}
    bbox_space = {'type': ['string', 'null'], 'enum': ['full_view', None]}
    if candidate_ids is None:
        cand_id = {'type': ['string', 'null']}
    else:
        clean_cids = list(dict.fromkeys(c for c in candidate_ids if type(c) is str and c.strip()))[:64]
        cand_id = {'type': ['string', 'null'], 'enum': [None, *clean_cids]}
    region = {'type': ['object', 'null'], 'additionalProperties': False,
              'properties': {'view': {'type': 'string', 'enum': ['front', 'left', 'right', 'back']},
                             'bbox': bbox,
                             'bbox_space': bbox_space,
                             'role': {'type': 'string', 'enum': ['door_jamb', 'between_object', 'spatial_landmark', 'unknown']},
                             'entity_id': {'type': ['integer', 'null'], 'enum': [None, *entity_ids]},
                             'candidate_id': cand_id},
              'required': ['view', 'bbox', 'bbox_space', 'entity_id', 'candidate_id', 'role']}
    landmark_region = {'type': ['object', 'null'], 'additionalProperties': False,
                       'properties': {'view': {'type': 'string', 'enum': ['front', 'left', 'right', 'back']},
                                      'bbox': bbox,
                                      'bbox_space': bbox_space,
                                      'role': {'type': 'string', 'enum': ['spatial_landmark']},
                                      'entity_id': {'type': ['integer', 'null'], 'enum': [None, *entity_ids]},
                                      'candidate_id': cand_id},
                       'required': ['view', 'bbox', 'bbox_space', 'entity_id', 'candidate_id', 'role']}
    reasoning_props = {
        'observation': {
            'type': 'object', 'additionalProperties': False,
            'properties': {
                'front view': {'type': 'string'},
                'left view': {'type': 'string'},
                'right view': {'type': 'string'},
                'back view': {'type': 'string'},
                'map': {'type': 'string'},
            },
            'required': ['front view', 'left view', 'right view', ' map'.strip()],
        },
        'physical_check': {'type': 'string'},
        'progress_analysis': {'type': 'string'},
        'scene_consistent': {'type': ['boolean', 'null']},
        'evidence': {'type': 'string'},
        'contradiction_evidence': {'type': 'string'},
        'route_contradiction': {
            'type': ['object', 'null'], 'additionalProperties': False,
            'properties': {
                'kind': {'type': 'string', 'enum': ['turn_direction', 'straight_path', 'wrong_room', 'wrong_branch', 'wrong_floor']},
                'condition_id': {'type': 'string', 'enum': [c['id'] for c in conditions]},
                'steps': {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 1},
                'expected': {'type': 'string'}, 'observed': {'type': 'string'}},
            'required': ['kind', 'condition_id', 'steps', 'expected', 'observed']},
        'condition_checks': {
            'type': 'array', 'minItems': len(conditions), 'maxItems': len(conditions),
            'items': {'type': 'object', 'additionalProperties': False,
                      'properties': {'id': {'type': 'string', 'enum': [c['id'] for c in conditions]},
                                     'status': {'type': 'string', 'enum': list(STATES)},
                                     'evidence': {'type': 'string'},
                                     'observation_id': {'type': ['integer', 'null'], 'minimum': 0},
                                     'steps': {'type': 'array', 'items': {'type': 'integer'}, 'maxItems': 4}},
                      'required': ['id', 'status', 'evidence', 'steps', 'observation_id']},
        },
        'target_entity_id': {'type': ['integer', 'null'], 'enum': [None, *entity_ids]},
        'target_candidate_id': cand_id,
        'target_identity': {'type': 'string'},
        'target_proximity': {'type': 'string', 'enum': list(STATES)},
        'target_evidence': {'type': 'string'},
        'target_view': {'type': ['string', 'null'], 'enum': ['front', 'left', 'right', 'back', None]},
        'target_bbox': bbox,
        'target_bbox_space': bbox_space,
        'reference_region': region,
        'spatial_landmark_region': landmark_region,
    }
    output_props = {
        'outcome': {'type': 'boolean'},
        'reason': {'type': 'string'},
        'failure_type': {'type': 'string', 'enum': ['none', 'not_reached', 'wrong_path']},
        'relation_satisfied': {'type': 'boolean'},
        'confidence': {'type': 'string', 'enum': ['low', 'medium', 'high']},
        'next_start_consistent': {'type': ['boolean', 'null']},
        'negative_landmarks_detected': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 4},
        'spatial_clue': {'type': 'string'},
        'rejected_target': {'type': 'string'},
        'terminal_correction': {'type': 'boolean'},
    }
    return {
        'name': 'navigation_verification',
        'schema': {
            'type': 'object', 'additionalProperties': False,
            'properties': {
                'reasoning': {
                    'type': 'object', 'additionalProperties': False,
                    'properties': reasoning_props,
                    'required': list(reasoning_props),
                },
                'output': {
                    'type': 'object', 'additionalProperties': False,
                    'properties': output_props,
                    'required': list(output_props),
                },
            },
            'required': ['reasoning', 'output'],
        },
    }


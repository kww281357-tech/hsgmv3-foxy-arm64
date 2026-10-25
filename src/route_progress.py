"""Measured route progress and positive wrong-route witnesses, without goal truth.

Mapper yaw follows camera forward (-R[:, 2]): LEFT is a negative measured
``yaw_delta`` and RIGHT is positive. Command ``theta`` has the opposite sign.
"""
import math
import re

import numpy as np


CONTRADICTION_KINDS = ('turn_direction', 'straight_path', 'wrong_room', 'wrong_branch', 'wrong_floor')
_TURN_DIRECTION = re.compile(
    r'\b(?:turn|rotate|pivot|face|take|make)\s+'
    r'(?:(?:to|on)\s+)?(?:(?:your|the|a|another|first|second|third|next)\s+)*'
    r'(left|right|around|back)\b', re.I)
_STRAIGHT = re.compile(r'\b(?:walk|go|move|continue|proceed|head|travel|run|advance|step)\s+'
                       r'(?:(?:on|ahead|forward)\s+)?straight\b', re.I)
_ENTERED_FACT = re.compile(
    r'\b(?:entered|re-?entered|crossed|exited|took|followed|'
    r'(?:walked|moved|went|stepped|travelled|traveled)\s+(?:into|through|out of|along|back\s+to)|'
    r'(?:travelled|traveled)\s+to|returned\s+(?:back\s+)?(?:to|into|through))\b', re.I)
_NONFACTUAL_TRANSITION = re.compile(
    r'\b(?:if|unless|would|could|should|may|might|will|can|must|maybe|perhaps|possibly|'
    r'probably|hypothetically|assuming|suppose|seems?|appears?|almost|nearly)\b|'
    r'\b(?:plans?|planned|intends?|intended|expects?|expected|needs?|going)\s+to\b|'
    r'\b(?:plan|planned|intended|expected|hypothetical)\s*:|'
    r'\bnot\s+(?:confirmed|observed|established|proven)\b|\?', re.I)
_NEGATED_TRANSITION = re.compile(r"\b(?:not|never|no|without|neither)\b|n['’]t\b", re.I)
_UNSUPPORTED_TRANSITION = re.compile(
    r'\b(?:unconfirmed|unverified|unproven|unsupported|uncertain|unclear)\b|'
    r'\b(?:false|incorrect|untrue)\s+that\b|'
    r'\b(?:insufficient|inadequate|no)\s+(?:evidence|proof|confirmation)\b|'
    r'\b(?:not|never)\s+(?:been\s+)?(?:confirmed|observed|established|proven|verified|supported|true)\b', re.I)
_ABSENCE_ONLY = re.compile(
    r'\b(?:not (?:yet |been )?(?:seen|visible|reached|entered|executed|completed|performed|standing)|'
    r'no (?:evidence|left turns?|right turns?|doorway|target)|'
    r'(?:cannot|can.t|could not) see|not in (?:the )?(?:front|left|right|back) view|'
    r'still (?:needs?|has|in)|has not|have not|only (?:moved|walked)|'
    r'endpoint not|target (?:absent|missing)|not at)\b', re.I)


def executed_records(ledger):
    return {r['step']: r for r in ledger if isinstance(r, dict)
            and type(r.get('step')) is int and not r.get('recovery')
            and not r.get('rejected_by_rollback')}


def observed_room_transition(text):
    """Recognize an asserted, completed transition, not a plan or negation.

    This is language validation only; movement, route scope and step references
    are checked separately. It does not establish the visual room identity or
    independently prove that the cited movement crossed a room boundary.
    """
    text = str(text or '')
    # A trailing disclaimer can negate an earlier assertion across "but" or
    # punctuation; never discard it by accepting the earlier clause first.
    if _UNSUPPORTED_TRANSITION.search(text):
        return False
    for clause in re.split(r'[.;\n]|\bbut\b', text, flags=re.I):
        if _NONFACTUAL_TRANSITION.search(clause):
            continue
        for match in _ENTERED_FACT.finditer(clause):
            if not _NEGATED_TRANSITION.search(clause[:match.start()]):
                return True
    return False


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _xy(value):
    try:
        point = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return point[:2] if point.ndim == 1 and point.size >= 2 and np.isfinite(point).all() else None


def turn_directions(text):
    """Match the turning verb, never a landmark 'to the right/left of ...'."""
    return [m.group(1).lower() for m in _TURN_DIRECTION.finditer(str(text or ''))]


def _turn_blocks(records, ledger=None):
    """Resolve citations to physical rotation blocks, separated by translation.

    Sparse citations must neither join turns separated by real movement nor
    hide an uncited in-place reversal. Use the full measured ledger whenever it
    is available, including adjacent turns outside the cited step interval.
    """
    selected = {r.get('step') for r in records if r.get('action_type') == 'turn'}
    rows = records if ledger is None else sorted(
        (r for r in ledger if isinstance(r, dict) and type(r.get('step')) is int), key=lambda r: r['step'])
    blocks, current, previous_step, translation = [], [], None, 0.
    for row in rows:
        if ledger is not None and previous_step is not None and row['step'] != previous_step + 1:
            # A filtered or missing interval has unknown physical motion.
            # It cannot establish continuous rotation across the gap.
            if current:
                blocks.append(current)
            current = []
            translation = 0.
        previous_step = row.get('step')
        if row.get('recovery') or row.get('rejected_by_rollback'):
            # Recovery is not route evidence, but it also cannot splice two
            # ordinary turns on different attempts into one rotation block.
            if current:
                blocks.append(current)
            current = []
            translation = 0.
        elif row.get('action_type') == 'turn' and _number(row.get('yaw_delta')) is not None:
            current.append(row)
            translation = 0.
        else:
            translation += max(0., _number(row.get('motion_distance')) or 0.)
            # A blocked proposal or numeric pose drift cannot hide an in-place
            # reversal. Accumulate consecutive small moves to the same 3 cm
            # measured-progress threshold used by route movement witnesses.
            if ledger is None or translation >= .03 - 1e-9:
                if current:
                    blocks.append(current)
                current = []
                translation = 0.
    if current:
        blocks.append(current)
    return [block for block in blocks if ledger is None or any(r['step'] in selected for r in block)]


def _turn_groups(records, ledger=None):
    turns = [sum(float(r['yaw_delta']) for r in block) for block in _turn_blocks(records, ledger)]
    groups = []
    for yaw in turns:
        magnitude = abs(math.atan2(math.sin(yaw), math.cos(yaw)))
        if magnitude >= .05:
            # Direction describes measured rotation, not the mathematical
            # shortest way to its final heading: left 270 is never right 90.
            # Removing complete revolutions still rejects a full-circle spin.
            groups.append(math.copysign(magnitude, yaw))
    return groups


def turn_witness_steps(records, ledger):
    """All steps owned by the cited physical rotation blocks."""
    return {r['step'] for block in _turn_blocks(records, ledger) for r in block}


def turn_witness(text, records, ledger=None):
    groups = _turn_groups(records, ledger)
    directions = turn_directions(text)
    if not directions:
        return bool(groups), 'No referenced turn was executed.'
    start = 0
    for direction in directions:
        for i in range(start, len(groups)):
            yaw = groups[i]
            matches = ((direction == 'left' and yaw <= -math.radians(20))
                       or (direction == 'right' and yaw >= math.radians(20))
                       or (direction in ('around', 'back') and abs(yaw) >= math.radians(150)))
            if matches:
                start = i + 1
                break
        else:
            return False, 'Referenced measured turns do not match the instructed direction and magnitude.'
    return True, ''


def _instruction_turn_blocks(condition, records, ledger, subtask_id):
    """Find cited rotation blocks owned by this controller-issued instruction."""
    directions = turn_directions(condition.get('text'))
    if len(directions) != 1:
        return []
    direction = {'left': 'LEFT', 'right': 'RIGHT', 'around': 'BACK', 'back': 'BACK'}[directions[0]]
    def belongs(row):
        command = row.get('instruction_turn') if isinstance(row, dict) else None
        if (not isinstance(command, dict)
                or row.get('recovery') or row.get('rejected_by_rollback')
                or row.get('subtask_id') != subtask_id
                or row.get('action_type') != 'turn'
                or row.get('requested_action') != {'type': 'turn', 'direction': direction}
                or any(command.get(k) != value for k, value in (
                    ('subtask_id', subtask_id), ('condition_id', condition['id']),
                    ('text', condition.get('text')), ('direction', direction)))):
            return False
        return turn_witness(condition['text'], [row], [row])[0]
    return [block for block in _turn_blocks(records, ledger) if any(belongs(row) for row in block)]


def instruction_turn_witness(condition, records, ledger, subtask_id):
    """Bind a directional clause to intent and the full measured rotation block.

    A later waypoint or observation rotation cannot borrow this intent after
    translation. Conversely, an uncited in-place undo remains in its block.
    """
    # Only directional clauses supported by the explicit-turn controller can
    # carry this marker. Other rotation relations retain their existing gate.
    text = str(condition.get('text') or '')
    if len(turn_directions(text)) != 1:
        return True
    blocks = _instruction_turn_blocks(condition, records, ledger, subtask_id)
    if any(turn_witness(text, block)[0] for block in blocks):
        return True
    # When a non-initial turn clause has no named landmark ("at <landmark>") and
    # followed an unverified movement clause in the same subtask without an
    # explicit instruction_turn marker in the ledger, fall back to the measured
    # rotation block witness so compound instructions ("go into X and make a right")
    # are not deadlocked when executed within one waypoint sequence.
    cid = str(condition.get('id') or '')
    has_named_landmark = bool(re.search(r'\bat\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:[.;,]|$)', text, re.I))
    has_any_command_for_clause = any(
        isinstance(r.get('instruction_turn'), dict)
        and r['instruction_turn'].get('condition_id') == cid
        for r in (ledger or ()) if isinstance(r, dict)
    )
    if not has_named_landmark and not cid.endswith('.C1') and not has_any_command_for_clause:
        return bool(turn_witness(text, records, ledger)[0])
    return False


def requires_straight(text):
    return bool(_STRAIGHT.search(str(text or '')))


def straight_witness(records, ledger, after_step=None):
    """Check the whole cited interval, including turns omitted from step citations.

    Fresh ledgers provide yaw_before. Old records can establish heading from an
    immediately preceding measured move, or two aligned displacements. Missing
    geometry is unknown, never a directional contradiction.
    """
    if not records:
        return False, False, 'Straight movement needs measured pose references.'
    by_step = executed_records(ledger)
    lo, hi = min(r['step'] for r in records), max(r['step'] for r in records)
    if after_step is not None:
        lo = min(lo, int(after_step) + 1)
    window = [by_step[s] for s in sorted(by_step) if lo <= s <= hi]
    moves = []
    for row in window:
        if row.get('action_type') != 'move_forward':
            continue
        before, after = _xy(row.get('pose_before')), _xy(row.get('pose'))
        if before is None or after is None:
            return False, False, 'Straight movement needs finite before/after poses.'
        delta = after - before
        distance = float(np.linalg.norm(delta))
        if distance >= .03:
            moves.append((row, before, after, delta, distance))
    if not moves or sum(m[4] for m in moves) < .15:
        return False, False, 'No sufficient measured straight translation.'
    first = moves[0]
    heading = _number(first[0].get('yaw_before'))
    if heading is None:
        prior = [r for s, r in sorted(by_step.items()) if s < first[0]['step']
                 and r.get('action_type') == 'move_forward']
        if prior:
            row = prior[-1]
            before, after = _xy(row.get('pose_before')), _xy(row.get('pose'))
            if before is not None and after is not None and np.linalg.norm(after-before) >= .03:
                # A separate earlier clause may have turned; reconstruct the
                # actual heading at this interval's entry, not its old heading.
                if np.linalg.norm(first[1]-after) <= .1:
                    heading = math.atan2(*(after-before)[::-1])
                    heading += sum(_number(r.get('yaw_delta')) or 0. for s, r in by_step.items()
                                   if row['step'] < s < first[0]['step'])
    if heading is None:
        if len(moves) < 2:
            return False, False, 'A lone forward action without measured entry heading cannot prove straight travel.'
        heading = math.atan2(first[3][1], first[3][0])
    direction = np.array([math.cos(heading), math.sin(heading)])
    backwards_or_sideways = any(float(np.dot(m[3]/m[4], direction)) < math.cos(math.radians(25))
                               for m in moves)
    turns = sum(abs(_number(r.get('yaw_delta')) or 0.) for r in window if r.get('action_type') == 'turn')
    disconnected = any(np.linalg.norm(b[1]-a[2]) > .1 for a, b in zip(moves, moves[1:]))
    length = sum(m[4] for m in moves)
    chord = float(np.linalg.norm(moves[-1][2]-moves[0][1]))
    bent = backwards_or_sideways or turns > math.radians(25) or chord < .9 * length
    if bent and len(moves) >= 3 and length >= 1.5 and chord >= .92 * length:
        net_vec = moves[-1][2] - moves[0][1]
        net_turn = sum(_number(r.get('yaw_delta')) or 0. for r in window if r.get('action_type') == 'turn')
        max_single_turn = max((abs(_number(r.get('yaw_delta')) or 0.) for r in window if r.get('action_type') == 'turn'), default=0.)
        if (float(np.dot(net_vec / chord, direction)) >= math.cos(math.radians(28))
                and abs(net_turn) <= math.radians(28)
                and max_single_turn <= math.radians(65)
                and all(float(np.dot(m[3]/m[4], direction)) >= math.cos(math.radians(50)) for m in moves)):
            bent = False
    if disconnected:
        return False, False, 'Straight witnesses are not a continuous measured path.'
    return not bent, bent, 'Cited path bends or turns instead of continuing straight.' if bent else ''


def measured_condition_witness(condition, records, ledger, after_step=None):
    if condition.get('requires_rotation', condition.get('kind') == 'turn'):
        okay, reason = turn_witness(condition.get('text'), records, ledger)
        if not okay:
            return False, reason
    if requires_straight(condition.get('text')):
        okay, _, reason = straight_witness(records, ledger, after_step=after_step)
        if not okay:
            return False, reason
    return True, ''


def route_progress_packet(checks):
    prefix, current = [], None
    for condition in checks:
        if current is None and condition.get('status') == 'satisfied' and condition.get('execution_checked'):
            prefix.append(condition['id'])
        elif current is None:
            current = condition['id']
    return {'completed_prefix': prefix, 'current_condition_id': current,
            'future_condition_ids': [c['id'] for c in checks
                                     if c['id'] not in prefix and c['id'] != current]}


def contradiction_recovery_constraint(evidence, checks, ledger):
    """Bind a validated contradiction to an execution-before checkpoint range.

    A checkpoint's ledger_len is the next action index, not its last action.
    Preserve preceding verified clauses and return before the affected motion.
    In particular a straight clause owns its complete interval: a later cited
    bend cannot hide an earlier turn, nor authorize restarting a sliced suffix.
    This helper is called only after the contradiction itself has been checked.
    """
    cid, kind = evidence['condition_id'], evidence['kind']
    index = next(i for i, c in enumerate(checks) if c['id'] == cid)
    by_step = executed_records(ledger)
    prior = checks[:index]
    # Rejected earlier attempts do not become verified progress: after a
    # return-to-entry, the next live ledger row may have a large global index.
    # Only actual preceding verified clauses establish the lower boundary.
    boundary = max([-1] + [s for c in prior
        if c.get('status') == 'satisfied' and c.get('execution_checked')
        for s in c.get('steps', [])])
    rows = [by_step[s] for s in sorted(set(evidence['steps']))]
    affected = []
    if kind == 'straight_path':
        window = [r for s, r in sorted(by_step.items()) if boundary < s <= max(evidence['steps'])]
        # Use the original full-window validator to locate the first bent
        # prefix. A later turn must not hide an already-sideways translation.
        for end in window:
            if straight_witness([end], ledger, after_step=boundary)[1]:
                window = [r for r in window if r['step'] <= end['step']]
                break
        # Conservatively include the first turn in the violated clause, even
        # when its cumulative bend became decisive only at a later action.
        affected = [r['step'] for r in window if r.get('action_type') == 'turn'
                    and abs(_number(r.get('yaw_delta')) or 0.) > 1e-6]
        if not affected:
            # Sideways / bent translation without recorded turns has no finer
            # trustworthy boundary. Revisit the start of the measured clause.
            affected = [r['step'] for r in window if r.get('action_type') == 'move_forward'
                        and (_number(r.get('motion_distance')) or 0.) >= .03]
    elif kind == 'turn_direction':
        direction = turn_directions(checks[index].get('text'))
        for block in _turn_blocks(rows, ledger):
            groups = _turn_groups(block)
            if any((direction == ['left'] and yaw >= math.radians(20)) or
                   (direction == ['right'] and yaw <= -math.radians(20)) for yaw in groups):
                affected.extend(r['step'] for r in block)
    elif kind == 'wrong_floor':
        going_up = bool(re.search(r'\b(?:up|ascend|upstairs)\b', checks[index].get('text', ''), re.I))
        affected = [r['step'] for r in rows if ((_number(r.get('height_delta')) or 0.) < 0) == going_up
                    and abs(_number(r.get('height_delta')) or 0.) > 1e-6]
    else:
        affected = [r['step'] for r in rows if r.get('action_type') == 'move_forward'
                    and (_number(r.get('motion_distance')) or 0.) > 0]
    return {'condition_id': cid, 'kind': kind, 'condition_boundary_step': boundary,
            'cited_steps': sorted(set(evidence['steps'])),
            'earliest_affected_step': min(affected, default=min(evidence['steps'])),
            'min_checkpoint_ledger_len': boundary + 1,
            'max_checkpoint_ledger_len': min(affected, default=min(evidence['steps']))}


def checkpoint_precedes_contradiction(checkpoint, constraint, subtask_key):
    """Fail closed on missing / stale checkpoint indexes for evidenced recovery."""
    if not isinstance(constraint, dict) or not isinstance(checkpoint, dict):
        return False
    lower, upper = constraint.get('min_checkpoint_ledger_len'), constraint.get('max_checkpoint_ledger_len')
    cutoff = checkpoint.get('ledger_len')
    return bool(type(lower) is int and type(upper) is int and type(cutoff) is int
                and 0 <= lower <= cutoff <= upper
                and isinstance(constraint.get('condition_id'), str)
                and constraint['condition_id'].startswith(str(subtask_key) + '.')
                and checkpoint.get('subtask_key') == subtask_key)


def check_route_contradiction(parsed, checks, ledger, *, require_instruction_turn=False,
                             instruction_subtask_id=None):
    """Return a validated witness, or a reason that forbids route rollback.

    Semantic room identity is still an observation claim, not evaluator truth.
    Structural checks tie it to real current-prefix motion and original text.
    """
    evidence = parsed.get('route_contradiction')
    if parsed.get('scene_consistent') is not False:
        return None, 'Wrong-route recovery requires an explicitly inconsistent observed scene.'
    if parsed.get('confidence') != 'high' or not isinstance(evidence, dict):
        return None, 'Wrong-route recovery requires a structured high-confidence observed contradiction.'
    kind, cid = evidence.get('kind'), evidence.get('condition_id')
    condition = next((c for c in checks if c['id'] == cid), None)
    progress = route_progress_packet(checks)
    if (condition is None or cid in progress['future_condition_ids']
            or condition.get('status') != 'contradicted'):
        return None, 'A future or unexecuted condition is unfinished, not a wrong-route witness.'
    steps = evidence.get('steps')
    by_step = executed_records(ledger)
    if (not isinstance(steps, list) or not steps or any(type(s) is not int or s not in by_step for s in steps)
            or not set(steps).issubset(condition.get('steps', []))):
        return None, 'The contradiction must cite measured non-recovery steps of the affected condition.'
    observed, expected = evidence.get('observed'), evidence.get('expected')
    normalize = lambda text: ' '.join(re.findall(r'\w+', str(text).lower()))
    if (not isinstance(kind, str) or kind not in CONTRADICTION_KINDS or not isinstance(observed, str) or not observed.strip()
            or not isinstance(expected, str) or len(normalize(expected).split()) < 2
            or normalize(expected) not in normalize(condition.get('text', ''))
            or normalize(observed) == normalize(expected) or _ABSENCE_ONLY.search(observed)):
        return None, 'A contradiction needs a positive observation and an exact expected source-clause fragment.'
    rows = [by_step[s] for s in sorted(set(steps))]
    owned_turn_steps = None
    if kind == 'turn_direction':
        directions = turn_directions(condition.get('text'))
        if require_instruction_turn:
            owned = _instruction_turn_blocks(condition, rows, ledger, instruction_subtask_id)
            if not owned:
                return None, 'Cited opposite turn is not bound to this condition\'s instruction-turn rotation block.'
            groups = [yaw for block in owned for yaw in _turn_groups(block)]
            owned_turn_steps = sorted({row['step'] for block in owned for row in block})
        else:
            groups = _turn_groups(rows, ledger)
        opposite = ((directions == ['left'] and any(v >= math.radians(20) for v in groups))
                    or (directions == ['right'] and any(v <= -math.radians(20) for v in groups)))
        if not opposite:
            return None, 'No measured opposite turn supports this direction contradiction.'
    elif kind == 'straight_path':
        prior = checks[:next(i for i, c in enumerate(checks) if c['id'] == cid)]
        boundary = max([-1] + [s for c in prior if c.get('execution_checked') for s in c.get('steps', [])])
        _, contradicted, _ = straight_witness(rows, ledger, after_step=boundary)
        if not requires_straight(condition.get('text')) or not contradicted:
            return None, 'No measured bend or opposite travel contradicts an explicit straight instruction.'
    elif kind == 'wrong_floor':
        text = str(condition.get('text', '')).lower()
        up = bool(re.search(r'\b(?:up|ascend|upstairs)\b', text))
        down = bool(re.search(r'\b(?:down|descend|downstairs)\b', text))
        dz = sum(_number(r.get('height_delta')) or 0. for r in rows)
        if up == down or not ((up and dz < -.15) or (down and dz > .15)):
            return None, 'No measured opposite elevation change contradicts this route clause.'
    else:
        movement = sum(max(0., _number(r.get('motion_distance')) or 0.) for r in rows
                       if r.get('action_type') == 'move_forward')
        if (not condition.get('requires_translation', condition.get('kind') == 'movement')
                or movement < .15 or not observed_room_transition(observed)):
            return None, 'Room or branch contradictions require movement into the positively observed wrong place.'
    verified = {k: evidence[k] for k in ('kind', 'condition_id', 'steps', 'expected', 'observed')}
    recovery_evidence = verified
    if owned_turn_steps is not None:
        # Raw citations remain auditable, but unowned rotations cannot widen
        # the recovery range beyond the blocks that authorized the verdict.
        verified['instruction_turn_steps'] = owned_turn_steps
        recovery_evidence = dict(verified, steps=owned_turn_steps)
    verified['recovery_constraint'] = contradiction_recovery_constraint(recovery_evidence, checks, ledger)
    return verified, None


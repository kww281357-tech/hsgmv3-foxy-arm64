"""R2R-CE text contracts, independent of the simulator and model provider."""

import math
import re


class ModelServiceError(RuntimeError):
    """An exhausted API request is an infrastructure failure, not navigation evidence."""


class ModelOutputContractError(RuntimeError):
    """Model output violates schema/JSON syntax or candidate action contract after allowed repair attempts."""



def instruction_requires_movement(text: str) -> bool:
    """Return True if the text instructs physical translation/locomotion."""
    if not text:
        return True
    locomotion = r"\b(walk|go|proceed|head|enter|exit|pass|climb|continue|cross|leave|move|approach|reach|follow|step|run|advance|travel|navigate|stroll|stride|make your way|opposite|across|through|down|up|into|out of|past|toward|towards|to the)\b"
    return bool(re.search(locomotion, text.lower()))


def completion_probe_due(is_final, decision_id, subtask_id, last_check, stalled=False,
                         final_interval=1, intermediate_interval=4):
    """Probe at planner boundaries, never by reading evaluator goal coordinates."""
    last_check = last_check or {}
    same_segment = last_check.get('subtask_id') == subtask_id
    elapsed = decision_id - last_check.get('decision_id', 0)
    if same_segment and elapsed <= 0:
        return False
    if stalled or (is_final and not same_segment):
        return True
    interval = max(1, int(final_interval if is_final else intermediate_interval))
    return elapsed >= interval



def verifier_feedback(result):
    """A past rejection is not a permanent ban on proposing STOP."""
    if not result:
        return ''
    if result.get('available') is False:
        return ('Previous verification was unavailable, not evidence of failure. '
                'Reassess completion from the current views; do not move away just to satisfy an API error.')
    reason = str(result.get('reason', ''))[:360]
    if result.get('failure_type') == 'not_reached':
        return ('The verifier did not confirm the current segment. Recheck the observed spatial relation '
                'and follow the requested JSON format and current available action list. Propose action "STOP" '
                'only when it is currently available and the current views establish completion; '
                'otherwise choose a legal waypoint or one useful turn. '
                'Do not roll back for a segment that is merely not reached. '
                'If visibility alone is uncertain, prefer one informative turn over walking past the endpoint. '
                f'Previous reason: {reason}')
    return ''


def _words(text):
    return [m.group().lower() for m in re.finditer(r"\w+(?:['’]\w+)*", text)]


def partition_instruction(instruction, proposed=None, max_subtasks=6):
    """Keep every original word, in order, including ordinal/side/stop clauses.

    Accept model boundaries only if they form a full partition of the source.
    Invalid model output falls back to sentence/newline boundaries, never commas
    (commas frequently bind a turn or spatial relation to its landmark).
    """
    source = instruction.strip()
    tokens = list(re.finditer(r"\w+(?:['’]\w+)*", source))
    groups = proposed if isinstance(proposed, list) else []
    valid = bool(groups) and all(isinstance(s, str) and _words(s) for s in groups)
    valid = valid and [w for s in groups for w in _words(s)] == _words(source)
    if valid:
        boundaries = [0]
        consumed = 0
        for s in groups[:-1]:
            consumed += len(_words(s))
            boundaries.append(tokens[consumed].start())
        boundaries.append(len(source))
        spans = [(a, b) for a, b in zip(boundaries, boundaries[1:])]
    else:
        ends = [m.end() for m in re.finditer(r"[.!?;](?:\s+|$)|\n+", source)]
        boundaries = sorted(set([0, *ends, len(source)]))
        spans = [(a, b) for a, b in zip(boundaries, boundaries[1:]) if source[a:b].strip()]
    if not spans:
        return [source]

    def kind(span):
        text = source[slice(*span)].strip().lower()
        text = re.sub(r"^(?:(?:and|then|after that)\s+)+", "", text)
        if re.match(r"(?:stop|wait|halt|stand|pause)\b|(?:that's|that is) where", text):
            return "stop"
        locomotion = r"\b(walk|go|proceed|head|enter|exit|pass|climb|continue|cross|leave|move|approach|reach|follow|step)\b"
        if re.search(r"\b(turn|face|rotate|pivot)\b|(?:take|make) a (?:left|right)", text) and not re.search(locomotion, text):
            return "turn"
        if re.match(r"(?:there|it|this|that|which)\b", text):
            return "dependent"
        return "move"

    i = 0
    while len(spans) > 1 and i < len(spans):
        tag = kind(spans[i])
        if tag in ("stop", "dependent") and i > 0:
            spans[i-1:i+1] = [(spans[i-1][0], spans[i][1])]
            i = max(0, i-1)
        elif tag == "turn":
            j = i if i + 1 < len(spans) else i - 1
            spans[j:j+2] = [(spans[j][0], spans[j+1][1])]
            i = max(0, j-1)
        else:
            i += 1
    while len(spans) > max(1, int(max_subtasks)):
        j = min(range(len(spans)-1), key=lambda n: spans[n+1][1]-spans[n][0])
        spans[j:j+2] = [(spans[j][0], spans[j+1][1])]
    return [source[a:b].strip() for a, b in spans]


def normalize_verification(parsed):
    """Strict control contract. Missing/contradictory evidence means observe again.

    Center depth, object labels, planner prose, and historical outcome are not
    evidence of reaching a particular landmark or entering the wrong room.
    """
    result = {
        'outcome': False, 'failure_type': 'not_reached',
        'reason': 'Verification unavailable; observe and retry.',
        'confidence': 'low', 'evidence': '', 'contradiction_evidence': '',
        'relation_satisfied': False, 'scene_consistent': None,
        'next_start_consistent': None, 'handoff_consistent': None,
        'rollback_required': False,
        'negative_landmarks_detected': [],
        'spatial_clue': '',
        'rejected_target': '',
        'correction_reason': '',
        'correction_action': None,
        'available': False,
        'contract_conflict': False,
        'observation': {},
        'physical_check': '',
        'progress_analysis': '',
        'thought': '',
        'reasoning': {},
        'output': {},
    }
    if isinstance(parsed, dict) and 'output' in parsed and isinstance(parsed['output'], dict):
        merged = {}
        if isinstance(parsed.get('reasoning'), dict):
            merged.update(parsed['reasoning'])
        merged.update(parsed['output'])
        for k, v in parsed.items():
            if k not in ('reasoning', 'output'):
                merged[k] = v
        merged['reasoning'] = parsed['reasoning']
        merged['output'] = parsed['output']
        parsed = merged

    if not isinstance(parsed, dict) or type(parsed.get('outcome')) is not bool:
        return result
    if 'evidence' not in parsed:
        parsed['evidence'] = str(parsed.get('thought', '') or parsed.get('reason', ''))
    if 'confidence' not in parsed:
        parsed['confidence'] = 'high' if parsed.get('outcome') else 'low'
    if 'failure_type' not in parsed:
        parsed['failure_type'] = 'none' if parsed.get('outcome') else 'not_reached'
    required = ('relation_satisfied', 'confidence', 'evidence')
    if any(k not in parsed for k in required) or type(parsed['relation_satisfied']) is not bool:
        return result
    confidence = str(parsed['confidence']).lower()
    if confidence not in ('low', 'medium', 'high'):
        return result
    result.update(available=True, confidence=confidence)
    for k in ('reason', 'evidence', 'contradiction_evidence', 'physical_check', 'progress_analysis', 'thought', 'spatial_clue', 'rejected_target', 'correction_reason'):
        result[k] = str(parsed.get(k, '') or '')[:1200]
    if 'correction_action' in parsed:
        result['correction_action'] = parsed.get('correction_action')
    if isinstance(parsed.get('observation'), dict):
        result['observation'] = {str(ok): str(ov)[:400] for ok, ov in parsed['observation'].items()}
    if isinstance(parsed.get('reasoning'), dict):
        result['reasoning'] = parsed['reasoning']
    if isinstance(parsed.get('output'), dict):
        result['output'] = parsed['output']
    for k in ('scene_consistent', 'next_start_consistent', 'handoff_consistent'):
        result[k] = parsed.get(k) if type(parsed.get(k)) is bool else None
    if result.get('handoff_consistent') is not None and result.get('next_start_consistent') is None:
        result['next_start_consistent'] = result['handoff_consistent']
    elif result.get('next_start_consistent') is not None and result.get('handoff_consistent') is None:
        result['handoff_consistent'] = result['next_start_consistent']
    result['relation_satisfied'] = parsed['relation_satisfied']
    negatives = parsed.get('negative_landmarks_detected', [])
    if isinstance(negatives, list):
        result['negative_landmarks_detected'] = [str(x)[:80] for x in negatives[:8]]
    failure = str(parsed.get('failure_type', '')).lower().replace('-', '_')
    wrong = failure in ('wrong_path', 'wrong_route', 'off_route')
    if wrong and not parsed['outcome'] and not parsed['relation_satisfied']:
        if result['scene_consistent'] is None:
            result['scene_consistent'] = False
        if not result['contradiction_evidence']:
            result['contradiction_evidence'] = str(result.get('reason') or result.get('evidence') or '')[:1200]
    is_handoff_conflict = bool(parsed['outcome'] and parsed.get('handoff_consistent') is False)
    result['contract_conflict'] = bool(
        (parsed['outcome'] and (not parsed['relation_satisfied'] or failure not in ('none', '')
                               or result['scene_consistent'] is False or result['contradiction_evidence']))
        or is_handoff_conflict
        or (wrong and (parsed['outcome'] or parsed['relation_satisfied']))
        or (not parsed['outcome'] and failure == 'none'))
    # Contradictory outputs cannot accept or trigger physical recovery.
    if parsed['outcome'] and parsed['relation_satisfied'] and result['evidence'] and confidence != 'low' and not is_handoff_conflict:
        if failure in ('none', '') and result['scene_consistent'] is not False and not result['contradiction_evidence']:
            result.update(outcome=True, failure_type='none')
    elif (not parsed['outcome'] and not parsed['relation_satisfied']
          and wrong
          and confidence in ('medium', 'high')
          and (result['contradiction_evidence'] or result['reason']) and result['evidence']):
        result.update(failure_type='wrong_path', rollback_required=True, scene_consistent=False)
    if not result['rollback_required']:
        # A seen object in an unfinished/accepted segment is not a negative
        # landmark. Do not let routine not_reached tags become memory bans.
        result['negative_landmarks_detected'] = []
    return result


def checkpoint_yaw(rotation):
    """Mapper camera forward is -R[:, 2], in its z-up coordinate frame."""
    return math.atan2(-float(rotation[1][2]), -float(rotation[0][2]))


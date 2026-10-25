"""Deterministic CPU contracts for explicit endpoint left/right spatial relations."""
import math
import re
import numpy as np

try:
    from src.target_identity import object_category
except ImportError:
    from target_identity import object_category

_ARRIVAL_RE = re.compile(r'\b(?:stop|wait|stand|standing)\b', re.IGNORECASE)
_LEAD_NAV_RE = re.compile(
    r'^\s*(?:(?:walk|go|move|head|proceed|continue|stop|wait|stand|standing|'
    r'next\s+to|in\s+front\s+of|beside|to|at|by|near|in|on|the|a|an)\b\s*)+',
    re.IGNORECASE,
)
_ARTICLE_RE = re.compile(r'^\s*(?:the|a|an)\b\s*', re.IGNORECASE)
_DIR_REL_RE = re.compile(
    r'^(?P<target>\S.+?)\s+(?:(?:to|on|at|by|near|from|toward|towards)\s+)?'
    r'(?:the\s+)?(?P<dir>left|right)(?:[-\s]+(?:hand\s+)?side)?\s+of\s+(?P<landmark>\S.+?)\s*$',
    re.IGNORECASE,
)


def _is_int(val):
    return isinstance(val, int) and not isinstance(val, bool)


def _finite_float(val):
    if isinstance(val, bool) or not isinstance(val, (int, float, np.number)):
        return None
    try:
        out = float(val)
    except (OverflowError, ValueError, TypeError):
        return None
    return out if math.isfinite(out) else None


def _vec3(val):
    if not isinstance(val, (list, tuple, np.ndarray)) or len(val) != 3:
        return None
    coords = [_finite_float(x) for x in val]
    return coords if all(x is not None for x in coords) else None


def parse_endpoint_relation(text) -> dict:
    """Parse explicit object-to-object left_of/right_of relation from an endpoint clause."""
    if not isinstance(text, str) or not text.strip():
        return {
            'required': False, 'relation': None, 'target_text': None, 'target_class': None,
            'landmark_text': None, 'landmark_class': None, 'frame_kind': None,
            'status': 'unknown', 'reason': 'handled_by_existing_geometry',
        }
    clause = re.sub(r'[.;,\s]+$', '', text.strip())
    arrivals = list(_ARRIVAL_RE.finditer(clause))
    if arrivals:
        tail = _LEAD_NAV_RE.sub('', clause[arrivals[-1].end():]).strip()
        if object_category(tail) or _DIR_REL_RE.match(tail):
            clause = tail
    clause = _LEAD_NAV_RE.sub('', clause).strip()
    match = _DIR_REL_RE.match(clause)
    if not match:
        target_clean = _ARTICLE_RE.sub('', clause).strip() or None
        return {
            'required': False, 'relation': None, 'target_text': target_clean,
            'target_class': object_category(target_clean) if target_clean else None,
            'landmark_text': None, 'landmark_class': None, 'frame_kind': None,
            'status': 'unknown', 'reason': 'handled_by_existing_geometry',
        }
    target_text = _ARTICLE_RE.sub('', match.group('target')).strip()
    landmark_text = _ARTICLE_RE.sub('', match.group('landmark')).strip()
    relation = f"{match.group('dir').lower()}_of"
    target_class = object_category(target_text)
    landmark_class = object_category(landmark_text)
    if landmark_class is None:
        reason = 'unsupported_reference_head'
    elif target_class is None:
        reason = 'unsupported_target_head'
    else:
        reason = 'pending_evaluation'
    return {
        'required': True, 'relation': relation,
        'target_text': target_text, 'target_class': target_class,
        'landmark_text': landmark_text, 'landmark_class': landmark_class,
        'frame_kind': 'frozen_observer', 'status': 'unknown', 'reason': reason,
    }


def _validate_packet(pkt, observation_id, expected_class):
    if not isinstance(pkt, dict) or pkt.get('valid') is not True:
        return None, 'invalid_packet'
    if not _is_int(pkt.get('observation_id')) or pkt.get('observation_id') != observation_id:
        return None, 'stale_observation'
    center = _vec3(pkt.get('world_center'))
    if center is None:
        return None, 'invalid_world_center'
    cid = pkt.get('candidate_id')
    if cid is not None and (not isinstance(cid, str) or not cid.strip()):
        return None, 'invalid_candidate_id'
    tid = pkt.get('target_id')
    if tid is not None and not _is_int(tid):
        return None, 'invalid_target_id'
    eid = pkt.get('entity_id')
    if eid is not None and not _is_int(eid):
        return None, 'invalid_target_id'
    if tid is None and eid is not None:
        tid = eid
    is_structural_doorway = bool(
        pkt.get('endpoint_kind') == 'doorway'
        and expected_class == 'door'
        and pkt.get('evidence_role') == 'primary'
    )
    if cid is None and tid is None and not is_structural_doorway:
        return None, 'missing_instance_id'
    if 'expected_object_class' in pkt and pkt['expected_object_class'] is not None:
        exp_val = pkt.get('expected_object_class')
        if not isinstance(exp_val, str) or not exp_val.strip():
            return None, 'category_mismatch'
        if (object_category(exp_val) or exp_val.strip().lower()) != expected_class:
            return None, 'category_mismatch'
    observed_keys = ('selected_detector_class_name', 'detector_class_name', 'category', 'label', 'object_category')
    has_observed_class = False
    for key in observed_keys:
        if key in pkt:
            val = pkt.get(key)
            if val is None and is_structural_doorway and key == 'selected_detector_class_name':
                has_observed_class = True
                continue
            if not isinstance(val, str) or not val.strip():
                return None, 'missing_object_class'
            if object_category(val) != expected_class:
                return None, 'category_mismatch'
            has_observed_class = True
    if not has_observed_class:
        return None, 'missing_object_class'
    return {'center': center, 'cid': cid, 'tid': tid}, None


def evaluate_spatial_relation(spec, target, landmark, frame, *, observation_id, minimum_separation_m=0.15) -> dict:
    """Evaluate target vs landmark horizontal separation in a controller-frozen observer frame."""
    if isinstance(spec, str):
        spec = parse_endpoint_relation(spec)
    base = {
        'required': False, 'relation': None, 'status': 'unknown', 'reason': 'invalid_spec',
        'frame_kind': None, 'frame_observation_id': None,
        'target_candidate_id': None, 'landmark_candidate_ids': [],
    }
    if not isinstance(spec, dict):
        return base
    if not bool(spec.get('required')):
        base['reason'] = str(spec.get('reason') or 'handled_by_existing_geometry')
        return base
    relation = spec.get('relation')
    base.update({'required': True, 'relation': relation, 'frame_kind': 'frozen_observer'})
    if relation not in ('left_of', 'right_of'):
        return {**base, 'reason': 'unsupported_relation'}
    t_class = spec.get('target_class') or (object_category(spec['target_text']) if spec.get('target_text') else None)
    l_class = spec.get('landmark_class') or (object_category(spec['landmark_text']) if spec.get('landmark_text') else None)
    if not l_class:
        return {**base, 'reason': 'unsupported_reference_head'}
    if not t_class:
        return {**base, 'reason': 'unsupported_target_head'}
    min_sep = _finite_float(minimum_separation_m)
    if not _is_int(observation_id) or observation_id < 0 or min_sep is None or min_sep < 0.0:
        return {**base, 'reason': 'invalid_evaluation_parameters'}
    if not isinstance(frame, dict) or frame.get('kind') != 'frozen_observer' or frame.get('source') != 'measured_camera':
        return {**base, 'reason': 'invalid_reference_frame'}
    f_obs = frame.get('observation_id')
    origin = _vec3(frame.get('origin'))
    r_axis = _vec3(frame.get('right_axis'))
    if not _is_int(f_obs) or f_obs < 0 or origin is None or r_axis is None or abs(r_axis[2]) > 1e-2:
        return {**base, 'reason': 'invalid_reference_frame'}
    norm = math.hypot(r_axis[0], r_axis[1])
    if not (0.5 <= norm <= 1.5):
        return {**base, 'frame_observation_id': int(f_obs), 'reason': 'invalid_frame_axis'}
    base['frame_observation_id'] = int(f_obs)
    ref_cls = frame.get('reference_class_name')
    if ref_cls is not None:
        if not isinstance(ref_cls, str) or not ref_cls.strip() or object_category(ref_cls) != l_class:
            return {**base, 'reason': 'reference_frame_class_mismatch'}
    t_info, t_err = _validate_packet(target, observation_id, t_class)
    l_info, l_err = _validate_packet(landmark, observation_id, l_class)
    if t_info is not None:
        base['target_candidate_id'] = t_info['cid']
    if l_info is not None and l_info['cid'] is not None:
        base['landmark_candidate_ids'] = [l_info['cid']]
    if t_err or l_err:
        return {**base, 'reason': t_err or l_err}
    if t_info['cid'] is not None and t_info['cid'] == l_info['cid']:
        return {**base, 'reason': 'same_candidate_id'}
    if t_info['tid'] is not None and t_info['tid'] == l_info['tid']:
        return {**base, 'reason': 'same_target_entity'}
    ux, uy = r_axis[0] / norm, r_axis[1] / norm
    dx = t_info['center'][0] - l_info['center'][0]
    dy = t_info['center'][1] - l_info['center'][1]
    sep = _finite_float(dx * ux + dy * uy)
    if sep is None:
        return {**base, 'reason': 'invalid_separation'}
    base['signed_separation_m'] = float(sep)
    if abs(sep) < min_sep:
        return {**base, 'reason': 'insufficient_separation'}
    satisfied = (sep < 0.0) if relation == 'left_of' else (sep > 0.0)
    return {
        **base,
        'status': 'satisfied' if satisfied else 'contradicted',
        'reason': f'{relation}_satisfied' if satisfied else 'position_relation_contradicted',
    }


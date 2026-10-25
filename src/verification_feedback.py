"""Bounded verifier diagnostics; these never establish navigation evidence."""
import math
from numbers import Real


def endpoint_relocalization_required(parsed, result, config=None):
    """An affirmative, route-complete claim can merit a fresh localization.

    This authorizes observation only. Missing identity/coordinates, negative
    probes, far targets and failed route evidence are not localization repairs.
    A near candidate with explicitly rejected endpoint geometry merits bounded
    re-localization, never permission to stop or to resume the old approach.
    """
    if (not isinstance(parsed, dict) or parsed.get('outcome') is not True
            or parsed.get('scene_consistent') is not True
            or parsed.get('relation_satisfied') is not True
            or parsed.get('failure_type') not in (None, 'none')
            or result.get('available') is not True or result.get('outcome') is not False
            or result.get('contract_conflict') or result.get('condition_contract_conflict')
            or result.get('rollback_required')):
        return False
    checks = result.get('condition_checks') or []
    if not checks or any(c.get('status') != 'satisfied'
                         or not (c.get('execution_checked') or c.get('observation_checked')) for c in checks):
        return False
    if any(not c.get('execution_witnessed') for c in result.get('doorway_checks') or []):
        return False
    packet = result.get('endpoint_grounding')
    if not isinstance(packet, dict):
        return False
    regions = [(packet, {'view': parsed.get('target_view'), 'bbox': parsed.get('target_bbox')})]
    reference = result.get('endpoint_reference_grounding')
    if isinstance(reference, dict):
        regions.append((reference, parsed.get('reference_region') or {}))
    cfg = config or {}
    kind = packet.get('endpoint_kind')
    limit = float(cfg.get('endpoint_door_frame_max_distance_m', 2.0) if kind == 'doorway' else
                  cfg.get('endpoint_between_max_distance_m', 2.0) if kind == 'between' else
                  cfg.get('endpoint_max_distance_m', 1.5))
    failures = {'mixed_target_depth', 'insufficient_valid_target_depth',
                'too_few_target_pixels', 'insufficient_door_frame_support'}
    def explicit_region(claimed):
        box = claimed.get('bbox') if isinstance(claimed, dict) else None
        return bool(isinstance(box, (list, tuple)) and len(box) == 4
                    and all(isinstance(v, Real) and not isinstance(v, bool)
                            and math.isfinite(v) and 0 <= v <= 1 for v in box)
                    and box[0] < box[2] and box[1] < box[3]
                    and claimed.get('view') in ('front', 'left', 'right', 'back'))

    def localized_near(measured):
        center = measured.get('world_center') if isinstance(measured, dict) else None
        return bool(isinstance(measured, dict) and measured.get('valid') is True
                    and isinstance(measured.get('distance_m'), Real)
                    and not isinstance(measured.get('distance_m'), bool)
                    and math.isfinite(measured['distance_m']) and 0 <= measured['distance_m'] <= limit
                    and isinstance(center, (list, tuple)) and len(center) == 3
                    and all(isinstance(v, Real) and not isinstance(v, bool) and math.isfinite(v) for v in center))

    # This gate reason is emitted only after freshness and relation checks have
    # passed. Require explicit current, near regions; missing geometry is not a
    # measured veto. An old/missing reference cannot support this classification.
    observation_id = packet.get('observation_id')
    geometry_failure = bool(
        packet.get('geometry_verified') is False
        and packet.get('gate_reason') == 'measured_endpoint_geometry_required'
        and type(observation_id) is int and observation_id >= 0
        and kind in ('doorway', 'between', 'stair_landing')
        and (kind != 'between' or isinstance(reference, dict))
        and (kind != 'doorway' or isinstance(reference, dict)
             or all(localized_near((packet.get('frame_support') or {}).get(side)) for side in ('left', 'right')))
        and all(localized_near(measured)
                and measured.get('source') in ('detector_mask', 'vlm_bbox')
                and measured.get('observation_id') == observation_id
                and explicit_region(claimed)
                and measured.get('view') == claimed.get('view')
                for measured, claimed in regions))
    repairable = False
    for measured, claimed in regions:
        # A known far surface requires an approach, even if another region is
        # poorly localized. Never replace that decision with repeated scanning.
        parts = [measured, *(measured.get('frame_support') or {}).values()]
        for part in parts:
            distance = part.get('distance_m') if isinstance(part, dict) else None
            if (isinstance(part, dict) and part.get('valid') is True
                    and isinstance(distance, Real) and math.isfinite(distance) and distance > limit):
                return False
        if not (measured is packet and geometry_failure) and (measured.get('valid') or measured.get('reason') not in failures):
            continue
        repairable |= explicit_region(claimed)
    return repairable


def evidence_repair_required(parsed, result):
    """Separate an affirmative claim's invalid citations from a negative verdict.

    A retry may repair references, but still passes through the complete verifier
    gate. Geometry, scene contradictions and actual not-reached verdicts keep
    their normal observation/movement/recovery path.
    """
    raw_checks = parsed.get('condition_checks') if isinstance(parsed, dict) else None
    model_self_reported_unmet = bool(
        isinstance(raw_checks, list) and raw_checks and any(
            isinstance(c, dict)
            and str(c.get('status') or '').strip().lower() in (
                'unknown', 'contradicted', 'pending', 'incomplete', 'uncertain', 'unverified',
                'failed', 'not_satisfied', 'unmet', 'false', 'no', 'violated', 'rejected'
            )
            and not (c.get('steps') or [])
            and c.get('observation_id') is None
            for c in raw_checks
        )
    )
    return bool(
        isinstance(parsed, dict) and parsed.get('outcome') is True
        and not model_self_reported_unmet
        and result.get('available') is True and result.get('outcome') is False
        and not result.get('contract_conflict') and not result.get('rollback_required')
        and parsed.get('failure_type') not in ('wrong_path', 'not_reached')
        and not any(isinstance(row, dict) and row.get('status') == 'contradicted'
                    for row in result.get('condition_checks') or [])
        and result.get('reason') in (
            'Route condition coverage or execution references are invalid.',
            'Every route condition requires satisfied execution evidence.')
    )


def initial_arrival_observation_retry_required(first, second, current_observation_id):
    """A same-frame citation repair flip may buy one look, never an arrival.

    The caller bounds this to one stationary observation and rechecks before any
    movement. No history exists to justify a physical wrong-route correction.
    """
    return bool(
        type(current_observation_id) is int and current_observation_id >= 0
        and first.get('initial_arrival_observation_eligible') is True
        and first.get('evidence_repair_required') is True
        and first.get('current_observation_id') == current_observation_id
        and second.get('current_observation_id') == current_observation_id
        and second.get('available') is True and second.get('outcome') is False
        and not second.get('contract_conflict') and not second.get('rollback_required')
        and not (second.get('route_contradiction_gate') or {}).get('accepted')
        and not any(c.get('status') == 'contradicted' for c in second.get('condition_checks') or [])
        and not second.get('endpoint_grounding'))


def landing_observation_pending(parsed, result, config=None):
    """Allow bounded stationary sampling of a measured landing, never a STOP.

    Stable support is counted only by StairProgress.observe after execution;
    this predicate and repeated verifier queries cannot add observations.
    """
    if (not isinstance(parsed, dict) or parsed.get('outcome') is not True
            or parsed.get('relation_satisfied') is not True or parsed.get('scene_consistent') is not True
            or parsed.get('failure_type') not in (None, 'none')
            or result.get('available') is not True or result.get('outcome') is not False
            or result.get('stair_completion_ready') is not False
            or result.get('landing_support_current') is not True
            or result.get('contract_conflict') or result.get('condition_contract_conflict')
            or result.get('rollback_required')
            or result.get('reason') != 'Measured stair direction or supported landing is not yet established.'):
        return False
    checks = result.get('condition_checks') or []
    if (not checks or not any(c.get('requires_arrival') for c in checks)
            or any(c.get('status') != 'satisfied'
                   or not (c.get('execution_checked') is True or c.get('observation_checked') is True)
                   for c in checks)):
        return False
    progress = result.get('stair_progress') or {}
    if (progress.get('landing_required') is not True or progress.get('direction_violation') is not False
            or progress.get('phase') not in ('traverse', 'landing')):
        return False
    cfg = config or {}
    try:
        minimum = max(.25, float(cfg.get('stair_min_progress_m', .25)))
        required = int(cfg.get('stair_landing_observations', 3))
    except (TypeError, ValueError, OverflowError):
        return False
    count, height = progress.get('stable_supported_observations'), progress.get('measured_height_change_m')
    direction = {'UP': 1, 'DOWN': -1}.get(progress.get('direction'), 0)
    return bool(type(count) is int and 1 <= count < required
                and isinstance(height, Real) and not isinstance(height, bool) and math.isfinite(height)
                and math.isfinite(minimum) and direction * height >= minimum)


def remember_verification(history, key, result, *, trigger, step, pose):
    """Keep the latest two normalized results shared by probes and proposals."""
    def measured_region(packet):
        if not isinstance(packet, dict):
            return None
        fields = ('valid', 'reason', 'accepted', 'gate_reason', 'source', 'view',
                  'bbox', 'target_id', 'endpoint_kind', 'distance_m', 'depth_m',
                  'valid_fraction', 'geometry_verified', 'confirmed_changed_view',
                  'opening_depth_contrast_m', 'opening_traversable')
        fields += ('selected_detector_class_name', 'expected_object_class', 'evidence_role')
        region = {name: packet[name] for name in fields if name in packet}
        supports = packet.get('frame_support')
        if isinstance(supports, dict):
            region['frame_support'] = {
                side: {name: part[name] for name in ('valid', 'reason', 'distance_m', 'depth_m')
                       if name in part}
                for side, part in supports.items() if side in ('left', 'right') and isinstance(part, dict)}
        return region

    checks = []
    for row in result.get('condition_checks') or []:
        if not isinstance(row, dict):
            continue
        checks.append({
            'id': row.get('id'), 'clause': str(row.get('text') or '')[:160],
            'status': row.get('status'), 'execution_checked': row.get('execution_checked') is True,
            'observation_checked': row.get('observation_checked') is True,
            'observation_id': row.get('observation_id'),
            'steps': list(row.get('steps') or [])[:32],
            'diagnostic': str(row.get('evidence') or '')[:300],
        })
    row = {'trigger': trigger, 'step': step, 'pose': pose,
           'current_observation_id': result.get('current_observation_id'),
           'accepted': result.get('completion_evidence_reliable') is True,
           'reason': str(result.get('reason') or '')[:400],
           'evidence_repair_required': result.get('evidence_repair_required') is True,
           'condition_checks': checks}
    measurements = {name: measured_region(result.get(name)) for name in
                    ('endpoint_grounding', 'endpoint_reference_grounding')}
    row.update({name: packet for name, packet in measurements.items() if packet is not None})
    row['endpoint_confirmation_required'] = result.get('endpoint_confirmation_required') is True
    row['endpoint_relocalization_required'] = result.get('endpoint_relocalization_required') is True
    row['landing_observation_pending'] = result.get('landing_observation_pending') is True
    history[key] = [*history.get(key, [])[-1:], row]
    return history


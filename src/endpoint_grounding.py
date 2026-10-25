"""Current-view target depth evidence, independent of models and simulator goals.

Depth is unnormalised camera Z depth in metres. K must describe the *depth*
image. Rotation maps camera coordinates to the mapper's Z-up world frame,
using mapping_utils.geometry's convention [x, y_up, -z_forward]. None of the
functions substitute image-centre depth for a missing target localisation.

This module checks geometry, not whether an identified door is a bathroom
door. ``geometry_verified`` and ``confirmed`` are controller evidence; never
copy either flag from a language-model response.
"""

from collections.abc import Mapping
from numbers import Real

import numpy as np


VIEWS = frozenset(('front', 'left', 'right', 'back'))
ENDPOINT_KINDS = frozenset(('near_object', 'doorway', 'stair_landing', 'between'))


def _number(value):
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and np.isfinite(value)


def _index(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and value >= 0


def _array(value, shape=None):
    try:
        arr = np.asarray(value)
        if arr.dtype.kind not in 'iuf' or (shape is not None and arr.shape != shape):
            return None
        arr = arr.astype(float, copy=False)
        return arr if np.isfinite(arr).all() else None
    except (TypeError, ValueError, OverflowError):
        return None


def _invalid(packet, reason):
    return dict(packet, valid=False, reason=reason)


def _bbox(value, height, width, max_area=.75):
    if not isinstance(value, (list, tuple, np.ndarray)):
        return None
    try:
        if len(value) != 4 or not all(_number(v) for v in value):
            return None
    except TypeError:
        return None
    box = _array(value, (4,))
    if box is None or np.any(box < 0) or np.any(box > 1):
        return None
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return None
    # Bound the localisation itself, rather than accepting a whole-view box.
    if (x2-x1)*(y2-y1) > max_area or (x2-x1)*width < 3 or (y2-y1)*height < 3:
        return None
    pixels = (int(np.floor(x1*width)), int(np.floor(y1*height)),
              int(np.ceil(x2*width)), int(np.ceil(y2*height)))
    return box.tolist(), pixels


def _region_stats(depth, selected, intrinsic, rotation, position, *, min_pixels, min_valid_fraction,
                  allow_depth_extent=False, foreground_jamb=False):
    count = int(np.count_nonzero(selected))
    if count < min_pixels:
        return {'valid': False, 'reason': 'too_few_target_pixels', 'count': count, 'valid_fraction': 0.0}
    valid = selected & np.isfinite(depth) & (depth > .15) & (depth < 50.0)
    valid_count = int(np.count_nonzero(valid))
    fraction = valid_count / count
    if valid_count < min_pixels or fraction < min_valid_fraction:
        return {'valid': False, 'reason': 'insufficient_valid_target_depth', 'count': valid_count,
                'valid_fraction': float(fraction)}
    rows, cols = np.nonzero(valid)
    z = depth[rows, cols]
    camera_points = np.column_stack(((cols-intrinsic[0, 2])*z/intrinsic[0, 0],
                                    (depth.shape[0]-1-rows-intrinsic[1, 2])*z/intrinsic[1, 1], -z))
    offsets = camera_points @ rotation.T
    distances = np.linalg.norm(offsets[:, :2], axis=1)
    near, median, far = np.quantile(distances, [.2, .5, .8])
    depth_near, depth_median, depth_far = np.quantile(z, [.2, .5, .8])
    # A tight target region should not mostly straddle unrelated foreground and
    # background surfaces. Door openings are sampled separately below.
    depth_extent = depth_far-depth_near > max(.60, .50*depth_median)
    mixed = depth_extent and not allow_depth_extent
    jamb_opening_contrast = None
    if mixed and foreground_jamb and float(depth_near) <= 1.60:
        fg_cutoff = float(depth_near) + max(0.45, 0.35 * float(depth_near))
        fg_valid = valid & (depth <= fg_cutoff)
        fg_count = int(np.count_nonzero(fg_valid))
        if fg_count >= min_pixels:
            jamb_opening_contrast = float(depth_far - depth_near)
            rows, cols = np.nonzero(fg_valid)
            z = depth[rows, cols]
            camera_points = np.column_stack(((cols-intrinsic[0, 2])*z/intrinsic[0, 0],
                                            (depth.shape[0]-1-rows-intrinsic[1, 2])*z/intrinsic[1, 1], -z))
            offsets = camera_points @ rotation.T
            distances = np.linalg.norm(offsets[:, :2], axis=1)
            near, median, far = np.quantile(distances, [.2, .5, .8])
            depth_near, depth_median, depth_far = np.quantile(z, [.2, .5, .8])
            valid_count = fg_count
            mixed = False
    result = {'valid': not bool(mixed), 'reason': 'mixed_target_depth' if mixed else 'ok',
              'count': valid_count, 'valid_fraction': float(fraction),
              'depth_m': float(depth_median), 'depth_near_m': float(depth_near),
              'depth_far_m': float(depth_far), 'distance_m': float(near),
              'median_distance_m': float(median), 'far_distance_m': float(far),
              'depth_extent_m': float(depth_far-depth_near)}
    if jamb_opening_contrast is not None:
        result['jamb_opening_contrast_m'] = jamb_opening_contrast
    if position is not None:
        result['world_center'] = np.median(offsets+position, axis=0).tolist()
    return result


def ground_target_depth(depth_views, *, target_view, target_bbox=None, target_mask=None,
                        endpoint_kind='near_object', observation_id=None, observation_step=None,
                        camera_intrinsic=None, camera_rotation=None, camera_position=None,
                        mask_view=None, mask_observation_id=None, mask_confidence=None,
                        target_id=None, min_valid_fraction=.60, min_pixels=16,
                        foreground_jamb=False):
    """Measure a localized target in one explicitly named current view.

    ``depth_views`` maps view names to HxW (or HxWx1) depth arrays. bbox uses
    normalized [x1,y1,x2,y2] with exclusive right/bottom boundaries. A detector
    mask must be boolean, same-sized and tagged with its actual view and current
    observation ID; optional bbox must overlap it. An invalid supplied mask is
    rejected, never silently replaced by a VLM bbox or centre sample.

    ``distance_m`` is a robust (20th percentile) *horizontal* surface distance,
    computed after K backprojection and camera-to-world rotation. ``depth_m``
    is the median optical depth for diagnostics. A doorway samples the two
    vertical border bands independently; its centre background is only an
    opening diagnostic. Doorway arrival still needs controller threshold-plane
    geometry. A closed door approached from outside uses ``near_object``.
    """
    packet = {'valid': False, 'reason': 'unmeasured', 'source': 'detector_mask' if target_mask is not None else 'vlm_bbox',
              'view': target_view if isinstance(target_view, str) else None,
              'endpoint_kind': endpoint_kind if isinstance(endpoint_kind, str) else None,
              'observation_id': int(observation_id) if _index(observation_id) else None,
              'observation_step': int(observation_step) if _index(observation_step) else None,
              'target_id': target_id if isinstance(target_id, (str, int)) and not isinstance(target_id, bool) else None}
    if not isinstance(target_view, str) or target_view not in VIEWS:
        return _invalid(packet, 'explicit_target_view_required')
    if not isinstance(endpoint_kind, str) or endpoint_kind not in ENDPOINT_KINDS:
        return _invalid(packet, 'unsupported_endpoint_kind')
    if not _index(observation_id):
        return _invalid(packet, 'current_observation_id_required')
    if (not _number(min_valid_fraction) or not 0 < min_valid_fraction <= 1
            or not _index(min_pixels) or min_pixels < 4):
        return _invalid(packet, 'invalid_sampling_policy')
    if not isinstance(depth_views, Mapping) or target_view not in depth_views:
        return _invalid(packet, 'target_view_depth_missing')
    try:
        depth = np.asarray(depth_views[target_view])
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[:, :, 0]
        if depth.ndim != 2 or depth.dtype.kind not in 'iuf' or min(depth.shape) < 3:
            return _invalid(packet, 'invalid_depth_image')
        depth = depth.astype(float, copy=False)
    except (TypeError, ValueError):
        return _invalid(packet, 'invalid_depth_image')
    h, w = depth.shape
    k, rotation = _array(camera_intrinsic, (3, 3)), _array(camera_rotation, (3, 3))
    if (k is None or rotation is None or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1]) or abs(k[0, 1])+abs(k[1, 0]) > 1e-6
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3)
            # habitat_rotation also swaps mapper axes, so its orthogonal
            # camera-to-world transform has determinant -1, not necessarily +1.
            or not np.isclose(abs(np.linalg.det(rotation)), 1, atol=1e-3)):
        return _invalid(packet, 'valid_camera_calibration_required')
    position = _array(camera_position, (3,)) if camera_position is not None else None
    if camera_position is not None and position is None:
        return _invalid(packet, 'invalid_camera_position')
    # A close segmented bed/counter/stair can legitimately fill the image.
    # Keep the stricter area limit for an unsegmented language-model rectangle,
    # while allowing a close doorway threshold frame to span up to 0.92 of the view.
    unsegmented_max_area = .92 if (endpoint_kind == 'doorway' or foreground_jamb) else .75
    bbox = _bbox(target_bbox, h, w, max_area=1.0 if target_mask is not None else unsegmented_max_area) if target_bbox is not None else None
    if target_bbox is not None and bbox is None:
        return _invalid(packet, 'invalid_target_bbox')
    mask = None
    if target_mask is not None:
        if mask_view != target_view or not _index(mask_observation_id) or mask_observation_id != observation_id:
            return _invalid(packet, 'mask_not_from_current_target_view')
        try:
            mask = np.asarray(target_mask)
        except (TypeError, ValueError):
            return _invalid(packet, 'invalid_target_mask')
        if mask.shape != depth.shape or mask.dtype.kind != 'b' or np.count_nonzero(mask) < min_pixels:
            return _invalid(packet, 'invalid_target_mask')
        if not _number(mask_confidence) or not 0 <= mask_confidence <= 1:
            return _invalid(packet, 'mask_confidence_required')
        packet['confidence'] = float(mask_confidence)
        ys, xs = np.nonzero(mask)
        mask_bbox = _bbox([xs.min()/w, ys.min()/h, (xs.max()+1)/w, (ys.max()+1)/h], h, w, max_area=1.0)
        if mask_bbox is None:
            return _invalid(packet, 'invalid_mask_extent')
        if bbox is None:
            bbox = mask_bbox
        else:
            x1, y1, x2, y2 = bbox[1]
            overlap = np.count_nonzero(mask[y1:y2, x1:x2])
            mask_count = np.count_nonzero(mask)
            union = mask_count + (x2-x1)*(y2-y1) - overlap
            # A requested end/edge/first tread may be a proper subset of the
            # segmented instance. Require overlap with that localized subset.
            if overlap/max(1, min(mask_count, (x2-x1)*(y2-y1))) < .6:
                return _invalid(packet, 'mask_bbox_mismatch')
    if bbox is None:
        return _invalid(packet, 'explicit_target_localisation_required')
    packet.update(bbox=bbox[0], image_size=[w, h])
    x1, y1, x2, y2 = bbox[1]
    selected = np.zeros(depth.shape, dtype=bool)
    selected[y1:y2, x1:x2] = True
    if mask is not None:
        selected &= mask
    kwargs = dict(min_pixels=int(min_pixels), min_valid_fraction=float(min_valid_fraction))
    if endpoint_kind == 'doorway':
        band = max(1, int(np.ceil((x2-x1)*.18)))
        ystart, yend = y1+int((y2-y1)*.15), y2-int((y2-y1)*.15)
        supports = {}
        for side, lo, hi in [('left', x1, x1+band), ('right', x2-band, x2)]:
            strip = np.zeros(depth.shape, dtype=bool)
            strip[ystart:yend, lo:hi] = True
            supports[side] = _region_stats(depth, strip & selected, k, rotation, position,
                                           foreground_jamb=True, **kwargs)
        packet.update(sampling='door_frame_borders', frame_support=supports)
        if not all(s['valid'] for s in supports.values()):
            return _invalid(packet, 'insufficient_door_frame_support')
        # Never mix background seen THROUGH the door into its frame distance.
        interior = np.zeros(depth.shape, dtype=bool)
        interior[ystart:yend, x1+band:x2-band] = True
        opening = _region_stats(depth, interior, k, rotation, position, **kwargs)
        packet['opening_depth_m'] = opening.get('depth_m')
        packet['opening_depth_contrast_m'] = (
            opening['depth_m']-max(s['depth_m'] for s in supports.values())
            if opening.get('valid') else None)
        packet.update(valid=True, reason='ok', distance_m=max(s['distance_m'] for s in supports.values()),
                      depth_m=float(np.median([s['depth_m'] for s in supports.values()])),
                      valid_fraction=min(s['valid_fraction'] for s in supports.values()))
        if position is not None:
            packet['world_center'] = np.mean([s['world_center'] for s in supports.values()], axis=0).tolist()
        return packet
    packet.update(sampling='target_mask' if mask is not None else 'target_bbox')
    # The segmented object's real depth extent is not foreground/background
    # contamination. Robust surface distance still excludes isolated outliers.
    return dict(packet, **_region_stats(depth, selected, k, rotation, position,
                                       allow_depth_extent=mask is not None,
                                       foreground_jamb=bool(foreground_jamb), **kwargs))


def evaluate_endpoint_grounding(packet, *, current_observation_id, max_age_observations=0,
                                max_distance_m=1.2, relation_satisfied=False,
                                geometry_verified=False, confirmed=False, min_mask_confidence=.55):
    """Return a geometry gate verdict, not an overall navigation completion.

    Route conditions and target identity remain the caller's responsibility.
    VLM bbox evidence needs independent, changed-view localisation confirmation.
    Opening/platform/between relations need separately measured controller
    geometry, not a language-model assertion or merely a nearby surface.
    """
    reason = None
    if not isinstance(packet, Mapping) or packet.get('valid') is not True:
        reason = 'target_depth_not_grounded'
    elif not _index(current_observation_id) or not _index(max_age_observations):
        reason = 'invalid_freshness_policy'
    elif (not _index(packet.get('observation_id')) or
          not 0 <= current_observation_id-packet['observation_id'] <= max_age_observations):
        reason = 'stale_or_future_target_observation'
    elif not isinstance(packet.get('source'), str) or packet['source'] not in ('detector_mask', 'vlm_bbox'):
        reason = 'unsupported_target_evidence_source'
    elif (not isinstance(packet.get('view'), str) or packet['view'] not in VIEWS
          or not isinstance(packet.get('endpoint_kind'), str) or packet['endpoint_kind'] not in ENDPOINT_KINDS):
        reason = 'invalid_target_view_or_kind'
    elif (not _number(max_distance_m) or max_distance_m <= 0
          or not _number(packet.get('distance_m')) or packet['distance_m'] < 0):
        reason = 'invalid_target_distance'
    elif packet['distance_m'] > max_distance_m:
        reason = 'target_surface_too_far'
    elif relation_satisfied is not True:
        reason = 'target_relation_not_verified'
    elif packet['source'] == 'detector_mask' and (
            not _number(min_mask_confidence) or not 0 <= min_mask_confidence <= 1
            or not _number(packet.get('confidence')) or not min_mask_confidence <= packet['confidence'] <= 1):
        reason = 'target_mask_confidence_too_low'
    elif packet['endpoint_kind'] != 'near_object' and geometry_verified is not True:
        reason = 'measured_endpoint_geometry_required'
    elif packet['source'] == 'vlm_bbox' and confirmed is not True:
        reason = 'changed_view_confirmation_required'
    return {'accepted': reason is None, 'reason': reason or 'ok'}


def consistent_target_observations(first, second, *, first_position, second_position,
                                   first_rotation, second_rotation, max_center_shift_m=.75,
                                   min_translation_m=.15, min_turn_degrees=15.0):
    """Check distinct, changed-pose observations of the same localized region.

    This is geometric corroboration, not proof of semantic identity. Caller
    must retain the same target hypothesis. Missing IDs are allowed; explicit
    differing IDs, reused frames, no camera change or displaced 3D targets fail.
    Camera poses must match each packet's observation, not robot base headings.
    """
    result = {'consistent': False, 'reason': 'invalid_confirmation_observation'}
    if not all(isinstance(p, Mapping) and p.get('valid') is True for p in (first, second)):
        return result
    if not all(_index(p.get('observation_id')) for p in (first, second)) or second['observation_id'] <= first['observation_id']:
        return dict(result, reason='new_observation_required')
    if first.get('endpoint_kind') != second.get('endpoint_kind'):
        return dict(result, reason='endpoint_kind_changed')
    a, b = first.get('target_id'), second.get('target_id')
    if a is not None and b is not None and str(a) != str(b):
        return dict(result, reason='target_instance_changed')
    p1, p2 = _array(first_position, (3,)), _array(second_position, (3,))
    r1, r2 = _array(first_rotation, (3, 3)), _array(second_rotation, (3, 3))
    c1, c2 = _array(first.get('world_center'), (3,)), _array(second.get('world_center'), (3,))
    if any(v is None for v in (p1, p2, r1, r2, c1, c2)):
        return dict(result, reason='camera_pose_and_world_target_required')
    if (not all(_number(v) for v in (max_center_shift_m, min_translation_m, min_turn_degrees))
            or max_center_shift_m <= 0 or min_translation_m <= 0 or not 0 < min_turn_degrees <= 180
            or not all(np.allclose(r @ r.T, np.eye(3), atol=1e-3) and np.isclose(abs(np.linalg.det(r)), 1, atol=1e-3)
                       for r in (r1, r2))):
        return dict(result, reason='invalid_confirmation_geometry')
    if np.linalg.det(r1)*np.linalg.det(r2) < 0:
        return dict(result, reason='inconsistent_camera_coordinate_convention')
    turn = np.degrees(np.arccos(np.clip((np.trace(r2 @ r1.T)-1)/2, -1, 1)))
    moved = np.linalg.norm(p2-p1)
    if moved < min_translation_m and turn < min_turn_degrees:
        return dict(result, reason='changed_camera_pose_required')
    shift = float(np.linalg.norm(c2-c1))
    if shift > max_center_shift_m:
        return dict(result, reason='localized_target_moved', center_shift_m=shift)
    return {'consistent': True, 'reason': 'ok', 'center_shift_m': shift,
            'camera_translation_m': float(moved), 'camera_turn_degrees': float(turn)}


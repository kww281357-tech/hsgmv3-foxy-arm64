"""Bind endpoint claims to fresh image regions and measured local geometry.

No evaluator goal, reference trajectory or planner waypoint is an endpoint witness.
The verifier identifies the region/relation; this module measures that region.
"""
import math
import re

import numpy as np

from endpoint_grounding import ground_target_depth, evaluate_endpoint_grounding
from navigation_control import landing_support
from target_identity import detector_matches_endpoint, endpoint_object_roles, is_between_endpoint
try:
    from agent.visual_grounding import VisualGroundingMixin
except ImportError:
    from visual_grounding import VisualGroundingMixin


def endpoint_kind(text):
    text = str(text).lower()
    if is_between_endpoint(text):
        return 'between'
    if re.search(r'\b(stair|stairs|staircase|landing)\b|\b(?:top|bottom|foot|base)\s+of\s+(?:the\s+|a\s+|an\s+)?steps?\b', text):
        return 'stair_landing'
    if re.search(r'\b(beside|next to|near|in front of)\b.{0,45}\b(doorway|archway|door|opening)\b', text):
        return 'near_object'
    if re.search(r'\b(doorway|archway|threshold|opening)\b', text):
        return 'doorway'
    if re.search(r'\b(through|into|out of|after crossing)\b.*\bdoor\b', text):
        return 'doorway'
    return 'near_object'


def doorway_position(packet, position, max_plane_distance=.75):
    """A threshold needs two supported jambs; open-space depth is not its plane."""
    frame = packet.get('frame_support') or {}
    try:
        left = np.asarray(frame['left']['world_center'], dtype=float)[:2]
        right = np.asarray(frame['right']['world_center'], dtype=float)[:2]
        p = np.asarray(position, dtype=float)[:2]
        span = right - left
        width = float(np.linalg.norm(span))
        if not .45 <= width <= 3.0:
            return False
        t = float(np.dot(p-left, span) / (width * width))
        distance = float(np.linalg.norm(p - (left + t * span)))
        contrast = packet.get('opening_depth_contrast_m')
        opening = (contrast is not None and float(contrast) >= .25) or packet.get('opening_traversable') is True
        return bool(-.10 <= t <= 1.10 and distance <= max_plane_distance and opening)
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def between_position(first, second, position, max_distance=2.0):
    try:
        if first.get('target_id') is not None and first.get('target_id') == second.get('target_id'):
            return False
        a = np.asarray(first['world_center'], dtype=float)[:2]
        b = np.asarray(second['world_center'], dtype=float)[:2]
        p = np.asarray(position, dtype=float)[:2]
        ab = b-a
        length = float(np.linalg.norm(ab))
        if length < .5 or not np.isfinite([*a, *b, *p]).all():
            return False
        t = float(np.dot(p-a, ab) / (length * length))
        off_axis = float(np.linalg.norm(p - (a + t*ab)))
        return bool(0 < t < 1 and off_axis <= .8
                    and max(first['distance_m'], second['distance_m']) <= max_distance)
    except (KeyError, TypeError, ValueError):
        return False


class EndpointGroundingGuard(VisualGroundingMixin):
    def _observed_portal_traversable(self, first, second):
        """Check only the short passage between observed jambs, using the existing
        simulator-geometry route backend. No endpoint/goal metadata is read.
        """
        routes = getattr(self, 'waypoint_routes', None)
        if routes is None:
            return False
        try:
            a, b = np.asarray(first['world_center']), np.asarray(second['world_center'])
            span = b[:2]-a[:2]
            width = float(np.linalg.norm(span))
            if not .45 <= width <= 3.0:
                return False
            pose = np.asarray(self.mapper.current_position).copy()
            ground = pose + [0, 0, self.mapper.initial_floor_height]
            t = float(np.dot(pose[:2]-a[:2], span)/(width*width))
            normal = np.array([-span[1], span[0]])/width
            centers = [a[:2] + t*span, a[:2] + 0.5*span, ground[:2]]
            axes = [normal]
            rot = getattr(self.mapper, 'current_rotation', None)
            if rot is not None:
                try:
                    fwd_3d = np.asarray(rot, dtype=float) @ np.array([0.0, 0.0, -1.0], dtype=float)
                    fnorm = float(np.linalg.norm(fwd_3d[:2]))
                    if np.isfinite(fnorm) and fnorm > 1e-6:
                        axes.append(fwd_3d[:2] / fnorm)
                except (TypeError, ValueError):
                    pass
            for c_xy in centers:
                for axis_xy in axes:
                    plane = ground.copy()
                    plane[:2] = c_xy
                    ends = []
                    ok = True
                    for sign in (-1, 1):
                        point = plane.copy()
                        point[:2] += sign*.35*axis_xy
                        start, end = routes.to_habitat(ground), routes.to_habitat(point)
                        if not routes.pf.is_navigable(end) or not routes.segment(start, end):
                            ok = False
                            break
                        ends.append(end)
                    if ok and len(ends) == 2 and bool(routes.segment(ends[0], ends[1])):
                        return True
            return False
        except (AttributeError, TypeError, ValueError, KeyError):
            return False

    def _endpoint_approach_action(self, verification, actions):
        packet = verification.get('endpoint_grounding') or {}
        if not packet.get('valid') or packet.get('gate_reason') != 'target_surface_too_far':
            return None
        target = np.asarray(packet.get('world_center', []), dtype=float)
        pose = np.asarray(self.mapper.current_position, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            return None
        current_distance = float(np.linalg.norm(target[:2]-pose[:2]))
        candidates = []
        for name, action in (actions or {}).items():
            if action.get('type') != 'waypoint' or action.get('rejected'):
                continue
            point = np.asarray(action.get('target_point_world', []), dtype=float)
            if point.shape != (3,) or not np.isfinite(point).all():
                continue
            distance = float(np.linalg.norm(point[:2]-target[:2]))
            ground_z = pose[2] + float(self.mapper.initial_floor_height)
            if (distance < current_distance-.15 and distance >= .35
                    and abs(point[2]-ground_z) <= .35):
                candidates.append((distance, str(name), action))
        if not candidates:
            return None
        _, name, action = min(candidates, key=lambda row: (row[0], row[1]))
        return dict(action, action_id=name, is_verifier_correction=True,
                    requested_action={'type': 'navigate', 'waypoint_id': name})

    def _ground_region(self, region, kind='near_object'):
        obs = getattr(self, '_verification_obs', None) or getattr(self, 'curr_obs', None) or {}
        view = str(region.get('view') or '').lower()
        bbox = region.get('bbox')
        entity_id = region.get('entity_id')
        endpoint_text = region.get('endpoint_text') or (getattr(self, '_route_verification_context', None) or {}).get('target_text', '')
        evidence_role = region.get('evidence_role', 'primary')
        roles_map = endpoint_object_roles(endpoint_text)
        match_role = 'primary' if (evidence_role == 'spatial_landmark' or (region.get('role') == 'door_jamb' and roles_map.get('reference') is None)) else evidence_role
        binding = {'evidence_role': evidence_role,
                   'expected_object_class': roles_map.get(match_role),
                   'selected_detector_class_name': None,
                   'candidate_id': region.get('candidate_id')}
        mapper = self.mapper
        observation_id = getattr(mapper, 'observation_id', None)
        mask, mask_view, mask_id, confidence = None, None, None, None
        resolved = (
            self._resolve_visual_region({**region, 'endpoint_text': endpoint_text, 'evidence_role': evidence_role, '_match_role': match_role})
            if hasattr(self, '_resolve_visual_region')
            else None
        )
        if resolved is not None:
            if not resolved.get('valid'):
                if (resolved.get('reason') == 'class_mismatch_for_endpoint_role'
                        and kind == 'near_object'
                        and binding.get('expected_object_class') == 'door'
                        and bbox is not None
                        and getattr(self, '_pending_endpoint_confirmation', None) is not None):
                    resolved = None
                    binding['candidate_id'] = None
                    entity_id = None
                else:
                    return {'valid': False, 'reason': resolved.get('reason') or 'Invalid visual candidate region.',
                            'target_id': resolved.get('entity_id'), **binding}
        if resolved is not None:
            view = str(resolved.get('view') or view).lower()
            bbox = resolved.get('bbox')
            entity_id = resolved.get('entity_id')
            mask = resolved.get('mask')
            mask_view = resolved.get('mask_view') or view
            mask_id = resolved.get('mask_observation_id')
            confidence = resolved.get('confidence')
            binding['candidate_id'] = resolved.get('candidate_id')
            binding['selected_detector_class_name'] = resolved.get('selected_detector_class_name') or resolved.get('canonical_label')
            binding['bbox_space'] = resolved.get('bbox_space', 'full_view')
        else:
            if getattr(self, 'navigation_cfg', {}).get('task_visual_enabled', False):
                if (bbox is not None and region.get('bbox_space') != 'full_view') or region.get('bbox_space') not in (None, 'full_view'):
                    return {'valid': False, 'reason': 'Endpoint region bbox_space must be full_view.',
                            'target_id': entity_id, **binding}
            entities = getattr(mapper, 'object_entities', [])
            if type(entity_id) is int and 0 <= entity_id < len(entities):
                entity = entities[entity_id]
                label = str(entity.get('class_name') or '')
                binding['selected_detector_class_name'] = label or None
                # A supplied detector ID cannot turn a different object's mask
                # into target evidence, even when the caller relaxes ID validity.
                if not detector_matches_endpoint(label, endpoint_text, match_role):
                    return {'valid': False, 'reason': 'target_detector_class_mismatch',
                            'target_id': entity_id, **binding}
                rows = [r for r in entity.get('observations', [])
                        if r.get('observation_id') == observation_id and (not view or r.get('view') == view)]
                rows.sort(key=lambda r: float(r.get('confidence', 0)), reverse=True)
                if rows:
                    evidence = rows[0]
                    view = evidence['view']
                    detected_bbox = evidence.get('bbox')
                    # A selected ID cannot silently supply a different region's depth.
                    if bbox is not None and detected_bbox is not None:
                        try:
                            a, b = np.asarray(bbox, float), np.asarray(detected_bbox, float)
                            if a.shape != (4,) or b.shape != (4,):
                                raise ValueError('bbox shape')
                            intersection = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2]))
                            area = float(np.prod(intersection))
                            union = float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-area)
                            bbox_area = float(np.prod(a[2:]-a[:2]))
                            if union <= 0 or bbox_area <= 0 or (area/union < .20 and area/bbox_area < .60):
                                return {'valid': False, 'reason': 'Selected entity and claimed image region do not overlap.'}
                        except (TypeError, ValueError):
                            return {'valid': False, 'reason': 'Malformed target image region.'}
                    bbox = detected_bbox if bbox is None else bbox
                    mask = getattr(mapper, 'current_observation_masks', {}).get(evidence.get('mask_key'))
                    mask_view, mask_id = view, evidence.get('observation_id')
                    confidence = evidence.get('confidence')
                else:
                    return {'valid': False, 'reason': 'Selected target instance is not observed in the current view/frame.'}
            elif entity_id is not None:
                return {'valid': False, 'reason': 'Unknown target instance ID.'}
        packet_obs = obs if view == 'front' else (obs.get(view) or {})
        state = packet_obs.get('agent_state')
        if state is None or view not in ('front', 'left', 'right', 'back'):
            return {'valid': False, 'reason': 'Endpoint requires an explicit current camera view.'}
        depths = {name: (obs if name == 'front' else (obs.get(name) or {})).get('depth_sensor')
                  for name in ('front', 'left', 'right', 'back')}
        fg_kwargs = {'foreground_jamb': True} if region.get('foreground_jamb') else {}
        packet = ground_target_depth(
            depths, target_view=view, target_bbox=bbox, target_mask=mask,
            mask_view=mask_view, mask_observation_id=mask_id,
            mask_confidence=confidence, observation_id=observation_id,
            observation_step=getattr(mapper, 'observation_step', None),
            target_id=entity_id, endpoint_kind=kind,
            camera_intrinsic=mapper.camera_intrinsic,
            camera_rotation=mapper.rotation_func(state.sensor_states['color_sensor'].rotation),
            camera_position=mapper.translation_func(state.position)-mapper.initial_position,
            **fg_kwargs)
        if (not packet.get('valid')
                and packet.get('reason') in ('mixed_target_depth', 'invalid_target_bbox')
                and mask is None
                and kind == 'near_object'
                and binding.get('expected_object_class') == 'door'
                and not region.get('foreground_jamb')):
            packet = ground_target_depth(
                depths, target_view=view, target_bbox=bbox, target_mask=mask,
                mask_view=mask_view, mask_observation_id=mask_id,
                mask_confidence=confidence, observation_id=observation_id,
                observation_step=getattr(mapper, 'observation_step', None),
                target_id=entity_id, endpoint_kind=kind,
                camera_intrinsic=mapper.camera_intrinsic,
                camera_rotation=mapper.rotation_func(state.sensor_states['color_sensor'].rotation),
                camera_position=mapper.translation_func(state.position)-mapper.initial_position,
                foreground_jamb=True)
        packet['floor'] = int(getattr(mapper, 'floor_level', 0))
        packet.setdefault('view', view)
        packet.update(binding)
        return packet

    def _supported_endpoint_platform(self):
        if getattr(self, 'navigation_cfg', {}).get('habitat_control_enabled', False):
            measured = self._habitat_measurements()
            return measured['platform']['valid'] is True and measured['floor']['valid'] is True
        try:
            points = self.mapper.current_view_navigable_pcd.point.positions.cpu().numpy()
            ground = np.asarray(self.mapper.current_position) + [0, 0, self.mapper.initial_floor_height]
            return landing_support(points, ground, radius=.6, minimum_area=.20)
        except (AttributeError, TypeError, ValueError):
            return False

    def _ground_endpoint_verdict(self, parsed, ctx, result):
        """Return a deterministic rejection reason, or None for a measured endpoint."""
        kind = endpoint_kind(ctx.get('target_text', ''))
        if kind == 'stair_landing' and getattr(self, 'navigation_cfg', {}).get('habitat_control_enabled', False):
            from structural_endpoints import stair_endpoint_side
            if stair_endpoint_side(ctx.get('target_text', '')):
                return self._habitat_ground_stair_endpoint(parsed, ctx, result)
        region = {'view': parsed.get('target_view'), 'bbox': parsed.get('target_bbox'),
                  'entity_id': parsed.get('target_entity_id'),
                  'candidate_id': parsed.get('target_candidate_id'),
                  'bbox_space': parsed.get('target_bbox_space'),
                  'endpoint_text': ctx.get('target_text', ''),
                  'evidence_role': 'primary'}
        reference_region = parsed.get('reference_region')
        if isinstance(reference_region, dict):
            reference_region = {**reference_region, 'endpoint_text': ctx.get('target_text', ''),
                                'evidence_role': 'reference'}
        split_jambs = (kind == 'doorway' and isinstance(reference_region, dict)
                       and reference_region.get('role') == 'door_jamb')
        cross_view_jambs = bool(
            split_jambs
            and str(region.get('view') or '').lower()
            and str(reference_region.get('view') or '').lower()
            and str(region.get('view') or '').lower() != str(reference_region.get('view') or '').lower()
        )
        if cross_view_jambs:
            region = {**region, 'foreground_jamb': True}
            reference_region = {**reference_region, 'foreground_jamb': True}
        ignored_reference = (kind == 'doorway' and isinstance(reference_region, dict)
                             and not split_jambs)
        packet = self._ground_region(region, 'doorway' if kind == 'doorway' and not split_jambs else 'near_object')
        packet['endpoint_kind'] = kind
        if not packet.get('valid') and kind == 'doorway' and split_jambs:
            single_door = self._ground_region({k: v for k, v in region.items() if k != 'foreground_jamb'}, 'doorway')
            if single_door.get('valid'):
                single_door['endpoint_kind'] = kind
                packet = single_door
                split_jambs = False
        if ignored_reference:
            # A spatial landmark may identify a doorway but cannot supply its
            # second jamb. Fall back to both borders of the primary door ROI.
            packet['reference_region_ignored_reason'] = 'doorway_reference_requires_door_jamb_role'
        result['endpoint_grounding'] = packet
        if not packet.get('valid'):
            return packet.get('reason') or 'Current target region has no reliable depth.'
        geometry_ok = kind == 'near_object'
        reference = None
        if kind == 'doorway':
            if split_jambs:
                reference = self._ground_region(reference_region)
                result['endpoint_reference_grounding'] = reference
                if reference.get('valid') and not (hasattr(self, '_overlapping_same_candidate_jambs') and self._overlapping_same_candidate_jambs(packet, reference)):
                    packet['frame_support'] = {'left': {k: packet[k] for k in ('world_center', 'distance_m', 'depth_m')},
                                               'right': {k: reference[k] for k in ('world_center', 'distance_m', 'depth_m')}}
                    packet['opening_traversable'] = self._observed_portal_traversable(packet, reference)
            geometry_ok = doorway_position(packet, self.mapper.current_position)
            if not geometry_ok and split_jambs:
                single_door = self._ground_region({k: v for k, v in region.items() if k != 'foreground_jamb'}, 'doorway')
                if single_door.get('valid') and doorway_position(single_door, self.mapper.current_position):
                    single_door['endpoint_kind'] = kind
                    packet = single_door
                    result['endpoint_grounding'] = packet
                    reference = None
                    geometry_ok = True
                elif (cross_view_jambs
                      and float(packet.get('distance_m', 999.0)) <= 0.45
                      and float(packet.get('jamb_opening_contrast_m') or 0.0) >= 1.50
                      and self._observed_portal_traversable(packet, reference if isinstance(reference, dict) and reference.get('valid') else packet)):
                    packet['opening_traversable'] = True
                    geometry_ok = True
                    if not isinstance(reference, dict) or not reference.get('valid') or 'frame_support' not in packet:
                        reference = None
        elif kind == 'between':
            if isinstance(reference_region, dict):
                reference = self._ground_region(reference_region)
                result['endpoint_reference_grounding'] = reference
                same_candidate = (packet.get('candidate_id') is not None
                                  and packet.get('candidate_id') == reference.get('candidate_id'))
                geometry_ok = bool(reference.get('valid') and not same_candidate and between_position(
                    packet, reference, self.mapper.current_position))
        elif kind == 'stair_landing':
            # Being beside a stair must not require having already climbed it.
            # A supported floor and a localized current stair remain necessary.
            geometry_ok = self._supported_endpoint_platform()
        packet['geometry_verified'] = bool(geometry_ok)
        cfg = getattr(self, 'navigation_cfg', {})
        max_distance = float(cfg.get('endpoint_max_distance_m', 1.5))
        if kind == 'between':
            max_distance = float(cfg.get('endpoint_between_max_distance_m', 2.0))
        elif kind == 'doorway':
            max_distance = float(cfg.get('endpoint_door_frame_max_distance_m', 2.0))
        # An unidentified bbox requires fresh corroboration after a changed view.
        key = (ctx.get('key'), packet.get('floor'))
        pending = getattr(self, '_endpoint_observations', {})
        prior = pending.get(key)
        pose = np.asarray(self.mapper.current_position, float)
        rotation = np.asarray(self.mapper.current_rotation, float)
        confirmed = False
        if prior and prior['packet'].get('observation_id') != packet.get('observation_id'):
            old = prior['packet']
            try:
                same_id = (old.get('target_id') is None or packet.get('target_id') is None
                           or old.get('target_id') == packet.get('target_id'))
                point_tol = .90 if (kind == 'doorway' or packet.get('expected_object_class') == 'door') else .65
                same_point = np.linalg.norm(np.asarray(old['world_center'])-np.asarray(packet['world_center'])) <= point_tol
                moved = np.linalg.norm(pose-np.asarray(prior['pose'])) >= .10
                turned = math.acos(float(np.clip((np.trace(rotation @ np.asarray(prior['rotation']).T)-1)/2, -1, 1))) >= math.radians(8)
                fresh = int(packet['observation_id'])-int(old['observation_id']) <= 12
                if reference is not None:
                    old_ref = prior.get('reference') or {}
                    has_old_ref = isinstance(old_ref.get('world_center'), (list, tuple, np.ndarray))
                    has_cur_ref = isinstance(reference.get('world_center'), (list, tuple, np.ndarray))
                    direct_pair = bool(
                        same_point
                        and has_old_ref
                        and has_cur_ref
                        and np.linalg.norm(np.asarray(old_ref['world_center'])-np.asarray(reference['world_center'])) <= .65
                    )
                    swapped_pair = bool(
                        split_jambs
                        and has_old_ref
                        and has_cur_ref
                        and np.linalg.norm(np.asarray(old['world_center'])-np.asarray(reference['world_center'])) <= .65
                        and np.linalg.norm(np.asarray(old_ref['world_center'])-np.asarray(packet['world_center'])) <= .65
                    )
                    doorway_primary_match = bool(split_jambs and same_point)
                    confirmed = bool(same_id and (direct_pair or swapped_pair or doorway_primary_match) and (moved or turned) and fresh)
                else:
                    confirmed = bool(same_id and same_point and (moved or turned) and fresh)
            except (KeyError, TypeError, ValueError):
                confirmed = False
        # The reference-distance comparison can produce np.bool_; the evidence
        # gate deliberately accepts only an explicit native True.
        confirmed = bool(confirmed)
        decision = evaluate_endpoint_grounding(
            packet, current_observation_id=getattr(self.mapper, 'observation_id', None),
            max_distance_m=max_distance, relation_satisfied=result.get('relation_satisfied') is True,
            geometry_verified=geometry_ok, confirmed=confirmed,
            min_mask_confidence=float(cfg.get('endpoint_mask_min_confidence', .48)))
        # Both regions need independent support for a between relation.
        if decision.get('accepted') and reference is not None and not (kind == 'doorway' and split_jambs and confirmed and not reference.get('valid')):
            reference_decision = evaluate_endpoint_grounding(
                reference, current_observation_id=getattr(self.mapper, 'observation_id', None),
                max_distance_m=max_distance, relation_satisfied=True,
                geometry_verified=True, confirmed=confirmed,
                min_mask_confidence=float(cfg.get('endpoint_mask_min_confidence', .48)))
            if not reference_decision.get('accepted'):
                decision = reference_decision
        packet['confirmed_changed_view'] = confirmed
        packet['accepted'] = bool(decision.get('accepted'))
        packet['gate_reason'] = decision.get('reason')
        if decision.get('accepted'):
            self._pending_final_stop_confirmation = None
            pending.pop(key, None)
            self._endpoint_observations = pending
            return None
        # Only near, geometrically plausible claims merit a stationary confirmation
        # turn. A far target must be approached, not repeatedly reconfirmed.
        visual_region = packet.get('source') == 'vlm_bbox' or (
            reference is not None and reference.get('source') == 'vlm_bbox')
        needs_confirmation = (geometry_ok and visual_region
                              and float(packet.get('distance_m', 999)) <= max_distance
                              and (reference is None or float(reference.get('distance_m', 999)) <= max_distance)
                              and not confirmed)
        if needs_confirmation:
            pending[key] = {'packet': packet, 'reference': reference, 'pose': pose.tolist(),
                            'rotation': rotation.tolist()}
            self._endpoint_observations = pending
            result['endpoint_confirmation_required'] = True
            self._pending_final_stop_confirmation = {'key': ctx.get('key')}
        else:
            if (kind == 'doorway' and visual_region and packet.get('valid')
                    and float(packet.get('distance_m', 999)) <= max_distance and not confirmed
                    and (prior is None or not prior.get('packet', {}).get('geometry_verified'))):
                pending[key] = {'packet': packet, 'reference': reference, 'pose': pose.tolist(),
                                'rotation': rotation.tolist()}
                self._endpoint_observations = pending
            self._pending_final_stop_confirmation = None
        return decision.get('reason') or 'The measured target stopping region is not reached.'


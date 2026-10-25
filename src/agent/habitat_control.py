"""Local Habitat geometry and an independent bounded endpoint controller.

NavMesh queries describe physical support and reachability, never the evaluator
goal. Visual identity and witnessed route conditions remain mandatory.
"""
import json
import math

import numpy as np

from habitat_geometry import measured_platform, stair_profile
from structural_endpoints import (current_floor_depth, stair_endpoint_side,
                                  structural_stair_verdict, bind_visible_stair_surface)


class HabitatControl:
    def _habitat_enabled(self):
        return bool(getattr(self, 'navigation_cfg', {}).get('habitat_control_enabled', False))

    def _habitat_ground(self):
        return np.asarray(self.mapper.current_position, float) + [0., 0., self.mapper.initial_floor_height]

    def _habitat_camera_inputs(self):
        obs = getattr(self, 'curr_obs', None) or {}
        depths, cameras = {}, {}
        for view in ('front', 'left', 'right', 'back'):
            entry = obs if view == 'front' else (obs.get(view) or {})
            state = entry.get('agent_state')
            if state is None or entry.get('depth_sensor') is None:
                continue
            try:
                cameras[view] = {'position': self.mapper.translation_func(state.position)-self.mapper.initial_position,
                    'rotation': self.mapper.rotation_func(state.sensor_states['color_sensor'].rotation)}
                depths[view] = entry['depth_sensor']
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
        return depths, cameras

    def _habitat_measurements(self):
        observation_id = getattr(self.mapper, 'observation_id', None)
        ground = self._habitat_ground()
        old = getattr(self, '_habitat_measurement_cache', None)
        if (old and old['observation_id'] == observation_id
                and np.allclose(old['ground'], ground, atol=1e-6, rtol=0)):
            return old
        routes = getattr(self, 'waypoint_routes', None)
        platform = measured_platform(routes, ground, observation_id)
        profile = stair_profile(routes, ground, observation_id)
        depths, cameras = self._habitat_camera_inputs()
        floor = current_floor_depth(depths, cameras, self.mapper.camera_intrinsic, ground, observation_id)
        value = {'observation_id': observation_id, 'ground': ground.copy(),
                 'platform': platform, 'profile': profile, 'floor': floor}
        self._habitat_measurement_cache = value
        return value

    def _habitat_geometry_prompt(self, actions=None):
        if not self._habitat_enabled() or not stair_endpoint_side(getattr(self, 'instruction', '')):
            return ''
        data = self._habitat_measurements()
        platform, profile = data['platform'], data['profile']
        summary = {'current_foot_platform': {k: platform.get(k) for k in
            ('valid', 'reason', 'supported_area_m2', 'short_span_m', 'slope_deg')},
            'local_stair_profile': {k: profile.get(k) for k in
            ('valid', 'ascending', 'descending', 'ascending_rays', 'descending_rays')},
            'fresh_visible_floor': data['floor'].get('valid')}
        # Local NavMesh surfaces are projected into CURRENT RGB-D, so these
        # hints cannot silently identify stairs through walls or in old images.
        depths, cameras = self._habitat_camera_inputs()
        hints = []
        for direction in ('ascending', 'descending'):
            points = profile.get(direction+'_points') or []
            if not points:
                continue
            for view in depths:
                for part, bbox in (('left half', [0., 0., .5, 1.]), ('right half', [.5, 0., 1., 1.])):
                    bound = bind_visible_stair_surface(depths[view], cameras[view],
                        self.mapper.camera_intrinsic, bbox, points)
                    if bound['valid']:
                        hints.append({'view': view, 'region': part, 'surface_direction': direction})
        summary['current_depth_matched_stair_surfaces'] = hints
        return ('\nINDEPENDENT HABITAT LOCAL GEOMETRY (not semantic target truth): '+json.dumps(summary)
                +'\nA bottom landing needs a wide level foot-connected floor, a local rising stair, '
                'and no observed descending continuation; top is the opposite. A narrow tread or '
                'slope is not a landing. For a stair endpoint, localize the actual stair in the '
                'indicated current view; its whole mixed-depth bbox is not the floor distance. '
                'Use a nearby certified flat-platform waypoint to finish descending/ascending '
                'when your feet are still on a tread. Recheck the scene before declaring arrival.')

    def _habitat_terminal_candidate(self):
        """Brake on an observed structural endpoint, before the next long move.

        This only schedules a check. It cannot satisfy identity, route clauses,
        or STOP. Explicit stair traversal must have real vertical progress;
        a pure arrival clause need not invent such history.
        """
        text = getattr(self, 'instruction', '')
        side = stair_endpoint_side(text)
        if not self._habitat_enabled() or side is None:
            return False
        from navigation_contracts import completion_conditions
        from navigation_control import stair_condition_intents
        intents = stair_condition_intents(completion_conditions(text))
        if any(intent.get('direction') for _, intent in intents):
            required = -1 if side == 'bottom' else 1
            key = self.instruction_obj.get_current_subtask_key()
            heights = [float(row.get('height_delta', 0.)) for row in getattr(self, '_execution_ledger', [])
                       if row.get('subtask_id', key) == key and not row.get('recovery')
                       and not row.get('rejected_by_rollback')]
            if not all(math.isfinite(height) for height in heights):
                return False
            actual = sum(heights)
            if required*actual < float(self.navigation_cfg.get('stair_min_progress_m', .25)):
                return False
        data = self._habitat_measurements()
        if data.get('observation_id') != getattr(self.mapper, 'observation_id', None):
            return False
        packet = structural_stair_verdict(text, data['platform'], data['profile'], data['floor'],
                                         observation_id=data['observation_id'], ground=data['ground'])
        return packet['valid'] is True

    def _habitat_observe_unsettled_stair(self, parsed, result):
        """Collect current visual geometry while stable support is settling.

        Perception and stability can be sampled in the same physical looks.
        This never upgrades the verdict, execution evidence, or STOP readiness.
        """
        if (not self._habitat_enabled() or result.get('stair_completion_ready') is not False
                or result.get('reason') != 'Measured stair direction or supported landing is not yet established.'
                or parsed.get('outcome') is not True or parsed.get('scene_consistent') is not True
                or parsed.get('relation_satisfied') is not True or not self._habitat_terminal_candidate()):
            return
        identity = parsed.get('target_identity')
        evidence = parsed.get('target_evidence')
        if (not isinstance(identity, str) or identity.strip().lower() in (
                '', 'unknown', 'none', 'unidentified', 'contradicted', 'null', 'false', 'no', 'failed')
                or not isinstance(evidence, str) or not evidence.strip()
                or str(parsed.get('target_proximity', '')).strip().lower() not in (
                    'satisfied', 'reached', 'close', 'true', 'yes')):
            return
        context = getattr(self, '_route_verification_context', None) or {}
        if not stair_endpoint_side(context.get('target_text', '')):
            return
        measured = {}
        self._habitat_ground_stair_endpoint(parsed, context, measured)
        packet = measured.get('endpoint_grounding')
        if packet is not None:
            packet.update(accepted=False, gate_reason='waiting_for_stable_support')
            result['endpoint_grounding'] = packet
            result['support_settling_geometry_observed'] = packet.get('valid') is True

    def _habitat_ground_stair_endpoint(self, parsed, ctx, result):
        data = self._habitat_measurements()
        ground, observation_id = data['ground'], data['observation_id']
        text = ctx.get('target_text', '')
        packet = structural_stair_verdict(text, data['platform'], data['profile'], data['floor'],
                                         observation_id=observation_id, ground=ground)
        result['endpoint_grounding'] = packet
        visual = bool(getattr(self, 'navigation_cfg', {}).get('task_visual_enabled', False))
        side = stair_endpoint_side(text)
        witness_role = 'stair_structure' if (self._habitat_enabled() and side in ('top', 'bottom')) else None
        if witness_role is not None:
            packet['visual_witness_role'] = witness_role
        view = parsed.get('target_view')
        bbox = parsed.get('target_bbox')
        bbox_space = parsed.get('target_bbox_space')
        candidate_id = parsed.get('target_candidate_id') if 'target_candidate_id' in parsed else None
        entity_id = parsed.get('target_entity_id')
        if visual:
            packet['candidate_id'] = candidate_id
            packet['bbox_space'] = bbox_space
            if (bbox is not None and bbox_space != 'full_view') or bbox_space not in (None, 'full_view'):
                packet.update(valid=False, reason='Endpoint region bbox_space must be full_view.')
                return packet['reason']
        depths, cameras = self._habitat_camera_inputs()
        if entity_id is not None or (visual and candidate_id is not None):
            expected_text = 'stairs' if (visual and witness_role == 'stair_structure') else text
            region_req = {
                'view': view,
                'bbox': bbox,
                'entity_id': entity_id,
                'endpoint_text': expected_text,
                'evidence_role': 'primary',
            }
            if visual:
                region_req['candidate_id'] = candidate_id
                region_req['bbox_space'] = bbox_space if bbox_space is not None else 'full_view'
            identity = self._ground_region(region_req)
            if (visual and candidate_id is not None
                    and not identity.get('valid')
                    and identity.get('reason') == 'Endpoint requires an explicit current camera view.'):
                resolved = self._resolve_visual_region({
                    **region_req,
                    '_match_role': 'primary',
                })
                if isinstance(resolved, dict) and resolved.get('valid'):
                    r_view = str(resolved.get('view') or view or '').lower()
                    if r_view in depths and r_view in cameras:
                        from endpoint_grounding import ground_target_depth
                        identity = ground_target_depth(
                            depths,
                            target_view=r_view,
                            target_bbox=resolved.get('bbox'),
                            target_mask=resolved.get('mask'),
                            mask_view=resolved.get('mask_view') or r_view,
                            mask_observation_id=resolved.get('mask_observation_id'),
                            mask_confidence=resolved.get('confidence'),
                            observation_id=observation_id,
                            observation_step=getattr(self.mapper, 'observation_step', None),
                            target_id=resolved.get('entity_id'),
                            endpoint_kind='near_object',
                            camera_intrinsic=self.mapper.camera_intrinsic,
                            camera_rotation=cameras[r_view]['rotation'],
                            camera_position=cameras[r_view]['position'],
                        )
                        identity['view'] = r_view
                        identity['candidate_id'] = resolved.get('candidate_id')
                        identity['selected_detector_class_name'] = (
                            resolved.get('selected_detector_class_name') or resolved.get('canonical_label')
                        )
                        identity['bbox_space'] = resolved.get('bbox_space', 'full_view')
            for k in ('candidate_id', 'selected_detector_class_name', 'expected_object_class', 'bbox_space'):
                if k in identity:
                    packet[k] = identity[k]
            if 'confidence' in identity:
                packet['confidence'] = identity['confidence']
            packet.pop('mask', None)
            # Keep the existing strict class/view/mask/region identity checks.
            # Only the object's optical depth extent is replaced by separate
            # physical landing evidence, never a mismatched or stale mask.
            if (not identity.get('valid') and identity.get('reason') not in (
                    'mixed_target_depth', 'insufficient_valid_target_depth', 'too_few_target_pixels')):
                packet.update(valid=False, reason=identity.get('reason', 'current_stair_identity_not_bound'))
                return packet['reason']
            confidence = identity.get('confidence')
            if ((identity.get('source') == 'detector_mask' or (visual and candidate_id is not None)) and
                    (not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
                     or not math.isfinite(confidence)
                     or confidence < float(self.navigation_cfg.get('endpoint_mask_min_confidence', .48)))):
                packet.update(valid=False, reason='target_mask_confidence_too_low')
                return packet['reason']
            if identity.get('view'):
                view = identity['view']
            if identity.get('bbox') is not None:
                bbox = identity['bbox']
            if visual and candidate_id is not None:
                entity_id = identity.get('target_id')
        if not packet['valid']:
            return packet['reason']
        direction = 'ascending' if packet['side'] == 'bottom' else 'descending'
        bound = bind_visible_stair_surface(depths.get(view), cameras.get(view, {}),
            self.mapper.camera_intrinsic, bbox, data['profile'].get(direction+'_points'))
        packet.update(view=view, bbox=bbox, target_id=entity_id,
                      visible_stair_surface=bound)
        packet.pop('mask', None)
        if not bound['valid']:
            packet.update(valid=False, reason=bound['reason'])
            return packet['reason']
        key = (ctx.get('key'), packet['side'])
        observations = getattr(self, '_habitat_stair_observations', {})
        prior = observations.get(key)
        rotation = np.asarray(self.mapper.current_rotation, float)
        confirmed = False
        if prior and type(prior['observation_id']) is int and 0 < observation_id-prior['observation_id'] <= 12:
            angle = math.acos(float(np.clip((np.trace(rotation @ prior['rotation'].T)-1)/2, -1, 1)))
            confirmed = bool(np.linalg.norm(ground-prior['ground']) <= .35
                             and abs(ground[2]-prior['ground'][2]) <= .08
                             and (angle >= math.radians(8) or np.linalg.norm(ground-prior['ground']) >= .10)
                             and (prior['target_id'] is None or entity_id is None or prior['target_id'] == entity_id))
        observations[key] = {'observation_id': observation_id, 'ground': ground.copy(),
                             'rotation': rotation.copy(), 'target_id': entity_id}
        self._habitat_stair_observations = observations
        packet.update(accepted=confirmed, confirmed_changed_view=confirmed,
                      gate_reason='accepted' if confirmed else 'changed_view_confirmation_required')
        if not confirmed:
            result['endpoint_confirmation_required'] = True
            return 'changed_view_confirmation_required'
        return None

    def _habitat_certified_endpoint_actions(self, actions, verification):
        """Bound local adjustment to routes the real movement backend accepts."""
        pose = self._habitat_ground()
        rows, by_id = [], {}
        for name, action in (actions or {}).items():
            if not isinstance(action, dict) or action.get('type') != 'waypoint' or action.get('rejected'):
                continue
            point = np.asarray(action.get('target_point_world', []), float)
            if point.shape != (3,) or not np.isfinite(point).all() or np.linalg.norm(point-pose) > 1.4:
                continue
            route = self._validated_waypoint_route(point)
            if route is None:
                continue
            points = np.asarray(route['path'], float)
            length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            if length > 1.2:
                continue
            rows.append({'action_id': str(name), 'certified': True,
                         'target_point_world': np.asarray(route['target']).tolist(), 'path_length_m': length})
            by_id[str(name)] = dict(action, action_id=str(name), target_point_world=route['target'])
        progress = verification.get('stair_progress') or {}
        direction = {'DOWN': -1, 'UP': 1}.get(progress.get('direction'), 0)
        if (direction and verification.get('candidate_claimed')
                and direction*float(progress.get('measured_height_change_m', 0.)) >= .15
                and verification.get('stair_completion_ready') is False):
            candidates = []
            for row in rows:
                point = np.asarray(row['target_point_world'])
                if (row['path_length_m'] <= .9 and .03 <= direction*(point[2]-pose[2]) <= .5
                        and measured_platform(self.waypoint_routes, point,
                            getattr(self.mapper, 'observation_id', None))['valid']):
                    candidates.append(row)
            if candidates:
                candidate = min(candidates, key=lambda row: row['path_length_m'])
                verification['local_geometry_candidate'] = {
                    'action_id': candidate['action_id'], 'target_point_world': candidate['target_point_world'],
                    'kind': 'stair_landing_observation', 'physically_supported': True}
        return rows, by_id

    def _habitat_boundary_stop_decision(self, proposal, front, views, map_image, obs, actions):
        from endpoint_controller import EndpointController, EndpointConfig
        key, _ = self.instruction_obj.get_current_subtask()
        controller = getattr(self, '_endpoint_controller', None)
        if controller is None:
            controller = self._endpoint_controller = EndpointController(EndpointConfig(
                max_calls=max(1, min(8, int(self.navigation_cfg.get('endpoint_session_max_calls', 6)))),
                max_looks=max(1, min(4, int(self.navigation_cfg.get('endpoint_session_max_looks', 2)))),
                max_approaches=max(1, min(3, int(self.navigation_cfg.get('endpoint_session_max_approaches', 3))))))
        if not controller.active:
            self._endpoint_session_start_step = len(getattr(self, '_execution_ledger', []))
        elif controller.calls >= controller.config.max_calls:
            self.current_path, self.action_sequence = [], []
            self._waypoint_execution = None
            self._pending_endpoint_confirmation = None
            self._pending_endpoint_relocalization = None
            self._pending_final_stop_confirmation = None
            self._endpoint_escape_requested = {'reason': 'endpoint_session_verification_budget_exhausted',
                'subtask_id': key, 'scope_key': key+'.endpoint',
                'first_affected_step': getattr(self, '_endpoint_session_start_step', len(getattr(self, '_execution_ledger', [])))}
            return {'type': 'noop'}
        budget = getattr(self, 'runtime_budget', None)
        if budget is not None:
            budget.check()
        calls = getattr(self, '_habitat_verifier_calls', {})
        maximum = max(1, int(self.navigation_cfg.get('habitat_max_verification_calls_per_subtask', 12)))
        if calls.get(key, 0) >= maximum:
            from progress_watchdog import NavigationAborted
            raise NavigationAborted('habitat_verification_budget_exhausted')
        calls[key] = calls.get(key, 0)+1
        self._habitat_verifier_calls = calls
        if proposal.get('_controller_probe') and hasattr(self, '_claim_terminal_probe'):
            self._claim_terminal_probe(getattr(self, '_last_step_index', 0))
        verification = self._verify_current_subtask(proposal, front, views, map_image, obs=obs, actions=actions, max_turn=1)
        verification['physical_arrival_candidate'] = self._habitat_terminal_candidate()
        # Semantic wrong-route claims keep the measured contradiction gate.
        if verification.get('rollback_required'):
            controller.reset()
            return self._direct_verifier_correction(verification, actions)
        if (
            controller.active
            and not verification.get('outcome')
            and verification.get('evidence_state') == 'contradicted'
            and not verification.get('candidate_claimed')
            and not verification.get('physical_arrival_candidate')
            and not verification.get('evidence_repair_required')
            and isinstance(verification.get('correction_action'), dict)
            and verification['correction_action'].get('type') == 'navigate'
        ):
            controller.reset()
            self._pending_endpoint_confirmation = None
            self._pending_endpoint_relocalization = None
            self._pending_final_stop_confirmation = None
            return self._direct_verifier_correction(verification, actions)
        certified, by_id = self._habitat_certified_endpoint_actions(actions, verification)
        side = stair_endpoint_side(getattr(self, 'instruction', ''))
        if side and verification.get('candidate_claimed') and not verification.get('local_geometry_candidate'):
            measured = self._habitat_measurements()
            profile = measured['profile']
            wrong_structure = bool(profile.get('valid') and (
                measured['platform'].get('valid') is not True
                or profile.get('ascending') and profile.get('descending')
                or (side == 'bottom' and not profile.get('ascending'))
                or (side == 'top' and not profile.get('descending'))))
            # A physically contradicted landing claim is not a target to hold.
            # Continue the explicit stair direction through observed, certified
            # nearby waypoints; never manufacture completion evidence for it.
            if wrong_structure:
                direction = -1 if side == 'bottom' else 1
                pose = self._habitat_ground()
                choices = [row for row in certified
                    if .03 <= direction*(row['target_point_world'][2]-pose[2]) <= .50
                    and row['path_length_m'] <= 1.0]
                if choices:
                    choice = min(choices, key=lambda row: row['path_length_m'])
                    controller.reset()
                    self.current_path, self.action_sequence = [], []
                    self._waypoint_execution = None
                    self._pending_endpoint_confirmation = None
                    self._pending_endpoint_relocalization = None
                    self._pending_final_stop_confirmation = None
                    self.navigation_trace.append({'role': 'controller', 'event': 'habitat_stair_continuation',
                        'step': getattr(self, '_last_step_index', 0), 'action_id': choice['action_id'],
                        'reason': 'Current local geometry contradicts the claimed stair landing.',
                        'platform_reason': measured['platform'].get('reason'), 'side': side})
                    return {**by_id[choice['action_id']], 'is_verifier_correction': True,
                            'requested_action': {'type': 'navigate', 'waypoint_id': choice['action_id']}}
        sequence = int(getattr(self, '_habitat_verification_sequence', 0))+1
        self._habitat_verification_sequence = sequence
        ledger = getattr(self, '_execution_ledger', [])
        decision = controller.consume(verification=verification,
            observation_id=getattr(self.mapper, 'observation_id', None), pose=self._habitat_ground().tolist(),
            scope_key=key+'.endpoint', executed_action=ledger[-1] if ledger else None,
            certified_actions=certified, verification_id=sequence)
        self.navigation_trace.append({'role': 'controller', 'event': 'endpoint_session_decision',
            'step': getattr(self, '_last_step_index', 0), 'subtask_id': key,
            'observation_id': getattr(self.mapper, 'observation_id', None), **decision.as_dict()})
        if decision.kind == 'repair':
            # _verify_current_subtask already saved the rejected references in
            # prompt history. A second charged request uses the same images,
            # pose and ledger; only a model response can repair its citations.
            # Re-enter the shared session/global/runtime budget checks before
            # dispatch, and do not execute or count a new physical observation.
            self.navigation_trace.append({'role': 'controller', 'event': 'verification_evidence_repair',
                'step': getattr(self, '_last_step_index', 0), 'subtask_id': key,
                'observation_id': getattr(self.mapper, 'observation_id', None),
                'session_call': controller.calls, 'repair_attempt': controller.protocol_repairs,
                'source': 'habitat_endpoint_session'})
            return self._habitat_boundary_stop_decision(proposal, front, views, map_image, obs, actions)
        if decision.kind == 'protocol_failure':
            from progress_watchdog import NavigationAborted
            raise NavigationAborted(decision.reason)
        if decision.kind == 'continue':
            if (proposal.get('_controller_probe')
                    and getattr(self, 'current_path', None)
                    and str(verification.get('target_view') or '').lower() in ('back', 'left', 'right')):
                self.current_path, self.action_sequence = [], []
                self._waypoint_execution = None
                if hasattr(self, '_build_verifier_feedback'):
                    self._verifier_feedback = self._build_verifier_feedback(key, verification)
            return {'type': 'continue_path'} if proposal.get('_controller_probe') else {'type': 'noop'}
        self.current_path, self.action_sequence = [], []
        self._waypoint_execution = None
        self._pending_endpoint_confirmation = None
        self._pending_endpoint_relocalization = None
        self._pending_final_stop_confirmation = None
        if decision.kind == 'ready' and self._verification_accepts_stop(verification):
            self._verifier_feedback = None
            return {'type': 'stop', 'verified': True}
        if decision.kind == 'approach' and decision.action_id in by_id:
            return {**by_id[decision.action_id], 'is_verifier_correction': True,
                    'requested_action': {'type': 'navigate', 'waypoint_id': decision.action_id}}
        if decision.kind == 'inspect':
            self._pending_endpoint_confirmation = {'key': key, 'is_probe': bool(proposal.get('_controller_probe')),
                                                   'habitat_session': True}
            return {'type': 'turn', 'angle': math.radians(12 if controller.looks % 2 == 0 else -24),
                    'is_verifier_correction': True, 'endpoint_confirmation': True}
        self._endpoint_escape_requested = {'reason': decision.reason, 'subtask_id': key,
            'scope_key': key+'.endpoint', 'first_affected_step': getattr(self, '_endpoint_session_start_step', len(ledger))}
        return {'type': 'noop'}


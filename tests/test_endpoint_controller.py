"""CPU-only endpoint ownership and measured-evidence regressions."""
import copy
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from endpoint_controller import EndpointConfig, EndpointController


def verdict(obs=1, accepted=False, point=(1., 0., 0.), reason='changed_view_confirmation_required'):
    return {'candidate_claimed': True, 'outcome': accepted,
            'completion_evidence_reliable': accepted,
            'endpoint_grounding': {'valid': True, 'observation_id': obs,
                'world_center': list(point), 'distance_m': 1., 'target_id': None,
                'geometry_verified': True, 'accepted': accepted,
                'gate_reason': reason}}


def look(step=0, yaw=math.radians(12)):
    return {'step': step, 'action_type': 'turn', 'yaw_delta': yaw, 'motion_distance': 0.}


def local_action(name='near', target=(.5, 0., 0.), **updates):
    return {'action_id': name, 'certified': True, 'target_point_world': list(target),
            'path_length_m': .5, **updates}


def consume(controller, data=None, obs=1, pose=(0., 0., 0.), **kwargs):
    return controller.consume(verification=verdict(obs) if data is None else data,
                              observation_id=obs, pose=pose, scope_key='P1.C2', **kwargs)


class EndpointControllerTests(unittest.TestCase):
    def test_negative_probe_does_not_capture_normal_navigation(self):
        c = EndpointController()
        self.assertEqual(consume(c, {'outcome': False}).kind, 'continue')
        self.assertFalse(c.active)
        self.assertEqual(c.calls, 0)

    def test_candidate_signals_arm_but_far_unclaimed_packet_does_not(self):
        for field in ('candidate_claimed', 'endpoint_relocalization_required',
                      'endpoint_confirmation_required', 'landing_observation_pending',
                      'physical_arrival_candidate'):
            c = EndpointController()
            self.assertEqual(consume(c, {field: True}).kind, 'inspect')
            self.assertTrue(c.active)
        far = verdict()
        far['candidate_claimed'] = False
        far['endpoint_grounding']['distance_m'] = 5.
        self.assertEqual(consume(EndpointController(), far).kind, 'continue')

    def test_repeated_api_requests_do_not_corroborate_same_observation(self):
        c = EndpointController()
        self.assertEqual(consume(c, verdict(accepted=True), verification_id='call1').kind, 'inspect')
        self.assertEqual(consume(c, verdict(accepted=True), verification_id='call2',
                                 executed_action=look()).kind, 'inspect')
        self.assertEqual(c.calls, 2)
        consume(c, verdict(accepted=True), verification_id='call2', executed_action=look())
        self.assertEqual((c.calls, c.looks), (2, 1))

    def test_actual_changed_view_and_current_complete_guard_can_be_ready(self):
        c = EndpointController()
        consume(c)
        out = consume(c, verdict(2, accepted=True), obs=2, executed_action=look())
        self.assertEqual(out.kind, 'ready')
        self.assertNotIn('stop', out.as_dict())
        json.dumps(out.as_dict())

    def test_fresh_counter_without_motion_is_not_changed_view(self):
        c = EndpointController()
        consume(c)
        self.assertEqual(consume(c, verdict(2, accepted=True), obs=2).kind, 'inspect')

    def test_cancelled_turns_do_not_prove_changed_view(self):
        c = EndpointController()
        consume(c)
        consume(c, verdict(2), obs=2, executed_action=look(0))
        out = consume(c, verdict(3, accepted=True), obs=3, executed_action=look(1, -math.radians(12)))
        self.assertEqual(out.kind, 'recover')  # Two looks consumed, net rotation zero.

    def test_physical_translation_can_confirm_and_floor_labels_do_not_split_scope(self):
        c = EndpointController()
        first = verdict()
        first['endpoint_grounding']['floor'] = 2
        consume(c, first)
        second = verdict(2, accepted=True)
        second['endpoint_grounding']['floor'] = -1
        self.assertEqual(consume(c, second, obs=2, pose=(.15, 0., 0.)).kind, 'ready')

    def test_structural_floor_candidate_point_does_not_need_object_world_center(self):
        c = EndpointController()
        first = verdict()
        first['endpoint_grounding'].pop('world_center')
        first['endpoint_grounding'].update(source='observed_supported_floor', candidate_point=[0., 0., 0.])
        consume(c, first)
        second = copy.deepcopy(first)
        second.update(outcome=True, completion_evidence_reliable=True)
        second['endpoint_grounding'].update(observation_id=2, accepted=True)
        self.assertEqual(consume(c, second, obs=2, executed_action=look()).kind, 'ready')

    def test_world_region_and_detector_id_cannot_silently_switch(self):
        for update in ({'world_center': [4., 0., 0.]}, {'target_id': 8}):
            c = EndpointController()
            first = verdict(); first['endpoint_grounding']['target_id'] = 7
            consume(c, first)
            second = verdict(2, accepted=True); second['endpoint_grounding'].update(update)
            self.assertEqual(consume(c, second, obs=2, executed_action=look()).kind, 'inspect')
            self.assertEqual(c.candidate_point, (1., 0., 0.))
            self.assertEqual(c.target_id, 7)

    def test_stale_future_or_missing_geometry_never_ready(self):
        for updates in ({'observation_id': 1}, {'observation_id': 3}, {'geometry_verified': False}):
            c = EndpointController(); consume(c)
            second = verdict(2, accepted=True); second['endpoint_grounding'].update(updates)
            self.assertNotEqual(consume(c, second, obs=2, executed_action=look()).kind, 'ready')

    def test_expired_view_requires_a_new_pair(self):
        c = EndpointController(); consume(c)
        self.assertEqual(consume(c, verdict(20, accepted=True), obs=20, executed_action=look()).kind, 'inspect')
        self.assertEqual(consume(c, verdict(21, accepted=True), obs=21,
                                 executed_action=look(1)).kind, 'ready')

    def test_unbound_region_never_uses_its_reported_center_for_approach(self):
        c = EndpointController()
        bad = verdict(); bad['endpoint_grounding'].update(valid=False, gate_reason='target_surface_too_far')
        out = consume(c, bad, certified_actions=[local_action()])
        self.assertEqual(out.kind, 'inspect')
        self.assertIsNone(c.candidate_point)
        self.assertIsNone(out.action_id)

    def test_active_negative_stays_local_then_requests_physical_escape(self):
        c = EndpointController(); consume(c, {'candidate_claimed': True})
        self.assertEqual(consume(c, {'outcome': False}, obs=2, executed_action=look()).kind, 'inspect')
        out = consume(c, {'outcome': False}, obs=3, executed_action=look(1))
        self.assertEqual(out.kind, 'recover')
        self.assertTrue(c.active)
        self.assertEqual(consume(c, verdict(4, accepted=True), obs=4).kind, 'recover')

    def test_only_certified_short_height_preserving_progress_is_approach(self):
        c = EndpointController()
        far = verdict(point=(3., 0., 0.), reason='target_surface_too_far')
        choices = [local_action('uncertified', certified=False),
                   local_action('upstairs', (.5, 0., .5), path_length_m=.71),
                   local_action('long_detour', path_length_m=3.),
                   local_action('away', (-.5, 0., 0.)), local_action()]
        out = consume(c, far, certified_actions=choices)
        self.assertEqual((out.kind, out.action_id), ('approach', 'near'))
        self.assertEqual(out.target, (.5, 0., 0.))

    def test_unknown_route_length_or_absent_certification_cannot_approach(self):
        for updates in ({'path_length_m': None}, {'path_length_m': float('nan')}, {'certified': 1}):
            c = EndpointController()
            out = consume(c, verdict(point=(3., 0., 0.), reason='target_surface_too_far'),
                          certified_actions=[local_action(**updates)])
            self.assertEqual(out.kind, 'recover')

    def test_approach_and_model_call_budgets_are_finite(self):
        c = EndpointController(EndpointConfig(max_approaches=1))
        far = verdict(point=(3., 0., 0.), reason='target_surface_too_far')
        consume(c, far, certified_actions=[local_action()])
        far['endpoint_grounding']['observation_id'] = 2
        out = consume(c, far, obs=2, pose=(.5, 0., 0.),
                      executed_action={'step': 0, 'action_type': 'move_forward', 'motion_distance': .5},
                      certified_actions=[local_action(target=(1., 0., 0.))])
        self.assertEqual(out.reason, 'local_approach_budget_exhausted')
        c = EndpointController(EndpointConfig(max_calls=2))
        consume(c, verification_id='first')
        self.assertEqual(consume(c, verification_id='second').kind, 'recover')

    def test_height_departure_and_observation_reset_require_recovery(self):
        c = EndpointController(); consume(c)
        self.assertEqual(consume(c, verdict(2), obs=2, pose=(0., 0., .4)).reason,
                         'left_candidate_physical_height')
        c = EndpointController(); consume(c, verdict(5), obs=5)
        self.assertEqual(consume(c, verdict(1), obs=1).reason, 'observation_epoch_changed')

    def test_explicit_scope_change_or_reset_releases_old_hypothesis(self):
        c = EndpointController(); consume(c)
        out = c.consume(verification={'outcome': False}, observation_id=2,
                        pose=(0, 0, 0), scope_key='P2.C1')
        self.assertEqual(out.kind, 'continue')
        self.assertFalse(c.active)
        consume(c); c.reset()
        self.assertFalse(c.active)

    def test_real_probe_mixed_rois_cannot_hand_back_to_upstairs_waypoint(self):
        # 556 endpoint-only probe, frozen41fecb1, saved steps0/1.
        c = EndpointController(EndpointConfig(max_looks=1))
        first = {'candidate_claimed': True, 'outcome': False,
                 'endpoint_grounding': {'valid': False, 'observation_id': 1,
                     'view': 'back', 'bbox': [.4, .1, 1., 1.], 'depth_extent_m': 1.420,
                     'world_center': [.380, 1.019, -.622], 'reason': 'mixed_target_depth'}}
        self.assertEqual(consume(c, first).kind, 'inspect')
        second = copy.deepcopy(first)
        second['endpoint_grounding'].update(observation_id=2, view='right',
            bbox=[.65, .45, 1., .7], depth_extent_m=1.352, world_center=[2.194, -.501, -.839])
        upstairs = local_action('WP_11', (.125, .375, .301), path_length_m=.497)
        out = consume(c, second, obs=2, executed_action=look(), certified_actions=[upstairs])
        self.assertEqual(out.kind, 'recover')
        self.assertIsNone(out.action_id)
        self.assertIsNone(c.candidate_point)

    def test_measured_stair_observation_approach_is_not_semantic_ready(self):
        for direction, sign in [('DOWN', -1), ('UP', 1)]:
            c = EndpointController()
            target = (.5, 0., sign * .371)
            hint = {'action_id': 'flat', 'target_point_world': list(target),
                    'kind': 'stair_landing_observation', 'physically_supported': True}
            data = {'candidate_claimed': True, 'outcome': True,
                    'local_geometry_candidate': hint,
                    'stair_progress': {'direction': direction, 'measured_height_change_m': sign * 1.9}}
            out = consume(c, data, certified_actions=[local_action('flat', target, path_length_m=.80)])
            self.assertEqual((out.kind, out.action_id), ('approach', 'flat'))
            self.assertIsNone(c.candidate_point)
            self.assertIsNone(c._witness)
            # Measured arrival within the certified height envelope is allowed,
            # but no semantic ROI still means inspect rather than ready.
            data.pop('local_geometry_candidate')
            self.assertEqual(consume(c, data, obs=2, pose=target,
                executed_action={'step': 0, 'action_type': 'move_forward',
                                 'motion_distance': .80}).kind, 'inspect')

    def test_stair_observation_hint_needs_matching_certification_and_measured_direction(self):
        base = {'candidate_claimed': True,
                'local_geometry_candidate': {'action_id': 'flat', 'target_point_world': [.5, 0, -.371],
                    'kind': 'stair_landing_observation', 'physically_supported': True},
                'stair_progress': {'direction': 'DOWN', 'measured_height_change_m': -1.9}}
        bad_cases = [(base, [local_action('flat', (.5, 0, -.371), certified=False)]),
                     (base, [local_action('flat', (.5, 0, -.2), path_length_m=.8)]),
                     (base, [local_action('flat', (.5, 0, -.371), path_length_m=1.1)]),
                     ({**base, 'stair_progress': {}}, [local_action('flat', (.5, 0, -.371), path_length_m=.8)]),
                     ({**base, 'stair_progress': {'direction': 'UP', 'measured_height_change_m': 1.9}},
                      [local_action('flat', (.5, 0, -.371), path_length_m=.8)])]
        for data, actions in bad_cases:
            with self.subTest(data=data, actions=actions):
                self.assertEqual(consume(EndpointController(), data, certified_actions=actions).kind, 'inspect')

    def test_old_observation_approach_hint_cannot_override_fresh_verified_ready(self):
        c = EndpointController(); consume(c)
        current = verdict(2, accepted=True)
        current.update(local_geometry_candidate={'action_id': 'flat',
            'target_point_world': [.5, 0, -.371], 'kind': 'stair_landing_observation',
            'physically_supported': True},
            stair_progress={'direction': 'DOWN', 'measured_height_change_m': -1.9})
        out = consume(c, current, obs=2, executed_action=look(),
                      certified_actions=[local_action('flat', (.5, 0, -.371), path_length_m=.8)])
        self.assertEqual(out.kind, 'ready')


if __name__ == '__main__':
    unittest.main()


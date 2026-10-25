"""Bounded landing sampling uses real observations and keeps STOP geometry strict."""
import copy
import unittest
from unittest.mock import Mock

from navigation_control import StairProgress
from verification_feedback import landing_observation_pending
from test_v3_boundary_height import agent, verdict, ACTIONS
from test_v3_endpoint_integration import grounded_guard


RAW = {'outcome': True, 'relation_satisfied': True, 'scene_consistent': True, 'failure_type': 'none'}


def sample(stairs):
    result = verdict(outcome=False, relation_satisfied=False, completion_evidence_reliable=False,
                     failure_type='not_reached', stair_completion_ready=False, landing_support_current=True,
                     reason='Measured stair direction or supported landing is not yet established.',
                     stair_progress=stairs.packet(), condition_checks=[
                         {'status': 'satisfied', 'execution_checked': True, 'requires_arrival': False},
                         {'status': 'satisfied', 'observation_checked': True, 'requires_arrival': True}])
    result['landing_observation_pending'] = landing_observation_pending(RAW, result)
    return result


def supported_stairs():
    stairs = StairProgress('P1', 'P1.C1', {'direction': -1, 'ambiguous': False, 'landing_required': True}, 0.)
    stairs.observe({'action_type': 'move_forward', 'pose': [0, 0, -.5], 'height_delta': -.5}, False)
    stairs.observe({'action_type': 'move_forward', 'pose': [0, 0, -.5], 'height_delta': 0.}, True)
    return stairs


class LandingObservationTests(unittest.TestCase):
    def test_lost_support_revokes_landing_and_requires_a_new_full_sequence(self):
        stairs = supported_stairs()
        turn = {'action_type': 'turn', 'pose': [0, 0, -.5], 'height_delta': 0.}
        for _ in range(2):
            stairs.observe(turn, True)
        self.assertTrue(stairs.completion_ready())
        stairs.observe(turn, False)
        self.assertFalse(stairs.completion_ready())
        self.assertEqual(stairs.phase, 'traverse')
        self.assertEqual(stairs.stable_samples, 0)
        for count in (1, 2, 3):
            stairs.observe(turn, True)
            self.assertEqual(stairs.stable_samples, count)
            self.assertEqual(stairs.completion_ready(), count == 3)

    def test_continued_vertical_motion_revokes_landing_even_with_support_claim(self):
        for direction in (-1, 1):
            stairs = StairProgress('P1', 'P1.C1',
                {'direction': direction, 'ambiguous': False, 'landing_required': True}, 0.,
                {'stair_landing_observations': 2})
            height = direction * .5
            stairs.observe({'action_type': 'move_forward', 'pose': [0, 0, height],
                            'height_delta': height}, False)
            for _ in range(2):
                stairs.observe({'action_type': 'turn', 'pose': [0, 0, height], 'height_delta': 0.}, True)
            self.assertTrue(stairs.completion_ready())
            stairs.observe({'action_type': 'move_forward', 'pose': [0, 0, direction * .66],
                            'height_delta': direction * .16}, True)
            self.assertFalse(stairs.completion_ready())
            self.assertEqual(stairs.phase, 'traverse')
            self.assertEqual(stairs.stable_samples, 0)

    def test_supported_flat_progress_keeps_landing_without_weakening_reversal(self):
        stairs = supported_stairs()
        for _ in range(2):
            stairs.observe({'action_type': 'turn', 'pose': [0, 0, -.5], 'height_delta': 0.}, True)
        stairs.observe({'action_type': 'move_forward', 'pose': [.2, 0, -.49], 'height_delta': .01}, True)
        self.assertTrue(stairs.completion_ready())
        stairs.observe({'action_type': 'move_forward', 'pose': [.4, 0, -.29], 'height_delta': .20}, False)
        self.assertFalse(stairs.completion_ready())
        incident = stairs.observe({'action_type': 'move_forward', 'pose': [.6, 0, -.09], 'height_delta': .20}, False)
        self.assertEqual(incident, 'stair_motion_opposes_current_clause')
        self.assertEqual(stairs.phase, 'reobserve')

    def test_one_measured_support_can_request_sampling_but_reads_never_increment_it(self):
        stairs = supported_stairs()
        before = copy.deepcopy(stairs.packet())
        for _ in range(5):
            self.assertTrue(sample(stairs)['landing_observation_pending'])
            self.assertFalse(stairs.completion_ready())
        self.assertEqual(stairs.packet(), before)
        self.assertEqual(stairs.stable_samples, 1)

    def test_still_descending_far_insufficient_or_unverified_arrival_cannot_hold(self):
        good = sample(supported_stairs())
        for key, value in (('stable_supported_observations', 0), ('stable_supported_observations', 3),
                           ('stable_supported_observations', True), ('measured_height_change_m', -.1),
                           ('measured_height_change_m', .5), ('measured_height_change_m', float('inf')),
                           ('direction', 'NONE'), ('direction_violation', True), ('landing_required', False)):
            bad = copy.deepcopy(good)
            bad['stair_progress'][key] = value
            self.assertFalse(landing_observation_pending(RAW, bad), (key, value))
        for change in ({'landing_support_current': False}, {'condition_checks': []},
                       {'stair_completion_ready': True}, {'contract_conflict': True},
                       {'reason': 'target_surface_too_far'}, {'reason': None}, {'rollback_required': True}):
            self.assertFalse(landing_observation_pending(RAW, dict(good, **change)))
        for field in ('outcome', 'relation_satisfied', 'scene_consistent'):
            self.assertFalse(landing_observation_pending(dict(RAW, **{field: False}), good))

    def test_two_actual_stationary_observations_reverify_before_any_translation(self):
        for is_probe in (False, True):
            stairs = supported_stairs()
            a = agent()
            a.current_path, a.action_sequence, a._waypoint_execution = [[1, 0, 0]], [ACTIONS['WP_01']], {'old': True}
            a._verify_current_subtask.side_effect = lambda *args, **kwargs: (
                verdict() if stairs.completion_ready() else sample(stairs))
            dispatch = lambda: a._boundary_stop_decision({'_controller_probe': is_probe}, None, [], None, {}, ACTIONS)
            for expected_stable in (1, 2):
                action = dispatch()
                self.assertEqual(action['type'], 'turn')
                self.assertEqual(stairs.stable_samples, expected_stable)
                self.assertEqual(a.current_path, [])
                self.assertTrue(a._pending_endpoint_relocalization['landing_observation'])
                stairs.observe({'action_type': 'turn', 'pose': [0, 0, -.5], 'height_delta': 0.}, True)
                a._observation_revision = expected_stable
            self.assertEqual(dispatch()['type'], 'stop')
            self.assertEqual(a._verify_current_subtask.call_count, 3)
            self.assertEqual(a._endpoint_relocalization_turns['P1'], 2)
            self.assertTrue(all(e['source'] == 'landing_support_confirmation' for e in a.navigation_trace))

    def test_unresolved_landing_is_bounded_to_two_turns_then_handback(self):
        a = agent(sample(supported_stairs()))
        dispatch = lambda: a._boundary_stop_decision({'_controller_probe': True}, None, [], None, {}, ACTIONS)
        self.assertEqual(dispatch()['type'], 'turn')
        a._observation_revision = 1
        self.assertEqual(dispatch()['type'], 'turn')
        a._observation_revision = 2
        self.assertEqual(dispatch()['type'], 'noop')
        self.assertEqual(a._endpoint_relocalization_turns['P1'], 2)

    def test_ready_landing_still_needs_actual_endpoint_localization_and_geometry(self):
        a, raw = grounded_guard('Walk down the stairs. Stop at the bottom of the stairs.')
        a.st['record'][0]['height_delta'] = -.5
        a.stair_ready = False
        a.stair_packet = supported_stairs().packet()
        a._landing_support_observed = Mock(return_value=True)
        a.prepare()
        result = a.evaluate(raw)
        self.assertTrue(result['landing_support_current'])
        self.assertFalse(result['stair_completion_ready'])
        self.assertTrue(landing_observation_pending(raw, result))
        a.stair_ready = True
        a.prepare()
        raw.update(target_entity_id=None, target_view=None, target_bbox=None)
        result = a.evaluate(raw)
        self.assertFalse(result['outcome'])
        self.assertIn('endpoint_grounding', result)
        self.assertFalse(landing_observation_pending(raw, result))


if __name__ == '__main__':
    unittest.main()


"""Endpoint localization failures must brake before the next waypoint move."""
import copy
import json
import unittest
from unittest.mock import Mock

import test_v3_boundary_height as boundary_tests
import test_v3_navigation_control as control_tests
from test_v3_terminal_integration import moving_agent
from verification_feedback import endpoint_relocalization_required
from progress_watchdog import NavigationAborted


def affirmative():
    return {'outcome': True, 'scene_consistent': True, 'relation_satisfied': True,
            'failure_type': 'none', 'target_view': 'left', 'target_bbox': [.1, .2, .4, .8]}


def localization_failure(reason='mixed_target_depth'):
    result = boundary_tests.verdict(
        outcome=False, relation_satisfied=False, completion_evidence_reliable=False,
        evidence_state='unknown', confidence='low', failure_type='not_reached', reason=reason,
        condition_contract_conflict=False,
        condition_checks=[{'id': 'P1.C1', 'status': 'satisfied', 'execution_checked': True}],
        doorway_checks=[{'execution_witnessed': True}],
        endpoint_grounding={'valid': False, 'reason': reason, 'source': 'vlm_bbox',
                            'view': 'left', 'bbox': [.1, .2, .4, .8], 'endpoint_kind': 'doorway'})
    result['endpoint_relocalization_required'] = endpoint_relocalization_required(affirmative(), result)
    return result


def integrated_agent(results):
    a = moving_agent()
    for name in ('_boundary_stop_decision', '_verification_accepts_stop', '_direct_verifier_correction'):
        setattr(a, name, getattr(boundary_tests.Agent, name).__get__(a))
    a._verify_current_subtask = Mock(side_effect=results)
    a._observation_revision = 0
    a.navigation_cfg['verifier_auto_reverify'] = False  # Pending localization must independently force verification.

    def decide(**kwargs):
        proposal = {'_controller_probe': kwargs.get('is_terminal_probe', False)}
        decision = a._boundary_stop_decision(proposal, None, [], None, {}, boundary_tests.ACTIONS)
        return decision, 'verified current frame', None

    a.decide_waypoint = Mock(side_effect=decide)
    return a


class RelocalizationClassificationTests(unittest.TestCase):
    def test_near_measured_geometry_veto_requires_localization_but_missing_geometry_does_not(self):
        result = localization_failure()
        packet = result['endpoint_grounding'] = {
            'valid': True, 'reason': 'ok', 'source': 'vlm_bbox', 'view': 'left',
            'endpoint_kind': 'doorway', 'distance_m': .8, 'observation_id': 5,
            'world_center': [0., .8, 1.],
            'frame_support': {side: {'valid': True, 'distance_m': .8, 'world_center': [0., .8, 1.]}
                              for side in ('left', 'right')},
            'geometry_verified': False, 'gate_reason': 'measured_endpoint_geometry_required'}
        self.assertTrue(endpoint_relocalization_required(affirmative(), result))
        for update in ({'geometry_verified': None}, {'geometry_verified': True},
                       {'distance_m': 3.}, {'distance_m': None}, {'distance_m': float('nan')},
                       {'source': None}, {'observation_id': None}, {'view': 'front'},
                       {'world_center': None}, {'frame_support': {}},
                       {'gate_reason': 'stale_or_future_target_observation'},
                       {'endpoint_kind': 'near_object'}, {'endpoint_kind': 'between'}):
            with self.subTest(update=update):
                self.assertFalse(endpoint_relocalization_required(
                    affirmative(), {**result, 'endpoint_grounding': {**packet, **update}}))
        self.assertFalse(endpoint_relocalization_required({**affirmative(), 'outcome': False}, result))
        self.assertFalse(endpoint_relocalization_required(affirmative(), {**result, 'condition_checks': []}))
        result['endpoint_reference_grounding'] = {'valid': True, 'distance_m': .7, 'observation_id': 4}
        self.assertFalse(endpoint_relocalization_required(affirmative(), result))
        result['endpoint_reference_grounding'] = {**packet, 'observation_id': 5}
        self.assertFalse(endpoint_relocalization_required({**affirmative(), 'reference_region': 'invalid'}, result))
        result['endpoint_grounding'] = {**packet, 'endpoint_kind': 'between'}
        self.assertTrue(endpoint_relocalization_required(
            {**affirmative(), 'reference_region': {'view': 'left', 'bbox': [.2, .3, .5, .8]}}, result))

    def test_supported_localization_failures_require_observation(self):
        for reason in ('mixed_target_depth', 'insufficient_valid_target_depth',
                       'too_few_target_pixels', 'insufficient_door_frame_support'):
            with self.subTest(reason=reason):
                self.assertTrue(localization_failure(reason)['endpoint_relocalization_required'])

    def test_second_jamb_failure_also_requires_relocalization(self):
        raw = affirmative()
        raw['reference_region'] = {'view': 'right', 'bbox': [.2, .2, .5, .8], 'entity_id': None}
        result = localization_failure()
        result['endpoint_reference_grounding'] = result['endpoint_grounding']
        result['endpoint_grounding'] = {'valid': True, 'reason': 'ok', 'endpoint_kind': 'doorway', 'distance_m': .8}
        self.assertTrue(endpoint_relocalization_required(raw, result))

    def test_negative_probes_missing_regions_and_incomplete_routes_are_excluded(self):
        for update in ({'outcome': False}, {'failure_type': 'wrong_path'}, {'target_view': None},
                       {'target_bbox': None}, {'target_bbox': [0, 0, 0, 1]}, {'scene_consistent': False}):
            self.assertFalse(endpoint_relocalization_required({**affirmative(), **update}, localization_failure()))
        for name, value in (('condition_contract_conflict', True), ('rollback_required', True),
                            ('condition_checks', []), ('doorway_checks', [{'execution_witnessed': False}])):
            self.assertFalse(endpoint_relocalization_required(affirmative(), {**localization_failure(), name: value}))
        failed_route = localization_failure()
        failed_route['condition_checks'][0]['status'] = 'unknown'
        self.assertFalse(endpoint_relocalization_required(affirmative(), failed_route))

    def test_far_targets_and_geometry_vetoes_do_not_request_scanning(self):
        far = localization_failure()
        far['endpoint_grounding']['distance_m'] = 3.
        # Invalid mixed pixels may be background through an opening, not a
        # reliable far jamb. They still merit bounded localization repair.
        self.assertTrue(endpoint_relocalization_required(affirmative(), far))
        far['endpoint_grounding']['frame_support'] = {'left': {'valid': True, 'distance_m': 3.}}
        self.assertFalse(endpoint_relocalization_required(affirmative(), far))
        del far['endpoint_grounding']['frame_support']
        far['endpoint_reference_grounding'] = {'valid': True, 'distance_m': 3.}
        self.assertFalse(endpoint_relocalization_required(affirmative(), far))
        for reason in ('target_surface_too_far', 'invalid_target_bbox', 'measured_endpoint_geometry_required'):
            result = localization_failure(reason)
            self.assertFalse(result['endpoint_relocalization_required'])


class RelocalizationDispatchTests(unittest.TestCase):
    def test_two_regions_on_same_wall_fail_geometry_and_brake_before_forward(self):
        from test_v3_endpoint_integration import grounded_guard
        guard, raw = grounded_guard('Walk to the doorway. Stop in the doorway.')
        raw.update(target_entity_id=None, target_view='front', target_bbox=[.4, .4, .6, .6],
                   reference_region={'view': 'front', 'bbox': [.2, .4, .4, .6], 'entity_id': None,
                                     'role': 'door_jamb'})
        result = guard.evaluate(raw)
        self.assertFalse(result['outcome'])
        self.assertIs(result['endpoint_grounding']['geometry_verified'], False)
        self.assertEqual(result['reason'], 'measured_endpoint_geometry_required')
        self.assertFalse(result.get('endpoint_confirmation_required', False))
        result['endpoint_relocalization_required'] = endpoint_relocalization_required(raw, result)
        self.assertTrue(result['endpoint_relocalization_required'])
        a = integrated_agent([result, result, result])
        for step in (2, 3, 4):
            a._observation_revision += 1
            control_tests.PLANNER['step'](a, step)
            a._terminal_probe_due.return_value = False
        types = [call.args[0].type for call in a.execute_action.call_args_list]
        self.assertEqual(types[:2], ['turn', 'turn'])
        self.assertNotIn(types[-1], ('turn', 'move_forward', 'stop'))
        self.assertEqual(a.current_path, [])
        self.assertEqual(a.navigation_trace[-1]['source'], 'endpoint_relocalization_exhausted')

    def test_probe_pauses_waypoint_before_any_forward_step_and_forces_next_verification(self):
        a = integrated_agent([localization_failure(), boundary_tests.verdict()])
        control_tests.PLANNER['step'](a, 2)
        self.assertEqual(a.execute_action.call_args.args[0].type, 'turn')
        self.assertEqual(a.execute_action.call_args.args[0].r, 0)
        self.assertEqual(a.current_path, [])
        self.assertIsNotNone(a._pending_endpoint_relocalization)
        a._terminal_probe_due.return_value = False
        a._observation_revision += 1
        control_tests.PLANNER['step'](a, 3)
        self.assertEqual(a._verify_current_subtask.call_count, 2)
        self.assertEqual([call.args[0].type for call in a.execute_action.call_args_list], ['turn', 'stop'])
        self.assertIsNone(a._pending_endpoint_relocalization)

    def test_completion_origin_keeps_its_own_verification_budget(self):
        a = integrated_agent([localization_failure(), boundary_tests.verdict()])
        first = a._boundary_stop_decision({}, None, [], None, {}, boundary_tests.ACTIONS)
        self.assertEqual(first['type'], 'turn')
        a._terminal_probe_due.return_value = False
        a._observation_revision += 1
        control_tests.PLANNER['step'](a, 3)
        self.assertEqual(a._boundary_checks['P1']['calls'], 2)
        self.assertFalse(getattr(a, '_probe_checks', {}))
        self.assertFalse(a.decide_waypoint.call_args.kwargs['is_terminal_probe'])

    def test_two_failed_looks_hand_back_without_third_turn_or_forward(self):
        a = integrated_agent([localization_failure() for _ in range(3)])
        a.navigation_cfg['endpoint_relocalization_max_turns'] = 100  # Hard upper bound remains two.
        for step in (2, 3, 4):
            a._observation_revision += 1
            control_tests.PLANNER['step'](a, step)
            a._terminal_probe_due.return_value = False
        types = [call.args[0].type for call in a.execute_action.call_args_list]
        self.assertEqual(types[:2], ['turn', 'turn'])
        self.assertNotIn(types[-1], ('turn', 'move_forward', 'stop'))
        self.assertEqual(a._verifier_corrections['P1'], 1)
        self.assertIsNone(a._pending_endpoint_relocalization)
        self.assertEqual(a.navigation_trace[-1]['source'], 'endpoint_relocalization_exhausted')

    def test_pending_probe_budget_exhaustion_cannot_resume_old_route(self):
        a = integrated_agent([localization_failure()])
        a.navigation_cfg['arrival_probe_max_calls'] = 1
        control_tests.PLANNER['step'](a, 2)
        a._observation_revision += 1
        with self.assertRaisesRegex(NavigationAborted, 'endpoint_relocalization_verifier_budget_exhausted'):
            control_tests.PLANNER['step'](a, 3)
        self.assertEqual(a.execute_action.call_count, 1)
        self.assertEqual(a.execute_action.call_args.args[0].type, 'turn')

    def test_exhausted_relocalization_uses_finite_correction_budget(self):
        a = boundary_tests.agent(localization_failure())
        a.navigation_cfg.update(endpoint_relocalization_max_turns=0, max_verifier_corrections_per_subtask=1)
        self.assertEqual(boundary_tests.boundary(a)['type'], 'noop')
        with self.assertRaisesRegex(NavigationAborted, 'verifier_correction_budget_exhausted'):
            boundary_tests.boundary(a)

    def test_true_negative_probe_does_not_interrupt_forward_route(self):
        result = boundary_tests.verdict(outcome=False, completion_evidence_reliable=False,
                                        failure_type='not_reached', evidence_state='unknown')
        a = integrated_agent([result])
        control_tests.PLANNER['step'](a, 2)
        self.assertEqual(a.execute_action.call_args.args[0].type, 'move_forward')
        self.assertFalse(getattr(a, '_pending_endpoint_relocalization', None))

    def test_real_verifier_records_diagnostics_for_relocalization_followup(self):
        a = boundary_tests.agent()
        del a._verify_current_subtask
        a._get_forward_clearance_and_waypoints = Mock(return_value=(None, []))
        seen = []
        a._verification_prompt = lambda *args, **kwargs: seen.append(
            copy.deepcopy(getattr(a, '_verification_history', {}))) or 'prompt'
        a._get_verifier_response = Mock(return_value=json.dumps(affirmative()))
        a._evaluate_route_verification = Mock(return_value=localization_failure())
        a.instruction_obj.update_verifier = Mock()
        for _ in range(2):
            a._verify_current_subtask({'_controller_probe': True}, None, [], None, obs={}, actions=boundary_tests.ACTIONS)
        history = seen[1]['P1'][0]
        self.assertTrue(history['endpoint_relocalization_required'])
        self.assertEqual(history['endpoint_grounding']['reason'], 'mixed_target_depth')
        self.assertEqual(history['endpoint_grounding']['bbox'], [.1, .2, .4, .8])


def confirmation_required():
    return boundary_tests.verdict(
        outcome=False, relation_satisfied=False, completion_evidence_reliable=False,
        failure_type='not_reached', reason='changed_view_confirmation_required',
        endpoint_confirmation_required=True,
        endpoint_grounding={'valid': True, 'geometry_verified': True, 'distance_m': .8,
                            'gate_reason': 'changed_view_confirmation_required'})


class ConfirmationDispatchTests(unittest.TestCase):
    def test_real_two_frame_grounding_is_rechecked_before_next_forward_step(self):
        from test_v3_endpoint_integration import EndpointIntegrationTests
        first, second = EndpointIntegrationTests()._between_frames()
        a = integrated_agent([first, second])
        control_tests.PLANNER['step'](a, 2)
        self.assertEqual(a.current_path, [])
        self.assertIsNone(a._waypoint_execution)
        self.assertIsNotNone(a._pending_endpoint_confirmation)
        a._terminal_probe_due.return_value = False
        a._observation_revision += 1
        control_tests.PLANNER['step'](a, 3)
        self.assertEqual(a._verify_current_subtask.call_count, 2)
        self.assertEqual([call.args[0].type for call in a.execute_action.call_args_list], ['turn', 'stop'])
        self.assertIsNone(a._pending_endpoint_confirmation)
        self.assertIsNone(a._pending_final_stop_confirmation)

    def test_completion_confirmation_retains_completion_budget(self):
        a = integrated_agent([confirmation_required(), boundary_tests.verdict()])
        a._pending_final_stop_confirmation = {'key': 'P1'}
        self.assertEqual(a._boundary_stop_decision({}, None, [], None, {}, boundary_tests.ACTIONS)['type'], 'turn')
        self.assertEqual(a.current_path, [])
        a._terminal_probe_due.return_value = False
        a._observation_revision += 1
        control_tests.PLANNER['step'](a, 3)
        self.assertEqual(a._boundary_checks['P1']['calls'], 2)
        self.assertFalse(getattr(a, '_probe_checks', {}))
        self.assertFalse(a.decide_waypoint.call_args.kwargs['is_terminal_probe'])

    def test_two_confirmation_turns_hand_back_and_do_not_resume_old_route(self):
        a = integrated_agent([confirmation_required() for _ in range(3)])
        a.navigation_cfg['endpoint_confirmation_max_turns'] = 100
        for step in (2, 3, 4):
            a._observation_revision += 1
            control_tests.PLANNER['step'](a, step)
            a._terminal_probe_due.return_value = False
        types = [call.args[0].type for call in a.execute_action.call_args_list]
        self.assertEqual(types[:2], ['turn', 'turn'])
        self.assertNotIn(types[-1], ('turn', 'move_forward', 'stop'))
        self.assertEqual(a.current_path, [])
        self.assertEqual(a._verifier_corrections['P1'], 1)
        self.assertIsNone(a._pending_endpoint_confirmation)
        self.assertIsNone(a._pending_final_stop_confirmation)
        self.assertEqual(a.navigation_trace[-1]['source'], 'endpoint_confirmation_exhausted')

    def test_new_position_gets_bounded_confirmation_within_existing_call_budget(self):
        a = integrated_agent([confirmation_required() for _ in range(4)])
        for step in (2, 3, 4):
            a._observation_revision += 1
            control_tests.PLANNER['step'](a, step)
            a._terminal_probe_due.return_value = False
        a.mapper.current_position[0] += .5  # A real planner approach, not another rotation.
        a._terminal_probe_due.return_value = True
        a._observation_revision += 1
        control_tests.PLANNER['step'](a, 5)
        self.assertEqual(a.execute_action.call_args.args[0].type, 'turn')
        self.assertEqual(a._endpoint_confirmation_turns['P1'], 1)
        self.assertEqual(a._probe_checks['P1']['calls'], 4)
        a._observation_revision += 1
        with self.assertRaisesRegex(NavigationAborted, 'endpoint_confirmation_verifier_budget_exhausted'):
            control_tests.PLANNER['step'](a, 6)
        self.assertEqual(a.execute_action.call_count, 4)

    def test_rotation_and_small_pose_noise_do_not_replenish_confirmation_turns(self):
        a = integrated_agent([confirmation_required() for _ in range(3)])
        for step in (2, 3):
            a._observation_revision += 1
            control_tests.PLANNER['step'](a, step)
        a.mapper.current_position[0] += .03
        a.mapper.current_rotation[:] = [[0., -1, 0], [1, 0, 0], [0, 0, 1]]
        a._observation_revision += 1
        control_tests.PLANNER['step'](a, 4)
        self.assertNotEqual(a.execute_action.call_args.args[0].type, 'turn')
        self.assertEqual(a._endpoint_confirmation_turns['P1'], 2)

    def test_new_view_withdraws_claim_and_hands_back_without_forward(self):
        negative = boundary_tests.verdict(outcome=False, completion_evidence_reliable=False,
                                          failure_type='not_reached', evidence_state='unknown')
        a = integrated_agent([confirmation_required(), negative])
        control_tests.PLANNER['step'](a, 2)
        a._terminal_probe_due.return_value = False
        a._observation_revision += 1
        control_tests.PLANNER['step'](a, 3)
        self.assertEqual(a.decide_waypoint.call_count, 2)
        self.assertNotIn(a.execute_action.call_args.args[0].type, ('move_forward', 'turn', 'stop'))
        self.assertIsNone(a._pending_endpoint_confirmation)
        self.assertEqual(a.navigation_trace[-1]['source'], 'endpoint_confirmation_exhausted')

    def test_confirmation_verifier_failure_cannot_resume_route(self):
        a = integrated_agent([confirmation_required(), boundary_tests.verdict(available=False, outcome=False)])
        control_tests.PLANNER['step'](a, 2)
        a._observation_revision += 1
        with self.assertRaisesRegex(NavigationAborted, 'verification_unavailable'):
            control_tests.PLANNER['step'](a, 3)
        self.assertEqual(a.execute_action.call_count, 1)

    def test_pending_confirmation_budget_exhaustion_cannot_resume_route(self):
        a = integrated_agent([confirmation_required()])
        a.navigation_cfg['arrival_probe_max_calls'] = 1
        control_tests.PLANNER['step'](a, 2)
        a._observation_revision += 1
        with self.assertRaisesRegex(NavigationAborted, 'endpoint_confirmation_verifier_budget_exhausted'):
            control_tests.PLANNER['step'](a, 3)
        self.assertEqual(a.execute_action.call_count, 1)
        self.assertIsNone(a._pending_final_stop_confirmation)

    def test_confirmation_handback_uses_existing_correction_budget(self):
        a = boundary_tests.agent(confirmation_required())
        a.navigation_cfg.update(endpoint_confirmation_max_turns=0, max_verifier_corrections_per_subtask=1)
        self.assertEqual(boundary_tests.boundary(a)['type'], 'noop')
        with self.assertRaisesRegex(NavigationAborted, 'verifier_correction_budget_exhausted'):
            boundary_tests.boundary(a)

    def test_semantic_refresh_marker_alone_does_not_authorize_controller_stop(self):
        a = integrated_agent([])
        a._terminal_probe_due.return_value = False
        a._pending_final_stop_confirmation = {'key': 'P1'}
        control_tests.PLANNER['step'](a, 2)
        a._verify_current_subtask.assert_not_called()
        self.assertEqual(a.execute_action.call_args.args[0].type, 'move_forward')


if __name__ == '__main__':
    unittest.main()


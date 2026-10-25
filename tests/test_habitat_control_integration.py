"""Production endpoint dispatch and structural guards, CPU without models."""
import ast
import copy
import json
import math
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from agent.habitat_control import HabitatControl
from test_v3_verification_contract import Guard, verdict
from structural_endpoints import current_floor_depth
from progress_watchdog import NavigationAborted


SOURCE = Path(__file__).resolve().parents[1]/'src/agent/agent.py'
tree = ast.parse(SOURCE.read_text())
body = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'PathPlannerAgent')
methods = [node for node in body.body if isinstance(node, ast.FunctionDef) and node.name in (
    '_boundary_stop_decision', '_verification_accepts_stop', '_sim_forward_executable')]
namespace = {'np': np, 'math': math, 'local_to_global': lambda *args: (_ for _ in ()).throw(RuntimeError('offline'))}
exec(compile(ast.Module(body=methods, type_ignores=[]), str(SOURCE), 'exec'), namespace)


class Harness(HabitatControl):
    _boundary_stop_decision = namespace['_boundary_stop_decision']
    _verification_accepts_stop = namespace['_verification_accepts_stop']
    _sim_forward_executable = namespace['_sim_forward_executable']

    def __init__(self, verification):
        self.navigation_cfg = {'habitat_control_enabled': True}
        self.mapper = NS(current_position=np.zeros(3), initial_floor_height=0., observation_id=1)
        self.instruction_obj = NS(get_current_subtask=lambda: ('P1', {}), is_last_subtask=lambda: True)
        self.navigation_trace = []
        self.current_path, self.action_sequence = [[1, 0, 0]], []
        self._execution_ledger = []
        self.verification = verification

    def _verify_current_subtask(self, *args, **kwargs):
        value = copy.deepcopy(self.verification)
        if value.get('endpoint_grounding'):
            value['endpoint_grounding']['observation_id'] = self.mapper.observation_id
        return value

    def tick(self):
        return self._boundary_stop_decision({'_controller_probe': True}, None, [], None, {}, {})

    def turn(self, action):
        self._execution_ledger.append({'step': len(self._execution_ledger), 'action_type': 'turn',
            'yaw_delta': -action['angle'], 'motion_distance': 0., 'pose_before': [0., 0., 0.], 'pose': [0., 0., 0.]})
        self.mapper.observation_id += 1


class EndpointDispatchTests(unittest.TestCase):
    def test_current_guard_acceptance_still_needs_real_fresh_confirmation(self):
        a = Harness({'available': True, 'outcome': True, 'relation_satisfied': True, 'scene_consistent': True,
            'completion_evidence_reliable': True, 'candidate_claimed': True,
            'endpoint_grounding': {'valid': True, 'accepted': True, 'geometry_verified': True,
                'world_center': [1., 0., 0.], 'distance_m': 1., 'target_id': 2}})
        action = a.tick()
        self.assertEqual(action['type'], 'turn')
        self.assertEqual(a.current_path, [])
        self.assertEqual(a.tick()['type'], 'turn')  # same image cannot corroborate itself
        a.turn(action)
        self.assertEqual(a.tick()['type'], 'stop')

    def test_unbound_regions_request_physical_escape_instead_of_wandering(self):
        a = Harness({'available': True, 'outcome': False, 'candidate_claimed': True,
                     'endpoint_grounding': {'valid': False, 'reason': 'mixed_target_depth'}})
        for _ in range(2):
            action = a.tick()
            self.assertEqual(action['type'], 'turn')
            a.turn(action)
        self.assertEqual(a.tick()['type'], 'noop')
        self.assertEqual(a._endpoint_escape_requested['subtask_id'], 'P1')
        self.assertEqual(a._endpoint_escape_requested['first_affected_step'], 0)
        self.assertEqual(a.current_path, [])

    def test_negative_probe_without_candidate_preserves_valid_navigation(self):
        a = Harness({'available': True, 'outcome': False, 'candidate_claimed': False})
        self.assertEqual(a.tick()['type'], 'continue_path')
        self.assertEqual(a.current_path, [[1, 0, 0]])

    def test_pending_session_cannot_bypass_the_global_verifier_budget(self):
        a = Harness({'available': True, 'outcome': False, 'candidate_claimed': True})
        a.navigation_cfg['habitat_max_verification_calls_per_subtask'] = 1
        self.assertEqual(a.tick()['type'], 'turn')
        with self.assertRaisesRegex(NavigationAborted, 'habitat_verification_budget_exhausted'):
            a.tick()
        self.assertEqual(a._habitat_verifier_calls['P1'], 1)

    def test_local_session_checks_budget_before_sending_an_extra_request(self):
        a = Harness({'available': True, 'outcome': False, 'candidate_claimed': True})
        a.navigation_cfg['endpoint_session_max_calls'] = 2
        a.tick()
        a.tick()
        self.assertEqual(a._habitat_verifier_calls['P1'], 2)
        self.assertEqual(a.tick()['type'], 'noop')
        self.assertEqual(a._habitat_verifier_calls['P1'], 2)
        self.assertIn('budget_exhausted', a._endpoint_escape_requested['reason'])

    def test_failed_physical_preflight_cannot_authorize_forward_motion(self):
        a = Harness({})
        self.assertFalse(a._sim_forward_executable(np.zeros(3), np.eye(3), .2, np.zeros(3)))

    def test_physical_stair_rejection_continues_only_the_requested_certified_direction(self):
        a = Harness({'available': True, 'outcome': False, 'candidate_claimed': True})
        a.instruction = 'Stop at the bottom of the stairs.'
        a._habitat_measurements = lambda: {'platform': {'valid': False, 'reason': 'narrow_support'},
            'profile': {'valid': True, 'ascending': True, 'descending': True}}
        a._validated_waypoint_route = lambda point: {'path': [np.zeros(3), point], 'target': point}
        actions = {'WP_DOWN': {'type': 'waypoint', 'target_point_world': [.4, 0., -.2]},
                   'WP_UP': {'type': 'waypoint', 'target_point_world': [.4, 0., .2]}}
        result = a._boundary_stop_decision({'_controller_probe': True}, None, [], None, {}, actions)
        self.assertEqual(result['action_id'], 'WP_DOWN')
        self.assertFalse(a._endpoint_controller.active)
        self.assertEqual(a.navigation_trace[-1]['event'], 'habitat_stair_continuation')
        a._validated_waypoint_route = lambda point: None
        result = a._boundary_stop_decision({'_controller_probe': True}, None, [], None, {}, actions)
        self.assertEqual(result['type'], 'turn')


class EndpointProtocolRepairTests(unittest.TestCase):
    def measured(self, accepted=False):
        return {'available': True, 'outcome': accepted, 'relation_satisfied': accepted,
                'scene_consistent': True, 'completion_evidence_reliable': accepted,
                'candidate_claimed': True, 'endpoint_grounding': {
                    'valid': True, 'accepted': accepted, 'geometry_verified': True,
                    'world_center': [1., 0., 0.], 'distance_m': 1., 'target_id': None}}

    def citation_failure(self):
        return {'available': True, 'outcome': False, 'relation_satisfied': False,
                'scene_consistent': True, 'completion_evidence_reliable': False,
                'candidate_claimed': True, 'evidence_repair_required': True,
                'reason': 'Route condition coverage or execution references are invalid.',
                'condition_checks': [{'id': 'P1.C2', 'status': 'unknown', 'steps': []}],
                'raw': '{"condition_checks":[{"id":"P1.C2","steps":[20],"observation_id":null}]}'}

    def observed_candidate(self):
        a = Harness(self.measured())
        a._verify_current_subtask = Mock(wraps=a._verify_current_subtask)
        for _ in range(2):
            action = a.tick()
            self.assertEqual(action['type'], 'turn')
            a.turn(action)
        return a

    def responses(self, a, values):
        values = iter(values)
        def verify(*args, **kwargs):
            result = copy.deepcopy(next(values))
            if result.get('endpoint_grounding'):
                result['endpoint_grounding']['observation_id'] = a.mapper.observation_id
            return result
        a._verify_current_subtask.side_effect = verify

    def test_protocol_repair_after_two_looks_can_pass_full_guard_without_extra_motion(self):
        a = self.observed_candidate()
        bad = self.citation_failure(); original = copy.deepcopy(bad)
        self.responses(a, [bad, self.measured(accepted=True)])
        before = (a.mapper.observation_id, len(a._execution_ledger))
        self.assertEqual(a.tick()['type'], 'stop')
        self.assertEqual(before, (a.mapper.observation_id, len(a._execution_ledger)))
        self.assertEqual(a._endpoint_controller.calls, 4)
        self.assertEqual(a._habitat_verifier_calls['P1'], 4)
        self.assertEqual(a._endpoint_controller.looks, 2)
        self.assertEqual(a._endpoint_controller.protocol_repairs, 1)
        self.assertEqual(a._verify_current_subtask.call_count, 4)
        self.assertEqual(bad, original)  # No local rewriting of future citations.
        self.assertFalse(getattr(a, '_endpoint_escape_requested', None))

    def test_persistent_protocol_failure_does_not_request_a_physical_escape(self):
        a = self.observed_candidate()
        self.responses(a, [self.citation_failure(), self.citation_failure()])
        with self.assertRaisesRegex(NavigationAborted, 'endpoint_protocol_repair_exhausted'):
            a.tick()
        self.assertEqual(a._endpoint_controller.calls, 4)
        self.assertEqual(a._endpoint_controller.looks, 2)
        self.assertFalse(getattr(a, '_endpoint_escape_requested', None))
        self.assertEqual(a.navigation_trace[-1]['kind'], 'protocol_failure')

    def test_protocol_retry_respects_session_global_and_runtime_request_budgets(self):
        for budget in ('session', 'global', 'runtime'):
            with self.subTest(budget=budget):
                a = self.observed_candidate()
                self.responses(a, [self.citation_failure(), self.measured(accepted=True)])
                if budget == 'session':
                    from endpoint_controller import EndpointConfig
                    a._endpoint_controller.config = EndpointConfig(max_calls=3)
                    reason = 'endpoint_protocol_repair_budget_exhausted'
                elif budget == 'global':
                    a.navigation_cfg['habitat_max_verification_calls_per_subtask'] = 3
                    reason = 'habitat_verification_budget_exhausted'
                else:
                    a.runtime_budget = Mock()
                    a.runtime_budget.check.side_effect = [None, NavigationAborted('request_budget')]
                    reason = 'request_budget'
                with self.assertRaisesRegex(NavigationAborted, reason):
                    a.tick()
                self.assertEqual(a._verify_current_subtask.call_count, 3)
                self.assertEqual(a._habitat_verifier_calls['P1'], 3)
                self.assertFalse(getattr(a, '_endpoint_escape_requested', None))

    def test_same_frame_protocol_retry_cannot_supply_a_changed_view(self):
        a = Harness(self.measured())
        self.assertEqual(a.tick()['type'], 'turn')  # Deliberately not executed.
        a._verify_current_subtask = Mock()
        self.responses(a, [self.citation_failure(), self.measured(accepted=True)])
        self.assertEqual(a.tick()['type'], 'turn')
        self.assertEqual(a.mapper.observation_id, 1)
        self.assertEqual(a._endpoint_controller.looks, 0)
        self.assertEqual(a._endpoint_controller.calls, 3)
        self.assertEqual(a._endpoint_controller._last_decision.reason, 'independent_changed_view_required')

    def test_repaired_citations_cannot_override_current_geometry_rejection(self):
        a = self.observed_candidate()
        rejected = self.measured(accepted=False)
        rejected['endpoint_grounding'].update(valid=False, geometry_verified=False,
                                              reason='mixed_target_depth')
        self.responses(a, [self.citation_failure(), rejected])
        self.assertEqual(a.tick()['type'], 'noop')
        self.assertEqual(a._endpoint_controller._last_decision.kind, 'recover')
        self.assertEqual(a._endpoint_controller.looks, 2)
        self.assertFalse(any(r.get('kind') == 'ready' for r in a.navigation_trace))


class HybridGuard(HabitatControl, Guard):
    pass


class StructuralGuardTests(unittest.TestCase):
    def make(self):
        a = HybridGuard('Stop at the bottom of the stairs.')
        a.navigation_cfg.update(habitat_control_enabled=True, require_endpoint_grounding=True)
        a.mapper.current_position = np.array([0., 0., 1.2])
        a.mapper.observation_id = 5
        a.mapper.camera_intrinsic = np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]])
        a.mapper.object_entities = []
        a.mapper.current_rotation = np.eye(3)
        points = np.array([[0., 0., .2], [.1, 0., .2]])

        def inputs():
            rotation = a.mapper.current_rotation
            camera = {'position': [0., 0., 1.2], 'rotation': rotation}
            depth = np.full((100, 100), 1.2)
            for point in (points-[0, 0, 1.2]) @ rotation:
                x = int(round(point[0]*100/-point[2]+50))
                y = int(round(99-(point[1]*100/-point[2]+50)))
                depth[y-1:y+2, x-1:x+2] = -point[2]
            return {'front': depth}, {'front': camera}

        def measured():
            base = {'valid': True, 'current': True, 'checked_ground': [0., 0., 0.],
                    'observation_id': a.mapper.observation_id}
            depths, cameras = inputs()
            return {'ground': np.zeros(3), 'observation_id': a.mapper.observation_id,
                'platform': dict(base), 'profile': dict(base, ascending=True, descending=False,
                    ascending_points=points.tolist(), descending_points=[]),
                'floor': current_floor_depth(depths, cameras, a.mapper.camera_intrinsic, [0, 0, 0], a.mapper.observation_id)}

        a._habitat_camera_inputs = inputs
        a._habitat_measurements = measured
        return a

    def raw(self, a):
        raw = verdict(a.prepare())
        raw.update(target_identity='The bottom of the visible stair flight', target_view='front',
                   target_bbox=[.35, .3, .8, .7], target_evidence='Stairs rise beside the current flat floor.')
        raw['condition_checks'][0].update(steps=[], observation_id=a.mapper.observation_id)
        return raw

    def test_two_current_views_bind_floor_and_visible_rising_stairs(self):
        a = self.make()
        first = a.evaluate(self.raw(a))
        self.assertFalse(first['outcome'])
        self.assertEqual(first['reason'], 'changed_view_confirmation_required')
        self.assertTrue(first['endpoint_grounding']['floor_depth']['valid'])
        angle = math.radians(12)
        a.mapper.current_rotation = np.array([[math.cos(angle), -math.sin(angle), 0.],
            [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
        a.mapper.observation_id += 1
        result = a.evaluate(self.raw(a))
        self.assertTrue(result['outcome'], result.get('reason'))
        self.assertTrue(result['endpoint_grounding']['accepted'])
        self.assertEqual(result['endpoint_grounding']['source'], 'habitat_structural_landing')

    def test_wrong_image_region_and_actual_tread_cannot_stop(self):
        a = self.make()
        raw = self.raw(a)
        raw['target_bbox'] = [0., 0., .2, .2]
        self.assertFalse(a.evaluate(raw)['outcome'])
        measured = a._habitat_measurements
        def tread():
            data = measured(); data['platform']['valid'] = False; return data
        a._habitat_measurements = tread
        result = a.evaluate(self.raw(a))
        self.assertFalse(result['outcome'])
        self.assertEqual(result['reason'], 'platform_not_current_or_supported')

    def test_current_platform_interrupts_a_live_route_before_the_periodic_interval(self):
        a = self.make()
        a.navigation_cfg.update(arrival_check_enabled=True, arrival_probe_min_interval_steps=8)
        a._terminal_probe_checks = {'P1': {'calls': 1, 'step': 17}}
        a._is_near_endpoint_or_threshold = lambda **kwargs: False
        self.assertTrue(a._terminal_probe_due(21))
        measured = a._habitat_measurements
        def tread():
            value = measured(); value['platform']['valid'] = False; return value
        a._habitat_measurements = tread
        self.assertFalse(a._terminal_probe_due(21))

    def test_traversal_candidate_needs_real_signed_progress_but_pure_arrival_does_not(self):
        a = self.make()
        self.assertTrue(a._habitat_terminal_candidate())
        a.instruction = 'take the stairs then head for the kitchen. stop at the bottom of the stairs.'
        a._execution_ledger = []
        self.assertFalse(a._habitat_terminal_candidate())
        a._execution_ledger = [{'subtask_id': 'P1', 'height_delta': -.5}]
        self.assertTrue(a._habitat_terminal_candidate())
        a._execution_ledger[0]['recovery'] = True
        self.assertFalse(a._habitat_terminal_candidate())

    def test_visual_geometry_can_settle_in_parallel_without_accepting_completion(self):
        a = self.make()
        raw = self.raw(a)
        result = {'outcome': False, 'stair_completion_ready': False,
                  'reason': 'Measured stair direction or supported landing is not yet established.'}
        a._habitat_observe_unsettled_stair(raw, result)
        self.assertFalse(result['outcome'])
        self.assertTrue(result['support_settling_geometry_observed'])
        self.assertFalse(result['endpoint_grounding']['accepted'])
        angle = math.radians(12)
        a.mapper.current_rotation = np.array([[math.cos(angle), -math.sin(angle), 0.],
            [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
        a.mapper.observation_id += 1
        final = a.evaluate(self.raw(a))
        self.assertTrue(final['outcome'], final.get('reason'))

    def test_physical_candidate_arms_observation_without_a_positive_model_verdict(self):
        a = Harness({'available': True, 'outcome': False, 'candidate_claimed': False})
        a._habitat_terminal_candidate = lambda: True
        self.assertEqual(a.tick()['type'], 'turn')
        self.assertTrue(a._endpoint_controller.active)

    def test_unsettled_observation_does_not_fill_missing_semantic_identity(self):
        for identity in ('unknown', 'contradicted', 'no', 'failed'):
            with self.subTest(identity=identity):
                a = self.make()
                raw = self.raw(a); raw['target_identity'] = identity
                result = {'outcome': False, 'stair_completion_ready': False,
                          'reason': 'Measured stair direction or supported landing is not yet established.'}
                a._habitat_observe_unsettled_stair(raw, result)
                self.assertNotIn('endpoint_grounding', result)
                self.assertFalse(getattr(a, '_habitat_stair_observations', {}))

    def test_real_verifier_retry_reuses_frame_and_rechecks_route_and_depth(self):
        from test_v3_boundary_height import Agent
        a = self.make()
        a.current_path, a.action_sequence, a._execution_ledger = [], [], []
        a._verification_accepts_stop = MethodType(Harness._verification_accepts_stop, a)
        a._verify_current_subtask = MethodType(Agent._verify_current_subtask, a)
        a.instruction_obj.update_verifier = Mock()
        a._get_forward_clearance_and_waypoints = Mock(return_value=(None, []))
        prompts = []
        def prompt(*args, **kwargs):
            value = a._generate_verification_prompt(a.prepare(), None)
            prompts.append(value)
            return value
        a._verification_prompt = prompt
        raw_responses = []
        def response(*args, **kwargs):
            raw = self.raw(a)
            if len(raw_responses) == 1:
                raw['condition_checks'][0].update(steps=[20], observation_id=None)
            raw_responses.append(copy.deepcopy(raw))
            return json.dumps(raw)
        a._get_verifier_response = Mock(side_effect=response)
        first = a._habitat_boundary_stop_decision({'_controller_probe': True}, None, [], None, {}, {})
        self.assertEqual(first['type'], 'turn')
        angle = first['angle']
        turn = np.array([[math.cos(angle), -math.sin(angle), 0.],
                         [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
        a.mapper.current_rotation = turn @ a.mapper.current_rotation
        a.mapper.observation_id += 1
        a._execution_ledger.append({'step': 0, 'controller_step': 0, 'action_type': 'turn',
            'yaw_delta': angle, 'motion_distance': 0., 'pose_before': [0., 0., 0.], 'pose': [0., 0., 0.]})
        second = a._habitat_boundary_stop_decision({'_controller_probe': True}, None, [], None, {}, {})
        self.assertEqual(second['type'], 'stop')
        self.assertEqual(a._get_verifier_response.call_count, 3)
        self.assertEqual(a._endpoint_controller.calls, 3)
        self.assertEqual(a._endpoint_controller.looks, 1)
        self.assertEqual(a.mapper.observation_id, 6)
        self.assertEqual(len(a._execution_ledger), 1)
        self.assertEqual(raw_responses[1]['condition_checks'][0]['steps'], [20])
        self.assertEqual(raw_responses[2]['condition_checks'][0]['steps'], [])
        self.assertIn('Route condition coverage or execution references are invalid.', prompts[2])
        trace = [r for r in a.navigation_trace if r.get('role') == 'verifier']
        self.assertEqual([r['current_observation_id'] for r in trace], [5, 6, 6])
        self.assertIsNone(trace[1]['endpoint_grounding'])
        self.assertTrue(trace[2]['endpoint_grounding']['geometry_verified'])
        self.assertTrue(trace[2]['endpoint_grounding']['accepted'])


if __name__ == '__main__':
    unittest.main()


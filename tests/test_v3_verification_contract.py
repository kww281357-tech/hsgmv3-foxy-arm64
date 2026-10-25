"""CPU-only regressions for route evidence and bounded observation probes.

These import the production guard/contracts, without Habitat, models or API calls.
"""
import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from agent.route_guard import RouteEvidenceGuard
from navigation_contracts import completion_conditions, validate_condition_checks
from navigation_policy import normalize_verification


class Guard(RouteEvidenceGuard):
    def __init__(self, instruction='Walk to the chair. Stop beside it.'):
        self.instruction = instruction
        self.st = {'instruction': instruction, 'record': [], 'completed': False}
        self.instruction_obj = NS(
            get_current_subtask=lambda: ('P1', self.st),
            get_current_subtask_key=lambda: 'P1',
            get_subtask_by_key=lambda key: self.st,
            is_last_subtask=lambda: True, sub_instruction_keys=['P1'],
            current_subtask_index=0)
        self.mapper = NS(current_position=np.array([.5, 0., 1.2]),
                         current_rotation=np.eye(3), initial_floor_height=-1.2,
                         trajectory_position=[np.array([0., 0., 1.2]), np.array([.5, 0., 1.2])],
                         floor_level=0)
        # Route contract tests isolate endpoint geometry; dedicated integration
        # tests enable grounding and supply measured depth/target candidates.
        self.navigation_cfg = {'require_endpoint_grounding': False}
        self.navigation_trace = []
        self.traveled_distance = .5
        self.entities = []
        self.stair_ready = True
        self.stair_packet = {'direction': 'NONE', 'measured_height_change_m': 0.}
        self._observation_revision = 0
        self._normalize_angle = lambda x: (x + math.pi) % (2 * math.pi) - math.pi

    def _observed_target_candidates(self, text, limit=10):
        self.queried_endpoint = text
        return self.entities[:limit]

    def _ensure_stair_progress(self):
        return NS(packet=lambda: self.stair_packet, completion_ready=lambda: self.stair_ready)

    def prepare(self):
        return self._prepare_route_verification({})

    def evaluate(self, raw):
        return self._evaluate_route_verification(raw, normalize_verification(copy.deepcopy(raw)))


def record(step, kind='move_forward', pose=(.5, 0., 0.), motion=.5, yaw=0.):
    return {'step': step, 'executed_action': kind, 'action_type': kind,
            'motion_distance': motion, 'yaw_delta': yaw, 'height_delta': 0.,
            'pose_before': [0., 0., 0.], 'pose': list(pose)}


def verdict(context):
    return {'outcome': True, 'relation_satisfied': True, 'scene_consistent': True,
            'confidence': 'high', 'evidence': 'Current views establish the route and endpoint.',
            'reason': 'The requested endpoint is reached.', 'failure_type': 'none',
            'target_identity': 'Chair beside the window', 'target_proximity': 'satisfied',
            'target_entity_id': None, 'target_evidence': 'Chair is beside the current pose in RGB/depth.',
            'condition_checks': [dict(id=c['id'], status='satisfied',
                                     evidence='Current view and executed motion.', steps=[1])
                                 for c in context['checks']]}


class RouteContractTests(unittest.TestCase):
    def setUp(self):
        self.a = Guard()
        self.a.st['record'] = [record(1)]
        self.ctx = self.a.prepare()
        self.raw = verdict(self.ctx)

    def test_short_observed_route_can_pass_without_minimum_distance_heuristic(self):
        result = self.a.evaluate(self.raw)
        self.assertTrue(result['completion_evidence_reliable'])
        self.assertEqual(result['scope'], 'current_segment')
        self.assertFalse(self.a.st['completed'])

    def test_intermediate_arrival_uses_history_final_arrival_uses_current_pose(self):
        a = Guard('Stand beside the oven. Walk toward the chair. Stop beside the chair.')
        a.st['record'] = [record(1, pose=(0., 0., 0.)), record(2)]
        raw = verdict(a.prepare())
        raw['condition_checks'][0]['steps'] = [1]
        for row in raw['condition_checks'][1:]:
            row['steps'] = [2]
        self.assertTrue(a.evaluate(raw)['completion_evidence_reliable'])
        raw['condition_checks'][-1]['steps'] = [1]
        self.assertFalse(a.evaluate(raw)['completion_evidence_reliable'])

    def test_missing_or_duplicate_rows_never_pass(self):
        for rows in (self.raw['condition_checks'][:1], self.raw['condition_checks'][:1] * 2):
            with self.subTest(rows=rows):
                raw = dict(self.raw, condition_checks=rows)
                result = self.a.evaluate(raw)
                self.assertFalse(result['outcome'])
                self.assertTrue(result['condition_contract_conflict'])

    def test_normalized_failed_clause_and_unknown_both_reject(self):
        for status in ('failed', 'violated', 'unknown'):
            with self.subTest(status=status):
                raw = copy.deepcopy(self.raw)
                raw['condition_checks'][-1]['status'] = status
                self.assertFalse(self.a.evaluate(raw)['completion_evidence_reliable'])

    def test_episode_722_unwitnessed_final_turn_and_stop_cannot_pass(self):
        a = Guard('When you leave the sitting area and go to the dinning area, make a left '
                  'then make another left near the outside door. Now go into the kitchen '
                  'and make a right. You should stop between the island and counter.')
        a.st['record'] = [record(1), record(2, 'turn', motion=0., yaw=.5)]
        raw = verdict(a.prepare())
        self.assertEqual(len(raw['condition_checks']), 6)
        for row in raw['condition_checks']:
            row['steps'] = [1, 2]
        raw['condition_checks'][-1]['status'] = 'unknown'
        before = copy.deepcopy(a.st['completion_conditions'])
        result = a.evaluate(raw)
        self.assertFalse(result['outcome'])
        self.assertFalse(result['completion_evidence_reliable'])
        self.assertEqual(a.st['completion_conditions'], before)

    def test_each_endpoint_evidence_field_must_be_explicit(self):
        for field in ('target_identity', 'target_proximity', 'target_evidence'):
            for value in ('missing', '', None, True):
                with self.subTest(field=field, value=value):
                    raw = copy.deepcopy(self.raw)
                    if value == 'missing':
                        raw.pop(field)
                    else:
                        raw[field] = value
                    self.assertFalse(self.a.evaluate(raw)['outcome'])

    def test_null_detector_id_allowed_but_unknown_nonnull_id_requires_policy(self):
        self.assertTrue(self.a.evaluate(self.raw)['outcome'])
        self.raw['target_entity_id'] = 27
        self.assertFalse(self.a.evaluate(self.raw)['outcome'])
        self.a.navigation_cfg['require_strict_target_id'] = False
        self.assertTrue(self.a.evaluate(self.raw)['outcome'])

    def test_negative_verdict_never_promotes_its_claimed_conditions(self):
        before = copy.deepcopy(self.a.st['completion_conditions'])
        raw = dict(self.raw, outcome=False, relation_satisfied=False, failure_type='not_reached')
        self.a.evaluate(raw)
        self.assertEqual(self.a.st['completion_conditions'], before)
        self.assertFalse(self.a.st['completed'])

    def test_weak_wrong_path_claim_cannot_trigger_rollback(self):
        raw = dict(self.raw, outcome=False, relation_satisfied=False, failure_type='wrong_path',
                   scene_consistent=False, reason='I cannot see the chair.')
        result = self.a.evaluate(raw)
        self.assertFalse(result['rollback_required'])
        self.assertEqual(result['evidence_state'], 'unknown')
        self.assertIsNone(result['scene_consistent'])

    def test_explicit_wrong_path_contradiction_and_clause_can_trigger_recovery(self):
        a = Guard('Enter the kitchen. Stop beside the counter.')
        a.st['record'] = [record(1)]
        raw = dict(verdict(a.prepare()), outcome=False, relation_satisfied=False, failure_type='wrong_path',
                   scene_consistent=False, contradiction_evidence='Entered a bedroom with a bed and nightstands.',
                   route_contradiction={'kind': 'wrong_room', 'condition_id': 'P1.C1', 'steps': [1],
                                        'expected': 'Enter the kitchen',
                                        'observed': 'Entered a bedroom with a bed and nightstands.'})
        raw['condition_checks'][0]['status'] = 'contradicted'
        self.assertTrue(a.evaluate(raw)['rollback_required'])

    def test_strict_doorway_policy_checks_measured_crossing(self):
        a = Guard('Go through the door. Stop outside.')
        a.st['record'] = [record(1)]
        raw = verdict(a.prepare())
        self.assertFalse(a.evaluate(raw)['outcome'])
        raw['doorway_checks'] = [{'condition_id': 'P1.C1', 'steps': [1], 'entry_side': 'bedroom',
                                 'exit_side': 'hallway', 'visual_evidence': 'Doorframe behind agent; corridor outside.'}]
        self.assertTrue(a.evaluate(raw)['outcome'])
        raw.pop('doorway_checks')
        a.navigation_cfg['require_strict_doorway_checks'] = False
        self.assertTrue(a.evaluate(raw)['outcome'])

    def test_absolute_or_episode_height_does_not_bypass_stair_state(self):
        self.a.stair_ready = False
        self.a.stair_packet = {'direction': 'DOWN', 'measured_height_change_m': .8,
                               'direction_violation': False}
        self.a.navigation_cfg['require_strict_stair_landing'] = False
        self.a.prepare()
        self.assertFalse(self.a.evaluate(self.raw)['outcome'])


class ExplicitWitnessTests(unittest.TestCase):
    def test_neighboring_forward_step_does_not_validate_cited_turn(self):
        conditions = completion_conditions('Walk down the hall.')
        rows = [{'id': 'P1.C1', 'status': 'satisfied', 'evidence': 'Moved forward.', 'steps': [1]}]
        checked, _ = validate_condition_checks(conditions, rows, [record(1, 'turn', motion=0., yaw=.5), record(2)])
        self.assertEqual(checked[0]['status'], 'unknown')

    def test_prose_ranges_do_not_add_uncited_steps(self):
        conditions = completion_conditions('Walk down the hall.')
        rows = [{'id': 'P1.C1', 'status': 'satisfied', 'evidence': 'Moved at steps 1-2.', 'steps': [1]}]
        checked, _ = validate_condition_checks(conditions, rows, [record(1, 'turn', motion=0., yaw=.5), record(2)])
        self.assertEqual(checked[0]['status'], 'unknown')

    def test_malformed_step_ids_do_not_turn_into_other_numbers(self):
        conditions = completion_conditions('Walk to the chair.')
        for steps in ([True], [1.5], ['1.5'], ['step 1'], '1', ['0-999']):
            rows = [{'id': 'P1.C1', 'status': 'satisfied', 'evidence': 'Moved.', 'steps': steps}]
            checked, conflict = validate_condition_checks(conditions, rows, [record(1), record(5)])
            self.assertTrue(conflict, steps)
            self.assertEqual(checked[0]['status'], 'unknown')

    def test_arrival_requires_cited_current_pose_including_floor(self):
        conditions = completion_conditions('Stop near the chair.')
        ledger = [record(1, pose=(.5, 0, 2)), record(2)]
        for steps in ([], [1]):
            rows = [{'id': 'P1.C1', 'status': 'satisfied', 'evidence': 'Chair beside agent.', 'steps': steps}]
            checked, _ = validate_condition_checks(conditions, rows, ledger,
                require_current_arrival=True, current_pose=[.5, 0, 0])
            self.assertEqual(checked[0]['status'], 'unknown')


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.a = Guard('Pass the sofa. Stop by the chair.')
        self.a.navigation_cfg = {'arrival_check_enabled': True}
        self.a.entities = [{'id': 1, 'horizontal_distance': 1.}]
        self.a.st['record'] = [record(1)]

    def test_queries_final_landmark_not_earlier_transit_object(self):
        self.assertTrue(self.a._terminal_probe_due(4))
        self.assertNotIn('sofa', self.a.queried_endpoint)
        self.assertIn('chair', self.a.queried_endpoint)

    def test_no_probe_budget_spent_at_start_before_required_motion(self):
        self.a.st['record'] = []
        self.assertFalse(self.a._terminal_probe_due(1, at_route_boundary=True))

    def test_long_route_endpoint_extraction_does_not_merge_transit_landmarks(self):
        text = ('Walk ahead. Turn left. Go into the room. Turn right. Pass the sofa. '
                'Go by the table. Turn left. Stop by the chair.')
        self.assertEqual(self.a._endpoint_clause(text), 'Stop by the chair.')

    def test_repeated_identical_views_not_probed_even_when_stalled(self):
        self.a._claim_terminal_probe(4)
        self.assertFalse(self.a._terminal_probe_due(5))
        self.assertFalse(self.a._terminal_probe_due(100, stalled=True))

    def test_movement_rotation_and_delayed_new_observation_can_retry(self):
        self.a._claim_terminal_probe(4)
        self.a.mapper.current_position[0] += .5
        self.assertTrue(self.a._terminal_probe_due(6))
        self.a._claim_terminal_probe(6)
        self.a.mapper.current_rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        self.assertTrue(self.a._terminal_probe_due(8))
        self.a._claim_terminal_probe(8)
        self.a._observation_revision += 1
        self.assertFalse(self.a._terminal_probe_due(10, stalled=True))
        self.assertTrue(self.a._terminal_probe_due(16, stalled=True))

    def test_probes_bounded_and_disabled_during_recovery_or_other_segment(self):
        for step in range(4):
            self.a._claim_terminal_probe(step * 10)
        self.a.mapper.current_position[0] += 2
        self.assertFalse(self.a._terminal_probe_due(100))
        self.a._terminal_probe_checks = {}
        self.a.rolling_back = True
        self.assertFalse(self.a._terminal_probe_due(100))
        self.a.rolling_back = False
        self.a.instruction_obj.is_last_subtask = lambda: False
        self.assertFalse(self.a._terminal_probe_due(100))

    def test_generic_waypoint_arrival_is_not_an_endpoint_cue(self):
        self.a.entities = []
        self.a.target_world_position = [.5, 0., 0.]
        self.assertFalse(self.a._terminal_probe_due(10, at_route_boundary=True))

    def test_threshold_near_end_of_route_can_trigger_without_detector(self):
        self.a = Guard('Cross the hall. Stop in the doorway.')
        self.a.navigation_cfg = {'arrival_check_enabled': True}
        self.a.target_world_position = [1., 0., 0.]
        self.a.current_path = [[1., 0., 0.]]
        self.a.st['record'] = [record(1)]
        self.assertTrue(self.a._terminal_probe_due(10))


class BranchConstraintTests(unittest.TestCase):
    def test_first_strong_event_filters_all_matching_waypoints_but_keeps_turns(self):
        a = Guard()
        a.mapper.current_position = np.array([0., 0., 1.2])
        a._failed_branches = [{'subtask_id': 'P1', 'floor': 0, 'origin': [0, 0, 0],
                               'angle': 0., 'hits': 1, 'strong': True, 'expires': 12}]
        actions = {'WP_01': {'type': 'waypoint', 'target_point_world': [1, 0, 0]},
                   'TURN_LEFT': {'type': 'turn', 'angle': 1.57}}
        self.assertEqual(set(a._constrain_branch_actions(actions)), {'TURN_LEFT'})
        a._failed_branches[0]['strong'] = False
        a._failed_branches[0]['hits'] = 9
        self.assertEqual(set(a._constrain_branch_actions(actions)), set(actions))


if __name__ == '__main__':
    unittest.main()


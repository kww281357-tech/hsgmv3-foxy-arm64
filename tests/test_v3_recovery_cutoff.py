"""CPU production regressions for return-before-contradiction selection.

The simulator and route backend are deterministic stubs. Selection, execution,
route evidence and recovery lifecycle are the production implementations.
"""
import copy
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace as NS

import numpy as np

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
from route_progress import check_route_contradiction, checkpoint_precedes_contradiction
from navigation_contracts import completion_conditions
from test_v3_route_progress import contradiction, move, row
from test_v3_verification_contract import record
from test_v3_navigation_control import extract, NS_GLOBALS, ACTIONS
from test_v3_recovery_lifecycle import harness, Action
from agent.route_guard import RouteEvidenceGuard


METHODS = extract(SRC / 'agent/agent.py', {'plan_rollback_path', '_direct_verifier_correction'},
                  dict(NS_GLOBALS, copy=copy))


def constraint(lower=0, upper=17, cid='P1.C1'):
    return {'condition_id': cid, 'kind': 'straight_path', 'cited_steps': [17, 26],
            'condition_boundary_step': lower - 1, 'earliest_affected_step': upper,
            'min_checkpoint_ledger_len': lower, 'max_checkpoint_ledger_len': upper}


def checkpoint(step, x=2., floor=0):
    return {'subtask_key': 'P1', 'floor_level': floor, 'ledger_len': step, 'step': step,
            'ground_pose': [x, 0., -1.2], 'mapper_pose': [x, 0., 0.],
            'rotation': np.eye(3).tolist(), 'is_junction': True, 'rollback_visits': 0}


def controller():
    a = harness(enable_wrong_path_rollback=True, rollback_max_path_m=8.)
    for name, method in METHODS.items():
        setattr(a, name, method.__get__(a))
    a.init_pos, a.init_rot = np.array([0., 0., -1.2]), np.eye(3)
    a.mapper.current_position = np.array([4., 0., 0.])
    a.instruction_obj.sub_instruction_keys = ['P1']
    a._branch_checkpoints = []
    a._validated_waypoint_route = Mock(side_effect=lambda target, recovery=False: {
        'path': [a._recovery_ground_pose(), np.asarray(target)], 'reachable': True})
    return a


class ContradictionBoundaryTests(unittest.TestCase):
    def test_late_citations_cannot_hide_the_first_turn_of_a_straight_clause(self):
        # Ep626 cited 17/26, but its measured step0 alignment was already 45deg.
        ledger = [record(0, 'turn', motion=0., yaw=math.pi/4),
                  move(1, (0., 0.), (.5, 0.), 0.),
                  record(17, 'turn', motion=0., yaw=-2.192),
                  record(26, 'turn', motion=0., yaw=1.365)]
        checks = completion_conditions('Walk straight.')
        checks[0].update(status='contradicted', steps=[17, 26])
        result, error = check_route_contradiction(contradiction('straight_path', steps=[17, 26],
            expected='Walk straight', observed='The measured route turned left then right.'), checks, ledger)
        self.assertIsNone(error)
        bounds = result['recovery_constraint']
        self.assertEqual(bounds['max_checkpoint_ledger_len'], 0)
        self.assertFalse(checkpoint_precedes_contradiction(checkpoint(36), bounds, 'P1'))
        self.assertTrue(checkpoint_precedes_contradiction(checkpoint(0), bounds, 'P1'))

    def test_prior_verified_clause_is_a_lower_boundary(self):
        ledger = [record(3, 'turn', motion=0., yaw=-1.57),
                  move(4, (0., 0.), (.5, 0.), 0.), record(17, 'turn', motion=0., yaw=1.57),
                  move(18, (.5, 0.), (.5, .5), 1.57)]
        checks = completion_conditions('Turn left. Walk straight.')
        checks[0].update(status='satisfied', execution_checked=True, steps=[3])
        checks[1].update(status='contradicted', steps=[18])
        result, error = check_route_contradiction(contradiction('straight_path', cid='P1.C2', steps=[18],
            expected='Walk straight', observed='The path bent right.'), checks, ledger)
        self.assertIsNone(error)
        self.assertEqual(result['recovery_constraint'], {**constraint(4, 17, 'P1.C2'), 'cited_steps': [18]})

    def test_turn_cutoff_owns_the_complete_physical_block(self):
        ledger = [record(4, 'turn', motion=0., yaw=.4), record(5, 'turn', motion=0., yaw=.4)]
        checks = completion_conditions('Turn left.')
        checks[0].update(status='contradicted', steps=[5])
        result, error = check_route_contradiction(contradiction('turn_direction', steps=[5],
            expected='Turn left', observed='The robot rotated right.'), checks, ledger)
        self.assertIsNone(error)
        self.assertEqual(result['recovery_constraint']['max_checkpoint_ledger_len'], 4)

    def test_late_turn_cannot_hide_earlier_sideways_translation(self):
        ledger = [move(0, (0., 0.), (0., .5), 0.), record(17, 'turn', motion=0., yaw=.8)]
        checks = completion_conditions('Walk straight.')
        checks[0].update(status='contradicted', steps=[17])
        result, error = check_route_contradiction(contradiction('straight_path', steps=[17],
            expected='Walk straight', observed='The route moves sideways and turns right.'), checks, ledger)
        self.assertIsNone(error)
        self.assertEqual(result['recovery_constraint']['max_checkpoint_ledger_len'], 0)

    def test_semantic_room_claim_does_not_reclassify_an_earlier_observation_turn(self):
        ledger = [record(0, 'turn', motion=0., yaw=1.57),
                  move(70, (0., 0.), (.2, 0.), 0.), move(75, (.2, 0.), (.4, 0.), 0.)]
        checks = completion_conditions('Enter the kitchen.')
        checks[0].update(status='contradicted', steps=[70, 75])
        result, error = check_route_contradiction(contradiction(steps=[70, 75]), checks, ledger)
        self.assertIsNone(error)
        self.assertEqual(result['recovery_constraint']['max_checkpoint_ledger_len'], 70)

    def test_rejected_attempt_is_not_a_new_verified_prefix_boundary(self):
        old = record(0, 'turn', motion=0., yaw=.8)
        old['rejected_by_rollback'] = True
        ledger = [old, record(10, 'turn', motion=0., yaw=.8), move(11, (0., 0.), (.3, 0.), .8)]
        checks = completion_conditions('Walk straight.')
        checks[0].update(status='contradicted', steps=[10, 11])
        result, error = check_route_contradiction(contradiction('straight_path', steps=[10, 11],
            expected='Walk straight', observed='The new route turns right.'), checks, ledger)
        self.assertIsNone(error)
        bounds = result['recovery_constraint']
        self.assertEqual(bounds['min_checkpoint_ledger_len'], 0)
        self.assertTrue(checkpoint_precedes_contradiction(checkpoint(0), bounds, 'P1'))

    def test_missing_or_malformed_checkpoint_boundary_fails_closed(self):
        for cp, bounds in [(checkpoint(36), constraint()), (checkpoint(1), constraint(4)),
                           ({'subtask_key': 'P1'}, constraint()), (checkpoint(True), constraint()),
                           (checkpoint(1), constraint(cid='P2.C1')), (checkpoint(1), None),
                           (checkpoint(1), {**constraint(), 'condition_id': None})]:
            self.assertFalse(checkpoint_precedes_contradiction(cp, bounds, 'P1'))


class CheckpointSelectionTests(unittest.TestCase):
    def test_selects_before_bad_evidence_instead_of_latest_reachable_checkpoint(self):
        a = controller()
        a._branch_checkpoints = [checkpoint(10), checkpoint(36, x=2.5)]
        path, _ = a.plan_rollback_path('P1', recovery_constraint=constraint())
        self.assertEqual(a._active_rollback_checkpoint['step'], 10)
        np.testing.assert_allclose(path[-1], [2., 0., -1.2])

    def test_no_eligible_checkpoint_is_explicit_and_does_not_mutate_history(self):
        a = controller()
        a._branch_checkpoints = [checkpoint(36, x=2.)]
        a.init_pos = np.array([-10., 0., -1.2])  # entry exceeds physical 8m budget
        a._execution_ledger = [record(17), record(26)]
        before = copy.deepcopy((a._execution_ledger, a.history_msgs, a.st))
        self.assertEqual(a.plan_rollback_path('P1', constraint()), (None, None))
        self.assertIsNone(a._active_rollback_checkpoint)
        self.assertEqual((a._execution_ledger, a.history_msgs, a.st), before)
        self.assertEqual(a.navigation_trace[-1]['event'], 'wrong_path_rollback_unavailable')
        self.assertIn('exceeds', a.navigation_trace[-1]['reason'])

    def test_verified_prefix_cannot_be_erased_by_entry_fallback(self):
        a = controller()
        a._branch_checkpoints = [checkpoint(2), checkpoint(36)]
        self.assertEqual(a.plan_rollback_path('P1', constraint(4)), (None, None))
        a._validated_waypoint_route.assert_not_called()

    def test_same_floor_and_reachability_filters_still_apply(self):
        a = controller()
        a._branch_checkpoints = [checkpoint(10, floor=1), checkpoint(12)]
        a._validated_waypoint_route.side_effect = lambda *args, **kwargs: None
        self.assertEqual(a.plan_rollback_path('P1', constraint(4)), (None, None))
        self.assertTrue(a._validated_waypoint_route.called)
        self.assertTrue(all(np.allclose(call.args[0], [2., 0., -1.2])
                            for call in a._validated_waypoint_route.call_args_list))
        a.mapper.floor_level = 1
        a._branch_checkpoints = []
        self.assertEqual(a.plan_rollback_path('P1', constraint(0, 0)), (None, None))

    def test_initial_entry_is_a_real_before_action_zero_checkpoint(self):
        a = controller()
        path, _ = a.plan_rollback_path('P1', constraint(0, 0))
        self.assertIsNotNone(path)
        self.assertEqual(a._active_rollback_checkpoint['ledger_len'], 0)
        self.assertEqual(a._active_rollback_checkpoint['floor_level'], 0)

    def test_manual_or_stalled_selection_has_no_stale_wrong_path_cutoff(self):
        a = controller()
        a._branch_checkpoints = [checkpoint(10), checkpoint(36, x=2.5)]
        a.plan_rollback_path('P1', constraint())
        a.plan_rollback_path('P1')
        self.assertEqual(a._active_rollback_checkpoint['step'], 36)

    def test_production_correction_carries_verified_constraint_to_execution(self):
        a = controller()
        a._branch_checkpoints = [checkpoint(10), checkpoint(36, x=2.5)]
        result = a._direct_verifier_correction({'failure_type': 'wrong_path', 'rollback_required': True,
            'route_contradiction_gate': {'accepted': True},
            'route_contradiction': {'recovery_constraint': constraint()}}, ACTIONS)
        self.assertEqual(result['type'], 'rollback')
        self.assertEqual(result['recovery_source'], 'wrong_path')
        self.assertEqual(result['recovery_constraint'], constraint())
        self.assertEqual(a._active_rollback_checkpoint['step'], 10)


class RecoveryBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, {'simWrapper': NS(PolarAction=Action)})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_stale_preplanned_checkpoint_fails_before_any_history_invalidation(self):
        a = harness()
        a.execute_action(Action(.2, 0, 'move_forward'))
        before = copy.deepcopy((a._execution_ledger, a.st['record'], a.history_msgs))
        cp = checkpoint(36, x=2.)
        self.assertFalse(a._begin_checkpoint_recovery([a._recovery_ground_pose(), cp['ground_pose']],
            np.eye(3), cp, subtask_key='P1', recovery_constraint=constraint(), recovery_source='wrong_path'))
        self.assertEqual((a._execution_ledger, a.st['record'], a.history_msgs), before)
        self.assertFalse(a.rolling_back)
        self.assertEqual(a.navigation_trace[-1]['event'], 'wrong_path_rollback_failed')

    def test_preplanned_floor_change_or_unexecuted_boundary_cannot_begin(self):
        for floor, bounds in [(1, constraint(0, 0)), (0, constraint(0, 17)), (0, None)]:
            with self.subTest(floor=floor, bounds=bounds):
                a = harness()
                a.execute_action(Action(.2, 0, 'move_forward'))
                before = copy.deepcopy(a._execution_ledger)
                cp = checkpoint(0, x=2., floor=floor)
                self.assertFalse(a._begin_checkpoint_recovery([a._recovery_ground_pose(), cp['ground_pose']],
                    np.eye(3), cp, subtask_key='P1', recovery_constraint=bounds, recovery_source='wrong_path'))
                self.assertEqual(a._execution_ledger, before)

    def test_success_and_cancel_keep_measured_history_and_prefix(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                a = harness(rollback_position_tolerance=.06)
                for _ in range(3):
                    a.execute_action(Action(.2, 0, 'move_forward'))
                a.st['completion_conditions'] = [
                    {'id': 'P1.C1', 'status': 'satisfied', 'execution_checked': True, 'steps': [0]},
                    {'id': 'P1.C2', 'status': 'contradicted', 'execution_checked': False, 'steps': [2]}]
                cp = checkpoint(1, x=.2)
                cp['rotation'] = a.mapper.current_rotation.tolist()
                self.assertTrue(a._begin_checkpoint_recovery([a._recovery_ground_pose(), cp['ground_pose']],
                    a.mapper.current_rotation, cp, subtask_key='P1',
                    recovery_constraint=constraint(1, 2, 'P1.C2'), recovery_source='wrong_path'))
                self.assertEqual(a.st['completion_conditions'][0]['status'], 'satisfied')
                self.assertFalse(a._execution_ledger[0].get('rejected_by_rollback', False))
                self.assertTrue(all(r.get('rejected_by_rollback') for r in a._execution_ledger[1:3]))
                self.assertEqual(a.st['completion_conditions'][1]['status'], 'unknown')
                if cancel:
                    a._cancel_checkpoint_recovery('route invalidated')
                    self.assertIsNone(a._last_rollback_info)
                else:
                    for step in range(12):
                        a._step_recovery(step)
                        if a._pending_recovery is None:
                            break
                    self.assertEqual(a.navigation_trace[-1]['status'], 'completed')
                    self.assertLessEqual(a.navigation_trace[-1]['position_error_m'], .06)
                self.assertGreaterEqual(len(a._execution_ledger), 3)
                self.assertGreaterEqual(a.traveled_distance, .6 - 1e-8)
                self.assertEqual(a.navigation_trace[-1]['checkpoint_ledger_len'], 1)
                self.assertEqual(a.navigation_trace[-1]['recovery_source'], 'wrong_path')


class EntryBranchMemoryTests(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, {'simWrapper': NS(PolarAction=Action)})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def entry_agent(self):
        a = harness(enable_wrong_path_rollback=True, rollback_position_tolerance=.1)
        a.plan_rollback_path = METHODS['plan_rollback_path'].__get__(a)
        a._remember_failed_branch = RouteEvidenceGuard._remember_failed_branch.__get__(a)
        a.init_pos = a._recovery_ground_pose().copy()
        a.init_rot = a.mapper.current_rotation.copy()
        a._branch_checkpoints = []
        a.instruction_obj.sub_instruction_keys = ['P1']
        a._requested_action = {'type': 'navigate', 'waypoint_id': 'WP_22'}
        a.execute_action(Action(0., -math.pi/4, 'turn'))
        # The exact Ep626 step0 execution packet; positions remain simulator
        # measurements and selection/return/branch exclusion are production.
        a._execution_ledger[0]['waypoint_execution'] = {
            'action_id': 'WP_22', 'phase': 'aligning', 'target': [.375, .375, -1.2]}
        for _ in range(3):
            a.execute_action(Action(.2, 0., 'move_forward'))
        return a

    def test_successful_entry_return_remembers_real_initial_waypoint_not_distant_pose(self):
        a = self.entry_agent()
        path, rotation = a.plan_rollback_path('P1', constraint(0, 0))
        cp = a._active_rollback_checkpoint
        self.assertEqual(cp['chosen_target'], [.375, .375, -1.2])
        self.assertEqual(cp['chosen_target_evidence']['waypoint_phase'], 'aligning')
        self.assertTrue(a._begin_checkpoint_recovery(path, rotation, cp, subtask_key='P1',
            recovery_constraint=constraint(0, 0), recovery_source='wrong_path'))
        self.assertEqual(getattr(a, '_failed_branches', []), [])
        self.assertEqual(a._blocked_targets, [])
        for step in range(15):
            a._step_recovery(step)
            if a._pending_recovery is None:
                break
        self.assertEqual(a.navigation_trace[-1]['status'], 'completed')
        self.assertEqual(len(a._failed_branches), 1)
        self.assertAlmostEqual(a._failed_branches[0]['angle'], math.pi/4)
        self.assertTrue(a._failed_branches[0]['strong'])
        self.assertEqual(a._blocked_targets[0]['target'], [.375, .375, -1.2])
        self.assertGreater(np.linalg.norm(np.asarray(a._blocked_targets[0]['target'])[:2]
                            - np.asarray(a._blocked_targets[0]['origin'])[:2]), .15)
        actions = {'bad': {'type': 'waypoint', 'target_point_world': [.375, .375, -1.2]},
                   'good': {'type': 'waypoint', 'target_point_world': [.5, 0., -1.2]}}
        self.assertEqual(set(a._constrain_branch_actions(actions)), {'good'})
        self.assertIsNone(a._validated_waypoint_route(actions['bad']['target_point_world']))
        self.assertIsNotNone(a._validated_waypoint_route(actions['good']['target_point_world']))

    def test_cancelled_entry_return_does_not_commit_entry_branch_memory(self):
        a = self.entry_agent()
        path, rotation = a.plan_rollback_path('P1', constraint(0, 0))
        self.assertTrue(a._begin_checkpoint_recovery(path, rotation, a._active_rollback_checkpoint,
            subtask_key='P1', recovery_constraint=constraint(0, 0), recovery_source='wrong_path'))
        a._cancel_checkpoint_recovery('forward segment rejected')
        self.assertEqual(a.navigation_trace[-1]['status'], 'failed')
        self.assertEqual(getattr(a, '_failed_branches', []), [])
        self.assertEqual(a._blocked_targets, [])
        self.assertEqual(len(a._execution_ledger), 4)

    def test_unknown_or_unrelated_entry_choice_is_not_invented(self):
        for change in ('missing', 'different_action', 'away_from_entry', 'rejected'):
            with self.subTest(change=change):
                a = self.entry_agent()
                row = a._execution_ledger[0]
                if change == 'missing':
                    row.pop('waypoint_execution')
                elif change == 'different_action':
                    row['waypoint_execution']['action_id'] = 'WP_OTHER'
                elif change == 'away_from_entry':
                    row['pose_before'] = [2., 2., -1.2]
                else:
                    row['rejected_by_rollback'] = True
                a.plan_rollback_path('P1', constraint(0, 0))
                self.assertNotIn('chosen_target', a._active_rollback_checkpoint)


if __name__ == '__main__':
    unittest.main()


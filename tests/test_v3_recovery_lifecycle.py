"""CPU recovery regressions using production route and measured execution code.

Only the simulator, perception update and renderer dependencies are replaced.
No Habitat process, model, API, GPU or dataset is used.
"""
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
from agent.recovery import CheckpointRecovery
import test_v3_waypoint_reachability as fixtures

Action = fixtures.Action


class RecoveryController(fixtures.Controller):
    """The two tiny adapters match the hooks required in agent.py."""
    def _invalidate_blocked_path(self, reason):
        super()._invalidate_blocked_path(reason)
        if getattr(self, '_pending_recovery', None):
            self._cancel_checkpoint_recovery(reason)

    def execute_action(self, action):
        super().execute_action(action)
        self._observe_recovery_action(action)


def harness(**config):
    a = fixtures.controller()
    a.__class__ = RecoveryController
    a.navigation_cfg = dict(config)
    a.mapper.floor_level = 0
    a.current_path, a.action_sequence = [], []
    a._pending_recovery = None
    a._last_step_index = 0
    a.st = {'instruction': 'Walk along the corridor.', 'record': [], 'completion_conditions': []}
    a.instruction_obj.get_current_subtask = lambda: ('P1', a.st)
    a.instruction_obj.get_subtask_by_key = lambda key: a.st
    a.instruction_obj.add_record_to_current_subtask = lambda row: a.st['record'].append(row)
    a.instruction_obj.current_step_count = 0
    a._get_agent_yaw = lambda: fixtures.checkpoint_yaw(a.mapper.current_rotation)
    a._rot_to_yaw = fixtures.checkpoint_yaw
    a._remember_failed_branch = Mock()
    a.decide_waypoint = Mock(side_effect=AssertionError('Recovery must not call planner'))
    a.history_msgs = ['before branch', 'wrong branch narrative']
    a._last_verifier_result, a._verifier_feedback = {'outcome': False}, {'reason': 'old direction'}
    a.block_translation = False
    state = a.sim_wrapper.sim.get_agent(0).get_state()
    state.rotation = None

    def move(action):
        yaw = a._get_agent_yaw()
        if action.type == 'turn':
            yaw -= action.theta
        elif action.type == 'move_forward' and not a.block_translation:
            state.position = np.asarray(state.position) + [action.r * math.cos(yaw), 0, action.r * math.sin(yaw)]
        a.mapper.current_rotation = np.array([[0., 0, -math.cos(yaw)],
                                              [0., 0, -math.sin(yaw)], [0, -1., 0]])
        return {'agent_state': state}

    def update(obs):
        p = obs['agent_state'].position
        a.mapper.current_position = np.array([p[0], p[2], p[1]])
        return []

    a.sim_wrapper.step = move
    a.update_map = update
    a.curr_obs = {'agent_state': state}
    return a


def begin(a, distance=1.745, heading=0, checkpoint=None):
    ground = a._recovery_ground_pose()
    target = ground + [distance, 0, 0]
    rotation = np.array([[0., 0, -math.cos(heading)], [0, 0, -math.sin(heading)], [0, -1, 0]])
    cp = checkpoint if checkpoint is not None else {
        'subtask_key': 'P1', 'ground_pose': target.tolist(),
        'mapper_pose': (target - [0, 0, a.mapper.initial_floor_height]).tolist(),
        'rotation': rotation.tolist(), 'ledger_len': 0, 'step': 0, 'history_len': 1,
        'chosen_target': ground.tolist(), 'chosen_view': 'RIGHT', 'floor_level': 0,
    }
    return a._begin_checkpoint_recovery([ground, target], rotation, cp, 'Wrong branch was observed.', 'P1')


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, {'simWrapper': NS(PolarAction=Action)})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def terminal_events(self, a):
        return [r for r in a.navigation_trace if r.get('event') in
                ('wrong_path_rollback_completed', 'wrong_path_rollback_failed')]

    def assert_clean(self, a):
        self.assertIsNone(a._pending_recovery)
        self.assertFalse(a.rolling_back)
        self.assertFalse(a._need_restore_heading)
        self.assertIsNone(a._waypoint_execution)
        self.assertEqual(a.current_path, [])
        self.assertEqual(a.action_sequence, [])

    def test_ep1395_return_completes_from_measured_pose_once(self):
        a = harness()
        self.assertTrue(begin(a))
        self.assertIsNone(a._last_rollback_info)
        for step in range(10):
            a._step_recovery(step)
            if a._pending_recovery is None:
                break
        self.assert_clean(a)
        self.assertEqual(len(self.terminal_events(a)), 1)
        event = self.terminal_events(a)[0]
        self.assertEqual(event['status'], 'completed')
        self.assertAlmostEqual(event['horizontal_error_m'], 0, places=6)
        self.assertAlmostEqual(a.traveled_distance, 1.745, places=6)
        self.assertEqual(len(a._execution_ledger), 5)  # No duplicated recovery ledger.
        self.assertTrue(all(r['recovery'] for r in a._execution_ledger))
        self.assertEqual(a._last_rollback_info['status'], 'completed')
        self.assertEqual(a._rollback_counts, {'P1': 1})
        a.decide_waypoint.assert_not_called()
        a.execute_action(Action(.2, 0, 'move_forward'))
        self.assertFalse(a._execution_ledger[-1]['recovery'])

    def test_ep1159_blocked_return_does_not_poison_following_motion(self):
        a = harness()
        self.assertTrue(begin(a, distance=1.83))
        for length in (.5, .5, .296):
            a.execute_action(Action(length, 0, 'move_forward'))
        self.assertAlmostEqual(a._recovery_error(a._pending_recovery)['horizontal_error_m'], .534)
        a.sim_wrapper.sim.pathfinder.inside = lambda p: p[0] < 1.30
        self.assertEqual(a._recovery_action(54).type, 'null')
        self.assert_clean(a)
        self.assertEqual(self.terminal_events(a)[0]['status'], 'failed')
        self.assertAlmostEqual(self.terminal_events(a)[0]['horizontal_error_m'], .534)
        self.assertIsNone(a._last_rollback_info)
        self.assertEqual(a._last_recovery_context['status'], 'failed')
        a.sim_wrapper.sim.pathfinder.inside = lambda p: True
        a.execute_action(Action(.5, 0, 'move_forward'))
        self.assertFalse(a._execution_ledger[-1]['recovery'])
        self.assertFalse(a._execution_ledger[-1].get('rejected_by_rollback', False))

    def test_caller_cannot_commit_before_position_or_heading_arrival(self):
        for distance, heading in ((1.7, 0), (0, math.pi / 2)):
            with self.subTest(distance=distance, heading=heading):
                a = harness()
                self.assertTrue(begin(a, distance=distance, heading=heading))
                self.assertFalse(a._finish_checkpoint_recovery(True, 'Caller claims done'))
                self.assertEqual(self.terminal_events(a)[0]['status'], 'failed')
                self.assertIsNone(a._last_rollback_info)

    def test_heading_is_restored_before_completion(self):
        a = harness()
        self.assertTrue(begin(a, distance=0, heading=math.pi / 2))
        action = a._step_recovery(0)
        self.assertEqual(action.type, 'turn')
        self.assertAlmostEqual(a._get_agent_yaw(), math.pi / 2)
        self.assert_clean(a)
        self.assertEqual(self.terminal_events(a)[0]['status'], 'completed')
        self.assertEqual(a._execution_ledger[0]['recovery'], True)

    def test_last_budgeted_action_can_arrive(self):
        a = harness(rollback_max_steps=1)
        self.assertTrue(begin(a, distance=.35))
        a._step_recovery(0)
        self.assertEqual(self.terminal_events(a)[0]['status'], 'completed')
        self.assertEqual(a._recovery_steps_total, 1)

    def test_tick_budget_failure_never_means_completion(self):
        a = harness(rollback_max_steps=2)
        self.assertTrue(begin(a, distance=3))
        a._step_recovery(0)
        a._step_recovery(1)
        self.assert_clean(a)
        self.assertEqual(self.terminal_events(a)[0]['status'], 'failed')
        self.assertEqual(a._recovery_steps_total, 2)
        self.assertGreater(self.terminal_events(a)[0]['horizontal_error_m'], 2)

    def test_total_budget_and_per_task_attempts_are_bounded(self):
        a = harness(rollback_total_step_budget=1, max_rollbacks_per_subtask=9)
        self.assertTrue(begin(a, distance=3))
        a._step_recovery(0)
        self.assertFalse(begin(a, distance=3))
        self.assertEqual(a._recovery_attempts_total, 1)
        a = harness(max_rollbacks_per_subtask=1)
        self.assertTrue(begin(a))
        a._cancel_checkpoint_recovery('Cancelled by controller')
        self.assertFalse(begin(a))
        self.assertEqual(a._rollback_counts['P1'], 1)
        self.assert_clean(a)

    def test_cancel_is_idempotent_and_does_not_cancel_later_normal_path(self):
        a = harness()
        self.assertTrue(begin(a))
        a._cancel_checkpoint_recovery('Episode interrupted')
        self.assert_clean(a)
        a.current_path = [[9, 0, -1.2]]
        a._cancel_checkpoint_recovery('Repeated cleanup')
        self.assertEqual(a.current_path, [[9, 0, -1.2]])
        self.assertEqual(len(self.terminal_events(a)), 1)

    def test_execution_interruption_cleans_transaction_and_propagates(self):
        class BudgetAbort(BaseException):
            pass
        a = harness()
        self.assertTrue(begin(a))
        a.sim_wrapper.step = Mock(side_effect=BudgetAbort('token budget'))
        with self.assertRaises(BudgetAbort):
            a._step_recovery(0)
        self.assert_clean(a)
        self.assertEqual(self.terminal_events(a)[0]['status'], 'failed')

    def test_consecutive_measured_collisions_cancel_recovery(self):
        a = harness()
        self.assertTrue(begin(a))
        a.block_translation = True
        a._step_recovery(0)
        a._step_recovery(1)
        self.assert_clean(a)
        self.assertEqual(self.terminal_events(a)[0]['status'], 'failed')
        self.assertEqual(a.traveled_distance, 0)
        a.block_translation = False
        a.execute_action(Action(.2, 0, 'move_forward'))
        self.assertFalse(a._execution_ledger[-1]['recovery'])

    def test_path_ending_without_arrival_fails(self):
        a = harness()
        self.assertTrue(begin(a))
        a.current_path = []
        self.assertEqual(a._recovery_action(0).type, 'null')
        self.assert_clean(a)
        self.assertEqual(self.terminal_events(a)[0]['status'], 'failed')

    def test_failed_begin_and_invalid_checkpoint_do_not_leave_active_state(self):
        for distance in (20, float('nan')):
            a = harness()
            self.assertFalse(begin(a, distance=distance))
            self.assert_clean(a)
            self.assertEqual(self.terminal_events(a)[0]['status'], 'failed')
        a = harness()
        self.assertFalse(a._begin_checkpoint_recovery([], np.eye(3)))
        self.assertIsNone(self.terminal_events(a)[0]['horizontal_error_m'])

    def test_only_rejected_branch_steps_and_dependent_conditions_are_invalidated(self):
        a = harness()
        for _ in range(3):
            a.execute_action(Action(.2, 0, 'move_forward'))
        records = list(a.st['record'])
        a.st['completion_conditions'] = [
            {'id': 'C1', 'status': 'satisfied', 'steps': [0], 'execution_checked': True},
            {'id': 'C2', 'status': 'satisfied', 'steps': [2], 'execution_checked': True}]
        target = a._recovery_ground_pose() + [1.745, 0, 0]
        cp = {'subtask_key': 'P1', 'ground_pose': target.tolist(), 'ledger_len': 1,
              'step': 1, 'chosen_target': a._recovery_ground_pose().tolist(), 'rollback_visits': 0}
        self.assertTrue(begin(a, checkpoint=cp))
        self.assertEqual(cp['rollback_visits'], 1)
        self.assertFalse(records[0].get('rejected_by_rollback', False))
        self.assertTrue(records[1]['rejected_by_rollback'])
        self.assertTrue(records[2]['rejected_by_rollback'])
        self.assertEqual(a.st['completion_conditions'][0]['status'], 'satisfied')
        self.assertEqual(a.st['completion_conditions'][1]['status'], 'unknown')
        self.assertEqual(len(a._execution_ledger), 3)
        self.assertEqual(len(a.st['record']), 3)
        self.assertAlmostEqual(a.traveled_distance, .6)
        a._remember_failed_branch.assert_called_once()

    def test_ground_height_uses_mapper_height_not_hardcoded_camera_height(self):
        a = harness()
        a.mapper.initial_floor_height = -.9
        self.assertTrue(begin(a, distance=0))
        a._recovery_action(0)
        self.assertEqual(self.terminal_events(a)[0]['height_error_m'], 0)
        self.assertEqual(self.terminal_events(a)[0]['target_position'][2], -.9)

    def test_lifecycle_events_always_carry_checkpoint_and_measured_errors(self):
        a = harness()
        self.assertTrue(begin(a))
        a._cancel_checkpoint_recovery('Route invalidated')
        events = [r for r in a.navigation_trace if r.get('event', '').startswith('wrong_path_rollback_')]
        self.assertEqual([r['status'] for r in events], ['started', 'failed'])
        for event in events:
            self.assertIn('checkpoint_id', event)
            self.assertIn('actual_ground_pose', event)
            self.assertIn('horizontal_error_m', event)
            self.assertIn('height_error_m', event)
            self.assertIn('heading_error_deg', event)
            self.assertTrue(event['reason'])
        self.assertEqual(len(a.st['rollback_events']), 2)


if __name__ == '__main__':
    unittest.main()


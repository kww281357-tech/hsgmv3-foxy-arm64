"""Production control-flow regressions, with no simulator/model imports."""
import math
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np

import test_v3_boundary_height as boundary_tests
import test_v3_navigation_control as control_tests
import test_v3_waypoint_reachability as fixtures
from progress_watchdog import NavigationAborted


def moving_agent():
    a = fixtures.controller()
    state = a.sim_wrapper.sim.get_agent(0).get_state()
    a.prev_agent_position = None
    a.last_action = fixtures.Action.null
    a.last_actions, a.vlm_responses, a.img_buffer = {}, [], []
    a.last_vlm_response = ''
    a.config = {'camera': {'fov': 90}}
    a.curr_obs = {'agent_state': state}
    a.instruction_obj.current_step_count = 0
    a.instruction_obj.is_last_subtask = lambda: True
    a.instruction_obj.get_subtask_by_key = lambda key: {'completed': False}
    a.instruction_obj.mark_current_subtask_completed = Mock()
    a._record_completed_subtask_memory = Mock()
    a._stop = Mock()
    a.execute_action = Mock()
    a._terminal_probe_due = Mock(return_value=True)
    a._claim_terminal_probe = Mock()
    a._get_agent_yaw = lambda: fixtures.checkpoint_yaw(a.mapper.current_rotation)
    fixtures.Action.stop = fixtures.Action(0, 0, 'stop')
    return a


class TerminalIntegrationTests(unittest.TestCase):
    def test_observed_endpoint_approach_uses_twenty_cm_stride(self):
        a = fixtures.controller()
        a.current_path = [[2., 0., -1.2]]
        a.navigation_cfg = {'arrival_check_enabled': True, 'arrival_precision_step_m': .2}
        a._is_near_endpoint_or_threshold = Mock(return_value=True)
        action = a._get_action_for_next_waypoint()
        self.assertEqual(action.type, 'move_forward')
        self.assertAlmostEqual(action.r, .2)

    def test_doorway_can_be_crossed_in_more_than_four_short_steps(self):
        from navigation_control import doorway_witnesses
        checks = [{'id': 'P1.C1', 'text': 'Walk through the doorway.'}]
        ledger = [{'step': i, 'action_type': 'move_forward', 'pose_before': [i*.1, 0, 0],
                   'pose': [(i+1)*.1, 0, 0]} for i in range(8)]
        report = {'condition_id': 'P1.C1', 'steps': list(range(8)), 'entry_side': 'kitchen',
                  'exit_side': 'hall', 'visual_evidence': 'The jamb is now behind the agent.'}
        self.assertTrue(doorway_witnesses(checks, [report], ledger)[0]['execution_witnessed'])

    def test_positive_probe_stops_before_another_forward_step(self):
        a = moving_agent()
        a.decide_waypoint = Mock(return_value=({'type': 'stop', 'verified': True}, 'verified', None))
        control_tests.PLANNER['step'](a, 2)
        self.assertEqual(a.execute_action.call_args.args[0].type, 'stop')
        a._claim_terminal_probe.assert_called_once_with(2)
        a.instruction_obj.mark_current_subtask_completed.assert_called_once()
        a.decide_waypoint.assert_called_once_with(is_stuck=False, is_reverify=True, is_terminal_probe=True)

    def test_negative_probe_preserves_certified_route(self):
        a = moving_agent()
        path_before = [list(p) for p in a.current_path]
        a.decide_waypoint = Mock(return_value=({'type': 'continue_path'}, 'not there', None))
        control_tests.PLANNER['step'](a, 2)
        self.assertEqual(a.current_path, path_before)
        self.assertEqual(a.execute_action.call_args.args[0].type, 'move_forward')
        a.instruction_obj.mark_current_subtask_completed.assert_not_called()
        a.decide_waypoint.assert_called_once()

    def test_pending_recovery_skips_probe_and_planner_even_on_terminal_null(self):
        a = moving_agent()
        a._pending_recovery = {'ticks': 1}
        a._recovery_action = Mock(return_value=fixtures.Action.null)
        a.decide_waypoint = Mock()
        control_tests.PLANNER['step'](a, 2)
        a._recovery_action.assert_called_once_with(2)
        a._terminal_probe_due.assert_not_called()
        a.decide_waypoint.assert_not_called()
        a.execute_action.assert_called_once_with(fixtures.Action.null)

    def test_recovery_selection_abort_cleans_transaction_and_propagates(self):
        a = moving_agent()
        a._pending_recovery = {'ticks': 1}
        a._recovery_action = Mock(side_effect=NavigationAborted('request_budget'))
        a._cancel_checkpoint_recovery = Mock()
        with self.assertRaisesRegex(NavigationAborted, 'request_budget'):
            control_tests.PLANNER['step'](a, 2)
        a._cancel_checkpoint_recovery.assert_called_once()
        a.execute_action.assert_not_called()

    def test_small_move_rotation_and_new_observation_each_require_new_verification(self):
        a = boundary_tests.agent()
        boundary_tests.boundary(a)
        a.mapper.current_position[0] += .05
        boundary_tests.boundary(a)
        a.mapper.current_rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        boundary_tests.boundary(a)
        a._observation_revision = 1
        boundary_tests.boundary(a)
        self.assertEqual(a._verify_current_subtask.call_count, 4)

    def test_exhausted_verification_budget_never_returns_stop(self):
        a = boundary_tests.agent(boundary_tests.verdict(outcome=False))
        a.navigation_cfg.update(max_verifier_checks_per_subtask=1, soft_stop_on_verifier_budget=True)
        boundary_tests.boundary(a)
        a.mapper.current_position[0] += .05
        with self.assertRaisesRegex(NavigationAborted, 'verifier_check_budget_exhausted'):
            boundary_tests.boundary(a)
        self.assertFalse(a.st['completed'])

    def test_unknown_scene_does_not_become_wrong_path_on_second_rejection(self):
        a = boundary_tests.agent(boundary_tests.verdict(
            outcome=False, scene_consistent=False, failure_type='not_reached',
            evidence_state='unknown', rollback_required=False))
        a.navigation_cfg['enable_wrong_path_rollback'] = True
        a.plan_rollback_path = Mock()
        boundary_tests.boundary(a)
        a.mapper.current_position[0] += .2
        boundary_tests.boundary(a)
        a.plan_rollback_path.assert_not_called()


if __name__ == '__main__':
    unittest.main()


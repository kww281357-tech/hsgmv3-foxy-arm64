"""Actual-ledger physical recovery; CPU geometry/controller stubs only."""
import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from recovery_supervisor import RecoverySupervisor, witnessed_checkpoint
from progress_watchdog import NavigationAborted
from navigation_control import WaypointExecution
from test_v3_recovery_lifecycle import harness, Action


def record(step, before, after, kind='move_forward', requested=.2, **extra):
    return dict(step=step, subtask_id='P1', pose_before=list(before), pose=list(after),
                executed_action=kind, action_type=kind, requested_distance=requested,
                motion_distance=float(np.linalg.norm(np.asarray(after)-before)), **extra)


class SupervisorEvidenceTests(unittest.TestCase):
    def test_three_measured_blocks_trigger_without_model_fields(self):
        supervisor = RecoverySupervisor()
        ledger = [record(0, [0, 0, 0], [.5, 0, 0])]
        self.assertIsNone(supervisor.consume(ledger, 'P1'))
        for step in range(1, 4):
            ledger.append(record(step, [.5, 0, 0], [.5, 0, 0]))
            result = supervisor.consume(ledger, 'P1')
        self.assertEqual(result['reason'], 'blocked_translation')
        self.assertEqual(result['first_affected_step'], 1)
        self.assertEqual(result['blocked_attempts'], 3)
        self.assertIsNone(supervisor.consume(ledger, 'P1'))  # Once per actual execution.

    def test_claimed_blocked_flag_cannot_override_actual_motion(self):
        ledger = [record(i, [i*.2, 0, 0], [(i+1)*.2, 0, 0], blocked=True) for i in range(24)]
        self.assertIsNone(RecoverySupervisor().consume(ledger, 'P1'))

    def test_slow_door_and_vertical_stair_motion_are_real_progress(self):
        for axis in (0, 2):
            ledger = []
            for i in range(32):
                before, after = np.zeros(3), np.zeros(3)
                before[axis], after[axis] = i*.03, (i+1)*.03
                ledger.append(record(i, before, after, requested=.03))
            self.assertIsNone(RecoverySupervisor().consume(ledger, 'P1'))

    def test_measured_back_and_forth_cannot_reset_stall_by_small_motion(self):
        ledger = [record(i, [0 if i % 2 == 0 else .25, 0, 0],
                         [.25 if i % 2 == 0 else 0, 0, 0]) for i in range(16)]
        result = RecoverySupervisor().consume(ledger, 'P1')
        self.assertEqual(result['reason'], 'measured_revisit_loop')
        self.assertEqual(result['window_path_m'], 4.)
        self.assertEqual(result['window_net_m'], 0.)

    def test_stationary_turns_trigger_and_fresh_waypoint_ids_do_not_reset(self):
        ledger = [record(i, [0, 0, 0], [0, 0, 0], 'turn', 0,
                         requested_action={'waypoint_id': str(i)}) for i in range(16)]
        self.assertEqual(RecoverySupervisor().consume(ledger, 'P1')['reason'], 'stationary_execution')

    def test_endpoint_observation_exemption_is_bounded_at_one_pose(self):
        supervisor, ledger = RecoverySupervisor(), []
        for i in range(20):
            ledger.append(record(i, [0, 0, 0], [0, 0, 0], 'turn', 0))
            result = supervisor.consume(ledger, 'P1', confirming=True)
            if i < 19:
                self.assertIsNone(result)
        self.assertEqual(result['reason'], 'stationary_execution')
        self.assertEqual(supervisor.held_records, 4)
        ledger.append(record(20, [0, 0, 0], [.4, 0, 0]))
        supervisor.consume(ledger, 'P1', confirming=True)
        self.assertEqual(supervisor.held_records, 0)

    def test_flag_toggle_does_not_replenish_observation_exemption(self):
        supervisor, ledger = RecoverySupervisor(), []
        for i in range(8):
            ledger.append(record(i, [0, 0, 0], [0, 0, 0], 'turn', 0))
            supervisor.consume(ledger, 'P1', confirming=(i % 2 == 0))
        self.assertEqual(supervisor.held_records, 4)
        ledger.append(record(8, [0, 0, 0], [0, 0, 0], 'turn', 0))
        supervisor.consume(ledger, 'P1', confirming=True)
        self.assertEqual(len(supervisor.rows), 2)

    def test_recovery_motion_cannot_make_a_physical_loop(self):
        ledger = [record(i, [i % 2, 0, 0], [(i+1) % 2, 0, 0], recovery=True) for i in range(20)]
        self.assertIsNone(RecoverySupervisor().consume(ledger, 'P1'))

    def test_missing_semantic_progress_does_not_mean_wrong_path(self):
        ledger = [record(i, [i*.4, 0, 0], [(i+1)*.4, 0, 0]) for i in range(100)]
        self.assertIsNone(RecoverySupervisor().consume(ledger, 'P1'))

    def test_nonfinite_localization_and_rewound_ledger_are_explicit_aborts(self):
        supervisor = RecoverySupervisor()
        self.assertTrue(supervisor.consume([record(0, [0, 0, 0], [math.nan, 0, 0])], 'P1')['abort'])
        self.assertEqual(supervisor.consume([], 'P1')['reason'], 'physical_ledger_rewound')

    def test_production_empty_null_text_is_a_measured_tick_not_protocol_failure(self):
        supervisor = RecoverySupervisor()
        ledger = []
        for i in range(20):
            row = record(i, [0, 0, 0], [0, 0, 0], 'null', 0)
            row['executed_action'] = ''  # Actual production _action_to_text(null).
            ledger.append(row)
            result = supervisor.consume(ledger, 'P1', confirming=True)
        self.assertEqual(result['reason'], 'stationary_execution')
        self.assertFalse(result.get('abort'))
        for kind, after in (('move_forward', [0, 0, 0]), ('null', [.1, 0, 0])):
            row = record(0, [0, 0, 0], after, kind, 0)
            row['executed_action'] = ''
            self.assertEqual(RecoverySupervisor().consume([row], 'P1')['reason'],
                             'physical_execution_record_invalid')


def save_anchor(a):
    cp = {'subtask_key': 'P1', 'ledger_len': len(a._execution_ledger),
          'step': len(a._execution_ledger), 'ground_pose': a._recovery_ground_pose().tolist(),
          'mapper_pose': a.mapper.current_position.tolist(),
          'rotation': a.mapper.current_rotation.tolist(), 'floor_level': 0, 'history_len': 1}
    a._branch_checkpoints.append(cp)
    return cp


def blocked_agent(two_anchors=False):
    a = harness()
    a._branch_checkpoints = []
    save_anchor(a)
    a.execute_action(Action(.5 if two_anchors else .6, 0, 'move_forward'))
    if two_anchors:
        save_anchor(a)
        a.execute_action(Action(.5, 0, 'move_forward'))
    a._waypoint_execution = WaypointExecution(a._recovery_ground_pose() + [.5, 0, 0], 'WP1')
    a.block_translation = True
    for _ in range(3):
        a.execute_action(Action(.2, 0, 'move_forward'))
    a._last_verifier_result = {'available': False}  # No semantic wrong-path request.
    return a


class PhysicalRecoveryProductionTests(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, {'simWrapper': NS(PolarAction=Action)})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_near_anchor_actual_return_preserves_all_valid_route_evidence(self):
        a = blocked_agent()
        a.st['completion_conditions'] = [{'id': 'C1', 'status': 'satisfied', 'execution_checked': True, 'steps': [0]}]
        a.st['completion_evidence_reliable'] = True
        history, conditions = copy.deepcopy(a.history_msgs), copy.deepcopy(a.st['completion_conditions'])
        records = list(a._execution_ledger)
        self.assertTrue(a._supervise_physical_progress(4))
        self.assertTrue(a._supervise_physical_progress(4))  # Pending does not start another attempt.
        self.assertEqual(a._physical_recovery_counts, {'P1': 1})
        self.assertEqual(a.navigation_trace[-1]['event'], 'physical_recovery_started')
        self.assertEqual(a.navigation_trace[-1]['physical_evidence']['blocked_attempts'], 3)
        self.assertEqual(a._pending_recovery['checkpoint']['ledger_len'], 0)
        a.block_translation = False
        for step in range(5, 20):
            a._step_recovery(step)
            if not a._pending_recovery:
                break
        self.assertIsNone(a._pending_recovery)
        self.assertEqual(a.navigation_trace[-1]['event'], 'physical_recovery_completed')
        self.assertLessEqual(a.navigation_trace[-1]['horizontal_error_m'], .3)
        self.assertEqual(a.history_msgs, history)
        self.assertEqual(a.st['completion_conditions'], conditions)
        self.assertTrue(a.st['completion_evidence_reliable'])
        self.assertFalse(any(r.get('rejected_by_rollback') for r in records))
        a._remember_failed_branch.assert_not_called()
        self.assertIsNone(a._last_rollback_info)  # Never issue a wrong-room completion prompt.
        self.assertEqual(a._last_recovery_context['recovery_source'], 'physical')
        self.assertTrue(any(r.get('source') == 'physical_recovery' for r in a._blocked_targets))
        self.assertIsNone(a._physical_recovery_retry)
        a.decide_waypoint.assert_not_called()

    def test_fabricated_and_late_anchors_are_not_selected(self):
        for change in ({'ground_pose': [.1, 0, -1.2]}, {'ledger_len': 3}):
            a = blocked_agent()
            a._branch_checkpoints[0].update(change)
            with self.assertRaises(NavigationAborted) as caught:
                a._supervise_physical_progress(4)
            self.assertEqual(caught.exception.reason, 'physical_recovery_no_safe_witnessed_anchor')
            self.assertFalse(a.rolling_back)
            a._remember_failed_branch.assert_not_called()

    def test_floor_label_change_does_not_block_same_height_certified_return(self):
        a = blocked_agent()
        a.mapper.floor_level = 5
        self.assertTrue(a._supervise_physical_progress(4))
        event = a.navigation_trace[-1]
        self.assertEqual(event['checkpoint_floor_label'], 0)
        self.assertEqual(event['actual_floor_label'], 5)
        a.block_translation = False
        for step in range(5, 20):
            a._step_recovery(step)
            if not a._pending_recovery:
                break
        self.assertEqual(a.navigation_trace[-1]['event'], 'physical_recovery_completed')

    def test_same_floor_label_does_not_allow_return_across_real_height_gap(self):
        a = harness()
        a._branch_checkpoints = []
        cp = save_anchor(a)
        flat_move = a.sim_wrapper.step

        def climb(action):
            obs = flat_move(action)
            if action.type == 'move_forward':
                obs['agent_state'].position[1] += .3
            return obs

        a.sim_wrapper.step = climb
        for _ in range(2):
            a.execute_action(Action(.3, 0, 'move_forward'))
        a.sim_wrapper.step = flat_move
        a.block_translation = True
        for _ in range(3):
            a.execute_action(Action(.2, 0, 'move_forward'))
        self.assertTrue(witnessed_checkpoint(cp, a._execution_ledger, 'P1', 0, 2))
        self.assertAlmostEqual(a._recovery_ground_pose()[2] - cp['ground_pose'][2], .6)
        a._validated_waypoint_route = Mock()
        with self.assertRaises(NavigationAborted) as caught:
            a._supervise_physical_progress(5)
        self.assertEqual(caught.exception.reason, 'physical_recovery_no_safe_witnessed_anchor')
        a._validated_waypoint_route.assert_not_called()

    def test_blocked_current_route_requires_current_navmesh_certificate(self):
        a = blocked_agent()
        a.sim_wrapper.sim.pathfinder.connected = False
        with self.assertRaises(NavigationAborted):
            a._supervise_physical_progress(4)
        self.assertFalse(a.rolling_back)
        self.assertFalse(any(r.get('rejected_by_rollback') for r in a._execution_ledger))

    def test_failed_return_retries_another_real_anchor_then_aborts_after_two(self):
        a = blocked_agent(two_anchors=True)
        self.assertTrue(a._supervise_physical_progress(5))
        self.assertEqual(a._pending_recovery['checkpoint']['ledger_len'], 1)
        for step in range(6, 15):
            a._step_recovery(step)
            if not a._pending_recovery:
                break
        self.assertEqual(a._last_recovery_context['status'], 'failed')
        self.assertTrue(a._supervise_physical_progress(15))
        self.assertEqual(a._pending_recovery['checkpoint']['ledger_len'], 0)
        for step in range(16, 25):
            a._step_recovery(step)
            if not a._pending_recovery:
                break
        with self.assertRaises(NavigationAborted) as caught:
            a._supervise_physical_progress(25)
        self.assertEqual(caught.exception.reason, 'physical_recovery_attempt_budget_exhausted')
        self.assertEqual(a._physical_recovery_counts['P1'], 2)
        self.assertFalse(a.rolling_back)
        self.assertEqual(a.current_path, [])
        a._remember_failed_branch.assert_not_called()

    def test_controller_claim_cannot_complete_unreached_physical_anchor(self):
        a = blocked_agent()
        self.assertTrue(a._supervise_physical_progress(4))
        self.assertFalse(a._finish_checkpoint_recovery(True, 'Caller claims arrived'))
        self.assertEqual(a.navigation_trace[-1]['event'], 'physical_recovery_failed')
        self.assertIsNone(a._last_rollback_info)

    def test_semantic_recovery_still_invalidates_only_rejected_witnesses(self):
        a = blocked_agent()
        cp = a._branch_checkpoints[0]
        target = a._recovery_ground_pose()
        self.assertTrue(a._begin_checkpoint_recovery([target, cp['ground_pose']], cp['rotation'], cp,
                                                    recovery_source='manual'))
        self.assertTrue(all(r.get('rejected_by_rollback') for r in a._execution_ledger))
        a._remember_failed_branch.assert_called_once()
        self.assertEqual(a.navigation_trace[-1]['event'], 'wrong_path_rollback_started')

    def test_nearby_checkpoint_must_be_witnessed_at_saved_ledger_boundary(self):
        a = blocked_agent()
        cp = a._branch_checkpoints[0]
        self.assertTrue(witnessed_checkpoint(cp, a._execution_ledger, 'P1', 0, 1))
        cp['ground_pose'][0] += .03
        self.assertFalse(witnessed_checkpoint(cp, a._execution_ledger, 'P1', 0, 1))

    def test_bounded_endpoint_escape_returns_before_session_and_resets_pending(self):
        a = harness()
        a._branch_checkpoints = []
        save_anchor(a)
        a.execute_action(Action(.6, 0, 'move_forward'))
        a._endpoint_controller = Mock()
        for name in ('_pending_endpoint_confirmation', '_pending_endpoint_relocalization',
                     '_pending_final_stop_confirmation'):
            setattr(a, name, {'key': 'P1', 'habitat_session': True})
        a._endpoint_escape_requested = {'reason': 'physical_escape_required:relation_unresolved',
            'subtask_id': 'P1', 'scope_key': 'P1.endpoint', 'first_affected_step': 1}
        self.assertTrue(a._supervise_physical_progress(2))
        self.assertEqual(a._pending_recovery['physical_evidence']['reason'], 'endpoint_observation_exhausted')
        self.assertIsNone(a._endpoint_escape_requested)
        self.assertIsNone(a._pending_endpoint_confirmation)
        self.assertIsNone(a._pending_endpoint_relocalization)
        self.assertIsNone(a._pending_final_stop_confirmation)
        a._endpoint_controller.reset.assert_called_once()
        a._remember_failed_branch.assert_not_called()

    def test_endpoint_escape_without_previous_safe_anchor_is_abort_not_stop(self):
        a = harness()
        a._branch_checkpoints = []
        a._endpoint_escape_requested = {'reason': 'inspection_budget', 'subtask_id': 'P1',
            'scope_key': 'P1.endpoint', 'first_affected_step': 0}
        with self.assertRaises(NavigationAborted) as caught:
            a._supervise_physical_progress(2)
        self.assertEqual(caught.exception.reason, 'physical_recovery_no_safe_witnessed_anchor')
        a.decide_waypoint.assert_not_called()

    def test_escape_from_another_subtask_is_not_a_recovery_request(self):
        a = harness()
        a._endpoint_escape_requested = {'reason': 'inspection_budget', 'subtask_id': 'P2',
            'scope_key': 'P2.endpoint', 'first_affected_step': 0}
        self.assertFalse(a._supervise_physical_progress(2))
        self.assertIsNone(a._endpoint_escape_requested)

    def test_invalid_endpoint_escape_boundary_cannot_select_future_anchor(self):
        a = harness()
        a._endpoint_escape_requested = {'reason': 'inspection_budget', 'subtask_id': 'P1',
            'scope_key': 'P1.endpoint', 'first_affected_step': 10}
        with self.assertRaises(NavigationAborted) as caught:
            a._supervise_physical_progress(2)
        self.assertEqual(caught.exception.reason, 'physical_endpoint_escape_invalid')


if __name__ == '__main__':
    unittest.main()


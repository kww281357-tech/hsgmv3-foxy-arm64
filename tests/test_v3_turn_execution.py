"""CPU turn-intent integration; no models, simulator, or evaluator targets."""
import math
import copy
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from agent.turn_execution import TurnExecutionGuard, first_segment_heading, heading_matches
from navigation_control import ActionProtocolError, decode_action
from route_progress import turn_witness
from navigation_contracts import completion_conditions, validate_condition_checks
import test_v3_verification_contract as verification_fixture
import test_v3_navigation_control as planner_fixture
import test_v3_waypoint_reachability as execution_fixture


TEXT = 'Turn right at the eye chart and stand next to the bathroom door.'


def guard(a=None, instruction=TEXT):
    if a is None:
        a = NS()
        a.mapper = NS(current_position=np.zeros(3))
    for name, method in TurnExecutionGuard.__dict__.items():
        if name.startswith('_') and callable(method):
            setattr(a, name, method.__get__(a))
    a.st = {'instruction': instruction, 'record': []}
    a.instruction_obj = getattr(a, 'instruction_obj', NS())
    a.instruction_obj.get_current_subtask = lambda: ('P1', a.st)
    a.instruction_obj.get_current_subtask_key = lambda: 'P1'
    a._execution_ledger = []
    a.navigation_trace = []
    a._invalidate_blocked_path = Mock()
    return a


def context(status='at_landmark', **kwargs):
    return {'condition_id': 'P1.C1', 'status': status, 'landmark_identity': 'eye chart',
            'landmark_view': 'right', 'landmark_bbox': [.3, .2, .5, .6], **kwargs}


def selected(direction='RIGHT'):
    return decode_action({'type': 'turn', 'direction': direction}, planner_fixture.ACTIONS)


def turn_row(step=0, yaw=math.pi/2, before=0.):
    return {'step': step, 'action_type': 'turn', 'yaw_delta': yaw,
            'requested_action': {'type': 'turn', 'direction': 'BACK' if abs(yaw) >= 2.6 else 'RIGHT' if yaw > 0 else 'LEFT'},
            'yaw_before': before, 'yaw_after': before+yaw, 'pose': [0, 0, -1.2],
            'pose_before': [0, 0, -1.2], 'motion_distance': 0., 'recovery': False}


def arm(a, row=None):
    row = row or turn_row()
    a._instruction_turn_command = a._active_turn_clause()
    a._execution_ledger.append(row)
    a._observe_turn_execution(row)
    return row


class LandmarkSelectionTests(unittest.TestCase):
    def test_near_identity_bound_landmark_requires_explicit_right_action(self):
        a = guard()
        a._measure_turn_landmark = Mock(return_value={'valid': True, 'median_distance_m': 1.})
        parsed = {'turn_context': context()}
        waypoint = decode_action({'type': 'navigate', 'waypoint_id': 'WP_01'}, planner_fixture.ACTIONS)
        with self.assertRaisesRegex(ActionProtocolError, 'explicit RIGHT'):
            a._validate_turn_selection(parsed, waypoint)
        actual = a._validate_turn_selection(parsed, selected())
        self.assertEqual(actual['instruction_turn']['direction'], 'RIGHT')
        self.assertNotIn('instruction_turn', selected())

    def test_wrong_identity_far_missing_or_mixed_depth_does_not_arm(self):
        a = guard()
        a._measure_turn_landmark = Mock(return_value={'valid': True, 'median_distance_m': .2})
        for identity in ('bathroom door', 'bathroom door beside the eye chart', 'not the eye chart'):
            with self.assertRaisesRegex(ActionProtocolError, 'identity'):
                a._validate_turn_selection({'turn_context': context(landmark_identity=identity)}, selected())
        a._measure_turn_landmark.assert_not_called()
        for packet in ({'valid': True, 'median_distance_m': 2.}, {'valid': False},
                       {'valid': False, 'median_distance_m': .5, 'reason': 'mixed_target_depth'},
                       {'valid': True, 'distance_m': .1}):
            a._measure_turn_landmark.return_value = packet
            with self.assertRaises(ActionProtocolError):
                a._validate_turn_selection({'turn_context': context()}, selected())

    def test_unknown_visible_or_observation_only_never_forces_or_locks_turn(self):
        a = guard()
        a._measure_turn_landmark = Mock()
        for c in (None, context('observe'), context('approaching')):
            for action in (selected(), decode_action({'type': 'navigate', 'waypoint_id': 'WP_01'}, planner_fixture.ACTIONS)):
                self.assertNotIn('instruction_turn', a._validate_turn_selection({'turn_context': c}, action))
        a._measure_turn_landmark.assert_not_called()
        a.st['instruction'] = 'Turn around and enter the house.'
        self.assertNotIn('instruction_turn', a._validate_turn_selection({'turn_context': None}, selected('BACK')))
        self.assertIn('instruction_turn', a._validate_turn_selection({'turn_context': context('execute_turn')}, selected('BACK')))

    def test_later_turn_not_active_before_previous_movement(self):
        a = guard(instruction='Walk down the hall. Turn right at the eye chart.')
        self.assertIsNone(a._active_turn_clause())

    def test_real_current_depth_is_used_for_its_own_view_and_box(self):
        a = guard()
        a.mapper.observation_id, a.mapper.observation_step = 4, 8
        a.mapper.camera_intrinsic = np.array([[60., 0, 50], [0, 60, 50], [0, 0, 1]])
        a.mapper.rotation_func = lambda r: r
        a.mapper.translation_func = lambda p: p
        a.mapper.initial_position = np.zeros(3)
        rotation = np.array([[0., 0, 1], [1, 0, 0], [0, 1, 0]])
        state = NS(position=np.zeros(3), sensor_states={'color_sensor': NS(rotation=rotation)})
        a.curr_obs = {'right': {'agent_state': state, 'depth_sensor': np.ones((100, 100))}}
        packet = a._measure_turn_landmark(context())
        self.assertTrue(packet['valid'], packet)
        self.assertEqual(packet['view'], 'right')
        self.assertEqual(packet['observation_id'], 4)
        self.assertLess(packet['median_distance_m'], 1.5)
        a.curr_obs['right']['depth_sensor'] *= 3
        self.assertGreater(a._measure_turn_landmark(context())['median_distance_m'], 2.)


class DepartureTests(unittest.TestCase):
    def test_left_right_signs_and_no_alignment_promotion(self):
        for direction, yaw in [('right', math.pi/2), ('left', -math.pi/2), ('around', -math.pi)]:
            a = guard(instruction='Turn ' + direction + '.')
            arm(a, turn_row(yaw=yaw))
            self.assertAlmostEqual(a._current_turn_departure()['heading'], yaw)
            self.assertTrue(a._turn_route_allowed([[math.cos(yaw), math.sin(yaw), 0]]))
            self.assertFalse(a._turn_route_allowed([[-math.cos(yaw), -math.sin(yaw), 0]]))
        a = guard()
        arm(a, turn_row(yaw=-math.pi/2))
        self.assertIsNone(a._current_turn_departure())
        a = guard()
        row = turn_row()
        a._execution_ledger.append(row)
        a._observe_turn_execution(row)  # pure alignment / terminal scan, no command tag
        self.assertIsNone(a._current_turn_departure())

    def test_path_first_segment_not_target_direction(self):
        a = guard()
        arm(a)
        self.assertFalse(a._turn_route_allowed([[0, 0, 0], [1, 0, 0], [0, 2, 0]]))
        self.assertTrue(a._turn_route_allowed([[0, 0, 0], [0, .2, 0], [2, .2, 0]]))

    def test_observation_turn_and_blocked_move_do_not_release_departure(self):
        a = guard()
        arm(a)
        a._observe_turn_execution(turn_row(step=1, yaw=-math.pi/2))
        a._observe_turn_execution({'step': 2, 'action_type': 'move_forward',
                                  'pose_before': [0, 0, 0], 'pose': [0, 0, 0]})
        self.assertIsNotNone(a._current_turn_departure())
        a._observe_turn_execution({'step': 3, 'action_type': 'move_forward',
                                  'pose_before': [0, 0, 0], 'pose': [0, .2, 0]})
        self.assertIsNone(a._current_turn_departure())
        self.assertTrue(a.navigation_trace[-1]['accepted'])

    def test_actual_wrong_departure_invalidates_and_rollback_clears(self):
        a = guard()
        arm(a)
        a._observe_turn_execution({'step': 1, 'action_type': 'move_forward',
                                  'pose_before': [0, 0, 0], 'pose': [.2, 0, 0]})
        a._invalidate_blocked_path.assert_called_once_with('actual_departure_opposes_instructed_turn')
        self.assertFalse(a.navigation_trace[-1]['accepted'])
        a = guard()
        row = arm(a)
        row['rejected_by_rollback'] = True
        self.assertIsNone(a._current_turn_departure())
        a = guard()
        arm(a)
        a._execution_ledger.append({'step': 1, 'recovery': True})
        self.assertIsNone(a._current_turn_departure())

    def test_production_route_filter_and_runtime_alignment(self):
        a = guard(execution_fixture.controller())
        arm(a)
        self.assertIsNone(a._validated_waypoint_route([1, 0, -1.2]))
        self.assertIsNotNone(a._validated_waypoint_route([0, 1, -1.2]))
        self.assertIsNotNone(a._validated_waypoint_route([1, 0, -1.2], recovery=True))
        a.current_path = [[1, 0, -1.2]]
        result = a._get_action_for_next_waypoint()
        self.assertEqual(result.type, 'null')
        a._invalidate_blocked_path.assert_called_with('first_segment_opposes_instructed_turn')
        a.current_path = [[0, 1, -1.2]]
        result = a._get_action_for_next_waypoint()
        self.assertEqual(result.type, 'turn')
        self.assertAlmostEqual(result.theta, -math.pi/2)  # command RIGHT; measured yaw positive

    def test_production_execute_action_arms_from_actual_signed_yaw(self):
        a = guard(execution_fixture.controller())
        a.current_path = []
        a._instruction_turn_command = a._active_turn_clause()
        a._requested_action = {'type': 'turn', 'direction': 'RIGHT'}
        state = a.sim_wrapper.sim.get_agent(0).get_state()
        a.sim_wrapper.step = lambda action: {'agent_state': state}
        a.update_map = lambda obs: setattr(a.mapper, 'current_rotation',
                                           np.array([[1., 0, 0], [0, 0, -1], [0, -1, 0]]))
        a.execute_action(execution_fixture.Action(0, -math.pi/2, 'turn'))
        row = a._execution_ledger[-1]
        self.assertAlmostEqual(row['yaw_delta'], math.pi/2)
        self.assertEqual(row['instruction_turn']['direction'], 'RIGHT')
        self.assertAlmostEqual(a._current_turn_departure()['heading'], math.pi/2)


class PlannerIntegrationTests(unittest.TestCase):
    def _ready_turn_then_departure_planner(self, repeat=False):
        turn = {'action': {'type': 'turn', 'direction': 'RIGHT'}, 'turn_context': context()}
        departure = {'action': {'type': 'navigate', 'waypoint_id': 'WP_01'}, 'turn_context': context()}
        a = guard(planner_fixture.planner([turn, turn, departure] if repeat else [turn, departure]))
        actions = dict(planner_fixture.ACTIONS, WP_01={'type': 'waypoint', 'target_point_world': np.array([0., 1., -1.2])})
        a._preprocessing_module = lambda obs: (np.zeros((100, 100, 3), dtype=np.uint8), actions)
        a._measure_turn_landmark = Mock(return_value={'valid': True, 'median_distance_m': .8})
        decision, _, _ = planner_fixture.PLANNER['decide_waypoint'](a)
        a._instruction_turn_command = decision['instruction_turn']
        row = turn_row()
        a._execution_ledger.append(row)
        a._observe_turn_execution(row)
        return a

    def test_ready_turn_then_same_landmark_forward_departure_is_accepted(self):
        a = self._ready_turn_then_departure_planner()
        self.assertIsNotNone(a._current_turn_departure())
        decision, _, _ = planner_fixture.PLANNER['decide_waypoint'](a)
        self.assertEqual(decision['type'], 'waypoint')
        self.assertNotIn('instruction_turn', decision)
        self.assertEqual(len(a.prompts), 2)
        self.assertEqual(sum(r.get('event') == 'instruction_turn_executed' for r in a.navigation_trace), 1)

    def test_pending_repeat_turn_retries_to_departure_without_double_turn(self):
        a = self._ready_turn_then_departure_planner(repeat=True)
        a._commit_planner_history = Mock()
        decision, _, _ = planner_fixture.PLANNER['decide_waypoint'](a)
        self.assertEqual(decision['type'], 'waypoint')
        self.assertEqual(len(a.prompts), 3)
        rejected = [r for r in a.navigation_trace if r.get('event') == 'action_protocol_rejected']
        self.assertEqual(len(rejected), 1)
        self.assertIn('already executed', rejected[0]['reason'])
        self.assertEqual(len(a._execution_ledger), 1)
        a._commit_planner_history.assert_called_once()

    def test_departed_turn_not_repeated_without_cached_route_progress(self):
        a = self._ready_turn_then_departure_planner()
        row = {'step': 1, 'action_type': 'move_forward', 'pose_before': [0, 0, -1.2],
               'pose': [0, .2, -1.2], 'motion_distance': .2, 'recovery': False}
        a._execution_ledger.append(row)
        a._observe_turn_execution(row)
        a.mapper.current_position = np.array([0, .2, 0.])
        self.assertIsNone(a._current_turn_departure())
        self.assertNotIn('route_progress', a.st)
        self.assertEqual(row['turn_departure']['turn_step'], 0)
        self.assertIsNotNone(a._executed_instruction_turn(a._active_turn_clause()))
        decision, _, _ = planner_fixture.PLANNER['decide_waypoint'](a)
        self.assertEqual(decision['type'], 'waypoint')
        with self.assertRaisesRegex(ActionProtocolError, 'already executed'):
            a._validate_turn_selection({'turn_context': context()}, selected())
        a._execution_ledger[0]['rejected_by_rollback'] = True
        self.assertIsNone(a._executed_instruction_turn(a._active_turn_clause()))
        self.assertIn('instruction_turn', a._validate_turn_selection({'turn_context': context()}, selected()))

    def test_pending_departure_keeps_landmark_identity_depth_and_heading_checks(self):
        a = self._ready_turn_then_departure_planner()
        waypoint = decode_action({'type': 'navigate', 'waypoint_id': 'WP_01'}, planner_fixture.ACTIONS)
        with self.assertRaisesRegex(ActionProtocolError, 'preserving its first heading'):
            a._validate_turn_selection({'turn_context': context()}, waypoint)
        with self.assertRaisesRegex(ActionProtocolError, 'identity'):
            a._validate_turn_selection({'turn_context': context(landmark_identity='bathroom door')}, waypoint)
        a._measure_turn_landmark.return_value = {'valid': False}
        with self.assertRaisesRegex(ActionProtocolError, 'not localized nearby'):
            a._validate_turn_selection({'turn_context': context()}, waypoint)

    def test_real_1263_forward_proposal_requires_localisation_context_before_history(self):
        # Exact old action and semantic excerpts: recognizing RIGHT did not tell
        # the controller whether the model intended an approach or a route turn.
        old = {'observation': "To my 'RIGHT', I clearly see the eye chart on the wall, which is the landmark specified in the instruction.",
               'thought': "Since I am at the start point and the chart is immediately to my right, I should first proceed down the hallway towards the landmarks.",
               'action': {'type': 'navigate', 'waypoint_id': 'WP_02'}}
        revised = dict(old, turn_context=None)  # Unknown still permits approach, never fabricates readiness.
        a = guard(planner_fixture.planner([old, revised]))
        a._preprocessing_module = lambda obs: (np.zeros((100, 100, 3), dtype=np.uint8),
                                              dict(planner_fixture.ACTIONS, WP_02=planner_fixture.ACTIONS['WP_01']))
        a._commit_planner_history = Mock()
        result, _, _ = planner_fixture.PLANNER['decide_waypoint'](a)
        self.assertEqual(result['type'], 'waypoint')
        self.assertEqual(len(a.prompts), 2)
        self.assertIn('requires turn_context', a.navigation_trace[0]['reason'])
        a._commit_planner_history.assert_called_once_with(revised)

    def test_context_rejection_precedes_history_commit_and_keeps_three_attempts(self):
        bad = {'action': {'type': 'navigate', 'waypoint_id': 'WP_01'}, 'turn_context': context()}
        good = {'action': {'type': 'turn', 'direction': 'RIGHT'}, 'turn_context': context()}
        a = guard(planner_fixture.planner([bad, bad, good]))
        a._measure_turn_landmark = Mock(return_value={'valid': True, 'median_distance_m': .8})
        a._commit_planner_history = Mock()
        result, _, _ = planner_fixture.PLANNER['decide_waypoint'](a)
        self.assertEqual(result['instruction_turn']['direction'], 'RIGHT')
        self.assertEqual(len(a.prompts), 3)
        a._commit_planner_history.assert_called_once_with(good)
        self.assertIn('ACTIVE INSTRUCTION TURN', a.prompts[0])
        self.assertEqual(sum(r.get('event') == 'action_protocol_rejected' for r in a.navigation_trace), 2)
        a = guard(planner_fixture.planner([bad, bad, bad]))
        a._measure_turn_landmark = Mock(return_value={'valid': True, 'median_distance_m': .8})
        self.assertIsNone(planner_fixture.PLANNER['decide_waypoint'](a)[0])
        self.assertEqual(len(a.prompts), 3)

    def test_recorded_1263_alignment_angles_stay_unmet(self):
        # Saved segment 1 / steps 1 and 7: separate navigation-alignment blocks.
        # Rounded recorded values preserved; no invented landmark box or depth.
        a = guard()
        rows = [turn_row(step=1, yaw=.066, before=-3.141),
                {'step': 2, 'action_type': 'move_forward', 'motion_distance': .2},
                turn_row(step=7, yaw=.185, before=-3.075)]
        for row in rows:
            row['requested_action'] = {'type': 'navigate', 'waypoint_id': 'WP_02'}
        a.st['record'] = rows
        clause = a._active_turn_clause()
        self.assertFalse(clause['measured_turn_witness'])
        self.assertEqual([r['degrees'] for r in clause['recent_turns']], [3.8, 10.6])
        for row in rows:
            a._execution_ledger.append(row)
            a._observe_turn_execution(row)
        self.assertIsNone(a._current_turn_departure())

    def test_recorded_1159_first_translation_would_undo_explicit_back_turn(self):
        # Saved step0 yaw_after2.618, then WP11 first segment toward [.875,-.125].
        # This replay explicitly supplies the new instruction-purpose context;
        # old raw responses did not contain it and are not relabelled as such.
        a = guard(instruction='Turn around and enter the house.')
        command = a._validate_turn_selection({'turn_context': context('execute_turn')}, selected('BACK'))
        a._instruction_turn_command = command['instruction_turn']
        row = turn_row(yaw=-math.pi, before=2.618+math.pi)
        a._execution_ledger.append(row)
        a._observe_turn_execution(row)
        self.assertFalse(a._turn_route_allowed([[0, 0, -1.2], [.875, -.125, -1.2]]))
        self.assertFalse(heading_matches(2.618, first_segment_heading([[.198, -.028, -1.2]], [0, 0, -1.2])))


class InstructionTurnAcceptanceTests(unittest.TestCase):
    """Real 1263 alignment cannot satisfy the stricter live Habitat contract."""

    def make_guard(self, strict=True, marked=False):
        a = verification_fixture.Guard(TEXT)
        a.navigation_cfg['habitat_control_enabled'] = strict
        a.mapper.current_position = np.array([-1.815, .562, 0.])
        a.mapper.observation_id = 17
        row = turn_row(step=6, yaw=.608, before=-3.075)
        row.update(subtask_id='P1', executed_action='Turn Right 34.8 degrees',
                   requested_action={'type': 'navigate', 'waypoint_id': 'WP_02'},
                   pose_before=[-1.298, -.054, -1.2], pose=[-1.298, -.054, -1.2])
        a.st['record'] = [row, verification_fixture.record(7, pose=(-1.381, -.072, -1.2), motion=.11),
                          verification_fixture.record(15, pose=(-1.815, .562, -1.2), motion=.2)]
        if marked:
            # Positive synthetic control, not a relabeling of the saved run.
            clause = completion_conditions(TEXT, 'P1')[0]
            row.update(yaw_delta=math.pi/2, yaw_after=row['yaw_before']+math.pi/2,
                       requested_action={'type': 'turn', 'direction': 'RIGHT'},
                       instruction_turn={'condition_id': clause['id'], 'subtask_id': 'P1',
                                         'text': clause['text'], 'direction': 'RIGHT'})
        ctx = a.prepare()
        raw = verification_fixture.verdict(ctx)
        raw.update(target_identity='Bathroom door', target_view='front', target_bbox=[.4, .25, .6, .85])
        raw['condition_checks'] = [
            {'id': 'P1.C1', 'status': 'satisfied',
             'evidence': 'Step 6: Turn Right 34.8 degrees (past eye chart)', 'steps': [6]},
            {'id': 'P1.C2', 'status': 'satisfied',
             'evidence': 'Agent is standing next to the visible bathroom door', 'steps': [], 'observation_id': 17}]
        a._ground_endpoint_verdict = Mock(return_value=None)
        return a, raw

    def test_real_alignment_is_unknown_and_does_not_reach_endpoint_gate(self):
        a, raw = self.make_guard()
        result = a.evaluate(raw)
        self.assertFalse(result['outcome'])
        self.assertEqual(result['condition_checks'][0]['status'], 'unknown')
        self.assertIn('explicit instruction turn', result['condition_checks'][0]['evidence'])
        self.assertFalse(result['route_contradiction_gate']['requested'])
        self.assertFalse(result.get('rollback_required'))
        self.assertEqual(a.st['route_progress']['completed_prefix'], [])
        a._ground_endpoint_verdict.assert_not_called()

    def test_legacy_non_habitat_replay_remains_compatible(self):
        a, raw = self.make_guard(strict=False)
        self.assertTrue(a.evaluate(raw)['outcome'])
        a._ground_endpoint_verdict.assert_called_once()

    def test_bound_explicit_turn_can_pass_without_an_extra_departure(self):
        a, raw = self.make_guard(marked=True)
        result = a.evaluate(raw)
        self.assertTrue(result['outcome'])
        self.assertTrue(result['condition_checks'][0]['execution_checked'])

    def test_marker_needs_matching_scope_text_direction_and_actual_request(self):
        mutations = [lambda r: r.pop('instruction_turn'),
                     lambda r: r.update(requested_action={'type': 'navigate', 'waypoint_id': 'WP_02'}),
                     lambda r: r.update(subtask_id='P2'),
                     lambda r: r['instruction_turn'].update(subtask_id='P2'),
                     lambda r: r['instruction_turn'].update(condition_id='P1.C2'),
                     lambda r: r['instruction_turn'].update(text='Turn right at the bathroom door'),
                     lambda r: r['instruction_turn'].update(direction='LEFT'),
                     lambda r: r.update(recovery=True),
                     lambda r: r.update(rejected_by_rollback=True)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                a, raw = self.make_guard(marked=True)
                mutate(a.st['record'][0])
                a.prepare()
                self.assertFalse(a.evaluate(raw)['outcome'])

    def test_marked_turn_does_not_hide_uncited_in_place_undo(self):
        a, raw = self.make_guard(marked=True)
        undo = turn_row(step=7, yaw=-math.pi/2)
        undo['executed_action'] = 'Turn Left 90.0 degrees'
        a.st['record'][1] = undo  # No actual translation separates this reversal.
        a.prepare()
        result = a.evaluate(raw)
        self.assertFalse(result['outcome'])
        self.assertEqual(result['condition_checks'][0]['status'], 'unknown')

    def test_cached_unmarked_prefix_is_revalidated_in_both_live_entry_points(self):
        legacy, raw = self.make_guard(strict=False)
        self.assertTrue(legacy.evaluate(raw)['outcome'])
        saved = copy.deepcopy(legacy.st['route_progress'])
        a, raw = self.make_guard()
        a.st['route_progress'] = saved
        self.assertEqual(a.prepare()['checks'][0]['status'], 'unknown')
        self.assertFalse(a.evaluate(raw)['outcome'])
        planner = guard()
        planner.navigation_cfg = {'habitat_control_enabled': True}
        planner.st['record'] = copy.deepcopy(a.st['record'])
        planner.st['route_progress'] = saved
        self.assertEqual(planner._active_turn_clause()['condition_id'], 'P1.C1')

    def test_cached_marked_prefix_keeps_live_witness_and_rejects_rollback_marker(self):
        a, raw = self.make_guard(marked=True)
        self.assertTrue(a.evaluate(raw)['outcome'])
        self.assertEqual(a.prepare()['checks'][0]['status'], 'satisfied')
        a.st['record'][0]['rejected_by_rollback'] = True
        self.assertEqual(a.prepare()['checks'][0]['status'], 'unknown')

    def test_default_pure_contract_legacy_option_is_explicit(self):
        a, raw = self.make_guard()
        expected = a._route_verification_context['checks']
        rows = [dict(c, observation_id=None, steps=[15]) if c['id']=='P1.C2' else c
                for c in raw['condition_checks']]
        old, _ = validate_condition_checks(expected, rows, a.st['record'])
        strict, _ = validate_condition_checks(expected, rows, a.st['record'],
                                              require_instruction_turn=True, instruction_subtask_id='P1')
        self.assertEqual(old[0]['status'], 'satisfied')
        self.assertEqual(strict[0]['status'], 'unknown')


class TurnContradictionOwnershipTests(unittest.TestCase):
    def make_guard(self, strict=True):
        a = verification_fixture.Guard(TEXT)
        a.navigation_cfg['habitat_control_enabled'] = strict
        clause = completion_conditions(TEXT, 'P1')[0]
        first = turn_row(step=1, yaw=1.571, before=-3.141)
        first.update(subtask_id='P1', executed_action='Turn Right 90.0 degrees',
                     instruction_turn={'condition_id': clause['id'], 'subtask_id': 'P1',
                                       'text': clause['text'], 'direction': 'RIGHT'})
        align = turn_row(step=2, yaw=.321, before=-1.57)
        align.update(executed_action='Turn Right 18.4 degrees',
                     requested_action={'type': 'navigate', 'waypoint_id': 'WP_05'})
        later = turn_row(step=6, yaw=-1.101, before=-1.249)
        later.update(executed_action='Turn Left 63.1 degrees',
                     requested_action={'type': 'navigate', 'waypoint_id': 'WP_02'})
        departed = verification_fixture.record(3, pose=(.063, -.190, -1.2), motion=.2)
        departed['turn_departure'] = {'accepted': True, 'condition_id': 'P1.C1', 'turn_step': 1}
        a.st['record'] = [first, align, departed,
                          verification_fixture.record(4, pose=(.125, -.375, -1.2), motion=.2),
                          verification_fixture.record(5, kind='null', motion=0.), later,
                          verification_fixture.record(7, pose=(-.015, -.517, -1.2), motion=.2),
                          verification_fixture.record(8, pose=(-.124, -.627, -1.2), motion=.15)]
        raw = verification_fixture.verdict(a.prepare())
        raw.update(outcome=False, relation_satisfied=False, scene_consistent=False,
                   failure_type='wrong_path',
                   reason='Agent turned left (step 6) instead of turning right at the eye chart.',
                   contradiction_evidence='Turn Left 63.1 degrees',
                   correction_action={'type': 'turn', 'direction': 'RIGHT'},
                   route_contradiction={'kind': 'turn_direction', 'condition_id': 'P1.C1',
                                        'steps': [6], 'expected': 'Turn right at the eye chart',
                                        'observed': 'Turn Left 63.1 degrees'})
        raw['condition_checks'] = [
            {'id': 'P1.C1', 'status': 'contradicted', 'steps': [6],
             'evidence': 'The agent executed a left turn (step 6) and approached a wall.'},
            {'id': 'P1.C2', 'status': 'unknown', 'steps': [], 'evidence': 'Not reached.'}]
        return a, raw

    def test_real_later_waypoint_left_is_not_a_contradiction_of_bound_right(self):
        a, raw = self.make_guard()
        result = a.evaluate(raw)
        self.assertTrue(result['route_contradiction_gate']['requested'])
        self.assertFalse(result['route_contradiction_gate']['accepted'])
        self.assertIn('not bound', result['route_contradiction_gate']['reason'])
        self.assertEqual(result['failure_type'], 'not_reached')
        self.assertFalse(result['rollback_required'])
        self.assertEqual(a.st['route_progress']['completed_prefix'], [])
        self.assertFalse(any(r.get('rejected_by_rollback') for r in a.st['record']))

    def test_legacy_replay_keeps_old_unscoped_direction_policy(self):
        a, raw = self.make_guard(strict=False)
        self.assertTrue(a.evaluate(raw)['route_contradiction_gate']['accepted'])

    def test_citing_the_earlier_marker_cannot_launder_a_later_opposite_block(self):
        a, raw = self.make_guard()
        raw['route_contradiction']['steps'] = raw['condition_checks'][0]['steps'] = [1, 6]
        self.assertFalse(a.evaluate(raw)['route_contradiction_gate']['accepted'])

    def test_unmarked_explicit_look_after_translation_has_no_clause_ownership(self):
        a, raw = self.make_guard()
        a.st['record'][5]['requested_action'] = {'type': 'turn', 'direction': 'LEFT'}
        a.prepare()
        self.assertFalse(a.evaluate(raw)['route_contradiction_gate']['accepted'])

    def test_same_bound_rotation_block_net_opposite_still_authorizes_contradiction(self):
        a, raw = self.make_guard()
        undo = turn_row(step=2, yaw=-3*math.pi/4)
        undo['executed_action'] = 'Turn Left 135 degrees'
        a.st['record'] = [a.st['record'][0], undo,
                          verification_fixture.record(3, motion=.2)]
        raw['route_contradiction'].update(steps=[2], observed='Turned left 135 degrees in place after the right turn.')
        raw['condition_checks'][0]['steps'] = [2]
        a.prepare()
        result = a.evaluate(raw)
        self.assertTrue(result['route_contradiction_gate']['accepted'])
        self.assertEqual(result['route_contradiction']['recovery_constraint']['earliest_affected_step'], 1)

    def test_undo_cannot_borrow_wrong_condition_or_rejected_intent_marker(self):
        for mutate in (lambda r: r['instruction_turn'].update(condition_id='P1.C2'),
                       lambda r: r.update(rejected_by_rollback=True),
                       lambda r: r.update(recovery=True)):
            with self.subTest(mutation=mutate):
                a, raw = self.make_guard()
                undo = turn_row(step=2, yaw=-3*math.pi/4)
                undo['executed_action'] = 'Turn Left 135 degrees'
                mutate(a.st['record'][0])
                a.st['record'] = [a.st['record'][0], undo]
                raw['route_contradiction']['steps'] = raw['condition_checks'][0]['steps'] = [2]
                a.prepare()
                self.assertFalse(a.evaluate(raw)['route_contradiction_gate']['accepted'])

    def test_mixed_citations_cannot_expand_owned_turn_recovery_boundary(self):
        for strict, expected_start in ((True, 3), (False, 1)):
            with self.subTest(strict=strict):
                a, raw = self.make_guard(strict=strict)
                earlier = turn_row(step=1, yaw=-math.pi/2)
                earlier['executed_action'] = 'Turn Left 90 degrees'
                marked = copy.deepcopy(a.st['record'][0])
                marked['step'] = 3
                undo = turn_row(step=4, yaw=-3*math.pi/4)
                undo['executed_action'] = 'Turn Left 135 degrees'
                a.st['record'] = [earlier, verification_fixture.record(2, motion=.2), marked, undo]
                raw['route_contradiction'].update(steps=[1, 4], observed='Turned left in both cited rotation blocks.')
                raw['condition_checks'][0]['steps'] = [1, 4]
                a.prepare()
                result = a.evaluate(raw)
                self.assertTrue(result['route_contradiction_gate']['accepted'])
                evidence = result['route_contradiction']
                self.assertEqual(evidence['steps'], [1, 4])  # Preserve the raw claim.
                constraint = evidence['recovery_constraint']
                self.assertEqual(constraint['earliest_affected_step'], expected_start)
                self.assertEqual(constraint['max_checkpoint_ledger_len'], expected_start)
                if strict:
                    self.assertEqual(evidence['instruction_turn_steps'], [3, 4])
                    self.assertEqual(constraint['cited_steps'], [3, 4])
                else:
                    self.assertNotIn('instruction_turn_steps', evidence)

    def test_owned_correct_block_does_not_expand_opposite_block_cutoff(self):
        a, raw = self.make_guard()
        first = a.st['record'][0]
        marked = copy.deepcopy(first)
        marked['step'] = 3
        undo = turn_row(step=4, yaw=-3*math.pi/4)
        undo['executed_action'] = 'Turn Left 135 degrees'
        a.st['record'] = [first, verification_fixture.record(2, motion=.2), marked, undo]
        raw['route_contradiction'].update(steps=[1, 4], observed='The later instruction turn was undone by a left rotation.')
        raw['condition_checks'][0]['steps'] = [1, 4]
        a.prepare()
        result = a.evaluate(raw)
        self.assertTrue(result['route_contradiction_gate']['accepted'])
        self.assertEqual(result['route_contradiction']['instruction_turn_steps'], [1, 3, 4])
        self.assertEqual(result['route_contradiction']['recovery_constraint']['earliest_affected_step'], 3)

    def test_positive_wrong_room_motion_is_not_suppressed_by_turn_ownership(self):
        a = verification_fixture.Guard('Walk into the kitchen. Stand beside the table.')
        a.navigation_cfg['habitat_control_enabled'] = True
        a.st['record'] = [verification_fixture.record(1, motion=.5)]
        raw = verification_fixture.verdict(a.prepare())
        raw.update(outcome=False, relation_satisfied=False, scene_consistent=False,
                   failure_type='wrong_path', contradiction_evidence='Entered the bedroom.',
                   route_contradiction={'kind': 'wrong_room', 'condition_id': 'P1.C1', 'steps': [1],
                                        'expected': 'Walk into the kitchen',
                                        'observed': 'Entered the bedroom containing a bed.'})
        raw['condition_checks'][0].update(status='contradicted', steps=[1])
        raw['condition_checks'][1].update(status='unknown', steps=[])
        self.assertTrue(a.evaluate(raw)['route_contradiction_gate']['accepted'])


if __name__ == '__main__':
    unittest.main()


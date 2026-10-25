"""CPU tests of typed actions, measured stair phases and production integration."""
import copy
import json
import logging
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import numpy as np

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
from navigation_control import (ActionProtocolError, decode_action, action_prompt, legal_action_objects,
    WaypointExecution, StairProgress, stair_intent, pending_stair_condition, landing_support,
    check_stair_witnesses, doorway_crossing_required, doorway_witnesses)
from navigation_contracts import completion_conditions
from navigation_policy import normalize_verification, partition_instruction
from progress_watchdog import NavigationAborted
from agent.route_guard import RouteEvidenceGuard
import test_v3_waypoint_reachability as fixtures
from test_v3_boundary_height import parser_ns

extract = fixtures.production_functions
ACTIONS = {'WP_01': {'type': 'waypoint', 'target_point_world': np.array([1., 0, -1.2])},
           'TURN_LEFT': {'type': 'turn', 'angle': np.pi/2},
           'TURN_RIGHT': {'type': 'turn', 'angle': -np.pi/2},
           'TURN_BACK': {'type': 'turn', 'angle': np.pi}}


class ActionTests(unittest.TestCase):
    def test_bare_symbols_and_unsupported_types_rejected(self):
        for value in ('B', 'L', 'R', 1, '-1', 'STOP', {'type': 'move_forward'},
                      {'type': 'turn', 'direction': 'B'}, {'type': 'navigate', 'waypoint_id': 1}):
            with self.subTest(value=value), self.assertRaises(ActionProtocolError):
                decode_action(value, ACTIONS)

    def test_waypoint_and_turn_have_distinct_explicit_meanings(self):
        waypoint = decode_action({'type': 'navigate', 'waypoint_id': 'WP_01'}, ACTIONS)
        turn = decode_action({'type': 'turn', 'direction': 'BACK'}, ACTIONS)
        self.assertEqual(waypoint['type'], 'waypoint')
        self.assertEqual(turn['type'], 'turn')
        self.assertEqual(turn['angle'], np.pi)
        self.assertNotIn('requested_action', ACTIONS['WP_01'])

    def test_coordinates_angles_extra_keys_and_stale_waypoints_rejected(self):
        for value in ({'type': 'navigate', 'waypoint_id': 'WP_99'},
                      {'type': 'navigate', 'waypoint_id': 'WP_01', 'target': [4, 5, 6]},
                      {'type': 'turn', 'direction': 'LEFT', 'angle': 0},
                      {'type': 'propose_completion', 'completed': True}):
            with self.subTest(value=value), self.assertRaises(ActionProtocolError):
                decode_action(value, ACTIONS)

    def test_completion_only_proposes_and_correction_cannot_complete(self):
        self.assertEqual(decode_action({'type': 'propose_completion'}, ACTIONS)['type'], 'completion_proposal')
        with self.assertRaises(ActionProtocolError):
            decode_action({'type': 'propose_completion'}, ACTIONS, completion=False)

    def test_unavailable_and_rejected_actions_not_exposed(self):
        actions = dict(ACTIONS, WP_02={'type': 'waypoint', 'rejected': True})
        self.assertNotIn({'type': 'navigate', 'waypoint_id': 'WP_02'}, legal_action_objects(actions))
        self.assertNotIn({'type': 'propose_completion'}, legal_action_objects(actions, completion=False))
        self.assertEqual(decode_action({'type': 'turn', 'direction': 'RIGHT'}, actions)['action_id'], 'TURN_RIGHT')


def stair_record(before, after, kind='move_forward', step=0):
    return {'step': step, 'executed_action': kind, 'action_type': kind,
            'pose_before': [0, 0, before], 'pose': [0, 0, after], 'height_delta': after-before,
            'motion_distance': abs(after-before), 'requested_distance': .2}


class StairTests(unittest.TestCase):
    def test_only_explicit_source_stair_motion_activates(self):
        for text in ('Go down the hall to the stairs.', 'Wait in the upstairs room.',
                     'Stand at the top of the steps.', 'Walk to the staircase.'):
            self.assertEqual(stair_intent(text)['direction'], 0, text)
        for text in ('Go down the stairs.', 'Walk down one set of steps.', 'Go downstairs.'):
            self.assertEqual(stair_intent(text)['direction'], -1)
        self.assertEqual(stair_intent('Climb the stairs.')['direction'], 1)

    def test_current_pending_clause_controls_direction(self):
        checks = completion_conditions('Go up the stairs. Then go down the steps.', 'P1')
        first, intent = pending_stair_condition(checks)
        self.assertEqual(intent['direction'], 1)
        first.update(status='satisfied', execution_checked=True)
        self.assertEqual(pending_stair_condition(checks)[1]['direction'], -1)

    def test_upward_route_rejected_for_descent_but_flat_approach_and_switchback_allowed(self):
        state = StairProgress('P1', 'P1.C1', stair_intent('Go down the stairs'), 0.)
        self.assertFalse(state.route_allowed([[0, 0, 0], [1, 0, .4]]))
        self.assertTrue(state.route_allowed([[0, 0, 0], [1, 0, .02]]))
        self.assertTrue(state.route_allowed([[0, 0, 0], [.5, 0, -.3], [.5, .5, -.3], [0, .5, -.7]]))

    def test_two_wrong_direction_moves_require_reobservation(self):
        state = StairProgress('P1', 'P1.C1', stair_intent('Go down the stairs'), 0.)
        self.assertIsNone(state.observe(stair_record(0, .2), False))
        self.assertEqual(state.observe(stair_record(.2, .4), False), 'stair_motion_opposes_current_clause')
        self.assertEqual(state.phase, 'reobserve')
        self.assertFalse(state.completion_ready())

    def test_flat_ground_or_height_alone_does_not_prove_stair_completion(self):
        state = StairProgress('P1', 'P1.C1', stair_intent('Go up the stairs'), 0.)
        for _ in range(4):
            state.observe(stair_record(0, 0, kind='turn'), True)
        self.assertFalse(state.completion_ready())
        state.observe(stair_record(0, .5), False)
        for _ in range(4):
            state.observe(stair_record(.5, .5, kind='turn'), False)
        self.assertFalse(state.completion_ready())

    def test_three_stable_supported_observations_enable_verification_only(self):
        state = StairProgress('P1', 'P1.C1', stair_intent('Go up the stairs'), 0.)
        state.observe(stair_record(0, .5), False)
        for _ in range(2):
            state.observe(stair_record(.5, .5, kind='turn'), True)
            self.assertFalse(state.completion_ready())
        state.observe(stair_record(.5, .5, kind='turn'), True)
        self.assertEqual(state.phase, 'landing')
        self.assertTrue(state.completion_ready())
        self.assertNotIn('completed', state.packet())

    def test_counted_steps_do_not_require_a_full_platform(self):
        intent = stair_intent('Go up two steps and stop.')
        self.assertFalse(intent['landing_required'])
        state = StairProgress('P1', 'P1.C1', intent, 0.)
        state.observe(stair_record(0, .3), False)
        self.assertTrue(state.completion_ready())
        self.assertTrue(stair_intent('Go down one set of steps')['landing_required'])

    def test_actual_height_witness_required_even_if_model_says_satisfied(self):
        checks = completion_conditions('Go down the stairs.', 'P1')
        checks[0].update(status='satisfied', execution_checked=True, steps=[0])
        self.assertEqual(check_stair_witnesses(checks, [stair_record(0, .4)])[0]['status'], 'unknown')
        self.assertEqual(check_stair_witnesses(checks, [stair_record(0, -.4)])[0]['status'], 'satisfied')
        rejected = dict(stair_record(0, -.4), rejected_by_rollback=True)
        self.assertEqual(check_stair_witnesses(checks, [rejected])[0]['status'], 'unknown')

    def test_landing_needs_occupied_area_not_a_tread_or_sparse_hull(self):
        xy = np.array([[x, y, 0.] for x in np.arange(-.4, .41, .05) for y in np.arange(-.4, .41, .05)])
        self.assertTrue(landing_support(xy, [0, 0, 0]))
        self.assertFalse(landing_support(xy, [0, 0, .5]))
        tread = xy[np.abs(xy[:, 0]) < .08]
        self.assertFalse(landing_support(tread, [0, 0, 0]))
        self.assertFalse(landing_support([[-.4, -.4, 0], [.4, .4, 0]], [0, 0, 0]))
        ramp = xy.copy()
        ramp[:, 2] = .15*ramp[:, 0]
        self.assertFalse(landing_support(ramp, [0, 0, 0]))


class DoorTests(unittest.TestCase):
    def test_seeing_or_approaching_door_does_not_request_crossing(self):
        self.assertFalse(doorway_crossing_required('Walk towards the door.'))
        self.assertTrue(doorway_crossing_required('Go through the open doorway on the left.'))

    def test_crossing_needs_measured_before_after_poses_and_visual_sides(self):
        checks = completion_conditions('Go through the door.', 'P1')
        report = {'condition_id': 'P1.C1', 'steps': [1], 'entry_side': 'bedroom',
                  'exit_side': 'hallway', 'visual_evidence': 'Door jamb is now behind the agent.'}
        row = dict(stair_record(0, 0, step=1), pose_before=[-.2, 0, 0], pose=[.2, 0, 0])
        self.assertTrue(doorway_witnesses(checks, [report], [row])[0]['execution_witnessed'])
        self.assertFalse(doorway_witnesses(checks, [report], [dict(row, pose=[-.2, 0, 0])])[0]['execution_witnessed'])
        self.assertFalse(doorway_witnesses(checks, [dict(report, exit_side='')], [row])[0]['execution_witnessed'])
        self.assertFalse(doorway_witnesses(checks, [dict(report, steps=[1, 1])], [row])[0]['execution_witnessed'])


CV = NS(FONT_HERSHEY_SIMPLEX=0, COLOR_BGR2RGB=0, LINE_AA=0, getTextSize=lambda *args: ((12, 8), 0),
        rectangle=lambda *args: None, putText=lambda *args: None, addWeighted=lambda *args: None,
        cvtColor=lambda image, *args: image)
IMAGE = NS(fromarray=lambda *args: object())
NS_GLOBALS = dict(np=np, math=math, json=json, logging=logging, cv2=CV, Image=IMAGE,
                 PolarAction=fixtures.Action, WaypointExecution=WaypointExecution,
                 NavigationAborted=NavigationAborted, ActionProtocolError=ActionProtocolError,
                 action_prompt=action_prompt, decode_action=decode_action,
                 robust_json_parse=parser_ns['robust_json_parse'],
                 create_top_down_map_centered=lambda *args, **kwargs: np.zeros((100, 100, 3), dtype=np.uint8))
PLANNER = extract(SRC/'agent/agent.py', {'decide_waypoint', 'generate_prompt', 'step', 'decompose_instruction'},
                  dict(NS_GLOBALS, re=re, partition_instruction=partition_instruction))


def planner(responses):
    a = fixtures.controller()
    a.config = {'camera': {'fov': 90}}
    a.curr_obs = {'color_sensor': np.zeros((100, 100, 3), dtype=np.uint8)}
    a.instruction = 'Walk along the corridor.'
    a.st = {'instruction': a.instruction, 'record': []}
    a.instruction_obj.get_current_subtask = lambda: ('P1', a.st)
    a.instruction_obj.get_all_subtasks_str = lambda: a.instruction
    a.instruction_obj.add_record_to_current_subtask = Mock()
    a.instruction_obj.update_plan_for_current_subtask = Mock()
    a._preprocessing_module = lambda obs: (np.zeros((100, 100, 3), dtype=np.uint8), ACTIONS)
    a._create_panoramic_360_image = lambda *args: None
    a.generate_prompt = PLANNER['generate_prompt'].__get__(a)
    a._format_in_episode_working_memory = lambda: ''
    a._memory_prompt_context = lambda: ''
    a.navigation_memory = None
    a.img_buffer = []
    pending = iter(responses)
    a.prompts = []
    def respond(*args, **kwargs):
        a.prompts.append(args[5])
        return json.dumps(next(pending))
    a._get_vlm_response_multiview = respond
    a._boundary_stop_decision = Mock(return_value={'type': 'stop'})
    return a


class PlannerIntegrationTests(unittest.TestCase):
    def test_bare_B_repaired_to_typed_navigation_never_executed_as_turn(self):
        a = planner([{'action': 'B', 'thought': 'Move forward'},
                     {'action': {'type': 'navigate', 'waypoint_id': 'WP_01'}}])
        result, _, _ = PLANNER['decide_waypoint'](a)
        self.assertEqual(result['type'], 'waypoint')
        self.assertEqual(len(a.prompts), 2)
        self.assertEqual(a.navigation_trace[0]['event'], 'action_protocol_rejected')

    def test_repeated_legacy_B_returns_no_decision_not_180_degree_rotation(self):
        a = planner([{'action': 'B'}]*3)
        result, _, _ = PLANNER['decide_waypoint'](a)
        self.assertIsNone(result)
        a._boundary_stop_decision.assert_not_called()

    def test_completed_flag_and_model_movement_do_not_override_action_or_ledger(self):
        a = planner([{'action': {'type': 'navigate', 'waypoint_id': 'WP_01'},
                      'completed': True, 'movement': 'I walked through the doorway.'}])
        result, _, _ = PLANNER['decide_waypoint'](a)
        self.assertEqual(result['type'], 'waypoint')
        a._boundary_stop_decision.assert_not_called()
        a.instruction_obj.add_record_to_current_subtask.assert_not_called()
        self.assertIn('MEASURED EXECUTION FEEDBACK', a.prompts[0])

    def test_only_explicit_completion_proposal_calls_verifier(self):
        a = planner([{'action': {'type': 'propose_completion'}}])
        self.assertEqual(PLANNER['decide_waypoint'](a)[0]['type'], 'stop')
        a._boundary_stop_decision.assert_called_once()

    def test_reverse_turn_remains_legal_after_left_turn(self):
        a = planner([{'action': {'type': 'turn', 'direction': 'RIGHT'}}])
        a._last_turn_dir = 'L'
        result, _, _ = PLANNER['decide_waypoint'](a)
        self.assertEqual(result['action_id'], 'TURN_RIGHT')
        self.assertLess(result['angle'], 0)

    def test_source_partition_cannot_lose_downstairs_clause(self):
        source = 'Walk down the stairs. Turn left and wait at the door.'
        a = NS(_get_llm_response=lambda prompt: json.dumps({'subtasks': ['Go upstairs.', 'Wait.']}))
        result = PLANNER['decompose_instruction'](a, source)
        words = lambda text: re.findall(r'\w+', text.lower())
        self.assertEqual(words(' '.join(result)), words(source))


class ExecutionIntegrationTests(unittest.TestCase):
    def test_twelve_short_valid_movements_keep_route_without_replanning(self):
        a = fixtures.controller()
        state = a.sim_wrapper.sim.get_agent(0).get_state()
        a.current_path = [[2., 0, -1.2]]
        a.target_world_position = [2., 0, -1.2]
        a._waypoint_execution = WaypointExecution(a.target_world_position, 'WP_01')
        a.navigation_cfg = {'precision_step_m': .1}
        a.prev_agent_position, a.last_action = None, fixtures.Action.null
        a.last_actions, a.vlm_responses, a.img_buffer = {}, [], []
        a.last_vlm_response = 'Executing WP_01'
        a._get_agent_yaw = lambda: fixtures.checkpoint_yaw(a.mapper.current_rotation)
        a.config = {'camera': {'fov': 90}}
        a.instruction_obj.current_step_count = 0
        a.decide_waypoint = Mock(side_effect=AssertionError('Do not replan mid-route'))
        def move(action):
            state.position = np.asarray(state.position) + [action.r, 0, 0]
            return {'agent_state': state, 'color_sensor': np.zeros((100, 100, 3), dtype=np.uint8)}
        a.sim_wrapper.step = move
        def update(obs):
            p = obs['agent_state'].position
            a.mapper.current_position = np.array([p[0], p[2], p[1]])
            return []
        a.update_map = update
        a.curr_obs = {'agent_state': state, 'color_sensor': np.zeros((100, 100, 3), dtype=np.uint8)}
        for step in range(12):
            PLANNER['step'](a, step)
        self.assertAlmostEqual(a.traveled_distance, 1.2)
        self.assertEqual(a._waypoint_execution.phase, 'moving')
        self.assertEqual(len(a._execution_ledger), 12)
        self.assertTrue(all(r['requested_action']['waypoint_id'] == 'WP_01' for r in a._execution_ledger))
        a.decide_waypoint.assert_not_called()

    def test_repeated_observation_bounded_without_fake_completion(self):
        a = NS(navigation_cfg={}, _physical_progress_step=0, _observation_turns_without_motion=8)
        with self.assertRaisesRegex(NavigationAborted, 'repeated_observation_without_motion'):
            PLANNER['step'](a, 8)

    def test_actual_wrong_height_cancels_route(self):
        a = fixtures.controller()
        a.st = {'instruction': 'Go down the stairs.', 'record': []}
        a.instruction_obj.get_current_subtask = lambda: ('P1', a.st)
        state = a.sim_wrapper.sim.get_agent(0).get_state()
        def wrong_move(action):
            state.position = np.asarray(state.position) + [0, .2, 0]
            return {'agent_state': state}
        a.sim_wrapper.step = wrong_move
        a.update_map = lambda obs: setattr(a.mapper, 'current_position', np.array([0., 0, obs['agent_state'].position[1]]))
        a._landing_support_observed = lambda: False
        a.execute_action(fixtures.Action(.2, 0, 'move_forward'))
        a.execute_action(fixtures.Action(.2, 0, 'move_forward'))
        self.assertEqual(a.current_path, [])
        self.assertEqual(a._stair_progress.phase, 'reobserve')
        self.assertEqual(a.navigation_trace[-1]['reason'], 'stair_motion_opposes_current_clause')

    def test_route_filter_uses_current_subtask_and_recovery_can_reverse(self):
        a = fixtures.controller(fixtures.Mesh(height=lambda p: p[0]*.4))
        a.st = {'instruction': 'Go down the stairs.', 'record': []}
        a.instruction_obj.get_current_subtask = lambda: ('P1', a.st)
        self.assertIsNone(a._validated_waypoint_route([1, 0, -.8]))
        self.assertIsNotNone(a._validated_waypoint_route([1, 0, -.8], recovery=True))
        a.st = {'instruction': 'Walk along the corridor.', 'record': []}
        a.instruction_obj.get_current_subtask = lambda: ('P2', a.st)
        a.instruction = 'Go upstairs earlier, then walk along the corridor.'
        self.assertIsNotNone(a._validated_waypoint_route([1, 0, -.8]))

    def test_arrival_on_last_execution_tick_is_not_budget_failure(self):
        active = WaypointExecution([1, 0, 0], 'WP_01', max_ticks=1)
        self.assertIsNone(active.observe({'action_type': 'move_forward', 'motion_distance': .2,
                                         'requested_distance': .2, 'pose': [1, 0, 0]}))
        self.assertEqual(active.phase, 'arrived')

    def test_verified_nearby_handoff_does_not_require_reclimbing(self):
        a = fixtures.controller()
        a.mapper.current_position = np.array([.5, 0, 1.7])
        previous_record = dict(stair_record(0, .5, step=1), pose_before=[0, 0, 0], pose=[.5, 0, .5],
                               motion_distance=.707)
        previous = {'completed': True, 'completion_evidence_reliable': True, 'record': [previous_record]}
        current = {'instruction': 'Go up the stairs.', 'record': [], 'position_anchor_check': {'status': 'aligned'}}
        a.instruction_obj.get_current_subtask = lambda: ('P2', current)
        a.instruction_obj.sub_instruction_keys = ['P1', 'P2']
        a.instruction_obj.current_subtask_index = 1
        a.instruction_obj.get_subtask_by_key = lambda key: previous if key == 'P1' else current
        state = a._ensure_stair_progress()
        self.assertEqual(state.handoff_steps, [1])
        self.assertEqual(state.start_height, 0)
        for _ in range(3):
            state.observe(stair_record(.5, .5, kind='turn'), True)
        self.assertTrue(state.completion_ready())


class GateHarness(fixtures.Controller):
    pass


def gate_harness(instruction):
    a = GateHarness()
    a.st = {'instruction': instruction, 'record': [], 'completed': False}
    a.mapper = NS(current_position=np.array([0., 0, 1.2]), initial_floor_height=-1.2)
    a.instruction_obj = NS(get_current_subtask=lambda: ('P1', a.st),
        get_current_subtask_key=lambda: 'P1',
        get_subtask_by_key=lambda key: a.st, is_last_subtask=lambda: True,
        sub_instruction_keys=['P1'], current_subtask_index=0)
    # Isolate stair/crossing execution contracts; localized target depth has
    # its own production-path integration suite.
    a.navigation_cfg, a.navigation_trace = {'require_endpoint_grounding': False}, []
    a.instruction = instruction
    a._observed_target_candidates = lambda *args: []
    return a


def positive_verdict(context):
    return {'outcome': True, 'relation_satisfied': True, 'scene_consistent': True,
            'confidence': 'high', 'evidence': 'Current destination and stopping side are visible.',
            'target_identity': 'satisfied', 'target_proximity': 'satisfied',
            'target_entity_id': None, 'target_evidence': 'The instructed endpoint is visible at the current pose.',
            'condition_checks': [dict(id=c['id'], status='satisfied', evidence='Current RGB and executed step.',
                                      steps=[1 if c['kind'] == 'movement' else 2]) for c in context['checks']]}


class VerificationIntegrationTests(unittest.TestCase):
    def test_production_gate_rejects_positive_stair_claim_until_measured_landing(self):
        a = gate_harness('Go down the stairs. Stop at the bottom.')
        state = a._ensure_stair_progress()
        move = dict(stair_record(0, -.5, step=1), pose_before=[0, 0, 0], pose=[.5, 0, -.5])
        wait = dict(stair_record(-.5, -.5, kind='null', step=2), pose_before=[.5, 0, -.5], pose=[.5, 0, -.5])
        a.st['record'] = [move, wait]
        a.mapper.current_position = np.array([.5, 0, .7])
        state.observe(move, False)
        context = a._prepare_route_verification({})
        raw = positive_verdict(context)
        result = a._evaluate_route_verification(raw, normalize_verification(copy.deepcopy(raw)))
        self.assertFalse(result['outcome'])
        self.assertFalse(a.st['completed'])
        self.assertIn('supported landing', result['reason'])
        for step in range(3, 6):
            a.st['record'].append(dict(wait, step=step, landing_supported=True))
        context = a._prepare_route_verification({})
        raw = positive_verdict(context)
        result = a._evaluate_route_verification(raw, normalize_verification(copy.deepcopy(raw)))
        self.assertTrue(result['completion_evidence_reliable'])
        self.assertFalse(a.st['completed'])  # Only the controller may commit.

    def test_production_gate_requires_door_crossing_references_and_side_evidence(self):
        a = gate_harness('Go through the door. Stop outside.')
        move = dict(stair_record(0, 0, step=1), pose_before=[-.2, 0, 0], pose=[.2, 0, 0], motion_distance=.4)
        wait = dict(stair_record(0, 0, kind='null', step=2), pose_before=[.2, 0, 0], pose=[.2, 0, 0])
        a.mapper.current_position = np.array([.2, 0, 1.2])
        a.st['record'] = [move, wait]
        context = a._prepare_route_verification({})
        raw = positive_verdict(context)
        result = a._evaluate_route_verification(raw, normalize_verification(copy.deepcopy(raw)))
        self.assertFalse(result['outcome'])
        self.assertIn('Doorway crossing', result['reason'])
        raw['doorway_checks'] = [{'condition_id': 'P1.C1', 'steps': [1], 'entry_side': 'bedroom',
                                 'exit_side': 'hallway', 'visual_evidence': 'Jamb behind agent; hallway outside.'}]
        result = a._evaluate_route_verification(raw, normalize_verification(copy.deepcopy(raw)))
        self.assertTrue(result['completion_evidence_reliable'])

    def test_raw_wrong_path_label_cannot_bypass_validated_contradiction_boundary(self):
        raw = {
            'outcome': False,
            'relation_satisfied': False,
            'scene_consistent': False,
            'failure_type': 'wrong_path',
            'confidence': 'high',
            'reason': 'Entered a bedroom dead-end instead of the kitchen.',
            'evidence': 'Bed and nightstand visible; no kitchen counters.',
            'correction_action': None,
        }
        norm = normalize_verification(copy.deepcopy(raw))
        self.assertEqual(norm['failure_type'], 'wrong_path')
        self.assertTrue(norm['rollback_required'])
        self.assertFalse(norm['contract_conflict'])

        rb_methods = extract(
            SRC / 'agent/agent.py',
            {'_direct_verifier_correction', '_record_branch_checkpoint', '_update_last_checkpoint_choice', 'plan_rollback_path'},
            dict(NS_GLOBALS, copy=copy),
        )
        RollbackController = type('RollbackController', (fixtures.Controller,), rb_methods)
        a = RollbackController()
        a.navigation_cfg = {'enable_wrong_path_rollback': True, 'verifier_return_control_to_planner': True}
        a.navigation_trace = []
        a.instruction_obj = NS(
            get_current_subtask=lambda: ('P1', {'instruction': 'Walk to the kitchen.'}),
            get_current_subtask_key=lambda: 'P1',
            is_last_subtask=lambda: True,
        )
        a.plan_rollback_path = Mock(side_effect=AssertionError('Raw labels must not choose a rollback checkpoint.'))
        corr = a._direct_verifier_correction(norm, ACTIONS)
        self.assertIsNotNone(corr)
        self.assertNotEqual(corr['type'], 'rollback')
        a.plan_rollback_path.assert_not_called()
        self.assertEqual(a.navigation_trace[0]['event'], 'wrong_path_rollback_unavailable')

    def test_junction_checkpoint_rollback_prefers_intermediate_junction_over_start(self):
        rb_methods = extract(
            SRC / 'agent/agent.py',
            {'_direct_verifier_correction', '_record_branch_checkpoint', '_update_last_checkpoint_choice', 'plan_rollback_path'},
            dict(NS_GLOBALS, copy=copy),
        )
        RollbackController = type('RollbackController', (fixtures.Controller,), rb_methods)
        a = RollbackController()
        a.navigation_cfg = {'enable_wrong_path_rollback': True}
        a._last_step_index = 3
        a.history_msgs = ['msg1', 'msg2']
        a.init_pos = np.array([0.0, 0.0, -1.2])
        a.init_rot = np.eye(3)
        a.mapper = NS(current_position=np.array([2.0, 0.0, 0.0]), current_rotation=np.eye(3), initial_floor_height=-1.2)
        a.instruction_obj = NS(
            get_current_subtask=lambda: ('P1', {'record': [{'step': 1}, {'step': 2}]}),
            get_current_subtask_key=lambda: 'P1',
            get_subtask_by_key=lambda k: {'record': [{'step': 1}, {'step': 2}]},
            sub_instruction_keys=['P1'],
            current_subtask_index=0,
        )
        junction_actions = {
            'WP_01': {'type': 'waypoint', 'view': 'front', 'visited': False, 'target_point_world': np.array([3.5, 0.0, -1.2])},
            'WP_02': {'type': 'waypoint', 'view': 'left', 'visited': False, 'target_point_world': np.array([3.5, 1.5, -1.2])},
        }
        a._record_branch_checkpoint(junction_actions)
        a._update_last_checkpoint_choice({
            'type': 'waypoint', 'action_id': 'WP_02', 'view': 'left',
            'target_point_world': np.array([3.5, 1.5, -1.2]),
        })

        # Move into wrong branch at (4.0, 1.5, 0.0) in mapper coords -> (4.0, 1.5, -1.2) ground coords
        a._last_step_index = 7
        a.mapper.current_position = np.array([4.0, 1.5, 0.0])
        a._validated_waypoint_route = lambda target_pos, recovery=False: {
            'path': [np.array([4.0, 1.5, -1.2]), np.asarray(target_pos, dtype=float)],
            'reachable': True,
        }
        rb_path, rb_rot = a.plan_rollback_path('P1')
        self.assertIsNotNone(rb_path)
        self.assertTrue(np.allclose(rb_path[-1], np.array([2.0, 0.0, -1.2])))
        self.assertIsNotNone(a._active_rollback_checkpoint)
        self.assertEqual(a._active_rollback_checkpoint['chosen_view'], 'LEFT')


if __name__ == '__main__':
    unittest.main()


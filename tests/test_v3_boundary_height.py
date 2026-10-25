"""Offline production-method regressions; no Habitat, API, or experiment writes."""
import ast
import copy
import json
import os
import re
import logging
import math
import time
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
sys.path.insert(0, str(SRC))
from progress_watchdog import NavigationAborted
from navigation_policy import normalize_verification
from navigation_contracts import completion_conditions, validate_condition_checks
from navigation_control import (ActionProtocolError, decode_action, legal_action_objects, action_prompt,
                                check_stair_witnesses, doorway_witnesses)
from runtime_budget import EpisodeBudget, EpisodeBudgetExceeded
from verification_feedback import (evidence_repair_required, endpoint_relocalization_required,
                                   initial_arrival_observation_retry_required, landing_observation_pending,
                                   remember_verification)


def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names), (names, [n.name for n in nodes])
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return {n.name: namespace[n.name] for n in nodes}


parser_ns = dict(re=re, json=json)
extract(SRC / 'agent/agent.py', ['robust_json_parse'], parser_ns)


METHODS = ['_verify_current_subtask', '_boundary_stop_decision', '_direct_verifier_correction', '_verification_accepts_stop',
           '_retrieve_navigation_memories', '_record_completed_subtask_memory']
Agent = type('Agent', (), extract(SRC / 'agent/agent.py', METHODS,
             dict(np=np, logging=logging, NavigationAborted=NavigationAborted,
                  copy=copy, json=json, math=math, normalize_verification=normalize_verification,
                  evidence_repair_required=evidence_repair_required, remember_verification=remember_verification,
                  endpoint_relocalization_required=endpoint_relocalization_required,
                  initial_arrival_observation_retry_required=initial_arrival_observation_retry_required,
                  landing_observation_pending=landing_observation_pending,
                  robust_json_parse=parser_ns['robust_json_parse'], ActionProtocolError=ActionProtocolError,
                  decode_action=decode_action, legal_action_objects=legal_action_objects, action_prompt=action_prompt)))
Guard = type('Guard', (), extract(SRC / 'agent/route_guard.py', ['_evaluate_route_verification'], dict(np=np, validate_condition_checks=validate_condition_checks,
                  check_stair_witnesses=check_stair_witnesses, doorway_witnesses=doorway_witnesses)))
Mapper = type('Mapper', (), extract(SRC / 'mapper.py', ['get_bev_height_points'], dict(np=np)))
ns = dict(np=np)
extract(SRC / 'agent/agent.py', ['_overlay_bev_height'], ns)
overlay = ns['_overlay_bev_height']

gpt_ns = dict(os=os, time=time, logging=logging, json=json)
GPT_METHODS = extract(SRC / 'agent/agent.py',
                      ['_next_request_seed', '_get_verifier_client',
                       '_create_chat_completion_with_backoff', '_get_verifier_response'],
                      gpt_ns)
GPTAgentHarness = type('GPTAgentHarness', (), GPT_METHODS)


class APITimeoutError(Exception):
    pass


class FakeCompletions:
    def __init__(self, client):
        self.client = client
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.client.create_error is not None:
            err = self.client.create_error
            self.client.create_error = None
            raise err
        payload = self.client.canned_response or json.dumps({
            'outcome': True,
            'relation_satisfied': True,
            'scene_consistent': True,
            'confidence': 'high',
            'evidence_state': 'supported',
            'evidence': 'Waiting outside the double doors.'
        })
        return NS(choices=[NS(message=NS(content=payload))])


class FakeChat:
    def __init__(self, client):
        self.completions = FakeCompletions(client)


class FakeOpenAIClient:
    def __init__(self, max_retries=2, canned_response=None, create_error=None):
        self.max_retries = max_retries
        self.canned_response = canned_response
        self.create_error = create_error
        self.chat = FakeChat(self)
        self.options_history = []
        self.clones = []

    def with_options(self, **options):
        self.options_history.append(options)
        new_retries = options.get('max_retries', self.max_retries)
        cloned = FakeOpenAIClient(
            max_retries=new_retries,
            canned_response=self.canned_response,
            create_error=self.create_error
        )
        self.clones.append(cloned)
        return cloned


def verdict(**changes):
    result = dict(available=True, outcome=True, relation_satisfied=True, scene_consistent=True,
                  handoff_consistent=True, contract_conflict=False, completion_evidence_reliable=True,
                  confidence='high', evidence_state='contradicted', evidence='Doorway remains ahead.')
    result.update(changes)
    return result


def agent(result=None):
    a = Agent()
    a.st = {'instruction': 'walk past the stairs', 'completed': False}
    a.instruction_obj = NS(get_current_subtask=lambda: ('P1', a.st), is_last_subtask=lambda: False,
                          get_subtask_by_key=lambda k: a.st, add_planner_memory=Mock())
    a.mapper = NS(current_position=np.zeros(3), current_rotation=np.eye(3), initial_floor_height=-1.2)
    a.navigation_cfg = {}
    a.navigation_trace = []
    a._verify_current_subtask = Mock(return_value=result if result is not None else verdict())
    a._scene_scope = lambda: 'scene'
    a.instruction = 'walk past stairs'
    a.navigation_memory = NS(search=Mock(return_value=[{'memory': 'one'}, {'memory': 'two'}]), add=Mock(return_value={}))
    return a


ACTIONS = {'WP_01': {'type': 'waypoint', 'target_point_world': [1, 0, 0]},
           'TURN_LEFT': {'type': 'turn', 'angle': 1.57}, 'STOP': {'type': 'stop'}}


def boundary(a, actions=ACTIONS):
    return a._boundary_stop_decision({'action': {'type': 'propose_completion'}}, None, [], None, {}, actions)


class BoundaryTests(unittest.TestCase):
    def test_pass_reuses_pose_but_rechecks_changed_pose(self):
        a = agent()
        self.assertEqual(boundary(a)['type'], 'stop')
        self.assertEqual(boundary(a)['type'], 'stop')
        self.assertEqual(a._verify_current_subtask.call_count, 1)
        a.mapper.current_position[0] = .2
        boundary(a)
        self.assertEqual(a._verify_current_subtask.call_count, 2)

    def test_reject_never_teleports_or_completes(self):
        a = agent(verdict(outcome=False, relation_satisfied=False, correction_action={'type': 'navigate', 'waypoint_id': 'WP_01'}))
        a._teleport_agent_to_pose = Mock(side_effect=AssertionError('teleport'))
        self.assertEqual(boundary(a)['type'], 'waypoint')
        self.assertFalse(a.st['completed'])
        a._teleport_agent_to_pose.assert_not_called()
        a.navigation_memory.add.assert_not_called()
        self.assertNotIn('is_verifier_correction', ACTIONS['WP_01'])

    def test_unavailable_verifier_aborts_without_claiming_completion(self):
        a = agent(normalize_verification(None))
        with self.assertRaisesRegex(NavigationAborted, 'verification_unavailable'):
            boundary(a)
        self.assertEqual(a._verify_current_subtask.call_count, 1)
        self.assertFalse(a.st['completed'])

    def test_last_check_can_pass_after_corrections(self):
        a = agent(verdict(outcome=False, correction_action={'type': 'navigate', 'waypoint_id': 'WP_01'}))
        for _ in range(3):
            self.assertEqual(boundary(a)['type'], 'waypoint')
            a.mapper.current_position[1] += .2
        a._verify_current_subtask.return_value = verdict()
        self.assertEqual(boundary(a)['type'], 'stop')
        self.assertEqual(a._verify_current_subtask.call_count, 4)

    def test_repeated_changed_pose_rejection_terminates(self):
        a = agent(verdict(outcome=False, relation_satisfied=False, correction_action={'type': 'navigate', 'waypoint_id': 'WP_01'}))
        for _ in range(3):
            boundary(a)
            a.mapper.current_position[1] += .2
        with self.assertRaises(NavigationAborted): boundary(a)
        self.assertEqual(a._verify_current_subtask.call_count, 4)
        with self.assertRaises(NavigationAborted): boundary(a)
        self.assertEqual(a._verify_current_subtask.call_count, 4)
        self.assertFalse(a.st['completed'])

    def test_three_explicit_rejections_never_become_completion(self):
        a = agent(verdict(outcome=False, relation_satisfied=False, correction_action={'type': 'navigate', 'waypoint_id': 'WP_01'}))
        a.navigation_cfg['max_verifier_checks_per_subtask'] = 3
        for _ in range(3):
            self.assertEqual(boundary(a)['type'], 'waypoint')
            a.mapper.current_position[0] += .1
        with self.assertRaisesRegex(NavigationAborted, 'check_budget_exhausted'):
            boundary(a)
        self.assertFalse(a.st['completed'])
        a.navigation_memory.add.assert_not_called()

    def test_new_observation_same_pose_is_not_stale_verifier_cache(self):
        a = agent()
        a._observation_revision = 1
        boundary(a)
        a._observation_revision = 2
        boundary(a)
        self.assertEqual(a._verify_current_subtask.call_count, 2)

    def test_budget_zero_and_missing_actions(self):
        a = agent(verdict(outcome=False))
        a.navigation_cfg['max_rollbacks_per_subtask'] = 0
        with self.assertRaises(NavigationAborted): boundary(a)
        for actions in ({}, {'STOP': {'type': 'stop'}},
                        {'WP_01': dict(ACTIONS['WP_01'], rejected=True)},
                        {'WP_01': {'type': 'waypoint', 'target_point_world': [0, 0, 0]}},
                        {'TURN_LEFT': {'type': 'turn', 'angle': 0}}):
            with self.assertRaisesRegex(NavigationAborted, 'no_legal'):
                boundary(agent(verdict(outcome=False)), actions)

    def test_unknown_never_uses_untrusted_translation(self):
        a = agent(verdict(available=False, correction_action={'type': 'navigate', 'waypoint_id': 'WP_01'}))
        with self.assertRaises(NavigationAborted): boundary(a, {'WP_01': ACTIONS['WP_01']})
        with self.assertRaisesRegex(NavigationAborted, 'verification_unavailable'):
            boundary(a)

    def test_translation_requires_reliable_rejection(self):
        for change in (dict(confidence='low'), dict(confidence=None),
                       dict(evidence_state='unknown'), dict(evidence_state=None),
                       dict(evidence=''), dict(evidence='  '), dict(contract_conflict=True),
                       dict(available=False), dict(outcome=True)):
            with self.subTest(change=change):
                result = verdict(outcome=False, correction_action={'type': 'navigate', 'waypoint_id': 'WP_01'})
                result.update(change)
                a = agent(result)
                self.assertEqual(a._direct_verifier_correction(result, ACTIONS)['type'], 'turn')
                with self.assertRaisesRegex(NavigationAborted, 'no_legal'):
                    a._direct_verifier_correction(result, {'WP_01': ACTIONS['WP_01']})
                self.assertFalse(a.st['completed'])
                a.navigation_memory.add.assert_not_called()
        for confidence in ('medium', 'high'):
            a = agent()
            result = verdict(outcome=False, relation_satisfied=False, confidence=confidence,
                             correction_action={'type': 'navigate', 'waypoint_id': 'WP_01'})
            self.assertEqual(a._direct_verifier_correction(result, ACTIONS)['type'], 'waypoint')

    def test_correction_total_not_reset_by_subtask(self):
        a = agent(verdict(outcome=False))
        a.navigation_cfg['max_verifier_corrections_total'] = 3
        for i in range(3):
            a.instruction_obj.get_current_subtask = lambda i=i: (str(i), a.st)
            boundary(a)
        a.instruction_obj.get_current_subtask = lambda: ('new', a.st)
        with self.assertRaises(NavigationAborted): boundary(a)

    def test_episode_budget_still_applies(self):
        a = agent()
        a.runtime_budget = EpisodeBudget(seconds=1, clock=lambda: 0)
        a.runtime_budget.clock = lambda: 2
        with self.assertRaises(EpisodeBudgetExceeded): boundary(a)
        a._verify_current_subtask.assert_not_called()

    def test_unknown_handoff_not_passed_and_final_null_allowed(self):
        a = agent()
        self.assertFalse(a._verification_accepts_stop(verdict(handoff_consistent=None)))
        a.instruction_obj.is_last_subtask = lambda: True
        self.assertTrue(a._verification_accepts_stop(verdict(handoff_consistent=None)))
        for field in ('outcome', 'relation_satisfied', 'scene_consistent', 'available'):
            self.assertFalse(a._verification_accepts_stop(verdict(**{field: None})))

    def test_route_guard_rejects_missing_scene_and_handoff(self):
        a = agent()
        a._route_verification_context = dict(key='P1', remaining=False, final=False)
        for parsed in (dict(outcome=True, relation_satisfied=True, thought='arrived'),
                       dict(outcome=True, relation_satisfied=True, thought='arrived', scene_consistent=True)):
            r = Guard._evaluate_route_verification(a, parsed, normalize_verification(copy.deepcopy(parsed)))
            self.assertFalse(r['outcome'])
            self.assertFalse(r['completion_evidence_reliable'])

    def test_memory_once_top1_even_after_instruction_cache_clear(self):
        a = agent()
        a.instruction_obj.get_cached_retrieval = lambda k: None
        a.instruction_obj.set_cached_retrieval = Mock()
        for _ in range(3): self.assertEqual(len(a._retrieve_navigation_memories()), 1)
        a.navigation_memory.search.assert_called_once()
        a.instruction_obj.get_current_subtask = lambda: ('P2', a.st)
        a._retrieve_navigation_memories()
        self.assertEqual(a.navigation_memory.search.call_count, 2)

    def test_failed_memory_search_not_retried(self):
        a = agent()
        a.navigation_memory.search.side_effect = RuntimeError('offline')
        for _ in range(2): self.assertEqual(a._retrieve_navigation_memories(), [])
        a.navigation_memory.search.assert_called_once()

    def test_untrusted_memory_not_written(self):
        a = agent()
        for v in ({}, verdict(completion_evidence_reliable=False), verdict(outcome=False)):
            a._record_completed_subtask_memory('P1', dict(a.st, completed=True, verifier=v))
        a.navigation_memory.add.assert_not_called()
        a._record_completed_subtask_memory('P1', dict(a.st, completed=True, verifier=verdict()))
        a.navigation_memory.add.assert_called_once()

    def test_actual_planner_and_renderer_wiring(self):
        source = (SRC / 'agent/agent.py').read_text()
        tree = ast.parse(source)
        funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        planner = ast.get_source_segment(source, funcs['decide_waypoint'])
        self.assertIn('self._boundary_stop_decision', planner)
        self.assertNotIn('_verify_current_subtask(', planner)
        self.assertNotIn('_verified_stop_subtasks', planner)
        renderer = ast.get_source_segment(source, funcs['create_top_down_map_centered'])
        self.assertIn('mapper.get_bev_height_points()', renderer)
        self.assertIn('_overlay_bev_height(', renderer)
        self.assertIn('create_top_down_map_centered(', planner)
        verifier_resp = ast.get_source_segment(source, funcs['_get_verifier_response'])
        self.assertIn('client_override', verifier_resp)
        self.assertIn('_get_verifier_client', verifier_resp)
        verifier_client_fn = ast.get_source_segment(source, funcs['_get_verifier_client'])
        self.assertIn('with_options', verifier_client_fn)
        self.assertIn('max_retries=0', verifier_client_fn)
        backoff_fn = ast.get_source_segment(source, funcs['_create_chat_completion_with_backoff'])
        self.assertIn('client_override', backoff_fn)


class VerificationChainTests(unittest.TestCase):
    """Execute real parser -> verifier method -> normalizer -> guard -> STOP gate."""
    def run_chain(self, payload, final):
        a = agent()
        del a._verify_current_subtask  # Use the production method, not a verdict mock.
        # This harness isolates scene/handoff contracts. Full RGB-depth endpoint
        # integration is covered by test_v3_endpoint_integration.
        a.navigation_cfg['require_endpoint_grounding'] = False
        a.instruction_obj.is_last_subtask = lambda: final
        a.instruction_obj.update_verifier = lambda result: a.st.update(verifier=result)
        checks = completion_conditions('Walk into the corridor.', 'P1')
        ledger = [dict(step=1, action_type='move_forward', motion_distance=.5,
                       pose=[0, 0, -1.2])]
        a._route_verification_context = dict(key='P1', remaining=False, final=final,
                                             checks=checks, ledger=ledger, entities=[])
        body = payload.get('output', payload)
        body.setdefault('condition_checks', [dict(id='P1.C1', status='satisfied',
                              evidence='Measured entry through the doorway.', steps=[1])])
        body.setdefault('target_identity', 'satisfied')
        body.setdefault('target_proximity', 'satisfied')
        body.setdefault('target_evidence', 'Currently outside the instructed doorway.')
        a._evaluate_route_verification = Guard._evaluate_route_verification.__get__(a)
        a._get_forward_clearance_and_waypoints = lambda obs, actions: (None, [])
        a._verification_prompt = lambda *args, **kwargs: 'offline verification input'
        a._get_verifier_response = lambda *args: json.dumps(payload)
        result = a._verify_current_subtask({'action': {'type': 'propose_completion'}}, None, [], None, actions=ACTIONS)
        self.assertNotIn('verifier exception', result['raw'])
        self.assertIs(a.st['verifier'], result)
        return a, result

    def test_missing_scene_final_and_intermediate_real_chain(self):
        for final in (True, False):
            for wrapped in (False, True):
                with self.subTest(final=final, wrapped=wrapped):
                    raw = dict(outcome=True, relation_satisfied=True, thought='arrived', confidence='high')
                    if not final:
                        raw['handoff_consistent'] = True  # No other rejection can mask this bug.
                    payload = {'reasoning': {}, 'output': raw} if wrapped else raw
                    a, result = self.run_chain(payload, final)
                    self.assertFalse(result['outcome'])
                    self.assertFalse(result['completion_evidence_reliable'])
                    self.assertEqual(result['reason'], 'Route-scene evidence is unknown.')
                    self.assertFalse(a._verification_accepts_stop(result))
                    self.assertEqual(a._direct_verifier_correction(result, ACTIONS)['type'], 'turn')

    def test_explicit_scene_values_real_chain(self):
        for final in (True, False):
            for scene in (True, False, None):
                with self.subTest(final=final, scene=scene):
                    a, result = self.run_chain(dict(outcome=True, relation_satisfied=True,
                        thought='arrived', confidence='high', scene_consistent=scene,
                        handoff_consistent=None if final else True), final)
                    self.assertEqual(a._verification_accepts_stop(result), scene is True)
                    self.assertEqual(result['completion_evidence_reliable'], scene is True)

    def test_rejection_confidence_real_chain(self):
        for confidence in ('low', 'medium', 'high'):
            with self.subTest(confidence=confidence):
                a, result = self.run_chain(dict(outcome=False, relation_satisfied=False,
                    thought='Door threshold is still ahead in the front view.',
                    scene_consistent=True, handoff_consistent=False, confidence=confidence,
                    failure_type='not_reached', correction_action={'type': 'navigate', 'waypoint_id': 'WP_01'}), final=False)
                self.assertFalse(a._verification_accepts_stop(result))
                self.assertEqual(result['evidence_state'], 'contradicted')
                decision = a._direct_verifier_correction(result, ACTIONS)
                self.assertEqual(decision['type'], 'turn' if confidence == 'low' else 'waypoint')

    def test_positive_verdict_needs_executed_conditions(self):
        for rows in ([], [dict(id='P1.C1', status='satisfied', evidence='Claimed movement', steps=[999])],
                     [dict(id='P1.C1', status='unknown', evidence='', steps=[])]):
            with self.subTest(rows=rows):
                a, result = self.run_chain(dict(outcome=True, relation_satisfied=True,
                    scene_consistent=True, thought='Looks like the destination.',
                    condition_checks=rows), final=True)
                self.assertFalse(a._verification_accepts_stop(result))
                self.assertFalse(result['completion_evidence_reliable'])

    def test_positive_verdict_needs_current_target_identity(self):
        for fields in (dict(target_identity='unknown'), dict(target_proximity='unknown'),
                       dict(target_evidence=''), dict(target_entity_id=99)):
            a, result = self.run_chain(dict(outcome=True, relation_satisfied=True,
                scene_consistent=True, thought='Looks like the destination.', **fields), final=True)
            self.assertFalse(a._verification_accepts_stop(result))
            self.assertFalse(result['completion_evidence_reliable'])



class HeightTests(unittest.TestCase):
    def test_height_bands_empty_and_nonfinite(self):
        image = np.full((8, 8, 3), 200, dtype=np.uint8)
        project = lambda pts: pts[:, :2].astype(int)
        for pts in ([], [[np.nan, 0, 0]]):
            np.testing.assert_array_equal(overlay(image.copy(), pts, project, 0), image)
        pts = np.array([[1, 1, 0], [2, 2, .3], [3, 3, .8], [4, 4, -.3], [99, 99, 1]])
        before = pts.copy()
        out = overlay(image.copy(), pts, project, 0)
        np.testing.assert_array_equal(out[1, 1], image[1, 1])
        self.assertGreater(out[2, 2, 2], out[2, 2, 0])
        self.assertGreater(out[4, 4, 0], out[4, 4, 2])
        self.assertFalse(np.array_equal(out[2, 2], out[3, 3]))
        np.testing.assert_array_equal(pts, before)

    @unittest.skipIf(cv2 is None, 'OpenCV unavailable; use hsgm environment for full renderer')
    def test_actual_centered_renderer_empty_and_stair_overlay(self):
        namespace = dict(np=np, cv2=cv2, time=time, math=math, Instruct_Mapper=Mapper)
        extract(SRC / 'agent/agent.py', ['draw_direction_markers', '_overlay_bev_height',
                                       'create_top_down_map_centered'], namespace)
        empty = NS(is_empty=lambda: True)
        m = NS(current_position=np.zeros(3), current_rotation=np.eye(3),
               floor_height=0, navigable_pcd=empty, obstacle_pcd=empty, frontier_pcd=empty,
               object_entities=[], trajectory_position=[], get_bev_height_points=lambda: np.empty((0, 3)))
        render = namespace['create_top_down_map_centered']
        baseline = render(m, 90, map_size_px=256, map_scale=.025, crop_size_m=6)
        m.get_bev_height_points = lambda: np.array([[1, 1, .5], [1, -1, -.5]])
        colored = render(m, 90, map_size_px=256, map_scale=.025, crop_size_m=6)
        self.assertEqual(colored.shape, baseline.shape)
        self.assertGreater(np.count_nonzero(colored != baseline), 0)
        self.assertLess(np.count_nonzero(colored != baseline), 300)

    def test_mapper_retains_stairs_display_only_bounded(self):
        def cloud(points):
            points = np.asarray(points).reshape(-1, 3)
            return NS(is_empty=lambda: not len(points), point=NS(positions=NS(cpu=lambda: NS(numpy=lambda: points))))
        m = Mapper()
        m.current_position = np.zeros(3)
        m.floor_height = 0
        m.navigable_pcd = cloud([])
        m.stair_pcd = cloud([[1, 1, 2], [1, 1, -2], [1, 1, 4], [9, 9, 1]])
        np.testing.assert_array_equal(m.get_bev_height_points(), [[1, 1, 2], [1, 1, -2]])
        m.stair_pcd = cloud([])
        self.assertEqual(m.get_bev_height_points().shape, (0, 3))
        m.stair_pcd = cloud(np.tile([1, 1, .5], (25001, 1)))
        self.assertLessEqual(len(m.get_bev_height_points()), 10000)


class VerifierTimeoutClientTests(unittest.TestCase):
    """Offline regression tests for verifier client timeout and max_retries=0 isolation."""

    def _make_agent(self, client):
        a = GPTAgentHarness()
        a.client = client
        a.model_name = "qwen3.8-flash"
        a.navigation_cfg = {'model_request_timeout_s': 90, 'model_max_retries': 2}
        a._pil_to_base64 = lambda img: "mock_base64"
        return a

    def test_verifier_client_max_retries_zero_and_single_create_call(self):
        planner_client = FakeOpenAIClient(max_retries=2)
        a = self._make_agent(planner_client)

        resp = a._get_verifier_response('front', 'left', 'right', 'map', 'check stop')
        data = json.loads(resp)
        self.assertTrue(data.get('outcome'))

        # Planner client retry setting remains untouched at 2
        self.assertEqual(planner_client.max_retries, 2)
        self.assertEqual(len(planner_client.options_history), 1)
        self.assertEqual(planner_client.options_history[0], {'max_retries': 0})
        self.assertEqual(len(planner_client.chat.completions.calls), 0)

        # Cloned verifier client has max_retries=0 and makes exactly 1 create call
        self.assertEqual(len(planner_client.clones), 1)
        verifier_client = planner_client.clones[0]
        self.assertEqual(verifier_client.max_retries, 0)
        self.assertEqual(len(verifier_client.chat.completions.calls), 1)

        # Confirm client_override is isolated and not passed to endpoint
        call_kwargs = verifier_client.chat.completions.calls[0]
        self.assertNotIn('client_override', call_kwargs)
        self.assertEqual(call_kwargs['model'], "qwen3.8-flash")
        self.assertEqual(call_kwargs['timeout'], 90.0)
        self.assertIn('seed', call_kwargs)
        self.assertIn('messages', call_kwargs)

    def test_verifier_timeout_fails_fast_single_call_no_sdk_retries(self):
        planner_client = FakeOpenAIClient(max_retries=2, create_error=APITimeoutError("Request timed out."))
        a = self._make_agent(planner_client)

        resp = a._get_verifier_response('front', 'left', 'right', 'map', 'check stop')
        data = json.loads(resp)
        self.assertIn('error', data)
        self.assertIn('APITimeoutError', data['error'])

        # Exactly 1 call was attempted before failing fast without SDK retries
        self.assertEqual(len(planner_client.clones), 1)
        verifier_client = planner_client.clones[0]
        self.assertEqual(verifier_client.max_retries, 0)
        self.assertEqual(len(verifier_client.chat.completions.calls), 1)

    def test_backoff_client_override_kwargs_isolation_and_budget(self):
        fake_client = FakeOpenAIClient(max_retries=0)
        a = GPTAgentHarness()
        now = [10.0]
        a.client = FakeOpenAIClient(max_retries=2)
        a.runtime_budget = EpisodeBudget(seconds=60, clock=lambda: now[0])
        now[0] = 20.0
        a.navigation_cfg = {'model_request_timeout_s': 90, 'model_max_retries': 0}

        resp = a._create_chat_completion_with_backoff(
            operation="verifier_request",
            max_retries=0,
            client_override=fake_client,
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
        )
        self.assertIsNotNone(resp)
        self.assertEqual(len(fake_client.chat.completions.calls), 1)
        self.assertEqual(len(a.client.chat.completions.calls), 0)

        call_kwargs = fake_client.chat.completions.calls[0]
        self.assertNotIn('client_override', call_kwargs)
        # Timeout bounded by budget remaining (60 - 10 = 50s < 90s)
        self.assertAlmostEqual(call_kwargs['timeout'], 50.0)

    def test_thinking_configuration_reaches_actual_request_kwargs(self):
        cases = [({}, {}, None), ({'HSGM_DISABLE_THINKING': '0'}, {}, None),
                 ({'HSGM_ENABLE_THINKING': '1'}, {}, True),
                 ({'HSGM_ENABLE_THINKING': '0'}, {}, False),
                 ({'HSGM_DISABLE_THINKING': '1'}, {}, False),
                 ({'HSGM_ENABLE_THINKING': '1', 'HSGM_DISABLE_THINKING': '1'}, {}, False),
                 ({'HSGM_ENABLE_THINKING': '0'}, {'enable_thinking': True}, False),
                 ({'HSGM_DISABLE_THINKING': '1'}, {'enable_thinking': True}, False),
                 ({}, {'enable_thinking': False}, False)]
        for env, overrides, expected in cases:
            with self.subTest(env=env, overrides=overrides), patch.dict(os.environ, env, clear=True):
                client = FakeOpenAIClient(max_retries=0)
                a = self._make_agent(client)
                body = {'custom_option': 7, **overrides}
                a._create_chat_completion_with_backoff(model=a.model_name, messages=[],
                                                       max_tokens=1536, extra_body=body)
                kwargs = client.chat.completions.calls[0]
                if expected is None:
                    self.assertNotIn('enable_thinking', kwargs['extra_body'])
                else:
                    self.assertIs(kwargs['extra_body']['enable_thinking'], expected)
                self.assertEqual(kwargs['extra_body']['custom_option'], 7)
                self.assertEqual(kwargs['max_tokens'], 1536)
                self.assertEqual(body, {'custom_option': 7, **overrides})

    def test_verifier_request_actually_disables_thinking(self):
        with patch.dict(os.environ, {'HSGM_DISABLE_THINKING': '1', 'HSGM_ENABLE_THINKING': '0'}, clear=True):
            client = FakeOpenAIClient(max_retries=0)
            a = self._make_agent(client)
            a._get_verifier_response('front', 'left', 'right', 'map', 'check stop')
        kwargs = client.clones[0].chat.completions.calls[0]
        self.assertIs(kwargs['extra_body']['enable_thinking'], False)
        self.assertEqual(kwargs['max_tokens'], 1536)

    def test_local_contract_preflight_disables_thinking_in_both_compatibility_fields(self):
        conditions = [{'id': 'C1'}]
        parsed = {'condition_checks': [{'id': 'C1', 'status': 'unknown', 'steps': []}]}
        client = FakeOpenAIClient(canned_response=json.dumps(parsed))
        namespace = dict(os=os, json=json, OpenAI=lambda **kwargs: client,
                         completion_conditions=lambda *args: conditions,
                         verification_schema=lambda *args: {'schema': {'type': 'object'}},
                         normalize_verification=lambda value: dict(available=True, outcome=False, rollback_required=False))
        main = extract(SRC / 'route_contract_preflight.py', ['main'], namespace)['main']
        with patch.dict(os.environ, {'OPENAI_BASE_URL': 'http://127.0.0.1:8000/v1', 'OPENAI_MODEL': 'test'}, clear=True):
            main()
        body = client.chat.completions.calls[0]['extra_body']
        self.assertIs(body['enable_thinking'], False)
        self.assertIs(body['chat_template_kwargs']['enable_thinking'], False)
        self.assertIn('structured_outputs', body)


if __name__ == '__main__':
    unittest.main()


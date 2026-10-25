"""CPU checks through production decision, image-request and API-kwargs methods."""
import ast
import copy
from contextlib import redirect_stdout
import io
import json
import logging
import math
import os
import re
import time
from types import SimpleNamespace as NS
from typing import Optional
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import test_v3_navigation_control as control
from test_v3_boundary_height import SRC, extract
from navigation_control import normalize_action_object
from runtime_budget import EpisodeBudget, EpisodeBudgetExceeded


namespace = dict(os=os, re=re, json=json, logging=logging, time=time, np=np,
                 Image=Image, PIL=NS(Image=Image), Optional=Optional)
tree = ast.parse((SRC / 'agent/agent.py').read_text())
gpt = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'GPTAgent')
names = {'_get_vlm_response', '_get_vlm_response_multiview', '_commit_planner_history',
         '_next_request_seed', '_create_chat_completion_with_backoff',
         '_get_verifier_client', '_get_verifier_response', '_get_llm_response',
         '_select_history_keyframes'}
methods = [node for node in gpt.body if isinstance(node, ast.FunctionDef) and node.name in names]
assert len(methods) == len(names)
exec(compile(ast.Module(body=methods, type_ignores=[]), str(SRC / 'agent/agent.py'), 'exec'), namespace)
decide = extract(SRC / 'agent/agent.py', ['decide_waypoint'],
                 dict(control.NS_GLOBALS, normalize_action_object=normalize_action_object))['decide_waypoint']


def response(action=None, **fields):
    return json.dumps({'observation': 'Current corridor.', 'thought': 'Select current waypoint.',
                       'plan': 'Continue along the corridor.',
                       'action': action or {'type': 'navigate', 'waypoint_id': 'WP_01'}, **fields})


class ScriptedClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.chat = NS(completions=self)

    def with_options(self, **kwargs):
        return self

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        value = next(self.responses)
        if isinstance(value, BaseException):
            raise value
        return NS(choices=[NS(message=NS(content=value))])


def planner(responses, single_view=False):
    agent = control.planner([])
    for name in names:
        setattr(agent, name, namespace[name].__get__(agent))
    agent.decide_waypoint = decide.__get__(agent)
    agent.client = ScriptedClient(responses)
    agent.model_name = 'test-model'
    agent.history_msgs = [{'role': 'system', 'content': 'Initial policy.'}]
    agent._pil_to_base64 = lambda image: 'encoded_current_image'
    agent._pil_to_base64_small = lambda image, width: 'encoded_history_image'
    agent.runtime_budget = EpisodeBudget(seconds=600, max_requests=20)
    if single_view:
        # Exercise the same production decision/parse/decode/commit boundary
        # with the alternate request builder, without copying its logic.
        agent._get_vlm_response_multiview = (
            lambda front, left, right, back, map_image, prompt, images, **kwargs:
            agent._get_vlm_response(front, map_image, prompt, images))
    return agent


class PlannerHistoryTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.output = redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_invalid_long_answer_never_enters_retry_or_long_term_history(self):
        invalid = 'UNRELATED_LONG_REPLY ' * 2000
        agent = planner([invalid, response()])
        selected, _, _ = agent.decide_waypoint()
        self.assertEqual(selected['action_id'], 'WP_01')
        self.assertEqual(len(agent.client.calls), 2)
        self.assertEqual(agent.runtime_budget.requests, 2)
        self.assertEqual(len(agent.history_msgs), 3)
        for call in agent.client.calls:
            self.assertNotIn('UNRELATED_LONG_REPLY', json.dumps(call['messages']))
            self.assertEqual([message['role'] for message in call['messages']], ['system', 'user'])
            self.assertIn('encoded_current_image', json.dumps(call['messages'][-1]))
        retry = agent.client.calls[1]['messages'][-1]['content'][0]['text']
        original = agent.client.calls[0]['messages'][-1]['content'][0]['text']
        self.assertIn('Previous response rejected:', retry)
        self.assertLess(len(retry) - len(original), 500)
        self.assertNotIn('UNRELATED_LONG_REPLY', json.dumps(agent.history_msgs))
        self.assertEqual(json.loads(agent.history_msgs[-1]['content'])['action']['waypoint_id'], 'WP_01')

    def test_parseable_but_illegal_action_is_not_committed(self):
        agent = planner([response({'type': 'navigate', 'waypoint_id': 'WP_99'}, plan='INVALID_PLAN'), response()])
        agent.decide_waypoint()
        self.assertEqual(len(agent.client.calls), 2)
        self.assertNotIn('INVALID_PLAN', json.dumps(agent.history_msgs))
        self.assertNotIn('WP_99', json.dumps(agent.client.calls[1]['messages']))
        agent.instruction_obj.update_plan_for_current_subtask.assert_called_once_with('Continue along the corridor.')

    def test_three_failed_attempts_keep_history_clean_and_no_stop(self):
        agent = planner(['not JSON'] * 3 + [response()])
        self.assertIsNone(agent.decide_waypoint()[0])
        self.assertEqual(len(agent.client.calls), 3)
        self.assertEqual(agent.runtime_budget.requests, 3)
        self.assertEqual(len(agent.history_msgs), 1)
        self.assertIsNone(agent._pending_planner_history)
        agent._boundary_stop_decision.assert_not_called()
        self.assertEqual(len([e for e in agent.navigation_trace if e.get('event') == 'action_protocol_rejected']), 3)
        self.assertEqual(agent.decide_waypoint()[0]['action_id'], 'WP_01')
        self.assertEqual(len(agent.history_msgs), 3)

    def test_single_view_also_commits_only_after_real_action_decoding(self):
        agent = planner([response({'type': 'navigate', 'waypoint_id': 'WP_99'}), response()], single_view=True)
        self.assertEqual(agent.decide_waypoint()[0]['action_id'], 'WP_01')
        self.assertEqual(len(agent.history_msgs), 3)
        self.assertNotIn('WP_99', json.dumps(agent.client.calls[1]['messages']))
        self.assertNotIn('WP_99', json.dumps(agent.history_msgs))

    def test_transport_failure_cannot_commit_stale_observation(self):
        agent = planner([RuntimeError('offline request failure'), response()])
        agent._pending_planner_history = {'role': 'user', 'content': 'STALE_OBSERVATION'}
        self.assertEqual(agent.decide_waypoint()[0]['action_id'], 'WP_01')
        self.assertNotIn('STALE_OBSERVATION', json.dumps(agent.history_msgs))
        self.assertEqual(len(agent.history_msgs), 3)

    def test_existing_accepted_history_survives_all_failed_retries(self):
        agent = planner([response()] + ['not JSON'] * 3)
        agent.decide_waypoint()
        accepted = copy.deepcopy(agent.history_msgs)
        self.assertIsNone(agent.decide_waypoint()[0])
        self.assertEqual(agent.history_msgs, accepted)

    def test_valid_wrapped_json_is_stored_as_canonical_json(self):
        agent = planner(['<think>PRIVATE_REASONING</think>\n```json\n' + response() + '\n```'])
        self.assertEqual(agent.decide_waypoint()[0]['action_id'], 'WP_01')
        stored = agent.history_msgs[-1]['content']
        self.assertEqual(json.loads(stored)['action']['waypoint_id'], 'WP_01')
        self.assertNotIn('```', stored)
        self.assertNotIn('PRIVATE_REASONING', stored)

    def test_completion_proposal_still_requires_independent_verification(self):
        agent = planner([response({'type': 'propose_completion'})])
        agent._boundary_stop_decision.return_value = {'type': 'continue_path'}
        self.assertEqual(agent.decide_waypoint()[0]['type'], 'continue_path')
        agent._boundary_stop_decision.assert_called_once()
        self.assertEqual(json.loads(agent.history_msgs[-1]['content'])['action']['type'], 'propose_completion')

    def test_request_budget_exhaustion_propagates_without_more_attempts_or_history(self):
        agent = planner(['not JSON', response()])
        agent.runtime_budget = EpisodeBudget(seconds=600, max_requests=1)
        with self.assertRaises(EpisodeBudgetExceeded):
            agent.decide_waypoint()
        self.assertEqual(len(agent.client.calls), 1)
        self.assertEqual(len(agent.history_msgs), 1)
        self.assertIsNone(agent._pending_planner_history)


class RequestFormatTests(unittest.TestCase):
    def test_opt_in_only_changes_json_operations(self):
        for enabled in (None, '0', '1'):
            for operation in ('planner_request', 'verifier_request', 'instruction_decomposition', 'llm_request'):
                env = {} if enabled is None else {'HSGM_JSON_OBJECT_RESPONSES': enabled}
                with self.subTest(enabled=enabled, operation=operation), patch.dict(os.environ, env, clear=True):
                    agent = planner(['{}'])
                    agent._create_chat_completion_with_backoff(operation=operation, model='test', messages=[], max_tokens=1234)
                    sent = agent.client.calls[0]
                    if enabled == '1' and operation in ('planner_request', 'verifier_request'):
                        self.assertEqual(sent['response_format'], {'type': 'json_object'})
                    else:
                        self.assertNotIn('response_format', sent)
                    self.assertEqual(sent['max_tokens'], 1234)
                    self.assertEqual(agent.runtime_budget.requests, 1)

    def test_verifier_and_decomposition_use_real_request_builders(self):
        with patch.dict(os.environ, {'HSGM_JSON_OBJECT_RESPONSES': '1'}, clear=True), redirect_stdout(io.StringIO()):
            agent = planner(['{}', 'plain text route'])
            before = copy.deepcopy(agent.history_msgs)
            agent._get_verifier_response(None, None, None, None, 'Return JSON verification.')
            self.assertEqual(agent.client.calls[0]['response_format'], {'type': 'json_object'})
            self.assertEqual(agent._get_llm_response('Split this instruction.'), 'plain text route')
            self.assertNotIn('response_format', agent.client.calls[1])
            self.assertEqual(agent.history_msgs, before)

    def test_penalties_are_floats_only_when_configured(self):
        for env, expected in (({}, {}), ({'HSGM_PRESENCE_PENALTY': '0', 'HSGM_FREQUENCY_PENALTY': '0'},
                                      {'presence_penalty': 0., 'frequency_penalty': 0.}),
                              ({'HSGM_PRESENCE_PENALTY': '1.5'}, {'presence_penalty': 1.5})):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                agent = planner(['{}'])
                agent._create_chat_completion_with_backoff(operation='planner_request', model='test', messages=[])
                sent = agent.client.calls[0]
                for name in ('presence_penalty', 'frequency_penalty'):
                    if name in expected:
                        self.assertEqual(sent[name], expected[name])
                        self.assertIsInstance(sent[name], float)
                    else:
                        self.assertNotIn(name, sent)


if __name__ == '__main__':
    unittest.main()


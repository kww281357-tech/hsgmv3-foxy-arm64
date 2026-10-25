"""CPU integration tests for verification request inputs, init vocab reset, and override history."""
import ast
import json
import math
import os
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

import test_v3_navigation_control as control
from test_v3_boundary_height import SRC, extract
import test_v3_planner_history_protocol as history_fixture

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent.verification_inputs import VerificationInputs
from navigation_policy import checkpoint_yaw


class FakePolarAction:
    def __init__(self, r=0.0, theta=0.0, action_type=None):
        self.r = float(r)
        self.theta = float(theta)
        if action_type is not None:
            self.type = action_type
        elif self.r == 0.0 and self.theta != 0.0:
            self.type = 'turn'
        elif self.r == 0.0 and self.theta == 0.0:
            self.type = 'stop'
        else:
            self.type = 'move_forward'


FakePolarAction.null = FakePolarAction(0.0, 0.0, 'stop')

EXECUTE_ACTION = extract(
    SRC / 'agent/agent.py',
    ['execute_action'],
    dict(control.NS_GLOBALS, checkpoint_yaw=checkpoint_yaw, PolarAction=FakePolarAction),
)['execute_action']


class _StopAtFirstUpdateMap(RuntimeError):
    pass


def _make_raw_obs(front_val, left_val=60, right_val=90, back_val=120, pos=(0.0, 0.0, 0.0)):
    def _view(val):
        return {
            'color_sensor': np.full((24, 32, 3), val, dtype=np.uint8),
            'depth_sensor': np.full((24, 32), 1.5, dtype=np.float32),
        }

    front = _view(front_val)
    return {
        'color_sensor': front['color_sensor'],
        'depth_sensor': front['depth_sensor'],
        'left': _view(left_val),
        'right': _view(right_val),
        'back': _view(back_val),
        'agent_state': NS(position=np.asarray(pos, dtype=float), rotation=np.eye(3)),
    }


class VerificationRequestIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def _make_exec_agent(self, obs_id=5):
        agent = control.planner([])
        for attr in dir(VerificationInputs):
            if not attr.startswith('__'):
                val = getattr(VerificationInputs, attr)
                if callable(val):
                    setattr(agent, attr, val.__get__(agent))
        agent.execute_action = EXECUTE_ACTION.__get__(agent)
        agent.navigation_cfg = {
            'verification_evidence_enabled': True,
            'path_arrival_tolerance_m': 0.08,
            'route_height_tolerance_m': 0.25,
        }
        agent.curr_obs = _make_raw_obs(front_val=42, left_val=64, right_val=96, back_val=128, pos=(0.0, 0.0, 0.0))
        agent.mapper = NS(
            current_position=np.array([0.0, 0.0, 0.0], dtype=float),
            initial_floor_height=0.0,
            current_rotation=np.eye(3),
        )
        if obs_id is not None:
            agent.mapper.observation_id = obs_id
        agent._verification_obs = {'stale': True}
        agent.traveled_distance = 0.0
        agent._execution_ledger = []
        agent.current_path = []
        agent.rolling_back = False
        agent._pending_recovery = None
        agent._blocked_move_count = 0
        agent._last_step_index = 3
        agent.instruction_obj = NS(
            get_current_subtask_key=lambda: '0:0',
            add_record_to_current_subtask=lambda rec: None,
        )
        agent._action_to_text = lambda act: 'turn' if act.type == 'turn' else 'move_forward'
        agent._normalize_angle = lambda ang: float(ang)
        agent._landing_support_observed = lambda: True
        return agent

    def test_execute_action_snapshots_before_in_place_mutation_and_orders_cache(self):
        agent = self._make_exec_agent(obs_id=5)
        old_front_buf = agent.curr_obs['color_sensor']
        old_left_buf = agent.curr_obs['left']['color_sensor']
        events = []

        def _step(action):
            events.append(('sim_step_mutating_old_buffers', int(old_front_buf[0, 0, 0])))
            old_front_buf[:] = 250
            old_left_buf[:] = 251
            return _make_raw_obs(front_val=180, left_val=190, right_val=200, back_val=210, pos=(0.25, 0.0, 0.0))

        agent.sim_wrapper = NS(
            sim=NS(get_agent=lambda idx: NS(get_state=lambda: NS(position=np.array([0.0, 0.0, 0.0])))),
            step=_step,
        )

        def _update_map(obs):
            events.append(('update_map', agent.mapper.observation_id))
            agent.mapper.observation_id = 6
            agent.mapper.current_position = np.array([0.25, 0.0, 0.0], dtype=float)
            return {'inst_1': {}}

        agent.update_map = _update_map

        def _ensure_stairs():
            def _observe(record, landing_supported):
                events.append(('stairs_observe', landing_supported))
                return None

            return NS(direction='up', observe=_observe, packet=lambda: {'direction': 'up', 'steps': 1})

        agent._ensure_stair_progress = _ensure_stairs

        def _observe_turn(record):
            events.append(('observe_turn', record['observation_id_before'], record['observation_id_after']))
            record['turn_execution'] = {'purpose': 'landmark_turn', 'target_view': 'left'}

        agent._observe_turn_execution = _observe_turn

        real_cache = agent._cache_verification_frames
        captured_snapshots = {}

        def _wrapped_cache(record, before_snapshot, obs):
            events.append((
                'cache_frames',
                'turn_execution' in record,
                'stair_progress' in record,
                'landing_supported' in record,
                agent._verification_obs,
            ))
            captured_snapshots['before'] = before_snapshot
            real_cache(record, before_snapshot, obs)

        agent._cache_verification_frames = _wrapped_cache

        agent.execute_action(FakePolarAction(r=0.25, theta=0.0))

        self.assertIsNone(agent._verification_obs)
        self.assertEqual(len(agent._execution_ledger), 1)
        rec = agent._execution_ledger[0]
        self.assertEqual(rec['observation_id_before'], 5)
        self.assertEqual(rec['observation_id_after'], 6)
        self.assertIsNone(rec['recovery_source'])

        before_snap = captured_snapshots['before']
        self.assertIsNotNone(before_snap)
        self.assertEqual(before_snap['observation_id'], 5)
        self.assertEqual(before_snap['views']['front'].getpixel((0, 0)), (42, 42, 42))
        self.assertEqual(before_snap['views']['left'].getpixel((0, 0)), (64, 64, 64))

        cached = getattr(agent, '_verification_event_frames', [])
        self.assertEqual(len(cached), 8)

        def _find_cached(phase, view, obs_id):
            matches = [
                fr for fr in cached
                if fr['phase'] == phase and fr['view'] == view and fr['observation_id'] == obs_id
            ]
            self.assertEqual(len(matches), 1)
            return matches[0]

        before_front = _find_cached('BEFORE', 'front', 5)
        before_left = _find_cached('BEFORE', 'left', 5)
        after_front = _find_cached('AFTER', 'front', 6)
        after_left = _find_cached('AFTER', 'left', 6)
        self.assertEqual(before_front['step'], 0)
        if 'rec' in locals():
            self.assertEqual(rec['step'], 0)
            self.assertEqual(rec['controller_step'], 3)
        self.assertEqual(before_front['subtask_id'], '0:0')
        self.assertEqual(before_front['image'].getpixel((0, 0)), (42, 42, 42))
        self.assertEqual(before_left['image'].getpixel((0, 0)), (64, 64, 64))
        self.assertEqual(after_front['image'].getpixel((0, 0)), (180, 180, 180))
        self.assertEqual(after_left['image'].getpixel((0, 0)), (190, 190, 190))

        event_names = [e[0] for e in events]
        self.assertEqual(
            event_names,
            ['sim_step_mutating_old_buffers', 'update_map', 'observe_turn', 'stairs_observe', 'cache_frames'],
        )
        cache_evt = events[-1]
        self.assertEqual(cache_evt, ('cache_frames', True, True, True, None))

    def test_execute_action_captures_recovery_source_before_clear_and_keeps_missing_ids_none(self):
        agent = self._make_exec_agent(obs_id=None)
        agent.mapper.observation_id = True
        agent.rolling_back = True
        agent._pending_recovery = {'recovery_source': 'verifier_rejection', 'checkpoint_id': 'cp_0'}
        order = []

        agent.sim_wrapper = NS(
            sim=NS(get_agent=lambda idx: NS(get_state=lambda: NS(position=np.array([0.0, 0.0, 0.0])))),
            step=lambda action: _make_raw_obs(front_val=110, pos=(0.0, 0.0, 0.0)),
        )
        agent.update_map = lambda obs: setattr(agent.mapper, 'observation_id', -3) or {}
        agent._ensure_stair_progress = lambda: NS(direction=None, observe=lambda r, l: None, packet=lambda: {})

        real_cache = agent._cache_verification_frames

        def _spy_cache(record, before_snapshot, obs):
            order.append(('cache', getattr(agent, '_pending_recovery', None) is not None, record['recovery_source']))
            real_cache(record, before_snapshot, obs)

        def _observe_recovery(action):
            order.append(('observe_recovery_clear',))
            agent._pending_recovery = None
            agent.rolling_back = False

        agent._cache_verification_frames = _spy_cache
        agent._observe_recovery_action = _observe_recovery

        agent.execute_action(FakePolarAction(r=0.0, theta=0.5))

        rec = agent._execution_ledger[-1]
        self.assertIsNone(rec['observation_id_before'])
        self.assertIsNone(rec['observation_id_after'])
        self.assertTrue(rec['recovery'])
        self.assertEqual(rec['recovery_source'], 'verifier_rejection')
        self.assertIsNone(agent._pending_recovery)
        self.assertEqual(order, [('cache', True, 'verifier_rejection'), ('observe_recovery_clear',)])

        agent.rolling_back = True
        agent._pending_recovery = {'checkpoint_id': 'cp_without_source'}
        agent._last_rollback_info = {'recovery_source': 'must_not_infer'}
        agent.update_map = lambda obs: setattr(agent.mapper, 'observation_id', 4.5) or {}
        agent.execute_action(FakePolarAction(r=0.0, theta=0.5))
        rec2 = agent._execution_ledger[-1]
        self.assertIsNone(rec2['observation_id_before'])
        self.assertIsNone(rec2['observation_id_after'])
        self.assertIsNone(rec2['recovery_source'])

    def test_init_configures_task_vocabulary_even_when_disabled_before_first_update_map(self):
        call_log = []

        def _configure_task_vocabulary(instruction, enabled=True, max_extra_terms=16):
            call_log.append(('configure_task_vocabulary', instruction, enabled, max_extra_terms))
            return {'instruction': instruction, 'enabled': enabled, 'max_extra_terms': max_extra_terms}

        class _FakeMapper:
            def __init__(self, *args, **kwargs):
                call_log.append(('mapper_init',))

            def reset(self, position, rotation):
                call_log.append(('mapper_reset',))

        tree = ast.parse((SRC / 'agent/agent.py').read_text())
        planner_cls = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'PathPlannerAgent'
        )
        init_node = next(
            node for node in planner_cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__'
        )
        init_ns = dict(
            control.NS_GLOBALS,
            os=os,
            np=np,
            SimWrapper=object,
            super=lambda *args, **kwargs: NS(__init__=lambda *a, **kw: None),
            calculate_focal_length=lambda fov, w: 256.0,
            Instruction=lambda full_instruction, sub_instructions: NS(
                full_instruction=full_instruction, sub_instructions=sub_instructions
            ),
            Instruct_Mapper=_FakeMapper,
            PolarAction=FakePolarAction,
            segmentation=NS(instance_segmentation=NS(configure_task_vocabulary=_configure_task_vocabulary)),
        )
        exec(compile(ast.Module(body=[init_node], type_ignores=[]), str(SRC / 'agent/agent.py'), 'exec'), init_ns)
        init_fn = init_ns['__init__']

        def _run_init(nav_cfg):
            call_log.clear()
            dummy = NS()
            dummy.decompose_instruction = lambda instruction: [instruction]

            def _update_map(obs):
                call_log.append(('update_map',))
                raise _StopAtFirstUpdateMap()

            dummy.update_map = _update_map
            sim_wrapper = NS(
                sim=NS(get_agent=lambda idx: NS(get_state=lambda: NS(position=np.zeros(3), rotation=np.eye(3)))),
                step=lambda action: {'agent_state': NS(position=np.zeros(3), rotation=np.eye(3))},
            )
            cfg = {
                'camera': {
                    'height': 64,
                    'width': 64,
                    'fov': 90,
                    'fx': 32.0,
                    'fy': 32.0,
                    'min_depth': 0.1,
                    'max_depth': 10.0,
                },
                'navigation': nav_cfg,
            }
            with self.assertRaises(_StopAtFirstUpdateMap):
                init_fn(dummy, sim_wrapper, cfg, 'Walk past the glass table to the sofa.')

        _run_init({})
        vocab_idx = next(i for i, item in enumerate(call_log) if item[0] == 'configure_task_vocabulary')
        update_idx = next(i for i, item in enumerate(call_log) if item[0] == 'update_map')
        self.assertLess(vocab_idx, update_idx)
        self.assertEqual(
            call_log[vocab_idx],
            ('configure_task_vocabulary', 'Walk past the glass table to the sofa.', False, 16),
        )

        _run_init({'task_visual_enabled': True, 'task_visual_max_extra_terms': 48})
        vocab_call = next(item for item in call_log if item[0] == 'configure_task_vocabulary')
        self.assertEqual(
            vocab_call,
            ('configure_task_vocabulary', 'Walk past the glass table to the sofa.', True, 16),
        )

        _run_init({'task_visual_enabled': True, 'task_visual_max_extra_terms': -4})
        vocab_call_clamped_low = next(item for item in call_log if item[0] == 'configure_task_vocabulary')
        self.assertEqual(
            vocab_call_clamped_low,
            ('configure_task_vocabulary', 'Walk past the glass table to the sofa.', True, 0),
        )

    def test_decide_waypoint_records_pending_landmark_reorientation_override_and_preserves_normal_history(self):
        normal_agent = history_fixture.planner([
            history_fixture.response({'type': 'navigate', 'waypoint_id': 'WP_01'}),
        ])
        res_wp = normal_agent.decide_waypoint()
        selected_wp = res_wp[0] if isinstance(res_wp, tuple) else res_wp
        self.assertEqual(selected_wp['type'], 'waypoint')
        self.assertEqual(selected_wp['action_id'], 'WP_01')
        normal_committed = json.loads(normal_agent.history_msgs[-1]['content'])
        self.assertEqual(normal_committed['action'], {'type': 'navigate', 'waypoint_id': 'WP_01'})
        self.assertNotIn('original_action', normal_committed)
        self.assertNotIn('controller_override', normal_committed)
        self.assertEqual(
            normal_agent.navigation_trace[-1],
            {
                'role': 'planner',
                'event': 'action_selected',
                'action': {'type': 'navigate', 'waypoint_id': 'WP_01'},
                'decision_step': getattr(normal_agent, '_last_step_index', 10),
            },
        )

        override_agent = history_fixture.planner([
            history_fixture.response({'type': 'turn', 'direction': 'RIGHT'}),
        ])
        override_agent._validate_turn_selection = lambda parsed_json, selected: {
            'type': 'turn',
            'direction': 'LEFT',
            'theta': math.pi / 2,
            'requested_action': {
                'type': 'landmark_reorientation',
                'direction': 'LEFT',
                'target_view': 'left',
                'reason': 'restore_landmark_visibility',
            },
            'waypoint_xyz': np.array([1.0, 2.0, 3.0]),
            '_private_handle': object(),
        }
        res_turn = override_agent.decide_waypoint()
        selected_turn = res_turn[0] if isinstance(res_turn, tuple) else res_turn
        self.assertEqual(selected_turn['type'], 'turn')
        self.assertNotEqual(selected_turn['type'], 'completion_proposal')

        override_committed = json.loads(override_agent.history_msgs[-1]['content'])
        expected_pending = {
            'type': 'landmark_reorientation',
            'direction': 'LEFT',
            'target_view': 'left',
            'reason': 'restore_landmark_visibility',
        }
        self.assertEqual(override_committed['action'], expected_pending)
        self.assertEqual(override_committed['selected_pending_action'], expected_pending)
        self.assertEqual(override_committed['original_action'], {'type': 'turn', 'direction': 'RIGHT'})
        self.assertEqual(override_committed['action_status'], 'pending')
        self.assertTrue(override_committed['controller_override'])
        self.assertNotIn('completed', override_committed)
        self.assertNotIn('waypoint_xyz', override_committed)
        self.assertNotIn('waypoint_xyz', override_committed['action'])

        trace_last = override_agent.navigation_trace[-1]
        self.assertEqual(trace_last['event'], 'action_selected')
        self.assertEqual(trace_last['action'], expected_pending)
        self.assertEqual(trace_last['selected_pending_action'], expected_pending)
        self.assertEqual(trace_last['original_action'], {'type': 'turn', 'direction': 'RIGHT'})
        self.assertEqual(trace_last['action_status'], 'pending')
        self.assertTrue(trace_last['controller_override'])


import copy
import importlib.abc
from PIL import Image
import test_v3_boundary_height as boundary_fixture
import verification_bev
import verification_evidence


class _HeavyImportBlocker(importlib.abc.MetaPathFinder):
    BLOCKED = {
        'torch',
        'ultralytics',
        'mapper',
        'agent.agent',
        'segmentation.instance_segmentation',
    }

    def find_spec(self, fullname, path, target=None):
        if fullname in self.BLOCKED or any(fullname.startswith(b + '.') for b in self.BLOCKED):
            raise ImportError(f'Blocked heavy import in unit test: {fullname}')
        return None


class VerificationBatchC1b2IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self._blocker = _HeavyImportBlocker()
        sys.meta_path.insert(0, self._blocker)
        self.addCleanup(lambda: sys.meta_path.remove(self._blocker) if self._blocker in sys.meta_path else None)

    def _bind_verification_inputs(self, agent):
        for attr in dir(VerificationInputs):
            if not attr.startswith('__'):
                val = getattr(VerificationInputs, attr)
                if callable(val):
                    setattr(agent, attr, val.__get__(agent))
        return agent

    def _make_views(self, seed_val):
        return {
            'front': Image.fromarray(np.full((24, 32, 3), seed_val, dtype=np.uint8), mode='RGB'),
            'left': Image.fromarray(np.full((24, 32, 3), (seed_val + 10) % 255, dtype=np.uint8), mode='RGB'),
            'right': Image.fromarray(np.full((24, 32, 3), (seed_val + 20) % 255, dtype=np.uint8), mode='RGB'),
            'back': Image.fromarray(np.full((24, 32, 3), (seed_val + 30) % 255, dtype=np.uint8), mode='RGB'),
        }

    def _make_ledger_row(self, step, ctrl_step, obs_before, obs_after, x0, x1, subtask_id='P1', recovery=False):
        return {
            'step': step,
            'controller_step': ctrl_step,
            'subtask_id': subtask_id,
            'action_type': 'move_forward',
            'executed_action': 'Move Forward 0.25m',
            'requested_distance': 0.25,
            'requested_action': {'type': 'navigate', 'waypoint_id': 'WP_01'},
            'pose_before': [float(x0), 0.0, 0.0],
            'pose': [float(x1), 0.0, 0.0],
            'yaw_before': 0.0,
            'yaw_after': 0.0,
            'yaw_delta': 0.0,
            'motion_distance': abs(float(x1) - float(x0)),
            'height_delta': 0.0,
            'observation_id_before': obs_before,
            'observation_id_after': obs_after,
            'recovery': bool(recovery),
        }

    def test_import_safety_guard_blocks_heavy_modules_and_preserves_sys_path(self):
        bad_dir = str((SRC / 'agent').resolve())
        for p in sys.path:
            self.assertNotEqual(os.path.abspath(p), bad_dir)
        saved_torch = sys.modules.pop('torch', None)
        saved_agent = sys.modules.pop('agent.agent', None)
        try:
            with self.assertRaises(ImportError):
                __import__('torch')
            with self.assertRaises(ImportError):
                __import__('agent.agent')
        finally:
            if saved_torch is not None:
                sys.modules['torch'] = saved_torch
            if saved_agent is not None:
                sys.modules['agent.agent'] = saved_agent

    def test_verifier_single_request_multimodal_assembly_and_15_image_cap(self):
        agent = history_fixture.planner([json.dumps({'output': {'verdict': 'PASS'}})])
        self._bind_verification_inputs(agent)
        os.environ['HSGM_VERIFIER_HISTORY_FRAMES'] = '7'
        agent.navigation_cfg = {
            'verification_evidence_enabled': True,
            'verifier_fixed_bev': True,
            'verifier_use_bev': True,
            'task_visual_enabled': True,
            'verifier_history_frames': 8,
            'verifier_event_max_frames': 9,
            'verifier_max_candidate_crops': 9,
            'verifier_max_output_tokens': 1536,
            'verifier_max_retries': 0,
        }
        agent.mapper = NS(
            current_position=np.array([1.25, -0.5, 0.0], dtype=float),
            current_rotation=np.eye(3, dtype=float),
            observation_id=40,
            initial_floor_height=0.0,
        )
        agent.instruction_obj = NS(get_current_subtask=lambda: ('P1', {'instruction': 'stop at the chair'}))
        agent._route_verification_context = {
            'key': 'P1',
            'checks': [{'id': 'P1.C1', 'text': 'walk forward', 'kind': 'route'}],
        }
        agent._execution_ledger = [
            self._make_ledger_row(11, 90, 31, 32, 0.0, 0.25),
            self._make_ledger_row(12, 91, 33, 34, 0.25, 0.50),
            self._make_ledger_row(13, 92, 35, 36, 0.50, 0.75),
        ]
        agent._verification_event_frames = [
            {'step': 11, 'subtask_id': 'P1', 'phase': 'BEFORE', 'observation_id': 31, 'view': 'front', 'image': self._make_views(10)['front']},
            {'step': 11, 'subtask_id': 'P1', 'phase': 'AFTER', 'observation_id': 32, 'view': 'front', 'image': self._make_views(20)['front']},
            {'step': 12, 'subtask_id': 'P1', 'phase': 'BEFORE', 'observation_id': 33, 'view': 'front', 'image': self._make_views(30)['front']},
            {'step': 12, 'subtask_id': 'P1', 'phase': 'AFTER', 'observation_id': 34, 'view': 'front', 'image': self._make_views(40)['front']},
            {'step': 13, 'subtask_id': 'P1', 'phase': 'BEFORE', 'observation_id': 35, 'view': 'front', 'image': self._make_views(50)['front']},
            {'step': 13, 'subtask_id': 'P1', 'phase': 'AFTER', 'observation_id': 36, 'view': 'front', 'image': self._make_views(60)['front']},
        ]

        def _fake_crops(for_verifier=False):
            self.assertTrue(for_verifier)
            blocks = [{'type': 'text', 'text': 'CANDIDATE OBJECT CROPS FOR VERIFICATION:'}]
            for idx in range(1, 7):
                blocks.append({'type': 'text', 'text': f'[CROP_SENTINEL_{idx}]'})
                blocks.append({
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/jpeg;base64,crop_img_{idx}', 'detail': 'high'},
                })
            return blocks

        agent._visual_candidate_content = _fake_crops
        bev_calls = []
        real_render = verification_bev.render_verification_bev

        def _spy_render(*args, **kwargs):
            bev_calls.append((args, kwargs))
            return real_render(*args, **kwargs)

        front = Image.new('RGB', (32, 24), (10, 20, 30))
        left = Image.new('RGB', (32, 24), (20, 30, 40))
        right = Image.new('RGB', (32, 24), (30, 40, 50))
        back = Image.new('RGB', (32, 24), (40, 50, 60))
        legacy_map = Image.new('RGB', (32, 24), (99, 99, 99))
        hist_imgs = [Image.new('RGB', (32, 24), (i * 15, 0, 0)) for i in range(6)]
        hist_snapshot = copy.deepcopy(agent.history_msgs)

        with patch.object(verification_bev, 'render_verification_bev', side_effect=_spy_render):
            out = agent._get_verifier_response(
                front, left, right, legacy_map, 'VERIFY PROMPT',
                back_rgb_image=back, img_buffer=hist_imgs,
            )

        self.assertIn('PASS', out)
        self.assertEqual(len(agent.client.calls), 1)
        self.assertEqual(agent.runtime_budget.requests, 1)
        self.assertEqual(agent.history_msgs, hist_snapshot)
        self.assertEqual(len(bev_calls), 1)
        call_kwargs = agent.client.calls[0]
        self.assertEqual(call_kwargs['max_tokens'], 1536)
        sys_text = call_kwargs['messages'][0]['content']
        self.assertIn('Fixed-axis metric verifier map', sys_text)
        self.assertNotIn('Heading-up metric map', sys_text)

        user_content = call_kwargs['messages'][1]['content']
        all_texts = [b['text'] for b in user_content if b.get('type') == 'text']
        all_imgs = [b for b in user_content if b.get('type') == 'image_url']
        self.assertFalse(any('TOP-DOWN MAP (you face upward)' in t for t in all_texts))
        self.assertTrue(any('PAST FIRST-PERSON KEYFRAMES (2,' in t for t in all_texts))
        self.assertTrue(any('obs=31' in t or 'BEFORE' in t for t in all_texts))
        self.assertTrue(any('obs=32' in t or 'AFTER' in t for t in all_texts))
        self.assertTrue(any('FIXED' in t.upper() or 'BEV' in t.upper() for t in all_texts))
        self.assertEqual(len(all_imgs), 15)
        self.assertIn('[CROP_SENTINEL_1]', all_texts)
        self.assertIn('[CROP_SENTINEL_4]', all_texts)
        self.assertNotIn('[CROP_SENTINEL_5]', all_texts)
        for idx in range(1, 5):
            cap_pos = user_content.index({'type': 'text', 'text': f'[CROP_SENTINEL_{idx}]'})
            self.assertEqual(
                user_content[cap_pos + 1]['image_url']['url'],
                f'data:image/jpeg;base64,crop_img_{idx}',
            )

    def test_verifier_fixed_bev_failure_produces_no_phantom_map_and_legacy_mode_keeps_old_behavior(self):
        agent = history_fixture.planner([
            json.dumps({'output': {'verdict': 'PASS'}}),
            json.dumps({'output': {'verdict': 'PASS'}}),
        ])
        self._bind_verification_inputs(agent)
        agent.navigation_cfg = {
            'verification_evidence_enabled': False,
            'verifier_fixed_bev': True,
            'verifier_use_bev': True,
        }
        agent.mapper = NS(
            current_position=np.zeros(3),
            current_rotation=np.eye(3),
            observation_id=5,
        )
        front = Image.new('RGB', (32, 24), (10, 20, 30))
        legacy_map = Image.new('RGB', (32, 24), (99, 99, 99))
        with patch.object(verification_bev, 'render_verification_bev', side_effect=RuntimeError('boom')):
            agent._get_verifier_response(front, front, front, legacy_map, 'PROMPT', back_rgb_image=front)
        first_content = agent.client.calls[0]['messages'][1]['content']
        first_texts = [b['text'] for b in first_content if b.get('type') == 'text']
        self.assertFalse(any('TOP-DOWN MAP' in t.upper() for t in first_texts))
        self.assertTrue(any('fixed_bev_render_error' in e for e in agent._last_verification_input_metadata['errors']))

        # Legacy mode with new flags absent/disabled preserves heading-up description and 4 history frames
        agent.navigation_cfg = {'verifier_use_bev': True, 'verifier_history_frames': 4}
        hist_imgs = [Image.new('RGB', (32, 24), (i * 20, 0, 0)) for i in range(4)]
        agent._get_verifier_response(
            front, front, front, legacy_map, 'LEGACY PROMPT',
            back_rgb_image=front, img_buffer=hist_imgs,
        )
        second_call = agent.client.calls[1]
        self.assertIn('Heading-up metric map', second_call['messages'][0]['content'])
        second_texts = [b['text'] for b in second_call['messages'][1]['content'] if b.get('type') == 'text']
        self.assertTrue(any('PAST FIRST-PERSON KEYFRAMES (4,' in t for t in second_texts))
        self.assertTrue(any('TOP-DOWN MAP (you face upward)' in t for t in second_texts))

    def test_planner_multiview_includes_crops_in_request_only_not_history_and_updates_rules(self):
        agent = history_fixture.planner([history_fixture.response()])
        agent.navigation_cfg = {'task_visual_enabled': True, 'planner_history_turns_with_images': 2}
        agent._visual_candidate_content = lambda for_verifier=False: [
            {'type': 'text', 'text': 'PLANNER CANDIDATE CROPS:'},
            {'type': 'text', 'text': '[PLANNER_CROP_1]'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,pcrop_1', 'detail': 'high'}},
            {'type': 'text', 'text': '[PLANNER_CROP_2]'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,pcrop_2', 'detail': 'high'}},
            {'type': 'text', 'text': '[PLANNER_CROP_3_DROPPED]'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,pcrop_3', 'detail': 'high'}},
        ]
        img = Image.new('RGB', (32, 24), (10, 20, 30))
        reply = agent._get_vlm_response_multiview(img, img, img, img, img, 'PLANNER PROMPT', [img])
        agent._commit_planner_history(reply)

        sent_msgs = agent.client.calls[0]['messages']
        sys_prompt = sent_msgs[0]['content']
        self.assertIn('ordinary `navigate` alignment is **not** proof of an explicit instruction turn', sys_prompt)
        self.assertIn('explicitly requests a bounded fresh observation', sys_prompt)

        req_content = sent_msgs[-1]['content']
        req_texts = [b.get('text', '') for b in req_content if b.get('type') == 'text']
        self.assertIn('[TOP-DOWN MAP (you face upward)]', req_texts)
        self.assertIn('[PLANNER_CROP_1]', req_texts)
        self.assertIn('[PLANNER_CROP_2]', req_texts)
        self.assertNotIn('[PLANNER_CROP_3_DROPPED]', req_texts)

        hist_user_msg = agent.history_msgs[1]
        hist_serialized = json.dumps(hist_user_msg)
        self.assertNotIn('PLANNER_CROP', hist_serialized)
        self.assertNotIn('pcrop_', hist_serialized)

    def test_recovery_trace_filter_and_route_prompt_full_ledger_with_diagnostics(self):
        from agent.route_guard import RouteEvidenceGuard
        agent = control.planner([])
        self._bind_verification_inputs(agent)
        agent._visual_candidate_prompt = RouteEvidenceGuard._visual_candidate_prompt.__get__(agent)
        gen_prompt = RouteEvidenceGuard._generate_verification_prompt.__get__(agent)
        agent.navigation_cfg = {
            'verification_evidence_enabled': True,
            'verifier_fixed_bev': True,
        }
        agent.mapper = NS(current_position=np.array([1.0, 0.0, 0.0]), initial_floor_height=0.0)
        ledger = [
            self._make_ledger_row(1, 10, 1, 2, 0.0, 0.25),
            self._make_ledger_row(2, 11, 2, 3, 0.25, 0.50),
            self._make_ledger_row(3, 12, 3, 4, 0.50, 0.25, recovery=True),
        ]
        agent._execution_ledger = ledger
        agent.navigation_trace = [
            {'role': 'planner', 'event': 'physical_recovery_started', 'status': 'started', 'subtask_id': 'P1', 'step': 50},
            {'role': 'verifier', 'event': 'verification_result', 'status': 'started', 'subtask_id': 'P1', 'step': 51},
            {'role': 'controller', 'event': 'endpoint_session_look', 'status': 'started', 'subtask_id': 'P1', 'step': 52},
            {'role': 'controller', 'event': 'physical_recovery_unavailable', 'status': 'unavailable', 'subtask_id': 'P1', 'step': 53},
            {'role': 'controller', 'event': 'physical_recovery_started', 'status': 'completed', 'subtask_id': 'P1', 'step': 54},
            {'role': 'controller', 'event': 'physical_recovery_started', 'status': 'started', 'subtask_id': '', 'step': 55},
            {
                'role': 'controller',
                'event': 'physical_recovery_started',
                'status': 'started',
                'subtask_id': 'P1',
                'step': 77,
                'recovery_source': 'physical',
                'actual_ground_pose': [0.5, 0.0, 0.0],
            },
            {
                'role': 'controller',
                'event': 'wrong_path_rollback_failed',
                'status': 'failed',
                'subtask_id': 'P1',
                'step': 88,
                'actual_ground_pose': [0.25, 0.0, 0.0],
            },
        ]
        ctx = {
            'key': 'P1',
            'curr_instruction': 'walk forward into the hallway',
            'final': False,
            'checks': [{'id': 'P1.C1', 'text': 'walk forward', 'kind': 'route'}],
            'ledger': ledger,
            'entities': [],
        }
        agent._route_verification_context = ctx

        from agent.verification_inputs import _filter_recovery_trace_events
        filtered = _filter_recovery_trace_events(agent.navigation_trace)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]['event'], 'physical_recovery_started')
        self.assertEqual(filtered[0]['recovery_source'], 'physical')
        self.assertEqual(filtered[0]['controller_step'], 77)
        self.assertIsNone(filtered[0]['step'])
        self.assertEqual(filtered[1]['event'], 'wrong_path_rollback_failed')
        self.assertEqual(filtered[1]['recovery_source'], 'unknown')
        self.assertEqual(filtered[1]['source'], 'unknown')
        self.assertEqual(filtered[1]['controller_step'], 88)
        self.assertIsNone(filtered[1]['step'])

        # Even without lifecycle trace, recovery ledger row remains diagnostic
        events_no_trace = verification_evidence.build_execution_events(ledger, recovery_events=[])
        rec_rows = [e for e in events_no_trace if e.get('step') == 3]
        self.assertEqual(len(rec_rows), 1)
        self.assertTrue(rec_rows[0].get('diagnostic_only') or rec_rows[0].get('recovery'))

        prompt = gen_prompt(ctx, 1.5, 'FRONT: center=1.5m')
        self.assertIn('step=1', prompt)
        self.assertIn('step=3', prompt)
        self.assertIn('Fixed-axis verifier BEV context (+X right, +Y up', prompt)
        self.assertIn('Mandatory execution conditions:', prompt)
        self.assertIn('ctrl_step=77 type=recovery status=started recovery_source=physical', prompt)

        # Legacy mode when disabled
        agent.navigation_cfg = {'verification_evidence_enabled': False, 'verifier_fixed_bev': False}
        legacy_prompt = gen_prompt(ctx, 1.5, 'FRONT: center=1.5m')
        self.assertIn('Explored map memory (agent is the red arrow facing upward; trajectory trail is red).', legacy_prompt)

    def test_verify_current_subtask_trace_sanitizes_diagnostics_and_bounds_bev_rows(self):
        harness = boundary_fixture.VerificationChainTests()
        harness.setUp() if hasattr(harness, 'setUp') else None
        payload = {
            'output': {
                'verdict': 'PASS',
                'confidence': 'high',
                'evidence_state': 'supported',
                'evidence': 'At the doorway.',
                'scene_consistent': True,
                'handoff_consistent': True,
                'contract_conflict': False,
                'target_candidate_id': 'RAW_MODEL_LEAK',
                'target_bbox_space': 'raw_untrusted_space',
                'spatial_relation': {'relation': 'hack', 'untrusted': 'raw'},
            }
        }
        a1, _ = harness.run_chain(copy.deepcopy(payload), final=False)
        tr1 = a1.navigation_trace[-1]
        self.assertIsNone(tr1['target_candidate_id'])
        self.assertIsNone(tr1['target_bbox_space'])
        self.assertIsNone(tr1['spatial_relation'])
        self.assertIsNone(tr1['spatial_landmark_grounding'])

        # Now test with controller-verified packets, verifier crops, and >128 BEV rows
        a2 = boundary_fixture.agent()
        del a2._verify_current_subtask
        a2.navigation_cfg['require_endpoint_grounding'] = False
        a2.instruction_obj.is_last_subtask = lambda: True
        a2.instruction_obj.update_verifier = lambda result: a2.st.update(verifier=result)
        checks = boundary_fixture.completion_conditions('Walk into the corridor.', 'P1')
        ledger = [dict(step=1, action_type='move_forward', motion_distance=0.5, pose=[0, 0, -1.2])]
        a2._route_verification_context = dict(
            key='P1', remaining=False, final=True, checks=checks, ledger=ledger, entities=[]
        )
        orig_eval = boundary_fixture.Guard._evaluate_route_verification.__get__(a2)

        def _eval_with_verified_packets(parsed, result):
            res = orig_eval(parsed, result)
            res['endpoint_grounding'] = {
                'grounded': True,
                'status': 'satisfied',
                'candidate_id': 'OBJ_02',
                'target_id': 'chair_2',
                'bbox_space': 'full_view',
                'selected_detector_class_name': 'chair',
                'expected_object_class': 'chair',
                'evidence_role': 'target',
                '_private_array': np.zeros((4, 4), dtype=np.float32),
                'mask': np.ones((8, 8), dtype=np.uint8),
                'pil_image': Image.new('RGB', (4, 4)),
                'nan_metric': float('nan'),
            }
            res['spatial_landmark_grounding'] = {
                'grounded': True,
                'status': 'satisfied',
                'candidate_id': 'OBJ_05',
                'evidence_role': 'landmark',
                '_secret': 'drop_me',
            }
            res['spatial_relation'] = {
                'required': True,
                'relation': 'past',
                'status': 'satisfied',
                'reason': 'Target is past landmark.',
                'frame_kind': 'current',
                'frame_observation_id': 42,
                'target_candidate_id': 'OBJ_02',
                'landmark_candidate_ids': ['OBJ_05', 999],
                'signed_separation_m': 0.65,
                'untrusted_model_key': {'bad': 'drop'},
                '_private': 123,
            }
            return res

        a2._evaluate_route_verification = _eval_with_verified_packets
        a2._get_forward_clearance_and_waypoints = lambda obs, actions: (None, [])
        a2._verification_prompt = lambda *args, **kwargs: 'offline verification input'
        body = copy.deepcopy(payload['output'])
        body['condition_checks'] = [
            dict(id='P1.C1', status='satisfied', evidence='Measured entry.', steps=[1])
        ]
        body['target_identity'] = 'satisfied'
        body['target_proximity'] = 'satisfied'
        body['target_evidence'] = 'At target.'
        a2._get_verifier_response = lambda *args: json.dumps({'output': body})
        a2._last_visual_candidate_crops_for_verifier = [
            {
                'candidate_id': f'OBJ_0{i}',
                'view': 'front',
                'bbox_xyxy': [1, 2, 10, 20],
                'depth_m': 1.5,
                'image': Image.new('RGB', (4, 4)),
                'mask': np.zeros((4, 4)),
                '_private': 'hidden',
            }
            for i in range(1, 7)
        ]
        a2._last_visual_candidate_crops_for_planner = [{'candidate_id': 'PLANNER_ONLY'}]
        a2._last_verification_input_metadata = {
            'evidence_enabled': True,
            'fixed_bev_enabled': True,
            'event_frames': [{'step': 1, 'phase': 'after', 'observation_id': 2, 'view': 'front'}],
            'bev': {
                'image_size': [520, 520],
                'axes': {
                    'x_positive': 'right',
                    'y_positive': 'up',
                    'pixel_y_positive': 'down',
                    'yaw_convention': 'ccw_from_plus_x_rad',
                },
                'current_pose_valid': True,
                'current_yaw_valid': True,
                'current_yaw': 0.25,
                'current_foot_pose': [1.0, 2.0, 0.0],
                'counts': {
                    'total_ledger_rows': 140,
                    'retained_rows': 128,
                    'omitted_history_rows': 12,
                    'omitted_steps': list(range(12)),
                },
                'rendered_rows': [{'step': i} for i in range(140)],
                'row_styles': [{'step': i} for i in range(140)],
                'turn_events': [{'step': 1}],
                'map_labels': [{'text': 'START'}],
                'z_profile': {'unit': 'm', 'min_z': -1.2, 'max_z': -1.0, 'current_z': -1.1, 'segments': [1, 2, 3]},
                'legend': {'current_subtask': '#f00', 'start': '#0f0'},
            },
            'errors': [],
        }

        a2._verify_current_subtask({'action': {'type': 'propose_completion'}}, None, [], None, actions=boundary_fixture.ACTIONS)
        tr2 = a2.navigation_trace[-1]
        serialized = json.dumps(tr2, allow_nan=False)
        self.assertIsInstance(serialized, str)
        self.assertEqual(tr2['target_candidate_id'], 'OBJ_02')
        self.assertEqual(tr2['target_bbox_space'], 'full_view')
        self.assertNotIn('_private_array', tr2['endpoint_grounding'])
        self.assertNotIn('mask', tr2['endpoint_grounding'])
        self.assertNotIn('pil_image', tr2['endpoint_grounding'])
        self.assertIsNone(tr2['endpoint_grounding']['nan_metric'])
        self.assertEqual(tr2['spatial_landmark_grounding']['candidate_id'], 'OBJ_05')
        self.assertNotIn('_secret', tr2['spatial_landmark_grounding'])
        self.assertEqual(
            set(tr2['spatial_relation'].keys()),
            {
                'required', 'relation', 'status', 'reason', 'frame_kind',
                'frame_observation_id', 'target_candidate_id', 'landmark_candidate_ids', 'signed_separation_m'
            },
        )
        self.assertEqual(tr2['spatial_relation']['landmark_candidate_ids'], ['OBJ_05'])
        self.assertEqual(len(tr2['visual_candidate_crops']), 4)
        self.assertNotIn('PLANNER_ONLY', serialized)
        bev_meta = tr2['verification_input_metadata']['bev']
        self.assertEqual(bev_meta['current_yaw'], 0.25)
        self.assertEqual(bev_meta['counts']['retained_rows'], 128)
        self.assertNotIn('omitted_steps', bev_meta['counts'])
        self.assertNotIn('rendered_rows', bev_meta)
        self.assertNotIn('row_styles', bev_meta)
        self.assertNotIn('turn_events', bev_meta)
        self.assertNotIn('map_labels', bev_meta)
        self.assertNotIn('segments', bev_meta['z_profile'])


if __name__ == '__main__':
    unittest.main()


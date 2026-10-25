"""Tests for visual turn landmark grounding, dual-pose confirmation, and incoming-heading safety."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from agent.endpoint_guard import EndpointGroundingGuard
from agent.turn_execution import TurnExecutionGuard
from agent.visual_grounding import VisualGroundingMixin
from navigation_contracts import completion_conditions
from navigation_control import ActionProtocolError, decode_action
from route_progress import instruction_turn_witness, turn_witness
import test_v3_navigation_control as planner_fixture


TEXT = 'Turn right at the eye chart and stand next to the bathroom door.'


def base_rot(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[-s, 0.0, -c], [c, 0.0, -s], [0.0, -1.0, 0.0]], dtype=float)


def sensor_rot(yaw=0.0, view='front'):
    offsets = {'front': 0.0, 'right': math.pi / 2, 'back': math.pi, 'left': -math.pi / 2}
    return base_rot(float(yaw) + offsets.get(view, 0.0))


def make_obs(depth_val=1.0, pos=(0.0, 0.0, 0.0), cam_yaw=0.0):
    depth = np.full((100, 100), depth_val, dtype=np.float32)
    p = np.asarray(pos, dtype=float)
    views = {}
    for name in ('front', 'left', 'right', 'back'):
        r = sensor_rot(cam_yaw, name)
        state = NS(position=p.copy(), sensor_states={'color_sensor': NS(rotation=r)})
        views[name] = {'depth_sensor': depth.copy(), 'agent_state': state}
    obs = dict(views['front'])
    for name in ('left', 'right', 'back'):
        obs[name] = views[name]
    return obs


def make_guard(visual_enabled=True, instruction=TEXT, yaw=0.0, pos=(0.0, 0.0, 0.0)):
    a = NS()
    a.navigation_cfg = {'task_visual_enabled': True, 'endpoint_mask_min_confidence': 0.48} if visual_enabled else {}
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:60, 30:50] = True
    cid = '1:right:0'
    cand = {
        'candidate_id': cid,
        'mask_key': cid,
        'detector_class_name': 'eye_chart',
        'canonical_label': 'eye_chart',
        'view': 'right',
        'observation_id': 1,
        'image_size': [100, 100],
        'bbox': [0.3, 0.2, 0.5, 0.6],
        'confidence': 0.82,
        'entity_id': None,
    }
    a.mapper = NS(
        observation_id=1,
        observation_step=1,
        current_position=np.asarray(pos, dtype=float),
        current_rotation=base_rot(yaw),
        camera_intrinsic=np.array([[60.0, 0.0, 50.0], [0.0, 60.0, 50.0], [0.0, 0.0, 1.0]]),
        rotation_func=lambda r: np.asarray(r, dtype=float),
        translation_func=lambda p: np.asarray(p, dtype=float),
        initial_position=np.zeros(3),
        current_visual_candidates=[cand],
        current_observation_masks={cid: mask.copy()},
        get_entities=lambda: {},
        entities={},
    )
    a.curr_obs = make_obs(1.0, pos=pos, cam_yaw=yaw)
    for cls in (VisualGroundingMixin, EndpointGroundingGuard, TurnExecutionGuard):
        for name, method in cls.__dict__.items():
            if name.startswith('_') and callable(method):
                setattr(a, name, method.__get__(a))
    a.st = {'instruction': instruction, 'record': []}
    a.instruction_obj = NS(
        get_current_subtask=lambda: ('P1', a.st),
        get_current_subtask_key=lambda: 'P1',
    )
    a._execution_ledger = []
    a.navigation_trace = []
    a.rolling_back = False
    a._invalidate_blocked_path = lambda reason=None: a.navigation_trace.append({
        'role': 'controller',
        'event': 'wrong_path_rollback_triggered',
        'reason': reason,
    })

    def _set_pose_and_candidates(obs_id, position, base_yaw, views=('right',), label='eye_chart',
                                confidence=0.82, entity_id=None, depth_val=1.0):
        p = np.asarray(position, dtype=float)
        a.mapper.observation_id = int(obs_id)
        a.mapper.observation_step = int(obs_id)
        a.mapper.current_position = p.copy()
        a.mapper.current_rotation = base_rot(base_yaw)
        a.curr_obs = make_obs(depth_val, pos=p, cam_yaw=base_yaw)
        cands = []
        masks = {}
        for v in views:
            c_id = f'{int(obs_id)}:{v}:0'
            cands.append({
                'candidate_id': c_id,
                'mask_key': c_id,
                'detector_class_name': label,
                'canonical_label': label,
                'view': v,
                'observation_id': int(obs_id),
                'image_size': [100, 100],
                'bbox': [0.3, 0.2, 0.5, 0.6],
                'confidence': confidence,
                'entity_id': entity_id,
            })
            masks[c_id] = mask.copy()
        a.mapper.current_visual_candidates = cands
        a.mapper.current_observation_masks = masks

    a.set_pose_and_candidates = _set_pose_and_candidates
    return a


def turn_ctx(status='at_landmark', **kwargs):
    base = {
        'condition_id': 'P1.C1',
        'status': status,
        'landmark_identity': 'eye chart',
        'landmark_view': 'right',
        'landmark_bbox': [0.3, 0.2, 0.5, 0.6],
        'landmark_bbox_space': 'full_view',
        'landmark_candidate_id': '1:right:0',
    }
    base.update(kwargs)
    return base


def act_turn(direction='RIGHT'):
    return decode_action({'type': 'turn', 'direction': direction}, planner_fixture.ACTIONS)


class VisualTurnGroundingTests(unittest.TestCase):
    def test_prompt_and_candidate_measurement_contracts(self):
        a = make_guard(yaw=0.0, pos=(0.0, 0.0, 0.0))
        prompt = a._turn_execution_prompt()
        self.assertIn('landmark_candidate_id', prompt)
        self.assertIn('landmark_bbox_space', prompt)
        self.assertIn('second_changed_base_pose_needed', prompt)

        bad_space = a._measure_turn_landmark(turn_ctx(landmark_bbox_space=None, landmark_candidate_id=None))
        self.assertFalse(bad_space['valid'])

        # Stale _verification_obs depth=9.0 vs current curr_obs depth=1.0 must measure 1.0 using real resolver + real ground_target_depth.
        a._verification_obs = make_obs(9.0, pos=(0.0, 0.0, 0.0), cam_yaw=0.0)
        pkt = a._measure_turn_landmark(turn_ctx(landmark_bbox=None, landmark_candidate_id='1:right:0'))
        self.assertTrue(pkt['valid'])
        self.assertEqual(pkt['candidate_id'], '1:right:0')
        self.assertIsNone(pkt['entity_id'])
        self.assertEqual(pkt['source'], 'detector_mask')
        self.assertEqual(pkt['binding_source'], 'candidate')
        self.assertEqual(pkt['selected_detector_class_name'], 'eye_chart')
        self.assertAlmostEqual(pkt['depth_m'], 1.0, places=3)
        self.assertAlmostEqual(pkt['median_distance_m'], 1.0, delta=0.05)
        self.assertNotIn('mask', pkt)

        # Wrong detector class (door vs eye_chart), stale candidate, or wrong view must fail with NO fallback to free bbox.
        a.set_pose_and_candidates(1, (0.0, 0.0, 0.0), 0.0, views=('right',), label='door')
        wrong_cls = a._measure_turn_landmark(turn_ctx(landmark_candidate_id='1:right:0', landmark_bbox=[0.3, 0.2, 0.5, 0.6]))
        self.assertFalse(wrong_cls['valid'])

        a.set_pose_and_candidates(1, (0.0, 0.0, 0.0), 0.0, views=('right',), label='eye_chart')
        a.mapper.observation_id = 2
        stale = a._measure_turn_landmark(turn_ctx(landmark_candidate_id='1:right:0', landmark_bbox=[0.3, 0.2, 0.5, 0.6]))
        self.assertFalse(stale['valid'])

        a.set_pose_and_candidates(2, (0.0, 0.0, 0.0), 0.0, views=('front',), label='eye_chart')
        wrong_view = a._measure_turn_landmark(turn_ctx(landmark_view='right', landmark_candidate_id='2:front:0', landmark_bbox=[0.3, 0.2, 0.5, 0.6]))
        self.assertFalse(wrong_view['valid'])

        # Low confidence, boolean confidence, non-finite confidence, and invalid depth must reject.
        for bad_conf in (0.30, True, float('nan')):
            a.set_pose_and_candidates(2, (0.0, 0.0, 0.0), 0.0, views=('right',), label='eye_chart', confidence=bad_conf)
            bad_pkt = a._measure_turn_landmark(turn_ctx(landmark_candidate_id='2:right:0'))
            self.assertFalse(bad_pkt['valid'])

        a.set_pose_and_candidates(2, (0.0, 0.0, 0.0), 0.0, views=('right',), label='eye_chart', depth_val=0.0)
        bad_depth = a._measure_turn_landmark(turn_ctx(landmark_candidate_id='2:right:0'))
        self.assertFalse(bad_depth['valid'])

        # Feature flag disabled preserves legacy bbox measurement.
        legacy = make_guard(visual_enabled=False, yaw=0.0)
        legacy_pkt = legacy._measure_turn_landmark({
            'condition_id': 'P1.C1',
            'status': 'at_landmark',
            'landmark_identity': 'eye chart',
            'landmark_view': 'right',
            'landmark_bbox': [0.3, 0.2, 0.5, 0.6],
        })
        self.assertTrue(legacy_pkt['valid'])

    def test_requires_two_changed_base_pose_observations_and_preserves_seed(self):
        a = make_guard(yaw=0.0, pos=(0.0, 0.0, 0.0))
        # First at_landmark saves seed before rejecting.
        with self.assertRaisesRegex(ActionProtocolError, 'two distinct changed-pose'):
            a._validate_turn_selection({'turn_context': turn_ctx('at_landmark')}, act_turn('RIGHT'))
        entry = next(iter(a._visual_turn_state.values()))
        self.assertEqual(len(entry['samples']), 1)

        # Same-obs retry cannot confirm.
        with self.assertRaisesRegex(ActionProtocolError, 'two distinct changed-pose'):
            a._validate_turn_selection({'turn_context': turn_ctx('at_landmark')}, act_turn('RIGHT'))

        # New obs_id with unchanged base pose and switched view must NOT confirm.
        a.set_pose_and_candidates(2, (0.0, 0.0, 0.0), 0.0, views=('front', 'right'))
        with self.assertRaisesRegex(ActionProtocolError, 'two distinct changed-pose'):
            a._validate_turn_selection(
                {'turn_context': turn_ctx('at_landmark', landmark_view='front', landmark_candidate_id='2:front:0')},
                act_turn('RIGHT'),
            )

        # 90-deg body look in place switching right->front has identical world camera pose and alone cannot confirm.
        a.set_pose_and_candidates(2, (0.0, 0.0, 0.0), math.pi / 2, views=('front',))
        with self.assertRaisesRegex(ActionProtocolError, 'two distinct changed-pose'):
            a._validate_turn_selection(
                {'turn_context': turn_ctx('at_landmark', landmark_view='front', landmark_candidate_id='2:front:0')},
                act_turn('RIGHT'),
            )

        # Observe status with invalid/unconfirmed localization never raises or attaches instruction_turn.
        res_obs = a._validate_turn_selection(
            {'turn_context': turn_ctx('observe', landmark_candidate_id='2:right:99')},
            act_turn('RIGHT'),
        )
        self.assertNotIn('instruction_turn', res_obs)

        # Move base forward >= 0.10m on obs 3 with consistent world center.
        a.set_pose_and_candidates(3, (0.20, 0.0, 0.0), 0.0, views=('right',))
        armed = a._validate_turn_selection(
            {'turn_context': turn_ctx('at_landmark', landmark_candidate_id='3:right:0')},
            act_turn('RIGHT'),
        )
        self.assertIn('instruction_turn', armed)
        self.assertIs(armed['instruction_turn']['landmark_evidence']['confirmed_changed_view'], True)
        self.assertEqual(armed['instruction_turn']['landmark_evidence']['confirmation_observation_ids'], [1, 3])

    def test_incoming_heading_restore_before_owned_turn_and_bound_enforcement(self):
        a = make_guard(yaw=0.0, pos=(0.0, 0.0, 0.0))
        # Seed observation 1 in observe status at pos (0, 0, 0), yaw 0.
        a._validate_turn_selection(
            {'turn_context': turn_ctx('observe', landmark_candidate_id='1:right:0')},
            act_turn('RIGHT'),
        )

        # Approach forward >= 0.10m to pos (0.20, 0, 0) at heading 0 on obs 2 -> confirms dual pose!
        a._execution_ledger.append({
            'step': 1, 'subtask_id': 'P1', 'action_type': 'move_forward',
            'pose_before': [0.0, 0.0, 0.0], 'pose': [0.20, 0.0, 0.0],
            'motion_distance': 0.20, 'recovery': False, 'rejected_by_rollback': False,
        })
        a.set_pose_and_candidates(2, (0.20, 0.0, 0.0), 0.0, views=('right',))
        a._validate_turn_selection(
            {'turn_context': turn_ctx('approaching', landmark_candidate_id='2:right:0')},
            act_turn('RIGHT'),
        )

        # Execute a 90-deg RIGHT observation look in place at (0.20, 0, 0) -> actual base yaw is +pi/2.
        # Now front view at yaw=+pi/2 has the exact same world camera pose as right view had at yaw=0.
        a._execution_ledger.append({
            'step': 2, 'subtask_id': 'P1', 'action_type': 'turn',
            'yaw_before': 0.0, 'yaw_after': math.pi / 2, 'yaw_delta': math.pi / 2,
            'requested_action': {'type': 'turn', 'direction': 'RIGHT'},
            'pose_before': [0.20, 0.0, 0.0], 'pose': [0.20, 0.0, 0.0],
            'motion_distance': 0.0, 'recovery': False, 'rejected_by_rollback': False,
        })
        a.set_pose_and_candidates(3, (0.20, 0.0, 0.0), math.pi / 2, views=('front',))

        # Confirmed chart, but base yaw (+pi/2) drifts 90deg from incoming (0): returns restore turn (+pi/2).
        decision = a._validate_turn_selection(
            {'turn_context': turn_ctx('at_landmark', landmark_view='front', landmark_candidate_id='3:front:0')},
            act_turn('RIGHT'),
        )
        self.assertEqual(decision['type'], 'turn')
        self.assertAlmostEqual(decision['angle'], math.pi / 2, places=5)
        self.assertNotIn('instruction_turn', decision)
        self.assertEqual(decision['requested_action']['type'], 'landmark_reorientation')
        self.assertEqual(decision['requested_action']['purpose'], 'restore_incoming_heading')

        # Execute restore turn in ledger (real yaw delta -pi/2 back to 0); must not count as owned turn.
        restore_row = {
            'step': 3, 'subtask_id': 'P1', 'action_type': 'turn',
            'yaw_before': math.pi / 2, 'yaw_after': 0.0, 'yaw_delta': -math.pi / 2,
            'requested_action': decision['requested_action'],
            'pose_before': [0.20, 0.0, 0.0], 'pose': [0.20, 0.0, 0.0],
            'motion_distance': 0.0, 'recovery': False, 'rejected_by_rollback': False,
        }
        a._instruction_turn_command = decision.get('instruction_turn')
        a._execution_ledger.append(restore_row)
        a.st['record'].append(restore_row)
        a._observe_turn_execution(restore_row)
        self.assertIsNone(a._executed_instruction_turn(a._active_turn_clause()))
        cond = completion_conditions(TEXT, 'P1')[0]
        self.assertFalse(instruction_turn_witness(cond, a.st['record'], a._execution_ledger, 'P1'))

        # Next named RIGHT from restored yaw 0 returns owned explicit turn with expected heading +pi/2.
        a.set_pose_and_candidates(4, (0.20, 0.0, 0.0), 0.0, views=('right',))
        owned = a._validate_turn_selection(
            {'turn_context': turn_ctx('at_landmark', landmark_view='right', landmark_candidate_id='4:right:0')},
            act_turn('RIGHT'),
        )
        self.assertIn('instruction_turn', owned)
        owned_row = {
            'step': 4, 'subtask_id': 'P1', 'action_type': 'turn',
            'yaw_before': 0.0, 'yaw_after': math.pi / 2, 'yaw_delta': math.pi / 2,
            'requested_action': {'type': 'turn', 'direction': 'RIGHT'},
            'pose_before': [0.20, 0.0, 0.0], 'pose': [0.20, 0.0, 0.0],
            'motion_distance': 0.0, 'recovery': False, 'rejected_by_rollback': False,
        }
        a._instruction_turn_command = owned['instruction_turn']
        a._execution_ledger.append(owned_row)
        a.st['record'].append(owned_row)
        a._observe_turn_execution(owned_row)
        self.assertAlmostEqual(a._current_turn_departure()['heading'], math.pi / 2, places=5)
        self.assertTrue(turn_witness(cond['text'], [owned_row], a._execution_ledger)[0])
        self.assertTrue(instruction_turn_witness(cond, a.st['record'], a._execution_ledger, 'P1'))

    def test_zero_yaw_restore_bound_and_unknown_heading_and_rollback_invalidation(self):
        a = make_guard(yaw=0.0, pos=(0.0, 0.0, 0.0))
        a._validate_turn_selection(
            {'turn_context': turn_ctx('observe', landmark_candidate_id='1:right:0')},
            act_turn('RIGHT'),
        )
        a._execution_ledger.append({
            'step': 1, 'subtask_id': 'P1', 'action_type': 'move_forward',
            'pose_before': [0.0, 0.0, 0.0], 'pose': [0.20, 0.0, 0.0],
            'motion_distance': 0.20, 'recovery': False, 'rejected_by_rollback': False,
        })
        a.set_pose_and_candidates(2, (0.20, 0.0, 0.0), 0.0, views=('right',))
        a._validate_turn_selection(
            {'turn_context': turn_ctx('approaching', landmark_candidate_id='2:right:0')},
            act_turn('RIGHT'),
        )
        a.set_pose_and_candidates(3, (0.20, 0.0, 0.0), math.pi / 2, views=('front',))
        for step in (2, 3):
            d = a._validate_turn_selection(
                {'turn_context': turn_ctx('at_landmark', landmark_view='front', landmark_candidate_id='3:front:0')},
                act_turn('RIGHT'),
            )
            self.assertNotIn('instruction_turn', d)
            a._execution_ledger.append({
                'step': step, 'subtask_id': 'P1', 'action_type': 'turn',
                'yaw_before': math.pi / 2, 'yaw_after': math.pi / 2, 'yaw_delta': 0.0,
                'requested_action': d['requested_action'],
                'pose_before': [0.20, 0.0, 0.0], 'pose': [0.20, 0.0, 0.0],
                'motion_distance': 0.0, 'recovery': False, 'rejected_by_rollback': False,
            })
        with self.assertRaisesRegex(ActionProtocolError, 'restore bound exceeded'):
            a._validate_turn_selection(
                {'turn_context': turn_ctx('at_landmark', landmark_view='front', landmark_candidate_id='3:front:0')},
                act_turn('RIGHT'),
            )

        # Rollback trace event invalidates cached confirmation and reorientation state.
        a._invalidate_blocked_path('wrong_branch')
        a.set_pose_and_candidates(4, (0.20, 0.0, 0.0), 0.0, views=('right',))
        with self.assertRaisesRegex(ActionProtocolError, 'two distinct changed-pose'):
            a._validate_turn_selection(
                {'turn_context': turn_ctx('at_landmark', landmark_view='right', landmark_candidate_id='4:right:0')},
                act_turn('RIGHT'),
            )

        # Undefined horizontal forward axis must never guess heading 0.
        b = make_guard(yaw=0.0, pos=(0.0, 0.0, 0.0))
        b.mapper.current_rotation = np.eye(3)
        b._validate_turn_selection(
            {'turn_context': turn_ctx('observe', landmark_candidate_id='1:right:0')},
            act_turn('RIGHT'),
        )
        b.set_pose_and_candidates(2, (0.20, 0.0, 0.0), 0.0, views=('right',))
        b.mapper.current_rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        with self.assertRaisesRegex(ActionProtocolError, 'incoming heading'):
            b._validate_turn_selection(
                {'turn_context': turn_ctx('at_landmark', landmark_candidate_id='2:right:0')},
                act_turn('RIGHT'),
            )


if __name__ == '__main__':
    unittest.main()


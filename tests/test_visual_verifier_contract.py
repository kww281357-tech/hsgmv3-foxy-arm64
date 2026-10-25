import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from agent.route_guard import RouteEvidenceGuard
from navigation_contracts import completion_conditions, verification_schema
from navigation_policy import normalize_verification

CALIB_R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=float)


def _make_state(pos=(0.0, 0.0, 0.0), rot=CALIB_R):
    sensor = NS(position=np.array(pos, dtype=float), rotation=np.array(rot, dtype=float))
    return NS(
        position=np.array(pos, dtype=float),
        rotation=np.array(rot, dtype=float),
        sensor_states={'color_sensor': sensor, 'depth_sensor': sensor},
    )


class Guard(RouteEvidenceGuard):
    def __init__(self, instruction='Walk toward the chair. Stop beside the chair to the left of the plant.'):
        self.instruction = instruction
        self.st = {'instruction': instruction, 'record': [], 'completed': False}
        self.instruction_obj = NS(
            get_current_subtask=lambda: ('P1', self.st),
            get_current_subtask_key=lambda: 'P1',
            get_subtask_by_key=lambda key: self.st,
            is_last_subtask=lambda: True,
            sub_instruction_keys=['P1'],
            current_subtask_index=0,
        )
        self.mapper = NS(
            current_position=np.array([0.0, 0.0, 0.0], dtype=float),
            current_rotation=CALIB_R.copy(),
            camera_intrinsic=np.array([[100.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]], dtype=float),
            initial_position=np.array([0.0, 0.0, 0.0], dtype=float),
            initial_floor_height=0.0,
            trajectory_position=[np.array([0.0, -0.5, 0.0]), np.array([0.0, 0.0, 0.0])],
            floor_level=0,
            observation_id=5,
            observation_step=3,
            translation_func=lambda p: np.asarray(p, dtype=float),
            rotation_func=lambda r: np.asarray(r, dtype=float),
            current_visual_candidates={},
            current_observation_masks={},
            object_entities=[],
        )
        depth = np.ones((200, 200), dtype=np.float32) * 1.1
        rgb = np.zeros((200, 200, 3), dtype=np.uint8)
        self.curr_obs = {
            'color_sensor': rgb,
            'depth_sensor': depth,
            'agent_state': _make_state(),
        }
        self.navigation_cfg = {
            'require_endpoint_grounding': False,
            'hfov': 90.0,
        }
        self.navigation_trace = []
        self.subtask_execution_ledger = {}
        self.traveled_distance = 0.5
        self.entities = []
        self.stair_ready = True
        self.stair_packet = {'direction': 'NONE', 'measured_height_change_m': 0.0}
        self._observation_revision = 5
        self._normalize_angle = lambda x: (x + math.pi) % (2 * math.pi) - math.pi

    def _observed_target_candidates(self, text, limit=10):
        return self.entities[:limit]

    def _ensure_stair_progress(self):
        return NS(packet=lambda: self.stair_packet, completion_ready=lambda: self.stair_ready)

    def prepare(self):
        self.st['record'] = [{
            'step': 1, 'executed_action': 'move_forward', 'action_type': 'move_forward',
            'motion_distance': 0.5, 'yaw_delta': 0.0, 'height_delta': 0.0,
            'pose_before': [0.0, -0.5, 0.0], 'pose': [0.0, 0.0, 0.0],
        }]
        return self._prepare_route_verification({})

    def evaluate(self, raw):
        return self._evaluate_route_verification(raw, normalize_verification(copy.deepcopy(raw)))


def _add_candidate(guard, cid, label, bbox, view='front', obs_id=5, conf=0.90, entity_id=None, depth_m=1.1):
    h, w = 200, 200
    x1, y1, x2, y2 = [int(round(v * 200)) for v in bbox]
    mask = np.zeros((h, w), dtype=bool)
    mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
    view_obs = guard.curr_obs if view == 'front' else guard.curr_obs.setdefault(view, {
        'color_sensor': np.zeros((h, w, 3), dtype=np.uint8),
        'depth_sensor': np.ones((h, w), dtype=np.float32) * depth_m,
        'agent_state': _make_state(),
    })
    view_obs['depth_sensor'][mask] = float(depth_m)
    guard.mapper.current_visual_candidates[cid] = {
        'candidate_id': cid,
        'mask_key': cid,
        'observation_id': obs_id,
        'step': guard.mapper.observation_step,
        'view': view,
        'canonical_label': label,
        'detector_class_name': label,
        'confidence': conf,
        'bbox': list(bbox),
        'image_size': [w, h],
        'entity_id': entity_id,
    }
    guard.mapper.current_observation_masks[cid] = mask


def _verdict(context):
    return {
        'outcome': True,
        'relation_satisfied': True,
        'scene_consistent': True,
        'confidence': 'high',
        'evidence': 'Current views establish the route and endpoint.',
        'reason': 'The requested endpoint is reached.',
        'failure_type': 'none',
        'target_identity': 'Chair to the left of the plant',
        'target_proximity': 'satisfied',
        'target_entity_id': None,
        'target_view': 'front',
        'target_bbox': [0.10, 0.35, 0.30, 0.75],
        'target_bbox_space': 'full_view',
        'target_candidate_id': '5:front:0',
        'target_evidence': 'Chair is directly ahead on the left side of the plant.',
        'condition_checks': [
            {'id': c['id'], 'status': 'satisfied', 'evidence': 'Observed and executed.', 'steps': [1], 'observation_id': 5}
            for c in context['checks']
        ],
    }


class TestVisualVerifierContract(unittest.TestCase):
    def test_verification_schema_backward_compatible_and_strict_candidate_enum(self):
        conds = completion_conditions('Walk to the chair and stop.')
        legacy = verification_schema(conds, [3, 7])
        r_props = legacy['schema']['properties']['reasoning']['properties']
        r_req = set(legacy['schema']['properties']['reasoning']['required'])
        self.assertIn('target_candidate_id', r_req)
        self.assertIn('target_bbox_space', r_req)
        self.assertIn('reference_region', r_req)
        self.assertIn('spatial_landmark_region', r_req)
        self.assertEqual(r_props['target_candidate_id'], {'type': ['string', 'null']})
        self.assertEqual(r_props['target_bbox_space'], {'type': ['string', 'null'], 'enum': ['full_view', None]})

        with_cids = verification_schema(
            conds, [3],
            candidate_ids=['5:front:0', '5:left:1', '5:front:0', True, False, 42, '', '   ', None],
        )
        props = with_cids['schema']['properties']['reasoning']['properties']
        self.assertEqual(props['target_candidate_id']['enum'], [None, '5:front:0', '5:left:1'])
        ref_region = props['reference_region']
        lm_region = props['spatial_landmark_region']
        self.assertFalse(ref_region['additionalProperties'])
        self.assertFalse(lm_region['additionalProperties'])
        self.assertEqual(
            set(ref_region['required']),
            {'view', 'bbox', 'bbox_space', 'entity_id', 'candidate_id', 'role'},
        )
        self.assertEqual(
            set(lm_region['required']),
            {'view', 'bbox', 'bbox_space', 'entity_id', 'candidate_id', 'role'},
        )
        self.assertEqual(lm_region['properties']['role']['enum'], ['spatial_landmark'])
        self.assertEqual(lm_region['properties']['candidate_id']['enum'], [None, '5:front:0', '5:left:1'])

    def test_normalizer_does_not_pass_through_unvalidated_visual_fields(self):
        norm = normalize_verification({
            'outcome': True,
            'relation_satisfied': True,
            'scene_consistent': True,
            'reason': 'ok',
            'target_candidate_id': 'forged:front:9',
            'target_bbox_space': 'crop',
            'spatial_landmark_region': {'role': 'spatial_landmark', 'view': 'front', 'bbox': [0.1, 0.1, 0.2, 0.2]},
        })
        self.assertNotIn('target_candidate_id', norm)
        self.assertNotIn('target_bbox_space', norm)
        self.assertNotIn('spatial_landmark_region', norm)

    def test_prompt_includes_independent_spatial_landmark_and_gates_candidate_helper_by_flag(self):
        guard = Guard()
        ctx = guard.prepare()
        _add_candidate(guard, '5:front:0', 'chair', [0.10, 0.35, 0.30, 0.75])
        prompt_off = guard._generate_verification_prompt(ctx, 1.1)
        self.assertIn('"target_candidate_id"', prompt_off)
        self.assertIn('"target_bbox_space"', prompt_off)
        self.assertIn('"spatial_landmark_region"', prompt_off)
        self.assertNotIn('5:front:0', prompt_off)

        guard.navigation_cfg['task_visual_enabled'] = True
        ctx_on = guard.prepare()
        prompt_on = guard._generate_verification_prompt(ctx_on, 1.1)
        self.assertIn('5:front:0', prompt_on)
        self.assertIn('full_view', prompt_on)
        self.assertIsNotNone(ctx_on)

    def test_route_verification_flag_off_vs_on_spatial_and_depth_precedence(self):
        guard_legacy = Guard()
        ctx_legacy = guard_legacy.prepare()
        res_legacy = guard_legacy.evaluate(_verdict(ctx_legacy))
        self.assertTrue(res_legacy['outcome'])

        guard = Guard()
        guard.navigation_cfg['task_visual_enabled'] = True
        ctx = guard.prepare()
        _add_candidate(guard, '5:front:0', 'chair', [0.10, 0.35, 0.30, 0.75])
        _add_candidate(guard, '5:front:1', 'plant', [0.70, 0.35, 0.90, 0.75])

        raw_missing_lm = _verdict(ctx)
        res_missing = guard.evaluate(raw_missing_lm)
        self.assertFalse(res_missing['outcome'])
        self.assertFalse(res_missing['relation_satisfied'])
        self.assertEqual(res_missing['failure_type'], 'not_reached')
        self.assertFalse(res_missing['rollback_required'])
        self.assertEqual(res_missing['target_candidate_id'], '5:front:0')
        self.assertEqual(res_missing['target_bbox_space'], 'full_view')
        self.assertEqual(res_missing['spatial_relation']['status'], 'unknown')

        raw_bad_cid = _verdict(ctx)
        raw_bad_cid['target_candidate_id'] = '5:front:99'
        res_bad_cid = guard.evaluate(raw_bad_cid)
        self.assertFalse(res_bad_cid['outcome'])
        self.assertIsNone(res_bad_cid['target_candidate_id'])
        self.assertIsNone(res_bad_cid['target_bbox_space'])
        self.assertFalse(res_bad_cid['endpoint_grounding']['valid'])

        guard_far = Guard()
        guard_far.navigation_cfg['task_visual_enabled'] = True
        ctx_far = guard_far.prepare()
        _add_candidate(guard_far, '5:front:0', 'chair', [0.10, 0.35, 0.30, 0.75], depth_m=4.5)
        _add_candidate(guard_far, '5:front:1', 'plant', [0.70, 0.35, 0.90, 0.75], depth_m=4.5)
        raw_far = _verdict(ctx_far)
        raw_far['spatial_landmark_region'] = {
            'role': 'spatial_landmark', 'view': 'front', 'bbox': [0.70, 0.35, 0.90, 0.75],
            'bbox_space': 'full_view', 'candidate_id': '5:front:1', 'entity_id': None,
        }
        res_far = guard_far.evaluate(raw_far)
        self.assertFalse(res_far['outcome'])
        self.assertEqual(res_far['spatial_relation']['status'], 'satisfied')
        self.assertEqual(res_far['spatial_relation']['frame_observation_id'], 5)
        self.assertIn('far', res_far['reason'].lower())

        raw_ok = _verdict(ctx)
        raw_ok['spatial_landmark_region'] = {
            'role': 'spatial_landmark', 'view': 'front', 'bbox': [0.70, 0.35, 0.90, 0.75],
            'bbox_space': 'full_view', 'candidate_id': '5:front:1', 'entity_id': None,
        }
        res_ok = guard.evaluate(raw_ok)
        self.assertTrue(res_ok['outcome'])
        self.assertTrue(res_ok['completion_evidence_reliable'])

        raw_incomplete = copy.deepcopy(raw_ok)
        raw_incomplete['condition_checks'][0]['status'] = 'unknown'
        res_incomplete = guard.evaluate(raw_incomplete)
        self.assertFalse(res_incomplete['completion_evidence_reliable'])

    def test_frozen_frame_rejects_freebbox_lowconf_and_reference_instance_switch(self):
        guard = Guard('Walk toward the chair. Stop beside the chair to the left of the statue.')
        guard.navigation_cfg['task_visual_enabled'] = True
        ctx = guard.prepare()
        _add_candidate(guard, '5:front:0', 'chair', [0.10, 0.35, 0.30, 0.75], entity_id=10)
        _add_candidate(guard, '5:front:1', 'statue', [0.70, 0.35, 0.90, 0.75], entity_id=20)

        raw5 = _verdict(ctx)
        raw5['spatial_landmark_region'] = {
            'role': 'spatial_landmark', 'view': 'front', 'bbox': [0.70, 0.35, 0.90, 0.75],
            'bbox_space': 'full_view', 'candidate_id': '5:front:1', 'entity_id': 20,
        }
        res5 = guard.evaluate(raw5)
        self.assertTrue(res5['outcome'])
        self.assertEqual(res5['spatial_relation']['frame_observation_id'], 5)

        guard.mapper.observation_id = 6
        guard.mapper.observation_step = 4
        guard._observation_revision = 6
        guard.mapper.current_visual_candidates = {}
        guard.mapper.current_observation_masks = {}
        ctx = guard.prepare()
        _add_candidate(guard, '6:front:0', 'chair', [0.10, 0.35, 0.30, 0.75], obs_id=6, entity_id=10)
        _add_candidate(guard, '6:front:1', 'statue', [0.70, 0.35, 0.90, 0.75], obs_id=6, conf=0.25, entity_id=20)

        raw_free = _verdict(ctx)
        for chk in raw_free['condition_checks']:
            chk['observation_id'] = 6
        raw_free['target_candidate_id'] = '6:front:0'
        raw_free['spatial_landmark_region'] = {
            'role': 'spatial_landmark', 'view': 'front', 'bbox': [0.70, 0.35, 0.90, 0.75],
            'bbox_space': 'full_view', 'candidate_id': None, 'entity_id': None,
        }
        res_free = guard.evaluate(raw_free)
        self.assertFalse(res_free['outcome'])
        self.assertEqual(res_free['failure_type'], 'not_reached')
        self.assertFalse(res_free['rollback_required'])
        self.assertEqual(res_free['spatial_relation']['status'], 'unknown')

        raw_lowconf = copy.deepcopy(raw_free)
        raw_lowconf['spatial_landmark_region']['candidate_id'] = '6:front:1'
        res_lowconf = guard.evaluate(raw_lowconf)
        self.assertFalse(res_lowconf['outcome'])
        self.assertEqual(res_lowconf['failure_type'], 'not_reached')
        self.assertFalse(res_lowconf['rollback_required'])
        self.assertEqual(res_lowconf['spatial_landmark_grounding']['reason'], 'low_confidence_spatial_landmark')
        self.assertEqual(res_lowconf['spatial_relation']['status'], 'unknown')

        guard.mapper.current_visual_candidates = {}
        guard.mapper.current_observation_masks = {}
        _add_candidate(guard, '6:front:0', 'chair', [0.10, 0.35, 0.30, 0.75], obs_id=6, entity_id=10)
        _add_candidate(guard, '6:front:2', 'statue', [0.70, 0.35, 0.90, 0.75], obs_id=6, conf=0.92, entity_id=99)
        raw_diff_id = copy.deepcopy(raw_free)
        raw_diff_id['spatial_landmark_region'].update(candidate_id='6:front:2', entity_id=99)
        res_diff_id = guard.evaluate(raw_diff_id)
        self.assertFalse(res_diff_id['outcome'])
        self.assertEqual(res_diff_id['spatial_landmark_grounding']['reason'], 'current_reference_instance_changed')
        self.assertEqual(res_diff_id['spatial_relation']['status'], 'unknown')

        guard.mapper.current_visual_candidates = {}
        guard.mapper.current_observation_masks = {}
        _add_candidate(guard, '6:front:0', 'chair', [0.10, 0.35, 0.30, 0.75], obs_id=6, entity_id=10)
        _add_candidate(guard, '6:front:3', 'statue', [0.70, 0.35, 0.90, 0.75], obs_id=6, conf=0.92, entity_id=None, depth_m=2.3)
        raw_shifted = copy.deepcopy(raw_free)
        raw_shifted['spatial_landmark_region'].update(candidate_id='6:front:3', entity_id=None)
        res_shifted = guard.evaluate(raw_shifted)
        self.assertFalse(res_shifted['outcome'])
        self.assertEqual(res_shifted['spatial_landmark_grounding']['reason'], 'current_reference_instance_changed')
        self.assertEqual(res_shifted['spatial_relation']['status'], 'unknown')

        guard.mapper.current_visual_candidates = {}
        guard.mapper.current_observation_masks = {}
        _add_candidate(guard, '6:front:0', 'chair', [0.10, 0.35, 0.30, 0.75], obs_id=6, entity_id=10)
        _add_candidate(guard, '6:front:4', 'statue', [0.70, 0.35, 0.90, 0.75], obs_id=6, conf=0.92, entity_id=20, depth_m=1.1)
        raw_same = copy.deepcopy(raw_free)
        raw_same['spatial_landmark_region'].update(candidate_id='6:front:4', entity_id=20)
        res_same = guard.evaluate(raw_same)
        self.assertTrue(res_same['outcome'])
        self.assertEqual(res_same['spatial_relation']['status'], 'satisfied')

    def test_approaching_outcome_false_freezes_frame_before_camera_180_turn(self):
        yaw_180 = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        rot_180 = yaw_180 @ CALIB_R

        guard = Guard('Walk toward the chair. Stop beside the chair to the left of the statue.')
        guard.navigation_cfg['task_visual_enabled'] = True
        ctx5 = guard.prepare()
        _add_candidate(guard, '5:front:0', 'chair', [0.10, 0.35, 0.30, 0.75], obs_id=5, entity_id=10, depth_m=1.0)
        _add_candidate(guard, '5:front:1', 'statue', [0.70, 0.35, 0.90, 0.75], obs_id=5, entity_id=20, depth_m=1.0)

        raw5_false = _verdict(ctx5)
        raw5_false['outcome'] = False
        raw5_false['relation_satisfied'] = False
        raw5_false['target_proximity'] = 'approaching'
        raw5_false['spatial_landmark_region'] = {
            'role': 'spatial_landmark', 'view': 'front', 'bbox': [0.70, 0.35, 0.90, 0.75],
            'bbox_space': 'full_view', 'candidate_id': '5:front:1', 'entity_id': 20,
        }
        res5 = guard.evaluate(raw5_false)
        self.assertFalse(res5['outcome'])
        self.assertFalse(res5['relation_satisfied'])
        self.assertTrue(res5['spatial_observation_grounding']['valid'])
        self.assertFalse(res5['spatial_observation_grounding']['accepted'])
        self.assertEqual(res5['spatial_relation']['status'], 'satisfied')
        self.assertEqual(res5['spatial_relation']['frame_observation_id'], 5)

        guard.mapper.observation_id = 6
        guard.mapper.observation_step = 4
        guard._observation_revision = 6
        guard.curr_obs['agent_state'] = _make_state(pos=(0.0, 2.0, 0.0), rot=rot_180)
        guard.mapper.current_visual_candidates = {}
        guard.mapper.current_observation_masks = {}
        ctx6 = guard.prepare()
        _add_candidate(guard, '6:front:0', 'chair', [0.70, 0.35, 0.90, 0.75], obs_id=6, entity_id=10, depth_m=1.0)
        _add_candidate(guard, '6:front:1', 'statue', [0.10, 0.35, 0.30, 0.75], obs_id=6, entity_id=20, depth_m=1.0)

        raw6_true = _verdict(ctx6)
        for chk in raw6_true['condition_checks']:
            chk['observation_id'] = 6
        raw6_true['target_candidate_id'] = '6:front:0'
        raw6_true['target_bbox'] = [0.70, 0.35, 0.90, 0.75]
        raw6_true['spatial_landmark_region'] = {
            'role': 'spatial_landmark', 'view': 'front', 'bbox': [0.10, 0.35, 0.30, 0.75],
            'bbox_space': 'full_view', 'candidate_id': '6:front:1', 'entity_id': 20,
        }
        res6 = guard.evaluate(raw6_true)
        self.assertTrue(res6['outcome'])
        self.assertEqual(res6['spatial_relation']['status'], 'satisfied')
        self.assertEqual(res6['spatial_relation']['frame_observation_id'], 5)

        guard_contra = Guard('Walk toward the chair. Stop beside the chair to the left of the statue.')
        guard_contra.navigation_cfg['task_visual_enabled'] = True
        ctx_c5 = guard_contra.prepare()
        _add_candidate(guard_contra, '5:front:0', 'chair', [0.70, 0.35, 0.90, 0.75], obs_id=5, entity_id=10, depth_m=1.0)
        _add_candidate(guard_contra, '5:front:1', 'statue', [0.10, 0.35, 0.30, 0.75], obs_id=5, entity_id=20, depth_m=1.0)
        raw_c5 = _verdict(ctx_c5)
        raw_c5['outcome'] = False
        raw_c5['relation_satisfied'] = False
        raw_c5['target_bbox'] = [0.70, 0.35, 0.90, 0.75]
        raw_c5['spatial_landmark_region'] = {
            'role': 'spatial_landmark', 'view': 'front', 'bbox': [0.10, 0.35, 0.30, 0.75],
            'bbox_space': 'full_view', 'candidate_id': '5:front:1', 'entity_id': 20,
        }
        res_c5 = guard_contra.evaluate(raw_c5)
        self.assertFalse(res_c5['outcome'])
        self.assertEqual(res_c5['spatial_relation']['status'], 'contradicted')
        self.assertEqual(res_c5['spatial_relation']['frame_observation_id'], 5)

        guard_contra.mapper.observation_id = 6
        guard_contra.mapper.observation_step = 4
        guard_contra._observation_revision = 6
        guard_contra.curr_obs['agent_state'] = _make_state(pos=(0.0, 2.0, 0.0), rot=rot_180)
        guard_contra.mapper.current_visual_candidates = {}
        guard_contra.mapper.current_observation_masks = {}
        ctx_c6 = guard_contra.prepare()
        _add_candidate(guard_contra, '6:front:0', 'chair', [0.10, 0.35, 0.30, 0.75], obs_id=6, entity_id=10, depth_m=1.0)
        _add_candidate(guard_contra, '6:front:1', 'statue', [0.70, 0.35, 0.90, 0.75], obs_id=6, entity_id=20, depth_m=1.0)
        raw_c6 = _verdict(ctx_c6)
        for chk in raw_c6['condition_checks']:
            chk['observation_id'] = 6
        raw_c6['target_candidate_id'] = '6:front:0'
        raw_c6['target_bbox'] = [0.10, 0.35, 0.30, 0.75]
        raw_c6['spatial_landmark_region'] = {
            'role': 'spatial_landmark', 'view': 'front', 'bbox': [0.70, 0.35, 0.90, 0.75],
            'bbox_space': 'full_view', 'candidate_id': '6:front:1', 'entity_id': 20,
        }
        res_c6 = guard_contra.evaluate(raw_c6)
        self.assertFalse(res_c6['outcome'])
        self.assertEqual(res_c6['spatial_relation']['status'], 'contradicted')
        self.assertEqual(res_c6['spatial_relation']['frame_observation_id'], 5)


if __name__ == '__main__':
    unittest.main()


import json
from pathlib import Path
import sys
import types
import unittest
from unittest import mock
import numpy as np

SRC = Path(__file__).resolve().parents[1] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agent.endpoint_guard import EndpointGroundingGuard
from agent.route_guard import RouteEvidenceGuard
from agent.visual_grounding import VisualGroundingMixin

CALIB_R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=float)
YAW_180 = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _make_state(pos=(0.0, 0.0, 0.0), rot=None):
    if rot is None:
        rot = CALIB_R
    sensor = types.SimpleNamespace(rotation=np.asarray(rot, dtype=float))
    return types.SimpleNamespace(position=np.asarray(pos, dtype=float), sensor_states={'color_sensor': sensor})


def _make_guard(visual_enabled=True, cls=RouteEvidenceGuard):
    guard = cls()
    guard.navigation_cfg = {
        'task_visual_enabled': visual_enabled,
        'endpoint_max_distance_m': 1.8,
        'endpoint_between_max_distance_m': 2.2,
        'endpoint_door_frame_max_distance_m': 2.2,
        'endpoint_mask_min_confidence': 0.48,
    }
    intrinsic = np.array([[100.0, 0.0, 100.0], [0.0, 100.0, 100.0], [0.0, 0.0, 1.0]], dtype=float)
    guard.mapper = types.SimpleNamespace(
        observation_id=5,
        observation_step=5,
        floor_level=0,
        current_position=np.array([0.0, 0.0, 0.0], dtype=float),
        current_rotation=CALIB_R.copy(),
        initial_position=np.array([0.0, 0.0, 0.0], dtype=float),
        initial_floor_height=0.0,
        camera_intrinsic=intrinsic,
        rotation_func=lambda r: np.asarray(r, dtype=float),
        translation_func=lambda p: np.asarray(p, dtype=float),
        object_entities=[],
        current_observation_masks={},
        current_visual_candidates={},
    )
    depth = np.full((200, 200), 1.1, dtype=np.float32)
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    rgb[:, :, 1] = 120
    guard.curr_obs = {
        'color_sensor': rgb.copy(),
        'depth_sensor': depth.copy(),
        'agent_state': _make_state(),
        'left': {'color_sensor': rgb.copy(), 'depth_sensor': depth.copy(), 'agent_state': _make_state()},
        'right': {'color_sensor': rgb.copy(), 'depth_sensor': depth.copy(), 'agent_state': _make_state()},
        'back': {'color_sensor': rgb.copy(), 'depth_sensor': depth.copy(), 'agent_state': _make_state()},
    }
    return guard


def _add_candidate(guard, cid, cls_name, bbox, view='front', entity_id=None, obs_id=5, conf=0.88,
                   roles=None, horizontal_distance_m=None, custom_mask=None):
    w, h = 200, 200
    if custom_mask is not None:
        mask = np.asarray(custom_mask, dtype=bool)
    else:
        mask = np.zeros((h, w), dtype=bool)
        x1, y1 = int(round(float(bbox[0]) * w)), int(round(float(bbox[1]) * h))
        x2, y2 = int(round(float(bbox[2]) * w)), int(round(float(bbox[3]) * h))
        mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
    guard.mapper.current_observation_masks[cid] = mask
    canonical = cls_name.strip().lower().replace(' ', '_')
    rec = {
        'candidate_id': cid,
        'observation_id': obs_id,
        'view': view,
        'detector_class_name': cls_name,
        'canonical_label': canonical,
        'class_name': cls_name,
        'entity_id': entity_id,
        'bbox': list(bbox),
        'original_bbox': list(bbox),
        'bbox_space': 'full_view',
        'mask_key': cid,
        'image_size': [w, h],
        'confidence': conf,
        'geometry_status': '2d_only' if entity_id is None else '3d_measured',
        'horizontal_distance_m': horizontal_distance_m if horizontal_distance_m is not None else (1.1 if entity_id is not None else None),
    }
    if roles is not None:
        rec['roles'] = list(roles)
    guard.mapper.current_visual_candidates[cid] = rec
    return rec


class VisualGroundingIntegrationTests(unittest.TestCase):
    def test_2d_candidate_entity_none_measures_real_mask_and_depth(self):
        self.assertTrue(issubclass(EndpointGroundingGuard, VisualGroundingMixin))
        self.assertTrue(issubclass(RouteEvidenceGuard, VisualGroundingMixin))
        guard = _make_guard(visual_enabled=True)
        _add_candidate(guard, '5:front:0', 'chair', [0.30, 0.30, 0.70, 0.75], entity_id=None)
        result = {'relation_satisfied': True}
        reason = guard._ground_endpoint_verdict(
            {'target_view': 'front', 'target_bbox': [0.32, 0.32, 0.68, 0.72],
             'target_candidate_id': '5:front:0', 'target_bbox_space': 'full_view', 'target_entity_id': None},
            {'key': 'ep_chair', 'target_text': 'stop near the chair'},
            result,
        )
        self.assertIsNone(reason)
        pkt = result['endpoint_grounding']
        self.assertTrue(pkt['valid'] and pkt['accepted'])
        self.assertEqual(pkt['candidate_id'], '5:front:0')
        self.assertIsNone(pkt['target_id'])
        self.assertEqual(pkt['source'], 'detector_mask')
        self.assertTrue(0.9 <= pkt['depth_m'] <= 1.3)
        self.assertNotIn('mask', pkt)

    def test_invalid_explicit_candidate_never_falls_back_to_entity_or_free_bbox(self):
        guard = _make_guard(visual_enabled=True)
        _add_candidate(guard, '5:front:0', 'chair', [0.30, 0.30, 0.70, 0.75], entity_id=0)
        _add_candidate(guard, '5:front:1', 'plant', [0.30, 0.30, 0.70, 0.75], entity_id=None)
        guard.mapper.object_entities = [{
            'class_name': 'chair',
            'observations': [{'observation_id': 5, 'view': 'front', 'bbox': [0.30, 0.30, 0.70, 0.75],
                              'mask_key': '5:front:0', 'confidence': 0.9}],
        }]
        for bad_id in ('5:front:99', '4:front:0', '5:front:1', 0, ''):
            result = {}
            reason = guard._ground_endpoint_verdict(
                {'target_view': 'front', 'target_bbox': [0.30, 0.30, 0.70, 0.75],
                 'target_candidate_id': bad_id, 'target_bbox_space': 'full_view', 'target_entity_id': 0},
                {'key': 'ep_chair', 'target_text': 'stop near the chair'},
                result,
            )
            self.assertIsNotNone(reason)
            self.assertFalse(result['endpoint_grounding']['valid'])

    def test_crop_bbox_space_rejected_when_enabled_and_legacy_preserved_when_off(self):
        guard = _make_guard(visual_enabled=True)
        res_crop = {}
        reason_crop = guard._ground_endpoint_verdict(
            {'target_view': 'front', 'target_bbox': [0.30, 0.30, 0.70, 0.75], 'target_bbox_space': 'crop'},
            {'key': 'ep_chair', 'target_text': 'stop near the chair'},
            res_crop,
        )
        self.assertIsNotNone(reason_crop)
        self.assertFalse(res_crop['endpoint_grounding']['valid'])

        guard_off = _make_guard(visual_enabled=False)
        res_legacy = {}
        guard_off._ground_endpoint_verdict(
            {'target_view': 'front', 'target_bbox': [0.30, 0.30, 0.70, 0.75]},
            {'key': 'ep_chair', 'target_text': 'stop near the chair'},
            res_legacy,
        )
        self.assertTrue(res_legacy['endpoint_grounding']['valid'])

    def test_between_rejects_same_candidate_and_doorway_verifies_split_jambs(self):
        guard = _make_guard(visual_enabled=True)
        _add_candidate(guard, '5:front:0', 'chair', [0.05, 0.20, 0.95, 0.80], entity_id=None)
        res_between = {'relation_satisfied': True}
        reason_between = guard._ground_endpoint_verdict(
            {
                'target_view': 'front', 'target_bbox': [0.05, 0.25, 0.25, 0.75],
                'target_candidate_id': '5:front:0', 'target_bbox_space': 'full_view',
                'reference_region': {'view': 'front', 'bbox': [0.75, 0.25, 0.95, 0.75],
                                     'candidate_id': '5:front:0', 'bbox_space': 'full_view'},
            },
            {'key': 'ep_bw', 'target_text': 'stop between the two chairs'},
            res_between,
        )
        self.assertIsNotNone(reason_between)
        self.assertFalse(res_between['endpoint_grounding']['geometry_verified'])

        jamb_mask = np.zeros((200, 200), dtype=bool)
        jamb_mask[30:180, 48:68] = True
        jamb_mask[30:180, 132:152] = True
        _add_candidate(
            guard, '5:front:1', 'door', [0.20, 0.15, 0.80, 0.90],
            entity_id=None, custom_mask=jamb_mask,
        )
        guard.curr_obs['depth_sensor'][:, :] = 1.2
        guard.curr_obs['depth_sensor'][:, 70:130] = 2.6
        door_parsed = {
            'target_view': 'front', 'target_bbox': [0.24, 0.15, 0.34, 0.90],
            'target_candidate_id': '5:front:1', 'target_bbox_space': 'full_view',
            'reference_region': {'role': 'door_jamb', 'view': 'front', 'bbox': [0.66, 0.15, 0.76, 0.90],
                                 'candidate_id': '5:front:1', 'bbox_space': 'full_view'},
        }
        res_no_path = {'relation_satisfied': True}
        reason_no_path = guard._ground_endpoint_verdict(door_parsed, {'key': 'ep_door', 'target_text': 'stop at the doorway'}, res_no_path)
        self.assertTrue(res_no_path['endpoint_grounding']['valid'])
        self.assertTrue(res_no_path['endpoint_reference_grounding']['valid'])
        frame_support = res_no_path['endpoint_grounding']['frame_support']
        self.assertIn('left', frame_support)
        self.assertIn('right', frame_support)
        self.assertEqual(len(frame_support['left']['world_center']), 3)
        self.assertEqual(len(frame_support['right']['world_center']), 3)
        self.assertGreater(frame_support['left']['depth_m'], 0.0)
        self.assertGreater(frame_support['right']['depth_m'], 0.0)
        self.assertFalse(res_no_path['endpoint_grounding']['geometry_verified'])
        self.assertFalse(res_no_path['endpoint_grounding']['accepted'])
        self.assertIsNotNone(reason_no_path)

        tests_dir = Path(__file__).resolve().parent
        if str(tests_dir) not in sys.path:
            sys.path.insert(0, str(tests_dir))
        import test_v3_endpoint_integration as v3_ep
        door_guard, raw_door = v3_ep.EndpointIntegrationTests()._split_door_frames(role='door_jamb')
        door_guard.navigation_cfg['task_visual_enabled'] = True
        obs_id = getattr(door_guard.mapper, 'observation_id', 5)
        cands = {}
        for eid, ent in enumerate(door_guard.mapper.object_entities):
            obs_entry = ent['observations'][0]
            view_name = obs_entry['view']
            cid = f'{obs_id}:{view_name}:0'
            old_mask = door_guard.mapper.current_observation_masks[obs_entry['mask_key']]
            h, w = old_mask.shape[:2]
            door_guard.mapper.current_observation_masks[cid] = old_mask
            cands[cid] = {
                'candidate_id': cid,
                'mask_key': cid,
                'observation_id': obs_id,
                'step': getattr(door_guard.mapper, 'observation_step', 1),
                'view': view_name,
                'canonical_label': ent['class_name'],
                'detector_class_name': ent['class_name'],
                'confidence': float(obs_entry.get('confidence', 0.9)),
                'bbox': list(obs_entry['bbox']),
                'image_size': [w, h],
                'entity_id': eid,
            }
        door_guard.mapper.current_visual_candidates = cands
        left_cid = f'{obs_id}:left:0'
        right_cid = f'{obs_id}:right:0'
        raw_door['target_candidate_id'] = left_cid
        raw_door['target_bbox_space'] = 'full_view'
        raw_door['reference_region']['candidate_id'] = right_cid
        raw_door['reference_region']['bbox_space'] = 'full_view'
        res_door = {'relation_satisfied': True}
        reason_door = door_guard._ground_endpoint_verdict(
            raw_door, {'key': 'ep_door_ok', 'target_text': 'stop at the doorway'}, res_door,
        )
        self.assertIsNone(reason_door)
        self.assertTrue(res_door['endpoint_grounding']['geometry_verified'])
        self.assertTrue(res_door['endpoint_grounding']['accepted'])
        self.assertTrue(res_door['endpoint_reference_grounding']['valid'])

    def test_production_route_evidence_guard_signature_prompt_and_crop_priority(self):
        guard = _make_guard(visual_enabled=True, cls=RouteEvidenceGuard)
        self.assertEqual(guard._visual_candidate_prompt(), guard._visual_candidate_prompt())
        self.assertEqual(guard._visual_candidate_content(for_verifier=False), [])

        _add_candidate(guard, '5:front:0', 'door', [0.10, 0.10, 0.40, 0.90], conf=0.95)
        _add_candidate(guard, '5:right:0', 'door', [0.15, 0.10, 0.45, 0.90], view='right', conf=0.92)
        _add_candidate(guard, '5:left:0', 'eye chart', [0.20, 0.20, 0.60, 0.60], view='left', conf=0.80)
        _add_candidate(guard, '5:back:0', 'chair', [0.25, 0.25, 0.65, 0.75], view='back', conf=0.75)
        _add_candidate(guard, '5:front:9', 'door', [0.10, 0.10, 0.30, 0.30], conf=float('nan'))
        guard._active_turn_clause = lambda: {'landmark': 'eye chart', 'text': 'turn left at the eye chart'}
        guard._route_verification_context = {
            'target_text': 'turn left at the eye chart and stop at the doorway to the left of the chair',
        }

        prompt = guard._visual_candidate_prompt()
        self.assertIn('normalized [0, 1] full-view xyxy', prompt)
        self.assertIn('5:left:0', prompt)
        self.assertIn('horizontal_distance_m=null', prompt)
        self.assertNotIn('5:front:9', prompt)

        planner_blocks = guard._visual_candidate_content(for_verifier=False)
        self.assertEqual(len(planner_blocks), 4)
        planner_meta = guard._last_visual_candidate_crops_for_planner
        self.assertEqual(len(planner_meta), 2)
        self.assertEqual(planner_meta[0]['candidate_id'], '5:left:0')
        self.assertIn('turn', planner_meta[0]['roles'])
        json.dumps(planner_meta)

        for i in range(1, 7):
            _add_candidate(guard, f'5:front:{i}', 'door', [0.10, 0.10, 0.40, 0.90], conf=0.90 - i * 0.01)
        verifier_blocks = guard._visual_candidate_content(for_verifier=True)
        verifier_meta = guard._last_visual_candidate_crops_for_verifier
        self.assertEqual(len(verifier_meta), 4)
        self.assertEqual(len(verifier_blocks), 8)
        verifier_cids = {m['candidate_id'] for m in verifier_meta}
        self.assertIn('5:back:0', verifier_cids)

        guard.curr_obs = {'cached_images': {'front': np.zeros((200, 200, 3), dtype=np.uint8)}}
        self.assertEqual(guard._visual_candidate_content(for_verifier=True), [])

    def test_spatial_relation_role_gate_preserves_caller_flags_and_freezes_sensor_frame(self):
        guard = _make_guard(visual_enabled=True)
        _add_candidate(guard, '5:front:0', 'chair', [0.10, 0.35, 0.30, 0.70], entity_id=None)
        _add_candidate(guard, '5:front:1', 'plant', [0.70, 0.35, 0.90, 0.70], entity_id=None)
        ctx = {'key': 'ep_rel', 'checks': [{'condition_id': 'cond_1'}], 'target_text': 'stop at the chair to the left of the plant'}
        parsed = {
            'target_view': 'front', 'target_bbox': [0.10, 0.35, 0.30, 0.70],
            'target_candidate_id': '5:front:0', 'target_bbox_space': 'full_view',
            'reference_region': {'role': 'door_jamb', 'view': 'front', 'bbox': [0.70, 0.35, 0.90, 0.70],
                                 'candidate_id': '5:front:1', 'bbox_space': 'full_view'},
        }
        result = {'relation_satisfied': True, 'outcome': False}
        self.assertIsNone(guard._ground_endpoint_verdict(parsed, ctx, result))
        result['relation_satisfied'] = False
        self.assertIsNotNone(guard._verify_spatial_relation(parsed, ctx, result))
        self.assertEqual(result['spatial_relation']['status'], 'unknown')

        for bad_role in ('door_jamb', 'between', 'unknown', None):
            parsed['spatial_landmark_region'] = {
                'role': bad_role, 'view': 'front', 'bbox': [0.70, 0.35, 0.90, 0.70],
                'candidate_id': '5:front:1', 'bbox_space': 'full_view',
            }
            self.assertIsNotNone(guard._verify_spatial_relation(parsed, ctx, result))
            self.assertEqual(result['spatial_landmark_grounding']['reason'], 'invalid_spatial_landmark_role')
            self.assertEqual(result['spatial_relation']['status'], 'unknown')
            self.assertFalse(result['relation_satisfied'])
            self.assertFalse(result['outcome'])

        parsed['spatial_landmark_region'] = {
            'role': 'spatial_landmark', 'view': 'front', 'bbox': [0.70, 0.35, 0.90, 0.70],
            'candidate_id': '5:front:1', 'bbox_space': 'full_view',
        }
        self.assertIsNone(guard._verify_spatial_relation(parsed, ctx, result))
        self.assertEqual(result['spatial_relation']['status'], 'satisfied')
        self.assertEqual(result['spatial_relation']['frame_observation_id'], 5)
        self.assertFalse(result['relation_satisfied'])
        self.assertFalse(result['outcome'])

        rot_180 = YAW_180 @ CALIB_R
        guard.mapper.observation_id = 6
        guard.mapper.observation_step = 6
        state_180 = _make_state(rot=rot_180)
        state_180.position = np.array([0.0, 2.0, 0.0], dtype=float)
        state_180.sensor_states['color_sensor'].position = np.array([0.0, 2.0, 0.0], dtype=float)
        if 'depth_sensor' in state_180.sensor_states:
            state_180.sensor_states['depth_sensor'].position = np.array([0.0, 2.0, 0.0], dtype=float)
        guard.curr_obs['agent_state'] = state_180
        guard.mapper.current_visual_candidates = {}
        guard.mapper.current_observation_masks = {}
        _add_candidate(guard, '6:front:0', 'chair', [0.70, 0.35, 0.90, 0.70], obs_id=6)
        _add_candidate(guard, '6:front:1', 'plant', [0.10, 0.35, 0.30, 0.70], obs_id=6)
        parsed_turn = {
            'target_view': 'front', 'target_bbox': [0.70, 0.35, 0.90, 0.70],
            'target_candidate_id': '6:front:0', 'target_bbox_space': 'full_view',
            'spatial_landmark_region': {
                'role': 'spatial_landmark', 'view': 'front', 'bbox': [0.10, 0.35, 0.30, 0.70],
                'candidate_id': '6:front:1', 'bbox_space': 'full_view',
            },
        }
        result_after_turn = {'relation_satisfied': True}
        self.assertIsNone(guard._ground_endpoint_verdict(parsed_turn, ctx, result_after_turn))
        self.assertIsNone(guard._verify_spatial_relation(parsed_turn, ctx, result_after_turn))
        self.assertEqual(result_after_turn['spatial_relation']['status'], 'satisfied')
        self.assertEqual(result_after_turn['spatial_relation']['frame_observation_id'], 5)

    def test_spatial_frame_cache_invalidation_on_rollback_and_bounds(self):
        guard = _make_guard(visual_enabled=True)
        _add_candidate(guard, '5:front:0', 'chair', [0.10, 0.35, 0.30, 0.70], entity_id=None)
        _add_candidate(guard, '5:front:1', 'plant', [0.70, 0.35, 0.90, 0.70], entity_id=None)
        ctx = {'checks': [{'condition_id': 'cond_ep_1'}], 'target_text': 'stop at the chair to the left of the plant'}
        parsed = {
            'target_view': 'front', 'target_bbox': [0.10, 0.35, 0.30, 0.70],
            'target_candidate_id': '5:front:0', 'target_bbox_space': 'full_view',
            'spatial_landmark_region': {
                'role': 'spatial_landmark', 'view': 'front', 'bbox': [0.70, 0.35, 0.90, 0.70],
                'candidate_id': '5:front:1', 'bbox_space': 'full_view',
            },
        }
        res5 = {'relation_satisfied': True}
        self.assertIsNone(guard._ground_endpoint_verdict(parsed, ctx, res5))
        self.assertIsNone(guard._verify_spatial_relation(parsed, ctx, res5))
        self.assertEqual(res5['spatial_relation']['frame_observation_id'], 5)

        guard.navigation_trace = [{'event': 'wrong_path_rollback_triggered', 'step': 6, 'condition_id': 'cond_ep_1'}]
        guard.subtask_execution_ledger = {'cond_ep_1': {'rejected_by_rollback': True, 'rollback_count': 1}}
        guard.mapper.observation_id = 7
        guard.mapper.observation_step = 7
        guard.mapper.current_visual_candidates = {}
        guard.mapper.current_observation_masks = {}
        _add_candidate(guard, '7:front:0', 'chair', [0.10, 0.35, 0.30, 0.70], obs_id=7)
        _add_candidate(guard, '7:left:0', 'plant', [0.70, 0.35, 0.90, 0.70], view='left', obs_id=7)
        parsed_diff_view = {
            'target_view': 'front', 'target_bbox': [0.10, 0.35, 0.30, 0.70],
            'target_candidate_id': '7:front:0', 'target_bbox_space': 'full_view',
            'spatial_landmark_region': {
                'role': 'spatial_landmark', 'view': 'left', 'bbox': [0.70, 0.35, 0.90, 0.70],
                'candidate_id': '7:left:0', 'bbox_space': 'full_view',
            },
        }
        res7 = {'relation_satisfied': True}
        self.assertIsNone(guard._ground_endpoint_verdict(parsed_diff_view, ctx, res7))
        self.assertIsNotNone(guard._verify_spatial_relation(parsed_diff_view, ctx, res7))
        self.assertEqual(res7['spatial_relation']['status'], 'unknown')

        _add_candidate(guard, '7:front:1', 'plant', [0.70, 0.35, 0.90, 0.70], view='front', obs_id=7)
        parsed_same_view = {
            'target_view': 'front', 'target_bbox': [0.10, 0.35, 0.30, 0.70],
            'target_candidate_id': '7:front:0', 'target_bbox_space': 'full_view',
            'spatial_landmark_region': {
                'role': 'spatial_landmark', 'view': 'front', 'bbox': [0.70, 0.35, 0.90, 0.70],
                'candidate_id': '7:front:1', 'bbox_space': 'full_view',
            },
        }
        for i in range(20):
            res_i = {'relation_satisfied': True}
            ctx_i = {'checks': [{'condition_id': f'cond_{i}'}], 'target_text': 'stop at the chair to the left of the plant'}
            self.assertIsNone(guard._ground_endpoint_verdict(parsed_same_view, ctx_i, res_i))
            self.assertIsNone(guard._verify_spatial_relation(parsed_same_view, ctx_i, res_i))
            self.assertEqual(res_i['spatial_relation']['frame_observation_id'], 7)
        self.assertLessEqual(len(guard._spatial_relation_frames), 16)


if __name__ == '__main__':
    unittest.main()


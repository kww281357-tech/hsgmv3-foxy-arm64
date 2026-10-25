"""Tests for visual candidate grounding in Habitat structural stair endpoints."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from agent.habitat_control import HabitatControl
from structural_endpoints import current_floor_depth, stair_endpoint_side
from target_identity import detector_matches_endpoint
from test_v3_verification_contract import Guard, record, verdict


class HybridStairGuard(HabitatControl, Guard):
    pass


class VisualStairGroundingTests(unittest.TestCase):
    def make(self, instruction='Stop at the first landing at the bottom of the stairs.', visual_enabled=True):
        a = HybridStairGuard(instruction)
        a.st['record'] = [record(1)]
        a.navigation_cfg.update(
            habitat_control_enabled=True,
            require_endpoint_grounding=True,
            task_visual_enabled=bool(visual_enabled),
            endpoint_mask_min_confidence=0.48,
        )
        a.mapper.current_position = np.array([0.0, 0.0, 1.2])
        a.mapper.observation_id = 5
        a.mapper.observation_step = 5
        a.mapper.camera_intrinsic = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
        a.mapper.rotation_func = lambda r: np.asarray(r, dtype=float)
        a.mapper.translation_func = lambda p: np.asarray(p, dtype=float)
        a.mapper.initial_position = np.zeros(3)
        a.mapper.object_entities = []
        a.mapper.current_rotation = np.eye(3)
        points = np.array([[0.0, 0.0, 0.2], [0.1, 0.0, 0.2]])
        mask = np.zeros((100, 100), dtype=bool)
        mask[30:70, 35:80] = True

        def inputs():
            rotation = a.mapper.current_rotation
            camera = {'position': [0.0, 0.0, 1.2], 'rotation': rotation}
            depth = np.full((100, 100), 1.2, dtype=np.float32)
            for point in (points - [0.0, 0.0, 1.2]) @ rotation:
                x = int(round(point[0] * 100.0 / -point[2] + 50.0))
                y = int(round(99.0 - (point[1] * 100.0 / -point[2] + 50.0)))
                depth[y - 1:y + 2, x - 1:x + 2] = float(-point[2])
            state = NS(position=np.array([0.0, 0.0, 1.2]), sensor_states={'color_sensor': NS(rotation=rotation)})
            a.curr_obs = {'depth_sensor': depth, 'agent_state': state}
            return {'front': depth}, {'front': camera}

        def measured():
            base = {
                'valid': True,
                'current': True,
                'checked_ground': [0.0, 0.0, 0.0],
                'observation_id': a.mapper.observation_id,
            }
            depths, cameras = inputs()
            return {
                'ground': np.zeros(3),
                'observation_id': a.mapper.observation_id,
                'platform': dict(base),
                'profile': dict(
                    base,
                    ascending=True,
                    descending=False,
                    ascending_points=points.tolist(),
                    descending_points=[],
                ),
                'floor': current_floor_depth(
                    depths, cameras, a.mapper.camera_intrinsic, [0.0, 0.0, 0.0], a.mapper.observation_id
                ),
            }

        def set_candidate(obs_id=5, view='front', label='stairs', confidence=0.84, bbox=None, entity_id=None):
            cid = f'{int(obs_id)}:{view}:0'
            box = [0.35, 0.30, 0.80, 0.70] if bbox is None else list(bbox)
            cand = {
                'candidate_id': cid,
                'mask_key': cid,
                'detector_class_name': label,
                'canonical_label': label,
                'view': view,
                'observation_id': int(obs_id),
                'image_size': [100, 100],
                'bbox': box,
                'confidence': confidence,
                'entity_id': entity_id,
            }
            a.mapper.current_visual_candidates = {cid: cand}
            a.mapper.current_observation_masks = {cid: mask.copy()}
            inputs()
            return cid

        a._habitat_camera_inputs = inputs
        a._habitat_measurements = measured
        a.set_candidate = set_candidate
        a.set_candidate()
        return a

    def raw(self, a, **overrides):
        raw = verdict(a.prepare())
        cid = f'{a.mapper.observation_id}:front:0'
        raw.update(
            target_identity='The bottom landing of the visible stair flight',
            target_view='front',
            target_bbox=[0.35, 0.30, 0.80, 0.70],
            target_bbox_space='full_view',
            target_candidate_id=cid,
            target_entity_id=None,
            target_evidence='Stairs rise beside the current flat floor.',
        )
        for check in raw['condition_checks']:
            check.update(steps=[], observation_id=a.mapper.observation_id)
        raw.update(overrides)
        return raw

    def test_valid_stairs_candidate_witness_binds_authoritative_roi_and_requires_platform_and_two_views(self):
        a = self.make('Stop at the first landing at the bottom of the stairs.')
        # Even when target_bbox is omitted in raw response, candidate supplies authoritative full_view bbox.
        first = a.evaluate(self.raw(a, target_bbox=None))
        self.assertFalse(first['outcome'])
        self.assertEqual(first['reason'], 'changed_view_confirmation_required')
        eg1 = first['endpoint_grounding']
        self.assertEqual(eg1['visual_witness_role'], 'stair_structure')
        self.assertEqual(eg1['candidate_id'], '5:front:0')
        self.assertEqual(eg1['selected_detector_class_name'], 'stairs')
        self.assertEqual(eg1['bbox_space'], 'full_view')
        self.assertEqual(eg1['bbox'], [0.35, 0.30, 0.80, 0.70])
        self.assertNotIn('mask', eg1)

        # If platform has not arrived, valid stairs candidate passes class stage but still cannot STOP.
        orig_measured = a._habitat_measurements

        def tread_only():
            data = orig_measured()
            data['platform']['valid'] = False
            return data

        a._habitat_measurements = tread_only
        tread_res = a.evaluate(self.raw(a, target_bbox=None))
        self.assertFalse(tread_res['outcome'])
        self.assertEqual(tread_res['reason'], 'platform_not_current_or_supported')
        self.assertEqual(tread_res['endpoint_grounding']['selected_detector_class_name'], 'stairs')
        a._habitat_measurements = orig_measured

        # Second changed-pose observation with valid candidate and valid platform confirms and accepts STOP.
        angle = math.radians(12)
        a.mapper.current_rotation = np.array([
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        a.mapper.observation_id = 6
        a.mapper.observation_step = 6
        a.set_candidate(obs_id=6, view='front', label='stairs')
        second = a.evaluate(self.raw(a, target_bbox=None))
        self.assertTrue(second['outcome'], second.get('reason'))
        self.assertTrue(second['endpoint_grounding']['accepted'])
        self.assertEqual(second['endpoint_grounding']['candidate_id'], '6:front:0')
        self.assertNotIn('mask', second['endpoint_grounding'])

    def test_invalid_wrong_class_stale_low_conf_or_bad_bbox_space_fail_closed_in_direct_and_settling_paths(self):
        a = self.make('Stop at the first landing at the bottom of the stairs.')

        # 1. Missing full_view bbox_space with explicit target_bbox must fail closed.
        bad_space_raw = self.raw(a, target_candidate_id=None, target_bbox_space=None)
        res_space = a.evaluate(bad_space_raw)
        self.assertFalse(res_space['outcome'])
        self.assertFalse(res_space['endpoint_grounding']['valid'])

        settling = {
            'outcome': False,
            'stair_completion_ready': False,
            'reason': 'Measured stair direction or supported landing is not yet established.',
        }
        a._habitat_observe_unsettled_stair(bad_space_raw, settling)
        self.assertFalse(settling.get('support_settling_geometry_observed', False))

        # 2. Desk candidate cannot serve as stair_structure witness even when platform/stair profile are valid.
        a.set_candidate(obs_id=5, view='front', label='desk')
        desk_raw = self.raw(a, target_candidate_id='5:front:0')
        res_desk = a.evaluate(desk_raw)
        self.assertFalse(res_desk['outcome'])
        self.assertFalse(res_desk['endpoint_grounding']['valid'])
        self.assertEqual(res_desk['endpoint_grounding']['reason'], 'class_mismatch_for_endpoint_role')

        settling_desk = {
            'outcome': False,
            'stair_completion_ready': False,
            'reason': 'Measured stair direction or supported landing is not yet established.',
        }
        a._habitat_observe_unsettled_stair(desk_raw, settling_desk)
        self.assertFalse(settling_desk.get('support_settling_geometry_observed', False))

        # 3. Unknown / stale / wrong-view / low-confidence candidate must fail closed without raw bbox fallback.
        a.set_candidate(obs_id=5, view='front', label='stairs', confidence=0.35)
        low_conf_raw = self.raw(a, target_candidate_id='5:front:0')
        res_low = a.evaluate(low_conf_raw)
        self.assertFalse(res_low['outcome'])
        self.assertFalse(res_low['endpoint_grounding']['valid'])
        self.assertEqual(res_low['endpoint_grounding']['reason'], 'target_mask_confidence_too_low')

        a.set_candidate(obs_id=5, view='front', label='stairs', confidence=0.85)
        wrong_view_raw = self.raw(a, target_view='left', target_candidate_id='5:front:0')
        res_view = a.evaluate(wrong_view_raw)
        self.assertFalse(res_view['outcome'])
        self.assertFalse(res_view['endpoint_grounding']['valid'])

        unknown_id_raw = self.raw(a, target_candidate_id='5:front:99')
        res_unknown = a.evaluate(unknown_id_raw)
        self.assertFalse(res_unknown['outcome'])
        self.assertFalse(res_unknown['endpoint_grounding']['valid'])

        settling_unknown = {
            'outcome': False,
            'stair_completion_ready': False,
            'reason': 'Measured stair direction or supported landing is not yet established.',
        }
        a._habitat_observe_unsettled_stair(unknown_id_raw, settling_unknown)
        self.assertFalse(settling_unknown.get('support_settling_geometry_observed', False))

        # Valid candidate in settling path DOES observe settling geometry without accepting STOP.
        settling_ok = {
            'outcome': False,
            'stair_completion_ready': False,
            'reason': 'Measured stair direction or supported landing is not yet established.',
        }
        a._habitat_observe_unsettled_stair(self.raw(a, target_candidate_id='5:front:0'), settling_ok)
        self.assertTrue(settling_ok.get('support_settling_geometry_observed'))
        self.assertFalse(settling_ok['endpoint_grounding']['accepted'])

    def test_plain_landing_without_stair_side_is_not_globally_aliased(self):
        self.assertIsNone(stair_endpoint_side('Stop at the first landing.'))
        self.assertFalse(detector_matches_endpoint('stairs', 'Stop at the first landing.', 'primary'))


if __name__ == '__main__':
    unittest.main()


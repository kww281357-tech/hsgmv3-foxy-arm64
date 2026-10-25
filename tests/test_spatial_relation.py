import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from spatial_relation import evaluate_spatial_relation, parse_endpoint_relation


def _frame(right_axis=(1.0, 0.0, 0.0), obs_id=2, source='measured_camera', kind='frozen_observer'):
    return {
        'kind': kind, 'source': source, 'observation_id': obs_id,
        'right_axis': list(right_axis), 'origin': [0.0, 0.0, 0.0],
    }


def _pkt(center, obs_id=5, cid='c1', tid=1, valid=True, cls='door'):
    pkt = {
        'valid': valid, 'observation_id': obs_id, 'world_center': list(center),
        'candidate_id': cid, 'target_id': tid, 'view': 'front',
    }
    if cls is not None:
        pkt['selected_detector_class_name'] = cls
    return pkt


class TestSpatialRelation(unittest.TestCase):
    def test_parse_explicit_left_and_right_relations(self):
        left = parse_endpoint_relation('doorway to the left of the white statue of two people')
        self.assertTrue(left['required'])
        self.assertEqual(left['relation'], 'left_of')
        self.assertEqual(left['target_class'], 'door')
        self.assertEqual(left['landmark_class'], 'statue')
        self.assertEqual(left['frame_kind'], 'frozen_observer')
        self.assertNotIn('doorway', left['landmark_text'])

        right = parse_endpoint_relation('door on the right side of the cabinet')
        self.assertTrue(right['required'])
        self.assertEqual(right['relation'], 'right_of')
        self.assertEqual(right['target_class'], 'door')
        self.assertEqual(right['landmark_class'], 'cabinet')

    def test_parse_unknown_reference_head_and_nondirectional_clauses(self):
        unknown = parse_endpoint_relation('door to the left of the forcefield')
        self.assertTrue(unknown['required'])
        self.assertEqual(unknown['relation'], 'left_of')
        self.assertIsNone(unknown['landmark_class'])
        self.assertEqual(unknown['status'], 'unknown')
        self.assertIn('unsupported', unknown['reason'])

        for clause in (
            'Turn right at chart and stand next to door',
            'between the sofa and the chair',
            'wait at the bottom of the stairs',
        ):
            res = parse_endpoint_relation(clause)
            self.assertFalse(res['required'])
            self.assertIsNone(res['relation'])
            self.assertEqual(res['status'], 'unknown')
            self.assertEqual(res['reason'], 'handled_by_existing_geometry')
            eval_res = evaluate_spatial_relation(res, _pkt([0, 0, 0]), _pkt([1, 0, 0], cid='c2', tid=2), _frame(), observation_id=5)
            self.assertFalse(eval_res['required'])
            self.assertEqual(eval_res['status'], 'unknown')
            self.assertEqual(eval_res['reason'], 'handled_by_existing_geometry')
            self.assertNotIn('signed_separation_m', eval_res)

    def test_evaluate_signs_frozen_frame_invariance_and_rotated_axis(self):
        spec_left = parse_endpoint_relation('doorway to the left of the white statue of two people')
        frozen = _frame(right_axis=[1.0, 0.0, 0.0], obs_id=1)
        target = _pkt([-0.4, 2.0, 0.0], obs_id=5, cid='door_1', tid=10, cls='door')
        landmark = _pkt([0.2, 2.0, 0.0], obs_id=5, cid='statue_1', tid=11, cls='statue')

        res1 = evaluate_spatial_relation(spec_left, target, landmark, frozen, observation_id=5)
        self.assertEqual(res1['status'], 'satisfied')
        self.assertAlmostEqual(res1['signed_separation_m'], -0.6)
        self.assertEqual(res1['frame_observation_id'], 1)
        self.assertNotIn('outcome', json.dumps(res1))

        target_later = {**target, 'observation_id': 9, 'view': 'rotated_camera'}
        landmark_later = {**landmark, 'observation_id': 9, 'view': 'rotated_camera'}
        res2 = evaluate_spatial_relation(spec_left, target_later, landmark_later, frozen, observation_id=9)
        self.assertEqual(res2['status'], 'satisfied')
        self.assertAlmostEqual(res2['signed_separation_m'], res1['signed_separation_m'])

        rotated_frame = _frame(right_axis=[0.0, 1.0, 0.0], obs_id=3)
        t_rot = _pkt([2.0, 0.8, 0.0], obs_id=5, cid='door_1', tid=10, cls='door')
        l_rot = _pkt([2.0, 0.2, 0.0], obs_id=5, cid='cab_1', tid=11, cls='cabinet')
        spec_right = parse_endpoint_relation('door on the right side of the cabinet')
        res_rot = evaluate_spatial_relation(spec_right, t_rot, l_rot, rotated_frame, observation_id=5)
        self.assertEqual(res_rot['status'], 'satisfied')
        self.assertAlmostEqual(res_rot['signed_separation_m'], 0.6)

        contra = evaluate_spatial_relation(
            spec_left, t_rot, {**l_rot, 'selected_detector_class_name': 'statue'}, rotated_frame, observation_id=5,
        )
        self.assertEqual(contra['status'], 'contradicted')
        self.assertEqual(contra['reason'], 'position_relation_contradicted')

    def test_fail_closed_validation_and_ambiguous_inputs(self):
        spec = parse_endpoint_relation('door on the right side of the cabinet')
        f = _frame()
        t = _pkt([0.5, 1.0, 0.0], obs_id=5, cid='c1', tid=1, cls='door')
        l = _pkt([0.0, 1.0, 0.0], obs_id=5, cid='c2', tid=2, cls='cabinet')

        close_t = _pkt([0.08, 1.0, 0.0], obs_id=5, cid='c1', tid=1, cls='door')
        self.assertEqual(evaluate_spatial_relation(spec, close_t, l, f, observation_id=5)['status'], 'unknown')
        self.assertEqual(evaluate_spatial_relation(spec, {**t, 'observation_id': 4}, l, f, observation_id=5)['status'], 'unknown')
        self.assertEqual(evaluate_spatial_relation(spec, {**t, 'world_center': [float('nan'), 0.0, 0.0]}, l, f, observation_id=5)['status'], 'unknown')
        self.assertEqual(evaluate_spatial_relation(spec, {**t, 'world_center': [10**1000, 0.0, 0.0]}, l, f, observation_id=5)['status'], 'unknown')
        self.assertEqual(evaluate_spatial_relation(spec, {**t, 'world_center': [True, 0.0, 0.0]}, l, f, observation_id=5)['status'], 'unknown')
        self.assertEqual(evaluate_spatial_relation(spec, {**t, 'world_center': None}, l, f, observation_id=5)['status'], 'unknown')
        self.assertEqual(evaluate_spatial_relation(spec, t, l, None, observation_id=5)['status'], 'unknown')
        self.assertEqual(evaluate_spatial_relation(spec, t, l, _frame(source='model_guess'), observation_id=5)['status'], 'unknown')
        self.assertEqual(evaluate_spatial_relation(spec, t, {**l, 'candidate_id': 'c1'}, f, observation_id=5)['status'], 'unknown')
        self.assertEqual(evaluate_spatial_relation(spec, t, {**l, 'target_id': 1}, f, observation_id=5)['status'], 'unknown')

        no_id_l = _pkt([0.0, 1.0, 0.0], obs_id=5, cid='c2', tid=2, cls=None)
        self.assertEqual(evaluate_spatial_relation(spec, t, no_id_l, f, observation_id=5)['status'], 'unknown')
        exp_only_l = {**no_id_l, 'expected_object_class': 'cabinet', 'selected_detector_class_name': None}
        self.assertEqual(evaluate_spatial_relation(spec, t, exp_only_l, f, observation_id=5)['status'], 'unknown')
        wrong_det_l = {**l, 'expected_object_class': 'cabinet', 'selected_detector_class_name': 'sofa'}
        self.assertEqual(evaluate_spatial_relation(spec, t, wrong_det_l, f, observation_id=5)['status'], 'unknown')
        spec_statue = parse_endpoint_relation('doorway to the left of the white statue of two people')
        wrong_statue_l = _pkt([1.0, 1.0, 0.0], obs_id=5, cid='c2', tid=2, cls='sofa')
        wrong_statue_l['expected_object_class'] = 'statue'
        self.assertEqual(evaluate_spatial_relation(spec_statue, t, wrong_statue_l, f, observation_id=5)['status'], 'unknown')
        conflict_cat_l = {**l, 'category': 'sofa'}
        self.assertEqual(evaluate_spatial_relation(spec, t, conflict_cat_l, f, observation_id=5)['status'], 'unknown')
        free_box_l = {**l, 'candidate_id': None, 'target_id': None}
        f_with_ref = {**f, 'reference_class_name': 'cabinet', 'reference_world_center': [0.0, 1.0, 0.0]}
        self.assertEqual(evaluate_spatial_relation(spec, t, free_box_l, f_with_ref, observation_id=5)['status'], 'unknown')
        entity_only_l = {**l, 'candidate_id': None, 'target_id': 2}
        self.assertEqual(evaluate_spatial_relation(spec, t, entity_only_l, f, observation_id=5)['status'], 'satisfied')


import ast
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import numpy as np

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
from semantic_evidence import (canonical_label, detection_observation,
                               add_entity_observation, merge_entity_evidence)


def extract(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return {node.name: namespace[node.name] for node in nodes}


class Array:
    def __init__(self, value):
        self.value = np.asarray(value)

    @property
    def shape(self):
        return self.value.shape

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def mean(self, dim=0):
        return Array(self.value.mean(axis=dim))

    def __lt__(self, other):
        return Array(self.value < other)

    def logical_not(self):
        return Array(~self.value)

    def sum(self):
        return Array(self.value.sum())

    def __gt__(self, other):
        return bool(self.value > other)

    def nonzero(self):
        return [np.nonzero(self.value)[0]]


class Cloud:
    def __init__(self, points=()):
        pts = np.asarray(points, dtype=float).reshape((-1, 3))
        self.point = NS(positions=Array(pts))

    def is_empty(self):
        return self.point.positions.shape[0] == 0

    def voxel_down_sample(self, resolution):
        return self

    def select_by_index(self, indices):
        return Cloud(self.point.positions.numpy()[np.asarray(indices)])


def fake_o3d():
    return NS(
        core=NS(Tensor=lambda arr, device=None: Array(arr)),
        t=NS(geometry=NS(PointCloud=lambda device=None: Cloud())),
    )


def pcd_distance(eval_pcd, ref_pcd):
    a = eval_pcd.point.positions.numpy()
    b = ref_pcd.point.positions.numpy()
    dists = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2).min(axis=1)
    return Array(dists)


def merge_pcd(a, b):
    if a.is_empty():
        return Cloud(b.point.positions.numpy())
    if b.is_empty():
        return Cloud(a.point.positions.numpy())
    return Cloud(np.vstack([a.point.positions.numpy(), b.point.positions.numpy()]))


class VisualCandidateMappingTests(unittest.TestCase):
    def _make_mapper_methods(self, segment_fn=None, point_fn=None):
        if point_fn is None:
            point_fn = lambda d, m, k: np.tile([1.5, 0.0, -0.2], (int(np.sum(m > 0)), 1))
        namespace = dict(
            np=np, canonical_label=canonical_label, detection_observation=detection_observation,
            add_entity_observation=add_entity_observation, merge_entity_evidence=merge_entity_evidence,
            get_pointcloud_from_depth_mask=point_fn, translate_to_world=lambda p, t, r: p + t,
            get_class_color=lambda c: (1, 2, 3), gpu_pointcloud_from_array=lambda p, c, dev: Cloud(p),
            gpu_cluster_filter=lambda p: p, pointcloud_distance=pcd_distance,
            gpu_merge_pointcloud=merge_pcd, o3d=fake_o3d(),
            instance_segmentation=segment_fn or (lambda rgb: []),
        )
        names = ['_begin_observation', '_update_semantic_views', 'get_object_entities', 'associate_object_entities']
        return extract(SRC / 'mapper.py', names, namespace)

    def test_zero_depth_keeps_2d_candidate_without_3d_entity_or_fake_distance(self):
        seen_masks = []
        def backproject(depth, mask, intrinsic):
            seen_masks.append(np.asarray(mask, dtype=bool).copy())
            return np.tile([1.8, 0.0, -0.3], (int(np.sum(mask)), 1))

        methods = self._make_mapper_methods(point_fn=backproject)
        mapper = NS(ceiling_height=1.0, camera_intrinsic=np.eye(3), current_position=np.zeros(3),
                    current_rotation=np.eye(3), pcd_device='cpu', pcd_resolution=0.05,
                    observation_id=4, observation_step=9)
        mask = np.zeros((4, 6), dtype=np.uint8)
        mask[1:3, 2:5] = 1
        entities = methods['get_object_entities'](
            mapper, np.zeros((4, 6), dtype=float), [10], ['staircase'], [mask], [0.82], view='left')
        self.assertEqual(entities, [])
        self.assertEqual(seen_masks, [])
        cid = '4:left:0'
        self.assertIn(cid, mapper.current_visual_candidates)
        cand = mapper.current_visual_candidates[cid]
        self.assertEqual(cand['candidate_id'], cid)
        self.assertEqual(cand['canonical_label'], 'stairs')
        self.assertIsNone(cand['entity_id'])
        self.assertEqual(cand['geometry_status'], 'unknown')
        self.assertEqual(cand['geometry_reason'], 'insufficient_depth')
        self.assertEqual(cand['valid_depth_fraction'], 0.0)
        self.assertIsNone(cand['horizontal_distance_m'])
        self.assertEqual(cand['bbox'], [2 / 6, 1 / 4, 5 / 6, 3 / 4])
        self.assertEqual(cand['image_size'], [6, 4])
        np.testing.assert_array_equal(mapper.current_observation_masks[cid], mask.astype(bool))

        mixed_depth = np.zeros((4, 6), dtype=float)
        mixed_depth[1:3, 2:5] = 2.0
        mixed_depth[1, 2] = 0.0
        full_mask = np.ones((4, 6), dtype=np.uint8)
        entities = methods['get_object_entities'](
            mapper, mixed_depth, [10], ['staircase'], [full_mask], [0.85], view='front')
        self.assertEqual(len(entities), 0)
        self.assertEqual(mapper.current_visual_candidates['4:front:0']['geometry_reason'], 'insufficient_points')
        self.assertEqual(int(seen_masks[0].sum()), 5)
        np.testing.assert_array_equal(mapper.current_observation_masks['4:front:0'], full_mask.astype(bool))

    def test_invalid_mask_and_depth_shape_mismatch_fail_closed(self):
        methods = self._make_mapper_methods()
        mapper = NS(ceiling_height=1.0, camera_intrinsic=np.eye(3), current_position=np.zeros(3),
                    current_rotation=np.eye(3), pcd_device='cpu', pcd_resolution=0.05,
                    observation_id=2, observation_step=5)
        empty_mask = np.zeros((4, 6), dtype=np.uint8)
        res_empty = methods['get_object_entities'](
            mapper, np.ones((4, 6)), [1], ['chair'], [empty_mask], [0.9], view='front')
        res_mismatch = methods['get_object_entities'](
            mapper, np.ones((2, 2)), [1], ['chair'], [np.ones((4, 6))], [0.9], view='front')
        res_nan_conf = methods['get_object_entities'](
            mapper, np.ones((4, 6)), [1], ['chair'], [np.ones((4, 6), dtype=np.uint8)], [float('nan')], view='front')
        self.assertEqual(res_empty, [])
        self.assertEqual(res_mismatch, [])
        self.assertEqual(res_nan_conf, [])
        self.assertEqual(mapper.current_visual_candidates, {})
        self.assertEqual(mapper.current_observation_masks, {})

        nan_methods = self._make_mapper_methods(
            point_fn=lambda d, m, k: np.full((int(np.sum(m > 0)), 3), np.nan))
        res_bad_proj = nan_methods['get_object_entities'](
            mapper, np.full((4, 6), 2.0), [1], ['chair'], [np.ones((4, 6), dtype=np.uint8)], [0.9], view='front')
        self.assertEqual(res_bad_proj, [])
        bad_cand = mapper.current_visual_candidates['2:front:0']
        self.assertEqual(bad_cand['geometry_status'], 'unknown')
        self.assertEqual(bad_cand['geometry_reason'], 'invalid_projection')
        self.assertIsNone(bad_cand['horizontal_distance_m'])
        self.assertIsNone(bad_cand['entity_id'])

    def test_multiview_linking_exact_mask_key_and_reset(self):
        mask = np.ones((4, 5), dtype=np.uint8)
        detections = {
            'left': [{'class_id': 1, 'class_name': 'sculpture', 'mask': mask, 'conf': 0.7}],
            'right': [{'class_id': 2, 'class_name': 'chair', 'mask': mask, 'conf': 0.8}],
            'back': [],
            'front': [{'class_id': 1, 'class_name': 'statue', 'mask': mask, 'conf': 0.9}],
        }
        methods = self._make_mapper_methods(segment_fn=lambda rgb: detections[rgb])
        Mapper = type('Mapper', (), methods)
        mapper = Mapper()
        mapper.ceiling_height = 1.0
        mapper.camera_intrinsic = np.eye(3)
        mapper.current_position = np.zeros(3)
        mapper.current_rotation = np.eye(3)
        mapper.pcd_device = 'cpu'
        mapper.pcd_resolution = 0.05
        mapper.current_pcd = 'orig'

        stale_entity = {'class': 1, 'class_name': 'statue', 'canonical_label': 'statue',
                        'pcd': Cloud(np.tile([5.0, 5.0, 0.0], (20, 1))), 'center': np.array([5.0, 5.0, 0.0])}
        add_entity_observation(stale_entity, {
            'observation_id': 1, 'step': 1, 'view': 'front', 'confidence': 0.9,
            'mask_key': '2:right:0', 'candidate_id': '2:right:0'
        })
        mapper.object_entities = [stale_entity]
        mapper.observation_id = 1
        mapper._begin_observation(10)
        self.assertEqual(mapper.observation_id, 2)
        self.assertEqual(mapper.current_visual_candidates, {})
        self.assertEqual(mapper.current_observation_masks, {})

        views = [
            (0, {'view': 'left', 'rgb': 'left', 'depth': np.full((4, 5), 1.5), 'pcd': 'l',
                 'position': np.zeros(3), 'rotation': np.eye(3)}),
            (1, {'view': 'right', 'rgb': 'right', 'depth': np.zeros((4, 5)), 'pcd': 'r',
                 'position': np.zeros(3), 'rotation': np.eye(3)}),
            (2, {'view': 'back', 'rgb': 'back', 'depth': np.full((4, 5), 1.5), 'pcd': 'b',
                 'position': np.zeros(3), 'rotation': np.eye(3)}),
            (3, {'view': 'front', 'rgb': 'front', 'depth': np.full((4, 5), 1.5), 'pcd': 'f',
                 'position': np.zeros(3), 'rotation': np.eye(3)}),
        ]
        mapper._update_semantic_views(views, 3)
        self.assertEqual(mapper.current_pcd, 'orig')
        self.assertEqual(set(mapper.current_visual_candidates), {'2:left:0', '2:right:0', '2:front:0'})
        self.assertEqual(len(mapper.object_entities), 2)
        self.assertEqual(mapper.current_visual_candidates['2:left:0']['entity_id'], 1)
        self.assertEqual(mapper.current_visual_candidates['2:front:0']['entity_id'], 1)
        self.assertIsNone(mapper.current_visual_candidates['2:right:0']['entity_id'])
        self.assertEqual(mapper.current_visual_candidates['2:front:0']['geometry_status'], 'available')
        self.assertAlmostEqual(mapper.current_visual_candidates['2:front:0']['horizontal_distance_m'], 1.5)

        mapper._begin_observation(11)
        self.assertEqual(mapper.current_visual_candidates, {})

    def test_order_invariant_overlap_and_tie_keeps_separate_without_point_loss(self):
        methods = self._make_mapper_methods()
        associate = methods['associate_object_entities']
        mapper = NS(pcd_device='cpu')

        xx, yy = np.meshgrid(np.arange(5) * 0.2, np.arange(4) * 0.2, indexing='ij')
        base_pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(20)])
        ref_a_pts = base_pts[:5].copy()
        ref_b_pts = base_pts[:8].copy()

        def make_ref(pts):
            return {'class': 1, 'class_name': 'chair', 'canonical_label': 'chair',
                    'pcd': Cloud(pts), 'center': pts.mean(axis=0)}

        def make_eval(pts=base_pts):
            ent = {'class': 1, 'class_name': 'chair', 'canonical_label': 'chair',
                   'pcd': Cloud(pts), 'center': pts.mean(axis=0)}
            add_entity_observation(ent, {
                'observation_id': 3, 'step': 4, 'view': 'front', 'confidence': 0.8, 'mask_key': '3:front:0'
            })
            return ent

        res_ab = associate(mapper, [make_ref(ref_a_pts), make_ref(ref_b_pts)], [make_eval()])
        res_ba = associate(mapper, [make_ref(ref_b_pts), make_ref(ref_a_pts)], [make_eval()])
        self.assertEqual(len(res_ab), 2)
        self.assertEqual(len(res_ba), 2)
        self.assertEqual(res_ab[0]['pcd'].point.positions.shape[0], 5)
        self.assertEqual(res_ab[1]['pcd'].point.positions.shape[0], 8 + 12)
        self.assertEqual(res_ba[0]['pcd'].point.positions.shape[0], 8 + 12)
        self.assertEqual(res_ba[1]['pcd'].point.positions.shape[0], 5)
        np.testing.assert_allclose(res_ab[0]['pcd'].point.positions.numpy(), ref_a_pts)
        np.testing.assert_allclose(res_ba[1]['pcd'].point.positions.numpy(), ref_a_pts)
        np.testing.assert_allclose(res_ab[1]['pcd'].point.positions.numpy(), base_pts)
        np.testing.assert_allclose(res_ba[0]['pcd'].point.positions.numpy(), base_pts)
        self.assertEqual(res_ab[1].get('last_seen_observation_id'), 3)
        self.assertEqual(res_ba[0].get('last_seen_observation_id'), 3)
        self.assertIsNone(res_ab[0].get('last_seen_observation_id'))

        tie_ref1 = make_ref(base_pts[:7].copy())
        tie_ref2 = make_ref(base_pts[2:9].copy())
        eval_tie = make_eval()
        res_tie = associate(mapper, [tie_ref1, tie_ref2], [eval_tie])
        self.assertEqual(len(res_tie), 3)
        self.assertEqual(res_tie[0]['pcd'].point.positions.shape[0], 7)
        self.assertEqual(res_tie[1]['pcd'].point.positions.shape[0], 7)
        self.assertIs(res_tie[2], eval_tie)
        self.assertEqual(res_tie[2]['pcd'].point.positions.shape[0], 20)

        xx25, yy25 = np.meshgrid(np.arange(5) * 0.2, np.arange(5) * 0.2, indexing='ij')
        pts25 = np.column_stack([xx25.ravel(), yy25.ravel(), np.zeros(25)])
        near_ref1 = make_ref(pts25[:8].copy())
        near_ref2 = make_ref(pts25[8:17].copy())
        eval_near = make_eval(pts25)
        res_near = associate(mapper, [near_ref1, near_ref2], [eval_near])
        self.assertEqual(len(res_near), 3)
        self.assertEqual(res_near[0]['pcd'].point.positions.shape[0], 8)
        self.assertEqual(res_near[1]['pcd'].point.positions.shape[0], 9)
        self.assertIs(res_near[2], eval_near)
        self.assertEqual(res_near[2]['pcd'].point.positions.shape[0], 25)

        distinct_pts = base_pts + np.array([0.0, 0.0, 0.25])
        distinct_eval = make_eval(distinct_pts)
        res_distinct = associate(mapper, [make_ref(ref_a_pts)], [distinct_eval])
        self.assertEqual(len(res_distinct), 2)
        self.assertEqual(res_distinct[0]['pcd'].point.positions.shape[0], 5)
        self.assertIs(res_distinct[1], distinct_eval)
        self.assertEqual(res_distinct[1]['pcd'].point.positions.shape[0], 20)

        small_pts = base_pts[:12] + np.array([0.0, 0.0, 0.25])
        small_eval = {'class': 1, 'class_name': 'chair', 'canonical_label': 'chair',
                      'pcd': Cloud(small_pts), 'center': small_pts.mean(axis=0)}
        res_small = associate(mapper, [make_ref(ref_a_pts)], [small_eval])
        self.assertEqual(len(res_small), 1)


if __name__ == '__main__':
    unittest.main()


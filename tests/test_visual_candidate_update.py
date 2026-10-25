import ast
from pathlib import Path
import sys
import time
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

    def __getitem__(self, item):
        return Array(self.value[item])

    def __lt__(self, other):
        other_val = other.value if isinstance(other, Array) else other
        return Array(self.value < other_val)

    def __le__(self, other):
        other_val = other.value if isinstance(other, Array) else other
        return Array(self.value <= other_val)

    def __gt__(self, other):
        other_val = other.value if isinstance(other, Array) else other
        return Array(self.value > other_val)

    def __ge__(self, other):
        other_val = other.value if isinstance(other, Array) else other
        return Array(self.value >= other_val)

    def __and__(self, other):
        other_val = other.value if isinstance(other, Array) else other
        return Array(self.value & other_val)

    def __or__(self, other):
        other_val = other.value if isinstance(other, Array) else other
        return Array(self.value | other_val)

    def __bool__(self):
        return bool(np.any(self.value))

    def logical_not(self):
        return Array(~self.value)

    def sum(self):
        return Array(self.value.sum())

    def nonzero(self):
        return [np.nonzero(self.value)[0]]


class Cloud:
    def __init__(self, points=(), device='cpu'):
        if isinstance(points, str) or points is None:
            pts = np.zeros((0, 3), dtype=float)
            device = points if isinstance(points, str) else device
        else:
            pts = np.asarray(points, dtype=float).reshape((-1, 3))
        self.point = NS(positions=Array(pts))
        self.device = device

    def is_empty(self):
        return self.point.positions.shape[0] == 0

    def voxel_down_sample(self, resolution):
        return self

    def select_by_index(self, indices):
        idx = indices.numpy() if isinstance(indices, Array) else np.asarray(indices)
        return Cloud(self.point.positions.numpy()[idx], device=self.device)

    def clone(self):
        return Cloud(self.point.positions.numpy().copy(), device=self.device)

    def clear(self):
        self.point = NS(positions=Array(np.zeros((0, 3), dtype=float)))


def fake_o3d():
    return NS(
        core=NS(
            Tensor=lambda arr, dtype=None, device=None: Array(arr),
            Dtype=NS(Int64='int64'),
        ),
        t=NS(geometry=NS(PointCloud=Cloud)),
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


def make_rgb(tag, shape=(6, 8, 3)):
    rgb = np.zeros(shape, dtype=np.uint8)
    rgb[0, 0, 0] = int(tag)
    return rgb


class VisualCandidateUpdateChainTests(unittest.TestCase):
    def _make_mapper(self, segment_fn=None, depth_pcd_fn=None, mask_pcd_fn=None):
        depth_calls = []
        mask_calls = []

        def default_depth_pcd(rgb, depth, intrinsic):
            depth_calls.append(np.asarray(depth).copy())
            valid = np.isfinite(depth) & (depth > 0)
            count = max(int(np.count_nonzero(valid)), 1)
            pts = np.tile([1.6, 0.0, 0.0], (count, 1))
            cols = np.zeros((count, 3), dtype=float)
            return pts, cols

        def default_mask_pcd(depth, mask, intrinsic):
            mask_calls.append((np.asarray(depth).copy(), np.asarray(mask, dtype=bool).copy()))
            count = int(np.sum(mask > 0))
            d_vals = np.asarray(depth)[np.asarray(mask, dtype=bool)]
            dist = float(np.median(d_vals)) if d_vals.size > 0 else 1.6
            return np.tile([dist, 0.0, 0.0], (count, 1))

        namespace = dict(
            time=time,
            np=np,
            o3d=fake_o3d(),
            canonical_label=canonical_label,
            detection_observation=detection_observation,
            add_entity_observation=add_entity_observation,
            merge_entity_evidence=merge_entity_evidence,
            preprocess_depth=lambda d: np.asarray(d, dtype=float),
            preprocess_image=lambda img: np.asarray(img),
            get_pointcloud_from_depth=depth_pcd_fn or default_depth_pcd,
            get_pointcloud_from_depth_mask=mask_pcd_fn or default_mask_pcd,
            translate_to_world=lambda p, t, r: np.asarray(p, dtype=float) + np.asarray(t, dtype=float),
            get_class_color=lambda c: (1, 2, 3),
            gpu_pointcloud_from_array=lambda p, c, dev: Cloud(p, device=dev),
            gpu_cluster_filter=lambda p: p,
            pointcloud_distance=pcd_distance,
            gpu_merge_pointcloud=merge_pcd,
            instance_segmentation=segment_fn or (lambda rgb: []),
        )
        names = [
            '_begin_observation',
            '_update_semantic_views',
            'update_multiview',
            'update',
            'get_object_entities',
            'associate_object_entities',
        ]
        methods = extract(SRC / 'mapper.py', names, namespace)
        Mapper = type('Mapper', (), methods)
        mapper = Mapper()
        mapper.translation_func = lambda p: np.asarray(p, dtype=float)
        mapper.rotation_func = lambda r: np.asarray(r, dtype=float)
        mapper.initial_position = np.zeros(3, dtype=float)
        mapper.initial_floor_height = 0.0
        mapper._detect_and_update_floor = lambda: None
        mapper.trajectory_position = []
        mapper.pcd_device = 'cpu'
        mapper.pcd_resolution = 0.05
        mapper.grid_resolution = 0.1
        mapper.camera_intrinsic = np.eye(3)
        mapper.floor_height = 0.0
        mapper.ceiling_height = 2.0
        mapper.scene_pcd = Cloud()
        mapper.useful_pcd = Cloud()
        mapper.navigable_pcd = Cloud()
        mapper.current_view_navigable_pcd = Cloud()
        mapper.obstacle_pcd = Cloud()
        mapper.stair_pcd = Cloud()
        mapper.trajectory_pcd = Cloud()
        mapper.current_pcd = Cloud()
        mapper.enable_stair_detection = False
        mapper.local_stair_radius = 2.5
        mapper.local_stair_min_points = 20
        mapper.update_iterations = 0
        mapper.release_vram = lambda: None
        mapper.object_entities = []
        mapper.observation_id = 0
        mapper.observation_step = None
        mapper.current_observation_masks = {}
        mapper.current_visual_candidates = {}
        return mapper, depth_calls, mask_calls

    def test_update_multiview_all_four_zero_depth_preserves_2d_candidates_and_clears_geometry(self):
        mask = np.zeros((6, 8), dtype=np.uint8)
        mask[1:5, 2:7] = 1
        view_ByTag = {
            0: [{'class_id': 10, 'class_name': 'staircase', 'mask': mask, 'conf': 0.81}],
            1: [{'class_id': 11, 'class_name': 'chair', 'mask': mask, 'conf': 0.76}],
            2: [{'class_id': 12, 'class_name': 'bed', 'mask': mask, 'conf': 0.79}],
            3: [{'class_id': 13, 'class_name': 'sofa', 'mask': mask, 'conf': 0.88}],
        }
        predict_calls = []

        def segment_fn(rgb):
            tag = int(rgb[0, 0, 0])
            predict_calls.append(tag)
            return view_ByTag[tag]

        mapper, depth_calls, mask_calls = self._make_mapper(segment_fn=segment_fn)
        mapper.current_pcd = Cloud([[1.0, 2.0, 0.0]])
        mapper.current_view_navigable_pcd = Cloud([[1.0, 2.0, 0.0]])

        views = [
            {'view': 'left', 'rgb': make_rgb(0), 'depth': np.zeros((6, 8), dtype=float),
             'position': np.array([0.0, 0.0, 0.0]), 'rotation': np.eye(3)},
            {'view': 'right', 'rgb': make_rgb(1), 'depth': np.zeros((6, 8), dtype=float),
             'position': np.array([0.0, 0.0, 0.0]), 'rotation': np.eye(3)},
            {'view': 'back', 'rgb': make_rgb(2), 'depth': np.zeros((6, 8), dtype=float),
             'position': np.array([0.0, 0.0, 0.0]), 'rotation': np.eye(3)},
            {'view': 'front', 'rgb': make_rgb(3), 'depth': np.zeros((6, 8), dtype=float),
             'position': np.array([1.0, 2.0, 0.0]), 'rotation': np.eye(3)},
        ]

        result = mapper.update_multiview(views, observation_step=7)
        self.assertEqual(predict_calls, [0, 1, 2, 3])
        self.assertEqual(len(depth_calls), 0)
        self.assertEqual(len(mask_calls), 0)
        self.assertEqual(result, view_ByTag[3])
        self.assertEqual(mapper.observation_id, 1)
        self.assertEqual(mapper.observation_step, 7)
        self.assertEqual(int(mapper.current_rgb[0, 0, 0]), 3)
        np.testing.assert_allclose(mapper.current_position, np.array([1.0, 2.0, 0.0]))
        self.assertTrue(mapper.current_pcd.is_empty())
        self.assertTrue(mapper.current_view_navigable_pcd.is_empty())
        self.assertTrue(mapper.navigable_pcd.is_empty())
        self.assertEqual(mapper.object_entities, [])

        expected_keys = {'1:left:0', '1:right:0', '1:back:0', '1:front:0'}
        self.assertEqual(set(mapper.current_visual_candidates.keys()), expected_keys)
        self.assertEqual(set(mapper.current_observation_masks.keys()), expected_keys)
        for key, expected_view, expected_label in (
            ('1:left:0', 'left', 'stairs'),
            ('1:right:0', 'right', 'chair'),
            ('1:back:0', 'back', 'bed'),
            ('1:front:0', 'front', 'sofa'),
        ):
            cand = mapper.current_visual_candidates[key]
            self.assertEqual(cand['view'], expected_view)
            self.assertEqual(cand['canonical_label'], expected_label)
            self.assertEqual(cand['observation_id'], 1)
            self.assertEqual(cand['step'], 7)
            self.assertIsNone(cand['entity_id'])
            self.assertEqual(cand['geometry_status'], 'unknown')
            self.assertEqual(cand['geometry_reason'], 'insufficient_depth')
            self.assertEqual(cand['valid_depth_fraction'], 0.0)
            self.assertIsNone(cand['horizontal_distance_m'])
            np.testing.assert_array_equal(mapper.current_observation_masks[key], mask.astype(bool))

    def test_update_multiview_mixed_zero_and_valid_depth_retains_zero_view_and_links_valid(self):
        mask = np.zeros((6, 8), dtype=np.uint8)
        mask[1:5, 2:7] = 1
        view_ByTag = {
            0: [{'class_id': 1, 'class_name': 'statue', 'mask': mask, 'conf': 0.85}],
            1: [{'class_id': 2, 'class_name': 'staircase', 'mask': mask, 'conf': 0.90}],
            2: [],
            3: [{'class_id': 1, 'class_name': 'sculpture', 'mask': mask, 'conf': 0.92}],
        }
        predict_calls = []

        def segment_fn(rgb):
            tag = int(rgb[0, 0, 0])
            predict_calls.append(tag)
            return view_ByTag[tag]

        mapper, depth_calls, mask_calls = self._make_mapper(segment_fn=segment_fn)
        views = [
            {'view': 'left', 'rgb': make_rgb(0), 'depth': np.full((6, 8), 1.8, dtype=float),
             'position': np.zeros(3), 'rotation': np.eye(3)},
            {'view': 'right', 'rgb': make_rgb(1), 'depth': np.zeros((6, 8), dtype=float),
             'position': np.zeros(3), 'rotation': np.eye(3)},
            {'view': 'back', 'rgb': make_rgb(2), 'depth': np.full((6, 8), 1.8, dtype=float),
             'position': np.zeros(3), 'rotation': np.eye(3)},
            {'view': 'front', 'rgb': make_rgb(3), 'depth': np.full((6, 8), 1.8, dtype=float),
             'position': np.zeros(3), 'rotation': np.eye(3)},
        ]

        result = mapper.update_multiview(views, observation_step=12)
        self.assertEqual(predict_calls, [0, 1, 2, 3])
        self.assertEqual(len(depth_calls), 3)
        self.assertEqual(len(mask_calls), 2)
        self.assertEqual(result, view_ByTag[3])
        self.assertEqual(mapper.observation_id, 1)
        self.assertEqual(set(mapper.current_visual_candidates.keys()), {'1:left:0', '1:right:0', '1:front:0'})

        zero_cand = mapper.current_visual_candidates['1:right:0']
        self.assertEqual(zero_cand['view'], 'right')
        self.assertEqual(zero_cand['observation_id'], 1)
        self.assertEqual(zero_cand['canonical_label'], 'stairs')
        self.assertEqual(zero_cand['geometry_status'], 'unknown')
        self.assertEqual(zero_cand['geometry_reason'], 'insufficient_depth')
        self.assertIsNone(zero_cand['horizontal_distance_m'])
        self.assertIsNone(zero_cand['entity_id'])

        self.assertEqual(len(mapper.object_entities), 1)
        self.assertEqual(mapper.current_visual_candidates['1:left:0']['entity_id'], 0)
        self.assertEqual(mapper.current_visual_candidates['1:front:0']['entity_id'], 0)
        self.assertEqual(mapper.current_visual_candidates['1:front:0']['geometry_status'], 'available')
        self.assertAlmostEqual(mapper.current_visual_candidates['1:front:0']['horizontal_distance_m'], 1.8)

    def test_update_multiview_nonfinite_and_negative_depth_creates_empty_cloud_and_keeps_2d_candidates(self):
        mask = np.ones((6, 8), dtype=np.uint8)
        mapper, depth_calls, mask_calls = self._make_mapper(
            segment_fn=lambda rgb: [{'class_id': 3, 'class_name': 'plant', 'mask': mask, 'conf': 0.8}]
        )
        views = [
            {'view': 'left', 'rgb': make_rgb(0), 'depth': np.full((6, 8), np.nan),
             'position': np.zeros(3), 'rotation': np.eye(3)},
            {'view': 'right', 'rgb': make_rgb(1), 'depth': np.full((6, 8), -1.0),
             'position': np.zeros(3), 'rotation': np.eye(3)},
            {'view': 'back', 'rgb': make_rgb(2), 'depth': np.full((6, 8), np.inf),
             'position': np.zeros(3), 'rotation': np.eye(3)},
            {'view': 'front', 'rgb': make_rgb(3), 'depth': np.full((6, 8), -np.inf),
             'position': np.zeros(3), 'rotation': np.eye(3)},
        ]
        result = mapper.update_multiview(views, observation_step=3)
        self.assertEqual(len(result), 1)
        self.assertEqual(len(depth_calls), 0)
        self.assertEqual(len(mask_calls), 0)
        self.assertTrue(mapper.current_pcd.is_empty())
        self.assertTrue(mapper.current_view_navigable_pcd.is_empty())
        self.assertEqual(len(mapper.current_visual_candidates), 4)
        for cand in mapper.current_visual_candidates.values():
            self.assertEqual(cand['geometry_status'], 'unknown')
            self.assertEqual(cand['geometry_reason'], 'insufficient_depth')
            self.assertIsNone(cand['horizontal_distance_m'])
            self.assertIsNone(cand['entity_id'])

    def test_single_update_all_zero_depth_preserves_2d_candidate_and_returns_instances(self):
        mask = np.zeros((6, 8), dtype=np.uint8)
        mask[1:5, 1:7] = 1
        expected_instances = [{'class_id': 5, 'class_name': 'toilet', 'mask': mask, 'conf': 0.87}]
        predict_calls = []

        def segment_fn(rgb):
            predict_calls.append(int(rgb[0, 0, 0]))
            return expected_instances

        mapper, depth_calls, mask_calls = self._make_mapper(segment_fn=segment_fn)
        mapper.current_pcd = Cloud([[2.0, 2.0, 0.0]])
        mapper.current_view_navigable_pcd = Cloud([[2.0, 2.0, 0.0]])

        result = mapper.update(
            make_rgb(9), np.zeros((6, 8), dtype=float),
            np.array([0.5, -0.5, 0.0]), np.eye(3), observation_step=4
        )
        self.assertEqual(predict_calls, [9])
        self.assertEqual(len(depth_calls), 0)
        self.assertEqual(len(mask_calls), 0)
        self.assertEqual(result, expected_instances)
        self.assertEqual(mapper.observation_id, 1)
        self.assertEqual(mapper.observation_step, 4)
        self.assertTrue(mapper.current_pcd.is_empty())
        self.assertTrue(mapper.current_view_navigable_pcd.is_empty())
        self.assertEqual(mapper.object_entities, [])
        self.assertIn('1:front:0', mapper.current_visual_candidates)
        cand = mapper.current_visual_candidates['1:front:0']
        self.assertEqual(cand['view'], 'front')
        self.assertEqual(cand['canonical_label'], 'toilet')
        self.assertEqual(cand['geometry_status'], 'unknown')
        self.assertEqual(cand['geometry_reason'], 'insufficient_depth')
        self.assertIsNone(cand['horizontal_distance_m'])
        self.assertIsNone(cand['entity_id'])
        np.testing.assert_array_equal(mapper.current_observation_masks['1:front:0'], mask.astype(bool))

    def test_single_update_valid_depth_links_candidate_entity_id_via_shared_semantic_helper(self):
        mask = np.zeros((6, 8), dtype=np.uint8)
        mask[1:5, 1:7] = 1
        expected_instances = [{'class_id': 7, 'class_name': 'armchair', 'mask': mask, 'conf': 0.91}]
        predict_calls = []

        def segment_fn(rgb):
            predict_calls.append(int(rgb[0, 0, 0]))
            return expected_instances

        mapper, depth_calls, mask_calls = self._make_mapper(segment_fn=segment_fn)
        result = mapper.update(
            make_rgb(4), np.full((6, 8), 2.2, dtype=float),
            np.zeros(3), np.eye(3), observation_step=8
        )
        self.assertEqual(predict_calls, [4])
        self.assertEqual(len(depth_calls), 1)
        self.assertEqual(len(mask_calls), 1)
        self.assertEqual(result, expected_instances)
        self.assertEqual(len(mapper.object_entities), 1)
        cand = mapper.current_visual_candidates['1:front:0']
        self.assertEqual(cand['entity_id'], 0)
        self.assertEqual(cand['geometry_status'], 'available')
        self.assertEqual(cand['geometry_reason'], 'available')
        self.assertAlmostEqual(cand['horizontal_distance_m'], 2.2)

    def test_do_instance_segmentation_false_skips_detector_and_candidates(self):
        mask = np.ones((6, 8), dtype=np.uint8)
        predict_calls = []

        def segment_fn(rgb):
            predict_calls.append(1)
            return [{'class_id': 1, 'class_name': 'chair', 'mask': mask, 'conf': 0.9}]

        mapper, _, _ = self._make_mapper(segment_fn=segment_fn)
        views = [
            {'view': 'left', 'rgb': make_rgb(0), 'depth': np.zeros((6, 8)), 'position': np.zeros(3), 'rotation': np.eye(3)},
            {'view': 'right', 'rgb': make_rgb(1), 'depth': np.zeros((6, 8)), 'position': np.zeros(3), 'rotation': np.eye(3)},
            {'view': 'back', 'rgb': make_rgb(2), 'depth': np.zeros((6, 8)), 'position': np.zeros(3), 'rotation': np.eye(3)},
            {'view': 'front', 'rgb': make_rgb(3), 'depth': np.zeros((6, 8)), 'position': np.zeros(3), 'rotation': np.eye(3)},
        ]
        res_mv = mapper.update_multiview(views, do_instance_segmentation=False, observation_step=1)
        res_single = mapper.update(
            make_rgb(3), np.zeros((6, 8)), np.zeros(3), np.eye(3),
            do_instance_segmentation=False, observation_step=2
        )
        self.assertEqual(res_mv, [])
        self.assertEqual(res_single, [])
        self.assertEqual(predict_calls, [])
        self.assertEqual(mapper.current_visual_candidates, {})
        self.assertEqual(mapper.current_observation_masks, {})

    def test_consecutive_observations_reset_candidates_and_do_not_leak_stale_entity_links(self):
        mask = np.zeros((6, 8), dtype=np.uint8)
        mask[1:5, 1:7] = 1
        mapper, _, _ = self._make_mapper(
            segment_fn=lambda rgb: [{'class_id': 1, 'class_name': 'chair', 'mask': mask, 'conf': 0.9}]
        )
        mapper.update(make_rgb(1), np.full((6, 8), 1.5), np.zeros(3), np.eye(3), observation_step=1)
        self.assertEqual(mapper.current_visual_candidates['1:front:0']['entity_id'], 0)

        mapper.update(make_rgb(2), np.zeros((6, 8)), np.zeros(3), np.eye(3), observation_step=2)
        self.assertEqual(set(mapper.current_visual_candidates.keys()), {'2:front:0'})
        self.assertIsNone(mapper.current_visual_candidates['2:front:0']['entity_id'])
        self.assertEqual(mapper.current_visual_candidates['2:front:0']['geometry_status'], 'unknown')


if __name__ == '__main__':
    unittest.main()


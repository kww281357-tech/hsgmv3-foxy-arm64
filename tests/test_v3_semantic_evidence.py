"""CPU perception contracts, using production methods without loading a model."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import numpy as np

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
from semantic_evidence import (canonical_label, rgb_to_bgr, detection_observation,
                               add_entity_observation, merge_entity_evidence, entity_observed_now)


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


class Cloud:
    def __init__(self, points=()):
        self.point = NS(positions=Array(np.asarray(points).reshape((-1, 3))))

    def is_empty(self):
        return self.point.positions.shape[0] == 0

    def voxel_down_sample(self, resolution):
        return self


def observation(revision=1, view='front', confidence=.8, step=3):
    return detection_observation(np.ones((4, 6)), np.ones((4, 6)), confidence,
                                 observation_id=revision, step=step, view=view, mask_key=f'{revision}:{view}:0')


class SemanticEvidenceTests(unittest.TestCase):
    def test_rgb_and_rgba_are_converted_once_without_mutating_source(self):
        for pixel in ([230, 40, 10], [230, 40, 10, 90]):
            source = np.array([[pixel]], dtype=np.uint8)
            before = source.copy()
            bgr = rgb_to_bgr(source)
            np.testing.assert_array_equal(bgr, [[[10, 40, 230]]])
            np.testing.assert_array_equal(source, before)
            self.assertTrue(bgr.flags.c_contiguous)
            # Ultralytics reverses its BGR input before inference.
            np.testing.assert_array_equal(bgr[:, :, ::-1], source[:, :, :3])

    def test_invalid_image_is_not_silently_reinterpreted(self):
        for bad in (np.zeros((4, 6)), np.zeros((4, 6, 2), dtype=np.uint8), np.zeros((4, 6, 3))):
            with self.assertRaises(ValueError):
                rgb_to_bgr(bad)

    def test_aliases_do_not_conflate_door_and_open_threshold(self):
        for name in ('stair', 'stairs', 'Staircase', 'staircases'):
            self.assertEqual(canonical_label(name), 'stairs')
        for name in ('statue', 'sculpture', 'statues', 'sculptures'):
            self.assertEqual(canonical_label(name), 'statue')
        self.assertNotEqual(canonical_label('door'), canonical_label('doorway'))

    def test_bbox_and_depth_are_from_mask_not_background(self):
        mask = np.zeros((4, 6), dtype=np.uint8)
        mask[1:3, 0:2] = 255
        depth = np.full((4, 6), 100.)
        depth[1:3, 0:2] = [[1., 3.], [0., np.nan]]
        row = detection_observation(mask, depth, .6, observation_id=2, step=9, view='left', mask_key='k')
        np.testing.assert_allclose(row['bbox'], [0, .25, 1/3, .75])
        self.assertEqual(row['image_size'], [6, 4])
        self.assertEqual(row['valid_depth_count'], 2)
        self.assertEqual(row['valid_depth_fraction'], .5)
        self.assertEqual(row['depth_median_m'], 2.)
        self.assertTrue(row['touches_image_edge'])

    def test_missing_depth_is_unknown_not_zero_distance(self):
        row = detection_observation(np.ones((2, 2)), np.zeros((2, 2)), .7,
                                    observation_id=2, step=9, view='back', mask_key='k')
        self.assertEqual(row['valid_depth_count'], 0)
        self.assertIsNone(row['depth_p10_m'])
        self.assertIsNone(row['depth_median_m'])

    def test_mask_and_depth_must_share_coordinates(self):
        with self.assertRaises(ValueError):
            detection_observation(np.ones((2, 2)), np.ones((3, 3)), .7,
                                  observation_id=2, step=9, view='back', mask_key='k')

    def test_multiple_views_do_not_count_as_multiple_temporal_observations(self):
        entity = {'class_name': 'stairs'}
        add_entity_observation(entity, observation(view='left'))
        add_entity_observation(entity, observation(view='front', confidence=.6))
        self.assertEqual(entity['observation_count'], 1)
        self.assertEqual(entity['detection_count'], 2)
        self.assertEqual(entity['last_seen_views'], ['front', 'left'])
        self.assertTrue(entity_observed_now(entity, 1))
        self.assertFalse(entity_observed_now(entity, 2))
        add_entity_observation(entity, observation(2, 'right', .5, step=4))
        self.assertEqual(entity['observation_count'], 2)
        self.assertEqual(entity['last_seen_views'], ['right'])
        self.assertEqual(entity['confidence'], .5)
        self.assertEqual(entity['max_confidence'], .8)
        self.assertEqual(entity['first_seen_step'], 3)
        self.assertEqual(entity['last_seen_step'], 4)

    def test_class_conflict_cannot_relabel_tracked_identity(self):
        old = {'class_name': 'chair'}
        add_entity_observation(old, observation())
        new = {'class_name': 'stairs'}
        add_entity_observation(new, observation(2))
        with self.assertRaises(ValueError):
            merge_entity_evidence(old, new)
        self.assertEqual(old['class_name'], 'chair')
        self.assertEqual(old['last_seen_observation_id'], 1)

    def test_alias_observation_can_refresh_existing_identity(self):
        old = {'class_name': 'staircase'}
        add_entity_observation(old, observation())
        new = {'class_name': 'stairs'}
        add_entity_observation(new, observation(2))
        merge_entity_evidence(old, new)
        self.assertEqual(old['observation_count'], 2)
        self.assertTrue(entity_observed_now(old, 2))
        self.assertFalse(entity_observed_now({}, 2))


class ProductionPerceptionTests(unittest.TestCase):
    def test_mapper_records_current_mask_depth_and_horizontal_distance(self):
        namespace = dict(np=np, canonical_label=canonical_label,
                         detection_observation=detection_observation, add_entity_observation=add_entity_observation,
                         get_pointcloud_from_depth_mask=lambda d, m, k: np.tile([1.3, 0, -.4], (24, 1)),
                         translate_to_world=lambda p, t, r: p + t, get_class_color=lambda c: (1, 2, 3),
                         gpu_pointcloud_from_array=lambda p, c, device: Cloud(p), gpu_cluster_filter=lambda p: p)
        build = extract(SRC/'mapper.py', ['get_object_entities'], namespace)['get_object_entities']
        mapper = NS(ceiling_height=1., camera_intrinsic=np.eye(3), current_position=np.zeros(3),
                    current_rotation=np.eye(3), pcd_device='cpu', pcd_resolution=.05,
                    observation_id=12, observation_step=27, current_observation_masks={})
        mask = np.ones((4, 6), dtype=np.uint8)
        result = build(mapper, np.full((4, 6), 1.4), [219], ['sculpture'], [mask], [.51], view='back')
        self.assertEqual(len(result), 1)
        entity = result[0]
        self.assertEqual(entity['class_name'], 'statue')
        self.assertTrue(entity_observed_now(entity, 12))
        row = entity['observations'][0]
        self.assertEqual((row['step'], row['view']), (27, 'back'))
        self.assertAlmostEqual(row['horizontal_distance_m'], 1.3)
        self.assertEqual(row['valid_depth_count'], 24)
        self.assertAlmostEqual(entity['confidence'], .51)
        np.testing.assert_array_equal(mapper.current_observation_masks[row['mask_key']], mask.astype(bool))

    def test_detector_keeps_edge_target_and_receives_bgr(self):
        source = np.zeros((20, 20, 3), dtype=np.uint8)
        source[:] = [230, 40, 10]
        mask = np.zeros((1, 20, 20))
        mask[:, :3, :2] = 1  # Old center-in-8%-margin rejection removed this.
        calls = []
        def predict(image, **kwargs):
            calls.append(image)
            return [NS(masks=NS(data=Array(mask)),
                       boxes=NS(cls=np.array([0]), conf=np.array([.51])), names={0: 'staircase'})]
        namespace = dict(np=np, model=NS(predict=predict), canonical_label=canonical_label,
                         rgb_to_bgr=rgb_to_bgr, cv2=NS(INTER_NEAREST=0, resize=lambda a, size, interpolation: a))
        segment = extract(SRC/'segmentation/instance_segmentation.py', ['instance_segmentation'], namespace)['instance_segmentation']
        detections = segment(source)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]['canonical_label'], 'stairs')
        np.testing.assert_array_equal(calls[0][0, 0], [10, 40, 230])

    def test_all_views_keep_endpoint_classes_and_actual_confidence(self):
        names = ['stairs', 'doorway', 'sculpture', 'carpet']
        detections = [[{'class_id': i, 'class_name': name, 'mask': np.ones((2, 2)), 'conf': .50}]
                      for i, name in enumerate(names)]
        calls = []
        def segment(rgb):
            calls.append(rgb)
            return detections[rgb]
        namespace = dict(instance_segmentation=segment)
        Mapper = type('Mapper', (), extract(SRC/'mapper.py', ['_update_semantic_views', '_begin_observation'], namespace))
        mapper = Mapper()
        mapper.current_pcd = original = object()
        mapper.camera_intrinsic = np.eye(3)
        mapper.object_entities = []
        received = []
        def entities(depth, classes, class_names, masks, confidences, **kwargs):
            received.append((class_names, confidences, kwargs['view']))
            return [dict(class_name=class_names[0])]
        mapper.get_object_entities = entities
        mapper.associate_object_entities = lambda old, new: old + new
        views = [(i, dict(view=v, rgb=i, depth=np.ones((2, 2)), pcd=object(),
                          position=np.zeros(3), rotation=np.eye(3)))
                 for i, v in enumerate(['left', 'right', 'back', 'front'])]
        result = mapper._update_semantic_views(views, 3)
        self.assertEqual(calls, [0, 1, 2, 3])
        self.assertEqual([r[0][0] for r in received], names)
        self.assertEqual([r[1][0] for r in received], [.5] * 4)
        self.assertEqual([r[2] for r in received], ['left', 'right', 'back', 'front'])
        self.assertIs(mapper.current_pcd, original)
        self.assertIs(result, detections[3])
        mapper.current_observation_masks = {'old': np.ones((2, 2))}
        mapper._begin_observation(17)
        self.assertEqual(mapper.observation_id, 1)
        self.assertEqual(mapper.observation_step, 17)
        self.assertEqual(mapper.current_observation_masks, {})

    def test_mapper_keeps_conflicting_class_as_separate_entity(self):
        namespace = dict(np=np, canonical_label=canonical_label)
        associate = extract(SRC/'mapper.py', ['associate_object_entities'], namespace)['associate_object_entities']
        cloud = NS(point=NS(positions=np.ones((30, 3))), is_empty=lambda: False)
        old = dict(class_name='chair', **{'class': 2}, center=np.zeros(3), pcd=cloud)
        new = dict(class_name='stairs', **{'class': 165}, center=np.ones(3)*.01, pcd=cloud)
        result = associate(NS(), [old], [new])
        self.assertEqual(len(result), 2)
        self.assertIs(result[0], old)
        self.assertEqual(result[0]['class_name'], 'chair')
        self.assertEqual(result[1]['class_name'], 'stairs')

    def test_mapper_alias_merge_refreshes_evidence_without_reassigning_id_class(self):
        namespace = dict(np=np, canonical_label=canonical_label, merge_entity_evidence=merge_entity_evidence,
                         pointcloud_distance=lambda a, b: Array(np.zeros(a.point.positions.shape[0])),
                         gpu_merge_pointcloud=lambda a, b: a if b.is_empty() else b,
                         o3d=NS(t=NS(geometry=NS(PointCloud=lambda device: Cloud()))))
        associate = extract(SRC/'mapper.py', ['associate_object_entities'], namespace)['associate_object_entities']
        cloud = Cloud(np.zeros((30, 3)))
        old = dict(class_name='staircase', **{'class': 237}, center=np.zeros(3), pcd=cloud)
        new = dict(class_name='stairs', **{'class': 165}, center=np.zeros(3), pcd=cloud)
        add_entity_observation(old, observation())
        add_entity_observation(new, observation(2, 'left', .6, 4))
        result = associate(NS(pcd_device='cpu'), [old], [new])
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], old)
        self.assertEqual(old['class'], 237)
        self.assertEqual(old['last_seen_views'], ['left'])
        self.assertTrue(entity_observed_now(old, 2))


if __name__ == '__main__':
    unittest.main()


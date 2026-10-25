"""Synthetic geometry/schema tests only; no Habitat run, scene claim, detector or API."""
from dataclasses import FrozenInstanceError, replace
import hashlib
import math
import unittest

from blue_help.manual_route import (AnchorSpec, COCO80_NAMES, DummyGeometry, FrameEvidence,
    GeometryConfig, GoalAnnotation, LandmarkEvidence, LocalConnector, Pose,
    RouteCard, build_route_card)


IMAGE_BYTES = b'synthetic evidence fixture, not a real scene image'
MANUAL_TEXT = ('SYNTHETIC TEST CARD ONLY. Walk forward, turn left, then right, then left, '
               'continue beyond the reviewed chair, and stop at the annotated fixture endpoint. '
               'This literal text is authored by the test, not generated from path geometry.')


def fields():
    frame = FrameEvidence('synthetic-scene', hashlib.sha256(IMAGE_BYTES).hexdigest(),
                          320, 240, 'synthetic-test-frame')
    landmark = LandmarkEvidence(frame.frame_id, 'chair', (10, 20, 80, 140),
                                'manual_annotation', 'Synthetic test review receipt')
    return dict(scene_id='synthetic-scene', geometry_revision='fixture-layout-and-navmesh-v1',
        dummy=DummyGeometry('dummy-fixture', (0, 0, 0)),
        anchor=AnchorSpec('exterior-fixture-anchor', Pose((0, 0, -.8), 0),
                          'Synthetic exterior anchor review receipt'),
        goal=GoalAnnotation('annotated-fixture-goal', (6, 0, -15.234), frame.frame_id,
                            'Synthetic endpoint annotation review receipt'),
        route_text=MANUAL_TEXT, route_review_evidence='Synthetic full-route manual review receipt',
        frames=(frame,), landmarks=(landmark,), mentioned_evidence_ids=(landmark.evidence_id,))


FULL_PATH = [(0, 0, -.8), (0, 0, -4), (3, 0, -4), (3, 0, -8), (6, 0, -8), (6, 0, -15.234)]


def card(**changes):
    args = fields()
    args.update(changes)
    return build_route_card(find_safe_path=lambda start, end: FULL_PATH, **args)


def request(connector, pose=None, **changes):
    args = dict(scene_id='synthetic-scene', geometry_revision='fixture-layout-and-navmesh-v1',
        dummy_id='dummy-fixture', observation_step=17, request_step=17,
        visible=True, line_of_sight_clear=True, same_floor=True)
    args.update(changes)
    return connector.answer_text(pose or Pose((.4, 0, -.8), 0), **args)


class EvidenceTests(unittest.TestCase):
    def test_coco80_exact_names_and_no_architecture_or_alias_invention(self):
        self.assertEqual(len(COCO80_NAMES), 80)
        original = fields()['landmarks'][0]
        for label in ('door', 'stairs', 'hallway', 'sofa', 'Chair', 'fake object'):
            with self.subTest(label=label), self.assertRaises(ValueError):
                replace(original, coco_label=label)
        self.assertEqual(replace(original, coco_label='couch').coco_label, 'couch')

    def test_frame_evidence_and_card_are_frozen_and_content_addressed(self):
        original = fields()
        frame, evidence = original['frames'][0], original['landmarks'][0]
        value = card()
        for obj, name, changed in ((frame, 'source_frame_label', 'changed'),
                (evidence, 'coco_label', 'couch'), (value, 'route_text', 'changed')):
            with self.subTest(name=name), self.assertRaises(FrozenInstanceError):
                setattr(obj, name, changed)
        self.assertNotEqual(replace(frame, source_frame_label='new-revision').frame_id, frame.frame_id)
        self.assertNotEqual(replace(evidence, coco_label='couch').evidence_id, evidence.evidence_id)
        self.assertNotEqual(replace(value, route_text=MANUAL_TEXT+' Reviewed change.').route_card_id,
                            value.route_card_id)
        self.assertEqual(card().route_card_id, value.route_card_id)
        self.assertIsInstance(value.private_full_path, tuple)
        self.assertTrue(all(isinstance(p, tuple) for p in value.private_full_path))

    def test_image_hash_is_byte_identity_not_an_object_detector(self):
        frame = fields()['frames'][0]
        self.assertTrue(frame.matches_image_bytes(IMAGE_BYTES))
        self.assertFalse(frame.matches_image_bytes(b'changed'))
        for digest in ('unknown', 'G'*64, 'a'*63):
            with self.assertRaises(ValueError):
                replace(frame, image_sha256=digest)

    def test_bbox_and_frame_scene_cross_references_fail_closed(self):
        args = fields()
        evidence = args['landmarks'][0]
        for changes in ({'frames': (replace(args['frames'][0], scene_id='wrong-scene'),)},
                {'landmarks': (replace(evidence, frame_id='unseen-frame'),)},
                {'landmarks': (replace(evidence, bbox_xyxy=(0, 0, 321, 240)),)},
                {'mentioned_evidence_ids': ('unprovided-object-evidence',)},
                {'frames': args['frames']*2}, {'landmarks': args['landmarks']*2},
                {'goal': replace(args['goal'], frame_id='missing-goal-frame')}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                card(**changes)

    def test_detector_provenance_must_be_explicit_and_scene_reviewed(self):
        original = fields()['landmarks'][0]
        with self.assertRaises(ValueError):
            replace(original, label_source='detector_output')
        with self.assertRaises(ValueError):
            replace(original, detector_id='model-that-was-not-used')
        with self.assertRaises(ValueError):
            replace(original, review_evidence='')
        detector = replace(original, label_source='detector_output', detector_id='fixture-detector-v1')
        self.assertEqual(detector.label_source, 'detector_output')
        output = request(LocalConnector(card(landmarks=(detector,),
            mentioned_evidence_ids=(detector.evidence_id,)), lambda a, b: [a, b]))
        self.assertIn('detector outputs that were manually scene-reviewed', output)
        self.assertIn('no detector is run now', output)


class OfflineCardTests(unittest.TestCase):
    def test_offline_origin_is_exterior_anchor_and_not_vehicle_or_dummy_foot(self):
        args = fields()
        calls = []
        def backend(start, goal):
            calls.append((start, goal))
            return FULL_PATH
        result = build_route_card(find_safe_path=backend, **args)
        self.assertEqual(calls, [(args['anchor'].pose.position, args['goal'].position)])
        self.assertNotEqual(calls[0][0], args['dummy'].foot)
        self.assertEqual(result.route_text, MANUAL_TEXT)

    def test_complete_long_route_has_no_eight_metre_or_three_leg_cap(self):
        result = card()
        length = sum(math.dist(a, b) for a, b in zip(result.private_full_path, result.private_full_path[1:]))
        self.assertGreater(length, 20.)
        self.assertEqual(len(result.private_full_path), 6)
        self.assertEqual(result.private_full_path[-1], result.goal.position)
        output = request(LocalConnector(result, lambda a, b: [a, b]))
        self.assertTrue(output.endswith(MANUAL_TEXT))

    def test_invalid_exterior_anchor_rejected_before_full_route_query(self):
        for point in ((0, 0, 0), (0, 0, -.45), (0, 0, -1.01), (0, .3, -.8)):
            calls = []
            args = fields()
            args['anchor'] = replace(args['anchor'], pose=Pose(point, 0))
            with self.subTest(point=point), self.assertRaises(ValueError):
                build_route_card(find_safe_path=lambda *a: calls.append(a), **args)
            self.assertEqual(calls, [])

    def test_no_route_empty_wrong_endpoint_nonfinite_and_human_crossing_rejected(self):
        bad_paths = [None, [], [(0, 0, -.8), (1, 0, 1)],
            [(0, 0, -.8), (math.nan, 0, 0), (6, 0, -15.234)],
            [(0, 0, -.8), (0, 0, .8), (6, 0, -15.234)]]
        for path in bad_paths:
            with self.subTest(path=path), self.assertRaises(ValueError):
                build_route_card(find_safe_path=lambda *args: path, **fields())

    def test_mutable_inputs_do_not_modify_frozen_records(self):
        points = [list(point) for point in FULL_PATH]
        result = build_route_card(find_safe_path=lambda *args: points, **fields())
        points[-1][0] = 999
        self.assertEqual(result.private_full_path[-1], result.goal.position)
        self.assertEqual(result.anchor.pose.position, (0., 0., -.8))


class ConnectorTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.route = card()
        def backend(a, b):
            self.calls.append((a, b))
            return [a, b]
        self.connector = LocalConnector(self.route, backend)

    def test_runtime_only_queries_current_vehicle_to_same_anchor_and_returns_str(self):
        output = request(self.connector)
        self.assertIsInstance(output, str)
        self.assertEqual(self.calls, [((.4, 0., -.8), self.route.anchor.pose.position)])
        self.assertIn('turn left about 90.0 degrees', output)
        self.assertIn('go forward about 0.400 meters', output)
        self.assertIn('turn right about 90.0 degrees', output)
        self.assertIn('manually annotated', output)
        self.assertNotIn('15.234', output)
        self.assertNotIn(self.route.route_card_id, output)
        self.assertNotIn(self.route.goal.goal_id, output)
        self.assertNotIn('private_full_path', self.connector.__dict__)
        self.assertNotIn('goal', self.connector.__dict__)
        self.assertFalse(any(isinstance(v, RouteCard) for v in self.connector.__dict__.values()))
        self.assertTrue(output.endswith(MANUAL_TEXT))

    def test_current_heading_changes_only_connector_not_manual_full_route(self):
        first = request(self.connector)
        second = request(self.connector, Pose((.4, 0, -.8), math.pi/2))
        self.assertNotEqual(first.split('FULL MANUAL ROUTE:\n')[0], second.split('FULL MANUAL ROUTE:\n')[0])
        self.assertEqual(first.split('FULL MANUAL ROUTE:\n')[1], MANUAL_TEXT)
        self.assertEqual(second.split('FULL MANUAL ROUTE:\n')[1], MANUAL_TEXT)

    def test_at_anchor_still_checks_safe_path_and_orients_before_verbatim_route(self):
        output = request(self.connector, Pose((0, 0, -.8), math.pi/2))
        self.assertEqual(self.calls, [((0., 0., -.8), (0., 0., -.8))])
        self.assertIn('turn right about 90.0 degrees', output)
        self.assertNotIn('go forward', output.split('FULL MANUAL ROUTE:\n')[0])
        self.assertIn('exterior anchor and aligned', request(self.connector, self.route.anchor.pose))

    def test_outside_unverified_other_floor_or_body_overlap_never_queries(self):
        for changes in ({'visible': False}, {'line_of_sight_clear': None}, {'same_floor': False},
                {'scene_id': 'wrong'}, {'geometry_revision': 'changed-layout'},
                {'dummy_id': 'other-dummy'}, {'observation_step': 18}, {'request_step': True}):
            with self.subTest(changes=changes):
                self.assertIsNone(request(self.connector, **changes))
        for pose in (Pose((0, 0, -1.001), 0), Pose((0, .3, -.8), 0), Pose((0, 0, -.3), 0)):
            self.assertIsNone(request(self.connector, pose))
        self.assertEqual(self.calls, [])

    def test_one_metre_boundary_inclusive_and_tight_bounds_not_expandable(self):
        self.assertIsNotNone(request(self.connector, Pose((0, 0, -1), 0)))
        for change in ({'interaction_radius_m': 1.001}, {'max_connector_length_m': 1.501},
                       {'body_margin_m': -.1}, {'robot_radius_m': True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                GeometryConfig(**change)

    def test_safe_connector_length_over_one_point_five_rejected_not_shortcut(self):
        calls = []
        def long_path(a, b):
            calls.append((a, b))
            return [a, (.4, 0, -2), (0, 0, -2), b]
        self.assertIsNone(request(LocalConnector(self.route, long_path)))
        self.assertEqual(len(calls), 1)

    def test_unavailable_empty_wrong_endpoint_service_error_and_body_crossing_fail_closed(self):
        raw_paths = [None, [], [(99, 0, 99)],
                     [(.4, 0, -.8), (0, 0, .8), (0, 0, -.8)],
                     [(.4, 0, -.8), (math.inf, 0, -.8), (0, 0, -.8)]]
        for path in raw_paths:
            with self.subTest(path=path):
                self.assertIsNone(request(LocalConnector(self.route, lambda *a: path)))
        def failure(*args):
            raise RuntimeError('fixture local path unavailable')
        self.assertIsNone(request(LocalConnector(self.route, failure)))

    def test_small_safe_curved_connector_keeps_all_corners(self):
        curved = lambda a, b: [a, (.4, 0, -1), (0, 0, -1), b]
        output = request(LocalConnector(self.route, curved))
        local = output.split('FULL MANUAL ROUTE:\n')[0]
        self.assertEqual(local.count('go forward about'), 3)
        self.assertEqual(local.count('turn left about 90.0 degrees'), 2)
        self.assertIn('180.0 degrees', local)
        self.assertTrue(output.endswith(MANUAL_TEXT))

    def test_actual_endpoint_is_kept_without_inventing_small_final_motion(self):
        near = lambda a, b: [a, (.005, 0, -.8)]
        output = request(LocalConnector(self.route, near))
        self.assertIn('go forward about 0.395 meters', output)
        self.assertIn('no extra unverified motion', output)
        self.assertNotIn('go forward about 0.400 meters', output)
        shifted_start = lambda a, b: [(.405, 0, -.8), b]
        self.assertIsNone(request(LocalConnector(self.route, shifted_start)))

    def test_config_allows_tighter_connector_but_no_hidden_full_route_cap(self):
        result = card(config=GeometryConfig(max_connector_length_m=.2))
        self.assertEqual(len(result.private_full_path), 6)
        self.assertIsNone(request(LocalConnector(result, lambda a, b: [a, b])))


if __name__ == '__main__':
    unittest.main()

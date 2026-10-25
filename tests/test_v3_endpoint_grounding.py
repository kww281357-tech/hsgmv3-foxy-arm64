"""CPU regressions for target-localised geometry, never image-centre STOPs."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from endpoint_grounding import (ground_target_depth, evaluate_endpoint_grounding,
                                consistent_target_observations)


K = np.array([[50., 0, 49.5], [0, 50., 49.5], [0, 0, 1.]])
# Camera x->world x, y_up->world z, -z_forward->world y.
R = np.array([[1., 0, 0], [0, 0, -1.], [0, 1., 0]])
BOX = [.35, .35, .65, .65]


def measure(depth=None, **kwargs):
    params = dict(target_view='front', target_bbox=BOX, observation_id=10,
                  observation_step=4, camera_intrinsic=K, camera_rotation=R,
                  camera_position=np.zeros(3))
    params.update(kwargs)
    return ground_target_depth({'front': np.full((100, 100), .8) if depth is None else depth}, **params)


def gate(packet, **kwargs):
    params = dict(current_observation_id=10, relation_satisfied=True, confirmed=True)
    params.update(kwargs)
    return evaluate_endpoint_grounding(packet, **params)


def mask_packet(**kwargs):
    mask = np.zeros((100, 100), bool)
    mask[35:65, 35:65] = True
    params = dict(target_mask=mask, mask_view='front', mask_observation_id=10, mask_confidence=.8)
    params.update(kwargs)
    return measure(**params)


class RegionDepthTests(unittest.TestCase):
    def test_large_segmented_object_and_real_depth_extent_are_not_missing(self):
        depth = np.tile(np.linspace(.5, 4., 100), (100, 1))
        p = measure(depth, target_mask=np.ones((100, 100), bool), target_bbox=[0, 0, 1, 1],
                    mask_view='front', mask_observation_id=10, mask_confidence=.9)
        self.assertTrue(p['valid'], p['reason'])
        self.assertGreater(p['depth_extent_m'], 1.)

    def test_localized_part_uses_only_corresponding_instance_mask_region(self):
        depth = np.full((100, 100), 4.)
        depth[40:50, 40:50] = .8
        p = measure(depth, target_mask=np.ones((100, 100), bool), target_bbox=[.4, .4, .5, .5],
                    mask_view='front', mask_observation_id=10, mask_confidence=.9)
        self.assertTrue(p['valid'], p['reason'])
        self.assertAlmostEqual(p['depth_m'], .8)

    def test_distant_target_cannot_use_near_image_centre(self):
        depth = np.full((100, 100), .5)
        depth[35:65, 5:25] = 4.0
        p = measure(depth, target_bbox=[.05, .35, .25, .65])
        self.assertTrue(p['valid'])
        self.assertEqual(p['depth_m'], 4.0)
        self.assertGreater(p['distance_m'], 4.0)
        self.assertEqual(gate(p)['reason'], 'target_surface_too_far')

    def test_missing_localisation_never_defaults_to_centre(self):
        self.assertEqual(measure(target_bbox=None)['reason'], 'explicit_target_localisation_required')
        self.assertEqual(measure(target_view='right')['reason'], 'target_view_depth_missing')

    def test_missing_and_malformed_view_rejected(self):
        for value in (None, 'RIGHT VIEW', '', [], True):
            with self.subTest(value=value):
                self.assertFalse(measure(target_view=value)['valid'])

    def test_malformed_nan_oob_huge_and_tiny_boxes_rejected(self):
        for box in ([0, 0, 1, 1], [-.1, .2, .4, .5], [.3, .3, 1.01, .5],
                    [.3, .3, .3, .5], [.5, .3, .2, .5], [.3, .3, .31, .31],
                    [.3, .3, np.nan, .7], [.3, .3, np.inf, .7], ['.3', '.3', '.7', '.7'],
                    [True, False, True, True], [False, .3, .6, .7], [.3, .3, .7],
                    [[.3], .3, .6, .7], np.asarray(.5), 'whole view'):
            with self.subTest(box=box):
                self.assertFalse(measure(target_bbox=box)['valid'])

    def test_minimum_pixel_support_is_required(self):
        self.assertFalse(measure(target_bbox=[.3, .3, .33, .33])['valid'])

    def test_invalid_depth_pixels_are_counted_in_fraction(self):
        depth = np.full((100, 100), .8)
        depth[35:60, 35:65] = np.nan
        p = measure(depth)
        self.assertFalse(p['valid'])
        self.assertAlmostEqual(p['valid_fraction'], 1/6)

    def test_zero_negative_infinite_and_normalized_like_depth_rejected(self):
        for value in (0., -.8, np.inf, np.nan, .1):
            with self.subTest(value=value):
                self.assertFalse(measure(np.full((100, 100), value))['valid'])

    def test_small_foreground_outliers_do_not_make_far_target_near(self):
        depth = np.full((100, 100), 4.)
        depth[35:36, 35:65] = .3
        p = measure(depth)
        self.assertTrue(p['valid'])
        self.assertGreater(p['distance_m'], 3.9)

    def test_foreground_background_mixture_is_not_reliable_target(self):
        depth = np.full((100, 100), 4.)
        depth[35:50, 35:65] = .4
        p = measure(depth)
        self.assertEqual(p['reason'], 'mixed_target_depth')

    def test_horizontal_distance_uses_pitch_and_intrinsics(self):
        angle = np.radians(-30)
        pitch = np.array([[1., 0, 0], [0, np.cos(angle), -np.sin(angle)],
                          [0, np.sin(angle), np.cos(angle)]])
        p = measure(np.ones((100, 100)), camera_rotation=pitch @ R,
                    target_bbox=[.47, .47, .53, .53])
        self.assertTrue(p['valid'])
        self.assertAlmostEqual(p['depth_m'], 1.)
        self.assertLess(p['distance_m'], .90)
        self.assertGreater(p['distance_m'], .83)
        self.assertAlmostEqual(p['world_center'][1], np.cos(angle), places=2)

    def test_missing_or_invalid_calibration_never_assumes_forward_range(self):
        for key, val in [('camera_intrinsic', None), ('camera_rotation', None),
                         ('camera_intrinsic', np.zeros((3, 3))), ('camera_rotation', np.ones((3, 3))),
                         ('camera_rotation', np.diag([2., 1, 1])),
                         ('camera_position', [0, np.nan, 0])]:
            with self.subTest(key=key):
                self.assertFalse(measure(**{key: val})['valid'])

    def test_three_dimensional_depth_single_channel_supported(self):
        self.assertTrue(measure(np.full((100, 100, 1), .8))['valid'])
        self.assertFalse(measure(np.ones((100, 100, 3)))['valid'])

    def test_production_habitat_axis_swap_and_camera_pitch_are_supported(self):
        # habitat_rotation: mapper XYZ = Habitat XZY, then sensor quaternion.
        axis_swap = np.array([[1., 0, 0], [0, 0, 1.], [0, 1., 0]])
        pitch = np.radians(-30)
        sensor_rotation = np.array([[1., 0, 0], [0, np.cos(pitch), -np.sin(pitch)],
                                    [0, np.sin(pitch), np.cos(pitch)]])
        rotation = axis_swap @ sensor_rotation
        self.assertAlmostEqual(np.linalg.det(rotation), -1.)
        p = measure(np.ones((100, 100)), camera_rotation=rotation,
                    target_bbox=[.47, .47, .53, .53])
        self.assertTrue(p['valid'])
        self.assertGreater(p['distance_m'], .83)
        self.assertLess(p['distance_m'], .90)
        self.assertAlmostEqual(p['world_center'][1], -np.cos(pitch), places=2)

    def test_packets_are_json_serializable(self):
        json.dumps(measure(), allow_nan=False)
        json.dumps(measure(np.full((100, 100), np.nan)), allow_nan=False)


class DetectorMaskTests(unittest.TestCase):
    def test_current_mask_takes_priority_over_bbox_background(self):
        depth = np.full((100, 100), 8.)
        depth[35:65, 35:65] = .8
        p = mask_packet(depth=depth, target_bbox=[.2, .2, .8, .8])
        self.assertTrue(p['valid'])
        self.assertEqual(p['source'], 'detector_mask')
        self.assertAlmostEqual(p['depth_m'], .8)
        self.assertTrue(gate(p, confirmed=False)['accepted'])

    def test_mask_can_supply_its_own_tight_bbox(self):
        p = mask_packet(target_bbox=None)
        self.assertTrue(p['valid'])
        self.assertEqual(p['bbox'], BOX)

    def test_stale_or_wrong_view_mask_never_falls_back_to_valid_bbox(self):
        for kwargs in ({'mask_view': 'right'}, {'mask_observation_id': 9},
                       {'mask_observation_id': None}, {'mask_observation_id': True}):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(mask_packet(**kwargs)['reason'], 'mask_not_from_current_target_view')

    def test_shifted_bbox_is_not_bound_to_unrelated_mask(self):
        self.assertEqual(mask_packet(target_bbox=[.01, .01, .30, .30])['reason'], 'mask_bbox_mismatch')

    def test_wrong_size_non_boolean_empty_and_untrusted_masks_rejected(self):
        for mask in (np.ones((50, 50), bool), np.ones((100, 100)),
                     np.zeros((100, 100), bool), [[None]]):
            with self.subTest(shape=np.shape(mask)):
                self.assertFalse(mask_packet(target_mask=mask)['valid'])
        self.assertFalse(mask_packet(mask_confidence=None)['valid'])
        self.assertEqual(gate(mask_packet(mask_confidence=.5))['reason'], 'target_mask_confidence_too_low')


class EndpointGateTests(unittest.TestCase):
    def test_localised_near_surface_still_requires_identity_relation(self):
        self.assertTrue(gate(measure())['accepted'])
        self.assertEqual(gate(measure(), relation_satisfied=False)['reason'], 'target_relation_not_verified')
        self.assertEqual(gate(measure(), confirmed=False)['reason'], 'changed_view_confirmation_required')

    def test_freshness_future_invalid_and_unknown_sources_fail_closed(self):
        for kwargs in ({'current_observation_id': 11}, {'current_observation_id': 9},
                       {'current_observation_id': None}):
            self.assertFalse(gate(measure(), **kwargs)['accepted'])
        self.assertTrue(gate(measure(), current_observation_id=11, max_age_observations=1)['accepted'])
        for bad in (None, {}, dict(measure(), source='image_center'), dict(measure(), distance_m=np.nan),
                    dict(measure(), view=[]), dict(measure(), endpoint_kind=[]), dict(measure(), source=[])):
            self.assertFalse(gate(bad)['accepted'])

    def test_stair_platform_and_between_need_measured_geometry(self):
        for kind in ('stair_landing', 'between'):
            p = measure(endpoint_kind=kind)
            self.assertEqual(gate(p)['reason'], 'measured_endpoint_geometry_required')
            self.assertTrue(gate(p, geometry_verified=True)['accepted'])
        self.assertFalse(measure(endpoint_kind='room_entry')['valid'])

    def test_door_frame_depth_excludes_far_opening_background(self):
        depth = np.full((100, 100), 6.)
        depth[20:80, 20:31] = .8
        depth[20:80, 69:80] = .8
        p = measure(depth, target_bbox=[.2, .2, .8, .8], endpoint_kind='doorway')
        self.assertTrue(p['valid'])
        self.assertEqual(p['sampling'], 'door_frame_borders')
        self.assertAlmostEqual(p['depth_m'], .8)
        self.assertAlmostEqual(p['opening_depth_m'], 6.)
        self.assertAlmostEqual(p['opening_depth_contrast_m'], 5.2)
        self.assertLess(p['frame_support']['left']['world_center'][0], 0)
        self.assertGreater(p['frame_support']['right']['world_center'][0], 0)
        self.assertEqual(gate(p)['reason'], 'measured_endpoint_geometry_required')
        self.assertTrue(gate(p, geometry_verified=True)['accepted'])

    def test_doorway_cannot_pass_with_only_one_visible_jamb(self):
        depth = np.full((100, 100), .8)
        depth[20:80, 69:80] = np.nan
        p = measure(depth, target_bbox=[.2, .2, .8, .8], endpoint_kind='doorway')
        self.assertEqual(p['reason'], 'insufficient_door_frame_support')


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.first = measure()
        self.second = measure(observation_id=11, camera_position=np.array([.2, 0, 0]))
        self.pose = dict(first_position=[0, 0, 0], second_position=[.2, 0, 0],
                         first_rotation=R, second_rotation=R)

    def confirm(self, a=None, b=None, **kwargs):
        return consistent_target_observations(self.first if a is None else a,
            self.second if b is None else b, **dict(self.pose, **kwargs))

    def test_new_translated_localisation_can_corroborate_same_hypothesis(self):
        self.assertTrue(self.confirm()['consistent'])

    def test_same_frame_or_unchanged_pose_does_not_confirm(self):
        self.assertEqual(self.confirm(b=self.first)['reason'], 'new_observation_required')
        self.assertEqual(self.confirm(second_position=[0, 0, 0])['reason'], 'changed_camera_pose_required')

    def test_new_rotated_view_can_confirm_without_forcing_translation(self):
        angle = np.radians(20)
        yaw = np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
        self.assertTrue(self.confirm(second_position=[0, 0, 0], second_rotation=yaw @ R)['consistent'])

    def test_changed_instance_or_displaced_world_target_cannot_confirm(self):
        a, b = dict(self.first, target_id=1), dict(self.second, target_id=2)
        self.assertEqual(self.confirm(a, b)['reason'], 'target_instance_changed')
        b = copy.deepcopy(self.second)
        b['world_center'][0] += 3
        self.assertEqual(self.confirm(b=b)['reason'], 'localized_target_moved')

    def test_missing_world_position_or_invalid_rotation_cannot_confirm(self):
        b = dict(self.second)
        del b['world_center']
        self.assertFalse(self.confirm(b=b)['consistent'])
        self.assertFalse(self.confirm(second_rotation=np.zeros((3, 3)))['consistent'])

    def test_production_left_handed_mapper_rotations_can_confirm(self):
        axis_swap = np.array([[1., 0, 0], [0, 0, 1.], [0, 1., 0]])
        self.assertTrue(self.confirm(first_rotation=axis_swap, second_rotation=axis_swap)['consistent'])
        self.assertEqual(self.confirm(first_rotation=axis_swap, second_rotation=R)['reason'],
                         'inconsistent_camera_coordinate_convention')


if __name__ == '__main__':
    unittest.main()


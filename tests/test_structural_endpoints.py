"""CPU geometry contracts for current landing evidence; no model/simulator."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from structural_endpoints import (current_floor_depth, stair_endpoint_side,
                                  structural_stair_verdict, bind_visible_stair_surface)


class CurrentFloorTests(unittest.TestCase):
    def sample(self, depth=None, **changes):
        args = dict(depth_views={'front': np.ones((100, 100)) if depth is None else depth},
                    camera_views={'front': {'position': [0., 0., 1.], 'rotation': np.eye(3)}},
                    intrinsic=np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]]),
                    ground=[0., 0., 0.], observation_id=5)
        args.update(changes)
        return current_floor_depth(**args)

    def test_current_depth_observes_floor_but_does_not_establish_task_completion(self):
        packet = self.sample()
        self.assertTrue(packet['valid'])
        self.assertGreater(packet['observed_area_m2'], .3)
        self.assertEqual(packet['world_center'], [0., 0., 0.])
        self.assertNotIn('accepted', packet)

    def test_other_floor_invalid_depth_and_stale_camera_pose_are_not_support(self):
        self.assertFalse(self.sample(np.full((100, 100), 2.))['valid'])
        self.assertFalse(self.sample(np.full((100, 100), np.nan))['valid'])
        self.assertFalse(self.sample(camera_views={'front': {'position': [2., 0., 1.], 'rotation': np.eye(3)}})['valid'])
        self.assertFalse(self.sample(observation_id=True)['valid'])

    def test_sloped_surface_does_not_become_a_floor_by_height_clipping(self):
        _, cols = np.indices((100, 100))
        depth = 1./(1.+.2*(cols-50)/100)
        packet = self.sample(depth)
        self.assertFalse(packet['valid'])
        self.assertEqual(packet['reason'], 'current_floor_depth_not_flat')

    def test_sparse_disconnected_points_do_not_fill_unknown_gaps(self):
        depth = np.full((100, 100), np.nan)
        depth[::10, ::10] = 1.
        self.assertFalse(self.sample(depth)['valid'])


class StructuralStairTests(unittest.TestCase):
    def packets(self):
        base = {'valid': True, 'current': True, 'observation_id': 5, 'checked_ground': [0., 0., 0.]}
        return copy.deepcopy(base), dict(base, ascending=True, descending=False), copy.deepcopy(base)

    def verdict(self, packets=None, text='Stop at the bottom of the stairs.'):
        return structural_stair_verdict(text, *(packets or self.packets()), observation_id=5, ground=[0., 0., 0.])

    def test_directional_structure_and_floor_are_separate_from_semantic_stop(self):
        result = self.verdict()
        self.assertTrue(result['valid'])
        self.assertTrue(result['geometry_verified'])
        self.assertFalse(result['accepted'])
        self.assertEqual(result['source'], 'habitat_structural_landing')

    def test_ramp_intermediate_landing_and_wrong_stair_side_cannot_pass(self):
        for name in ('platform', 'floor', 'ramp', 'intermediate', 'no_stair'):
            with self.subTest(name=name):
                packets = list(self.packets())
                if name == 'platform': packets[0]['valid'] = False
                elif name == 'floor': packets[2]['valid'] = False
                elif name == 'ramp': packets[1].update(ascending=False, descending=True)
                elif name == 'intermediate': packets[1]['descending'] = True
                else: packets[1].update(ascending=False, descending=False)
                self.assertFalse(self.verdict(packets)['valid'])

    def test_top_uses_opposite_local_stair_direction(self):
        packets = list(self.packets())
        packets[1].update(ascending=False, descending=True)
        self.assertTrue(self.verdict(packets, 'Stop at the top of the stairs.')['valid'])
        self.assertFalse(self.verdict(text='Stop at the top of the stairs.')['valid'])

    def test_stale_missing_nonfinite_or_other_pose_cannot_bind_current_geometry(self):
        for index in range(3):
            for change in ({'current': False}, {'observation_id': 4}, {'observation_id': True}, {'checked_ground': None},
                           {'checked_ground': [float('nan'), 0., 0.]}, {'checked_ground': [0., 0., 1.]}):
                with self.subTest(index=index, change=change):
                    packets = list(self.packets()); packets[index].update(change)
                    self.assertFalse(self.verdict(packets)['valid'])

    def test_head_for_kitchen_is_not_the_top_endpoint(self):
        self.assertEqual(stair_endpoint_side('take the stairs then head for the kitchen. stop at the bottom of the stairs.'), 'bottom')
        self.assertIsNone(stair_endpoint_side('Stand beside the stairs.'))
        self.assertIsNone(stair_endpoint_side('Stop at the top or bottom of the stairs.'))


class StairSurfaceBindingTests(unittest.TestCase):
    def bound(self, **changes):
        args = dict(depth=np.ones((100, 100)),
                    camera={'position': [0., 0., 1.], 'rotation': np.eye(3)},
                    intrinsic=np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]]),
                    bbox=[.2, .2, .8, .8], stair_points=[[0., 0., 0.], [.1, 0., 0.]])
        args.update(changes)
        return bind_visible_stair_surface(**args)

    def test_independent_surface_points_must_be_visible_inside_the_claimed_region(self):
        self.assertTrue(self.bound()['valid'])
        self.assertFalse(self.bound(bbox=[0., 0., .2, .2])['valid'])
        self.assertFalse(self.bound(depth=np.full((100, 100), .4))['valid'])
        self.assertFalse(self.bound(depth=np.full((100, 100), 4.))['valid'])

    def test_behind_camera_duplicate_points_and_whole_view_cannot_bind(self):
        self.assertFalse(self.bound(stair_points=[[0., 0., 2.], [.1, 0., 2.]])['valid'])
        self.assertFalse(self.bound(stair_points=[[0., 0., 0.], [0., 0., 0.]])['valid'])
        self.assertFalse(self.bound(bbox=[0., 0., 1., 1.])['valid'])

    def test_neighbourhood_background_cannot_hide_a_center_occluder(self):
        depth = np.full((100, 100), 2.)
        depth[49, 50] = depth[49, 55] = .4
        self.assertFalse(self.bound(depth=depth, stair_points=[[0., 0., -1.], [.1, 0., -1.]])['valid'])
        depth[49, 50] = depth[49, 55] = np.nan
        self.assertFalse(self.bound(depth=depth, stair_points=[[0., 0., -1.], [.1, 0., -1.]])['valid'])

    def test_subpixel_regions_invalid_calibration_and_overflow_are_unknown(self):
        self.assertFalse(self.bound(bbox=[.4, .489, .7, .491])['valid'])
        self.assertFalse(self.bound(intrinsic=np.diag([100., 100., 2.]))['valid'])
        self.assertFalse(self.bound(stair_points=[[10**400, 0, 0], [0, 0, 0]])['valid'])
        self.assertFalse(self.bound(depth=np.full((100, 100), 2.),
            stair_points=[[0., 0., -1.], [0., 0., -1.05]])['valid'])


if __name__ == '__main__':
    unittest.main()


"""Local platform evidence from analytic NavMeshes; CPU only, no Habitat import."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from habitat_geometry import measured_platform, stair_profile
from waypoint_navigation import WaypointRoutes


class Mesh:
    is_loaded = True

    def __init__(self, inside=lambda p: True, height=lambda p: 0., island=lambda p: 0):
        self.inside, self.height, self.island = inside, height, island
        self.calls = 0

    def snap_point(self, p):
        self.calls += 1
        result = np.array(p, float)
        result[1] = self.height(p)
        return result

    def is_navigable(self, p):
        self.calls += 1
        return self.inside(p)

    def get_island(self, p):
        self.calls += 1
        return self.island(p)

    def try_step_no_sliding(self, start, end):
        self.calls += 1
        # Check the entire tiny segment so a wall between samples blocks it.
        for a in np.linspace(0., 1., 11):
            if not self.inside(start + a * (end - start)):
                return np.array(start)
        result = np.array(end, float)
        result[1] = self.height(end)
        return result


def measure(mesh, pose=(0., 0., 0.), obs=1, policy=None):
    return measured_platform(NS(pf=mesh, to_habitat=lambda p: np.array([p[0], p[2], p[1]])),
                             pose, obs, policy=policy)


class HabitatPlatformTests(unittest.TestCase):
    def test_flat_platform_is_current_bound_and_does_not_change_pose(self):
        pose = np.zeros(3)
        mesh = Mesh()
        result = measure(mesh, pose, obs=7)
        self.assertTrue(result['valid'], result)
        self.assertTrue(result['current'])
        self.assertEqual(result['observation_id'], 7)
        self.assertEqual(result['checked_ground'], pose.tolist())
        self.assertGreater(result['supported_area_m2'], .3)
        self.assertGreater(result['short_span_m'], .4)
        self.assertAlmostEqual(result['slope_deg'], 0.)
        self.assertEqual(result['query_count'], mesh.calls)
        np.testing.assert_array_equal(pose, np.zeros(3))

    def test_stair_ramps_reject_cumulative_height_at_any_rotation(self):
        for angle in (0., .31, .78, 1.2):
            v = np.array([math.cos(angle), math.sin(angle)])
            for gradient in (.20, .8):
                result = measure(Mesh(height=lambda p: gradient * np.dot(p[[0, 2]], v)))
                self.assertFalse(result['valid'], (angle, gradient, result))
                self.assertGreater(result.get('height_rejection_count', 0), 0)

    def test_wide_thin_tread_rejects_independent_of_rotation(self):
        for angle in (0., .31, .78, 1.2):
            c, s = math.cos(angle), math.sin(angle)
            rotation = np.array([[c, -s], [s, c]])
            def inside(p):
                x, z = p[[0, 2]] @ rotation
                return abs(x) <= .55 and abs(z) <= .125
            result = measure(Mesh(inside=inside))
            self.assertFalse(result['valid'], (angle, result))
            self.assertIn(result['reason'], ('narrow_support', 'insufficient_connected_area'))

    def test_multiple_tread_heights_do_not_supply_one_large_platform(self):
        mesh = Mesh(height=lambda p: .18 * math.floor((p[2] + .125) / .25))
        self.assertFalse(measure(mesh)['valid'])

    def test_disconnected_nearby_floor_cannot_supply_support(self):
        mesh = Mesh(inside=lambda p: abs(p[0]) < .12 or abs(p[0]) > .18,
                    island=lambda p: 0 if abs(p[0]) < .12 else 1)
        self.assertFalse(measure(mesh)['valid'])
        # Even a malformed backend that jumps islands is not accepted.
        mesh = Mesh(island=lambda p: 0 if abs(p[0]) < .12 else 1)
        self.assertFalse(measure(mesh)['valid'])

    def test_overlap_floor_snap_and_mid_path_height_jumps_are_rejected(self):
        result = measure(Mesh(height=lambda p: 2.8))
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'current_pose_not_on_navmesh')
        result = measure(Mesh(height=lambda p: 0. if np.linalg.norm(p[[0, 2]]) < .08 else 2.8))
        self.assertFalse(result['valid'])

    def test_missing_nan_exception_and_budget_fail_closed(self):
        self.assertEqual(measure(None)['reason'], 'navmesh_unavailable')
        mesh = Mesh()
        mesh.try_step_no_sliding = lambda *args: np.full(3, np.nan)
        self.assertFalse(measure(mesh)['valid'])
        def fail(*args):
            raise RuntimeError('unavailable')
        mesh.try_step_no_sliding = fail
        self.assertEqual(measure(mesh)['reason'], 'navmesh_query_failed')
        mesh = Mesh()
        result = measure(mesh, policy={'max_queries': 7})
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'query_budget_exhausted')
        self.assertEqual(mesh.calls, 7)

    def test_no_cache_can_reuse_an_old_pose_or_frame(self):
        mesh = Mesh(inside=lambda p: p[0] < .7)
        first = measure(mesh, obs=1)
        second = measure(mesh, pose=(.65, 0., 0.), obs=2)
        self.assertEqual(first['checked_ground'], [0., 0., 0.])
        self.assertEqual(second['checked_ground'], [.65, 0., 0.])
        self.assertEqual(second['observation_id'], 2)
        self.assertGreater(mesh.calls, first['query_count'])
        for value in (True, -1, '2', 2.):
            self.assertFalse(measure(mesh, obs=value)['valid'])

    def test_unsafe_policy_cannot_raise_height_or_unbound_queries(self):
        for policy in ({'height_tolerance_m': .2}, {'radius_m': 10}, {'max_queries': 100000},
                       {'minimum_area_m2': float('nan')}, {'micro_step_m': .5}):
            self.assertEqual(measure(Mesh(), policy=policy)['reason'], 'invalid_platform_policy')


class StairProfileTests(unittest.TestCase):
    def profile(self, mesh, **kwargs):
        return stair_profile(NS(pf=mesh, to_habitat=lambda p: np.array([p[0], p[2], p[1]]),
                                to_mapper=lambda p: np.array([p[0], p[2], p[1]])),
                             [0., 0., 0.], 3, **kwargs)

    def test_flat_has_no_local_rise_or_fall_but_is_not_semantic_bottom(self):
        result = self.profile(Mesh())
        self.assertTrue(result['valid'])
        self.assertTrue(result['current'])
        self.assertFalse(result['ascending'])
        self.assertFalse(result['descending'])
        self.assertEqual(result['completed_rays'], 16)
        self.assertEqual(result['coverage'], 'finite_radial_samples_only')

    def test_bottom_top_and_intermediate_landing_have_distinct_local_profiles(self):
        for direction in (-1, 1):
            result = self.profile(Mesh(height=lambda p: direction * max(0., p[0] - .3) * .8))
            self.assertTrue(result['valid'], result)
            self.assertEqual(result['ascending'], direction == 1)
            self.assertEqual(result['descending'], direction == -1)
            points = result['ascending_points'] if direction == 1 else result['descending_points']
            self.assertGreater(len(points), 0)
            self.assertLessEqual(len(points), 16 * 3)
            self.assertTrue(all(direction * point[2] >= .20 for point in points))
            self.assertTrue(all(np.linalg.norm(np.asarray(point)[:2]) <= 1.2 + 1e-8 for point in points))
        result = self.profile(Mesh(height=lambda p: .8 * np.sign(p[0]) * max(0., abs(p[0]) - .3)))
        self.assertTrue(result['ascending'])
        self.assertTrue(result['descending'])

    def test_real_boundary_block_is_known_and_does_not_invent_surface_beyond_it(self):
        result = self.profile(Mesh(inside=lambda p: abs(p[0]) < .12))
        self.assertTrue(result['valid'])
        self.assertGreater(result['blocked_rays'], 0)
        self.assertFalse(result['ascending'])
        self.assertFalse(result['descending'])

    def test_profile_surface_points_use_real_mapper_ground_coordinates(self):
        mesh = Mesh(height=lambda p: 2. + .8 * max(0., p[0] - 4.3))
        routes = WaypointRoutes(mesh, [4., 7., 2.], NS)
        result = stair_profile(routes, [0., 0., -1.2], 4)
        self.assertTrue(result['valid'], result)
        self.assertTrue(result['ascending'])
        for point in result['ascending_points']:
            self.assertGreaterEqual(point[2] + 1.2, .20 - 1e-6)
            self.assertLessEqual(np.linalg.norm(np.asarray(point)[:2]), 1.2 + 1e-6)

    def test_height_jump_island_jump_missing_query_and_budget_are_unknown(self):
        result = self.profile(Mesh(height=lambda p: 0. if abs(p[0]) < .15 else 2.8))
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'unsafe_height_jump')
        result = self.profile(Mesh(island=lambda p: 0 if abs(p[0]) < .15 else 1))
        self.assertFalse(result['valid'])
        self.assertEqual(result['reason'], 'unexpected_island_change')
        mesh = Mesh()
        mesh.try_step_no_sliding = lambda *args: np.full(3, np.nan)
        self.assertEqual(self.profile(mesh)['reason'], 'navmesh_query_failed')
        mesh = Mesh()
        result = self.profile(mesh, max_queries=9)
        self.assertFalse(result['valid'])
        self.assertEqual(mesh.calls, 9)
        self.assertEqual(result['reason'], 'query_budget_exhausted')


if __name__ == '__main__':
    unittest.main()


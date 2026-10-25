"""Offline v3 regression tests for path planning graph reuse and invalidation.

Proves:
1. Shared graph construction (KD-trees, base nodes, node KD-tree, neighbor lists)
   is built once and reused across all candidate target queries in a pass.
2. Invalidation triggers on:
   - Map updates (update, update_multiview)
   - Pose changes (current_position modification)
   - Waypoint buffer changes (waypoints replacement)
   - Floor transitions (_switch_to_floor)
   - Mapper reset
3. Original navigation behavior is preserved when cache is absent.
4. Collision and floor height checks are fully preserved and not weakened.
5. Scoped context manager (path_planning_context) and batch API (plan_paths_to_targets).
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from mapper import Instruct_Mapper


def create_test_mapper():
    """Create a mapper on CPU with synthetic waypoints and navigable points."""
    K = np.array([[320.0, 0.0, 320.0],
                  [0.0, 320.0, 240.0],
                  [0.0, 0.0, 1.0]], dtype=np.float32)
    mapper = Instruct_Mapper(camera_intrinsic=K, device="CPU:0")

    # Set up a 2D grid of waypoints from (0,0,0) to (5,5,0) at 0.5m spacing
    grid = np.mgrid[0:6:0.5, 0:6:0.5, 0:0.1:0.1].reshape(3, -1).T.astype(np.float32)
    mapper.waypoints = grid
    mapper.current_position = np.array([0.0, 0.0, 1.2], dtype=np.float32)
    mapper.floor_height = 0.0
    mapper.initial_floor_height = -1.2
    return mapper


class TestV3PathPlanningCache(unittest.TestCase):
    def setUp(self):
        self.mapper = create_test_mapper()

    def test_shared_graph_reused_within_pass(self):
        """Proves shared graph construction is built once and reused for multiple candidates."""
        self.mapper.prepare_path_planning_cache()
        self.assertTrue(self.mapper.is_path_cache_valid())
        initial_builds = self.mapper._path_cache_build_count
        self.assertEqual(initial_builds, 1)

        targets = [
            np.array([1.0, 1.0, 0.0], dtype=np.float32),      # in grid
            np.array([2.5, 3.0, 0.0], dtype=np.float32),      # in grid
            np.array([1.25, 2.75, 0.0], dtype=np.float32),    # not in grid, attaches to neighbors
            np.array([0.5, 4.5, 0.0], dtype=np.float32),      # in grid
            np.array([4.0, 4.0, 0.0], dtype=np.float32),      # in grid
        ]

        paths = []
        for tg in targets:
            p = self.mapper.plan_path_to_target(tg)
            self.assertTrue(len(p) > 0, f"Expected path to {tg}")
            paths.append(p)

        # Build count must remain 1 — no rebuilding during the pass
        self.assertEqual(self.mapper._path_cache_build_count, initial_builds)
        self.assertEqual(self.mapper._path_planning_cache["hit_count"], len(targets))

    def test_equivalence_with_uncached_path(self):
        """Proves path returned with cache matches uncached plan_path_to_target."""
        targets = [
            np.array([2.0, 2.0, 0.0], dtype=np.float32),
            np.array([3.5, 1.5, 0.0], dtype=np.float32),
            np.array([0.75, 1.25, 0.0], dtype=np.float32),
        ]

        for tg in targets:
            # Uncached run
            self.mapper.invalidate_path_cache()
            self.assertFalse(self.mapper.is_path_cache_valid())
            p_uncached = self.mapper.plan_path_to_target(tg)

            # Cached run
            self.mapper.prepare_path_planning_cache()
            self.assertTrue(self.mapper.is_path_cache_valid())
            p_cached = self.mapper.plan_path_to_target(tg)

            self.assertEqual(p_uncached, p_cached, f"Cached path differed from uncached path for target {tg}")

    def test_invalidated_on_pose_change(self):
        """Proves moving the agent invalidates cache validity."""
        self.mapper.prepare_path_planning_cache()
        self.assertTrue(self.mapper.is_path_cache_valid())

        # Change position
        self.mapper.current_position = np.array([2.0, 1.0, 1.2], dtype=np.float32)
        self.assertFalse(self.mapper.is_path_cache_valid())

        # Calling plan_path_to_target still works (falls back to uncached)
        tg = np.array([2.5, 2.0, 0.0], dtype=np.float32)
        path = self.mapper.plan_path_to_target(tg)
        self.assertTrue(len(path) > 0)

    def test_invalidated_on_map_revision(self):
        """Proves map update methods invalidate cache and bump revision."""
        self.mapper.prepare_path_planning_cache()
        self.assertTrue(self.mapper.is_path_cache_valid())
        rev_before = self.mapper._map_revision

        # Map reset
        import quaternion
        self.mapper.reset(np.array([0.0, 0.0, 0.0]), quaternion.quaternion(1.0, 0.0, 0.0, 0.0))
        self.assertFalse(self.mapper.is_path_cache_valid())
        self.assertIsNone(self.mapper._path_planning_cache)
        self.assertGreater(self.mapper._map_revision, rev_before)

        # Floor switch
        self.mapper.waypoints = np.array([[1, 1, 0], [2, 2, 0]], dtype=np.float32)
        self.mapper.prepare_path_planning_cache()
        self.assertTrue(self.mapper.is_path_cache_valid())
        self.mapper._switch_to_floor(1, 2.5)
        self.assertFalse(self.mapper.is_path_cache_valid())
        self.assertIsNone(self.mapper._path_planning_cache)

    def test_invalidated_on_waypoint_buffer_change(self):
        """Proves replacing waypoint buffer invalidates the cache."""
        self.mapper.prepare_path_planning_cache()
        self.assertTrue(self.mapper.is_path_cache_valid())

        # Replace waypoints with a different set
        self.mapper.waypoints = np.array([[10.0, 10.0, 0.0]], dtype=np.float32)
        self.assertFalse(self.mapper.is_path_cache_valid())

    def test_floor_and_reachability_checks_preserved(self):
        """Proves floor check and unreachable checks are strictly preserved."""
        self.mapper.prepare_path_planning_cache()

        # Target on wrong floor (delta z > 1.0 from agent floor)
        wrong_floor_tg = np.array([1.0, 1.0, 5.0], dtype=np.float32)
        p_wrong = self.mapper.plan_path_to_target(wrong_floor_tg)
        self.assertEqual(p_wrong, [], "Must reject target on wrong floor")

        # Unreachable target far beyond graph connectivity
        unreachable_tg = np.array([100.0, 100.0, 0.0], dtype=np.float32)
        p_unreachable = self.mapper.plan_path_to_target(unreachable_tg)
        self.assertEqual(p_unreachable, [], "Must return empty path for unreachable target")

    def test_query_height_bounds_searches_same_floor_detour_instead_of_shortest_drop(self):
        """A floor-approach query can avoid a shorter route through lower nodes."""
        mapper = self.mapper
        mapper.current_position = np.array([0.0, 0.0, 1.2], dtype=np.float32)
        mapper.floor_height = 0.0
        mapper.initial_floor_height = -1.2
        mapper.waypoints = np.array([
            [0.5, 0.0, -0.2], [1.5, 0.0, -0.2],
            [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0], [2.0, 0.0, 0.0],
        ], dtype=np.float32)

        unconstrained = mapper.plan_path_to_target(
            np.array([2.0, 0.0, 0.0], dtype=np.float32),
            max_distance_to_neighbor=1.1,
            max_iterations=100,
            use_waypoint_buffer=True,
        )
        self.assertTrue(unconstrained)
        self.assertLess(float(np.min(np.asarray(unconstrained)[:, 2])), -0.15)

        mapper.invalidate_path_cache()
        bounded = mapper.plan_path_to_target(
            np.array([2.0, 0.0, 0.0], dtype=np.float32),
            max_distance_to_neighbor=1.1,
            max_iterations=100,
            use_waypoint_buffer=True,
            z_bounds=(-0.15, 0.15),
        )
        self.assertTrue(bounded, "A same-floor detour should exist around the lower shortcut")
        z = np.asarray(bounded, dtype=float)[:, 2]
        self.assertTrue(np.all((z >= -0.15) & (z <= 0.15)), bounded)
        self.assertAlmostEqual(float(z[-1]), 0.0, places=5)

    def test_stair_edge_checks_support_at_interpolated_height(self):
        """A stair edge is supported at each local step height, not one fixed floor slice."""
        class FakeTree:
            def __init__(self, query):
                self.query = query
                self.points = []

            def search_radius_vector_3d(self, point, radius):
                p = np.asarray(point, dtype=float)
                self.points.append(p.copy())
                return self.query(p, radius)

        mapper = self.mapper
        navigable = FakeTree(lambda p, _: (
            (10, list(range(10)), [0.0] * 10)
            if abs(p[2] - 0.5 * p[0]) < 1e-6 else (0, [], [])
        ))
        obstacle = FakeTree(lambda _p, _r: (0, [], []))

        clear = mapper._is_path_clear(
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.5], navigable, obstacle,
            check_radius=0.1, obstacle_radius=0.1, step_size=0.2,
            obstacle_points=np.empty((0, 3)),
        )

        self.assertTrue(clear)
        self.assertGreater(len(navigable.points), 1)
        self.assertGreater(navigable.points[-1][2], navigable.points[0][2])

    def test_astar_uses_validated_tread_centers_as_intermediate_nodes(self):
        """A stair route climbs via tread nodes rather than a long unsupported edge."""
        mapper = self.mapper
        mapper.current_position = np.array([0.0, 0.0, 1.2], dtype=np.float32)
        mapper.waypoints = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        tread_points = []
        for step in range(1, 5):
            cx, cz = 0.2 * step, 0.2 * step
            for x in np.linspace(cx - 0.08, cx + 0.08, 5):
                for y in np.linspace(-0.08, 0.08, 5):
                    tread_points.append([x, y, cz])
        tread_points = np.asarray(tread_points, dtype=np.float32)

        def tensor_cloud(points):
            cloud = o3d.t.geometry.PointCloud(o3d.core.Device('CPU:0'))
            cloud.point.positions = o3d.core.Tensor(np.asarray(points, dtype=np.float32))
            return cloud

        mapper.stair_tread_pcd = tensor_cloud(tread_points)
        mapper.stair_pcd = tensor_cloud(tread_points)
        mapper.navigable_pcd = tensor_cloud(tread_points)
        mapper.obstacle_pcd = tensor_cloud([[50.0, 50.0, 50.0]])

        path = mapper.plan_path_to_target(
            np.array([0.8, 0.0, 0.8], dtype=np.float32), max_iterations=100)

        self.assertGreaterEqual(len(path), 5, f'Expected a stepwise path, got {path}')
        path_np = np.asarray(path, dtype=float)
        self.assertTrue(np.all(np.diff(path_np[:, 2]) >= -1e-6), path)
        self.assertTrue(np.all(np.diff(path_np[:, 2]) <= 0.30 + 1e-6), path)
        self.assertAlmostEqual(path_np[-1, 2], 0.8, places=4)

    def test_astar_rejects_stair_tread_gap_above_step_height_limit(self):
        """Sparse or vertically separated tread detections cannot form a jump route."""
        mapper = self.mapper
        mapper.current_position = np.array([0.0, 0.0, 1.2], dtype=np.float32)
        mapper.waypoints = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        tread_points = []
        for cx, cz in ((0.2, 0.2), (0.4, 0.6)):
            for x in np.linspace(cx - 0.08, cx + 0.08, 5):
                for y in np.linspace(-0.08, 0.08, 5):
                    tread_points.append([x, y, cz])
        tread_points = np.asarray(tread_points, dtype=np.float32)

        def tensor_cloud(points):
            cloud = o3d.t.geometry.PointCloud(o3d.core.Device('CPU:0'))
            cloud.point.positions = o3d.core.Tensor(np.asarray(points, dtype=np.float32))
            return cloud

        mapper.stair_tread_pcd = tensor_cloud(tread_points)
        mapper.stair_pcd = tensor_cloud(tread_points)
        mapper.navigable_pcd = tensor_cloud(tread_points)
        mapper.obstacle_pcd = tensor_cloud([[50.0, 50.0, 50.0]])

        path = mapper.plan_path_to_target(
            np.array([0.4, 0.0, 0.6], dtype=np.float32), max_iterations=100)

        self.assertEqual(path, [], f'An over-height step gap must remain unreachable: {path}')

    def test_stair_tread_support_does_not_allow_wall_collision(self):
        """The stair-only support exception never bypasses body collision checks."""
        class FakeTree:
            def __init__(self, query):
                self.query = query

            def search_radius_vector_3d(self, point, radius):
                return self.query(np.asarray(point, dtype=float), radius)

        mapper = self.mapper
        tread = np.array([[x, 0.0, 0.0] for x in (0.2, 0.4, 0.6, 0.8)])
        navigable = FakeTree(lambda _p, _r: (0, [], []))
        tread_tree = FakeTree(lambda p, radius: (
            (1, [int(np.argmin(np.linalg.norm(tread - p, axis=1)))], [0.0])
            if np.min(np.linalg.norm(tread - p, axis=1)) <= radius else (0, [], [])
        ))
        wall = np.array([[0.2, 0.0, 0.4]])
        obstacle = FakeTree(lambda p, _r: (
            (1, [0], [0.0]) if np.linalg.norm(p - wall[0]) < 0.05 else (0, [], [])
        ))
        no_stair_surface = FakeTree(lambda _p, _r: (0, [], []))
        diagnostics = {'no_floor_support': 0, 'obstacle_collisions': 0,
                       'stair_surface_exemptions': 0, 'stair_tread_supports': 0}

        clear = mapper._is_path_clear(
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], navigable, obstacle,
            no_stair_surface, check_radius=0.1, obstacle_radius=0.1, step_size=0.2,
            obstacle_points=wall, stair_points=np.empty((0, 3)), diagnostics=diagnostics,
            stair_tread_kdtree=tread_tree, stair_tread_points=tread,
        )

        self.assertFalse(clear)
        self.assertEqual(diagnostics['no_floor_support'], 0)
        self.assertGreater(diagnostics['stair_tread_supports'], 0)
        self.assertEqual(diagnostics['obstacle_collisions'], 1)

    def test_nearby_stair_surface_does_not_hide_wall_collision(self):
        """A wall remains blocked even when stair points lie within the old broad exemption radius."""
        class FakeTree:
            def __init__(self, query=None, nearest=None):
                self.query = query
                self.nearest = nearest

            def search_radius_vector_3d(self, point, radius):
                return self.query(np.asarray(point, dtype=float), radius)

            def search_knn_vector_3d(self, point, k):
                return self.nearest(np.asarray(point, dtype=float), k)

        mapper = self.mapper
        navigable = FakeTree(query=lambda _p, _r: (10, list(range(10)), [0.0] * 10))
        wall_point = np.array([0.2, 0.0, 0.4])
        obstacle = FakeTree(query=lambda p, _r: (
            (1, [0], [0.0]) if np.linalg.norm(p - np.array([0.2, 0.0, 0.4])) < 0.05
            else (0, [], [])
        ))
        nearby_stairs = FakeTree(
            query=lambda _p, _r: (6, list(range(6)), [0.0] * 6),
            nearest=lambda _p, _k: (1, [0], [0.1 ** 2]),
        )

        clear = mapper._is_path_clear(
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], navigable, obstacle, nearby_stairs,
            check_radius=0.1, obstacle_radius=0.1, step_size=0.2,
            obstacle_points=wall_point.reshape(1, 3),
            stair_points=np.repeat(np.array([[0.2, 0.1, 0.4]]), 6, axis=0),
        )

        self.assertFalse(clear)

        stair_surface = FakeTree(
            query=lambda _p, _r: (6, list(range(6)), [0.0] * 6),
            nearest=lambda _p, _k: (1, [0], [0.0]),
        )
        self.assertTrue(mapper._is_path_clear(
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], navigable, obstacle, stair_surface,
            check_radius=0.1, obstacle_radius=0.1, step_size=0.2,
            obstacle_points=wall_point.reshape(1, 3),
            stair_points=np.repeat(wall_point.reshape(1, 3), 6, axis=0),
        ))

    def test_flat_floor_connectivity_and_collision_are_preserved(self):
        class FakeTree:
            def __init__(self, query):
                self.query = query

            def search_radius_vector_3d(self, point, radius):
                return self.query(np.asarray(point, dtype=float), radius)

        mapper = self.mapper
        navigable = FakeTree(lambda _p, _r: (10, list(range(10)), [0.0] * 10))
        empty_obstacles = FakeTree(lambda _p, _r: (0, [], []))
        self.assertTrue(mapper._is_path_clear(
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], navigable, empty_obstacles,
            step_size=0.2, obstacle_points=np.empty((0, 3)),
        ))

        wall_point = np.array([[0.2, 0.0, 0.4]])
        blocked_obstacles = FakeTree(lambda p, _r: (
            (1, [0], [0.0]) if np.linalg.norm(p - wall_point[0]) < 0.05 else (0, [], [])
        ))
        self.assertFalse(mapper._is_path_clear(
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], navigable, blocked_obstacles,
            step_size=0.2, obstacle_points=wall_point,
        ))

    def test_scoped_context_manager(self):
        """Proves path_planning_context scopes cache lifetime cleanly."""
        self.assertIsNone(self.mapper._path_planning_cache)
        with self.mapper.path_planning_context():
            self.assertTrue(self.mapper.is_path_cache_valid())
            p = self.mapper.plan_path_to_target(np.array([1.0, 1.0, 0.0], dtype=np.float32))
            self.assertTrue(len(p) > 0)
        # Outside context, cache is cleared
        self.assertIsNone(self.mapper._path_planning_cache)
        self.assertFalse(self.mapper.is_path_cache_valid())

    def test_batch_path_query_api(self):
        """Proves plan_paths_to_targets batches queries and cleans up when standalone."""
        targets = [
            np.array([1.0, 1.0, 0.0], dtype=np.float32),
            np.array([2.0, 2.0, 0.0], dtype=np.float32),
            np.array([50.0, 50.0, 0.0], dtype=np.float32), # unreachable
        ]
        results = self.mapper.plan_paths_to_targets(targets)
        self.assertEqual(len(results), 3)
        self.assertTrue(len(results[0]) > 0)
        self.assertTrue(len(results[1]) > 0)
        self.assertEqual(results[2], [])
        # When called without existing cache, cleans up afterward
        self.assertIsNone(self.mapper._path_planning_cache)


if __name__ == "__main__":
    unittest.main()


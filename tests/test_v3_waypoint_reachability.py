"""CPU regressions for physical waypoint routes and production controller methods.

Analytic geometry substitutes Habitat's NavMesh; no API, GPU, datasets or writes.
These tests do not establish real-scene SR or replace a Habitat smoke run.
"""
import ast
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
from waypoint_navigation import WaypointRoutes, candidate_order, representative_points
from navigation_policy import checkpoint_yaw
from progress_watchdog import NavigationAborted
from agent.route_guard import RouteEvidenceGuard
from agent.recovery import CheckpointRecovery
from navigation_contracts import completion_conditions
from navigation_control import WaypointExecution, StairProgress, pending_stair_condition, stair_condition_intents, landing_support


class Mesh:
    """Analytic traversability with a robot footprint already eroded from walls."""
    def __init__(self, inside=lambda p: True, height=lambda p: 0., corners=None):
        self.inside, self.height, self.corners = inside, height, corners
        self.connected = True
        self.query_error = False

    def snap_point(self, p):
        q = np.array(p, dtype=float)
        q[1] = self.height(q)
        return q

    def is_navigable(self, p):
        return self.inside(p)

    def find_path(self, path):
        path.points = [path.requested_start, *(self.corners or []), path.requested_end]
        return self.connected

    def try_step_no_sliding(self, a, b):
        if self.query_error:
            raise RuntimeError('geometry unavailable')
        if not self.inside(b):
            return np.array(a)
        return self.snap_point(b)

    def distance_to_closest_obstacle(self, p, radius):
        return .1


def routes(mesh, **cfg):
    return WaypointRoutes(mesh, np.zeros(3), NS, cfg)


class ReachabilityTests(unittest.TestCase):
    def test_coordinate_roundtrip_with_nonzero_origin(self):
        planner = WaypointRoutes(Mesh(), [4, 7, 2], NS)
        for point in ([.4, .7, -1.2], [0, 0, 2.1]):
            np.testing.assert_allclose(planner.to_mapper(planner.to_habitat(point)), point, atol=1e-6)
        np.testing.assert_allclose(planner.to_habitat([0, 0, -1.2]), [4, 2, 7])

    def test_endpoint_navigable_but_wall_between_rejected(self):
        mesh = Mesh(inside=lambda p: not (.4 < p[0] < .6))
        self.assertTrue(mesh.is_navigable([1, 0, 0]))
        self.assertIsNone(routes(mesh).plan([0, 0, 0], [1, 0, -1.2]))

    def test_disconnected_island_rejected(self):
        mesh = Mesh()
        mesh.connected = False
        self.assertIsNone(routes(mesh).plan([0, 0, 0], [1, 0, -1.2]))

    def test_narrow_corridor_preserved_without_extra_point_three_inflation(self):
        mesh = Mesh(inside=lambda p: abs(p[2]) <= .04)
        route = routes(mesh).plan([0, 0, 0], [1, 0, -1.2])
        self.assertIsNotNone(route)
        self.assertIsNone(routes(mesh).plan([0, 0, 0], [1, .06, -1.2]))

    def test_exact_corner_route_preserved(self):
        mesh = Mesh(inside=lambda p: abs(p[2]) < .08 or abs(p[0]-.6) < .08,
                    corners=[[.6, 0, 0]])
        result = routes(mesh, waypoint_max_detour_ratio=2).plan([0, 0, 0], [.6, 1, -1.2])
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result['path'][1], [.6, 0, -1.2])
        mesh.corners = None
        self.assertIsNone(routes(mesh).plan([0, 0, 0], [.6, 1, -1.2]))

    def test_up_and_down_stairs_use_navmesh_tread_height(self):
        height = lambda p: .2 * min(4, max(0, int(math.floor((p[0]+1e-6)/.25))))
        mesh = Mesh(height=height)
        self.assertIsNotNone(routes(mesh).plan([0, 0, 0], [1.01, 0, -.4]))
        self.assertIsNotNone(routes(mesh).plan([1.01, .8, 0], [0, 0, -1.2]))

    def test_unclimbable_jump_rejected(self):
        mesh = Mesh(height=lambda p: .8 if p[0] >= .5 else 0.)
        self.assertIsNone(routes(mesh).plan([0, 0, 0], [1, 0, -.4]))

    def test_floor_snap_cannot_jump_to_other_storey(self):
        mesh = Mesh(height=lambda p: 2.8)
        self.assertIsNone(routes(mesh).plan([0, 0, 0], [1, 0, -1.2]))

    def test_large_horizontal_snap_rejected(self):
        mesh = Mesh()
        mesh.snap_point = lambda p: np.asarray(p)+[.4, 0, 0]
        self.assertIsNone(routes(mesh).plan([0, 0, 0], [1, 0, -1.2]))

    def test_missing_or_failed_navmesh_never_passes(self):
        self.assertIsNone(routes(None).plan([0, 0, 0], [1, 0, -1.2]))
        mesh = Mesh()
        mesh.query_error = True
        self.assertIsNone(routes(mesh).plan([0, 0, 0], [1, 0, -1.2]))

    def test_zero_progress_short_substep_uses_existing_bounded_fallback(self):
        mesh = Mesh()
        mesh.try_step_no_sliding = lambda a, b: np.asarray(a)
        # At a real NavMesh corner, no-slide can stall while the actual
        # primitive's bounded fallback advances with micrometre lateral error.
        mesh.try_step = Mock(side_effect=lambda a, b: np.asarray(b) + [0, 0, 1e-6])
        self.assertTrue(routes(mesh).segment([0, 0, 0], [.2, 0, 0]))
        self.assertEqual(mesh.try_step.call_count, 4)

    def test_zero_progress_fallback_cannot_cross_a_real_wall(self):
        mesh = Mesh()
        mesh.try_step_no_sliding = lambda a, b: np.asarray(a)
        mesh.try_step = Mock(side_effect=lambda a, b: np.asarray(a))
        self.assertFalse(routes(mesh).segment([0, 0, 0], [.2, 0, 0]))
        mesh.try_step.assert_called_once()

    def test_zero_progress_fallback_keeps_lateral_deviation_limit(self):
        mesh = Mesh()
        mesh.try_step_no_sliding = lambda a, b: np.asarray(a)
        mesh.try_step = Mock(side_effect=lambda a, b: np.asarray(b) + [0, 0, .09])
        self.assertFalse(routes(mesh).segment([0, 0, 0], [.2, 0, 0]))

    def test_zero_progress_fallback_keeps_height_jump_limit(self):
        mesh = Mesh()
        mesh.try_step_no_sliding = lambda a, b: np.asarray(a)
        mesh.try_step = Mock(side_effect=lambda a, b: np.asarray(b) + [0, .4, 0])
        self.assertFalse(routes(mesh).segment([0, 0, 0], [.2, 0, 0]))

    def test_normal_progress_does_not_query_sliding_fallback(self):
        mesh = Mesh()
        mesh.try_step = Mock(side_effect=AssertionError('Unnecessary fallback'))
        self.assertTrue(routes(mesh).segment([0, 0, 0], [.2, 0, 0]))
        mesh.try_step.assert_not_called()

    def test_nan_and_long_unknown_detour_rejected(self):
        self.assertIsNone(routes(Mesh()).plan([0, 0, 0], [np.nan, 0, -1.2]))
        mesh = Mesh(corners=[[0, 0, 4], [1, 0, 4]])
        self.assertIsNone(routes(mesh).plan([0, 0, 0], [1, 0, -1.2]))

    def test_surface_sampling_does_not_invent_centroid_in_gap(self):
        points = np.array([[.1, .1, .1], [.4, .4, .4], [.15, .1, .1]])
        sampled = representative_points(points, .5)
        self.assertEqual(len(sampled), 1)
        self.assertTrue(any(np.array_equal(sampled[0], p.astype(np.float32)) for p in points))
        self.assertEqual(representative_points([[np.nan, 0, 0]], .2).shape, (0, 3))

    def test_candidate_cap_keeps_side_door(self):
        points = [[x, 0, 0] for x in np.linspace(.2, 2, 100)] + [[0, 1, 0]]
        selected = candidate_order(points, [0, 0, 0], limit=16)
        self.assertEqual(len(selected), 16)
        self.assertIn(100, selected)


def production_functions(path, names, namespace):
    nodes = [n for n in ast.walk(ast.parse(path.read_text()))
             if isinstance(n, ast.FunctionDef) and n.name in names]
    for node in nodes:
        node.returns = None
        for arg in node.args.args:
            arg.annotation = None
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return {n.name: namespace[n.name] for n in nodes}


class Action:
    def __init__(self, r, theta, type):
        self.r, self.theta, self.type = r, theta, type


Action.null = Action(0, 0, 'null')
methods = production_functions(SRC / 'agent/agent.py',
    {'_get_action_for_next_waypoint', '_invalidate_blocked_path', '_validated_waypoint_route',
     'execute_action', '_preprocessing_module', '_ensure_stair_progress', '_landing_support_observed', '_execution_feedback'},
    dict(np=np, math=math, PolarAction=Action, checkpoint_yaw=checkpoint_yaw,
         candidate_order=candidate_order, completion_conditions=completion_conditions,
         StairProgress=StairProgress, pending_stair_condition=pending_stair_condition,
         stair_condition_intents=stair_condition_intents, landing_support=landing_support))
Controller = type('Controller', (CheckpointRecovery, RouteEvidenceGuard), methods)


def controller(mesh=None):
    a = Controller()
    mesh = mesh or Mesh()
    state = NS(position=np.zeros(3))
    a.sim_wrapper = NS(sim=NS(pathfinder=mesh, get_agent=lambda _: NS(get_state=lambda: state)), agent_radius=.1)
    # Camera basis yielding a forward vector along +x in mapper coordinates.
    a.mapper = NS(current_position=np.zeros(3), initial_floor_height=-1.2,
                  current_rotation=np.array([[0., 0, -1], [1, 0, 0], [0, -1, 0]]))
    a.navigation_cfg = {}
    a.current_path = [[.2, 0, -1.2], [.2, .8, -1.2]]
    a.waypoint_routes = routes(mesh)
    a.target_world_position = [.2, .8, -1.2]
    a.navigation_trace = []
    a._last_step_index = 10
    a._normalize_angle = lambda x: (x+np.pi) % (2*np.pi)-np.pi
    a._blocked_targets, a._execution_ledger = [], []
    a._blocked_move_count, a.traveled_distance = 0, 0.
    a._physical_progress_step = 0
    a.rolling_back = False
    a._action_to_text = lambda action: action.type
    a.instruction_obj = NS(get_current_subtask_key=lambda: 'P1', add_record_to_current_subtask=lambda r: None,
                          get_subtasks_key_coord=lambda: {}, get_current_subtask=lambda: ('P1', {'instruction': ''}))
    return a


class ControllerTests(unittest.TestCase):
    def test_corner_twenty_cm_away_is_not_skipped(self):
        a = controller()
        result = a._get_action_for_next_waypoint()
        self.assertEqual(result.type, 'move_forward')
        self.assertAlmostEqual(result.r, .2)
        self.assertEqual(len(a.current_path), 2)

    def test_blocked_forward_not_executed_and_target_temporarily_excluded(self):
        a = controller(Mesh(inside=lambda p: p[0] < .01))
        a._executing_verifier_correction = True
        result = a._get_action_for_next_waypoint()
        self.assertEqual(result.type, 'null')
        self.assertEqual(a.current_path, [])
        self.assertFalse(a._executing_verifier_correction)
        self.assertIsNone(a._validated_waypoint_route(a.target_world_position))
        a._last_step_index = 30
        a.sim_wrapper.sim.pathfinder.inside = lambda p: True
        self.assertIsNotNone(a._validated_waypoint_route(a.target_world_position))

    def test_shorter_stride_when_long_stride_blocked(self):
        a = controller(Mesh(inside=lambda p: p[0] <= .12))
        result = a._get_action_for_next_waypoint()
        self.assertEqual(result.type, 'move_forward')
        self.assertLessEqual(result.r, .12)

    def test_wrong_height_at_same_xy_never_pops_corner(self):
        a = controller()
        a.current_path = [[0, 0, .8]]
        result = a._get_action_for_next_waypoint()
        self.assertEqual(result.type, 'null')
        self.assertTrue(a._blocked_targets)

    def test_two_real_blocked_moves_invalidate_and_records_are_measured(self):
        a = controller()
        state = a.sim_wrapper.sim.get_agent(0).get_state()
        a.sim_wrapper.step = lambda action: {'agent_state': state}
        a.update_map = lambda obs: []
        for _ in range(2):
            a.execute_action(Action(.2, 0, 'move_forward'))
        self.assertEqual(a.current_path, [])
        self.assertEqual([r['motion_distance'] for r in a._execution_ledger], [0, 0])
        self.assertEqual([r['step'] for r in a._execution_ledger], [0, 1])
        self.assertEqual(a.traveled_distance, 0)

    def test_production_preprocessing_exposes_only_certified_waypoints(self):
        mesh = Mesh(inside=lambda p: not (.4 < p[0] < .6))
        a = controller(mesh)
        # The first target is a navigable endpoint BEHIND a wall. The second
        # is reachable. A large center depth must never override the wall.
        points = np.array([[1, 0, -1.2], [0, 1, -1.2]], dtype=float)
        a.mapper.get_current_view_candidate_waypoints = lambda **kw: (points, points[:, [0, 2, 1]])
        a.mapper.trajectory_position = [[0, 0, 0]]
        a.mapper.camera_intrinsic = np.eye(3)
        a.sim_wrapper.agent_height = 1.5
        a.image_edge_threshold, a.min_move_distance, a.max_move_distance = .04, .25, 3.
        a.first_stop, a.turn_angle_rad = False, np.pi/2
        a.instruction_obj.get_subtasks_key_coord = lambda: {}
        a._world_to_pixel_coords = lambda pts, *args: (np.tile([50, 70], (len(pts), 1)),
                                                      np.arange(len(pts)), np.ones(len(pts)))
        # Only rendering and an unused visited-point index are replaced.
        cv = NS(FONT_HERSHEY_SIMPLEX=0, getTextSize=lambda *args: ((12, 8), 0),
                circle=lambda *args: None, putText=lambda *args: None,
                addWeighted=lambda *args: None)
        observation = {'color_sensor': np.zeros((100, 100, 3), dtype=np.uint8),
                       'depth_sensor': np.full((100, 100), 10.)}
        with patch.dict(sys.modules, {'cv2': cv, 'scipy': NS(), 'scipy.spatial': NS(cKDTree=None)}):
            _, actions = a._preprocessing_module(observation)
            waypoints = [v for v in actions.values() if v['type'] == 'waypoint']
            self.assertEqual(len(waypoints), 1)
            np.testing.assert_allclose(waypoints[0]['target_point_world'], [0, 1, -1.2])
            self.assertEqual(waypoints[0]['route_backend'], 'navmesh_no_sliding')
            mesh.connected = False
            a.last_turn_dir = 'L'
            _, actions = a._preprocessing_module(observation)
            self.assertFalse(any(v['type'] == 'waypoint' for v in actions.values()))
            self.assertEqual(set(actions), {'TURN_LEFT', 'TURN_RIGHT', 'TURN_BACK'})

    def test_production_step_follows_certified_corner_without_astar_fallback(self):
        step = production_functions(SRC/'agent/agent.py', {'step'},
            dict(np=np, math=math, PolarAction=Action, NavigationAborted=NavigationAborted, WaypointExecution=WaypointExecution,
                 create_top_down_map_centered=lambda *args, **kw: np.zeros((2, 2, 3))))['step']
        a = controller(Mesh(corners=[[.2, 0, 0]]))
        a.current_path, a.action_sequence = [], []
        a.prev_agent_position = None
        a.last_action = Action.null
        a.last_actions, a.vlm_responses = {}, []
        a.config = {'camera': {'fov': 90}}
        a.curr_obs = {'agent_state': a.sim_wrapper.sim.get_agent(0).get_state()}
        a.instruction_obj.current_step_count = 0
        a.decide_waypoint = lambda **kw: ({'type': 'waypoint', 'target_point_world': [.2, .8, -1.2]},
                                          'selected a certified waypoint', object())
        def forbidden(*args):
            raise AssertionError('Independent A* must not replace the certified route')
        a.mapper.plan_path_to_target = forbidden
        executed = []
        a.execute_action = executed.append
        step(a, 0)
        self.assertEqual(executed[0].type, 'move_forward')
        self.assertAlmostEqual(executed[0].r, .2)
        np.testing.assert_allclose(a.current_path[0], [.2, 0, -1.2])


class BodyClearanceTests(unittest.TestCase):
    def setUp(self):
        self.fn = production_functions(SRC/'mapper.py', {'_waypoint_body_clear'}, dict(np=np))['_waypoint_body_clear']

    @staticmethod
    def tree(points):
        def search(p, radius):
            indices = np.flatnonzero(np.linalg.norm(np.asarray(points)-p, axis=1) <= radius)
            return len(indices), indices, []
        return NS(search_radius_vector_3d=search)

    def test_head_obstacle_cannot_hide_between_old_sphere_checks(self):
        obstacles = np.array([[0, 0, 1.35]])
        self.assertFalse(self.fn(None, [0, 0, 0], self.tree(obstacles), obstacles, None, .13, 1.5))

    def test_stairs_never_exempt_wall_or_railing(self):
        obstacles = np.array([[0, 0, .2], [0, 0, .8]])
        stair = self.tree([[0, 0, .2], [0, 0, .8]])
        self.assertFalse(self.fn(None, [0, 0, 0], self.tree(obstacles), obstacles, stair, .13, 1.5))
        self.assertTrue(self.fn(None, [0, 0, 0], self.tree(obstacles[:1]), obstacles[:1], stair, .13, 1.5))

    def test_footprint_margin_preserves_narrow_passage(self):
        obstacles = np.array([[0, -.16, .5], [0, .16, .5]])
        self.assertTrue(self.fn(None, [0, 0, 0], self.tree(obstacles), obstacles, None, .13, 1.5))
        self.assertFalse(self.fn(None, [0, 0, 0], self.tree(obstacles), obstacles, None, .2, 1.5))


class MetricsTests(unittest.TestCase):
    def test_near_goal_without_stop_is_only_oracle_success(self):
        funcs = production_functions(SRC/'run_experiments.py', {'_calculate_metrics'}, dict(np=np, fastdtw=None))
        env = NS(path_calculator=NS(), simWrapper=NS(get_path=lambda p: 1.), env_cfg={'success_threshold': 2.},
                 agent=NS(traveled_distance=3.), agent_trajectory=[], oracle_best_distance=100.,
                 oracle_best_spl=0., oracle_success=False)
        moving = funcs['_calculate_metrics'](env, NS(position=[0, 0, 0]), NS(type='move_forward'), 2.)
        self.assertFalse(moving['goal_reached'])
        self.assertEqual(moving['oracle_success'], 1)
        self.assertEqual(moving['spl'], 0.)
        stopped = funcs['_calculate_metrics'](env, NS(position=[0, 0, 0]), NS(type='stop'), 2.)
        self.assertTrue(stopped['goal_reached'])
        self.assertEqual(stopped['finish_status'], 'success')


if __name__ == '__main__':
    unittest.main()


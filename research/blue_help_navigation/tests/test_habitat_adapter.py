"""CPU-only API facades: these tests do not claim to render Habitat scenes."""
import copy
from dataclasses import replace
import math
from types import SimpleNamespace as NS
import unittest
import numpy as np

from blue_help.contracts import DummySpec, Episode
from blue_help.habitat_adapter import AdapterConfig, HabitatNavigationEnvironment, _yaw


class FakePathfinder:
    is_loaded = True
    def __init__(self):
        self.unreachable = False
        self.stuck = False
        self.path_override = None
        self.step_calls = 0
        self.snap_offset = np.zeros(3)
    def seed(self, value): self.seed_value = value
    def is_navigable(self, p, max_y_delta=.5): return np.isfinite(p).all()
    def snap_point(self, p): return np.asarray(p)+self.snap_offset
    def find_path(self, query):
        if self.unreachable: return False
        query.points = self.path_override or [query.requested_start, query.requested_end]
        query.geodesic_distance = sum(math.dist(a,b) for a,b in zip(query.points,query.points[1:]))
        return True
    def try_step_no_sliding(self, a, b):
        self.step_calls += 1
        return np.asarray(a if self.stuck else b)


class FakeAgent:
    def __init__(self, sim):
        self.sim = sim
        self.position = np.zeros(3)
        self.rotation = np.array([0,0,0,1.])
        self.controls = NS(move_filter_fn=lambda a,b: sim.pathfinder.try_step_no_sliding(a,b))
        self.camera_missing = False
    def get_state(self):
        sensors = {} if self.camera_missing else {"rgb": NS(position=self.position+[0,1.25,0],rotation=self.rotation.copy())}
        return NS(position=self.position.copy(), rotation=self.rotation.copy(), sensor_states=sensors)
    def set_state(self, state, reset_sensors):
        assert reset_sensors
        self.position, self.rotation = state.position.copy(), state.rotation.copy()


class FakeObject:
    def __init__(self, identity, owner):
        self.object_id, self.owner = identity, owner
        self._translation = np.zeros(3)
        self.motion_type = "DYNAMIC"
        self.rotation = 0
        self.aabb = NS(min=(-.275,-.8,-.135),max=(.275,.8,.135))
    @property
    def translation(self): return self._translation.copy()
    @translation.setter
    def translation(self, value):
        if self.motion_type != "STATIC": self._translation = np.asarray(value).copy()
    @property
    def transformation(self):
        matrix = np.eye(4)
        matrix[:3,3] = self.translation
        c,s = math.cos(self.rotation),math.sin(self.rotation)
        matrix[:3,:3] = [[c,0,s],[0,1,0],[-s,0,c]]
        return matrix
    def contact_test(self): return self.owner.contact


class FakeObjects:
    def __init__(self):
        self.objects = {}
        self.removed = []
        self.next_id = 21
        self.contact = False
    def add_object_by_template_id(self, identity):
        assert identity == 7
        obj = FakeObject(self.next_id,self)
        self.objects[obj.object_id] = obj
        self.next_id += 1
        return obj
    def remove_object_by_id(self, identity):
        self.removed.append(identity)
        del self.objects[identity]


class FakeSimulator:
    def __init__(self):
        self.pathfinder = FakePathfinder()
        self.objects = FakeObjects()
        self.agent = FakeAgent(self)
        self.rgb = np.full((12,16,4), [50,130,240,255], dtype=np.uint8)
        self.depth = np.ones((12,16), dtype=np.float32)
        self.semantic = np.full((12,16), 10000, dtype=np.int32)
        self.ray_hits = None
        self.placement_hits = []
        self.ray_calls = []
        self.actions = []
        self.closed = False
        self.forward_step = .25
        self.sensors_missing = set()
    def get_agent(self, index): assert index == 0; return self.agent
    def get_rigid_object_manager(self): return self.objects
    def get_object_template_manager(self): return NS(load_configs=lambda p: [7])
    def seed(self, value): self.seed_value = value
    def reset(self): self.agent.position = np.zeros(3)
    def get_sensor_observations(self):
        result = {"rgb": self.rgb,"depth": self.depth,"semantic": self.semantic}
        return {k:v for k,v in result.items() if k not in self.sensors_missing}
    def cast_ray(self, ray, max_distance, buffer_distance):
        self.ray_calls.append((ray, max_distance, buffer_distance))
        if max_distance < .4:
            return NS(hits=self.placement_hits)
        hits = self.ray_hits
        if hits is None:
            hits = [NS(object_id=next(iter(self.objects.objects)), ray_distance=.5)]
        return NS(hits=hits)
    def step(self, action):
        self.actions.append(action)
        collided = False
        if action == "move_forward":
            start = self.agent.position.copy()
            yaw = _yaw(self.agent.rotation)
            wanted = start+self.forward_step*np.array([-math.sin(yaw),0,-math.cos(yaw)])
            self.agent.position = np.asarray(self.agent.controls.move_filter_fn(start,wanted))
            collided = np.linalg.norm(self.agent.position-wanted) > 1e-4
        else:
            sign = 1 if action == "turn_left" else -1
            yaw = _yaw(self.agent.rotation)+sign*math.pi/6
            self.agent.rotation = np.array([0,math.sin(yaw/2),0,math.cos(yaw/2)])
        return {**self.get_sensor_observations(),"collided":collided}
    def close(self): self.closed = True


def episode():
    return Episode("1","scene.glb","Walk forward.",(0,0,0),(0,0,0,1),(9,8,7))


def env_and_sim(config=None):
    sim = FakeSimulator()
    env = HabitatNavigationEnvironment(sim, "actor.object_config.json", config=config or AdapterConfig(),
        shortest_path_factory=NS, rotation_from_xyzw=lambda q: np.asarray(q),
        object_rotation_from_yaw=lambda yaw: yaw, static_motion_type="STATIC",
        ray_factory=lambda origin,direction: NS(origin=np.asarray(origin),direction=np.asarray(direction)))
    return env, sim


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.env, self.sim = env_and_sim()
        self.dummy = DummySpec("blue", (0,0,-.8), 0)
    def tearDown(self): self.env.close()

    def test_current_semantic_blue_rgb_true_firsthit_and_reachability(self):
        frame = self.env.reset(episode(), [self.dummy], seed=12)
        row = frame.dummy_candidates[0]
        self.assertTrue(row["visible"])
        self.assertTrue(row["spatial_eligible"])
        self.assertEqual(row["interaction_radius_m"],1)
        self.assertEqual(self.sim.pathfinder.seed_value,12)
        self.assertTrue(self.sim.ray_calls)
        ray, _, buffer = next(item for item in self.sim.ray_calls if item[1] > .4)
        self.assertAlmostEqual(np.linalg.norm(ray.direction),1)
        np.testing.assert_allclose(ray.origin,[0,1.25,0])
        self.assertEqual(buffer,0)
        actor = next(iter(self.sim.objects.objects.values()))
        self.assertEqual(actor.semantic_id,10000)
        self.assertNotEqual(actor.semantic_id,actor.object_id)
        self.assertTrue(actor.collidable)
        self.assertEqual(actor.motion_type,"STATIC")
        np.testing.assert_allclose(actor.translation,[0,.8,-.8])

    def test_wall_firsthit_fails_even_when_target_also_hit(self):
        self.sim.ray_hits = [NS(object_id=21,ray_distance=.8),NS(object_id=-1,ray_distance=.2)]
        row = self.env.reset(episode(),[self.dummy]).dummy_candidates[0]
        self.assertFalse(row["spatial_eligible"])
        self.assertIn("occluded_or_unverified_line_of_sight",row["reason_codes"])

    def test_no_ray_hits_or_missing_camera_fail_closed(self):
        self.sim.ray_hits = []
        self.assertFalse(self.env.reset(episode(),[self.dummy]).dummy_candidates[0]["spatial_eligible"])
        self.sim.ray_hits = None
        self.sim.agent.camera_missing = True
        self.assertFalse(self.env.observe().dummy_candidates[0]["spatial_eligible"])

    def test_hidden_semantic_no_registry_omniscience(self):
        self.sim.semantic[:] = 0
        row = self.env.reset(episode(),[self.dummy]).dummy_candidates[0]
        self.assertFalse(row["visible"])
        self.assertFalse(row["spatial_eligible"])
        self.assertTrue(all(item[1] < .4 for item in self.sim.ray_calls))

    def test_missing_semantic_fails_closed(self):
        self.sim.sensors_missing.add("semantic")
        self.assertFalse(self.env.reset(episode(),[self.dummy]).dummy_candidates[0]["visible"])

    def test_nonblue_and_too_few_pixels_fail_closed(self):
        self.sim.rgb[:] = [240,20,20,255]
        self.assertFalse(self.env.reset(episode(),[self.dummy]).dummy_candidates[0]["visible"])
        self.sim.rgb[:] = [50,130,240,255]
        self.sim.semantic[:] = 0
        self.sim.semantic[0,0] = 10000
        self.assertFalse(self.env.observe().dummy_candidates[0]["visible"])

    def test_outside_one_metre_and_different_floor_fail(self):
        row = self.env.reset(episode(),[replace(self.dummy,ground_position=(0,0,-1.01))]).dummy_candidates[0]
        self.assertFalse(row["spatial_eligible"])
        row = self.env.reset(episode(),[replace(self.dummy,ground_position=(0,3,-.8))]).dummy_candidates[0]
        self.assertFalse(row["spatial_eligible"])
        self.assertIn("different_or_unverified_floor",row["reason_codes"])

    def test_boundary_inclusive(self):
        row = self.env.reset(episode(),[replace(self.dummy,ground_position=(0,0,-1))]).dummy_candidates[0]
        self.assertTrue(row["spatial_eligible"])

    def test_unreachable_navmesh_fails_closed(self):
        self.sim.pathfinder.unreachable = True
        self.assertFalse(self.env.reset(episode(),[self.dummy]).dummy_candidates[0]["spatial_eligible"])

    def test_interaction_path_goes_other_floor_rejected(self):
        self.sim.pathfinder.path_override = [(0,0,0),(1,1,0),(1,1,-.22),(0,0,-.22)]
        self.assertFalse(self.env.reset(episode(),[self.dummy]).dummy_candidates[0]["spatial_eligible"])

    def test_swept_movement_rejects_tunneling_with_clear_endpoints(self):
        self.env.reset(episode(),[self.dummy])
        self.sim.forward_step = 2
        frame = self.env.step("move_forward")
        self.assertTrue(frame.collision)
        np.testing.assert_allclose(frame.pose.position,(0,0,0))
        self.assertEqual(self.env.path_length_m,0)

    def test_legal_actions_omit_known_collision_but_step_still_enforces(self):
        close = replace(self.dummy,ground_position=(0,0,-.6))
        frame = self.env.reset(episode(),[close])
        self.assertNotIn("move_forward",frame.legal_actions)
        self.assertFalse(frame.collision)  # preview is not a collision event
        self.assertTrue(self.env.step("move_forward").collision)

    def test_safe_path_uses_same_proxy_and_does_not_invent_detour(self):
        self.env.reset(episode(),[self.dummy])
        self.assertIsNone(self.env.find_safe_path((0,0,0),(0,0,-2)))
        self.assertIsNotNone(self.env.find_safe_path((.7,0,0),(.7,0,-2)))

    def test_path_navmesh_no_sliding_required_and_executability_checked(self):
        self.env.reset(episode(),[])
        self.sim.pathfinder.stuck = True
        self.assertIsNone(self.env.find_safe_path((0,0,0),(0,0,-1)))
        self.sim.pathfinder.try_step_no_sliding = None
        self.assertIsNone(self.env.find_safe_path((0,0,0),(0,0,-1)))

    def test_reset_removes_owned_actors_and_counters(self):
        self.env.reset(episode(),[self.dummy])
        self.env.step("turn_left")
        self.env.step("stop")
        self.env.reset(episode(),[])
        self.assertEqual(self.sim.objects.removed,[21])
        self.assertEqual(self.sim.objects.objects,{})
        self.assertEqual(self.env.step_count,0)
        self.assertFalse(self.env.stopped)

    def test_failed_partial_spawn_cleans_up(self):
        with self.assertRaises(ValueError):
            self.env.reset(episode(),[self.dummy,self.dummy])
        self.assertFalse(self.sim.objects.objects)
        with self.assertRaises(RuntimeError): self.env.observe()

    def test_start_overlap_rejected(self):
        with self.assertRaises(ValueError):
            self.env.reset(episode(),[replace(self.dummy,ground_position=(0,0,-.2))])

    def test_rgb_depth_are_readonly_copies_and_no_goal_forwarded(self):
        frame = self.env.reset(episode(),[self.dummy])
        self.assertFalse(frame.rgb.flags.writeable)
        self.assertFalse(frame.depth.flags.writeable)
        self.assertFalse(np.shares_memory(frame.rgb,self.sim.rgb))
        self.assertFalse(hasattr(frame,"goal_position"))
        self.assertFalse(hasattr(self.env,"goal_position"))
        self.assertNotIn("position",frame.dummy_candidates[0])
        self.sim.rgb[:] = 0
        self.assertGreater(frame.rgb.sum(),0)

    def test_basic_navigation_yaw_wait_stop_and_path_length(self):
        self.env.reset(episode(),[])
        frame = self.env.step("move_forward")
        np.testing.assert_allclose(frame.pose.position,[0,0,-.25])
        self.assertAlmostEqual(self.env.path_length_m,.25)
        self.assertAlmostEqual(self.env.step("turn_left").pose.yaw_rad,math.pi/6)
        self.assertAlmostEqual(self.env.step("turn_right").pose.yaw_rad,0)
        self.env.step("wait")
        self.env.step("stop")
        self.assertEqual(self.sim.actions,["move_forward","turn_left","turn_right"])
        self.assertEqual(self.env.step_count,5)
        with self.assertRaises(RuntimeError): self.env.step("move_forward")

    def test_missing_rgb_depth_and_invalid_actions(self):
        self.env.reset(episode(),[])
        with self.assertRaises(ValueError): self.env.step("teleport")
        self.sim.sensors_missing.add("depth")
        with self.assertRaises(RuntimeError): self.env.observe()

    def test_close_restores_original_filter_and_owned_lifecycle(self):
        original = self.env._base_filter
        self.env.reset(episode(),[self.dummy])
        self.env.close()
        self.assertIs(self.sim.agent.controls.move_filter_fn,original)
        self.assertFalse(self.sim.closed)
        self.assertFalse(self.sim.objects.objects)
        self.env.close()
        with self.assertRaises(RuntimeError): self.env.observe()

    def test_config_validation(self):
        with self.assertRaises(ValueError): AdapterConfig(navmesh_error_m=.1)
        with self.assertRaises(ValueError): AdapterConfig(width=0)

    def test_movement_query_failure_returns_start_instead_of_unfiltered_move(self):
        self.env.reset(episode(),[])
        def broken(*args): raise RuntimeError("query unavailable")
        self.env._base_filter = broken
        frame = self.env.step("move_forward")
        np.testing.assert_allclose(frame.pose.position,[0,0,0])
        self.assertTrue(frame.collision)
        self.assertNotIn("move_forward",frame.legal_actions)

    def test_path_never_requests_interpolated_vertical_teleport(self):
        self.env.reset(episode(),[])
        self.assertIsNone(self.env.find_safe_path((0,0,0),(0,2,0)))
        requested_dy = []
        def stair(a,b):
            requested_dy.append(float(b[1]-a[1]))
            result = np.asarray(b).copy()
            result[1] = -result[2]*.2
            return result
        self.sim.pathfinder.try_step_no_sliding = stair
        self.assertIsNotNone(self.env.find_safe_path((0,0,0),(0,.2,-1)))
        self.assertTrue(all(dy == 0 for dy in requested_dy))

    def test_navmesh_floor_jump_rejected(self):
        self.env.reset(episode(),[])
        def teleport(a,b):
            result = np.asarray(b).copy(); result[1] = 2; return result
        self.sim.pathfinder.try_step_no_sliding = teleport
        self.assertIsNone(self.env.find_safe_path((0,0,0),(0,2,-1)))

    def test_cleanup_failure_retains_ownership_for_retry(self):
        self.env.reset(episode(),[self.dummy])
        actual = self.sim.objects.remove_object_by_id
        def fail(identity): raise RuntimeError("temporary remove failure")
        self.sim.objects.remove_object_by_id = fail
        with self.assertRaises(RuntimeError): self.env.close()
        self.assertEqual(len(self.env._actors),1)
        self.assertFalse(self.env._closed)
        self.sim.objects.remove_object_by_id = actual
        self.env.close()
        self.assertFalse(self.sim.objects.objects)

    def test_static_pose_order_readback_and_real_bounds_preflight(self):
        self.env.reset(episode(),[self.dummy])
        audit = self.env.placement_audit[0]
        self.assertEqual(audit["status"],"pass_coarse_checks_only")
        self.assertFalse(audit["full_topology_validated"])
        self.assertFalse(audit["bullet_contact"])
        self.assertTrue(audit["radial_rays_clear"])

    def test_body_contact_and_near_wall_rays_reject_placement(self):
        self.sim.objects.contact = True
        with self.assertRaisesRegex(ValueError,"contacts"):
            self.env.reset(episode(),[self.dummy])
        self.assertFalse(self.sim.objects.objects)
        self.sim.objects.contact = False
        self.sim.placement_hits = [NS(object_id=-1,ray_distance=.1)]
        with self.assertRaisesRegex(ValueError,"nearby"):
            self.env.reset(episode(),[self.dummy])

    def test_proxy_must_enclose_actual_asset(self):
        with self.assertRaisesRegex(ValueError,"enclose"):
            self.env.reset(episode(),[replace(self.dummy,radius_m=.1)])

    def test_dummy_pair_overlap_rejected(self):
        other = replace(self.dummy,dummy_id="other",ground_position=(.1,0,-.8))
        with self.assertRaisesRegex(ValueError,"bodies overlap"):
            self.env.reset(episode(),[self.dummy,other])
        self.assertFalse(self.sim.objects.objects)

    def test_action_specs_are_relative_only(self):
        frame = self.env.reset(episode(),[])
        self.assertEqual(frame.action_specs["move_forward"],
                         {"relative_direction":"forward","requested_distance_m":.25})
        self.assertEqual(frame.action_specs["turn_left"]["degrees"],30)
        self.assertEqual(set(frame.action_specs),set(frame.legal_actions))

    def test_episode_goal_is_never_accessed(self):
        class NoGoalEpisode:
            episode_id = "private-goal-canary"
            start_position = (0,0,0)
            start_rotation_xyzw = (0,0,0,1)
            @property
            def goal_position(self): raise AssertionError("goal leaked into environment")
        self.env.reset(NoGoalEpisode(),[])

    def test_nonlevel_start_rejected_for_upright_body_baseline(self):
        tilted = replace(episode(),start_rotation_xyzw=(math.sin(.1),0,0,math.cos(.1)))
        with self.assertRaisesRegex(ValueError,"yaw-only"):
            self.env.reset(tilted,[])

    def test_semantic_ids_avoid_existing_scene_annotations(self):
        self.sim.semantic_scene = NS(objects=[None,NS(semantic_id=10000)])
        self.sim.semantic[:] = 10001
        frame = self.env.reset(episode(),[self.dummy])
        self.assertEqual(next(iter(self.sim.objects.objects.values())).semantic_id,10001)
        self.assertTrue(frame.dummy_candidates[0]["visible"])

    def test_actual_magnum_vector_return_contract_when_installed(self):
        try:
            import magnum as mn
        except ImportError:
            self.skipTest("Actual Magnum available only in isolated Habitat environment")
        self.env.reset(episode(),[self.dummy])
        self.env._base_filter = lambda a,b: b
        start, end = mn.Vector3(0,0,0), mn.Vector3(0,0,-.1)
        accepted = self.env._movement_filter(start,end)
        self.assertIsInstance(accepted,mn.Vector3)
        self.assertAlmostEqual((accepted-start).dot(),.01)
        rejected = self.env._movement_filter(start,mn.Vector3(0,0,-2))
        self.assertIsInstance(rejected,mn.Vector3)
        self.assertEqual((rejected-start).dot(),0)

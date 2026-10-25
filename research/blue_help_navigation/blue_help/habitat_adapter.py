"""Real Habitat-Sim API adapter, with optional dependency imports kept lazy.

The injected Simulator owns world knowledge; SensorFrame contains no goal/path.
find_safe_path is a privileged oracle/evaluation service, never a policy input.
The ordinary Habitat Agent is a scene node, not a Bullet rigid body: its movement
filter is explicitly chained to our continuous human-body occupancy checks.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable

import numpy as np

from .collision import Cylinder, first_blocker, swept_cylinder_hits, vec3
from .contracts import DummySpec, Episode, Pose, SensorFrame
from .interaction_range import RangeConfig, evaluate_interaction


@dataclass(frozen=True)
class AdapterConfig:
    allow_sliding: bool = False
    agent_radius_m: float = 0.18
    agent_height_m: float = 1.5
    camera_height_m: float = 1.25
    collision_margin_m: float = 0.03
    forward_step_m: float = 0.25
    turn_degrees: float = 30.0
    interaction_radius_m: float = 1.0
    floor_tolerance_m: float = 0.25
    max_interaction_path_m: float = 1.5
    navmesh_step_m: float = 0.05
    navmesh_error_m: float = 0.015
    endpoint_error_m: float = 0.05
    max_step_height_m: float = 0.25
    helper_visibility_mode: str = "semantic_blue"
    minimum_visible_pixels: int = 8
    minimum_blue_fraction: float = 0.35
    width: int = 320
    height: int = 240
    hfov_degrees: float = 90.0

    def __post_init__(self):
        if self.helper_visibility_mode not in {"semantic_blue", "semantic_instance"}:
            raise ValueError("Unknown helper visibility mode")
        if type(self.allow_sliding) is not bool:
            raise ValueError("allow_sliding must be bool")
        positive = (self.agent_radius_m, self.agent_height_m, self.camera_height_m,
                    self.forward_step_m, self.turn_degrees, self.interaction_radius_m,
                    self.floor_tolerance_m, self.max_interaction_path_m,
                    self.navmesh_step_m, self.navmesh_error_m, self.endpoint_error_m,
                    self.max_step_height_m)
        if any(not math.isfinite(x) or x <= 0 for x in positive):
            raise ValueError("Adapter dimensions must be finite and positive")
        if not math.isfinite(self.collision_margin_m) or self.collision_margin_m < 0:
            raise ValueError("Invalid collision margin")
        if not 1 < self.hfov_degrees < 179 or not 0 < self.minimum_blue_fraction <= 1:
            raise ValueError("Invalid sensor settings")
        if any(type(x) is not int or x <= 0 for x in (self.minimum_visible_pixels, self.width, self.height)):
            raise ValueError("Integer sensor settings required")
        if self.navmesh_error_m >= self.navmesh_step_m:
            raise ValueError("NavMesh tolerance must be smaller than validation step")


def _xyzw(rotation):
    if isinstance(rotation, (tuple, list, np.ndarray)):
        result = np.asarray(rotation, dtype=float)
    elif hasattr(rotation, "imag") and hasattr(rotation, "real"):
        result = np.r_[np.asarray(rotation.imag, dtype=float), float(rotation.real)]
    elif hasattr(rotation, "vector") and hasattr(rotation, "scalar"):
        result = np.r_[np.asarray(rotation.vector, dtype=float), float(rotation.scalar)]
    else:
        result = np.array([rotation.x, rotation.y, rotation.z, rotation.w], dtype=float)
    if result.shape != (4,) or not np.isfinite(result).all() or abs(float(result @ result)-1) > 1e-3:
        raise ValueError("Expected normalized XYZW quaternion")
    return result


def _yaw(rotation):
    x, y, z, w = _xyzw(rotation)
    # Rotated camera forward (0,0,-1); positive yaw turns left in Habitat Y-up.
    forward_x, forward_z = -2*(x*z+y*w), -(1-2*(x*x+y*y))
    return math.atan2(-forward_x, -forward_z)


def _rotation_from_xyzw(value):
    from habitat_sim.utils.common import quat_from_coeffs
    return quat_from_coeffs(np.asarray(value, dtype=float))


def _object_rotation_from_yaw(yaw):
    import magnum as mn
    return mn.Quaternion.rotation(mn.Rad(float(yaw)), mn.Vector3.y_axis())


def _ray(origin, direction):
    import magnum as mn
    from habitat_sim.geo import Ray
    return Ray(mn.Vector3(*vec3(origin)), mn.Vector3(*vec3(direction)))


class HabitatNavigationEnvironment:
    """Goal-isolated wrapper around an existing habitat_sim.Simulator.

    Factories are injectable only to test the actual API calls without installing
    Habitat. Defaults import real Habitat/Magnum classes; no mock auto-fallback.
    reset reads only episode identity/start, never episode.goal_position.
    """
    ACTIONS = ("move_forward", "turn_left", "turn_right", "stop", "wait")

    def __init__(self, simulator, object_config_path, *, config=AdapterConfig(),
                 shortest_path_factory: Callable | None = None,
                 rotation_from_xyzw: Callable | None = None,
                 object_rotation_from_yaw: Callable | None = None,
                 static_motion_type=None, ray_factory: Callable | None = None,
                 owns_simulator=False):
        self.sim = simulator
        self.config = config
        self.object_config_path = str(object_config_path)
        self._shortest_path_factory = shortest_path_factory
        self._rotation_from_xyzw = rotation_from_xyzw or _rotation_from_xyzw
        self._object_rotation_from_yaw = object_rotation_from_yaw or _object_rotation_from_yaw
        self._static_motion_type = static_motion_type
        self._ray_factory = ray_factory or _ray
        self._owns_simulator = bool(owns_simulator)
        self._closed = False
        self._ready = False
        self._actors = []
        self._bodies = []
        self._template_id = None
        self._agent = self.sim.get_agent(0)
        self._base_filter = self._agent.controls.move_filter_fn
        if not callable(self._base_filter):
            raise ValueError("Simulator must supply the original NavMesh movement filter")
        self._installed_filter = self._movement_filter
        self._agent.controls.move_filter_fn = self._installed_filter
        self.step_count = 0
        self.path_length_m = 0.0
        self.stopped = False
        self._collision = False
        self._proxy_collision = False
        self.placement_audit = []
        self.last_path_check = {"status": "not_queried"}

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Environment is closed")

    def _blocker(self, start, end):
        return first_blocker(start, end, self._bodies,
                             agent_radius_m=self.config.agent_radius_m,
                             agent_height_m=self.config.agent_height_m,
                             margin_m=self.config.collision_margin_m)

    def _movement_filter(self, start, end):
        """Preserve original NavMesh filtering, then reject a swept body crossing."""
        # Keep Habitat's Magnum Vector3 return type. ObjectControls calls the
        # vector's zero-argument .dot(); NumPy.ndarray is NOT API-compatible.
        original_start = start
        start_coordinates = vec3(start)
        try:
            vec3(end)
            original_filtered = self._base_filter(start, end)
            filtered = vec3(original_filtered)
            valid = (self._navigable(start_coordinates) and self._navigable(filtered)
                     and self._blocker(start_coordinates, filtered) is None)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            valid = False
        if not valid:
            self._proxy_collision = True
            return original_start
        return original_filtered

    def _remove_owned_actors(self):
        manager = self.sim.get_rigid_object_manager()
        errors = []
        failed = []
        for item in self._actors:
            try:
                manager.remove_object_by_id(item["object_id"])
            except Exception as exc:
                errors.append(exc)
                failed.append(item)
        self._actors = failed
        failed_ids = {item["spec"].dummy_id for item in failed}
        self._bodies = [body for body in self._bodies if body.identity in failed_ids]
        if errors:
            raise RuntimeError("Could not clean up all owned dummy actors") from errors[0]

    def _navigable(self, point):
        pf = self.sim.pathfinder
        try:
            return bool(pf.is_loaded and pf.is_navigable(np.asarray(point),
                                                       max_y_delta=self.config.floor_tolerance_m))
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return False

    def reset(self, episode: Episode, dummies: list[DummySpec], seed: int = 42) -> SensorFrame:
        self._check_open()
        self._ready = False
        self._remove_owned_actors()
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Seed must be a uint32 integer")
        start = vec3(episode.start_position)
        rotation = _xyzw(episode.start_rotation_xyzw)
        if abs(rotation[0]) > 1e-5 or abs(rotation[2]) > 1e-5:
            raise ValueError("This upright-cylinder baseline requires a level yaw-only start rotation")
        if not self._navigable(start):
            raise ValueError("Episode start is not on the loaded NavMesh")
        self.sim.seed(seed)
        self.sim.pathfinder.seed(seed)
        self.sim.reset()
        state = self._agent.get_state()
        state.position = np.asarray(start)
        state.rotation = self._rotation_from_xyzw(rotation)
        self._agent.set_state(state, reset_sensors=True)
        self.episode_id = str(episode.episode_id)
        self.step_count, self.path_length_m = 0, 0.0
        self.stopped, self._collision, self._proxy_collision = False, False, False
        self.placement_audit = []
        self.last_path_check = {"status": "not_queried"}
        seen = set()
        semantic_ids = set()
        scene = getattr(self.sim, "semantic_scene", None)
        for item in getattr(scene, "objects", ()):
            if item is not None and hasattr(item, "semantic_id"):
                semantic_ids.add(int(item.semantic_id))
        manager = self.sim.get_rigid_object_manager()
        if hasattr(manager, "get_objects_by_handle_substring"):
            semantic_ids.update(int(obj.semantic_id) for obj in manager.get_objects_by_handle_substring("").values())
        next_semantic_id = 10000
        try:
            for index, dummy in enumerate(dummies):
                if dummy.dummy_id in seen:
                    raise ValueError("Duplicate dummy ID")
                seen.add(dummy.dummy_id)
                body = Cylinder(dummy.dummy_id, dummy.ground_position, dummy.radius_m, dummy.height_m)
                if not self._navigable(body.foot):
                    raise ValueError("Dummy foot is not on NavMesh: " + dummy.dummy_id)
                if (not math.isfinite(dummy.yaw_rad) or not math.isfinite(dummy.floor_offset_m)
                        or not 0 <= dummy.floor_offset_m <= dummy.height_m):
                    raise ValueError("Invalid dummy pose or asset floor offset")
                if first_blocker(start, start, [body], agent_radius_m=self.config.agent_radius_m,
                                 agent_height_m=self.config.agent_height_m,
                                 margin_m=self.config.collision_margin_m):
                    raise ValueError("Dummy overlaps episode start: " + dummy.dummy_id)
                if any(swept_cylinder_hits(body.foot, body.foot, other,
                        agent_radius_m=body.radius_m, agent_height_m=body.height_m,
                        margin_m=self.config.collision_margin_m) for other in self._bodies):
                    raise ValueError("Dummy bodies overlap: " + dummy.dummy_id)
                if self._template_id is None:
                    handles = self.sim.get_object_template_manager().load_configs(self.object_config_path)
                    if len(handles) != 1:
                        raise ValueError("Expected exactly one dummy object template")
                    self._template_id = handles[0]
                obj = self.sim.get_rigid_object_manager().add_object_by_template_id(self._template_id)
                if obj is None:
                    raise RuntimeError("Habitat failed to instantiate dummy")
                while next_semantic_id in semantic_ids:
                    next_semantic_id += 1
                record = {"spec": dummy, "object_id": int(obj.object_id), "semantic_id": next_semantic_id}
                semantic_ids.add(next_semantic_id)
                self._actors.append(record)  # track immediately, even if a later setter fails
                static = self._static_motion_type
                if static is None:
                    from habitat_sim.physics import MotionType
                    static = MotionType.STATIC
                obj.collidable = True  # Needed for true Bullet raycasts, NOT sufficient to stop Agent.
                wanted_translation = np.asarray(body.foot) + np.array([0.0, dummy.floor_offset_m, 0.0])
                # Habitat 0.3.3 STATIC bodies ignore later transform setters.
                # Set pose while movable, freeze LAST, then verify real readback.
                obj.translation = wanted_translation
                obj.rotation = self._object_rotation_from_yaw(dummy.yaw_rad)
                obj.semantic_id = record["semantic_id"]
                obj.motion_type = static
                if np.linalg.norm(np.asarray(vec3(obj.translation))-wanted_translation) > 1e-4:
                    raise RuntimeError("Dummy pose setter did not take effect; template must spawn movable")
                self._preflight_actor(obj, body)
                self._bodies.append(body)
            self._ready = True
            return self.observe()
        except Exception:
            self._ready = False
            self._remove_owned_actors()
            raise

    def _preflight_actor(self, obj, body):
        """Conservative placement checks, not proof of full topology clearance.

        Exact engine AABB corners must fit the declared occupancy envelope;
        Bullet contacts and sampled horizontal rays must be clear. Ground contact
        may conservatively reject a placement: never silently raise/move the asset.
        """
        audit = {"dummy_id": body.identity, "status": "pending",
                 "full_topology_validated": False}
        self.placement_audit.append(audit)
        try:
            bounds = obj.aabb
            low, high = np.asarray(vec3(bounds.min)), np.asarray(vec3(bounds.max))
            matrix = np.asarray(obj.transformation, dtype=float)
            if matrix.shape != (4,4) or not np.isfinite(matrix).all() or np.any(low > high):
                raise ValueError("Invalid actual object bounds/transform")
            world = []
            for x in (low[0],high[0]):
                for y in (low[1],high[1]):
                    for z in (low[2],high[2]):
                        world.append((matrix @ np.array([x,y,z,1.0]))[:3])
            delta = np.asarray(world)-np.asarray(body.foot)
            radius = float(np.linalg.norm(delta[:,[0,2]],axis=1).max())
            min_y, max_y = float(delta[:,1].min()), float(delta[:,1].max())
            audit.update(actual_aabb_radius_m=radius, actual_aabb_foot_relative_y=[min_y,max_y])
            if radius > body.radius_m+1e-3 or min_y < -1e-3 or max_y > body.height_m+1e-3:
                raise ValueError("Declared cylinder does not enclose actual dummy geometry")
            contact = bool(obj.contact_test())
            audit["bullet_contact"] = contact
            if contact:
                raise ValueError("Dummy contacts existing collision geometry")
            for height in (0.1, body.height_m*.5, body.height_m-.1):
                center = np.asarray(body.foot)+[0,height,0]
                for angle in np.linspace(0, 2*math.pi, 16, endpoint=False):
                    direction = np.array([math.cos(angle),0,math.sin(angle)])
                    result = self.sim.cast_ray(self._ray_factory(center,direction),
                        max_distance=body.radius_m+self.config.collision_margin_m, buffer_distance=0.0)
                    for hit in result.hits:
                        if not math.isfinite(float(hit.ray_distance)) or hit.ray_distance < 0:
                            raise ValueError("Invalid placement ray evidence")
                        if (int(hit.object_id) != int(obj.object_id)
                                and hit.ray_distance <= body.radius_m+self.config.collision_margin_m):
                            raise ValueError("Dummy occupancy proxy intersects nearby collidable geometry")
            audit["status"] = "pass_coarse_checks_only"
            audit["radial_rays_clear"] = True
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            audit["status"], audit["reason"] = "rejected_or_unverified", str(exc)
            raise ValueError("Dummy placement rejected: " + body.identity + ": " + str(exc)) from exc

    def _visible(self, rgb, semantic, semantic_id):
        if semantic is None:
            return False
        labels = np.asarray(semantic)
        if labels.shape != rgb.shape[:2] or labels.dtype.kind not in "iu":
            return False
        selected = labels == semantic_id
        count = int(selected.sum())
        if count < self.config.minimum_visible_pixels:
            return False
        if self.config.helper_visibility_mode == "semantic_instance":
            # Simulator-assisted instance recognition, NOT a YOLO/person detector.
            # Only currently rendered instance pixels count; LOS/floor/range/path
            # gates still run independently and no semantic map reaches policy.
            return True
        pixels = rgb[selected, :3].astype(np.float32)
        blue = ((pixels[:, 2] >= 80) & (pixels[:, 2] > pixels[:, 0]*1.20)
                & (pixels[:, 2] > pixels[:, 1]*1.05))
        return float(blue.mean()) >= self.config.minimum_blue_fraction

    def _line_of_sight(self, state, actor):
        """First collidable ray hit must be the currently visible actor, not a wall."""
        sensors = getattr(state, "sensor_states", {})
        camera = sensors.get("rgb")
        if camera is None or not callable(getattr(self.sim, "cast_ray", None)):
            return False
        try:
            origin = np.asarray(vec3(camera.position))
            dummy = actor["spec"]
            for ratio in (0.6, 0.85, 0.35):
                target = np.asarray(dummy.ground_position) + np.array([0, dummy.height_m*ratio, 0])
                direction = target-origin
                distance = float(np.linalg.norm(direction))
                if not math.isfinite(distance) or distance < 1e-6:
                    continue
                ray = self._ray_factory(origin, direction/distance)
                result = self.sim.cast_ray(ray, max_distance=distance+0.05, buffer_distance=0.0)
                hits = list(result.hits)
                if not hits:
                    continue
                if any(not math.isfinite(float(hit.ray_distance)) or hit.ray_distance < 0 for hit in hits):
                    continue
                hit = min(hits, key=lambda h: float(h.ray_distance))
                if int(hit.object_id) == actor["object_id"] and hit.ray_distance <= distance+0.05:
                    return True
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return False
        return False

    def _interaction_reachable(self, position, dummy):
        """Reach an exterior stand-off point, never plan into the person's body."""
        p, foot = np.asarray(position), np.asarray(dummy.ground_position)
        direction = p-foot
        direction[1] = 0
        distance = float(np.linalg.norm(direction))
        if distance < 1e-8:
            return False
        standoff = dummy.radius_m + self.config.agent_radius_m + self.config.collision_margin_m + 0.05
        target = foot + direction/distance*standoff
        try:
            snapped = np.asarray(vec3(self.sim.pathfinder.snap_point(target)))
        except (AttributeError, TypeError, ValueError, RuntimeError):
            return False
        if np.linalg.norm(snapped-target) > self.config.endpoint_error_m:
            return False
        path = self.find_safe_path(position, snapped)
        return bool(path is not None
                    and self._length(path) <= self.config.max_interaction_path_m
                    and all(abs(q[1]-foot[1]) <= self.config.floor_tolerance_m for q in path))

    def observe(self) -> SensorFrame:
        self._check_open()
        if not self._ready:
            raise RuntimeError("reset must succeed before observation")
        sensors = self.sim.get_sensor_observations()
        if "rgb" not in sensors or "depth" not in sensors:
            raise RuntimeError("Current RGB and depth sensors are required; no fabricated image fallback")
        rgb, depth = np.asarray(sensors["rgb"]), np.asarray(sensors["depth"])
        if (rgb.ndim != 3 or rgb.shape[2] not in (3, 4) or rgb.dtype != np.uint8
                or depth.shape != rgb.shape[:2] or depth.dtype.kind not in "fiu"):
            raise RuntimeError("Sensor arrays have incompatible RGB-D shape/dtype")
        state = self._agent.get_state()
        position = vec3(state.position)
        pose = Pose(position, _yaw(state.rotation))
        candidates = []
        for actor in self._actors:
            dummy = actor["spec"]
            distance = math.hypot(position[0]-dummy.ground_position[0], position[2]-dummy.ground_position[2])
            visible = self._visible(rgb, sensors.get("semantic"), actor["semantic_id"])
            same_floor = abs(position[1]-dummy.ground_position[1]) <= self.config.floor_tolerance_m
            los = visible and self._line_of_sight(state, actor)
            reachable = bool(visible and los and same_floor and distance <= self.config.interaction_radius_m
                             and self._interaction_reachable(position, dummy))
            # Internal state-machine evidence also includes measured out-of-disk distances
            # for hysteresis. The policy payload builder must whitelist visible offers only.
            evidence = evaluate_interaction(robot_ground_position=position,
                dummy_foot_position=dummy.ground_position, dummy_id=dummy.dummy_id,
                visible=visible, line_of_sight_clear=los, same_floor=same_floor,
                navmesh_reachable=reachable, remaining_questions=3, cooldown_ready=True,
                config=RangeConfig(self.config.interaction_radius_m, self.config.floor_tolerance_m))
            evidence["visible"] = bool(visible)
            candidates.append(evidence)
        rgb_copy, depth_copy = np.array(rgb[..., :3], copy=True), np.array(depth, copy=True)
        rgb_copy.setflags(write=False)
        depth_copy.setflags(write=False)
        legal = (() if self.stopped else
                 tuple(a for a in self.ACTIONS if a != "move_forward" or self._forward_is_clear(pose)))
        specs = {"move_forward": {"relative_direction": "forward", "requested_distance_m": self.config.forward_step_m},
                 "turn_left": {"relative_direction": "left", "degrees": self.config.turn_degrees},
                 "turn_right": {"relative_direction": "right", "degrees": self.config.turn_degrees},
                 "wait": {"relative_direction": "stationary"},
                 "stop": {"relative_direction": "stationary"}}
        return SensorFrame(rgb_copy, depth_copy, pose, self.step_count, self._collision, candidates, legal,
                           {action: specs[action] for action in legal})

    def _forward_is_clear(self, pose):
        start = np.asarray(pose.position)
        wanted = start + self.config.forward_step_m*np.array([-math.sin(pose.yaw_rad), 0, -math.cos(pose.yaw_rad)])
        try:
            accepted = vec3(self._base_filter(start, wanted))
            return (self._navigable(start) and self._navigable(accepted)
                    and math.dist(start, accepted) >= self.config.forward_step_m*0.9
                    and self._blocker(start, accepted) is None)
        except (TypeError, ValueError, RuntimeError):
            return False

    def step(self, action: str) -> SensorFrame:
        self._check_open()
        if not self._ready or self.stopped:
            raise RuntimeError("Environment is not active; reset before stepping")
        if action not in self.ACTIONS:
            raise ValueError("Unsupported navigation action")
        before = np.asarray(vec3(self._agent.get_state().position))
        self._proxy_collision, self._collision = False, False
        if action == "stop":
            self.stopped = True
        elif action != "wait":
            result = self.sim.step(action)
            self._collision = self._proxy_collision or bool(result.get("collided", False))
        after = np.asarray(vec3(self._agent.get_state().position))
        self.path_length_m += float(np.linalg.norm(after-before))
        self.step_count += 1
        return self.observe()

    @staticmethod
    def _length(points):
        return sum(math.dist(a, b) for a, b in zip(points, points[1:]))

    def find_safe_path(self, start, end) -> list[tuple[float, float, float]] | None:
        """Shortest NavMesh path validated against the SAME occupancy as motion.

        No reroute is invented when the loaded NavMesh ignores a new static human.
        Missing query support, unsafe segments, floor discontinuities, or a human
        crossing return None. This service does not store/read the episode goal.
        """
        self._check_open()
        self.last_path_check = {"status": "checking"}
        def reject(reason, **details):
            # Diagnostic scalars only; never persist a private requested goal/path.
            self.last_path_check = {"status": "rejected", "reason": reason, **details}
            return None
        start, end = vec3(start), vec3(end)
        pf = self.sim.pathfinder
        if not self._navigable(start) or not self._navigable(end):
            return reject("endpoint_not_navigable")
        query_method = "try_step" if self.config.allow_sliding else "try_step_no_sliding"
        if not callable(getattr(pf, query_method, None)):
            return reject("missing_matching_motion_query", method=query_method)
        try:
            factory = self._shortest_path_factory
            if factory is None:
                from habitat_sim import ShortestPath
                factory = ShortestPath
            query = factory()
            query.requested_start, query.requested_end = np.asarray(start), np.asarray(end)
            if (not pf.find_path(query) or not math.isfinite(float(query.geodesic_distance))
                    or query.geodesic_distance < 0):
                return reject("shortest_path_unavailable")
            points = [vec3(p) for p in query.points]
            if (not points or math.dist(points[0], start) > self.config.endpoint_error_m
                    or math.dist(points[-1], end) > self.config.endpoint_error_m):
                return reject("path_endpoint_mismatch")
            # Include small endpoint offsets: do not leave unchecked start/end gaps.
            points = [start] + points + [end]
            if self.config.allow_sliding:
                return self._walk_sliding_path(points, reject)
            for segment_index, (a, b) in enumerate(zip(points, points[1:])):
                if self._blocker(a, b):
                    return reject("human_body_crossing", segment_index=segment_index)
                # Adapt the verified hsgmv3 WaypointRoutes.segment execution
                # convention: horizontal requests, NavMesh supplies tread height.
                # Unlike that helper, NEVER fall back to try_step/sliding.
                length = math.hypot(b[0]-a[0], b[2]-a[2])
                if length > 1000:  # reject malformed unbounded query results
                    return reject("unbounded_segment", segment_index=segment_index)
                if length < 1e-8 and abs(b[1]-a[1]) > self.config.endpoint_error_m:
                    return reject("vertical_teleport_segment", segment_index=segment_index)
                count = max(1, math.ceil(length/self.config.navmesh_step_m))
                current = np.asarray(a)
                delta_xz = (np.asarray(b)[[0, 2]]-np.asarray(a)[[0, 2]])/count
                for i in range(1, count+1):
                    wanted = current.copy()
                    wanted[[0, 2]] += delta_xz
                    accepted = np.asarray(vec3(pf.try_step_no_sliding(current, wanted)))
                    if (np.linalg.norm((accepted-wanted)[[0, 2]]) > self.config.navmesh_error_m
                            or abs(accepted[1]-current[1]) > self.config.max_step_height_m):
                        return reject("no_sliding_step_mismatch", segment_index=segment_index,
                            substep=i, horizontal_error_m=float(np.linalg.norm((accepted-wanted)[[0,2]])),
                            height_delta_m=float(abs(accepted[1]-current[1])))
                    if self._blocker(current, accepted):
                        return reject("executed_substep_body_crossing", segment_index=segment_index, substep=i)
                    current = accepted
                if math.dist(current, b) > self.config.endpoint_error_m:
                    return reject("segment_endpoint_drift", segment_index=segment_index,
                                  error_m=float(math.dist(current,b)))
            self.last_path_check = {"status": "passed", "segment_count": len(points)-1}
            return [p for i, p in enumerate(points) if i == 0 or math.dist(points[i-1], p) > 1e-8]
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            return reject("query_exception", exception_type=type(exc).__name__)

    def _walk_sliding_path(self, points, reject):
        """Explicit sliding profile: return actual queried polyline, not teleport gaps.

        Each query preserves floor handling and gets the same human sweep as real
        motion. Sliding may deviate at a wall corner, but must be bounded, finite,
        make progress toward the next path corner and meet endpoint tolerances.
        It is not a fallback inside the strict no-sliding experiment.
        """
        pf = self.sim.pathfinder
        current = np.asarray(points[0], dtype=float)
        walked = [vec3(current)]
        if self._blocker(current,current):
            return reject("sliding_start_body_overlap")
        total_steps = 0
        for index, target_tuple in enumerate(points[1:]):
            target = np.asarray(target_tuple, dtype=float)
            initial = float(np.linalg.norm((target-current)[[0,2]]))
            if initial > 1000:
                return reject("unbounded_segment", segment_index=index)
            limit = max(4, 3*math.ceil(initial/self.config.navmesh_step_m)+4)
            for substep in range(limit):
                remaining = float(np.linalg.norm((target-current)[[0,2]]))
                if remaining <= self.config.navmesh_error_m:
                    if abs(target[1]-current[1]) > self.config.endpoint_error_m:
                        return reject("sliding_endpoint_height", segment_index=index)
                    break
                wanted = current.copy()
                requested = min(remaining,self.config.navmesh_step_m)
                wanted[[0,2]] += (target-current)[[0,2]]/remaining*requested
                accepted = np.asarray(vec3(pf.try_step(current,wanted)))
                travelled = float(np.linalg.norm((accepted-current)[[0,2]]))
                new_remaining = float(np.linalg.norm((target-accepted)[[0,2]]))
                if (travelled > requested+self.config.navmesh_error_m
                        or abs(accepted[1]-current[1]) > self.config.max_step_height_m):
                    return reject("sliding_unbounded_motion", segment_index=index, substep=substep)
                if new_remaining >= remaining-1e-6:
                    return reject("sliding_no_progress", segment_index=index, substep=substep)
                if self._blocker(current,accepted):
                    return reject("executed_substep_body_crossing", segment_index=index,substep=substep)
                current=accepted
                walked.append(vec3(current))
                total_steps+=1
                if total_steps>20000:
                    return reject("sliding_query_budget")
            else:
                return reject("sliding_iteration_budget", segment_index=index)
        if math.dist(current,points[-1])>self.config.endpoint_error_m:
            return reject("sliding_final_endpoint_mismatch")
        self.last_path_check={"status":"passed","motion_profile":"sliding", "actual_substeps":total_steps}
        return walked

    def debug_camera(self):
        """Privileged debug projection metadata; never part of policy RGB/payload."""
        state = self._agent.get_state().sensor_states["rgb"]
        return {"camera_position": vec3(state.position),
                "camera_rotation_xyzw": tuple(float(x) for x in _xyzw(state.rotation)),
                "hfov_degrees": self.config.hfov_degrees}

    def close(self):
        if self._closed:
            return
        self._ready = False
        self._remove_owned_actors()  # On failure remain retryable and keep occupancy filter.
        self._agent.controls.move_filter_fn = self._base_filter
        self._closed = True
        if self._owns_simulator:
            self.sim.close()


def create_habitat_sim(scene_path, navmesh_path, *, config=AdapterConfig(), cpu=True):
    """Create RGB/depth/semantic Habitat-Sim with Bullet, level camera, no downloads.

    CPU mode must be accompanied by software-EGL environment variables before
    process startup (see docs/HABITAT_ENV.md). gpu_device_id=-1 skips CUDA-device
    matching. No GPU mode is authorized by the default configuration.
    """
    if cpu is not True:
        raise ValueError("This project factory is CPU-only; no implicit GPU allocation")
    for path in (scene_path, navmesh_path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    import habitat_sim
    import magnum as mn
    sim_config = habitat_sim.SimulatorConfiguration()
    sim_config.scene_id = str(Path(scene_path).resolve())
    sim_config.enable_physics = True
    sim_config.gpu_device_id = -1
    sim_config.allow_sliding = config.allow_sliding
    agent = habitat_sim.agent.AgentConfiguration()
    agent.radius = config.agent_radius_m
    agent.height = config.agent_height_m
    specs = []
    for name, sensor_type in (("rgb", habitat_sim.SensorType.COLOR),
                              ("depth", habitat_sim.SensorType.DEPTH),
                              ("semantic", habitat_sim.SensorType.SEMANTIC)):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid, spec.sensor_type = name, sensor_type
        spec.resolution = [config.height, config.width]
        spec.position = mn.Vector3(0.0, config.camera_height_m, 0.0)
        spec.orientation = mn.Vector3(0.0, 0.0, 0.0)
        spec.hfov = mn.Deg(config.hfov_degrees)
        spec.gpu2gpu_transfer = False
        specs.append(spec)
    agent.sensor_specifications = specs
    agent.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=config.forward_step_m)),
        "turn_left": habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=config.turn_degrees)),
        "turn_right": habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=config.turn_degrees)),
    }
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_config, [agent]))
    try:
        if not sim.pathfinder.load_nav_mesh(str(Path(navmesh_path).resolve())):
            raise RuntimeError("Failed to load supplied NavMesh")
        if not sim.pathfinder.is_loaded:
            raise RuntimeError("NavMesh is not loaded")
        settings = sim.pathfinder.nav_mesh_settings
        if settings is None:
            raise RuntimeError("NavMesh robot dimensions are unknown; validate/rebuild a separate mesh, not the input file")
        if (settings.agent_radius + 1e-6 < config.agent_radius_m
                or settings.agent_height + 1e-6 < config.agent_height_m):
            raise RuntimeError("Loaded NavMesh was built for a smaller robot; build a compatible separate NavMesh")
        return sim
    except Exception:
        sim.close()
        raise


def create_environment(scene_path, navmesh_path, object_config_path, *, config=AdapterConfig()):
    if not Path(object_config_path).is_file():
        raise FileNotFoundError(object_config_path)
    sim = create_habitat_sim(scene_path, navmesh_path, config=config, cpu=True)
    try:
        return HabitatNavigationEnvironment(sim, object_config_path, config=config, owns_simulator=True)
    except Exception:
        sim.close()
        raise

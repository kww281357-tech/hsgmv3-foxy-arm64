import math

from mapping_utils.geometry import *
from mapping_utils.preprocess import *
from mapping_utils.projection import *
from mapping_utils.transform import *
from mapping_utils.path_planning import *

from matplotlib import colormaps
from habitat_sim.utils.common import d3_40_colors_rgb

import open3d as o3d
from segmentation.instance_segmentation import instance_segmentation
from segmentation.object_list import categories
from segmentation.instance_segmentation import get_class_color
import contextlib
import heapq

import time
import numpy as np
import gc
from typing import Any
from waypoint_navigation import representative_points
from semantic_evidence import (canonical_label, detection_observation,
                               add_entity_observation, merge_entity_evidence)

class Instruct_Mapper:
    def __init__(self,
                 camera_intrinsic,
                 pcd_resolution=0.05,
                 grid_resolution=0.1,
                 grid_size=5,
                 floor_height=-1.2,
                 ceiling_height=0.6,
                 translation_func=habitat_translation,
                 rotation_func=habitat_rotation,
                 rotate_axis=[0, 1, 0],
                 resolution=(480, 640),
                 device='cuda:0'):
        self.device = device
        self.camera_intrinsic = camera_intrinsic
        self.pcd_resolution = pcd_resolution
        self.grid_resolution = grid_resolution
        self.grid_size = grid_size
        self.initial_floor_height = floor_height
        self.initial_ceiling_height = ceiling_height
        self.floor_height = floor_height
        self.last_floor_height = self.initial_floor_height  # height at last floor update
        self.ceiling_height = ceiling_height
        self.last_ceiling_height = self.initial_ceiling_height  # height at last floor update
        self.translation_func = translation_func
        self.rotation_func = rotation_func
        self.rotate_axis = np.array(rotate_axis)
        self.pcd_device = o3d.core.Device(device.upper())
        self.current_view_navigable_pcd = o3d.t.geometry.PointCloud(self.pcd_device)  # new: added this line
        self.stair_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.stair_tread_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.stair_portals = []  # Persistent stair entry/exit portals across floor switches
        self.observation_id = 0
        self.observation_step = None
        self.current_observation_masks = {}
        self._map_revision = 0
        self._path_planning_cache = None
        self._path_cache_build_count = 0

        self.floor_level = 0
        self.floor_height_difference = None  # floor height
        self.last_stable_z = None  # last stable height
        self.last_z = None  # last changed height
        self.height_change_timestamp = None  # timestamp when height started changing
        self.is_changing_floor = False  # whether floor change is being detected
        self.stable_update_counter = 0
        self.resolution = resolution

        self.pcds_per_floor = {}
        self._floor_departure_data = None
        self._landing_z = None
        self._anchor_confirmation_counter = 0
        self._anchor_candidate_z = None
        self._floor_candidate_z = None
        self._floor_candidate_origin_xy = None
        self._switch_to_floor(0, 0)

        self.local_stair_radius = 2.5
        self.local_stair_min_points = 80
        self.enable_stair_detection = True
        self.trajectory_position = []

        self.waypoints = np.array([])

    def _switch_to_floor(self, floor_level, height_diff, last_stable_z=0):
        """Switch to the specified floor, loading or initializing its point cloud and instance data."""

        if self.floor_level in self.pcds_per_floor:
            print(f"Saving data for floor level: {self.floor_level}")
            self.pcds_per_floor[self.floor_level]['scene_pcd'] = self.scene_pcd
            self.pcds_per_floor[self.floor_level]['navigable_pcd'] = self.navigable_pcd
            self.pcds_per_floor[self.floor_level]['obstacle_pcd'] = self.obstacle_pcd
            self.pcds_per_floor[self.floor_level]['trajectory_position'] = self.trajectory_position
            self.pcds_per_floor[self.floor_level]['object_entities'] = self.object_entities

        if self.floor_level in self.pcds_per_floor and getattr(self, '_floor_departure_data', None):
            self.pcds_per_floor[self.floor_level] = self._floor_departure_data

        if floor_level not in self.pcds_per_floor:
            init_scene = o3d.t.geometry.PointCloud(self.pcd_device)
            init_nav = o3d.t.geometry.PointCloud(self.pcd_device)
            init_obs = o3d.t.geometry.PointCloud(self.pcd_device)

            if hasattr(self, 'current_pcd') and not self.current_pcd.is_empty():
                try:
                    pts_z = self.current_pcd.point.positions[:, 2]
                    target_floor_h = last_stable_z + height_diff + self.initial_floor_height
                    floor_mask = (pts_z >= target_floor_h - 0.25) & (pts_z <= target_floor_h + 0.25)
                    floor_indices = floor_mask.nonzero()[0]
                    if floor_indices.shape[0] > 0:
                        init_nav = self.current_pcd.select_by_index(floor_indices)
                        init_scene = self.current_pcd.clone()
                except Exception as e:
                    print(f"Error seeding new floor pointcloud: {e}")

            self.pcds_per_floor[floor_level] = {
                'scene_pcd': init_scene,
                'navigable_pcd': init_nav,
                'obstacle_pcd': init_obs,
                'trajectory_position': [],
                'object_entities': [],
                'floor_height': last_stable_z + height_diff + self.initial_floor_height,
            }
            print(f"Initialized data for new floor level: {floor_level}")

        print("Switching pcds to floor level:", floor_level)
        current_floor_data = self.pcds_per_floor[floor_level]
        self.scene_pcd = current_floor_data['scene_pcd']
        self.navigable_pcd = current_floor_data['navigable_pcd']
        self.obstacle_pcd = current_floor_data['obstacle_pcd']
        self.trajectory_position = current_floor_data['trajectory_position']
        self.object_entities = current_floor_data['object_entities']
        self.floor_level = floor_level

        self.floor_height = current_floor_data['floor_height']
        self.ceiling_height = self.floor_height - self.initial_floor_height + self.initial_ceiling_height
        self.last_floor_height = self.floor_height
        self.last_ceiling_height = self.ceiling_height
        self.waypoints = np.array([])
        self._map_revision = getattr(self, '_map_revision', 0) + 1
        self._path_planning_cache = None

    def _calculate_floor_height_difference(self):
        """
        Compute floor height from ceiling-floor height difference in point cloud.
        """
        if self.scene_pcd.is_empty():
            return None
        points_z = self.scene_pcd.point.positions[:, 2].cpu().numpy()

        floor_z = np.percentile(points_z, 5)
        ceiling_z = np.percentile(points_z, 95)

        height_diff = ceiling_z - floor_z

        if 2.0 < height_diff < 5.0:
            self.floor_height_difference = height_diff
            print(f"Calculated floor height difference: {self.floor_height_difference:.2f}m")
        return self.floor_height_difference

    def _detect_and_update_floor(self):
        """Confirm a landing over several observations, using a fixed floor anchor.

        The visible ceiling span is not a reliable storey height. Stair proximity
        is also not a veto: a valid upper landing is usually next to stairs.
        """
        current_z = float(self.current_position[2])
        if self.last_z is None:
            self.last_z = current_z
        delta = current_z - (self.last_floor_height - self.initial_floor_height)
        # The local navigation slice follows our feet even while climbing.
        self.floor_height = current_z + self.initial_floor_height
        self.ceiling_height = current_z + self.initial_ceiling_height
        if abs(delta) < 0.4:
            self._floor_departure_data = None
        elif getattr(self, '_floor_departure_data', None) is None:
            # Preserve the departure map before the moving height slice clips it.
            self._floor_departure_data = {
                key: getattr(self, key).clone()
                for key in ('scene_pcd', 'navigable_pcd', 'obstacle_pcd')
            }
            self._floor_departure_data.update(
                trajectory_position=list(self.trajectory_position),
                object_entities=list(self.object_entities),
                floor_height=self.last_floor_height,
            )

        moved_z = abs(current_z - self.last_z) > 0.08
        self.last_z = current_z
        self.is_changing_floor = abs(delta) >= 0.4
        # Re-evaluate on every frame: a landing can become visible after a turn.
        if moved_z:
            self.stable_update_counter = 0
            self._landing_z = None
            self._anchor_confirmation_counter = 0
            self._anchor_candidate_z = None
            candidate_z = getattr(self, '_floor_candidate_z', None)
            if candidate_z is None or abs(current_z - candidate_z) > 0.12:
                self._floor_candidate_z = None
                self._floor_candidate_origin_xy = None
            return
        evidence = self.current_pcd if (hasattr(self, 'current_pcd') and not self.current_pcd.is_empty()) else self.scene_pcd
        foot_z = current_z + self.initial_floor_height
        known_elevation = any(
            level != self.floor_level and abs(data.get('floor_height', float('inf')) - foot_z) <= 0.35
            for level, data in self.pcds_per_floor.items())
        required_area = 0.6 if known_elevation else 0.75

        # Check for anchor refinement on the SAME floor (e.g. multi-flight stairs continuation)
        if (self.floor_level != 0 and self.floor_level * delta > 0
                and 0.25 <= abs(delta) < 1.6):
            flag, metrics = self.is_on_platform(evidence, min_area=0.6)
            if not flag and hasattr(self, 'scene_pcd') and not self.scene_pcd.is_empty():
                scene_flag, scene_metrics = self.is_on_platform(self.scene_pcd, min_area=0.6)
                if scene_flag:
                    flag, metrics = scene_flag, scene_metrics
            supported = (flag and metrics.get('vertical_diff', float('inf')) <= 0.18
                         and metrics.get('area_xy', 0.0) >= 0.6
                         and metrics.get('z_std', float('inf')) <= 0.08)
            candidate_z = getattr(self, '_anchor_candidate_z', None)
            if not supported:
                self._anchor_confirmation_counter = 0
                self._anchor_candidate_z = None
            elif candidate_z is None or abs(current_z - candidate_z) > 0.12:
                self._anchor_candidate_z = current_z
                self._anchor_confirmation_counter = 1
            else:
                self._anchor_confirmation_counter += 1

            if self._anchor_confirmation_counter >= 3:
                self.last_floor_height = foot_z
                self.last_ceiling_height = current_z + self.initial_ceiling_height
                self.pcds_per_floor[self.floor_level]['floor_height'] = foot_z
                self._floor_departure_data = None
                self.last_stable_z = current_z
                self.is_changing_floor = False
                self._anchor_confirmation_counter = 0
                self._anchor_candidate_z = None
                print(f"[FloorTransition] refined level={self.floor_level} floor_z={foot_z:.3f}")
            return

        self._anchor_confirmation_counter = 0
        self._anchor_candidate_z = None

        # For a new floor or returning to an existing floor, require |delta| >= 0.7m (or known elevation)
        if abs(delta) < 0.7 and not known_elevation:
            self.stable_update_counter = 0
            self._landing_z = None
            self._floor_candidate_z = None
            self._floor_candidate_origin_xy = None
            return

        flag, metrics = self.is_on_platform(evidence, min_area=required_area)
        if not flag and hasattr(self, 'scene_pcd') and not self.scene_pcd.is_empty():
            scene_flag, scene_metrics = self.is_on_platform(self.scene_pcd, min_area=required_area)
            if scene_flag:
                flag, metrics = scene_flag, scene_metrics

        local_support = (metrics.get('vertical_diff', float('inf')) <= 0.18
                         and metrics.get('z_std', float('inf')) <= 0.08)
        area = metrics.get('area_xy', 0.0)
        candidate_z = getattr(self, '_floor_candidate_z', None)
        if local_support and area >= 0.45:
            if candidate_z is None or abs(current_z - candidate_z) > 0.12:
                self._floor_candidate_z = current_z
                self._floor_candidate_origin_xy = self.current_position[:2].copy()
        else:
            self._floor_candidate_z = None
            self._floor_candidate_origin_xy = None

        origin_xy = getattr(self, '_floor_candidate_origin_xy', None)
        level_travel = (float(np.linalg.norm(self.current_position[:2] - origin_xy))
                        if origin_xy is not None else 0.0)
        landing = flag and local_support and area >= required_area
        if not landing and not known_elevation and local_support and area >= 0.6:
            scene_flag, scene_metrics = self.is_on_platform(self.scene_pcd, min_area=0.7)
            scene_support = (scene_flag and scene_metrics.get('area_xy', 0.0) >= 0.7
                             and scene_metrics.get('vertical_diff', float('inf')) <= 0.18
                             and scene_metrics.get('z_std', float('inf')) <= 0.08)
            landing = scene_support or level_travel >= 0.8

        candidate_z = getattr(self, '_landing_z', None)
        if not landing:
            self.stable_update_counter = 0
            self._landing_z = None
        elif candidate_z is None or abs(current_z - candidate_z) > 0.12:
            self._landing_z = current_z
            self.stable_update_counter = 1
        else:
            self.stable_update_counter += 1

        print(f"[FloorTransition] level={self.floor_level} delta={delta:.3f} "
              f"landing={landing} confirmations={self.stable_update_counter}/3 "
              f"area={area:.3f}/{required_area:.2f} travel={level_travel:.2f} known={known_elevation} "
              f"support_error={metrics.get('vertical_diff', float('inf')):.3f}")

        if self.stable_update_counter < 3:
            return

        target_height = current_z + self.initial_floor_height
        matches = [(abs(data.get('floor_height', float('inf')) - target_height), level)
                   for level, data in self.pcds_per_floor.items() if level != self.floor_level]
        match = min(matches, default=(float('inf'), None))

        between_known_floors = any(
            min(self.last_floor_height, data['floor_height']) + 0.3 < target_height
            < max(self.last_floor_height, data['floor_height']) - 0.3
            for level, data in self.pcds_per_floor.items() if level != self.floor_level)

        if match[0] > 0.45 and between_known_floors:
            self.stable_update_counter = 0
            self._landing_z = None
            return

        if match[0] <= 0.45:
            new_level = match[1]
        else:
            candidate_level = self.floor_level + (1 if delta > 0 else -1)
            if candidate_level in self.pcds_per_floor:
                new_level = (max(self.pcds_per_floor) + 1 if delta > 0
                             else min(self.pcds_per_floor) - 1)
            else:
                new_level = candidate_level

            base_floor_h = self.pcds_per_floor.get(0, {}).get('floor_height', self.initial_floor_height)
            if target_height < base_floor_h - 0.5 and new_level > 0:
                new_level = min(self.pcds_per_floor) - 1 if min(self.pcds_per_floor) <= 0 else -1
            elif target_height > base_floor_h + 0.5 and new_level < 0:
                new_level = max(self.pcds_per_floor) + 1 if max(self.pcds_per_floor) >= 0 else 1

        self._switch_to_floor(new_level, 0, current_z)
        self._floor_departure_data = None
        self._floor_candidate_z = None
        self._floor_candidate_origin_xy = None
        self.last_stable_z = current_z
        self.stable_update_counter = 0
        self._landing_z = None
        self.is_changing_floor = False
        print(f"[FloorTransition] switched level={new_level} floor_z={self.floor_height:.3f}")

    def reset(self, position, rotation):
        self.update_iterations = 0
        self.observation_id = 0
        self.observation_step = None
        self.current_observation_masks = {}
        self.initial_position = self.translation_func(position)
        self.current_position = self.translation_func(position) - self.initial_position
        self.current_rotation = self.rotation_func(rotation)

        self.pcds_per_floor.clear()
        self.current_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self._floor_departure_data = None
        self._landing_z = None
        self._anchor_confirmation_counter = 0
        self._anchor_candidate_z = None
        self._floor_candidate_z = None
        self._floor_candidate_origin_xy = None
        self.floor_height_difference = None
        self.last_floor_height = self.initial_floor_height
        self.last_ceiling_height = self.initial_ceiling_height
        self.floor_level = 0
        self.floor_height = self.initial_floor_height
        self.ceiling_height = self.initial_ceiling_height
        self.last_stable_z = None
        self.last_z = None
        self.is_changing_floor = False
        self.stable_update_counter = 0
        self.current_view_navigable_pcd = o3d.t.geometry.PointCloud(self.pcd_device)  # new: added this line
        self.stair_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.stair_tread_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

        self._switch_to_floor(0, 0)
        self._map_revision = getattr(self, '_map_revision', 0) + 1
        self._path_planning_cache = None

    def get_candidate_waypoints(self,
                                waypoint_grid_resolution=1.0,
                                min_distance=0.0,
                                max_distance=5.0,
                                cylinder_radius=0.2,
                                cylinder_height=1.0,
                                use_distance_filter=False):
        """
        Generate candidate waypoints using cylinder-occupancy rules:
        1. voxel downsample to candidates
        2. for each candidate, build a cylinder (base center at point, radius=cylinder_radius, height=cylinder_height)
        3. check the complete body cylinder; only coincident low stair
           tread support is exempt, never nearby walls or railings
        4. optionally apply 2D distance filter against obstacles
        Returns: (N,3) numpy.ndarray
        """
        has_stair = hasattr(self, 'stair_pcd') and not self.stair_pcd.is_empty()
        if self.navigable_pcd.is_empty() and not has_stair:
            self.waypoints = np.array([])
            return np.array([])

        try:
            if has_stair and not self.navigable_pcd.is_empty():
                base_pcd = gpu_merge_pointcloud(self.navigable_pcd, self.stair_pcd)
            elif has_stair:
                base_pcd = self.stair_pcd
            else:
                base_pcd = self.navigable_pcd

            if self.obstacle_pcd.is_empty():
                try:
                    cand = base_pcd.voxel_down_sample(waypoint_grid_resolution)
                    self.waypoints = cand.point.positions.cpu().numpy()
                    return self.waypoints
                except:
                    self.waypoints = np.array([])
                    return np.array([])

            candidate_waypoints_pcd = base_pcd.voxel_down_sample(waypoint_grid_resolution)
            if candidate_waypoints_pcd.is_empty():
                self.waypoints = np.array([])
                return np.array([])

            cand_pts = candidate_waypoints_pcd.point.positions.cpu().numpy()

            obs_legacy = self.obstacle_pcd.to_legacy()
            obs_kdtree = o3d.geometry.KDTreeFlann(obs_legacy)

            stair_kdtree = None
            if has_stair:
                stair_legacy = self.stair_pcd.to_legacy()
                stair_kdtree = o3d.geometry.KDTreeFlann(stair_legacy)

            obstacle_pts = self.obstacle_pcd.point.positions.cpu().numpy()
            stair_pts = self.stair_pcd.point.positions.cpu().numpy() if has_stair else None

            valid_flags = np.zeros(cand_pts.shape[0], dtype=bool)

            for i, p in enumerate(cand_pts):
                valid_flags[i] = self._waypoint_body_clear(
                    p, obs_kdtree, obstacle_pts, stair_kdtree,
                    cylinder_radius, cylinder_height)

            if not np.any(valid_flags):
                self.waypoints = np.array([])
                return np.array([])

            if use_distance_filter:
                distance_to_obstacle = pointcloud_2d_distance(candidate_waypoints_pcd, self.obstacle_pcd)
                dist_np = distance_to_obstacle.cpu().numpy()
                dist_mask = (dist_np >= min_distance) & (dist_np < max_distance)
                valid_flags = valid_flags & dist_mask
                if not np.any(valid_flags):
                    self.waypoints = np.array([])
                    return np.array([])

            indices_np = np.where(valid_flags)[0]
            indices_tensor = o3d.core.Tensor(indices_np, device=self.pcd_device)
            final_pcd = candidate_waypoints_pcd.select_by_index(indices_tensor)
            self.waypoints = final_pcd.point.positions.cpu().numpy()
            return self.waypoints

        except Exception as e:
            print(f"[get_candidate_waypoints] Error: {e}")
            self.waypoints = np.array([])
            return np.array([])

    def _denoise_pcd(self, pcd: o3d.t.geometry.PointCloud,
                     nb_neighbors: int = 20,
                     std_ratio: float = 1.0,
                     radius: float = 0.30,
                     min_points: int = 10) -> o3d.t.geometry.PointCloud:
        """
        Two-stage denoising:
        1) statistical outlier removal (mean distance > mean + std_ratio * std)
        2) radius outlier removal (fewer than min_points neighbors)
        compatible with tensor/legacy conversion.
        """
        try:
            if pcd.is_empty() or pcd.point.positions.shape[0] < (min_points * 2):
                return pcd

            legacy = pcd.to_legacy()
            legacy_sor, _ = legacy.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
            legacy_rad, _ = legacy_sor.remove_radius_outlier(nb_points=min_points, radius=radius)
            return o3d.t.geometry.PointCloud.from_legacy(legacy_rad, self.pcd_device)
        except Exception:
            return pcd

    def get_current_view_candidate_waypoints(self,
                                             waypoint_grid_resolution=1.0,
                                             min_distance=0.1,
                                             max_distance=1.0,
                                             merge_distance=0.5,
                                             cylinder_radius=0.2,
                                             cylinder_height=1.2,
                                             use_distance_filter=False,
                                             max_view_angle_deg=None,
                                             head_clearance=0.6,
                                             head_check_radius=0.30):
        has_stair = hasattr(self, 'stair_pcd') and (not self.stair_pcd.is_empty())
        if self.current_view_navigable_pcd.is_empty() and not has_stair:
            return np.array([]), np.array([])

        try:
            base_view_pcd = self.current_view_navigable_pcd
            if has_stair and not base_view_pcd.is_empty():
                base_view_pcd = gpu_merge_pointcloud(base_view_pcd, self.stair_pcd)
            elif has_stair:
                base_view_pcd = self.stair_pcd
            src = self._denoise_pcd(base_view_pcd,
                                    nb_neighbors=20, std_ratio=1.0,
                                    radius=0.35, min_points=12)

            # A voxel centroid can lie between stair treads or inside a jamb.
            # Preserve an actually observed support point instead.
            sampled = representative_points(src.point.positions.cpu().numpy(), waypoint_grid_resolution)
            candidate_waypoints_pcd = gpu_pointcloud_from_array(
                sampled, np.zeros_like(sampled), self.pcd_device)
            if candidate_waypoints_pcd.is_empty():
                return np.array([]), np.array([])

            cand_pts = candidate_waypoints_pcd.point.positions.cpu().numpy()

            has_obstacle = not self.obstacle_pcd.is_empty()
            if has_obstacle:
                obs_legacy = self.obstacle_pcd.to_legacy()
                obs_kdtree = o3d.geometry.KDTreeFlann(obs_legacy)
                obstacle_pts = self.obstacle_pcd.point.positions.cpu().numpy()
            else:
                obs_kdtree = None
                obstacle_pts = None

            if has_stair:
                stair_legacy = self.stair_pcd.to_legacy()
                stair_kdtree = o3d.geometry.KDTreeFlann(stair_legacy)
                stair_pts = self.stair_pcd.point.positions.cpu().numpy()
            else:
                stair_kdtree = None
                stair_pts = None

            valid_flags = np.zeros(cand_pts.shape[0], dtype=bool)

            for i, p in enumerate(cand_pts):
                valid_flags[i] = self._waypoint_body_clear(
                    p, obs_kdtree, obstacle_pts, stair_kdtree,
                    cylinder_radius, cylinder_height)

            if not np.any(valid_flags):
                return np.array([]), np.array([])

            if use_distance_filter and has_obstacle:
                distance_to_obstacle = pointcloud_2d_distance(candidate_waypoints_pcd, self.obstacle_pcd).cpu().numpy()
                dist_mask = (distance_to_obstacle >= min_distance) & (distance_to_obstacle < max_distance)
                valid_flags = valid_flags & dist_mask
                if not np.any(valid_flags):
                    return np.array([]), np.array([])

            indices_np = np.where(valid_flags)[0]
            indices_tensor = o3d.core.Tensor(indices_np, device=self.pcd_device)
            final_waypoints_pcd = candidate_waypoints_pcd.select_by_index(indices_tensor)
            waypoints_world = final_waypoints_pcd.point.positions.cpu().numpy()

            if merge_distance and merge_distance > 0 and waypoints_world.shape[0] > 1:
                cell_size = float(merge_distance)
                voxel_idx = np.floor(waypoints_world / cell_size + 0.5).astype(np.int32)
                keys = [tuple(v) for v in voxel_idx]
                accum = {}
                for k, pp in zip(keys, waypoints_world):
                    if k in accum:
                        # Keep a real support sample; never average two treads.
                        pass
                    else:
                        accum[k] = [pp.copy(), 1]
                waypoints_world = np.vstack([v[0] / v[1] for v in accum.values()]).astype(np.float32)

            if waypoints_world.shape[0] > 0 and max_view_angle_deg is not None:
                R = self.current_rotation
                if R.shape == (4, 4):
                    R = R[:3, :3]

                local_forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
                forward_world = (R @ local_forward).astype(np.float32)

                forward2d = forward_world[:2]
                n = np.linalg.norm(forward2d)
                if n < 1e-6:
                    forward_world = R @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
                    forward2d = forward_world[:2]
                    n = np.linalg.norm(forward2d)

                if n > 1e-6:
                    forward2d /= n
                    vecs2d = waypoints_world[:, :2] - self.current_position[:2]
                    vnorm = np.linalg.norm(vecs2d, axis=1, keepdims=True) + 1e-8
                    dir2d = vecs2d / vnorm

                    dist2d = np.linalg.norm(vecs2d, axis=1)
                    close_keep_radius = 1.0
                    close_mask = dist2d <= float(close_keep_radius)

                    cosv = np.clip(np.sum(dir2d * forward2d[None, :], axis=1), -1.0, 1.0)
                    angles_deg = np.degrees(np.arccos(cosv))
                    ang_mask = angles_deg <= float(max_view_angle_deg)

                    combined_mask = close_mask | ang_mask
                    if not np.any(combined_mask):
                        return np.array([]), np.array([])

                    waypoints_world = waypoints_world[combined_mask]

            if waypoints_world.shape[0] == 0:
                return np.array([]), np.array([])

            ground_z = self.current_position[2] + self.initial_floor_height
            has_stair = hasattr(self, 'stair_pcd') and not self.stair_pcd.is_empty()
            max_allowed_z = ground_z + (2.4 if has_stair else 0.35)
            min_allowed_z = ground_z - (2.8 if has_stair else 0.35)
            hmask = (waypoints_world[:, 2] <= max_allowed_z) & (waypoints_world[:, 2] >= min_allowed_z)
            if not np.any(hmask):
                return np.array([]), np.array([])

            waypoints_world = waypoints_world[hmask]
            waypoints_relative = waypoints_world - self.current_position
            return waypoints_world, waypoints_relative[:, [0, 2, 1]]

        except Exception as e:
            print(f"[get_current_view_candidate_waypoints] Error: {e}")
            return np.array([]), np.array([])

    def _waypoint_body_clear(self, ground, obstacle_tree, obstacle_points, stair_tree,
                             radius, height):
        """Check the full body cylinder, not two small spheres at fixed heights."""
        if obstacle_tree is None:
            return True
        ground = np.asarray(ground, dtype=float)
        center = ground + [0, 0, height / 2]
        _, indices, _ = obstacle_tree.search_radius_vector_3d(
            center, float(np.hypot(radius, height / 2)))
        if not len(indices):
            return True
        nearby = np.asarray(obstacle_points)[indices]
        delta = nearby - ground
        inside = (np.linalg.norm(delta[:, :2], axis=1) <= radius) & (delta[:, 2] > .10) & (delta[:, 2] < height)
        for point in nearby[inside]:
            # Only a coincident low tread/riser may be stair support. Nearby
            # stairs never exempt a wall, railing or overhead obstacle.
            if stair_tree is not None and point[2] - ground[2] <= .40:
                k, _, _ = stair_tree.search_radius_vector_3d(point, .10)
                if k:
                    continue
            return False
        return True

    def _is_path_clear(self, p1, p2, navigable_kdtree, obstacle_kdtree, stair_kdtree=None, check_radius=0.1,
                       obstacle_radius=0.1, step_size=0.1, obstacle_points=None, stair_points=None,
                       diagnostics=None, stair_tread_kdtree=None, stair_tread_points=None):
        """2D distance-based path reachability check (obstacle-free), considering stairs."""
        vec_2d = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        dist_2d = np.linalg.norm(vec_2d)
        if dist_2d < step_size:
            return True
        num_steps = int(dist_2d / step_size)
        if num_steps == 0:
            return True
        dir_vec_2d = vec_2d / dist_2d
        ground_z = self.floor_height
        obs_pts_arr = np.asarray(obstacle_points, dtype=float) if obstacle_points is not None else None

        for i in range(1, num_steps):
            fraction = float(i) / float(num_steps)
            inter_point_2d = np.array([p1[0], p1[1]]) + dir_vec_2d * (i * step_size)
            inter_z = float(p1[2] + fraction * (p2[2] - p1[2])) if (
                abs(p2[2] - p1[2]) > 0.05 or abs(p1[2] - ground_z) > 0.05 or abs(p2[2] - ground_z) > 0.05
            ) else float(ground_z)
            check_point = np.array([inter_point_2d[0], inter_point_2d[1], inter_z])
            check_point_2 = np.array([inter_point_2d[0], inter_point_2d[1], inter_z + 0.4])
            check_point_3 = np.array([inter_point_2d[0], inter_point_2d[1], inter_z + 0.8])

            k_nav = 0
            if navigable_kdtree is not None:
                k_nav, _, _ = navigable_kdtree.search_radius_vector_3d(check_point, max(check_radius, 0.2))
            k_stair_ground = 0
            if stair_kdtree is not None:
                k_stair_ground, _, _ = stair_kdtree.search_radius_vector_3d(check_point, max(check_radius, 0.25))

            supported = (k_nav + k_stair_ground) >= 3
            if not supported and stair_tread_kdtree is not None:
                k_tread, _, _ = stair_tread_kdtree.search_radius_vector_3d(check_point, max(check_radius, 0.25))
                if k_tread >= 1:
                    supported = True
                    if isinstance(diagnostics, dict):
                        diagnostics['stair_tread_supports'] = int(diagnostics.get('stair_tread_supports', 0)) + 1

            if not supported:
                if isinstance(diagnostics, dict):
                    diagnostics['no_floor_support'] = int(diagnostics.get('no_floor_support', 0)) + 1
                return False

            if obstacle_kdtree is not None:
                k_obs, idx_obs_2, _ = obstacle_kdtree.search_radius_vector_3d(check_point_2, obstacle_radius)
                k_obs_2, idx_obs_3, _ = obstacle_kdtree.search_radius_vector_3d(check_point_3, obstacle_radius)

                if k_obs + k_obs_2 > 0:
                    exempt = False
                    if stair_kdtree is not None:
                        k_stair_2, _, _ = stair_kdtree.search_radius_vector_3d(check_point_2, 0.35)
                        k_stair_3, _, _ = stair_kdtree.search_radius_vector_3d(check_point_3, 0.35)
                        if k_stair_2 + k_stair_3 >= 3:
                            hit_indices = list(idx_obs_2) + list(idx_obs_3)
                            if obs_pts_arr is not None and obs_pts_arr.ndim == 2 and len(obs_pts_arr) > 0 and hit_indices:
                                all_on_stair = True
                                for oi in hit_indices:
                                    if 0 <= int(oi) < len(obs_pts_arr):
                                        op = obs_pts_arr[int(oi)]
                                        if hasattr(stair_kdtree, 'search_knn_vector_3d'):
                                            kn, _, d2 = stair_kdtree.search_knn_vector_3d(op, 1)
                                            if not kn or float(d2[0]) > 0.05 ** 2:
                                                all_on_stair = False
                                                break
                                        else:
                                            kr, _, _ = stair_kdtree.search_radius_vector_3d(op, 0.05)
                                            if not kr:
                                                all_on_stair = False
                                                break
                                exempt = all_on_stair
                            elif obs_pts_arr is None:
                                exempt = True
                    if exempt:
                        if isinstance(diagnostics, dict):
                            diagnostics['stair_surface_exemptions'] = int(diagnostics.get('stair_surface_exemptions', 0)) + 1
                        continue
                    if isinstance(diagnostics, dict):
                        diagnostics['obstacle_collisions'] = int(diagnostics.get('obstacle_collisions', 0)) + 1
                    return False

        return True

    def _waypoint_signature(self):
        wp = getattr(self, 'waypoints', None)
        if wp is None or not isinstance(wp, np.ndarray) or wp.size == 0:
            return (0,)
        arr = np.ascontiguousarray(wp[:, :3], dtype=np.float32)
        return (arr.shape, float(arr.sum()), float(arr[0, 0]), float(arr[-1, -1]))

    def _extract_validated_tread_centers(self):
        tread_pcd = getattr(self, 'stair_tread_pcd', None)
        if tread_pcd is None or tread_pcd.is_empty():
            return np.empty((0, 3), dtype=np.float32)
        try:
            pts = np.asarray(tread_pcd.point.positions.cpu().numpy(), dtype=np.float32)
            if pts.ndim != 2 or pts.shape[0] < 10:
                return np.empty((0, 3), dtype=np.float32)
            z_bins = np.round(pts[:, 2] / 0.08).astype(np.int32)
            centers = []
            for zb in np.unique(z_bins):
                sub = pts[z_bins == zb]
                if sub.shape[0] >= 8:
                    centers.append(sub.mean(axis=0))
            return np.asarray(centers, dtype=np.float32) if centers else np.empty((0, 3), dtype=np.float32)
        except Exception:
            return np.empty((0, 3), dtype=np.float32)

    def invalidate_path_cache(self):
        self._path_planning_cache = None

    def is_path_cache_valid(self, max_distance_to_neighbor=1.0):
        cache = getattr(self, '_path_planning_cache', None)
        if not isinstance(cache, dict):
            return False
        pos = tuple(np.asarray(self.current_position[:3], dtype=np.float32).tolist())
        return (
            cache.get('map_revision') == getattr(self, '_map_revision', 0)
            and cache.get('position') == pos
            and cache.get('waypoint_sig') == self._waypoint_signature()
            and abs(float(cache.get('max_distance_to_neighbor', 1.0)) - float(max_distance_to_neighbor)) <= 1e-6
        )

    def prepare_path_planning_cache(self, max_distance_to_neighbor=1.0):
        navigable_kdtree = None if self.navigable_pcd.is_empty() else o3d.geometry.KDTreeFlann(
            self.navigable_pcd.to_legacy())
        obstacle_kdtree = None if self.obstacle_pcd.is_empty() else o3d.geometry.KDTreeFlann(
            self.obstacle_pcd.to_legacy())
        obstacle_points = None if self.obstacle_pcd.is_empty() else np.asarray(
            self.obstacle_pcd.point.positions.cpu().numpy(), dtype=np.float32)
        stair_kdtree = None if self.stair_pcd.is_empty() else o3d.geometry.KDTreeFlann(self.stair_pcd.to_legacy())
        stair_points = None if self.stair_pcd.is_empty() else np.asarray(
            self.stair_pcd.point.positions.cpu().numpy(), dtype=np.float32)
        tread_pcd = getattr(self, 'stair_tread_pcd', None)
        stair_tread_kdtree = None if (tread_pcd is None or tread_pcd.is_empty()) else o3d.geometry.KDTreeFlann(
            tread_pcd.to_legacy())
        stair_tread_points = None if (tread_pcd is None or tread_pcd.is_empty()) else np.asarray(
            tread_pcd.point.positions.cpu().numpy(), dtype=np.float32)
        tread_centers = self._extract_validated_tread_centers()

        start_point = (self.current_position - np.array([0, 0, 1.2], dtype=np.float32))[:3]
        wp = self.waypoints[:, :3] if (isinstance(self.waypoints, np.ndarray) and self.waypoints.size > 0) else np.empty((0, 3), dtype=np.float32)
        parts = [p for p in (wp, tread_centers, start_point[None, :]) if p.shape[0] > 0]
        base_nodes = np.vstack(parts).astype(np.float32)
        nodes_unique, unique_idx = np.unique(base_nodes, axis=0, return_index=True)
        base_nodes = nodes_unique[np.argsort(unique_idx)]

        node_kdtree = None
        base_neighbors = []
        max_step_z = 0.30 if (tread_centers.shape[0] > 0 or stair_kdtree is None) else 0.45
        if base_nodes.shape[0] > 0:
            node_pcd = o3d.geometry.PointCloud()
            node_pcd.points = o3d.utility.Vector3dVector(base_nodes.astype(np.float64))
            node_kdtree = o3d.geometry.KDTreeFlann(node_pcd)
            base_neighbors = [[] for _ in range(base_nodes.shape[0])]
            for i in range(base_nodes.shape[0]):
                p = base_nodes[i].astype(np.float64)
                k, idxs, _ = node_kdtree.search_radius_vector_3d(p, float(max_distance_to_neighbor) + 1e-6)
                if k <= 1:
                    continue
                pz = base_nodes[i, 2]
                base_neighbors[i] = [j for j in idxs if j != i and abs(float(pz - base_nodes[j, 2])) <= max_step_z]

        self._path_cache_build_count = int(getattr(self, '_path_cache_build_count', 0)) + 1
        self._path_planning_cache = {
            'map_revision': getattr(self, '_map_revision', 0),
            'position': tuple(np.asarray(self.current_position[:3], dtype=np.float32).tolist()),
            'waypoint_sig': self._waypoint_signature(),
            'max_distance_to_neighbor': float(max_distance_to_neighbor),
            'navigable_kdtree': navigable_kdtree,
            'obstacle_kdtree': obstacle_kdtree,
            'obstacle_points': obstacle_points,
            'stair_kdtree': stair_kdtree,
            'stair_points': stair_points,
            'stair_tread_kdtree': stair_tread_kdtree,
            'stair_tread_points': stair_tread_points,
            'tread_centers': tread_centers,
            'base_nodes': base_nodes,
            'node_kdtree': node_kdtree,
            'base_neighbors': base_neighbors,
            'max_step_z': max_step_z,
            'hit_count': 0,
        }
        return self._path_planning_cache

    @contextlib.contextmanager
    def path_planning_context(self, max_distance_to_neighbor=1.0):
        self.prepare_path_planning_cache(max_distance_to_neighbor=max_distance_to_neighbor)
        try:
            yield self._path_planning_cache
        finally:
            self.invalidate_path_cache()

    def plan_paths_to_targets(self, targets, **kwargs):
        had_valid_cache = self.is_path_cache_valid(
            max_distance_to_neighbor=kwargs.get('max_distance_to_neighbor', 1.0)
        )
        if not had_valid_cache:
            with self.path_planning_context(max_distance_to_neighbor=kwargs.get('max_distance_to_neighbor', 1.0)):
                return [self.plan_path_to_target(tg, **kwargs) for tg in targets]
        return [self.plan_path_to_target(tg, **kwargs) for tg in targets]

    def plan_path_to_target(self, target_point, max_distance_to_neighbor=1.0, max_iterations=100,
                            use_waypoint_buffer=True, z_bounds=None):
        """
        A* 2D path planning from current position to target (optimized):
        - precompute adjacencies to avoid O(N^2) full scan
        - standard heap update strategy (duplicate entries allowed, stale ones skipped on pop)
        - vectorized heuristic distance via numpy
        """
        t1 = time.time()

        has_stair = hasattr(self, 'stair_pcd') and not self.stair_pcd.is_empty()
        max_target_z = 2.2 if has_stair else 1.0
        if abs(target_point[2] - (self.current_position[2] - 1.2)) > max_target_z:
            return []

        if not use_waypoint_buffer:
            self.waypoints = self.get_candidate_waypoints(
                min_distance=0.3, max_distance=2.5, waypoint_grid_resolution=0.5
            )

        tread_centers = self._extract_validated_tread_centers()
        if (not isinstance(self.waypoints, np.ndarray) or self.waypoints.shape[0] == 0) and tread_centers.shape[0] == 0:
            return []

        start_point = (self.current_position - np.array([0, 0, 1.2], dtype=np.float32))[:3]
        goal_point = np.asarray(target_point[:3], dtype=np.float32)

        def _find_idx(arr, p):
            m = np.where(np.all(arr == p, axis=1))[0]
            if m.size > 0:
                return int(m[0])
            d2 = np.sum((arr - p[None, :]) ** 2, axis=1)
            return int(np.argmin(d2))

        use_cache = (z_bounds is None and self.is_path_cache_valid(max_distance_to_neighbor=max_distance_to_neighbor))
        if use_cache:
            cache = self._path_planning_cache
            cache['hit_count'] = int(cache.get('hit_count', 0)) + 1
            navigable_kdtree = cache['navigable_kdtree']
            obstacle_kdtree = cache['obstacle_kdtree']
            obstacle_points = cache['obstacle_points']
            stair_kdtree = cache['stair_kdtree']
            stair_points = cache['stair_points']
            stair_tread_kdtree = cache['stair_tread_kdtree']
            stair_tread_points = cache['stair_tread_points']
            base_nodes = cache['base_nodes']
            max_step_z = cache['max_step_z']
            goal_matches = np.where(np.all(base_nodes == goal_point, axis=1))[0]
            if goal_matches.size > 0:
                nodes = base_nodes
                neighbors = [list(nb) for nb in cache['base_neighbors']]
                start_node_idx = _find_idx(nodes, start_point)
                goal_node_idx = int(goal_matches[0])
            else:
                nodes = np.vstack([base_nodes, goal_point[None, :]]).astype(np.float32)
                goal_node_idx = nodes.shape[0] - 1
                start_node_idx = _find_idx(nodes, start_point)
                neighbors = [list(nb) for nb in cache['base_neighbors']] + [[]]
                node_kdtree = cache['node_kdtree']
                if node_kdtree is not None:
                    k, idxs, _ = node_kdtree.search_radius_vector_3d(
                        goal_point.astype(np.float64), float(max_distance_to_neighbor) + 1e-6
                    )
                    if k > 0:
                        gz = goal_point[2]
                        cand = [j for j in idxs if abs(float(gz - base_nodes[j, 2])) <= max_step_z]
                        neighbors[goal_node_idx] = cand
                        for j in cand:
                            neighbors[j].append(goal_node_idx)
        else:
            navigable_kdtree = None if self.navigable_pcd.is_empty() else o3d.geometry.KDTreeFlann(
                self.navigable_pcd.to_legacy())
            obstacle_kdtree = None if self.obstacle_pcd.is_empty() else o3d.geometry.KDTreeFlann(
                self.obstacle_pcd.to_legacy())
            obstacle_points = None if self.obstacle_pcd.is_empty() else np.asarray(
                self.obstacle_pcd.point.positions.cpu().numpy(), dtype=np.float32)
            stair_kdtree = None if self.stair_pcd.is_empty() else o3d.geometry.KDTreeFlann(self.stair_pcd.to_legacy())
            stair_points = None if self.stair_pcd.is_empty() else np.asarray(
                self.stair_pcd.point.positions.cpu().numpy(), dtype=np.float32)
            tread_pcd = getattr(self, 'stair_tread_pcd', None)
            stair_tread_kdtree = None if (tread_pcd is None or tread_pcd.is_empty()) else o3d.geometry.KDTreeFlann(
                tread_pcd.to_legacy())
            stair_tread_points = None if (tread_pcd is None or tread_pcd.is_empty()) else np.asarray(
                tread_pcd.point.positions.cpu().numpy(), dtype=np.float32)

            wp = self.waypoints[:, :3] if (isinstance(self.waypoints, np.ndarray) and self.waypoints.size > 0) else np.empty((0, 3), dtype=np.float32)
            parts = [p for p in (wp, tread_centers, start_point[None, :], goal_point[None, :]) if p.shape[0] > 0]
            nodes = np.vstack(parts).astype(np.float32)
            nodes_unique, unique_idx = np.unique(nodes, axis=0, return_index=True)
            nodes = nodes_unique[np.argsort(unique_idx)]

            if z_bounds is not None and len(z_bounds) == 2:
                z_min, z_max = float(z_bounds[0]), float(z_bounds[1])
                keep_mask = (nodes[:, 2] >= z_min - 1e-6) & (nodes[:, 2] <= z_max + 1e-6)
                nodes = nodes[keep_mask]
                if nodes.shape[0] == 0:
                    return []

            start_node_idx = _find_idx(nodes, start_point)
            goal_node_idx = _find_idx(nodes, goal_point)

            n_nodes = nodes.shape[0]
            if n_nodes == 0:
                return []

            node_pcd = o3d.geometry.PointCloud()
            node_pcd.points = o3d.utility.Vector3dVector(nodes.astype(np.float64))
            node_kdtree = o3d.geometry.KDTreeFlann(node_pcd)

            max_step_z = 0.30 if (tread_centers.shape[0] > 0 or stair_kdtree is None) else 0.45
            neighbors = [[] for _ in range(n_nodes)]
            for i in range(n_nodes):
                p = nodes[i].astype(np.float64)
                k, idxs, _ = node_kdtree.search_radius_vector_3d(p, float(max_distance_to_neighbor) + 1e-6)
                if k <= 1:
                    continue
                pz = nodes[i, 2]
                cand = [j for j in idxs if j != i and abs(float(pz - nodes[j, 2])) <= max_step_z]
                neighbors[i] = cand

        n_nodes = nodes.shape[0]
        if n_nodes == 0:
            return []

        goal_xyz = nodes[goal_node_idx]

        def heuristic_idx(i: int) -> float:
            d = nodes[i] - goal_xyz
            return float(np.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]))

        open_heap = []
        came_from = np.full((n_nodes,), -1, dtype=np.int32)

        g_score = np.full((n_nodes,), np.inf, dtype=np.float64)
        f_score = np.full((n_nodes,), np.inf, dtype=np.float64)

        g_score[start_node_idx] = 0.0
        f0 = heuristic_idx(start_node_idx)
        f_score[start_node_idx] = f0
        heapq.heappush(open_heap, (f0, start_node_idx))

        iterations = 0

        while open_heap:
            if iterations >= max_iterations:
                pass

                return []
            iterations += 1

            current_f, current_idx = heapq.heappop(open_heap)

            if current_f > f_score[current_idx]:
                continue

            if current_idx == goal_node_idx:

                path_idx = []
                cur = current_idx
                while cur != -1:
                    path_idx.append(cur)
                    cur = came_from[cur]
                path_idx.reverse()
                path = [tuple(nodes[i].tolist()) for i in path_idx]

                return path

            p_cur = nodes[current_idx]

            for neighbor_idx in neighbors[current_idx]:
                p_nei = nodes[neighbor_idx]

                if navigable_kdtree is not None or stair_tread_kdtree is not None:
                    if not self._is_path_clear(
                            p_cur, p_nei,
                            navigable_kdtree, obstacle_kdtree, stair_kdtree,
                            check_radius=0.3, obstacle_radius=0.2, step_size=0.2,
                            obstacle_points=obstacle_points, stair_points=stair_points,
                            stair_tread_kdtree=stair_tread_kdtree, stair_tread_points=stair_tread_points,
                    ):
                        continue

                d = p_cur - p_nei
                edge_cost = float(np.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]))
                tentative_g = g_score[current_idx] + edge_cost

                if tentative_g < g_score[neighbor_idx]:
                    came_from[neighbor_idx] = current_idx
                    g_score[neighbor_idx] = tentative_g
                    new_f = tentative_g + heuristic_idx(neighbor_idx)
                    f_score[neighbor_idx] = new_f
                    heapq.heappush(open_heap, (new_f, neighbor_idx))

        return []

    def plan_path_to_target_naive(self, target_point, step_length=1.0, clearance_step=0.2):
        """
        Ablation: straight-line strategy, no global search.
        Sample uniformly along a 2D line from current position to target, check clearance segment-wise.
        If entire path is clear, return waypoint list; otherwise return [].
        Parameters:
            target_point: (x, y, z) target world coords (same reference frame as get_candidate_waypoints)
            step_length:  sampling interval (meters)
            clearance_step: _is_path_clear internal step (meters)
        Returns:
            list[(x, y, z)] straight-line waypoints (including start/end); returns [] if unreachable
        """
        try:
            tp = np.asarray(target_point, dtype=np.float32)

            if abs(tp[2] - (self.current_position[2] - 1.2)) > 1.0:
                return []

            start_point = (self.current_position - np.array([0, 0, 1.2], dtype=np.float32)).astype(np.float32)
            vec2d = tp[:2] - start_point[:2]
            dist2d = float(np.linalg.norm(vec2d))
            if dist2d < 1e-6:
                return [tuple(start_point.tolist()), tuple(tp.tolist())]

            dir2d = vec2d / dist2d
            n_steps = int(dist2d // step_length)

            if self.navigable_pcd.is_empty():
                navigable_kdtree = None
            else:
                navigable_kdtree = o3d.geometry.KDTreeFlann(self.navigable_pcd.to_legacy())

            if self.obstacle_pcd.is_empty():
                obstacle_kdtree = None
            else:
                obstacle_kdtree = o3d.geometry.KDTreeFlann(self.obstacle_pcd.to_legacy())

            if self.stair_pcd.is_empty():
                stair_kdtree = None
            else:
                stair_kdtree = o3d.geometry.KDTreeFlann(self.stair_pcd.to_legacy())

            if navigable_kdtree is None or obstacle_kdtree is None:
                path = [tuple(start_point.tolist())]
                for i in range(1, n_steps + 1):
                    seg_len = min(step_length * i, dist2d)
                    p2 = start_point.copy()
                    p2[0] += dir2d[0] * seg_len
                    p2[1] += dir2d[1] * seg_len

                    p2[2] = start_point[2] + (tp[2] - start_point[2]) * (seg_len / (dist2d + 1e-6))
                    path.append(tuple(p2.tolist()))
                path.append(tuple(tp.tolist()))
                return path

            clear = self._is_path_clear(
                start_point, tp,
                navigable_kdtree, obstacle_kdtree,
                stair_kdtree=stair_kdtree,
                check_radius=0.3,
                obstacle_radius=0.2,
                step_size=clearance_step
            )
            if not clear:
                return []

            path = [tuple(start_point.tolist())]
            for i in range(1, n_steps + 1):
                seg_len = min(step_length * i, dist2d)
                p2 = start_point.copy()
                p2[0] += dir2d[0] * seg_len
                p2[1] += dir2d[1] * seg_len
                p2[2] = start_point[2] + (tp[2] - start_point[2]) * (seg_len / (dist2d + 1e-6))
                path.append(tuple(p2.tolist()))

            if path[-1] != tuple(tp.tolist()):
                path.append(tuple(tp.tolist()))
            return path
        except Exception as e:
            print(f"[plan_path_to_target_naive] Error: {e}")
            return []

    def _identify_and_process_stairs(self,
                                     scene_pcd,
                                     normal_radius=0.2,
                                     normal_max_nn=30,
                                     stair_min_abs_z=0.2,
                                     stair_max_abs_z=0.7,
                                     stair_cluster_eps=0.4,
                                     stair_cluster_min_points=150,
                                     stair_min_height_span=0.5,
                                     platform_abs_z_thresh=0.9,
                                     platform_z_delta=0.1,
                                     platform_cluster_eps=0.35,
                                     platform_cluster_min_points=35,
                                     platform_min_points_total=120,
                                     bbox_xy_expand=0.3,
                                     stair_min_xy_extent=0.45,
                                     stair_min_height_bins=4,
                                     stair_min_axis_z_corr=0.3,
                                     enable_tread_detection=True,
                                     tread_abs_z_thresh=0.85,
                                     tread_z_delta=0.08,
                                     tread_axis_expand=0.20,
                                     tread_side_expand=0.12,
                                     tread_cluster_min_points=30,
                                     _profile=True):
        """
        Detect stairs and their top/bottom platforms; return stair point cloud.

        Key optimizations:
        1. use voxel connectivity instead of per-frame DBSCAN to reduce clustering time.
        2. geometry consistency check on candidate clusters, filtering slanted walls, handrails, stray surfaces.
        3. platform search: coarse 2D KDTree filter, then precise rectangle + height filtering.
        """
        if scene_pcd.is_empty() or len(scene_pcd.point.positions) < 100:
            return o3d.t.geometry.PointCloud(self.pcd_device)

        t0 = time.perf_counter()

        def _voxel_key(vox):
            return tuple(int(v) for v in vox)

        def _voxel_cc_cluster(points_np, voxel, min_points, neighbor_range=1):
            if points_np is None or points_np.shape[0] == 0:
                return []

            voxel = max(float(voxel), float(self.pcd_resolution), 1e-3)
            vox = np.floor(points_np / voxel).astype(np.int32)
            vox_view = np.ascontiguousarray(vox).view(
                [("x", np.int32), ("y", np.int32), ("z", np.int32)]
            ).reshape(-1)
            uniq_view, inv = np.unique(vox_view, return_inverse=True)
            uniq_vox = uniq_view.view(np.int32).reshape(-1, 3)

            buckets = [[] for _ in range(uniq_vox.shape[0])]
            for point_idx, voxel_idx in enumerate(inv):
                buckets[int(voxel_idx)].append(point_idx)

            lut = {_voxel_key(v): i for i, v in enumerate(uniq_vox)}
            visited = np.zeros(uniq_vox.shape[0], dtype=bool)
            clusters = []

            offsets = [
                (dx, dy, dz)
                for dx in range(-neighbor_range, neighbor_range + 1)
                for dy in range(-neighbor_range, neighbor_range + 1)
                for dz in range(-neighbor_range, neighbor_range + 1)
                if not (dx == 0 and dy == 0 and dz == 0)
            ]

            for seed in range(uniq_vox.shape[0]):
                if visited[seed]:
                    continue

                queue = [seed]
                visited[seed] = True
                voxel_ids = []
                point_count = 0

                while queue:
                    cur = queue.pop()
                    voxel_ids.append(cur)
                    point_count += len(buckets[cur])
                    base = uniq_vox[cur]

                    for offset in offsets:
                        nb_key = (
                            int(base[0] + offset[0]),
                            int(base[1] + offset[1]),
                            int(base[2] + offset[2]),
                        )
                        nb = lut.get(nb_key)
                        if nb is None or visited[nb]:
                            continue
                        visited[nb] = True
                        queue.append(nb)

                if point_count < int(min_points):
                    continue

                point_ids = []
                for voxel_id in voxel_ids:
                    point_ids.extend(buckets[voxel_id])
                clusters.append(np.asarray(point_ids, dtype=np.int64))

            return clusters

        def _looks_like_stairs(points_np):
            if points_np.shape[0] < int(stair_cluster_min_points):
                return False

            xy = points_np[:, :2]
            z = points_np[:, 2]
            h_span = float(z.max() - z.min())
            if h_span < float(stair_min_height_span):
                return False

            xy_extent = np.ptp(xy, axis=0)
            if float(np.max(xy_extent)) < float(stair_min_xy_extent):
                return False

            centered_xy = xy - xy.mean(axis=0, keepdims=True)
            try:
                _, _, vh = np.linalg.svd(centered_xy, full_matrices=False)
                axis = vh[0]
            except np.linalg.LinAlgError:
                return False

            along = centered_xy @ axis
            along_span = float(along.max() - along.min())
            if along_span < float(stair_min_xy_extent):
                return False

            corr = np.corrcoef(along, z)[0, 1] if points_np.shape[0] > 2 else 0.0
            if not np.isfinite(corr) or abs(float(corr)) < float(stair_min_axis_z_corr):
                return False

            bin_size = max(float(self.pcd_resolution) * 2.0, 0.08)
            height_bins = np.unique(np.floor((z - z.min()) / bin_size).astype(np.int32))
            if height_bins.size < int(stair_min_height_bins):
                return False

            return True

        try:
            t_norm0 = time.perf_counter()
            scene_pcd.estimate_normals(radius=normal_radius, max_nn=normal_max_nn)
            t_norm1 = time.perf_counter()

            t_cpu0 = time.perf_counter()
            normals = scene_pcd.point.normals.cpu().numpy()
            abs_z = np.abs(normals[:, 2])
            t_cpu1 = time.perf_counter()

            stair_mask = (abs_z > stair_min_abs_z) & (abs_z < stair_max_abs_z)
            if not np.any(stair_mask):
                self.platform_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
                return o3d.t.geometry.PointCloud(self.pcd_device)

            stair_indices = np.where(stair_mask)[0]
            stair_indices_tensor = o3d.core.Tensor(stair_indices, dtype=o3d.core.Dtype.Int64, device=self.pcd_device)
            stair_candidates = scene_pcd.select_by_index(stair_indices_tensor)
            if stair_candidates.is_empty():
                self.platform_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
                return o3d.t.geometry.PointCloud(self.pcd_device)

            t_cl0 = time.perf_counter()
            stair_points_np = stair_candidates.point.positions.cpu().numpy()
            stair_clusters = _voxel_cc_cluster(
                stair_points_np,
                voxel=max(float(stair_cluster_eps) * 0.5, 0.05),
                min_points=stair_cluster_min_points,
                neighbor_range=1,
            )
            t_cl1 = time.perf_counter()
            if not stair_clusters:
                self.platform_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
                return o3d.t.geometry.PointCloud(self.pcd_device)

            platform_mask = abs_z > platform_abs_z_thresh
            platform_indices = np.where(platform_mask)[0]
            platform_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
            if platform_indices.size > 0:
                platform_indices_tensor = o3d.core.Tensor(platform_indices, dtype=o3d.core.Dtype.Int64,
                                                          device=self.pcd_device)
                platform_pcd = scene_pcd.select_by_index(platform_indices_tensor)

            tread_pcd_all = o3d.t.geometry.PointCloud(self.pcd_device)
            if enable_tread_detection:
                tread_mask = abs_z > tread_abs_z_thresh
                tread_indices = np.where(tread_mask)[0]
                if tread_indices.size > 0:
                    tread_indices_tensor = o3d.core.Tensor(tread_indices, dtype=o3d.core.Dtype.Int64,
                                                           device=self.pcd_device)
                    tread_pcd_all = scene_pcd.select_by_index(tread_indices_tensor)

            plat_pts = None
            plat_kdtree = None
            if not platform_pcd.is_empty():
                plat_pts = platform_pcd.point.positions.cpu().numpy()
                plat_pts_2d = plat_pts.copy()
                plat_pts_2d[:, 2] = 0.0
                plat_legacy_2d = o3d.geometry.PointCloud()
                plat_legacy_2d.points = o3d.utility.Vector3dVector(plat_pts_2d.astype(np.float64, copy=False))
                plat_kdtree = o3d.geometry.KDTreeFlann(plat_legacy_2d)

            tread_pts = None
            tread_kdtree = None
            if enable_tread_detection and not tread_pcd_all.is_empty():
                tread_pts = tread_pcd_all.point.positions.cpu().numpy()
                tread_pts_2d = tread_pts.copy()
                tread_pts_2d[:, 2] = 0.0
                tread_legacy_2d = o3d.geometry.PointCloud()
                tread_legacy_2d.points = o3d.utility.Vector3dVector(tread_pts_2d.astype(np.float64, copy=False))
                tread_kdtree = o3d.geometry.KDTreeFlann(tread_legacy_2d)

            stair_clusters_to_merge = []
            tread_clusters_to_merge = []
            platform_clusters_to_merge = []

            t_loop0 = time.perf_counter()
            for cluster_indices_np in stair_clusters:
                cluster_indices = o3d.core.Tensor(cluster_indices_np, dtype=o3d.core.Dtype.Int64,
                                                  device=self.pcd_device)
                cluster = stair_candidates.select_by_index(cluster_indices)
                if cluster.point.positions.shape[0] < stair_cluster_min_points:
                    continue

                cluster_pts = cluster.point.positions.cpu().numpy()
                if not _looks_like_stairs(cluster_pts):
                    continue

                stair_clusters_to_merge.append(cluster)

                if enable_tread_detection and tread_pts is not None and tread_kdtree is not None:
                    stair_xy = cluster_pts[:, :2]
                    z_min = float(cluster_pts[:, 2].min())
                    z_max = float(cluster_pts[:, 2].max())
                    center_xy = stair_xy.mean(axis=0)
                    centered_xy = stair_xy - center_xy[None, :]
                    try:
                        _, _, vh = np.linalg.svd(centered_xy, full_matrices=False)
                        axis = vh[0].astype(np.float64, copy=False)
                    except np.linalg.LinAlgError:
                        axis = None

                    if axis is not None:
                        axis_norm = float(np.linalg.norm(axis))
                        if axis_norm > 1e-6:
                            axis = axis / axis_norm
                            side_axis = np.array([-axis[1], axis[0]], dtype=np.float64)

                            stair_along = centered_xy @ axis
                            stair_side = centered_xy @ side_axis
                            along_min = float(stair_along.min()) - float(tread_axis_expand)
                            along_max = float(stair_along.max()) + float(tread_axis_expand)
                            side_min = float(stair_side.min()) - float(tread_side_expand)
                            side_max = float(stair_side.max()) + float(tread_side_expand)

                            radius = float(np.sqrt(max(abs(along_min), abs(along_max)) ** 2 +
                                                   max(abs(side_min), abs(side_max)) ** 2) + 1e-6)
                            query = np.array([center_xy[0], center_xy[1], 0.0], dtype=np.float64)
                            k_tread, candidate_tread_indices, _ = tread_kdtree.search_radius_vector_3d(query, radius)
                            if k_tread > 0:
                                candidate_tread_indices = np.asarray(candidate_tread_indices, dtype=np.int64)
                                candidate_tread_pts = tread_pts[candidate_tread_indices]
                                rel_xy = candidate_tread_pts[:, :2] - center_xy[None, :]
                                cand_along = rel_xy @ axis
                                cand_side = rel_xy @ side_axis
                                footprint_cond = (
                                    (cand_along >= along_min) & (cand_along <= along_max) &
                                    (cand_side >= side_min) & (cand_side <= side_max)
                                )
                                z_cond = (
                                    (candidate_tread_pts[:, 2] >= z_min - float(tread_z_delta)) &
                                    (candidate_tread_pts[:, 2] <= z_max + float(tread_z_delta))
                                )
                                tread_sel = candidate_tread_indices[footprint_cond & z_cond]
                                if tread_sel.size >= int(tread_cluster_min_points):
                                    tread_sel_tensor = o3d.core.Tensor(tread_sel, dtype=o3d.core.Dtype.Int64,
                                                                       device=self.pcd_device)
                                    candidate_treads = tread_pcd_all.select_by_index(tread_sel_tensor)
                                    candidate_tread_pts = candidate_treads.point.positions.cpu().numpy()
                                    tread_clusters = _voxel_cc_cluster(
                                        candidate_tread_pts,
                                        voxel=max(float(platform_cluster_eps) * 0.5, 0.05),
                                        min_points=tread_cluster_min_points,
                                        neighbor_range=1,
                                    )
                                    for tread_indices_np in tread_clusters:
                                        tread_indices_tensor = o3d.core.Tensor(
                                            tread_indices_np,
                                            dtype=o3d.core.Dtype.Int64,
                                            device=self.pcd_device
                                        )
                                        tread_sub_pcd = candidate_treads.select_by_index(tread_indices_tensor)
                                        if tread_sub_pcd.point.positions.shape[0] >= tread_cluster_min_points:
                                            tread_clusters_to_merge.append(tread_sub_pcd)

                if plat_pts is None or plat_kdtree is None:
                    continue

                min_x, max_x = cluster_pts[:, 0].min(), cluster_pts[:, 0].max()
                min_y, max_y = cluster_pts[:, 1].min(), cluster_pts[:, 1].max()
                min_z, max_z = cluster_pts[:, 2].min(), cluster_pts[:, 2].max()

                cx = 0.5 * (min_x + max_x)
                cy = 0.5 * (min_y + max_y)
                hx = 0.5 * (max_x - min_x) + float(bbox_xy_expand)
                hy = 0.5 * (max_y - min_y) + float(bbox_xy_expand)
                query_radius = float(np.sqrt(hx * hx + hy * hy) + 1e-6)
                query = np.array([cx, cy, 0.0], dtype=np.float64)
                k, candidate_indices, _ = plat_kdtree.search_radius_vector_3d(query, query_radius)
                if k <= 0:
                    continue

                candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
                candidate_plat_pts = plat_pts[candidate_indices]

                xy_cond = (candidate_plat_pts[:, 0] >= min_x - bbox_xy_expand) & \
                          (candidate_plat_pts[:, 0] <= max_x + bbox_xy_expand) & \
                          (candidate_plat_pts[:, 1] >= min_y - bbox_xy_expand) & \
                          (candidate_plat_pts[:, 1] <= max_y + bbox_xy_expand)
                if not np.any(xy_cond):
                    continue

                sel_xy = candidate_indices[xy_cond]
                sel_xy_pts = plat_pts[sel_xy]

                z_cond = (np.abs(sel_xy_pts[:, 2] - min_z) < platform_z_delta) | \
                         (np.abs(sel_xy_pts[:, 2] - max_z) < platform_z_delta)

                sel = sel_xy[z_cond]
                if sel.size == 0:
                    continue

                sel_tensor = o3d.core.Tensor(sel, dtype=o3d.core.Dtype.Int64, device=self.pcd_device)
                candidate_platform = platform_pcd.select_by_index(sel_tensor)

                if candidate_platform.point.positions.shape[0] >= platform_min_points_total:
                    candidate_platform_pts = candidate_platform.point.positions.cpu().numpy()
                    platform_clusters = _voxel_cc_cluster(
                        candidate_platform_pts,
                        voxel=max(float(platform_cluster_eps) * 0.5, 0.05),
                        min_points=platform_cluster_min_points,
                        neighbor_range=1,
                    )
                    for sub_indices_np in platform_clusters:
                        sub_indices = o3d.core.Tensor(sub_indices_np, dtype=o3d.core.Dtype.Int64,
                                                      device=self.pcd_device)
                        sub_pcd = candidate_platform.select_by_index(sub_indices)
                        if sub_pcd.point.positions.shape[0] >= platform_cluster_min_points:
                            platform_clusters_to_merge.append(sub_pcd)
            t_loop1 = time.perf_counter()

            stair_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
            for cluster in stair_clusters_to_merge:
                stair_pcd = gpu_merge_pointcloud(stair_pcd, cluster)

            tread_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
            for tread in tread_clusters_to_merge:
                tread_pcd = gpu_merge_pointcloud(tread_pcd, tread)
            self.stair_tread_pcd = tread_pcd.voxel_down_sample(self.pcd_resolution) \
                if not tread_pcd.is_empty() else o3d.t.geometry.PointCloud(self.pcd_device)

            collected_platform = o3d.t.geometry.PointCloud(self.pcd_device)
            for platform in platform_clusters_to_merge:
                collected_platform = gpu_merge_pointcloud(collected_platform, platform)

            self.platform_pcd = collected_platform.voxel_down_sample(self.pcd_resolution) \
                if not collected_platform.is_empty() else o3d.t.geometry.PointCloud(self.pcd_device)

            if stair_pcd.is_empty() and self.platform_pcd.is_empty():
                return o3d.t.geometry.PointCloud(self.pcd_device)

            merged = gpu_merge_pointcloud(stair_pcd, tread_pcd) if enable_tread_detection else stair_pcd
            if not self.platform_pcd.is_empty():
                merged = gpu_merge_pointcloud(merged, self.platform_pcd)
            if not merged.is_empty():
                merged = merged.voxel_down_sample(self.pcd_resolution)

            if _profile:
                t1 = time.perf_counter()
                print(
                    "[StairDetect] "
                    f"normals={t_norm1 - t_norm0:.4f}s, "
                    f"cpu={t_cpu1 - t_cpu0:.4f}s, "
                    f"cluster={t_cl1 - t_cl0:.4f}s, "
                    f"loop={t_loop1 - t_loop0:.4f}s, "
                    f"total={t1 - t0:.4f}s"
                )

            return merged

        except Exception as e:
            print(f"[StairDetect] Error: {e}")
            self.platform_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
            return o3d.t.geometry.PointCloud(self.pcd_device)

    def is_on_platform(self,
                       scene_pcd,
                       search_radius=1.5,
                       plane_distance=0.05,
                       min_area=0.5,
                       min_support_ratio=0.35,
                       min_radius=0.35,
                       max_height_diff=0.4,
                       bottom_height_tolerance=0.1,
                       area_grid_resolution=0.12):
        """
        Check for sufficiently large horizontal platform near current position.
        use local height bucketing and 2D grid occupancy instead of RANSAC + convex hull for speed.
        Returns: (flag, metrics)
        """
        metrics = {}
        try:
            if scene_pcd.is_empty() or scene_pcd.point.positions.shape[0] < 50:
                return False, metrics

            pts_np = scene_pcd.point.positions.cpu().numpy()
            agent_xy = self.current_position[:2]
            dx = pts_np[:, 0] - agent_xy[0]
            dy = pts_np[:, 1] - agent_xy[1]
            dist2 = dx * dx + dy * dy
            local_mask = dist2 <= (search_radius ** 2)
            if not np.any(local_mask):
                return False, metrics

            local_pts = pts_np[local_mask]
            local_count = local_pts.shape[0]
            if local_count < 30:
                return False, metrics

            agent_z = float(self.current_position[2] - 1.2)
            local_z = local_pts[:, 2]
            finite_mask = np.isfinite(local_z)
            if not np.any(finite_mask):
                return False, metrics
            local_pts = local_pts[finite_mask]
            local_z = local_z[finite_mask]
            local_count = local_pts.shape[0]
            if local_count < 30:
                return False, metrics

            z_min = float(local_z.min())

            bin_size = max(float(plane_distance), 1e-3)
            z_bins = np.floor((local_z - z_min) / bin_size).astype(np.int32)
            bin_ids, counts = np.unique(z_bins, return_counts=True)
            if bin_ids.size == 0:
                return False, metrics

            bin_heights = z_min + (bin_ids.astype(np.float64) + 0.5) * bin_size
            close_bins = np.where(
                (np.abs(bin_heights - agent_z) < float(bottom_height_tolerance)) &
                (counts >= 20)
            )[0]
            if close_bins.size > 0:
                candidate_bins = close_bins
            else:
                near_bottom = np.abs(bin_heights - agent_z) <= max(
                    float(max_height_diff), float(bottom_height_tolerance)
                )
                candidate_bins = np.where(near_bottom)[0]
            if candidate_bins.size == 0:
                candidate_bins = np.arange(bin_ids.size)

            if close_bins.size > 0:
                best_rel = candidate_bins[np.argmin(np.abs(bin_heights[candidate_bins] - agent_z))]
            else:
                best_rel = candidate_bins[np.argmax(counts[candidate_bins])]
            plane_z_seed = bin_heights[best_rel]
            height_band = max(float(plane_distance) * 1.5, 0.06)
            inlier_mask = np.abs(local_z - plane_z_seed) <= height_band
            inlier_count = int(np.count_nonzero(inlier_mask))
            if inlier_count < 20:
                return False, metrics

            inlier_pts = local_pts[inlier_mask]
            plane_z_mean = float(np.median(inlier_pts[:, 2]))
            z_std = float(inlier_pts[:, 2].std())
            support_ratio = inlier_count / (local_count + 1e-6)

            pts_xy = inlier_pts[:, :2]
            grid_resolution = max(float(area_grid_resolution), float(self.pcd_resolution), 1e-3)
            grid_xy = np.floor(pts_xy / grid_resolution).astype(np.int32)
            grid_view = np.ascontiguousarray(grid_xy).view(
                [("x", np.int32), ("y", np.int32)]
            ).reshape(-1)
            occupied_cells = int(np.unique(grid_view).shape[0])
            area_xy = occupied_cells * grid_resolution * grid_resolution

            radius_est = math.sqrt(area_xy / math.pi) if area_xy > 0 else 0.0
            vertical_diff = abs(agent_z - plane_z_mean)
            vertical_score = 1.0 if z_std < 0.08 else max(0.0, 1.0 - z_std / 0.4)

            close_to_agent_bottom = vertical_diff < float(bottom_height_tolerance)

            flag = (
                close_to_agent_bottom or (
                    vertical_score >= 0.95 and
                    support_ratio >= min_support_ratio and
                    area_xy >= min_area and
                    radius_est >= min_radius and
                    vertical_diff <= max_height_diff and
                    z_std < 0.08
                )
            )

            if not flag and vertical_score >= 0.95 and area_xy >= (min_area * 0.8) and vertical_diff <= 0.5:
                flag = True

            metrics = {
                "inlier_count": int(inlier_count),
                "local_count": int(local_count),
                "support_ratio": float(support_ratio),
                "vertical_score": float(vertical_score),
                "plane_height_mean": float(plane_z_mean),
                "agent_z": float(agent_z),
                "vertical_diff": float(vertical_diff),
                "area_xy": float(area_xy),
                "radius_est": float(radius_est),
                "z_std": float(z_std),
                "occupied_cells": int(occupied_cells),
                "bottom_height_tolerance": float(bottom_height_tolerance)
            }
            return bool(flag), metrics
        except Exception as e:
            print(f"[is_on_platform] error: {e}")
            return False, metrics

    def _safe_clear_tpcd(self, pcd):
        """Safely clear an o3d.t point cloud and return empty on same device."""
        try:
            if isinstance(pcd, o3d.t.geometry.PointCloud):
                pcd.clear()
        except Exception:
            pass
        return o3d.t.geometry.PointCloud(self.pcd_device)

    def release_vram(self):
        """Attempt to release GPU memory cache and trigger Python GC."""
        try:

            if hasattr(o3d.core, "cuda") and hasattr(o3d.core.cuda, "release_cache"):
                o3d.core.cuda.release_cache()
        except Exception:
            pass
        gc.collect()

    def translate_2d_coordinate_to_3d_world(
            self,
            x,
            y,
            depth: np.ndarray,
            camera_intrinsic=None,
            agent_position=None,
            agent_rotation=None):
        """
        Convert 2D image coords + depth to 3D world coordinates.
        Coordinate system consistent with get_pointcloud_from_depth / translate_to_world:
        - camera coords: [pixel_x, pixel_z, -pixel_y]
        - world coords: world = R @ camera_point + t
        accept camera_intrinsic/agent_position/agent_rotation for non-primary camera views,
        e.g., a single panorama panel.
        """

        if depth is None:
            return None

        if len(depth.shape) == 3:
            depth = depth[:, :, 0]

        h, w = depth.shape[:2]
        xi, yi = int(round(x)), int(round(y))
        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            return None

        d = float(depth[yi, xi])
        if not np.isfinite(d) or d <= 0:
            return None

        intrinsic = self.camera_intrinsic if camera_intrinsic is None else camera_intrinsic
        fx = float(intrinsic[0][0])
        fy = float(intrinsic[1][1])
        cx = float(intrinsic[0][2])
        cy = float(intrinsic[1][2])

        pixel_z = (h - 1 - yi - cy) * d / fy
        pixel_x = (xi - cx) * d / fx
        pixel_y = d
        camera_point = np.array([pixel_x, pixel_z, -pixel_y], dtype=np.float32)

        position = self.current_position if agent_position is None else agent_position
        rotation = self.current_rotation if agent_rotation is None else agent_rotation
        if hasattr(rotation, 'as_rotation_matrix'):
            rotation = rotation.as_rotation_matrix()
        elif hasattr(rotation, 'shape') and rotation.shape == (4, 4):
            rotation = rotation[:3, :3]

        world_point = rotation @ camera_point + position
        return world_point

    def _clone_object_entities(self, entities=None):
        src = getattr(self, 'object_entities', ()) if entities is None else entities
        cloned = []
        for ent in (src or ()):
            if not isinstance(ent, dict):
                continue
            row = dict(ent)
            pcd = row.get('pcd')
            if pcd is not None and hasattr(pcd, 'clone'):
                row['pcd'] = pcd.clone()
            center = row.get('center')
            if isinstance(center, np.ndarray):
                row['center'] = center.copy()
            obs_list = row.get('observations')
            if isinstance(obs_list, list):
                row['observations'] = [dict(o) if isinstance(o, dict) else o for o in obs_list]
            views_list = row.get('last_seen_views')
            if isinstance(views_list, list):
                row['last_seen_views'] = list(views_list)
            cloned.append(row)
        return cloned

    def _begin_observation(self, observation_step):
        self.observation_id = getattr(self, 'observation_id', 0) + 1
        self.observation_step = int(observation_step) if observation_step is not None else None
        self.current_observation_masks = {}
        self.current_visual_candidates = {}
        self._pre_observation_entities = self._clone_object_entities() if hasattr(self, '_clone_object_entities') else []
        self._map_revision = getattr(self, '_map_revision', 0) + 1
        self._path_planning_cache = None

    def _update_semantic_views(self, processed_views, primary_index, segmenter_fn=None):
        """Use the same detector threshold and vocabulary for every current view."""
        instances = []
        original_pcd = self.current_pcd
        self._last_processed_views = list(processed_views or ())
        self._last_primary_index = primary_index
        if getattr(self, 'current_observation_masks', None) is None:
            self.current_observation_masks = {}
        if getattr(self, 'current_visual_candidates', None) is None:
            self.current_visual_candidates = {}
        try:
            for idx, proc in processed_views:
                if callable(segmenter_fn):
                    try:
                        view_instances = segmenter_fn(proc['rgb'], proc.get('view', 'front'))
                    except TypeError:
                        view_instances = segmenter_fn(proc['rgb'])
                else:
                    view_instances = instance_segmentation(proc['rgb'])
                if idx == primary_index:
                    instances = view_instances
                if not view_instances:
                    continue
                self.current_pcd = proc['pcd']
                current_entities = self.get_object_entities(
                    proc['depth'],
                    [inst['class_id'] for inst in view_instances],
                    [inst['class_name'] for inst in view_instances],
                    [inst['mask'] for inst in view_instances],
                    [1.0 if inst.get('conf') is None else inst.get('conf') for inst in view_instances],
                    camera_intrinsic=proc.get('camera_intrinsic', self.camera_intrinsic),
                    position=proc['position'], rotation=proc['rotation'], view=proc['view'])
                if current_entities:
                    self.object_entities = self.associate_object_entities(self.object_entities, current_entities)
            candidates = getattr(self, 'current_visual_candidates', None) or {}
            for cand in candidates.values():
                cand['entity_id'] = None
            obs_id = getattr(self, 'observation_id', None)
            for entity_idx, entity in enumerate(getattr(self, 'object_entities', None) or ()):
                if not isinstance(entity, dict):
                    continue
                if obs_id is not None and entity.get('last_seen_observation_id') != obs_id:
                    continue
                for obs in entity.get('observations', ()):
                    if obs_id is not None and obs.get('observation_id') != obs_id:
                        continue
                    mask_key = obs.get('mask_key')
                    cand = candidates.get(mask_key)
                    if (cand is not None and cand.get('mask_key') == mask_key
                            and cand.get('view') == obs.get('view')
                            and cand.get('observation_id') == obs.get('observation_id')):
                        cand['entity_id'] = int(entity_idx)
        finally:
            self.current_pcd = original_pcd
        return instances

    def refresh_current_views_with_segmenter(self, segmenter_fn, additive: bool = True):
        """Supplement or re-segment the current observation's 4 views with an alternate segmenter (e.g. DINO+SAM3 at verification)."""
        processed_views = getattr(self, '_last_processed_views', None)
        if not processed_views or not callable(segmenter_fn):
            return []
        primary_index = getattr(self, '_last_primary_index', 0)
        if not additive:
            if hasattr(self, '_pre_observation_entities'):
                self.object_entities = self._clone_object_entities(self._pre_observation_entities)
            self.current_observation_masks = {}
            self.current_visual_candidates = {}
            return self._update_semantic_views(processed_views, primary_index, segmenter_fn=segmenter_fn)

        obs_id = getattr(self, 'observation_id', None)
        if getattr(self, 'current_observation_masks', None) is None:
            self.current_observation_masks = {}
        if getattr(self, 'current_visual_candidates', None) is None:
            self.current_visual_candidates = {}
        original_pcd = self.current_pcd
        supplemented = []
        try:
            for idx, proc in processed_views:
                view_name = proc.get('view', 'front')
                try:
                    raw_extra = segmenter_fn(proc['rgb'], view_name)
                except TypeError:
                    raw_extra = segmenter_fn(proc['rgb'])
                if not raw_extra:
                    continue
                existing_view_cands = [
                    c for c in self.current_visual_candidates.values()
                    if isinstance(c, dict) and c.get('view') == view_name and c.get('observation_id') == obs_id
                ]
                start_idx = len(existing_view_cands)
                kept_extra = []
                for inst in raw_extra:
                    if not isinstance(inst, dict) or inst.get('mask') is None:
                        continue
                    inst_canon = inst.get('canonical_label') or canonical_label(inst.get('class_name', ''))
                    inst_mask_bool = np.asarray(inst['mask']) > 0
                    if not np.any(inst_mask_bool):
                        continue
                    duplicate = False
                    _incompat_yoloe_classes = {
                        'door', 'couch', 'sofa', 'chair', 'bed', 'toilet', 'sink',
                        'bathtub', 'window', 'curtain', 'plant', 'stairs', 'mirror', 'refrigerator'
                    }
                    for ec in existing_view_cands:
                        ec_canon = ec.get('canonical_label') or canonical_label(ec.get('detector_class_name', ''))
                        ec_mask = self.current_observation_masks.get(ec.get('mask_key'))
                        if ec_mask is None:
                            continue
                        inter = int(np.count_nonzero(inst_mask_bool & (np.asarray(ec_mask) > 0)))
                        if inter <= 0:
                            continue
                        union = int(np.count_nonzero(inst_mask_bool | (np.asarray(ec_mask) > 0)))
                        iou = (inter / union) if union > 0 else 0.0
                        if ec_canon == inst_canon and iou >= 0.30:
                            duplicate = True
                            break
                        ec_conf = float(ec.get('confidence') or 0.0)
                        if ec_canon != inst_canon and ec_canon in _incompat_yoloe_classes and ec_conf >= 0.55 and iou >= 0.45:
                            duplicate = True
                            break
                    if not duplicate:
                        kept_extra.append(inst)
                if not kept_extra:
                    continue
                if idx == primary_index:
                    supplemented = kept_extra
                self.current_pcd = proc['pcd']
                new_entities = self.get_object_entities(
                    proc['depth'],
                    [inst['class_id'] for inst in kept_extra],
                    [inst['class_name'] for inst in kept_extra],
                    [inst['mask'] for inst in kept_extra],
                    [1.0 if inst.get('conf') is None else inst.get('conf') for inst in kept_extra],
                    camera_intrinsic=proc.get('camera_intrinsic', self.camera_intrinsic),
                    position=proc['position'],
                    rotation=proc['rotation'],
                    view=view_name,
                    start_detection_index=start_idx,
                    detector_source='dino_sam3',
                )
                if new_entities:
                    self.object_entities = self.associate_object_entities(self.object_entities, new_entities)
            candidates = getattr(self, 'current_visual_candidates', None) or {}
            for entity_idx, entity in enumerate(getattr(self, 'object_entities', None) or ()):
                if not isinstance(entity, dict):
                    continue
                if obs_id is not None and entity.get('last_seen_observation_id') != obs_id:
                    continue
                for obs in entity.get('observations', ()):
                    if obs_id is not None and obs.get('observation_id') != obs_id:
                        continue
                    mask_key = obs.get('mask_key')
                    cand = candidates.get(mask_key)
                    if (cand is not None and cand.get('mask_key') == mask_key
                            and cand.get('view') == obs.get('view')
                            and cand.get('observation_id') == obs.get('observation_id')):
                        cand['entity_id'] = int(entity_idx)
        finally:
            self.current_pcd = original_pcd
        return supplemented

    def update_multiview(self, views, primary_index=None, do_instance_segmentation=True,
                         observation_step=None):
        """
        Batch fuse multiple synchronized views.

        Instead of calling update multiple times in the agent, only perform depth backprojection per view,
        then update global scene/navigable/obstacle point clouds once. Semantic
        detections cover all views; only primary detections are returned for display.
        views: [{'rgb': ..., 'depth': ..., 'position': ..., 'rotation': ..., 'camera_intrinsic': optional}, ...]
        """
        t0 = time.time()
        self._begin_observation(observation_step)
        if not views:
            return []

        if primary_index is None:
            primary_index = len(views) - 1
        primary_index = max(0, min(primary_index, len(views) - 1))
        primary_view = views[primary_index]

        self.current_position = self.translation_func(primary_view['position']) - self.initial_position
        self.current_rotation = self.rotation_func(primary_view['rotation'])
        curr_agent_height = self.current_position[2] + self.initial_floor_height
        self._detect_and_update_floor()

        if len(self.trajectory_position) == 0 or not np.allclose(self.trajectory_position[-1], self.current_position,
                                                                 atol=1e-1):
            if isinstance(self.current_position, np.ndarray):
                self.trajectory_position.append(self.current_position.copy())
            else:
                self.trajectory_position.append(np.array(self.current_position))

        merged_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        primary_processed = None
        processed_views = []

        for idx, view in enumerate(views):
            view_name = view.get('view', ('left', 'right', 'back', 'front')[idx]
                                 if len(views) == 4 else ('front' if idx == primary_index else f'view_{idx}'))
            current_depth = preprocess_depth(np.array(view['depth'], copy=True))
            current_rgb = preprocess_image(view['rgb'])[:, :, :3]
            view_position = self.translation_func(view['position']) - self.initial_position
            view_rotation = self.rotation_func(view['rotation'])
            view_intrinsic = view.get('camera_intrinsic', self.camera_intrinsic)
            if np.any(np.isfinite(current_depth) & (current_depth > 0)):
                camera_points, camera_colors = get_pointcloud_from_depth(current_rgb, current_depth, view_intrinsic)
                world_points = translate_to_world(camera_points, view_position, view_rotation)
                view_pcd = gpu_pointcloud_from_array(world_points, camera_colors,
                                                     self.pcd_device).voxel_down_sample(self.pcd_resolution)
                merged_pcd = gpu_merge_pointcloud(merged_pcd, view_pcd)
            else:
                view_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
            processed = {
                'view': view_name,
                'rgb': current_rgb,
                'depth': current_depth,
                'position': view_position,
                'rotation': view_rotation,
                'camera_intrinsic': view_intrinsic,
                'pcd': view_pcd,
            }
            processed_views.append((idx, processed))
            if idx == primary_index:
                primary_processed = processed

        if primary_processed is None:
            primary_processed = {
                'rgb': preprocess_image(primary_view['rgb'])[:, :, :3],
                'depth': preprocess_depth(np.array(primary_view['depth'], copy=True)),
                'position': self.current_position,
                'rotation': self.current_rotation,
                'camera_intrinsic': primary_view.get('camera_intrinsic', self.camera_intrinsic),
                'pcd': o3d.t.geometry.PointCloud(self.pcd_device),
            }

        self.current_rgb = primary_processed['rgb']
        self.current_depth = primary_processed['depth']
        self.current_position = primary_processed['position']
        self.current_rotation = primary_processed['rotation']

        if merged_pcd.is_empty():
            self.current_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
            self.current_view_navigable_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
            return self._update_semantic_views(processed_views, primary_index) if do_instance_segmentation else []

        self.current_pcd = merged_pcd.voxel_down_sample(self.pcd_resolution)

        instances = self._update_semantic_views(processed_views, primary_index) if do_instance_segmentation else []

        min_keep_z = min(float(self.floor_height), float(curr_agent_height)) - 3.0
        max_keep_z = max(float(self.ceiling_height), float(curr_agent_height) + 2.0)
        self.scene_pcd = gpu_merge_pointcloud(self.current_pcd, self.scene_pcd).voxel_down_sample(self.pcd_resolution)
        if not self.scene_pcd.is_empty():
            self.scene_pcd = self.scene_pcd.select_by_index(
                (self.scene_pcd.point.positions[:, 2] > min_keep_z).nonzero()[0])

        if not self.scene_pcd.is_empty():
            self.useful_pcd = self.scene_pcd.select_by_index(
                (self.scene_pcd.point.positions[:, 2] < max_keep_z).nonzero()[0])
        else:
            self.useful_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
            self.navigable_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

        if self.enable_stair_detection and (not self.useful_pcd.is_empty()):
            try:
                pts_np = self.useful_pcd.point.positions.cpu().numpy()
                agent_xy = self.current_position[:2]
                r = float(self.local_stair_radius + 1)
                dx = pts_np[:, 0] - agent_xy[0]
                dy = pts_np[:, 1] - agent_xy[1]
                local_mask = (dx * dx + dy * dy) <= (r * r)
                local_idx = np.where(local_mask)[0]

                if local_idx.size >= 100:
                    local_idx_tensor = o3d.core.Tensor(local_idx, dtype=o3d.core.Dtype.Int64,
                                                       device=self.useful_pcd.device)
                    local_pcd = self.useful_pcd.select_by_index(local_idx_tensor)
                    current_stair_pcd = self._identify_and_process_stairs(local_pcd)
                    local_pcd.clear()
                    del local_pcd
                else:
                    current_stair_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
            except Exception as _e:
                current_stair_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        else:
            current_stair_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

        if not current_stair_pcd.is_empty():
            pts = current_stair_pcd.point.positions.cpu().numpy()
            agent_xy = self.current_position[:2]
            dx = pts[:, 0] - agent_xy[0]
            dy = pts[:, 1] - agent_xy[1]
            local_mask = (dx * dx + dy * dy) <= (self.local_stair_radius ** 2)
            local_count = int(np.count_nonzero(local_mask))
            if local_count >= self.local_stair_min_points:
                pass
                self.stair_pcd = gpu_merge_pointcloud(self.stair_pcd, current_stair_pcd).voxel_down_sample(
                    self.pcd_resolution)
                try:
                    s_pts = self.stair_pcd.point.positions.cpu().numpy()
                    if s_pts.shape[0] >= 10:
                        min_idx = np.argmin(s_pts[:, 2])
                        max_idx = np.argmax(s_pts[:, 2])
                        self.stair_portals = [
                            {'type': 'stair_bottom', 'pos': s_pts[min_idx]},
                            {'type': 'stair_top', 'pos': s_pts[max_idx]}
                        ]
                except Exception:
                    pass

        cur_z = self.current_pcd.point.positions[:, 2]
        nav_band_mask = (
            ((cur_z >= self.floor_height - 0.25) & (cur_z <= self.floor_height + 0.22))
            | ((cur_z >= curr_agent_height - 0.25) & (cur_z <= curr_agent_height + 0.22))
        )
        original_navigable_pcd = self.current_pcd.select_by_index(nav_band_mask.nonzero()[0])

        current_navigable_point = gpu_merge_pointcloud(original_navigable_pcd, current_stair_pcd).voxel_down_sample(
            self.grid_resolution)

        current_navigable_position = current_navigable_point.point.positions.cpu().numpy()

        # Preserve measured support only. Linear interpolation used to fill
        # unseen space (including walls/voids) with fictitious floor samples.
        self.current_view_navigable_pcd = current_navigable_point.clone()
        if not self.current_view_navigable_pcd.is_empty():
            self.navigable_pcd = gpu_merge_pointcloud(
                self.navigable_pcd, self.current_view_navigable_pcd).voxel_down_sample(self.pcd_resolution)

        if not self.useful_pcd.is_empty():
            obs_floor_ref = min(float(self.floor_height), float(curr_agent_height))
            self.obstacle_pcd = self.useful_pcd.select_by_index(
                (self.useful_pcd.point.positions[:, 2] > obs_floor_ref + 0.2).nonzero()[0])
        else:
            self.obstacle_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

        self.trajectory_pcd = gpu_pointcloud_from_array(np.array(self.trajectory_position),
                                                        np.zeros((len(self.trajectory_position), 3)), self.pcd_device)
        if self.navigable_pcd.is_empty() and not self.useful_pcd.is_empty():
            self.navigable_pcd = self.useful_pcd.select_by_index(
                (self.useful_pcd.point.positions[:, 2] < curr_agent_height + 0.2).nonzero()[0])

        self.update_iterations += 1

        try:
            if 'original_navigable_pcd' in locals():
                original_navigable_pcd.clear(); del original_navigable_pcd
            if 'current_navigable_point' in locals():
                current_navigable_point.clear(); del current_navigable_point
            if 'current_stair_pcd' in locals():
                current_stair_pcd.clear(); del current_stair_pcd

            if isinstance(self.current_pcd, o3d.t.geometry.PointCloud):
                self.current_pcd.clear()
                self.current_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

            if isinstance(self.useful_pcd, o3d.t.geometry.PointCloud):
                self.useful_pcd.clear()
                self.useful_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        except Exception:
            pass

        if (self.update_iterations % 5) == 0:
            self.release_vram()

        return instances

    def update(self, rgb, depth, position, rotation, do_instance_segmentation=True, observation_step=None):

        t0 = time.time()
        self._begin_observation(observation_step)

        self.current_position = self.translation_func(position) - self.initial_position
        self.current_rotation = self.rotation_func(rotation)
        curr_agent_height = self.current_position[2] + self.initial_floor_height
        self._detect_and_update_floor()
        t1 = time.time()

        self.current_depth = preprocess_depth(np.array(depth, copy=True))
        self.current_rgb = preprocess_image(rgb)[:, :, :3]

        if len(self.trajectory_position) == 0 or not np.allclose(self.trajectory_position[-1], self.current_position,
                                                                 atol=1e-1):

            if isinstance(self.current_position, np.ndarray):
                self.trajectory_position.append(self.current_position.copy())
            else:
                self.trajectory_position.append(np.array(self.current_position))

        has_usable_depth = bool(np.any(np.isfinite(self.current_depth) & (self.current_depth > 0)))
        if has_usable_depth:
            camera_points, camera_colors = get_pointcloud_from_depth(self.current_rgb, self.current_depth,
                                                                     self.camera_intrinsic)
            world_points = translate_to_world(camera_points, self.current_position, self.current_rotation)
            self.current_pcd = gpu_pointcloud_from_array(world_points, camera_colors,
                                                         self.pcd_device).voxel_down_sample(self.pcd_resolution)
        else:
            self.current_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
            self.current_view_navigable_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        t2 = time.time()

        processed_views = [(0, {
            'view': 'front',
            'rgb': self.current_rgb,
            'depth': self.current_depth,
            'position': self.current_position,
            'rotation': self.current_rotation,
            'camera_intrinsic': self.camera_intrinsic,
            'pcd': self.current_pcd,
        })]
        instances = self._update_semantic_views(processed_views, 0) if do_instance_segmentation else []
        t3 = t4 = t5 = t6 = time.time()
        if not has_usable_depth:
            return instances

        min_keep_z = min(float(self.floor_height), float(curr_agent_height)) - 3.0
        max_keep_z = max(float(self.ceiling_height), float(curr_agent_height) + 2.0)
        self.scene_pcd = gpu_merge_pointcloud(self.current_pcd, self.scene_pcd).voxel_down_sample(self.pcd_resolution)
        if not self.scene_pcd.is_empty():
            self.scene_pcd = self.scene_pcd.select_by_index(
                (self.scene_pcd.point.positions[:, 2] > min_keep_z).nonzero()[0])

        if not self.scene_pcd.is_empty():
            self.useful_pcd = self.scene_pcd.select_by_index(
                (self.scene_pcd.point.positions[:, 2] < max_keep_z).nonzero()[0])
        else:
            self.useful_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
            self.navigable_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

        if self.enable_stair_detection and (not self.useful_pcd.is_empty()):
            try:
                pts_np = self.useful_pcd.point.positions.cpu().numpy()
                agent_xy = self.current_position[:2]
                r = float(self.local_stair_radius + 1)
                dx = pts_np[:, 0] - agent_xy[0]
                dy = pts_np[:, 1] - agent_xy[1]
                local_mask = (dx * dx + dy * dy) <= (r * r)
                local_idx = np.where(local_mask)[0]

                if local_idx.size >= 100:
                    local_idx_tensor = o3d.core.Tensor(local_idx, dtype=o3d.core.Dtype.Int64,
                                                       device=self.useful_pcd.device)
                    local_pcd = self.useful_pcd.select_by_index(local_idx_tensor)
                    current_stair_pcd = self._identify_and_process_stairs(local_pcd)

                    local_pcd.clear()
                    del local_pcd
                else:
                    current_stair_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
            except Exception as _e:
                current_stair_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        else:
            current_stair_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

        if not current_stair_pcd.is_empty():
            pts = current_stair_pcd.point.positions.cpu().numpy()
            agent_xy = self.current_position[:2]
            dx = pts[:, 0] - agent_xy[0]
            dy = pts[:, 1] - agent_xy[1]
            local_mask = (dx * dx + dy * dy) <= (self.local_stair_radius ** 2)
            local_count = int(np.count_nonzero(local_mask))
            if local_count >= self.local_stair_min_points:
                pass
                self.stair_pcd = gpu_merge_pointcloud(self.stair_pcd, current_stair_pcd).voxel_down_sample(
                    self.pcd_resolution)

        cur_z = self.current_pcd.point.positions[:, 2]
        nav_band_mask = (
            ((cur_z >= self.floor_height - 0.25) & (cur_z <= self.floor_height + 0.22))
            | ((cur_z >= curr_agent_height - 0.25) & (cur_z <= curr_agent_height + 0.22))
        )
        original_navigable_pcd = self.current_pcd.select_by_index(nav_band_mask.nonzero()[0])

        current_navigable_point = gpu_merge_pointcloud(original_navigable_pcd, current_stair_pcd).voxel_down_sample(self.pcd_resolution)

        current_navigable_position = current_navigable_point.point.positions.cpu().numpy()

        # Preserve measured support only. Linear interpolation used to fill
        # unseen space (including walls/voids) with fictitious floor samples.
        self.current_view_navigable_pcd = current_navigable_point.clone()
        if not self.current_view_navigable_pcd.is_empty():
            self.navigable_pcd = gpu_merge_pointcloud(
                self.navigable_pcd, self.current_view_navigable_pcd).voxel_down_sample(self.pcd_resolution)

        if not self.useful_pcd.is_empty():
            obs_floor_ref = min(float(self.floor_height), float(curr_agent_height))
            self.obstacle_pcd = self.useful_pcd.select_by_index(
                (self.useful_pcd.point.positions[:, 2] > obs_floor_ref + 0.1).nonzero()[0])
        else:
            self.obstacle_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

        # if not self.obstacle_pcd.is_empty() and not stair_pcd.is_empty():
        #     dist_to_stairs = pointcloud_distance(self.obstacle_pcd, stair_pcd)

        #     not_stair_mask = dist_to_stairs > self.pcd_resolution
        #     if not_stair_mask.sum() > 0:
        #         indices_tensor = o3d.core.Tensor(not_stair_mask.cpu().numpy(), device=self.pcd_device).nonzero()[0]
        #         self.obstacle_pcd = self.obstacle_pcd.select_by_index(indices_tensor)
        #     else:
        #         self.obstacle_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

        self.trajectory_pcd = gpu_pointcloud_from_array(np.array(self.trajectory_position),
                                                        np.zeros((len(self.trajectory_position), 3)), self.pcd_device)
        if self.navigable_pcd.is_empty() and not self.useful_pcd.is_empty():
            self.navigable_pcd = self.useful_pcd.select_by_index(
                (self.useful_pcd.point.positions[:, 2] < curr_agent_height + 0.1).nonzero()[0])

        # if not self.navigable_pcd.is_empty() and not self.obstacle_pcd.is_empty():
        #     self.frontier_pcd = project_frontier(self.obstacle_pcd, self.navigable_pcd, self.floor_height + 0.2,
        #                                          self.grid_resolution)
        #     if self.frontier_pcd.shape[0] > 0:
        #         self.frontier_pcd[:, 2] = self.navigable_pcd.point.positions.cpu().numpy()[:, 2].mean()
        #         self.frontier_pcd = gpu_pointcloud_from_array(self.frontier_pcd,
        #                                                       np.ones((self.frontier_pcd.shape[0], 3)) * np.array(
        #                                                           [[255, 0, 0]]), self.pcd_device)
        #     else:
        #         self.frontier_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        # else:
        #     self.frontier_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

        self.update_iterations += 1
        t7 = time.time()

        # ========================

        # ========================
        try:
            if 'original_navigable_pcd' in locals():
                original_navigable_pcd.clear(); del original_navigable_pcd
            if 'current_navigable_point' in locals():
                current_navigable_point.clear(); del current_navigable_point
            if 'current_stair_pcd' in locals():
                current_stair_pcd.clear(); del current_stair_pcd

            if isinstance(self.current_pcd, o3d.t.geometry.PointCloud):
                self.current_pcd.clear()
                self.current_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

            if isinstance(self.useful_pcd, o3d.t.geometry.PointCloud):
                self.useful_pcd.clear()
                self.useful_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        except Exception:
            pass

        if (self.update_iterations % 5) == 0:
            self.release_vram()

        # print(f"  get_object_entities: {t5 - t4:.3f}s")
        # print(f"  associate_object_entities: {t6 - t5:.3f}s")

        return instances

    def get_bev_height_points(self):
        """Display-only samples, including retained stairs outside the floor slice.

        Never merge these samples into navigability/obstacles or path planning.
        """
        samples = []
        for name in ('navigable_pcd', 'stair_pcd'):
            cloud = getattr(self, name, None)
            if cloud is None or cloud.is_empty():
                continue
            points = cloud.point.positions.cpu().numpy()
            mask = (np.isfinite(points).all(axis=1)
                    & (np.linalg.norm(points[:, :2] - self.current_position[:2], axis=1) <= 8.0)
                    & (np.abs(points[:, 2] - self.floor_height) <= 3.0))
            points = points[mask]
            # Bound CPU drawing work independently of the accumulated cloud size.
            samples.append(points[::max(1, int(np.ceil(len(points) / 10000)))])
        return np.concatenate(samples) if samples else np.empty((0, 3))

    def get_object_entities(self, depth, classes, class_names, masks, confidences, camera_intrinsic=None,
                            position=None, rotation=None, view='front', start_detection_index=0,
                            detector_source='yoloe'):
        entities = []
        if getattr(self, 'current_observation_masks', None) is None:
            self.current_observation_masks = {}
        if getattr(self, 'current_visual_candidates', None) is None:
            self.current_visual_candidates = {}
        observation_id = int(getattr(self, 'observation_id', 0) or 0)
        observation_step = getattr(self, 'observation_step', None)

        object_height_threshold = self.ceiling_height - 0.2
        intrinsic = self.camera_intrinsic if camera_intrinsic is None else camera_intrinsic
        pos = self.current_position if position is None else position
        rot = self.current_rotation if rotation is None else rotation

        depth_arr = np.asarray(depth)
        if depth_arr.ndim == 3 and depth_arr.shape[2] == 1:
            depth_arr = depth_arr[:, :, 0]

        for offset_index, (cls, cls_name, mask, score) in enumerate(zip(classes, class_names, masks, confidences)):
            detection_index = int(start_detection_index) + offset_index
            mask_key = f'{observation_id}:{view}:{detection_index}'
            try:
                score_val = float(score)
            except (ValueError, TypeError):
                continue
            if not np.isfinite(score_val):
                continue
            try:
                observation = detection_observation(
                    mask, depth_arr, score_val, observation_id=observation_id,
                    step=observation_step, view=view, mask_key=mask_key)
            except (ValueError, TypeError):
                continue
            if observation is None:
                continue

            mask_bool = (np.asarray(mask) > 0).copy()
            self.current_observation_masks[mask_key] = mask_bool
            name = canonical_label(cls_name)
            observation['candidate_id'] = mask_key
            observation['detector_class_name'] = cls_name
            observation['detector_class_id'] = int(cls)

            candidate = {
                'candidate_id': mask_key,
                'observation_id': observation['observation_id'],
                'step': observation['step'],
                'view': observation['view'],
                'detector_class_name': cls_name,
                'detector_class_id': int(cls),
                'canonical_label': name,
                'confidence': observation['confidence'],
                'bbox': list(observation['bbox']),
                'image_size': list(observation['image_size']),
                'mask_key': mask_key,
                'entity_id': None,
                'geometry_status': 'unknown',
                'geometry_reason': 'insufficient_depth',
                'valid_depth_fraction': observation['valid_depth_fraction'],
                'horizontal_distance_m': None,
                'detector_source': str(detector_source or 'yoloe'),
            }
            self.current_visual_candidates[mask_key] = candidate

            valid_mask = mask_bool & np.isfinite(depth_arr) & (depth_arr > 0)
            masked_depth = depth_arr[valid_mask]
            if masked_depth.size == 0:
                continue
            if masked_depth.min() < 1.0 and score_val < 0.45:
                candidate['geometry_reason'] = 'low_confidence_depth'
                continue

            camera_points = get_pointcloud_from_depth_mask(depth_arr, valid_mask, intrinsic)
            world_points = np.asarray(translate_to_world(camera_points, pos, rot), dtype=float)
            if world_points.ndim != 2 or world_points.shape[1] < 3:
                candidate['geometry_reason'] = 'invalid_projection'
                continue
            finite_rows = np.all(np.isfinite(world_points), axis=1)
            if not np.all(finite_rows):
                world_points = world_points[finite_rows]
                if world_points.shape[0] == 0:
                    candidate['geometry_reason'] = 'invalid_projection'
                    continue

            color = get_class_color(cls)
            point_colors = np.array([color] * world_points.shape[0])
            if world_points.shape[0] < 10:
                candidate['geometry_reason'] = 'insufficient_points'
                continue
            object_pcd = gpu_pointcloud_from_array(world_points, point_colors, self.pcd_device).voxel_down_sample(
                self.pcd_resolution * 2)
            object_pcd = gpu_cluster_filter(object_pcd)
            if object_pcd.point.positions.shape[0] < 10:
                candidate['geometry_reason'] = 'insufficient_points'
                continue

            center = np.asarray(object_pcd.point.positions.mean(dim=0).cpu().numpy(), dtype=float)
            points = np.asarray(object_pcd.point.positions.cpu().numpy(), dtype=float)
            pos_arr = np.asarray(pos, dtype=float)
            if (center.shape[0] < 3 or points.ndim != 2 or points.shape[1] < 3
                    or pos_arr.shape[0] < 2
                    or not np.all(np.isfinite(center))
                    or not np.all(np.isfinite(points))
                    or not np.all(np.isfinite(pos_arr[:2]))):
                candidate['geometry_reason'] = 'invalid_projection'
                continue

            if center[2] > object_height_threshold:
                candidate['geometry_reason'] = 'height_filtered'
                continue

            horizontal_distance_m = float(np.quantile(np.linalg.norm(points[:, :2] - pos_arr[:2], axis=1), .1))
            if not np.isfinite(horizontal_distance_m):
                candidate['geometry_reason'] = 'invalid_projection'
                continue
            observation['horizontal_distance_m'] = horizontal_distance_m
            candidate['geometry_status'] = 'available'
            candidate['geometry_reason'] = 'available'
            candidate['horizontal_distance_m'] = horizontal_distance_m
            entity = {'class': cls, 'class_name': name, 'canonical_label': name,
                      'pcd': object_pcd, 'center': center}
            add_entity_observation(entity, observation)
            entities.append(entity)
        return entities

    def associate_object_entities(self, ref_entities, eval_entities):
        if len(ref_entities) == 0:
            return eval_entities

        ref_centers = np.array([entity['center'] for entity in ref_entities])

        for entity in eval_entities:
            eval_center = entity['center']
            eval_pcd = entity['pcd']
            if eval_pcd.is_empty():
                continue
            eval_label = canonical_label(entity['class_name'])

            center_distances = np.linalg.norm(ref_centers - eval_center, axis=1)
            distance_threshold = 1.0
            candidate_indices = np.where(center_distances < distance_threshold)[0]

            if len(candidate_indices) == 0:
                ref_entities.append(entity)
                ref_centers = np.vstack([ref_centers, eval_center])
                continue

            same_class_candidates = []
            overlap_scores = []
            overlap_conditions = []

            for idx in candidate_indices:
                if canonical_label(ref_entities[idx]['class_name']) == eval_label:
                    ref_pcd = ref_entities[idx]['pcd']
                    if ref_pcd.is_empty():
                        continue
                    same_class_candidates.append(idx)
                    cdist = pointcloud_distance(eval_pcd, ref_pcd)
                    overlap_condition = (cdist < 0.1)
                    denom = float(overlap_condition.shape[0]) if overlap_condition.shape[0] > 0 else 1e-6
                    overlap_ratio = float(overlap_condition.sum().cpu().numpy()) / denom
                    overlap_scores.append(overlap_ratio)
                    overlap_conditions.append(overlap_condition)

            if len(overlap_scores) > 0:
                best_pos = int(np.argmax(overlap_scores))
                max_overlap_score = overlap_scores[best_pos]
                second_best_score = max(
                    (score for i, score in enumerate(overlap_scores) if i != best_pos),
                    default=0.0,
                )
                is_unambiguous = (len(overlap_scores) == 1
                                  or (max_overlap_score - second_best_score) >= 0.05 - 1e-9)
                if max_overlap_score >= 0.25 - 1e-9 and is_unambiguous:
                    best_candidate_idx = same_class_candidates[best_pos]
                    best_entity = ref_entities[best_candidate_idx]
                    nonoverlap_condition = overlap_conditions[best_pos].logical_not()
                    if nonoverlap_condition.sum() > 0:
                        remaining_pcd = eval_pcd.select_by_index(
                            o3d.core.Tensor(nonoverlap_condition.cpu().numpy(), device=self.pcd_device).nonzero()[0]
                        )
                    else:
                        remaining_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
                    best_entity['pcd'] = gpu_merge_pointcloud(best_entity['pcd'], remaining_pcd)
                    if not best_entity['pcd'].is_empty():
                        best_entity['center'] = best_entity['pcd'].point.positions.mean(dim=0).cpu().numpy()
                        ref_centers[best_candidate_idx] = best_entity['center']
                    merge_entity_evidence(best_entity, entity)
                    ref_entities[best_candidate_idx] = best_entity
                    continue

            if eval_pcd.point.positions.shape[0] >= 15:
                ref_entities.append(entity)
                ref_centers = np.vstack([ref_centers, entity['center']])

        return ref_entities

    def update_object_pcd(self):
        object_pcd = o3d.geometry.PointCloud()
        for entity in self.object_entities:
            points = entity['pcd'].point.positions.cpu().numpy()
            colors = entity['pcd'].point.colors.cpu().numpy()
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(points)
            new_pcd.colors = o3d.utility.Vector3dVector(colors)
            object_pcd = object_pcd + new_pcd
        try:
            return gpu_pointcloud(object_pcd, self.pcd_device)
        except:
            return self.scene_pcd

    def get_view_pointcloud(self, rgb, depth, translation, rotation):
        current_position = self.translation_func(translation) - self.initial_position
        current_rotation = self.rotation_func(rotation)
        current_depth = preprocess_depth(np.array(depth, copy=True))
        current_rgb = preprocess_image(rgb)
        camera_points, camera_colors = get_pointcloud_from_depth(current_rgb, current_depth, self.camera_intrinsic)
        world_points = translate_to_world(camera_points, current_position, current_rotation)
        current_pcd = gpu_pointcloud_from_array(world_points, camera_colors, self.pcd_device).voxel_down_sample(
            self.pcd_resolution)
        return current_pcd

    def get_obstacle_affordance(self):
        try:
            distance = pointcloud_distance(self.navigable_pcd, self.obstacle_pcd)
            affordance = (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
            affordance[distance < 0.25] = 0
            return affordance.cpu().numpy()
        except:
            return np.zeros((self.navigable_pcd.point.positions.shape[0],), dtype=np.float32)

    def get_trajectory_affordance(self):
        try:
            distance = pointcloud_distance(self.navigable_pcd, self.trajectory_pcd)
            affordance = (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
            return affordance.cpu().numpy()
        except:
            return np.zeros((self.navigable_pcd.point.positions.shape[0],), dtype=np.float32)

    def get_semantic_affordance(self, target_class, threshold=0.1):
        semantic_pointcloud = o3d.t.geometry.PointCloud()
        for entity in self.object_entities:
            if entity['class'] in target_class:
                semantic_pointcloud = gpu_merge_pointcloud(semantic_pointcloud, entity['pcd'])
        try:
            distance = pointcloud_2d_distance(self.navigable_pcd, semantic_pointcloud)
            affordance = 1 - (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
            affordance[distance > threshold] = 0
            affordance = affordance.cpu().numpy()
            return affordance
        except:
            return np.zeros((self.navigable_pcd.point.positions.shape[0],), dtype=np.float32)

    def get_gpt4v_affordance(self, gpt4v_pcd):
        try:
            distance = pointcloud_distance(self.navigable_pcd, gpt4v_pcd)
            affordance = 1 - (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
            affordance[distance > 0.1] = 0
            return affordance.cpu().numpy()
        except:
            return np.zeros((self.navigable_pcd.point.positions.shape[0],), dtype=np.float32)

    def get_action_affordance(self, action):
        try:
            if action == 'Explore':
                distance = pointcloud_2d_distance(self.navigable_pcd, self.frontier_pcd)
                affordance = 1 - (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
                affordance[distance > 0.2] = 0
                return affordance.cpu().numpy()
            elif action == 'Move_Forward':
                pixel_x, pixel_z, depth_values = project_to_camera(self.navigable_pcd, self.camera_intrinsic,
                                                                   self.current_position, self.current_rotation)
                filter_condition = (pixel_x >= 0) & (pixel_x < self.camera_intrinsic[0][2] * 2) & (pixel_z >= 0) & (
                            pixel_z < self.camera_intrinsic[1][2] * 2) & (depth_values > 1.5) & (depth_values < 2.5)
                filter_pcd = self.navigable_pcd.select_by_index(
                    o3d.core.Tensor(np.where(filter_condition == 1)[0], device=self.navigable_pcd.device))
                distance = pointcloud_distance(self.navigable_pcd, filter_pcd)
                affordance = 1 - (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
                affordance[distance > 0.1] = 0
                return affordance.cpu().numpy()
            elif action == 'Turn_Around':
                R = np.array([np.pi, np.pi, np.pi]) * self.rotate_axis
                turn_extrinsic = np.matmul(self.current_rotation,
                                           quaternion.as_rotation_matrix(quaternion.from_euler_angles(R)))
                pixel_x, pixel_z, depth_values = project_to_camera(self.navigable_pcd, self.camera_intrinsic,
                                                                   self.current_position, turn_extrinsic)
                filter_condition = (pixel_x >= 0) & (pixel_x < self.camera_intrinsic[0][2] * 2) & (pixel_z >= 0) & (
                            pixel_z < self.camera_intrinsic[1][2] * 2) & (depth_values > 1.5) & (depth_values < 2.5)
                filter_pcd = self.navigable_pcd.select_by_index(
                    o3d.core.Tensor(np.where(filter_condition == 1)[0], device=self.navigable_pcd.device))
                distance = pointcloud_distance(self.navigable_pcd, filter_pcd)
                affordance = 1 - (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
                affordance[distance > 0.1] = 0
                return affordance.cpu().numpy()
            elif action == 'Turn_Left':
                R = np.array([np.pi / 2, np.pi / 2, np.pi / 2]) * self.rotate_axis
                turn_extrinsic = np.matmul(self.current_rotation,
                                           quaternion.as_rotation_matrix(quaternion.from_euler_angles(R)))
                pixel_x, pixel_z, depth_values = project_to_camera(self.navigable_pcd, self.camera_intrinsic,
                                                                   self.current_position, turn_extrinsic)
                filter_condition = (pixel_x >= 0) & (pixel_x < self.camera_intrinsic[0][2] * 2) & (pixel_z >= 0) & (
                            pixel_z < self.camera_intrinsic[1][2] * 2) & (depth_values > 1.5) & (depth_values < 2.5)
                filter_pcd = self.navigable_pcd.select_by_index(
                    o3d.core.Tensor(np.where(filter_condition == 1)[0], device=self.navigable_pcd.device))
                distance = pointcloud_distance(self.navigable_pcd, filter_pcd)
                affordance = 1 - (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
                affordance[distance > 0.1] = 0
                return affordance.cpu().numpy()
            elif action == 'Turn_Right':
                R = np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2]) * self.rotate_axis
                turn_extrinsic = np.matmul(self.current_rotation,
                                           quaternion.as_rotation_matrix(quaternion.from_euler_angles(R)))
                pixel_x, pixel_z, depth_values = project_to_camera(self.navigable_pcd, self.camera_intrinsic,
                                                                   self.current_position, turn_extrinsic)
                filter_condition = (pixel_x >= 0) & (pixel_x < self.camera_intrinsic[0][2] * 2) & (pixel_z >= 0) & (
                            pixel_z < self.camera_intrinsic[1][2] * 2) & (depth_values > 1.5) & (depth_values < 2.5)
                filter_pcd = self.navigable_pcd.select_by_index(
                    o3d.core.Tensor(np.where(filter_condition == 1)[0], device=self.navigable_pcd.device))
                distance = pointcloud_distance(self.navigable_pcd, filter_pcd)
                affordance = 1 - (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
                affordance[distance > 0.1] = 0
                return affordance.cpu().numpy()
            elif action == 'Enter':
                return self.get_semantic_affordance(['doorway', 'door', 'entrance', 'exit'])
            elif action == 'Exit':
                return self.get_semantic_affordance(['doorway', 'door', 'entrance', 'exit'])
            else:
                return np.zeros((self.navigable_pcd.point.positions.shape[0],), dtype=np.float32)
        except:
            return np.zeros((self.navigable_pcd.point.positions.shape[0],), dtype=np.float32)

    def get_objnav_affordance_map(self, action, target_class, gpt4v_pcd, complete_flag=False, failure_mode=False):
        if failure_mode:
            obstacle_affordance = self.get_obstacle_affordance()
            affordance = self.get_action_affordance('Explore')
            affordance = np.clip(affordance, 0.1, 1.0)
            affordance[obstacle_affordance == 0] = 0
            return affordance, self.visualize_affordance(affordance)
        elif complete_flag:
            affordance = self.get_semantic_affordance([target_class], threshold=0.1)
            return affordance, self.visualize_affordance(affordance)
        else:
            obstacle_affordance = self.get_obstacle_affordance()
            semantic_affordance = self.get_semantic_affordance([target_class], threshold=1.5)
            action_affordance = self.get_action_affordance(action)
            gpt4v_affordance = self.get_gpt4v_affordance(gpt4v_pcd)
            history_affordance = self.get_trajectory_affordance()
            affordance = 0.25 * semantic_affordance + 0.25 * action_affordance + 0.25 * gpt4v_affordance + 0.25 * history_affordance
            affordance = np.clip(affordance, 0.1, 1.0)
            affordance[obstacle_affordance == 0] = 0
            return affordance, self.visualize_affordance(affordance / (affordance.max() + 1e-6))

    def get_debug_affordance_map(self, action, target_class, gpt4v_pcd):
        obstacle_affordance = self.get_obstacle_affordance()
        semantic_affordance = self.get_semantic_affordance([target_class], threshold=1.5)
        action_affordance = self.get_action_affordance(action)
        gpt4v_affordance = self.get_gpt4v_affordance(gpt4v_pcd)
        history_affordance = self.get_trajectory_affordance()
        return self.visualize_affordance(semantic_affordance / (semantic_affordance.max() + 1e-6)), \
            self.visualize_affordance(history_affordance / (history_affordance.max() + 1e-6)), \
            self.visualize_affordance(action_affordance / (action_affordance.max() + 1e-6)), \
            self.visualize_affordance(gpt4v_affordance / (gpt4v_affordance.max() + 1e-6)), \
            self.visualize_affordance(obstacle_affordance / (obstacle_affordance.max() + 1e-6))

    def visualize_affordance(self, affordance):
        cmap = colormaps.get('jet')
        color_affordance = cmap(affordance)[:, 0:3]
        color_affordance = cpu_pointcloud_from_array(self.navigable_pcd.point.positions.cpu().numpy(), color_affordance)
        return color_affordance

    def get_appeared_objects(self):
        return [entity['class'] for entity in self.object_entities]

    def save_pointcloud_debug(self, path="./"):
        save_pcd = o3d.geometry.PointCloud()
        try:
            assert self.useful_pcd.point.positions.shape[0] > 0
            save_pcd.points = o3d.utility.Vector3dVector(self.useful_pcd.point.positions.cpu().numpy())
            save_pcd.colors = o3d.utility.Vector3dVector(self.useful_pcd.point.colors.cpu().numpy())
            o3d.io.write_point_cloud(path + "scene.ply", save_pcd)
        except:
            pass
        try:
            assert self.navigable_pcd.point.positions.shape[0] > 0
            save_pcd.points = o3d.utility.Vector3dVector(self.navigable_pcd.point.positions.cpu().numpy())
            save_pcd.colors = o3d.utility.Vector3dVector(self.navigable_pcd.point.colors.cpu().numpy())
            o3d.io.write_point_cloud(path + "navigable.ply", save_pcd)
        except:
            pass
        try:
            assert self.obstacle_pcd.point.positions.shape[0] > 0
            save_pcd.points = o3d.utility.Vector3dVector(self.obstacle_pcd.point.positions.cpu().numpy())
            save_pcd.colors = o3d.utility.Vector3dVector(self.obstacle_pcd.point.colors.cpu().numpy())
            o3d.io.write_point_cloud(path + "obstacle.ply", save_pcd)
        except:
            pass

        object_pcd = o3d.geometry.PointCloud()
        for entity in self.object_entities:
            points = entity['pcd'].point.positions.cpu().numpy()
            colors = entity['pcd'].point.colors.cpu().numpy()
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(points)
            new_pcd.colors = o3d.utility.Vector3dVector(colors)
            object_pcd = object_pcd + new_pcd
        if len(object_pcd.points) > 0:
            o3d.io.write_point_cloud(path + "object.ply", object_pcd)

    def release_all_vram(self):
        """Release GPU memory and large object refs; safe to call repeatedly."""

        pcd_attr_names = [
            'scene_pcd', 'navigable_pcd', 'obstacle_pcd', 'trajectory_pcd',
            'current_pcd', 'useful_pcd', 'frontier_pcd', 'current_view_navigable_pcd',
            'stair_pcd', 'platform_pcd'
        ]
        for name in pcd_attr_names:
            self._safe_clear_tpcd(getattr(self, name, None))

        try:
            if hasattr(self, 'pcds_per_floor') and isinstance(self.pcds_per_floor, dict):
                for _, data in list(self.pcds_per_floor.items()):
                    for k in ['scene_pcd', 'navigable_pcd', 'obstacle_pcd']:
                        self._safe_clear_tpcd(data.get(k))

                    for ent in data.get('object_entities', []):
                        self._safe_clear_tpcd(ent.get('pcd'))
                    data['object_entities'] = []
                    data['trajectory_position'] = []
                self.pcds_per_floor.clear()
        except Exception:
            pass

        try:
            if hasattr(self, 'object_entities') and isinstance(self.object_entities, list):
                for ent in self.object_entities:
                    try:
                        self._safe_clear_tpcd(ent.get('pcd'))
                    except Exception:
                        pass
                self.object_entities.clear()
        except Exception:
            pass

        for name in ['current_depth', 'current_rgb', 'trajectory_position', 'frontier_pcd']:
            try:
                setattr(self, name, None)
            except Exception:
                pass

        try:
            import torch
            if hasattr(self, 'device') and isinstance(self.device, str) and self.device.lower().startswith('cuda'):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

        try:
            gc.collect()
        except Exception:
            pass

    def __enter__(self):
        """Supports with statement."""
        return self

    def __exit__(self, exc_type, exc, tb):
        """Auto-release GPU memory on with exit."""
        try:
            self.release_vram()
        except Exception:
            pass

        return False

    def __del__(self):
        """Fallback GPU memory release on destruction."""
        try:
            self.release_vram()
        except Exception:
            pass


import json
import math
import re
from abc import ABC, abstractmethod

import PIL.Image
from PIL import ImageDraw, ImageFont

from simWrapper import SimWrapper, PolarAction
from mapper import Instruct_Mapper
import time
from typing import Optional, List
from PIL.Image import Image
from utils import *
from scipy.spatial.transform import Rotation as R

import socket
import os
import copy
import logging
import time
from agent.route_guard import RouteEvidenceGuard
from agent.turn_execution import TurnExecutionGuard
from agent.recovery import CheckpointRecovery
from agent.habitat_control import HabitatControl
from agent.verification_inputs import VerificationInputs
from navigation_memory import NavigationMemory, get_shared_navigation_memory
from navigation_policy import (normalize_verification, checkpoint_yaw,
                               instruction_requires_movement, ModelServiceError, partition_instruction)
from spatial_state import SpatialState, semantic_refresh_due
from progress_watchdog import NavigationAborted
from waypoint_navigation import WaypointRoutes, candidate_order
from navigation_contracts import completion_conditions
from verification_feedback import (evidence_repair_required, endpoint_relocalization_required,
                                   initial_arrival_observation_retry_required, landing_observation_pending,
                                   remember_verification)
from navigation_control import (ActionProtocolError, action_prompt, decode_action, legal_action_objects,
                                normalize_action_object, WaypointExecution, StairProgress, pending_stair_condition,
                                stair_condition_intents, landing_support)

def _setup_proxy(preferred_url: str = "http://127.0.0.1:7897") -> None:
    """Set proxy only when:
    1. User has NOT already set http_proxy/https_proxy env vars
    2. The preferred proxy is TCP-reachable (otherwise skip)
    """
    if os.environ.get("http_proxy") or os.environ.get("https_proxy"):
        return
    try:
        host, port = preferred_url.replace("http://", "").replace("/", "").split(":")
        with socket.create_connection((host, int(port)), timeout=1.0):
            os.environ["http_proxy"] = preferred_url
            os.environ["https_proxy"] = preferred_url
    except Exception:
        pass

_setup_proxy()

def draw_direction_markers(img: np.ndarray) -> np.ndarray:
    """
    Draw red direction triangles (top/bottom/left/right) with FRONT/BACK/LEFT/RIGHT direction-only labels (auto-scaled).
    """
    h, w = img.shape[:2]
    if h < 20 or w < 20:
        return img

    s = max(6, int(min(h, w) * 0.025))  # triangle half-size (shrunk)
    m = max(2, int(s * 0.3))  # margin
    border_th = max(1, s // 8)  # stroke thickness
    font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_triangle_with_label(bg, center, direction, label):
        cx, cy = center
        if direction == 'up':
            pts = np.array([[cx, cy - s], [cx - s, cy + s], [cx + s, cy + s]], dtype=np.int32)
        elif direction == 'down':
            pts = np.array([[cx, cy + s], [cx - s, cy - s], [cx + s, cy - s]], dtype=np.int32)
        elif direction == 'left':
            pts = np.array([[cx - s, cy], [cx + s, cy - s], [cx + s, cy + s]], dtype=np.int32)
        else:  # right
            pts = np.array([[cx + s, cy], [cx - s, cy - s], [cx - s, cy + s]], dtype=np.int32)

        cv2.fillPoly(bg, [pts], (255, 255, 255), lineType=cv2.LINE_AA)
        cv2.polylines(bg, [pts], isClosed=True, color=(0, 0, 0), thickness=border_th, lineType=cv2.LINE_AA)

        centroid = pts.mean(axis=0).astype(int)
        x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
        x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
        allow_w = max(8, int((x_max - x_min) * 0.55))
        allow_h = max(8, int((y_max - y_min) * 0.45))

        (tw1, th1), _ = cv2.getTextSize(label, font, 0.7, 3)
        if tw1 == 0 or th1 == 0:
            return
        scale = min(allow_w / tw1, allow_h / th1)
        scale = max(0.4, min(2.0, scale))  # clamp range
        thickness = max(1, int(scale * 1.2))

        (tw, th), bl = cv2.getTextSize(label, font, scale, thickness)

        text_org = (int(centroid[0] - tw / 2), int(centroid[1] + th / 2) - 1)

        cv2.putText(bg, label, text_org, font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)

    draw_triangle_with_label(img, (w // 2, m + s), 'up', 'FRONT')

    draw_triangle_with_label(img, (w // 2, h - m - s), 'down', 'BACK')

    draw_triangle_with_label(img, (m + s, h // 2), 'left', 'LEFT')

    draw_triangle_with_label(img, (w - m - s, h // 2), 'right', 'RIGHT')

    return img

def _overlay_bev_height(image, points, project, floor_height):
    """Tint observed elevated/lower samples only; preserve base map geometry."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return image
    levels = np.floor(np.abs(points[:, 2] - floor_height) / 0.25).astype(int)
    pixels = project(points)
    valid = ((levels > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < image.shape[1])
             & (pixels[:, 1] >= 0) & (pixels[:, 1] < image.shape[0]))
    # Farther layers first, nearest floor wins if multiple heights share a pixel.
    order = np.argsort(-levels[valid], kind='stable')
    pixels, levels = pixels[valid][order], levels[valid][order]
    higher = (points[valid, 2] > floor_height)[order]
    for band in np.unique(levels)[::-1]:
        for above in (False, True):
            selected = pixels[(levels == band) & (higher == above)]
            if not len(selected):
                continue
            shade = max(60, 220 - 20 * int(band))
            color = np.array((40, shade, 240) if above else (240, shade, 40))  # BGR
            x, y = selected[:, 0], selected[:, 1]
            image[y, x] = (0.35 * image[y, x] + 0.65 * color).astype(np.uint8)
    return image


def create_top_down_map_centered(
    mapper: Instruct_Mapper,
    fov_angle_deg: float,
    target_point=None,
    map_scale=0.025,
    map_size_px=1024,
    fov_range=1.0,
    show_all_labels=False,
    label_volume_threshold=100,
    crop_size_m=16.0,
    action_candidates=None,
    subtasks=None,
):
    """
    action_candidates: Dict[str, Dict], each entry example:
      "waypoint 1": {
        "type": "waypoint",
        "target_point_world": [x, y, z]
      }
    subtasks: Dict[str, Sequence[float]] subtask name -> world coordinate ([x,y,z]).
    """
    t_start = time.time()
    timings = {}

    t = time.time()
    agent_pos = mapper.current_position
    agent_rot = mapper.current_rotation

    forward_vec_3d = agent_rot @ np.array([0, 0, -1])
    forward_vec_2d = forward_vec_3d[:2]
    if np.linalg.norm(forward_vec_2d) < 1e-6:
        forward_vec_2d = np.array([0., -1.])
    forward_vec_2d /= np.linalg.norm(forward_vec_2d)

    agent_angle_rad = np.arctan2(forward_vec_2d[1], forward_vec_2d[0])
    rotation_angle = -agent_angle_rad - np.pi / 2

    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    rotation_matrix = np.array([[c, -s], [s, c]])

    def transform_and_to_pixel(world_coords):
        if world_coords.ndim == 1:
            world_coords = world_coords.reshape(1, -1)

        translated_coords = world_coords[:, :2] - agent_pos[:2]

        rotated_coords = translated_coords @ rotation_matrix.T

        pixel_coords = (rotated_coords / map_scale) + np.array([map_size_px / 2, map_size_px / 2])
        return pixel_coords.astype(int)

    timings['calc_transform'] = time.time() - t

    top_down_map = np.full((map_size_px, map_size_px, 3), 255, dtype=np.uint8)
    map_h, map_w = top_down_map.shape[:2]

    pcd_resolution = 0.025
    dilation_radius = max(1, int(pcd_resolution / map_scale / 2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius * 2 + 1, dilation_radius * 2 + 1))

    agent_ground_z = float(agent_pos[2] - 1.2)

    t = time.time()
    if not mapper.navigable_pcd.is_empty() and mapper.navigable_pcd.point.positions.shape[0] > 0:
        nav_pts = mapper.navigable_pcd.point.positions.cpu().numpy()
        # Dynamic height slicing: show walkable ground for current floor level
        nav_z_mask = (nav_pts[:, 2] >= agent_ground_z - 0.4) & (nav_pts[:, 2] <= agent_ground_z + 1.2)
        if np.any(nav_z_mask):
            nav_pixels = transform_and_to_pixel(nav_pts[nav_z_mask])
            valid_mask = (nav_pixels[:, 0] >= 0) & (nav_pixels[:, 0] < map_w) & (nav_pixels[:, 1] >= 0) & (
                        nav_pixels[:, 1] < map_h)
            if np.any(valid_mask):
                nav_mask = np.zeros((map_h, map_w), dtype=np.uint8)
                nav_mask[nav_pixels[valid_mask, 1], nav_pixels[valid_mask, 0]] = 255
                dilated_nav_mask = cv2.dilate(nav_mask, kernel)
                top_down_map[dilated_nav_mask == 255] = (200, 200, 200)
    timings['draw_navigable'] = time.time() - t

    t = time.time()
    obstacle_mask_bool = np.zeros((map_h, map_w), dtype=bool)
    if not mapper.obstacle_pcd.is_empty() and mapper.obstacle_pcd.point.positions.shape[0] > 0:
        obs_pts = mapper.obstacle_pcd.point.positions.cpu().numpy()
        # Dynamic obstacle height slicing: eliminate cross-floor wall projections
        obs_z_mask = (obs_pts[:, 2] >= agent_ground_z + 0.15) & (obs_pts[:, 2] <= agent_ground_z + 1.8)
        if np.any(obs_z_mask):
            obs_pixels = transform_and_to_pixel(obs_pts[obs_z_mask])
            valid_mask = (obs_pixels[:, 0] >= 0) & (obs_pixels[:, 0] < map_w) & (obs_pixels[:, 1] >= 0) & (
                        obs_pixels[:, 1] < map_h)
            if np.any(valid_mask):
                obs_mask = np.zeros((map_h, map_w), dtype=np.uint8)
                obs_mask[obs_pixels[valid_mask, 1], obs_pixels[valid_mask, 0]] = 255
                dilated_obs_mask = cv2.dilate(obs_mask, kernel)
                top_down_map[dilated_obs_mask == 255] = (50, 50, 50)
                obstacle_mask_bool = (dilated_obs_mask == 255)

                contours, _ = cv2.findContours(dilated_obs_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(top_down_map, contours, -1, (0, 0, 0), thickness=5)
    timings['draw_obstacle'] = time.time() - t

    # Dynamic Stair & Stair Portal Rendering (Topological persistence)
    if hasattr(mapper, 'stair_pcd') and not mapper.stair_pcd.is_empty():
        st_pts = mapper.stair_pcd.point.positions.cpu().numpy()
        st_z_mask = (st_pts[:, 2] >= agent_ground_z - 1.6) & (st_pts[:, 2] <= agent_ground_z + 1.6)
        if np.any(st_z_mask):
            st_pixels = transform_and_to_pixel(st_pts[st_z_mask])
            st_valid = (st_pixels[:, 0] >= 0) & (st_pixels[:, 0] < map_w) & (st_pixels[:, 1] >= 0) & (st_pixels[:, 1] < map_h)
            if np.any(st_valid):
                st_mask = np.zeros((map_h, map_w), dtype=np.uint8)
                st_mask[st_pixels[st_valid, 1], st_pixels[st_valid, 0]] = 255
                dilated_st_mask = cv2.dilate(st_mask, kernel)
                top_down_map[dilated_st_mask == 255] = (220, 160, 255)

    if hasattr(mapper, 'stair_portals') and mapper.stair_portals:
        for portal in mapper.stair_portals:
            p_pos = portal.get('pos')
            if p_pos is not None:
                p_pix = transform_and_to_pixel(p_pos)
                if p_pix.shape[0] > 0:
                    px, py = int(p_pix[0, 0]), int(p_pix[0, 1])
                    if 15 < px < map_w - 15 and 15 < py < map_h - 15:
                        p_label = "STAIR?-UP" if portal.get('type') == 'stair_top' else "STAIR?-DN"
                        cv2.rectangle(top_down_map, (px - 14, py - 14), (px + 14, py + 14), (255, 140, 0), 2)
                        cv2.putText(top_down_map, p_label, (px - 22, py - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 140, 0), 1)

    t = time.time()
    if hasattr(mapper, 'frontier_pcd') and not mapper.frontier_pcd.is_empty():
        frontier_pixels = transform_and_to_pixel(mapper.frontier_pcd.point.positions.cpu().numpy())
        valid_mask = (frontier_pixels[:, 0] >= 0) & (frontier_pixels[:, 0] < map_w) & \
                     (frontier_pixels[:, 1] >= 0) & (frontier_pixels[:, 1] < map_h)
        valid_frontier_pixels = frontier_pixels[valid_mask]
        for pixel in valid_frontier_pixels:
            cv2.circle(top_down_map, tuple(pixel), 1, (0, 255, 255), -1)  # yellow dot
    timings['draw_frontier'] = time.time() - t

    t = time.time()
    if len(mapper.object_entities) > 0:
        for entity in mapper.object_entities:
            if entity['pcd'].point.positions.shape[0] > 0:
                obj_pixels = transform_and_to_pixel(entity['pcd'].point.positions.cpu().numpy())
                valid_pixels_mask = (obj_pixels[:, 0] >= 0) & (obj_pixels[:, 0] < map_w) & (obj_pixels[:, 1] >= 0) & (
                            obj_pixels[:, 1] < map_h)
                if not np.any(valid_pixels_mask): continue

                valid_obj_pixels = obj_pixels[valid_pixels_mask]
                intersection_mask = obstacle_mask_bool[valid_obj_pixels[:, 1], valid_obj_pixels[:, 0]]

                if np.any(intersection_mask):
                    final_pixels = valid_obj_pixels[intersection_mask]
                    if len(final_pixels) == 0: continue

                    obj_colors_rgb = entity['pcd'].point.colors.cpu().numpy()[valid_pixels_mask][intersection_mask]
                    object_color = (obj_colors_rgb[0] * 255)[::-1].astype(np.uint8)

                    object_mask_img = np.zeros((map_h, map_w), dtype=np.uint8)
                    object_mask_img[final_pixels[:, 1], final_pixels[:, 0]] = 255
                    fill_kernel = np.ones((5, 5), np.uint8)
                    filled_mask = cv2.morphologyEx(object_mask_img, cv2.MORPH_CLOSE, fill_kernel, iterations=2)
                    top_down_map[filled_mask == 255] = object_color
    timings['draw_objects'] = time.time() - t
    _overlay_bev_height(top_down_map, mapper.get_bev_height_points(),
                        transform_and_to_pixel, mapper.floor_height)

    label_upscale_factor = 2
    if label_upscale_factor > 1:
        new_h, new_w = map_h * label_upscale_factor, map_w * label_upscale_factor
        top_down_map = cv2.resize(top_down_map, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        new_h, new_w = map_h, map_w

    t = time.time()
    drawn_labels = []
    label_merge_distance_px = 50
    for entity in sorted(mapper.object_entities, key=lambda e: e['class']):
        if not show_all_labels and entity['pcd'].point.positions.shape[0] < label_volume_threshold: continue
        if entity['pcd'].point.positions.shape[0] == 0: continue

        center_pixel = transform_and_to_pixel(entity['center'])[0]
        class_name = entity['class_name']

        should_merge = False
        for i, label_info in enumerate(drawn_labels):
            if label_info['class'] == class_name and np.linalg.norm(
                    center_pixel - label_info['center_pixel']) < label_merge_distance_px:
                drawn_labels[i]['center_pixel'] = (label_info['center_pixel'] + center_pixel) / 2
                should_merge = True
                break
        if not should_merge:
            drawn_labels.append({'class': class_name, 'center_pixel': center_pixel})

    for label_info in drawn_labels:
        center_pixel_scaled = (label_info['center_pixel'] * label_upscale_factor).astype(int)
        if not (0 <= center_pixel_scaled[0] < new_w and 0 <= center_pixel_scaled[1] < new_h): continue

        class_name = label_info['class']
        font_scale, font_thickness = 0.7, 1
        (tw, th), bl = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        box_tl = (center_pixel_scaled[0] - tw // 2 - 3, center_pixel_scaled[1] - th - 3)
        box_br = (center_pixel_scaled[0] + tw // 2 + 3, center_pixel_scaled[1] + bl)

        alpha = 0.8
        overlay = top_down_map.copy()
        cv2.rectangle(overlay, box_tl, box_br, (255, 255, 255), -1)
        cv2.addWeighted(overlay, alpha, top_down_map, 1 - alpha, 0, top_down_map)
        cv2.rectangle(top_down_map, box_tl, box_br, (0, 0, 0), 1)

        text_org = (box_tl[0] + 3, box_br[1] - bl)
        cv2.putText(top_down_map, class_name, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), font_thickness, cv2.LINE_AA)
    timings['draw_labels'] = time.time() - t

    t = time.time()
    if len(mapper.trajectory_position) > 1:
        traj_pixels = transform_and_to_pixel(np.array(mapper.trajectory_position[-150:]))
        valid_traj_pixels = (traj_pixels * label_upscale_factor).astype(int)
        overlay = top_down_map.copy()
        cv2.polylines(overlay, [valid_traj_pixels], isClosed=False, color=(0, 0, 255),
                      thickness=2 * label_upscale_factor)
        alpha = 0.5  # trajectory alpha, adjustable
        top_down_map = cv2.addWeighted(overlay, alpha, top_down_map, 1 - alpha, 0)
    timings['draw_trajectory'] = time.time() - t

    # waypoints = mapper.get_candidate_waypoints(min_distance=0.2, max_distance=1.0, waypoint_grid_resolution=1.0)

    # if waypoints.shape[0] > 0:
    #     waypoint_pixels = transform_and_to_pixel(waypoints)
    #     waypoint_pixels_scaled = (waypoint_pixels * label_upscale_factor).astype(int)
    #     for pixel in waypoint_pixels_scaled:
    #         if 0 <= pixel[0] < new_w and 0 <= pixel[1] < new_h:

    #
    t = time.time()
    if target_point is not None:
        target_pixel = transform_and_to_pixel(np.array(target_point))[0]
        target_pixel_scaled = (target_pixel * label_upscale_factor).astype(int)

        if (0 <= target_pixel_scaled[0] < new_w) and (0 <= target_pixel_scaled[1] < new_h):

            cv2.circle(top_down_map, tuple(target_pixel_scaled), 8, (0, 255, 0), -1)  # green filled circle
            cv2.circle(top_down_map, tuple(target_pixel_scaled), 8, (0, 0, 0), 2)  # black border
    timings['draw_target'] = time.time() - t

    new_h, new_w = top_down_map.shape[:2]  # use resized dims if already resized

    if action_candidates:

        overlay = top_down_map.copy()

        for text, cand in action_candidates.items():
            if cand.get('type') != 'waypoint':
                continue  # only handle waypoint type for now

            if not re.fullmatch(r'WP_\d{2,}', str(text)):
                continue

            coord = cand.get('target_point_world')
            if coord is None:
                continue

            pix = transform_and_to_pixel(np.array(coord, dtype=float))[0]
            pix = (pix * label_upscale_factor).astype(int)

            if not (0 <= pix[0] < new_w and 0 <= pix[1] < new_h):
                continue

            radius = max(10, int(7 * label_upscale_factor))
            circle_color = (0, 128, 255)  # orange border (BGR)
            fill_color = (255, 255, 255)  # white background
            border_thickness = max(2, label_upscale_factor)

            cv2.circle(overlay, tuple(pix), radius, fill_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(overlay, tuple(pix), radius, circle_color, border_thickness, lineType=cv2.LINE_AA)

            text_idx = str(text)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5 * label_upscale_factor
            font_thickness = max(1, label_upscale_factor)
            (tw, th), bl = cv2.getTextSize(text_idx, font, font_scale, font_thickness)
            text_org = (int(pix[0] - tw / 2), int(pix[1] + th / 2))
            cv2.putText(overlay, text_idx, text_org, font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

        top_down_map = cv2.addWeighted(overlay, 0.9, top_down_map, 0.1, 0)

    t = time.time()
    agent_pixel_scaled = (np.array([map_size_px / 2, map_size_px / 2]) * label_upscale_factor).astype(int)

    fov_rad = np.deg2rad(fov_angle_deg)
    angles = np.linspace(-np.pi / 2 - fov_rad / 2, -np.pi / 2 + fov_rad / 2, 30)
    fov_range_px = fov_range / map_scale * label_upscale_factor
    sector_points = [(agent_pixel_scaled + (np.array([np.cos(a), np.sin(a)]) * fov_range_px)).astype(int) for a in
                     angles]
    sector_pts = np.array([agent_pixel_scaled] + sector_points, np.int32)

    overlay = top_down_map.copy()
    cv2.fillPoly(overlay, [sector_pts], (255, 255, 0), lineType=cv2.LINE_AA)
    top_down_map = cv2.addWeighted(overlay, 0.2, top_down_map, 0.8, 0)

    if subtasks:
        overlay = top_down_map.copy()

        new_h, new_w = overlay.shape[:2]

        for name, coord in subtasks.items():
            if coord is None:
                continue

            try:
                pix = transform_and_to_pixel(np.array(coord, dtype=float))[0]
            except Exception:
                continue
            pix = (pix * label_upscale_factor).astype(int)

            if not (0 <= pix[0] < new_w and 0 <= pix[1] < new_h):
                continue

            half = max(6, int(6 * label_upscale_factor))
            tl = (int(pix[0] - half), int(pix[1] - half))
            br = (int(pix[0] + half), int(pix[1] + half))

            tl = (max(0, tl[0]), max(0, tl[1]))
            br = (min(new_w - 1, br[0]), min(new_h - 1, br[1]))

            cv2.rectangle(overlay, tl, br, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.rectangle(overlay, tl, br, (0, 0, 0), 1, lineType=cv2.LINE_AA)

            text = str(name)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5 * label_upscale_factor
            font_thickness = max(1, label_upscale_factor)
            (tw, th), bl = cv2.getTextSize(text, font, font_scale, font_thickness)

            tx = br[0] + int(6 * label_upscale_factor)
            ty = tl[1] - int(4 * label_upscale_factor)

            bx1 = max(0, tx - 3)
            by1 = max(0, ty - th - 3)
            bx2 = min(new_w - 1, tx + tw + 3)
            by2 = min(new_h - 1, ty + 3)

            if bx2 > bx1 and by2 > by1:
                cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (255, 255, 255), -1)

            text_org = (min(max(0, tx), new_w - 1 - tw), min(max(th, ty), new_h - 1))
            cv2.putText(overlay, text, text_org, font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

        top_down_map = cv2.addWeighted(overlay, 0.95, top_down_map, 0.05, 0)

    arrow_length = int(12 * label_upscale_factor * 1.2)  # shorter arrow, less sharp
    base_half = max(12, int(arrow_length * 0.6))  # wider base for blunter triangle

    triangle_pts = np.array([
        agent_pixel_scaled + np.array([0, -arrow_length]),  # apex
        agent_pixel_scaled + np.array([-base_half, int(arrow_length * 0.6)]),  # bottom-left
        agent_pixel_scaled + np.array([base_half, int(arrow_length * 0.6)])  # bottom-right
    ], dtype=np.int32)

    cv2.fillPoly(top_down_map, [triangle_pts], (0, 0, 255), lineType=cv2.LINE_AA)

    outline_thickness = max(3, 2 * label_upscale_factor)
    cv2.polylines(top_down_map, [triangle_pts], isClosed=True, color=(100, 100, 100),
                  thickness=outline_thickness, lineType=cv2.LINE_AA)

    timings['draw_agent'] = time.time() - t

    try:
        start_world = np.array([0.0, 0.0, 0.0])
        start_pixel = transform_and_to_pixel(start_world)[0]  # returns (x, y)
        start_pixel_scaled = (start_pixel * label_upscale_factor).astype(int)

        if 0 <= start_pixel_scaled[0] < new_w and 0 <= start_pixel_scaled[1] < new_h:

            cv2.circle(top_down_map, tuple(start_pixel_scaled), 7, (255, 0, 0), -1)  # solid blue
            cv2.circle(top_down_map, tuple(start_pixel_scaled), 9, (0, 0, 0), 2)  # black border

            text = "START"
            font_scale = 0.7
            thickness = 1
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_org = (start_pixel_scaled[0] + 10, start_pixel_scaled[1] + th // 2)

            bx1 = text_org[0] - 3
            by1 = text_org[1] - th - 3
            bx2 = text_org[0] + tw + 3
            by2 = text_org[1] + 3

            bx1 = max(bx1, 0);
            by1 = max(by1, 0);
            bx2 = min(bx2, new_w - 1);
            by2 = min(by2, new_h - 1)
            overlay = top_down_map.copy()
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.8, top_down_map, 0.2, 0, top_down_map)

            cv2.putText(top_down_map, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness,
                        cv2.LINE_AA)
    except Exception:

        pass

    crop_size_px = int(crop_size_m / map_scale * label_upscale_factor)
    agent_pixel_scaled = (np.array([map_size_px / 2, map_size_px / 2]) * label_upscale_factor).astype(int)
    half_crop = crop_size_px // 2
    y1 = max(agent_pixel_scaled[1] - half_crop, 0)
    y2 = min(agent_pixel_scaled[1] + half_crop, top_down_map.shape[0])
    x1 = max(agent_pixel_scaled[0] - half_crop, 0)
    x2 = min(agent_pixel_scaled[0] + half_crop, top_down_map.shape[1])
    top_down_map_cropped = top_down_map[y1:y2, x1:x2]

    top_down_map_cropped = draw_direction_markers(top_down_map_cropped)

    t_end = time.time()

    return top_down_map_cropped

def create_top_down_map_global(
    mapper: Instruct_Mapper,
    map_scale: float = 0.025,
    padding_px: int = 60,
    fov_angle_deg: float = 60.0,
    fov_range: float = 2.5,
    target_point=None,
    action_candidates=None,
    subtasks=None,
    enable_semantic: bool = True,
    enable_navigation: bool = True,
    show_all_labels: bool = False,
    label_volume_threshold: int = 100,
):
    """
    Global top-down view (not agent-centered).:
    - Auto-compute scene bounds and expand canvas.
    - enable_semantic: whether to render object point clouds and semantic labels.
    - enable_navigation: whether to render target / candidate waypoints / subtask nodes.
    """
    t_start = time.time()

    point_sets = []

    if hasattr(mapper, "navigable_pcd") and not mapper.navigable_pcd.is_empty():
        point_sets.append(mapper.navigable_pcd.point.positions.cpu().numpy()[:, :2])
    if hasattr(mapper, "obstacle_pcd") and not mapper.obstacle_pcd.is_empty():
        point_sets.append(mapper.obstacle_pcd.point.positions.cpu().numpy()[:, :2])
    if enable_semantic and hasattr(mapper, "object_entities"):
        for ent in mapper.object_entities:
            if ent['pcd'].point.positions.shape[0] > 0:
                point_sets.append(ent['pcd'].point.positions.cpu().numpy()[:, :2])
    if hasattr(mapper, "frontier_pcd") and not mapper.frontier_pcd.is_empty():
        point_sets.append(mapper.frontier_pcd.point.positions.cpu().numpy()[:, :2])
    if mapper.trajectory_position:
        traj_arr = np.array(mapper.trajectory_position)[:, :2]
        point_sets.append(traj_arr)
    if enable_navigation:
        if target_point is not None:
            point_sets.append(np.array(target_point)[None, :2])
        if action_candidates:
            for cand in action_candidates.values():
                coord = cand.get("target_point_world")
                if coord is not None:
                    point_sets.append(np.array(coord)[None, :2])
        if subtasks:
            for coord in subtasks.values():
                if coord is not None:
                    point_sets.append(np.array(coord)[None, :2])

    if not point_sets:

        map_size_px = 1024
        top_down_map = np.full((map_size_px, map_size_px, 3), 255, dtype=np.uint8)
        return top_down_map

    all_pts = np.vstack(point_sets)
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)

    span = max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1])

    base_size_px = int(span / map_scale) + 2 * padding_px

    base_size_px = max(base_size_px, 1024)

    width_px = int((max_xy[0] - min_xy[0]) / map_scale) + 2 * padding_px
    height_px = int((max_xy[1] - min_xy[1]) / map_scale) + 2 * padding_px
    width_px = max(width_px, 1024)
    height_px = max(height_px, 1024)

    top_down_map = np.full((height_px, width_px, 3), 255, dtype=np.uint8)
    map_h, map_w = top_down_map.shape[:2]
    offset = min_xy - padding_px * map_scale  # world-coord bottom-left offset

    def world_to_pixel(world_coords):
        arr = np.asarray(world_coords, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        px = (arr[:, :2] - offset) / map_scale
        return px.astype(int)

    pcd_resolution = 0.025
    dilation_radius = max(1, int(pcd_resolution / map_scale / 2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius * 2 + 1, dilation_radius * 2 + 1))

    if hasattr(mapper, "navigable_pcd") and not mapper.navigable_pcd.is_empty():
        nav_pts_world = mapper.navigable_pcd.point.positions.cpu().numpy()
        nav_px = world_to_pixel(nav_pts_world)
        valid = (nav_px[:, 0] >= 0) & (nav_px[:, 0] < map_w) & (nav_px[:, 1] >= 0) & (nav_px[:, 1] < map_h)
        if np.any(valid):
            mask = np.zeros((map_h, map_w), dtype=np.uint8)
            mask[nav_px[valid, 1], nav_px[valid, 0]] = 255
            mask = cv2.dilate(mask, kernel)
            top_down_map[mask == 255] = (200, 200, 200)

    if enable_navigation:
        try:
            wps = mapper.get_candidate_waypoints(min_distance=0.3, max_distance=1.5, waypoint_grid_resolution=0.5)
            if wps is not None:

                if hasattr(wps, "detach"):
                    wps = wps.detach().cpu().numpy()
                else:
                    wps = np.asarray(wps)
                if wps.size > 0:
                    if wps.ndim == 1:
                        wps = wps.reshape(1, -1)

                    wp_xy = wps[:, :2]
                    wp_px = world_to_pixel(wp_xy)
                    valid = (wp_px[:, 0] >= 0) & (wp_px[:, 0] < map_w) & (wp_px[:, 1] >= 0) & (wp_px[:, 1] < map_h)
                    if np.any(valid):
                        overlay = top_down_map.copy()
                        for p in wp_px[valid]:

                            cv2.circle(overlay, tuple(p), 2, (0, 165, 255), -1, cv2.LINE_AA)
                        top_down_map = cv2.addWeighted(overlay, 0.9, top_down_map, 0.1, 0)
        except Exception:

            pass

    obstacle_mask_bool = np.zeros((map_h, map_w), dtype=bool)
    if hasattr(mapper, "obstacle_pcd") and not mapper.obstacle_pcd.is_empty():
        obs_pts_world = mapper.obstacle_pcd.point.positions.cpu().numpy()
        obs_px = world_to_pixel(obs_pts_world)
        valid = (obs_px[:, 0] >= 0) & (obs_px[:, 0] < map_w) & (obs_px[:, 1] >= 0) & (obs_px[:, 1] < map_h)
        if np.any(valid):
            mask = np.zeros((map_h, map_w), dtype=np.uint8)
            mask[obs_px[valid, 1], obs_px[valid, 0]] = 255
            mask = cv2.dilate(mask, kernel)
            top_down_map[mask == 255] = (50, 50, 50)
            obstacle_mask_bool = (mask == 255)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(top_down_map, contours, -1, (0, 0, 0), thickness=4)

    if hasattr(mapper, "frontier_pcd") and not mapper.frontier_pcd.is_empty():
        fr_px = world_to_pixel(mapper.frontier_pcd.point.positions.cpu().numpy())
        valid = (fr_px[:, 0] >= 0) & (fr_px[:, 0] < map_w) & (fr_px[:, 1] >= 0) & (fr_px[:, 1] < map_h)
        for p in fr_px[valid]:
            cv2.circle(top_down_map, tuple(p), 1, (0, 255, 255), -1)

    if enable_semantic and hasattr(mapper, "object_entities"):
        for ent in mapper.object_entities:
            if ent['pcd'].point.positions.shape[0] == 0:
                continue
            obj_px = world_to_pixel(ent['pcd'].point.positions.cpu().numpy())
            valid = (obj_px[:, 0] >= 0) & (obj_px[:, 0] < map_w) & (obj_px[:, 1] >= 0) & (obj_px[:, 1] < map_h)
            if not np.any(valid):
                continue
            obj_valid_px = obj_px[valid]

            inter_mask = obstacle_mask_bool[obj_valid_px[:, 1], obj_valid_px[:, 0]]
            if np.any(inter_mask):
                final_px = obj_valid_px[inter_mask]
                color_arr = ent['pcd'].point.colors.cpu().numpy()[valid][inter_mask]
            else:
                final_px = obj_valid_px
                color_arr = ent['pcd'].point.colors.cpu().numpy()[valid]
            if final_px.shape[0] == 0:
                continue

            obj_color = (color_arr[0] * 255)[::-1].astype(np.uint8)
            mask_img = np.zeros((map_h, map_w), dtype=np.uint8)
            mask_img[final_px[:, 1], final_px[:, 0]] = 255
            fill_kernel = np.ones((5, 5), np.uint8)
            filled = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, fill_kernel, iterations=2)
            top_down_map[filled == 255] = obj_color

    if False and enable_semantic and hasattr(mapper, "object_entities"):
        drawn_labels = []
        merge_dist_px = 40
        for ent in sorted(mapper.object_entities, key=lambda e: e['class']):
            if ent['pcd'].point.positions.shape[0] == 0:
                continue
            if (not show_all_labels) and ent['pcd'].point.positions.shape[0] < label_volume_threshold:
                continue
            center_px = world_to_pixel(ent['center'])[0]
            if not (0 <= center_px[0] < map_w and 0 <= center_px[1] < map_h):
                continue
            cname = ent['class_name']
            merged = False
            for i, info in enumerate(drawn_labels):
                if info['class'] == cname and np.linalg.norm(center_px - info['center']) < merge_dist_px:
                    drawn_labels[i]['center'] = (info['center'] + center_px) / 2
                    merged = True
                    break
            if not merged:
                drawn_labels.append({'class': cname, 'center': center_px})

        for info in drawn_labels:
            cp = info['center'].astype(int)
            font_scale = 0.6
            thickness = 1
            (tw, th), bl = cv2.getTextSize(info['class'], cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            tl = (cp[0] - tw // 2 - 3, cp[1] - th - 4)
            br = (cp[0] + tw // 2 + 3, cp[1] + bl + 2)
            tl = (max(0, tl[0]), max(0, tl[1]))
            br = (min(map_w - 1, br[0]), min(map_h - 1, br[1]))
            overlay = top_down_map.copy()
            cv2.rectangle(overlay, tl, br, (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.85, top_down_map, 0.15, 0, top_down_map)
            cv2.rectangle(top_down_map, tl, br, (0, 0, 0), 1)
            text_org = (tl[0] + 3, br[1] - bl)
            cv2.putText(top_down_map, info['class'], text_org, cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    if enable_navigation and mapper.trajectory_position and len(mapper.trajectory_position) > 1:
        traj_px = world_to_pixel(np.array(mapper.trajectory_position[-200:])[:, :2]).astype(int)
        valid = (traj_px[:, 0] >= 0) & (traj_px[:, 0] < map_w) & (traj_px[:, 1] >= 0) & (traj_px[:, 1] < map_h)
        traj_px = traj_px[valid]
        if traj_px.shape[0] > 1:
            overlay = top_down_map.copy()
            cv2.polylines(overlay, [traj_px], False, (0, 0, 255), 2)
            top_down_map = cv2.addWeighted(overlay, 0.6, top_down_map, 0.4, 0)

    if enable_navigation:

        # if target_point is not None:
        #     tp = world_to_pixel(np.array(target_point))[0]
        #     if 0 <= tp[0] < map_w and 0 <= tp[1] < map_h:
        #         cv2.circle(top_down_map, tuple(tp), 9, (0, 255, 0), -1)
        #         cv2.circle(top_down_map, tuple(tp), 9, (0, 0, 0), 2)

        if action_candidates:
            overlay = top_down_map.copy()
            for text, cand in action_candidates.items():
                if cand.get("type") != "waypoint":
                    continue
                coord = cand.get("target_point_world")
                if coord is None:
                    continue
                pix = world_to_pixel(np.array(coord))[0]
                if not (0 <= pix[0] < map_w and 0 <= pix[1] < map_h):
                    continue

                label = str(text) if re.fullmatch(r'WP_\d{2,}', str(text)) else None
                radius = 11
                cv2.circle(overlay, tuple(pix), radius, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(overlay, tuple(pix), radius, (0, 128, 255), 2, cv2.LINE_AA)
                if label is not None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (tw, th), bl = cv2.getTextSize(label, font, 0.6, 2)
                    org = (int(pix[0] - tw / 2), int(pix[1] + th / 2))
                    cv2.putText(overlay, label, org, font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            top_down_map = cv2.addWeighted(overlay, 0.9, top_down_map, 0.1, 0)

        if subtasks:
            overlay = top_down_map.copy()
            for name, coord in subtasks.items():
                if coord is None:
                    continue
                pix = world_to_pixel(np.array(coord))[0]
                if not (0 <= pix[0] < map_w and 0 <= pix[1] < map_h):
                    continue

                half = 10
                tl = (pix[0] - half, pix[1] - half)
                br = (pix[0] + half, pix[1] + half)
                tl = (max(0, tl[0]), max(0, tl[1]))
                br = (min(map_w - 1, br[0]), min(map_h - 1, br[1]))

                cv2.rectangle(overlay, tl, br, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.rectangle(overlay, tl, br, (0, 0, 0), 1, cv2.LINE_AA)

                text = str(name)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.45  # fit 16x16 block
                thickness = 1
                (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)

                cx = (tl[0] + br[0]) // 2
                cy = (tl[1] + br[1]) // 2

                text_org = (int(cx - tw / 2), int(cy + th / 2) - 1)

                cv2.putText(overlay, text, text_org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            top_down_map = cv2.addWeighted(overlay, 0.95, top_down_map, 0.05, 0)

    agent_pos = getattr(mapper, "current_position", np.array([0.0, 0.0, 0.0]))
    agent_rot = getattr(mapper, "current_rotation", np.eye(3))
    agent_px = world_to_pixel(agent_pos)[0]
    if 0 <= agent_px[0] < map_w and 0 <= agent_px[1] < map_h:
        forward_vec_3d = agent_rot @ np.array([0, 0, -1])
        forward_2d = forward_vec_3d[:2]
        if np.linalg.norm(forward_2d) < 1e-6:
            forward_2d = np.array([1.0, 0.0])
        forward_2d /= np.linalg.norm(forward_2d)
        angle = np.arctan2(forward_2d[1], forward_2d[0])

        # fov_rad = np.deg2rad(fov_angle_deg)
        # num_pts = 40
        # angles = np.linspace(angle - fov_rad / 2, angle + fov_rad / 2, num_pts)
        # fov_range_px = fov_range / map_scale
        # sector_pts = [agent_px]
        # for a in angles:
        #     end_world = agent_pos[:2] + np.array([np.cos(a), np.sin(a)]) * fov_range
        #     sector_pts.append(world_to_pixel(end_world)[0])
        # overlay = top_down_map.copy()
        # cv2.fillPoly(overlay, [np.array(sector_pts, dtype=np.int32)], (255, 255, 0))
        # top_down_map = cv2.addWeighted(overlay, 0.2, top_down_map, 0.8, 0)

        length = 30
        base_half = 10
        tip = agent_px + (forward_2d * length).astype(int)
        left_dir = np.array([[0, -1], [1, 0]]) @ forward_2d
        right_dir = -left_dir
        p_left = agent_px + (forward_2d * (length * 0.2) + left_dir * base_half).astype(int)
        p_right = agent_px + (forward_2d * (length * 0.2) + right_dir * base_half).astype(int)
        tri = np.array([tip, p_left, p_right], dtype=np.int32)
        cv2.fillPoly(top_down_map, [tri], (0, 0, 255))
        cv2.polylines(top_down_map, [tri], True, (50, 50, 50), 2)

    start_px = world_to_pixel(np.array([0.0, 0.0]))[0]
    if enable_navigation and 0 <= start_px[0] < map_w and 0 <= start_px[1] < map_h:
        cv2.circle(top_down_map, tuple(start_px), 7, (255, 0, 0), -1)
        cv2.circle(top_down_map, tuple(start_px), 9, (0, 0, 0), 2)
        (tw, th), bl = cv2.getTextSize("START", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        org = (start_px[0] + 10, start_px[1] + th // 2)
        bx1, by1 = org[0] - 3, org[1] - th - 3
        bx2, by2 = org[0] + tw + 3, org[1] + 3
        bx1 = max(bx1, 0); by1 = max(by1, 0)
        bx2 = min(bx2, map_w - 1); by2 = min(by2, map_h - 1)
        overlay = top_down_map.copy()
        # cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.8, top_down_map, 0.2, 0, top_down_map)
        # cv2.putText(top_down_map, "START", org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    return cv2.cvtColor(top_down_map, cv2.COLOR_BGR2RGB)


class Instruction:
    """
    Class encapsulating task instructions, decomposition, and subtask management.
    Each subtask has a final_coord attribute, recorded only when marked complete.
    """
    def __init__(self, full_instruction: str, sub_instructions: list = None):
        """
        Initialize Instruction object.
        Contains the full task instruction and an optional list of subtask instructions.
        :param full_instruction: full task instruction text.
        :param sub_instructions: decomposed subtask instruction list.
        """
        self.full_instruction = full_instruction
        self.sub_instruction_dict = {}

        if sub_instructions is not None:
            for idx, sub_inst in enumerate(sub_instructions):
                key = f'P{idx + 1}'
                inst_body = {}
                inst_body['instruction'] = sub_inst
                inst_body['completed'] = False
                inst_body['plan'] = ""
                inst_body['record'] = []
                inst_body['final_coord'] = None
                inst_body['end_position'] = None
                inst_body['target_pos'] = None
                inst_body['end_rotation'] = None
                inst_body['target_rot'] = None
                self.sub_instruction_dict[key] = inst_body

        self.sub_instruction_keys = list(self.sub_instruction_dict.keys())
        self.current_subtask_index = 0
        self.current_step_count = 0
        self.num_subtasks = len(self.sub_instruction_keys)
        self._retrieval_cache = {}

    def get_current_subtask_key(self):
        """Get the current subtask key."""
        if self.current_subtask_index < self.num_subtasks:
            return self.sub_instruction_keys[self.current_subtask_index]
        return None

    def is_last_subtask(self):
        """Check if the current subtask is the last one."""
        return self.current_subtask_index == self.num_subtasks - 1

    def get_current_subtask(self):
        """Get current subtask instruction (key + dict)."""
        key = self.get_current_subtask_key()
        if key:
            return key, self.sub_instruction_dict[key]
        return None, None

    def get_cached_retrieval(self, key):
        """Get cached retrieval memories for a subtask key."""
        if not hasattr(self, '_retrieval_cache'):
            self._retrieval_cache = {}
        return self._retrieval_cache.get(key)

    def set_cached_retrieval(self, key, memories):
        """Cache retrieval memories for a subtask key."""
        if not hasattr(self, '_retrieval_cache'):
            self._retrieval_cache = {}
        self._retrieval_cache[key] = list(memories or [])

    def clear_retrieval_cache(self):
        """Clear retrieval memory cache upon subtask change."""
        self._retrieval_cache = {}

    def update_plan_for_current_subtask(self, plan: str):
        """Update the plan text for the current subtask."""
        key = self.get_current_subtask_key()
        if key:
            self.sub_instruction_dict[key]['plan'] = plan

    def add_record_to_current_subtask(self, record: dict):
        """Append an execution record to the current subtask."""
        key = self.get_current_subtask_key()
        if key:
            self.sub_instruction_dict[key]['record'].append(record)

    def reset_subtask(self, key: str):
        """Reset the subtask for the given key to incomplete, clearing plan/records/coords/rotation."""
        key_upper = key.upper()
        if key_upper in self.sub_instruction_dict:
            self.sub_instruction_dict[key_upper]['completed'] = False
            self.sub_instruction_dict[key_upper]['plan'] = ""
            self.sub_instruction_dict[key_upper]['record'] = []
            self.sub_instruction_dict[key_upper]['final_coord'] = None

            self.sub_instruction_dict[key_upper]['final_rotation'] = None
        if hasattr(self, '_retrieval_cache'):
            self._retrieval_cache.pop(key_upper, None)
            self._retrieval_cache.pop(key, None)
        self.current_step_count = 0

    def get_subtask_by_key(self, key: str):
        """Get subtask instruction by key (case-insensitive)."""
        key_upper = key.upper()
        return self.sub_instruction_dict.get(key_upper, None)

    def get_all_subtasks_str(self):
        """
        Get JSON string representation of all subtasks (only current has record + final_coord).
        """
        result = []
        for idx, key in enumerate(self.sub_instruction_keys):
            subtask = self.sub_instruction_dict[key]

            if subtask['completed']:
                status = "completed"
            elif idx == self.current_subtask_index:
                status = "ongoing(current task)"
            else:
                status = "queuing"

            if idx == self.current_subtask_index:
                record_dict = {}
                recs = subtask['record']
                max_rec = int(os.environ.get("HSGM_SUBTASK_PROMPT_RECORDS", 6))
                start_i = max(0, len(recs) - max_rec) if max_rec > 0 else 0
                for i in range(start_i, len(recs)):
                    rec = recs[i]
                    record_key = f"{key.lower()}.{i + 1}"
                    if isinstance(rec, dict) and 'executed_action' in rec:
                        pose = rec.get('pose') or []
                        pose_r = [round(float(x), 2) for x in pose[:3]] if isinstance(pose, (list, tuple)) else pose
                        record_dict[record_key] = {
                            'step': rec.get('step'),
                            'executed_action': rec.get('executed_action'),
                            'motion_distance': round(float(rec.get('motion_distance') or 0.0), 2),
                            'height_delta': round(float(rec.get('height_delta') or 0.0), 2),
                            'blocked': bool(rec.get('blocked')),
                            'pose': pose_r,
                        }
                    else:
                        record_dict[record_key] = rec
            else:
                record_dict = {}

            task_info = {
                "task_name": key.lower(),
                "instruction": subtask['instruction'],
                "status": status,
                "record": record_dict,
                "your_last_plan": subtask['plan'],
            }
            result.append(task_info)

        all_tasks = json.dumps(result, ensure_ascii=False, indent=0)
        cur_task = f"\n\nCurrent task: {self.get_current_subtask_key()}: {self.get_current_subtask()[1]['instruction']}"
        return f"Full instruction: {self.full_instruction}\n" + all_tasks + cur_task

    def mark_current_subtask_completed(self, coord=None, rotation=None):
        """
        Mark the current subtask as completed and advance to the next.
        coord: final coordinate (x, y, z).
        rotation: final orientation quaternion (supports mn.Quaternion / [w,x,y,z] / objects with w,x,y,z).
        """
        key = self.get_current_subtask_key()
        if key:
            if not self.sub_instruction_dict[key].get('completion_evidence_reliable'):
                raise NavigationAborted('subtask_completion_without_verified_evidence')
            self.sub_instruction_dict[key]['completed'] = True

            if coord is not None:
                try:
                    self.sub_instruction_dict[key]['final_coord'] = [
                        float(coord[0]), float(coord[1]), float(coord[2])
                    ]
                except Exception:
                    self.sub_instruction_dict[key]['final_coord'] = None

            if rotation is not None:
                self.sub_instruction_dict[key]['final_rotation'] = rotation

            if self.current_subtask_index < self.num_subtasks - 1:
                self.current_subtask_index += 1
                following = self.sub_instruction_dict[self.get_current_subtask_key()]
                following['position_anchor_check'] = {'status': 'aligned', 'pose': coord}
            self.current_step_count = 0
            self.clear_retrieval_cache()


    def is_all_completed(self):
        """Check if all subtasks are completed."""
        return all(subtask['completed'] for subtask in self.sub_instruction_dict.values())

    def roll_back_to_specific_subtask(self, key: str):
        """Roll back to the subtask identified by key."""
        key_upper = key.upper()
        if key_upper in self.sub_instruction_keys:
            target_index = self.sub_instruction_keys.index(key_upper)
            self.current_subtask_index = target_index

            for i in range(target_index, self.num_subtasks):
                reset_key = self.sub_instruction_keys[i]
                self.reset_subtask(reset_key)
            self.clear_retrieval_cache()

    def get_subtasks_key_coord(self):
        """Get a dict of all completed subtask keys with their final coordinates."""
        result = {}
        for key in self.sub_instruction_keys:
            subtask = self.sub_instruction_dict[key]
            if subtask['completed'] and subtask['final_coord'] is not None:
                result[key] = subtask['final_coord']
        return result

    def bind_episode_targets(self, goal_position=None, reference_path=None):
        """
        Bind ground truth target positions to subtasks.
        - The last subtask receives goal_position (or reference_path[-1]).
        - Intermediate subtasks are mapped to points along reference_path (if available).
        """
        num_keys = len(self.sub_instruction_keys)
        if num_keys == 0:
            return
        ref_path = list(reference_path) if reference_path is not None and len(reference_path) > 0 else None
        goal_pos = np.asarray(goal_position, dtype=np.float32).tolist() if goal_position is not None else (
            np.asarray(ref_path[-1], dtype=np.float32).tolist() if ref_path else None
        )
        for idx, key in enumerate(self.sub_instruction_keys):
            st = self.sub_instruction_dict[key]
            if idx == num_keys - 1 and goal_pos is not None:
                st['end_position'] = list(goal_pos)
                st['target_pos'] = list(goal_pos)
            elif ref_path is not None:
                pt_idx = int(round((idx + 1) / num_keys * (len(ref_path) - 1)))
                target_pt = np.asarray(ref_path[pt_idx], dtype=np.float32).tolist()
                st['end_position'] = target_pt
                st['target_pos'] = list(target_pt)
            elif goal_pos is not None and idx == num_keys - 1:
                st['end_position'] = list(goal_pos)
                st['target_pos'] = list(goal_pos)

    def get_subtask_target_pos_rot(self, key: str):
        """
        Retrieve target position and rotation for the given subtask key.
        Returns (target_pos, target_rot).
        """
        key_upper = key.upper()
        subtask = self.sub_instruction_dict.get(key_upper)
        if not subtask:
            return None, None
        target_pos = (
            subtask.get('end_position')
            or subtask.get('target_pos')
            or subtask.get('target_position')
            or subtask.get('end_pos')
            or subtask.get('final_coord')
        )
        target_rot = (
            subtask.get('end_rotation')
            or subtask.get('target_rot')
            or subtask.get('final_rotation')
        )
        return target_pos, target_rot

    def get_subtask_pos_rot_by_key(self, key: str):
        """Get the final coordinate and rotation for the given subtask key. Returns (coord, rotation) tuple."""
        key_upper = key.upper()
        subtask = self.sub_instruction_dict.get(key_upper, None)
        if subtask and subtask.get('completed'):
            coord = subtask.get('final_coord', None)
            rotation = subtask.get('final_rotation', None)
            return coord, rotation
        elif subtask:
            coord = subtask.get('end_position') or subtask.get('target_pos')
            rotation = subtask.get('end_rotation') or subtask.get('target_rot')
            if coord is not None:
                return coord, rotation
        return None, None

    def update_waypoint_reason(self, reason, waypoint=None, pose=None):
        key = self.get_current_subtask_key()
        if not key or key not in self.sub_instruction_dict:
            return
        self.sub_instruction_dict[key]['waypoint_reason'] = str(reason or '')
        metadata = self.sub_instruction_dict[key].setdefault('metadata', {})
        if waypoint is not None:
            try:
                metadata['waypoint'] = [float(waypoint[0]), float(waypoint[1]), float(waypoint[2])]
            except Exception:
                pass
        if pose is not None:
            try:
                metadata['pose'] = [float(pose[0]), float(pose[1]), float(pose[2])]
            except Exception:
                pass

    def add_planner_memory(self, memories):
        key = self.get_current_subtask_key()
        if key and key in self.sub_instruction_dict:
            self.sub_instruction_dict[key]['planner_memory'] = list(memories or [])[:1]

    def update_verifier(self, result: dict):
        key = self.get_current_subtask_key()
        if key and key in self.sub_instruction_dict:
            self.sub_instruction_dict[key]['verifier'] = result or {}
            outcome = (result or {}).get('outcome')
            if isinstance(outcome, bool):
                self.sub_instruction_dict[key]['outcome'] = outcome
            if (result or {}).get('reason'):
                self.sub_instruction_dict[key]['outcome_reason'] = str(result['reason'])

import json
import re

def robust_json_parse(json_str: str):
    """
    Parse a JSON string that may contain comments or common formatting errors; returns dict or None.
    """
    if not json_str or not isinstance(json_str, str):
        return None

    # Strip <think>...</think> reasoning blocks if present
    json_str = re.sub(r'<think>.*?</think>', '', json_str, flags=re.DOTALL)
    if '</think>' in json_str:
        json_str = json_str.split('</think>', 1)[-1]

    json_str = re.sub(r'//.*', '', json_str)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)

    # Expand integer ranges inside JSON arrays (e.g., `"steps": [1-52]` -> `"steps": [1, 52]`)
    json_str = re.sub(r'(\[\s*\d+(?:\s*,\s*\d+)*)\s*-\s*(\d+)', r'\1, \2', json_str)
    # Repair missing opening quote on JSON key after comma/brace (e.g., `, target_evidence":` -> `, "target_evidence":`)
    json_str = re.sub(r'([\{\,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)"\s*:', r'\1"\2":', json_str)
    # Repair missing comma between string value and next quoted key on a new line
    json_str = re.sub(r'"\s*\n\s*"([a-zA-Z_][a-zA-Z0-9_]*)"\s*:', r'",\n"\1":', json_str)

    # Safely strip trailing commas before closing braces/brackets (e.g., `,}` or `,]`)
    comma_pattern = r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')|,\s*([}\]])'
    while True:
        cleaned_commas = re.sub(comma_pattern, lambda m: m.group(1) if m.group(1) is not None else m.group(2), json_str)
        if cleaned_commas == json_str:
            break
        json_str = cleaned_commas

    json_str = json_str.strip()

    json_str = re.sub(r'^```json', '', json_str)
    json_str = re.sub(r'^```', '', json_str)
    json_str = re.sub(r'```$', '', json_str)

    # Try finding the outer braces { ... }
    start = json_str.find('{')
    end = json_str.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = json_str[start:end+1].strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

    try:
        return json.loads(json_str)
    except Exception as e:
        pass

    try:
        from utils import robust_json_parse as utils_robust_json_parse
        return utils_robust_json_parse(json_str)
    except Exception:
        pass
    return None

class PathPlannerAgent(HabitatControl, CheckpointRecovery, RouteEvidenceGuard, TurnExecutionGuard, VerificationInputs, ABC):
    """
    An Agent whose decision logic is:
    1. Generate waypoints in current view plus explicit TURN_* actions.
    2. Let the VLM choose a target waypoint or action.
    3. Use A* to plan a path to that point.
    4. Decompose the path into actions and execute step-by-step, updating the map each step.
    """

    def __init__(self, sim_wrapper: SimWrapper, config: dict, instruction: str, init_position=None, init_rotation=None, mode='objnav'):
        self.sim_wrapper = sim_wrapper
        self.config = config
        self.navigation_cfg = config.get('navigation', {})
        self.forward_step = config.get('forward_step', 1.0)
        self.turn_angle_rad = np.deg2rad(config.get('turn_angle_deg', 90))
        self.instruction = instruction
        self.curr_obs = None
        self.last_action = PolarAction.null
        self.last_actions = {}
        self._recent_positions = []
        self.resolution = (config['camera']['height'], config['camera']['width'])
        self.focal_length = calculate_focal_length(config['camera']['fov'], self.resolution[1])
        self.image_edge_threshold = 0.04
        self.traveled_distance = 0.0
        self.prev_agent_position = None
        self.vlm_responses = []
        self.mode = mode
        self.error = False
        self.decision_none_fail_count = 0
        self._decision_fail_count = 0
        self.is_parse_fail = False

        nav_cfg = (config or {}).get('navigation', {}) if isinstance(config, dict) else {}
        disable_decomp = bool(nav_cfg.get('disable_subtask_decomposition', False))
        enable_decomp = (
            not disable_decomp
            and (
                bool(nav_cfg.get('enable_subtask_decomposition', False))
                or os.environ.get('HSGM_DECOMPOSE_INSTRUCTION') == '1'
            )
        )
        self._instruction_decomposition_enabled = enable_decomp
        if enable_decomp:
            sub_insts = self.decompose_instruction(instruction)
        else:
            sub_insts = [instruction.strip()]

        self.instruction_obj = Instruction(full_instruction=instruction, sub_instructions=sub_insts)

        if init_position is not None and init_rotation is not None:
            self.sim_wrapper.set_initial_state(init_position, init_rotation)

        camera_params = config['camera']
        cam_intrinsics = np.array([
            [camera_params['fx'], 0, camera_params['width'] / 2],
            [0, camera_params['fy'], camera_params['height'] / 2],
            [0, 0, 1]
        ])

        _map = config.get('map', {})
        self.mapper = Instruct_Mapper(camera_intrinsic=cam_intrinsics,
                                      grid_resolution=_map.get('grid_resolution', 0.05),
                                      floor_height=_map.get('floor_height', -1.2),
                                      ceiling_height=_map.get('ceiling_height', 0.6),
                                      pcd_resolution=_map.get('pcd_resolution', 0.025),
                                      resolution=(config['camera']['height'], config['camera']['width']))

        initial_state = self.sim_wrapper.sim.get_agent(0).get_state()
        self.mapper.reset(initial_state.position, initial_state.rotation)

        obs = sim_wrapper.step(PolarAction(r=0, theta=0))
        self.mapper.reset(obs['agent_state'].position, obs['agent_state'].rotation)
        raw_max_extra = nav_cfg.get('task_visual_max_extra_terms', 16)
        if raw_max_extra is None or isinstance(raw_max_extra, bool):
            max_extra_terms = 16
        else:
            try:
                max_extra_terms = int(raw_max_extra)
            except (TypeError, ValueError):
                max_extra_terms = 16
        max_extra_terms = max(0, min(16, max_extra_terms))
        task_visual_enabled = bool(nav_cfg.get('task_visual_enabled', False))
        cfg_vocab = globals().get('configure_task_vocabulary')
        if not callable(cfg_vocab):
            seg_mod = globals().get('segmentation')
            if seg_mod is not None and hasattr(seg_mod, 'instance_segmentation'):
                cfg_vocab = getattr(seg_mod.instance_segmentation, 'configure_task_vocabulary', None)
        if not callable(cfg_vocab):
            from segmentation.instance_segmentation import configure_task_vocabulary as cfg_vocab
        self._task_vocabulary_info = cfg_vocab(
            instruction,
            enabled=task_visual_enabled,
            max_extra_terms=max_extra_terms,
        )
        self.instances = self.update_map(obs)
        self.curr_obs = obs
        self.turn_cooldown = 0
        self.turn_count = 0
        self.current_plan = None
        self.navigation_cfg = self.config.get('navigation', {})
        import habitat_sim
        route_cfg = dict(self.navigation_cfg)
        route_cfg['camera_height_m'] = -float(self.mapper.initial_floor_height)
        self.waypoint_routes = WaypointRoutes(
            self.sim_wrapper.sim.pathfinder, self.mapper.initial_position,
            habitat_sim.ShortestPath, route_cfg)
        self._execution_ledger = []
        self._blocked_targets = []
        self._blocked_move_count = 0
        self._physical_progress_step = 0
        self._waypoint_execution = None
        self._stair_progress = None
        self._observation_turns_without_motion = 0
        self.active_negative_memory = {}
        self._decision_verifications = {}
        self.navigation_trace = []
        memory_cfg = self.config.get('memory', {})
        if memory_cfg.get('enabled', False):
            self.navigation_memory = get_shared_navigation_memory(
                storage_path=memory_cfg.get('storage_path'),
                user_id=memory_cfg.get('user_id', 'hsgm_navigation'),
                agent_id=memory_cfg.get('agent_id', 'hsgm'),
                backend=memory_cfg.get('backend'),
            )
        else:
            self.navigation_memory = None
        self._waypoint_history = []
        self._last_memory_subtask_key = None
        self._last_retrieved_memories = []
        self._pending_negative_events = []
        self._memory_finalized = False
        self._failed_waypoints = []
        self.last_vlm_response = "Executing initial action."
        self.target_world_position = [0, 0, 0]
        self.img_buffer = []
        self.min_move_distance = float(self.navigation_cfg.get('waypoint_min_distance_m', .25))
        self.max_move_distance = 3

        self.init_pos = self.mapper.current_position - [0, 0, 1.2]
        self.init_rot = self.mapper.current_rotation

        self.first_stop = False
        self.second_stop = False

        self.rolling_back = False
        self._branch_checkpoints = []
        self._rollback_counts = {}
        self._active_rollback_checkpoint = None
        self._last_rollback_info = None
        self._pending_recovery = None
        self._navigation_decision_id = 0
        self._terminal_probe_checks = {}
        self._executing_verifier_correction = False
        self._verifier_correction_steps = {}
        self._verifier_feedback = None
        self._grounding_vetoes = {}
        self._has_teleport_correction = False
        self.goal_position = None
        self.reference_path = None

        stair_words = ['stair', 'step']
        if any(word in instruction.lower() for word in stair_words):
            pass
            self.mapper.enable_stair_detection = True
        else:
            self.mapper.enable_stair_detection = False

    def _stop(self):
        """Execute stop action and update sub-goal state."""
        if not self.first_stop:
            pass
            self.first_stop = True
            return PolarAction.stop
        elif not self.second_stop:
            pass
            self.second_stop = True
            return PolarAction.stop
        else:
            pass
            return PolarAction.null()

    def update_map(self, obs: dict):
        """Update map using four-direction observations (batched multiview)."""

        views = []

        if 'left' in obs and 'agent_state' in obs['left']:
            s = obs['left']['agent_state']
            views.append({
                'rgb': obs['left']['color_sensor'],
                'depth': obs['left']['depth_sensor'],
                'position': np.array(s.position),
                'rotation': s.sensor_states['color_sensor'].rotation,
            })

        if 'right' in obs and 'agent_state' in obs['right']:
            s = obs['right']['agent_state']
            views.append({
                'rgb': obs['right']['color_sensor'],
                'depth': obs['right']['depth_sensor'],
                'position': np.array(s.position),
                'rotation': s.sensor_states['color_sensor'].rotation,
            })

        if 'back' in obs and 'agent_state' in obs['back']:
            s = obs['back']['agent_state']
            views.append({
                'rgb': obs['back']['color_sensor'],
                'depth': obs['back']['depth_sensor'],
                'position': np.array(s.position),
                'rotation': s.sensor_states['color_sensor'].rotation,
            })

        agent_state = obs['agent_state']
        # Primary (front) view — always last so primary_index defaults to it
        views.append({
            'rgb': obs['color_sensor'],
            'depth': obs['depth_sensor'],
            'position': np.array(agent_state.position),
            'rotation': agent_state.sensor_states['color_sensor'].rotation,
        })

        step = int(getattr(self, '_controller_step', -1))
        cfg = getattr(self, 'navigation_cfg', {})
        spatial_fast = cfg.get('spatial_fast_enabled', False)
        pose = self.mapper.translation_func(views[-1]['position']) - self.mapper.initial_position
        yaw = checkpoint_yaw(self.mapper.rotation_func(views[-1]['rotation']))
        instruction_obj = getattr(self, 'instruction_obj', None)
        key = instruction_obj.get_current_subtask_key() if instruction_obj else None
        in_transit = len(getattr(self, 'current_path', ())) > 1 or bool(getattr(self, 'action_sequence', ()))
        critical = bool(
            not spatial_fast or instruction_obj is None
            or (instruction_obj.is_last_subtask() and not in_transit)
            or key != getattr(self, '_semantic_subtask_key', None)
            or getattr(self, '_pending_final_stop_confirmation', None)
            or getattr(self, '_pending_endpoint_confirmation', None)
            or getattr(self, '_pending_endpoint_relocalization', None)
            or getattr(self, '_pending_recovery', None)
            or getattr(self.mapper, 'is_changing_floor', False)
            or step <= getattr(self, '_full_observation_until_step', -100))
        refresh = semantic_refresh_due(
            step, getattr(self, '_semantic_step', -100), pose, getattr(self, '_semantic_pose', None),
            yaw, getattr(self, '_semantic_yaw', yaw), critical=critical,
            max_interval=max(1, int(cfg.get('semantic_max_interval', 3))),
            displacement=float(cfg.get('semantic_refresh_displacement_m', .6)))
        started = time.monotonic()
        instances = self.mapper.update_multiview(views, do_instance_segmentation=refresh, observation_step=step)
        self._observation_revision = getattr(self, '_observation_revision', 0) + 1
        self._decision_image_cache = None
        self._bev_render_cache = None
        if refresh:
            self._semantic_step, self._semantic_pose, self._semantic_yaw = step, pose.copy(), yaw
            self._semantic_subtask_key = key
        if spatial_fast:
            if not hasattr(self, 'spatial_state'):
                self.spatial_state = SpatialState()
            self.spatial_state.update(self.mapper.current_position, step)
        if hasattr(self, 'navigation_trace') and isinstance(self.navigation_trace, list):
            self.navigation_trace.append({'role': 'runtime_timing', 'stage': 'mapping', 'step': step,
                                          'duration_s': round(time.monotonic()-started, 4),
                                          'semantic_refresh': refresh})
        return instances

    def _world_to_pixel_coords(self, waypoints_world, agent_position, agent_rotation, camera_intrinsic):
        """
        Convert world-coordinate waypoints to image pixel coordinates.
        :param waypoints_world:
        :return:
        """
        if waypoints_world.shape[0] == 0:
            return np.array([]).reshape(0, 2), np.array([], dtype=int), np.array([], dtype=float)

        relative_points = waypoints_world - agent_position
        if hasattr(agent_rotation, 'as_rotation_matrix'):
            rotation_matrix = agent_rotation.as_rotation_matrix()
        elif hasattr(agent_rotation, 'shape') and len(agent_rotation.shape) == 1 and agent_rotation.shape[0] == 4:
            rotation_matrix = R.from_quat(agent_rotation).as_matrix()
        else:
            rotation_matrix = agent_rotation
        rotation_matrix_inv = rotation_matrix.T
        local_points = np.dot(relative_points, rotation_matrix_inv.T)

        camera_points = np.copy(local_points)
        camera_points[:, 1] = -local_points[:, 1]
        camera_points[:, 2] = -local_points[:, 2]

        original_indices = np.arange(waypoints_world.shape[0])
        valid_mask = camera_points[:, 2] > 0.1
        if not np.any(valid_mask):
            return np.array([]).reshape(0, 2), np.array([], dtype=int), np.array([], dtype=float)

        valid_points = camera_points[valid_mask]
        valid_original_indices = original_indices[valid_mask]
        camera_depths = valid_points[:, 2].astype(float)

        fx, fy = camera_intrinsic[0, 0], camera_intrinsic[1, 1]
        cx, cy = camera_intrinsic[0, 2], camera_intrinsic[1, 2]
        pixel_x = (valid_points[:, 0] * fx / valid_points[:, 2]) + cx
        pixel_y = (valid_points[:, 1] * fy / valid_points[:, 2]) + cy
        pixel_coords = np.column_stack([pixel_x, pixel_y])

        return pixel_coords, valid_original_indices, camera_depths

    def _preprocessing_module(self, obs, use_far_waypoints: bool = False):
        """
        Preprocessing: generate waypoints in current view, project onto image, including turn-left/right actions.
        Also mark visited candidates (<1m from trajectory) in red using trajectory_position history.
        """
        import numpy as np
        import cv2
        from scipy.spatial import cKDTree

        # Collect 4-direction views (Front, Left, Right, Back)
        image = obs['color_sensor'].copy()
        height, width, _ = image.shape

        views_data = [
            {
                'name': 'front',
                'img': image,
                'depth': obs.get('depth_sensor', None),
                'rotation': self.mapper.current_rotation,
            }
        ]
        if 'left' in obs and 'color_sensor' in obs['left']:
            rot_left = self.mapper.rotation_func(obs['left']['agent_state'].sensor_states['color_sensor'].rotation) if ('agent_state' in obs['left']) else self.mapper.current_rotation
            views_data.append({
                'name': 'left',
                'img': obs['left']['color_sensor'].copy(),
                'depth': obs['left'].get('depth_sensor', None),
                'rotation': rot_left,
            })
        if 'right' in obs and 'color_sensor' in obs['right']:
            rot_right = self.mapper.rotation_func(obs['right']['agent_state'].sensor_states['color_sensor'].rotation) if ('agent_state' in obs['right']) else self.mapper.current_rotation
            views_data.append({
                'name': 'right',
                'img': obs['right']['color_sensor'].copy(),
                'depth': obs['right'].get('depth_sensor', None),
                'rotation': rot_right,
            })
        if 'back' in obs and 'color_sensor' in obs['back']:
            rot_back = self.mapper.rotation_func(obs['back']['agent_state'].sensor_states['color_sensor'].rotation) if ('agent_state' in obs['back']) else self.mapper.current_rotation
            views_data.append({
                'name': 'back',
                'img': obs['back']['color_sensor'].copy(),
                'depth': obs['back'].get('depth_sensor', None),
                'rotation': rot_back,
            })

        for v in views_data:
            d_img = v['depth']
            if d_img is not None:
                if d_img.ndim == 3 and d_img.shape[2] == 1:
                    d_img = d_img[:, :, 0]
                v['depth'] = d_img
                dh, dw = d_img.shape[:2]
                v['sx'] = dw / float(width)
                v['sy'] = dh / float(height)
            else:
                v['sx'] = 1.0
                v['sy'] = 1.0

        depth_img = views_data[0]['depth']
        depth_h = depth_w = None
        sx = views_data[0]['sx']
        sy = views_data[0]['sy']
        if depth_img is not None:
            depth_h, depth_w = depth_img.shape[:2]

        actions = {}
        action_idx = 1

        body_radius = float(self.sim_wrapper.agent_radius) + float(self.navigation_cfg.get('waypoint_margin_m', .03))
        waypoints_world_near, waypoints_relative_near = self.mapper.get_current_view_candidate_waypoints(
            waypoint_grid_resolution=float(self.navigation_cfg.get('waypoint_grid_m', .25)),
            merge_distance=0, cylinder_radius=body_radius,
            cylinder_height=float(self.sim_wrapper.agent_height), max_view_angle_deg=None)

        if use_far_waypoints:
            waypoints_world_far, waypoints_relative_far = self.mapper.get_current_view_candidate_waypoints(
                waypoint_grid_resolution=2.0, min_distance=0.3, max_distance=1.5, merge_distance=1.5,
                max_view_angle_deg=None
            )
            agent_pos = self.mapper.current_position
            if waypoints_world_far.shape[0] > 0:
                distances_far = np.linalg.norm(waypoints_world_far - agent_pos, axis=1)
                far_mask = (distances_far > self.max_move_distance) & (distances_far < 8.0)
                waypoints_world_far = waypoints_world_far[far_mask]
                waypoints_relative_far = waypoints_relative_far[far_mask]
            else:
                waypoints_world_far = np.empty((0, 3))
                waypoints_relative_far = np.empty((0, 3))
        else:
            waypoints_world_far = np.empty((0, 3))
            waypoints_relative_far = np.empty((0, 3))

        if waypoints_world_far.shape[0] > 0:
            waypoints_world = np.vstack([waypoints_world_near, waypoints_world_far])
            waypoints_relative = np.vstack([waypoints_relative_near, waypoints_relative_far])
        else:
            waypoints_world = waypoints_world_near
            waypoints_relative = waypoints_relative_near

        visited_radius = 0.8
        if waypoints_world.shape[0] > 0:
            traj = np.array(getattr(self.mapper, 'trajectory_position', []), dtype=float)[:-1] - np.array([0, 0, 1.2])
            if traj.size > 0:
                traj_xz = traj[:, [0, 1]]
                wpts_xz = waypoints_world[:, [0, 1]]
                tree = cKDTree(traj_xz)
                dists, _ = tree.query(wpts_xz, k=1)
                visited_mask_all = dists < visited_radius
            else:
                visited_mask_all = np.zeros((waypoints_world.shape[0],), dtype=bool)
        else:
            visited_mask_all = np.zeros((0,), dtype=bool)

        # Bound physical queries while retaining candidates in every direction.
        if len(waypoints_world):
            ground = self.mapper.current_position + [0, 0, self.mapper.initial_floor_height]
            distances = np.linalg.norm(waypoints_world - ground, axis=1)
            keep = (distances >= self.min_move_distance) & (distances <= self.max_move_distance)
            waypoints_world, waypoints_relative = waypoints_world[keep], waypoints_relative[keep]
            order = candidate_order(waypoints_world, ground,
                                    int(self.navigation_cfg.get('waypoint_candidate_limit', 96)))
            waypoints_world, waypoints_relative = waypoints_world[order], waypoints_relative[order]
            visited_mask_all = visited_mask_all[keep][order]

        skip_by_edge = 0
        skip_by_distance = 0
        skip_by_path = 0
        skip_by_occlusion = 0
        skip_by_stair = 0
        print(f"waypoints (360-degree): {waypoints_world.shape[0]}")

        # Physical route checks decide climb/descent feasibility. Do not ban
        # landing exits because an old clause elsewhere mentions stairs.
        agent_pos = self.mapper.current_position

        # Project waypoints to all 4 views and map each waypoint to its best view (most centered)
        edge_margin_x = width * self.image_edge_threshold
        edge_margin_y = height * self.image_edge_threshold
        cand_best_view = {}

        if waypoints_world.shape[0] > 0:
            for v in views_data:
                p_coords, o_indices, c_depths = self._world_to_pixel_coords(
                    waypoints_world,
                    self.mapper.current_position,
                    v['rotation'],
                    self.mapper.camera_intrinsic
                )
                if p_coords.shape[0] == 0:
                    continue
                for i in range(p_coords.shape[0]):
                    xp, yp = int(p_coords[i, 0]), int(p_coords[i, 1])
                    orig_idx = int(o_indices[i])
                    if not (edge_margin_x < xp < width - edge_margin_x and edge_margin_y < yp < height - edge_margin_y):
                        continue
                    d_center = (xp - width / 2.0) ** 2 + (yp - height / 2.0) ** 2
                    if orig_idx not in cand_best_view or d_center < cand_best_view[orig_idx]['d_center']:
                        cand_best_view[orig_idx] = {
                            'view': v,
                            'x_pixel': xp,
                            'y_pixel': yp,
                            'cam_depth': float(c_depths[i]),
                            'd_center': d_center
                        }

        # Keep the displayed action set readable without making the sampling
        # grid too coarse for doors. Physics queries have a separate limit.
        displayed_waypoints = []
        # Process and validate candidate waypoints in their respective views
        for orig_idx, cand_info in sorted(cand_best_view.items(), key=lambda kv: kv[0]):
            if len(displayed_waypoints) >= int(self.navigation_cfg.get('waypoint_display_limit', 24)):
                break
            target_view = cand_info['view']
            x_pixel, y_pixel = cand_info['x_pixel'], cand_info['y_pixel']
            cam_depth = cand_info['cam_depth']
            v_depth = target_view['depth']

            if v_depth is not None:
                dx = int(np.clip(round(x_pixel * target_view['sx']), 0, (v_depth.shape[1] - 1)))
                dy = int(np.clip(round(y_pixel * target_view['sy']), 0, (v_depth.shape[0] - 1)))
                observed = float(v_depth[dy, dx])
                if np.isfinite(observed) and observed > 0:
                    tol = max(0.12, 0.03 * observed)
                    if cam_depth > observed + tol:
                        skip_by_occlusion += 1
                        continue

            wp_world = waypoints_world[orig_idx]
            wp_relative = waypoints_relative[orig_idx] + [0, 1.2, 0]
            wp_world_for_planning = np.copy(wp_world)
            if any(np.linalg.norm(wp_world_for_planning-p) < .30 for p in displayed_waypoints):
                continue
            distance = float(np.linalg.norm(wp_world - (agent_pos + [0, 0, self.mapper.initial_floor_height])))
            if distance < self.min_move_distance or distance > self.max_move_distance:
                skip_by_distance += 1
                continue
            route = self._validated_waypoint_route(wp_world_for_planning)
            if route is None:
                skip_by_path += 1
                continue
            wp_world_for_planning = route['target']
            # Snapping must not leave the displayed marker at another point.
            projected, _, snapped_depths = self._world_to_pixel_coords(
                wp_world_for_planning.reshape(1, 3), self.mapper.current_position,
                target_view['rotation'], self.mapper.camera_intrinsic)
            if not len(projected):
                skip_by_occlusion += 1
                continue
            x_pixel, y_pixel = map(int, projected[0])
            if not (edge_margin_x < x_pixel < width-edge_margin_x and edge_margin_y < y_pixel < height-edge_margin_y):
                skip_by_edge += 1
                continue
            if v_depth is not None:
                dx = int(np.clip(round(x_pixel*target_view['sx']), 0, v_depth.shape[1]-1))
                dy = int(np.clip(round(y_pixel*target_view['sy']), 0, v_depth.shape[0]-1))
                observed = float(v_depth[dy, dx])
                if not np.isfinite(observed) or observed <= 0 or snapped_depths[0] > observed + max(.12, .03*observed):
                    skip_by_occlusion += 1
                    continue
            wp_relative = (wp_world_for_planning - self.mapper.current_position)[[0, 2, 1]] + [0, -self.mapper.initial_floor_height, 0]

            is_visited = bool(visited_mask_all[orig_idx])

            # Apply failed-branch constraints before drawing or naming the point.
            candidate = {'type': 'waypoint', 'target_point_world': wp_world_for_planning,
                         'target_point_relative': wp_relative, 'visited': is_visited,
                         'view': target_view['name']}
            if not self._constrain_branch_actions({'candidate': candidate}):
                continue

            min_radius, max_radius = 6, 35
            min_font, max_font = 0.4, 1.1
            norm_dist = np.clip(distance / 3, 0, 1)
            font_scale = max_font - (max_font - min_font) * norm_dist

            text = f"WP_{action_idx:02d}"
            (ref_text_width, ref_text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            text_diag = int(np.sqrt(ref_text_width ** 2 + ref_text_height ** 2))
            radius = max(int(text_diag / 2 + 2), min_radius)

            target_img = target_view['img']
            overlay = target_img.copy()

            circle_color = (255, 200, 100) if distance > self.max_move_distance else (255, 255, 255)
            if is_visited:
                circle_color = (255, 100, 100)

            cv2.circle(overlay, (x_pixel, y_pixel), radius, circle_color, -1)
            cv2.circle(overlay, (x_pixel, y_pixel), radius, (0, 0, 0), 2)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, target_img, 1 - alpha, 0, target_img)

            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            cv2.putText(target_img, text, (x_pixel - text_width // 2, y_pixel + text_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)

            actions[text] = {
                'type': 'waypoint',
                'target_point_world': wp_world_for_planning,
                'target_point_relative': wp_relative,
                'view': target_view['name'],
                'visited': is_visited,
                'validated_path': route['path'],
                'route_length_m': route['length_m'],
                'route_backend': route['backend']
            }
            displayed_waypoints.append(wp_world_for_planning)
            action_idx += 1

        print(
            f"skipped waypoints: edge {skip_by_edge},  distance {skip_by_distance},  no_path {skip_by_path},  occlusion {skip_by_occlusion},  stair {skip_by_stair}")

        # No unvalidated depth-ray fallback: absence of a certified route
        # permits an observation turn, never an invented forward waypoint.

        if self.first_stop:
            try:
                num_points = 50
                radius = 1.0
                thetas = np.linspace(0, 2 * np.pi, num_points)
                agent_pos = self.mapper.current_position
                floor_z = agent_pos[2] - 1.2

                circle_x = agent_pos[0] + radius * np.cos(thetas)
                circle_y = agent_pos[1] + radius * np.sin(thetas)
                circle_z = np.full_like(circle_x, floor_z)
                circle_world_points = np.vstack([circle_x, circle_y, circle_z]).T

                pixel_coords_circle, _, _ = self._world_to_pixel_coords(
                    circle_world_points,
                    self.mapper.current_position,
                    self.mapper.current_rotation,
                    self.mapper.camera_intrinsic
                )

                if pixel_coords_circle.shape[0] > 1:
                    pts = pixel_coords_circle.astype(np.int32).reshape((-1, 1, 2))
                    overlay = image.copy()
                    cv2.polylines(
                        overlay,
                        [pts],
                        isClosed=True,
                        color=(0, 255, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )
                    alpha = 0.6
                    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

                    scale_factor = image.shape[0] / 1080.0
                    top_idx = np.argmin(pixel_coords_circle[:, 1])
                    top_point = pixel_coords_circle[top_idx]
                    text = "1m"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.2 * scale_factor
                    thickness = max(2, int(2 * scale_factor))
                    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

                    tx = int(top_point[0] - tw / 2)
                    ty = int(top_point[1] - 2 * scale_factor - th)

                    tx = max(2, min(tx, image.shape[1] - tw - 2))
                    ty = max(th + 2, min(ty, image.shape[0] - 2))

                    cv2.putText(image, text, (tx, ty), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                    cv2.putText(image, text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            except Exception as e:
                pass

        # Both directions remain legal, including reversing an earlier turn.
        # Repetition is measured from executed feedback, not a symbol ban.
        action_keys = ['TURN_LEFT', 'TURN_RIGHT', 'TURN_BACK']
        positions = [(75, height // 2), (width - 75, height // 2), (width // 2, 45)]
        angles = [self.turn_angle_rad, -self.turn_angle_rad, np.pi]
        for key, pos, angle in zip(action_keys, positions, angles):
            overlay = image.copy()
            (tw, th), _ = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, .45, 1)
            cv2.circle(overlay, pos, max(30, tw//2+5), (255, 255, 255), -1)
            cv2.addWeighted(overlay, .6, image, .4, 0, image)
            cv2.putText(image, key, (pos[0]-tw//2, pos[1]+th//2), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1)
            actions[key] = {'type': 'turn', 'angle': angle}

        subtask_points = self.instruction_obj.get_subtasks_key_coord()
        try:
            if subtask_points:
                keys, pts = [], []
                for k, coord in subtask_points.items():
                    if coord is None or len(coord) < 3:
                        continue
                    pts.append([float(coord[0]), float(coord[1]), float(coord[2])])
                    keys.append(k)

                if len(pts) > 0:
                    pts_np = np.array(pts, dtype=float)
                    pixel_coords_st, valid_idx_st, _ = self._world_to_pixel_coords(
                        pts_np,
                        self.mapper.current_position,
                        self.mapper.current_rotation,
                        self.mapper.camera_intrinsic
                    )

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale_factor = image.shape[0] / 1080.0
                    for i, px in enumerate(pixel_coords_st):
                        key = keys[int(valid_idx_st[i])]
                        x, y = int(px[0]), int(px[1])
                        if x < 0 or x >= width or y < 0 or y >= height:
                            continue

                        half = max(6, int(8 * scale_factor))
                        tl = (max(0, x - half), max(0, y - half))
                        br = (min(width - 1, x + half), min(height - 1, y + half))
                        cv2.rectangle(image, tl, br, (255, 0, 0), -1, lineType=cv2.LINE_AA)
                        cv2.rectangle(image, tl, br, (0, 0, 0), 1, lineType=cv2.LINE_AA)

                        label = key
                        font_scale = max(0.5, 0.8 * scale_factor)
                        thickness = max(1, int(2 * scale_factor))
                        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                        tx = br[0] + int(6 * scale_factor)
                        ty = tl[1] - int(4 * scale_factor)

                        bx1 = max(0, tx - 3)
                        by1 = max(0, ty - th - 3)
                        bx2 = min(width - 1, tx + tw + 3)
                        by2 = min(height - 1, ty + 3)
                        if bx2 > bx1 and by2 > by1:
                            cv2.rectangle(image, (bx1, by1), (bx2, by2), (255, 255, 255), -1)
                            cv2.rectangle(image, (bx1, by1), (bx2, by2), (0, 0, 0), 1)

                        text_org = (min(max(0, tx), width - 1 - tw), min(max(th, ty), height - 1))
                        cv2.putText(image, label, text_org, font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        except Exception as e:
            pass

        self.labeled_views = {v['name']: v['img'] for v in views_data}
        return image, actions

    def _create_panoramic_360_image(self, labeled_views_dict: dict) -> np.ndarray:
        """Join labeled RGB panels without reprojecting or covering scene annotations."""
        self.panorama_360_image = None
        directions = ('left', 'front', 'right', 'back')
        titles = ('[LEFT -90°]', '[FRONT 0°]', '[RIGHT +90°]', '[BACK 180°]')
        if not isinstance(labeled_views_dict, dict):
            raise ValueError("Panorama views must be a dictionary of RGB/RGBA arrays")

        rgb_views = []
        for direction in directions:
            view = labeled_views_dict.get(direction)
            if (not isinstance(view, np.ndarray) or view.ndim != 3
                    or view.shape[0] == 0 or view.shape[1] == 0
                    or view.shape[2] not in (3, 4)
                    or view.dtype.kind not in 'uif' or not np.isfinite(view).all()):
                raise ValueError(f"Missing or invalid panorama view: {direction}")
            # Drop alpha only; preserve RGB order and never write into source arrays.
            rgb_views.append(np.clip(view[:, :, :3], 0, 255).astype(np.uint8))

        common_height = max(view.shape[0] for view in rgb_views)
        panels = []
        for view in rgb_views:
            panel = PIL.Image.fromarray(view)
            width = max(1, round(view.shape[1] * common_height / view.shape[0]))
            if panel.size != (width, common_height):
                # Nearest-neighbor scaling keeps discrete waypoint labels sharp.
                panel = panel.resize((width, common_height), PIL.Image.Resampling.NEAREST)
            panels.append(panel)

        separator_width = 2
        total_width = sum(panel.width for panel in panels) + separator_width * 3
        font_size = max(12, min(48, common_height // 24))
        try:
            font = ImageFont.truetype('DejaVuSans.ttf', font_size)
        except OSError:
            font = ImageFont.load_default()
        padding = max(4, font_size // 4)
        title_height = max(font.getbbox(title)[3] - font.getbbox(title)[1]
                           for title in titles)
        header_height = title_height + padding * 2
        panorama = PIL.Image.new('RGB', (total_width, common_height + header_height),
                                 (128, 128, 128))
        # The translucent band occupies new rows, leaving all original labels intact.
        header = PIL.Image.new('RGBA', (total_width, header_height), (0, 0, 0, 180))
        panorama.paste(header, (0, 0), header)
        draw = ImageDraw.Draw(panorama)
        x = 0
        for panel, title in zip(panels, titles):
            panorama.paste(panel, (x, header_height))
            # Clip each title to its own header so narrow panels cannot overlap.
            title_band = panorama.crop((x, 0, x + panel.width, header_height))
            ImageDraw.Draw(title_band).text(
                (panel.width / 2, header_height / 2), title, font=font,
                fill=(255, 255, 255), anchor='mm')
            panorama.paste(title_band, (x, 0))
            x += panel.width
            if x < total_width:
                draw.line((x, 0, x, panorama.height - 1), fill=(128, 128, 128),
                          width=separator_width)
                x += separator_width

        self.panorama_360_image = panorama
        return np.asarray(panorama, dtype=np.uint8).copy()


    def _get_agent_yaw(self):
        return checkpoint_yaw(self.mapper.current_rotation)

    def _rot_to_yaw(self, rotation):
        return checkpoint_yaw(rotation)

    def _normalize_angle(self, a):
        return float((a + np.pi) % (2 * np.pi) - np.pi)

    def _scene_scope(self):
        scene_path = str((getattr(self, 'config', {}) or {}).get('scene_path', ''))
        match = re.search(r'([A-Za-z0-9_-]+)\.basis\.glb', scene_path)
        return match.group(1) if match else scene_path

    def _capture_waypoint_checkpoint(self):
        return {
            'pose': np.asarray(self.mapper.current_position, dtype=float).tolist(),
            'rotation': np.asarray(self.mapper.current_rotation, dtype=float).tolist(),
            'yaw': float(self._get_agent_yaw()),
            'step': getattr(self, '_controller_step', 0),
            'trajectory_before': [np.asarray(p, dtype=float).tolist() for p in getattr(self.mapper, 'trajectory_position', [])],
        }

    def _get_multiview_depth_stats(self, obs=None) -> dict:
        if obs is None:
            obs = getattr(self, 'curr_obs', None) or {}
        stats = {}
        view_sources = [
            ('front', obs.get('depth_sensor', None)),
            ('left', (obs.get('left') or {}).get('depth_sensor', None)),
            ('right', (obs.get('right') or {}).get('depth_sensor', None)),
            ('back', (obs.get('back') or {}).get('depth_sensor', None)),
        ]
        for vname, depth_img in view_sources:
            if depth_img is None:
                continue
            try:
                arr = np.array(depth_img, copy=False)
                if arr.ndim == 3 and arr.shape[2] == 1:
                    arr = arr[:, :, 0]
                h, w = arr.shape[:2]
                center = arr[int(h * 0.35):int(h * 0.65), int(w * 0.35):int(w * 0.65)]
                valid_c = center[np.isfinite(center) & (center > 0.15) & (center < 10.0)]
                mid_band = arr[int(h * 0.25):int(h * 0.75), int(w * 0.15):int(w * 0.85)]
                valid_m = mid_band[np.isfinite(mid_band) & (mid_band > 0.15) & (mid_band < 10.0)]
                c_med = float(np.median(valid_c)) if len(valid_c) > 0 else None
                p85 = float(np.percentile(valid_m, 85)) if len(valid_m) > 0 else c_med
                p20 = float(np.percentile(valid_m, 20)) if len(valid_m) > 0 else c_med
                if c_med is not None:
                    stats[vname] = {
                        'center_m': round(c_med, 2),
                        'near_m': round(p20, 2) if p20 is not None else round(c_med, 2),
                        'far_m': round(p85, 2) if p85 is not None else round(c_med, 2),
                    }
            except Exception:
                continue
        return stats

    def _format_multiview_depth_and_waypoints(self, obs=None, actions=None) -> str:
        if obs is None:
            obs = getattr(self, 'curr_obs', None) or {}
        if actions is None:
            actions = getattr(self, 'last_actions', None) or {}
        depth_stats = self._get_multiview_depth_stats(obs) if hasattr(self, '_get_multiview_depth_stats') else {}
        lines = ["**4-View Surface Depth & Waypoint Distances**:",
                 "View-center depth describes a central image patch, NOT the named target. "
                 "The p85 value is a depth percentile, NOT maximum depth or reachable space. "
                 "Only a localized target region supplies endpoint distance."]
        try:
            curr_pos = np.asarray(self.mapper.current_position, dtype=float)
            traj = getattr(self.mapper, 'trajectory_position', [])
            if traj and len(traj) > 0:
                start_pos = self._episode_start_position() if hasattr(self, '_episode_start_position') else np.asarray(traj[0], dtype=float)
                disp_xy = float(np.linalg.norm(curr_pos[:2] - start_pos[:2]))
                disp_z = float(curr_pos[2] - start_pos[2])
                lines.append(f"- Net displacement from episode start: {disp_xy:.2f}m horizontal, {disp_z:+.2f}m vertical (steps taken: {len(getattr(self, '_execution_ledger', []))})")
        except Exception:
            pass
        if depth_stats:
            d_parts = []
            for vname, label in [('front', 'FRONT'), ('left', 'LEFT'), ('right', 'RIGHT'), ('back', 'BACK')]:
                st = depth_stats.get(vname)
                if st:
                    d_parts.append(f"{label}: center_patch={st['center_m']:.1f}m (p85={st['far_m']:.1f}m)")
            if d_parts:
                lines.append("- Camera depth by view: " + " | ".join(d_parts))
        wp_parts = []
        if actions and isinstance(actions, dict):
            curr_pos = np.asarray(self.mapper.current_position, dtype=float) if hasattr(self, 'mapper') else np.zeros(3)
            for k, v in actions.items():
                if not isinstance(v, dict) or v.get('type') != 'waypoint':
                    continue
                vdir = str(v.get('view', 'front')).upper()
                tw = v.get('target_point_world')
                rel = v.get('target_point_relative')
                try:
                    if tw is not None:
                        tw_arr = np.asarray(tw, dtype=float)
                        dist_xy = float(np.linalg.norm(tw_arr[:2] - curr_pos[:2]))
                        dz = float(tw_arr[2] - curr_pos[2])
                    elif rel is not None:
                        rel_arr = np.asarray(rel, dtype=float)
                        dist_xy = float(np.linalg.norm(rel_arr[:2]))
                        dz = float(rel_arr[2]) if len(rel_arr) > 2 else 0.0
                    else:
                        continue
                    vis_tag = ", visited" if v.get('visited') else ""
                    dz_tag = f", dz={dz:+.2f}m" if abs(dz) >= 0.15 else ""
                    wp_parts.append(f"{k} ({vdir}, dist={dist_xy:.2f}m{dz_tag}{vis_tag})")
                except Exception:
                    continue
        if wp_parts:
            lines.append("- Candidate waypoint distances: " + "; ".join(wp_parts))
        if hasattr(self, '_habitat_geometry_prompt'):
            lines.append(self._habitat_geometry_prompt(actions))
        return "\n".join(lines)

    def _get_forward_clearance_and_waypoints(self, obs=None, actions=None):
        if obs is None:
            obs = getattr(self, 'curr_obs', None) or {}
        if actions is None:
            actions = getattr(self, 'last_actions', None) or {}

        depth_img = obs.get('depth_sensor', None)
        forward_clearance = None
        if depth_img is not None:
            try:
                arr = np.array(depth_img, copy=False)
                if arr.ndim == 3 and arr.shape[2] == 1:
                    arr = arr[:, :, 0]
                h, w = arr.shape[:2]
                center = arr[int(h * 0.35):int(h * 0.65), int(w * 0.35):int(w * 0.65)]
                valid = center[np.isfinite(center) & (center > 0.2)]
                if len(valid) > 0:
                    forward_clearance = float(np.median(valid))
            except Exception:
                pass

        forward_candidates = []
        if actions and isinstance(actions, dict):
            for k, v in actions.items():
                if not isinstance(v, dict) or v.get('type') != 'waypoint':
                    continue
                if v.get('visited') is True:
                    continue
                rel = v.get('target_point_relative')
                if rel is None:
                    continue
                try:
                    rel_arr = np.asarray(rel, dtype=np.float32)
                    dist = float(np.linalg.norm(rel_arr))
                    if 0.5 <= dist <= 3.0:
                        forward_candidates.append({
                            'id': str(k),
                            'distance': round(dist, 2),
                            'target_point_world': v.get('target_point_world'),
                            'target_point_relative': rel_arr.tolist(),
                        })
                except Exception:
                    continue

        forward_candidates.sort(key=lambda x: x['distance'])
        return forward_clearance, forward_candidates

    def _verification_prompt(self, parsed_json: dict, memories: list, forward_clearance: float = None, forward_candidates: list = None, candidate_str: str = None, depth_and_wp_str: str = None) -> str:
        return self._route_verification_prompt(parsed_json, forward_clearance, candidate_str=candidate_str, depth_and_wp_str=depth_and_wp_str)

    def _verify_current_subtask(self, parsed_json: dict, front_image, view_images: list, map_image, obs: dict = None, actions: dict = None, max_turn: int = 1) -> dict:
        self._verification_obs = obs if obs is not None else getattr(self, 'curr_obs', None)
        key, subtask = self.instruction_obj.get_current_subtask()
        if max_turn <= 0:
            result = normalize_verification(None)
            result.update(raw='verifier skipped: max_turn reached', subtask_id=key, checked_pose=np.asarray(self.mapper.current_position).tolist())
            return result
        clearance, candidates = self._get_forward_clearance_and_waypoints(obs, actions)
        depth_and_wp_str = self._format_multiview_depth_and_waypoints(obs, actions) if hasattr(self, '_format_multiview_depth_and_waypoints') else ""
        direct_correction = self.navigation_cfg.get('verifier_direct_correction', True)
        self._verifier_correction_actions = legal_action_objects(actions or {}, completion=False)
        candidate_str = action_prompt(actions or {}, completion=False)

        prompt = self._verification_prompt(parsed_json, [], clearance, candidates, candidate_str=candidate_str, depth_and_wp_str=depth_and_wp_str)
        left = view_images[0] if view_images else front_image
        right = view_images[1] if len(view_images) > 1 else front_image
        back = view_images[2] if len(view_images) > 2 else None
        raw = ''
        result = normalize_verification(None)
        try:
            try:
                raw = self._get_verifier_response(
                    front_image, left, right, map_image, prompt,
                    back, getattr(self, 'img_buffer', None))
            except TypeError:
                raw = self._get_verifier_response(
                    front_image, left, right, map_image, prompt)
            parsed = robust_json_parse(raw)
            if isinstance(parsed, dict) and 'output' in parsed and isinstance(parsed['output'], dict):
                merged = {}
                if isinstance(parsed.get('reasoning'), dict):
                    merged.update(parsed['reasoning'])
                merged.update(parsed['output'])
                for k, v in parsed.items():
                    if k not in ('reasoning', 'output'):
                        merged[k] = v
                parsed = merged
            # Normalization fills defaults in-place; preserve the model's raw
            # field presence/values for the independent route evidence gate.
            result = normalize_verification(copy.deepcopy(parsed))
            if hasattr(self, '_evaluate_route_verification'):
                result = self._evaluate_route_verification(parsed, result)
            if hasattr(self, '_habitat_observe_unsettled_stair') and isinstance(parsed, dict):
                self._habitat_observe_unsettled_stair(parsed, result)
            result['evidence_repair_required'] = evidence_repair_required(parsed, result)
            result['endpoint_relocalization_required'] = endpoint_relocalization_required(parsed, result, self.navigation_cfg)
            result['landing_observation_pending'] = landing_observation_pending(parsed, result, self.navigation_cfg)
            result['candidate_claimed'] = bool(isinstance(parsed, dict) and parsed.get('outcome') is True
                and parsed.get('scene_consistent') is True and parsed.get('relation_satisfied') is True)
            ledger_rows = getattr(self, '_execution_ledger', []) or []
            unexecuted_turn = any(
                isinstance(c, dict)
                and 'landmark-bound explicit instruction turn' in str(c.get('evidence') or '')
                and not any(
                    isinstance(r, dict)
                    and not r.get('recovery')
                    and not r.get('rejected_by_rollback')
                    and (r.get('instruction_turn') or {}).get('condition_id') == c.get('id')
                    for r in ledger_rows
                )
                for c in (result.get('condition_checks') or [])
            )
            if not unexecuted_turn and ledger_rows and not result.get('condition_contract_conflict'):
                from route_progress import instruction_turn_witness, requires_straight, straight_witness, turn_witness
                hab_ctrl = bool(getattr(self, 'navigation_cfg', {}).get('habitat_control_enabled', False))
                after_s = -1
                for c in (result.get('condition_checks') or []):
                    if not isinstance(c, dict):
                        continue
                    if c.get('status') == 'satisfied':
                        c_steps = [s for s in (c.get('steps') or []) if type(s) is int]
                        if c_steps:
                            after_s = max(after_s, max(c_steps))
                        continue
                    rem_rows = [
                        r for r in ledger_rows
                        if isinstance(r, dict) and not r.get('recovery') and not r.get('rejected_by_rollback')
                        and type(r.get('step')) is int and r['step'] > after_s
                    ]
                    if c.get('requires_rotation', c.get('kind') == 'turn'):
                        if not turn_witness(c.get('text', ''), rem_rows, ledger_rows)[0] or (
                            hab_ctrl and not instruction_turn_witness(c, rem_rows, ledger_rows, key)
                        ):
                            unexecuted_turn = True
                            break
                    if requires_straight(c.get('text', '')) and any(
                        msg in str(c.get('evidence') or '')
                        for msg in ('Cited path bends or turns instead of continuing straight.',
                                    'No sufficient measured straight translation.')
                    ):
                        has_straight = any(
                            straight_witness(rem_rows[i:j], ledger_rows, after_step=after_s)[0]
                            for i in range(len(rem_rows))
                            for j in range(i + 1, len(rem_rows) + 1)
                        )
                        if not has_straight:
                            unexecuted_turn = True
                            break
            if unexecuted_turn:
                result['evidence_repair_required'] = False
                result['candidate_claimed'] = False
            if (result.get('stair_completion_ready') is False
                    and result.get('reason') == 'Measured stair direction or supported landing is not yet established.'
                    and not result.get('support_settling_geometry_observed')
                    and not result.get('landing_observation_pending')):
                sp = result.get('stair_progress') or {}
                s_dir = {'DOWN': -1, 'UP': 1}.get(sp.get('direction'), 0)
                if s_dir * float(sp.get('measured_height_change_m', 0.0) or 0.0) < 0.15:
                    result['candidate_claimed'] = False
            if direct_correction and isinstance(parsed, dict):
                result['correction_reason'] = str(parsed.get('correction_reason') or '')
                result['correction_action'] = parsed.get('correction_action')
        except Exception as exc:
            raw = f'verifier exception: {exc}'
            result = normalize_verification(None)
        result.update(raw=raw, subtask_id=key, checked_pose=np.asarray(self.mapper.current_position).tolist())
        trigger = 'terminal_probe' if parsed_json.get('_controller_probe') else 'completion_proposal'
        self._verification_history = remember_verification(
            getattr(self, '_verification_history', {}), key, result,
            trigger=trigger, step=getattr(self, '_last_step_index', 0),
            pose=result['checked_pose'])
        self.instruction_obj.update_verifier(result)
        def _is_unsupported_diag(val):
            if val is None or isinstance(val, (bool, str, int, float, np.bool_, np.integer, np.floating)):
                return False
            if isinstance(val, (list, tuple, dict)):
                return False
            return True

        def _clean_diag(val, depth=0):
            if depth > 6 or val is None:
                return None
            if isinstance(val, (bool, np.bool_)):
                return bool(val)
            if isinstance(val, (int, np.integer)) and not isinstance(val, bool):
                return int(val)
            if isinstance(val, (float, np.floating)):
                fv = float(val)
                return fv if bool(np.isfinite(fv)) else None
            if isinstance(val, str):
                return val
            if isinstance(val, (list, tuple)):
                out_list = []
                for item in val[:64]:
                    if _is_unsupported_diag(item):
                        continue
                    cleaned = _clean_diag(item, depth + 1)
                    if cleaned is not None or item is None:
                        out_list.append(cleaned)
                return out_list
            if isinstance(val, dict):
                out_dict = {}
                for k, v in val.items():
                    if not isinstance(k, str) or k.startswith('_') or _is_unsupported_diag(v):
                        continue
                    if k in ('image', 'crop_image', 'pil_image', 'mask', 'raw_mask', 'segmentation_mask'):
                        continue
                    out_dict[k] = _clean_diag(v, depth + 1)
                return out_dict
            return None

        _grounding_markers = (
            'grounded', 'required', 'status', 'evidence_role', 'matched', 'valid', 'target_id', 'entity_id'
        )

        def _clean_grounding_pkt(pkt):
            if not isinstance(pkt, dict):
                return None
            if not any(k in pkt for k in _grounding_markers):
                return None
            return _clean_diag(pkt)

        def _clean_spatial_rel(sr):
            if not isinstance(sr, dict):
                return None
            if not isinstance(sr.get('required'), (bool, np.bool_)) and not isinstance(sr.get('status'), str):
                return None
            allow = (
                'required', 'relation', 'status', 'reason', 'frame_kind',
                'frame_observation_id', 'target_candidate_id', 'landmark_candidate_ids', 'signed_separation_m'
            )
            out_sr = {}
            for k in allow:
                if k not in sr:
                    continue
                v = sr[k]
                if k == 'required':
                    out_sr[k] = bool(v) if isinstance(v, (bool, np.bool_)) else None
                elif k in ('relation', 'status', 'reason', 'frame_kind', 'target_candidate_id'):
                    out_sr[k] = v.strip() if isinstance(v, str) and v.strip() else None
                elif k == 'frame_observation_id':
                    out_sr[k] = (
                        int(v)
                        if isinstance(v, (int, np.integer)) and not isinstance(v, bool) and int(v) >= 0
                        else None
                    )
                elif k == 'landmark_candidate_ids':
                    out_sr[k] = (
                        [x.strip() for x in v[:16] if isinstance(x, str) and x.strip()]
                        if isinstance(v, (list, tuple)) else []
                    )
                elif k == 'signed_separation_m':
                    out_sr[k] = (
                        float(v)
                        if isinstance(v, (int, float, np.integer, np.floating))
                        and not isinstance(v, bool)
                        and bool(np.isfinite(float(v)))
                        else None
                    )
            return out_sr

        _crop_allow = (
            'candidate_id', 'target_id', 'object_id', 'entity_id', 'label', 'class_name',
            'selected_detector_class_name', 'expected_object_class', 'view', 'view_name',
            'bbox', 'bbox_xyxy', 'crop_box', 'bbox_space', 'depth_m', 'distance_m',
            'confidence', 'score', 'observation_id', 'evidence_role', 'role', 'source'
        )

        def _clean_verifier_crops(raw_crops):
            if not isinstance(raw_crops, (list, tuple)):
                return []
            cleaned_crops = []
            for item in raw_crops[:4]:
                if not isinstance(item, dict):
                    continue
                c_item = {}
                for k in _crop_allow:
                    if k in item and not _is_unsupported_diag(item[k]):
                        c_item[k] = _clean_diag(item[k], 1)
                if c_item:
                    cleaned_crops.append(c_item)
            return cleaned_crops

        def _clean_bev_summary(bev_dict):
            if not isinstance(bev_dict, dict):
                return None
            bev_out = {}
            if 'image_size' in bev_dict:
                bev_out['image_size'] = _clean_diag(bev_dict.get('image_size'), 1)
            if isinstance(bev_dict.get('axes'), dict):
                ax = bev_dict['axes']
                bev_out['axes'] = {
                    k: _clean_diag(ax[k], 2)
                    for k in ('x_positive', 'y_positive', 'pixel_y_positive', 'yaw_convention')
                    if k in ax
                }
            for bkey in ('current_pose_valid', 'current_yaw_valid'):
                if bkey in bev_dict and isinstance(bev_dict.get(bkey), (bool, np.bool_)):
                    bev_out[bkey] = bool(bev_dict[bkey])
            for skey in ('current_yaw', 'current_foot_pose'):
                if skey in bev_dict and not _is_unsupported_diag(bev_dict.get(skey)):
                    bev_out[skey] = _clean_diag(bev_dict[skey], 1)
            if isinstance(bev_dict.get('counts'), dict):
                cnt = bev_dict['counts']
                bev_out['counts'] = {
                    k: _clean_diag(cnt[k], 2)
                    for k in (
                        'total_ledger_rows', 'retained_rows', 'omitted_history_rows', 'invalid_rows',
                        'rendered_observed_points', 'omitted_observed_points', 'invalid_observed_points',
                        'clipped_points_count', 'omitted_map_labels', 'move_arrow_count'
                    )
                    if k in cnt and not isinstance(cnt[k], (list, tuple, dict)) and not _is_unsupported_diag(cnt[k])
                }
            if isinstance(bev_dict.get('coordinate_clipping'), dict):
                clp = bev_dict['coordinate_clipping']
                bev_out['coordinate_clipping'] = {
                    k: _clean_diag(clp[k], 2)
                    for k in ('active', 'clipped_row_count', 'clipped_point_count', 'outlier_radius_m')
                    if k in clp and not _is_unsupported_diag(clp[k])
                }
            if isinstance(bev_dict.get('z_profile'), dict):
                zp = bev_dict['z_profile']
                bev_out['z_profile'] = {
                    k: _clean_diag(zp[k], 2)
                    for k in ('unit', 'min_z', 'max_z', 'current_z')
                    if k in zp and not _is_unsupported_diag(zp[k])
                }
            if isinstance(bev_dict.get('transform'), dict):
                tr = bev_dict['transform']
                bev_out['transform'] = {
                    k: _clean_diag(tr[k], 2)
                    for k in ('scale_px_per_m', 'x_min', 'x_max', 'y_min', 'y_max')
                    if k in tr and not _is_unsupported_diag(tr[k])
                }
            if 'legend' in bev_dict:
                leg = bev_dict.get('legend')
                if isinstance(leg, dict):
                    bev_out['legend'] = sorted(str(k) for k in leg.keys() if isinstance(k, str))
                elif isinstance(leg, (list, tuple)):
                    bev_out['legend'] = [str(k) for k in leg[:32] if isinstance(k, str)]
            return bev_out

        def _clean_verif_input_meta(raw_meta):
            if not isinstance(raw_meta, dict):
                return None
            meta_out = {}
            for bkey in ('evidence_enabled', 'bev_enabled', 'fixed_bev_enabled'):
                if bkey in raw_meta and isinstance(raw_meta.get(bkey), (bool, np.bool_)):
                    meta_out[bkey] = bool(raw_meta[bkey])
            if isinstance(raw_meta.get('selected_frames'), (list, tuple)):
                meta_out['selected_frames'] = _clean_diag(raw_meta['selected_frames'][:8], 1)
            if isinstance(raw_meta.get('event_frames'), (list, tuple)):
                meta_out['event_frames'] = _clean_diag(raw_meta['event_frames'][:8], 1)
            if isinstance(raw_meta.get('bev'), dict):
                meta_out['bev'] = _clean_bev_summary(raw_meta['bev'])
            elif raw_meta.get('bev') is None and 'bev' in raw_meta:
                meta_out['bev'] = None
            if isinstance(raw_meta.get('errors'), (list, tuple)):
                meta_out['errors'] = [str(x) for x in raw_meta['errors'][:16] if isinstance(x, str)]
            return meta_out

        clean_eg = _clean_grounding_pkt(result.get('endpoint_grounding'))
        clean_erg = _clean_grounding_pkt(result.get('endpoint_reference_grounding'))
        clean_slg = _clean_grounding_pkt(result.get('spatial_landmark_grounding'))
        clean_sr = _clean_spatial_rel(result.get('spatial_relation'))
        eg_valid = isinstance(clean_eg, dict) and (clean_eg.get('grounded') is True or clean_eg.get('valid') is True)
        if isinstance(clean_eg, dict) and (
            eg_valid
            or clean_eg.get('candidate_id') is not None
            or clean_eg.get('bbox_space') is not None
        ):
            raw_tcid = (
                clean_eg.get('candidate_id')
                if 'candidate_id' in clean_eg
                else (result.get('target_candidate_id') if eg_valid else None)
            )
            raw_tbspace = (
                clean_eg.get('bbox_space')
                if 'bbox_space' in clean_eg
                else (result.get('target_bbox_space') if eg_valid else None)
            )
            trace_target_cid = raw_tcid.strip() if isinstance(raw_tcid, str) and raw_tcid.strip() else None
            trace_target_bspace = raw_tbspace.strip() if isinstance(raw_tbspace, str) and raw_tbspace.strip() else None
        else:
            trace_target_cid = None
            trace_target_bspace = None

        self.navigation_trace.append({
            'role': 'verifier', 'event': 'verification_result', 'subtask_id': key,
            'step': getattr(self, '_last_step_index', 0),
            'pose': np.asarray(self.mapper.current_position).tolist(),
            'trigger': trigger,
            'available': result.get('available'), 'outcome': result.get('outcome'),
            'completion_evidence_reliable': result.get('completion_evidence_reliable', False),
            'evidence_repair_required': result.get('evidence_repair_required', False),
            'current_observation_id': result.get('current_observation_id'),
            'initial_arrival_observation_eligible': result.get('initial_arrival_observation_eligible', False),
            'landing_observation_pending': result.get('landing_observation_pending', False),
            'landing_support_current': result.get('landing_support_current'),
            'stair_completion_ready': result.get('stair_completion_ready'),
            'stair_progress': copy.deepcopy(result.get('stair_progress')),
            'failure_type': result.get('failure_type'), 'reason': result.get('reason'),
            'condition_checks': copy.deepcopy(result.get('condition_checks', [])),
            'target_identity': result.get('target_identity'),
            'target_proximity': result.get('target_proximity'),
            'target_entity_id': result.get('target_entity_id'),
            'selected_detector_class_name': result.get('selected_detector_class_name'),
            'target_candidate_id': trace_target_cid,
            'target_bbox_space': trace_target_bspace,
            'endpoint_grounding': clean_eg,
            'endpoint_reference_grounding': clean_erg,
            'spatial_landmark_grounding': clean_slg,
            'spatial_relation': clean_sr,
            'visual_candidate_crops': _clean_verifier_crops(
                getattr(self, '_last_visual_candidate_crops_for_verifier', None)
            ),
            'verification_input_metadata': _clean_verif_input_meta(
                getattr(self, '_last_verification_input_metadata', None)
            ),
            'endpoint_confirmation_required': result.get('endpoint_confirmation_required', False),
            'endpoint_relocalization_required': result.get('endpoint_relocalization_required', False),
            'route_contradiction': copy.deepcopy(result.get('route_contradiction')),
            'route_contradiction_gate': copy.deepcopy(result.get('route_contradiction_gate')),
            'route_progress': copy.deepcopy(result.get('route_progress')),
            'correction_action': result.get('correction_action'),
        })
        return result

    def _verification_accepts_stop(self, verification, require_remaining_route=False):
        if not verification or not verification.get('available'):
            return False
        is_last = self.instruction_obj.is_last_subtask() if hasattr(self.instruction_obj, 'is_last_subtask') else True
        handoff_ok = True if is_last else (verification.get('handoff_consistent') is True)
        return bool(
            verification.get('outcome') is True
            and not verification.get('contract_conflict')
            and verification.get('relation_satisfied') is True
            and verification.get('scene_consistent') is True
            and verification.get('completion_evidence_reliable') is True
            and (not require_remaining_route or verification.get('scope') == 'remaining_route')
            and handoff_ok
        )

    def bind_episode_targets(self, goal_position=None, reference_path=None):
        """Bind episode ground truth goal and reference path to agent and subtasks."""
        if goal_position is not None:
            self.goal_position = np.asarray(goal_position, dtype=np.float32)
        if reference_path is not None:
            self.reference_path = reference_path
        if hasattr(self, 'instruction_obj') and self.instruction_obj:
            self.instruction_obj.bind_episode_targets(goal_position, reference_path)

    def _get_subtask_target_pose(self, key: str = None, subtask: dict = None):
        """
        Locate target position and rotation for the given subtask.
        Returns (target_pos, target_rot, is_mapper_coord).
        """
        if not key and hasattr(self, 'instruction_obj') and self.instruction_obj:
            key = self.instruction_obj.get_current_subtask_key()
        if not key:
            return None, None, False
        if subtask is None and hasattr(self, 'instruction_obj') and self.instruction_obj:
            subtask = self.instruction_obj.get_subtask_by_key(key)

        target_pos = None
        target_rot = None
        is_mapper_coord = False

        if subtask:
            target_pos = (
                subtask.get('end_position')
                or subtask.get('target_pos')
                or subtask.get('target_position')
                or subtask.get('end_pos')
            )
            target_rot = (
                subtask.get('end_rotation')
                or subtask.get('target_rot')
                or subtask.get('final_rotation')
            )

        if target_pos is None and hasattr(self.instruction_obj, 'get_subtask_target_pos_rot'):
            target_pos, target_rot = self.instruction_obj.get_subtask_target_pos_rot(key)

        if target_pos is None and hasattr(self.instruction_obj, 'get_subtask_pos_rot_by_key'):
            target_pos, target_rot = self.instruction_obj.get_subtask_pos_rot_by_key(key)

        # If last subtask and target_pos not yet found, check episode goal
        is_last = self.instruction_obj.is_last_subtask() if hasattr(self.instruction_obj, 'is_last_subtask') else False
        if target_pos is None and is_last:
            goal = getattr(self, 'goal_position', None)
            if goal is None and hasattr(self, 'config'):
                goal = self.config.get('goal_position')
            if goal is None and hasattr(self.sim_wrapper, 'goal_position'):
                goal = getattr(self.sim_wrapper, 'goal_position')
            if goal is not None:
                target_pos = np.asarray(goal, dtype=np.float32)

        # If reference path is available, partition along path
        if target_pos is None and getattr(self, 'reference_path', None):
            ref = list(self.reference_path)
            if len(ref) > 0:
                try:
                    num_k = len(self.instruction_obj.sub_instruction_keys)
                    idx = self.instruction_obj.sub_instruction_keys.index(key.upper())
                    pt_idx = int(round((idx + 1) / num_k * (len(ref) - 1)))
                    target_pos = np.asarray(ref[pt_idx], dtype=np.float32)
                except Exception:
                    pass

        if target_pos is not None:
            target_pos = np.asarray(target_pos, dtype=np.float32)

        return target_pos, target_rot, is_mapper_coord

    def _teleport_agent_to_pose(self, target_pos, target_rot=None, is_mapper_coord=False):
        """
        Directly place the agent in simulator at target_pos (and target_rot),
        and synchronize mapper current_position and current_rotation.
        """
        target_pos = np.asarray(target_pos, dtype=np.float32)
        if is_mapper_coord:
            hab_pos = np.array([
                target_pos[0] + self.mapper.initial_position[0],
                target_pos[2] + self.mapper.initial_position[1] + 1.2,
                target_pos[1] + self.mapper.initial_position[2]
            ], dtype=np.float32)
        else:
            hab_pos = target_pos

        # Write to simWrapper
        if hasattr(self.sim_wrapper, 'set_agent_state'):
            self.sim_wrapper.set_agent_state(hab_pos, target_rot)
        elif hasattr(self.sim_wrapper, 'teleport'):
            self.sim_wrapper.teleport(hab_pos, target_rot)
        elif hasattr(self.sim_wrapper, 'set_initial_state'):
            self.sim_wrapper.set_initial_state(hab_pos, target_rot)
        else:
            agent = self.sim_wrapper.sim.get_agent(0)
            state = agent.get_state()
            state.position = hab_pos
            if target_rot is not None:
                state.rotation = target_rot
            agent.set_state(state)

        # Synchronize mapper: zero-movement observation at new placement
        obs = self.sim_wrapper.step(PolarAction.null)
        instances = self.update_map(obs)
        self.curr_obs = obs
        self.instances = instances
        self.current_path = []
        self.action_sequence = []
        self._need_restore_heading = False
        self.target_world_position = np.asarray(target_pos, dtype=np.float32)
        self.prev_agent_position = np.asarray(obs['agent_state'].position, dtype=np.float32)
        return obs

    def _boundary_stop_decision(self, proposal, front, views, map_image, obs, actions):
        """Only current, positive execution evidence authorizes completion.

        Automatic probes have their own bounded allowance. A negative probe
        leaves a valid route intact; it does not force a spurious correction.
        """
        if self.navigation_cfg.get('habitat_control_enabled', False):
            return self._habitat_boundary_stop_decision(proposal, front, views, map_image, obs, actions)
        key, _ = self.instruction_obj.get_current_subtask()
        budget = getattr(self, 'runtime_budget', None)
        if budget is not None:
            budget.check()
        is_probe = bool(proposal.get('_controller_probe'))
        states_name = '_probe_checks' if is_probe else '_boundary_checks'
        states = getattr(self, states_name, None)
        if states is None:
            states = {}
            setattr(self, states_name, states)
        state = states.setdefault(key, {'calls': 0})
        pose = np.concatenate((np.asarray(self.mapper.current_position).ravel(),
                               np.asarray(self.mapper.current_rotation).ravel()))
        revision = (getattr(self, '_observation_revision', 0), len(getattr(self, '_execution_ledger', [])))
        unchanged = ('pose' in state and np.allclose(pose, state['pose'], atol=1e-5, rtol=0)
                     and state.get('revision') == revision)
        limit = max(1, int(self.navigation_cfg.get(
            'arrival_probe_max_calls' if is_probe else 'max_verifier_checks_per_subtask',
            4)))
        if not unchanged:
            if state['calls'] >= limit:
                if getattr(self, '_pending_endpoint_confirmation', None):
                    self._pending_endpoint_confirmation = None
                    self._pending_final_stop_confirmation = None
                    raise NavigationAborted('endpoint_confirmation_verifier_budget_exhausted')
                if getattr(self, '_pending_endpoint_relocalization', None):
                    self._pending_endpoint_relocalization = None
                    raise NavigationAborted('endpoint_relocalization_verifier_budget_exhausted')
                if is_probe:
                    return {'type': 'continue_path'}
                raise NavigationAborted('verifier_check_budget_exhausted')
            state['calls'] += 1
            logging.info(f"[Verifier] Subtask '{key}' verification call {state['calls']}/{limit} "
                         f"(trigger={'terminal_probe' if is_probe else 'completion_proposal'}) "
                         f"at pose {self.mapper.current_position.round(2)}.")
            state['result'] = self._verify_current_subtask(
                proposal, front, views, map_image, obs=obs, actions=actions, max_turn=1)
            state['pose'], state['revision'] = pose.copy(), revision
        verification = state.get('result') or {}
        pending_arrival_look = bool((getattr(self, '_pending_endpoint_relocalization', None) or {}).get('observation_only'))
        pending_landing_look = bool((getattr(self, '_pending_endpoint_relocalization', None) or {}).get('landing_observation'))
        # Repair invalid references against the same images and execution ledger.
        # This is one additional charged check, not a physical correction. Mark
        # the revision before dispatch so cached proposals cannot replay it.
        if (verification.get('evidence_repair_required')
                and state.get('evidence_repair_revision') != revision):
            state['evidence_repair_revision'] = revision
            if state['calls'] < limit:
                if budget is not None:
                    budget.check()
                state['calls'] += 1
                self.navigation_trace.append({
                    'role': 'controller', 'event': 'verification_evidence_repair',
                    'subtask_id': key, 'step': getattr(self, '_last_step_index', 0),
                    'trigger': 'terminal_probe' if is_probe else 'completion_proposal',
                    'call': state['calls'], 'pose': np.asarray(self.mapper.current_position).tolist()})
                logging.info(f"[Verifier] Subtask '{key}' same-pose evidence repair "
                             f"{state['calls']}/{limit}.")
                first_verification = verification
                state['result'] = self._verify_current_subtask(
                    proposal, front, views, map_image, obs=obs, actions=actions, max_turn=1)
                verification = state['result']
                if initial_arrival_observation_retry_required(
                        first_verification, verification, getattr(self.mapper, 'observation_id', None)):
                    # A changed story about unchanged images is uncertainty,
                    # not permission to walk off the candidate or to STOP.
                    verification = {**verification, 'endpoint_relocalization_required': True,
                                    'initial_arrival_observation_retry': True}
                    state['result'] = verification
        if self._verification_accepts_stop(verification):
            self._verifier_feedback = None
            self._pending_endpoint_relocalization = None
            self._pending_endpoint_confirmation = None
            self._pending_final_stop_confirmation = None
            return {'type': 'stop', 'verified': True}
        if verification.get('endpoint_relocalization_required') or verification.get('landing_observation_pending'):
            self._pending_endpoint_confirmation = None
            self._pending_final_stop_confirmation = None
            attempts = getattr(self, '_endpoint_relocalization_turns', {})
            count = attempts.get(key, 0)
            maximum = min(2, max(0, int(self.navigation_cfg.get('endpoint_relocalization_max_turns', 2))))
            landing_look = verification.get('landing_observation_pending') is True
            observation_only = bool(not landing_look and (verification.get('initial_arrival_observation_retry')
                                    or (any(c.get('observation_checked') for c in verification.get('condition_checks') or [])
                                        and not (verification.get('endpoint_grounding') or {}).get('valid'))))
            if observation_only:
                maximum = min(maximum, 1)
            # Cancel the old route before returning a physical observation. The
            # next controller tick must verify this candidate before any move.
            self.current_path, self.action_sequence = [], []
            self._waypoint_execution = None
            if count < maximum:
                attempts[key] = count + 1
                self._endpoint_relocalization_turns = attempts
                self._pending_endpoint_relocalization = {'key': key, 'is_probe': is_probe,
                                                        'observation_only': observation_only,
                                                        'landing_observation': landing_look}
                self.navigation_trace.append({
                    'role': 'controller', 'event': 'endpoint_relocalization_turn',
                    'subtask_id': key, 'step': getattr(self, '_last_step_index', 0),
                    'attempt': count + 1, 'reason': verification.get('reason'),
                    'source': ('landing_support_confirmation' if landing_look else
                               'initial_arrival_observation_uncertainty' if observation_only else 'endpoint_region_relocalization'),
                    'stair_progress': copy.deepcopy(verification.get('stair_progress')) if landing_look else None,
                    'trigger': 'terminal_probe' if is_probe else 'completion_proposal'})
                return {'type': 'turn', 'angle': math.radians(12 if count == 0 else -24),
                        'is_verifier_correction': True, 'endpoint_relocalization': True}
            self._pending_endpoint_relocalization = None
            verification = {**verification, 'endpoint_relocalization_exhausted': True}
            return self._direct_verifier_correction(verification, actions, repeated_without_motion=unchanged)
        self._pending_endpoint_relocalization = None
        if (pending_arrival_look or pending_landing_look) and verification.get('available') is False:
            raise NavigationAborted('verification_unavailable')
        if ((pending_arrival_look or pending_landing_look) and not verification.get('endpoint_confirmation_required')
                and not verification.get('rollback_required') and verification.get('failure_type') != 'wrong_path'):
            verification = {**verification, 'endpoint_relocalization_exhausted': True}
            return self._direct_verifier_correction(verification, actions, repeated_without_motion=unchanged)
        if verification.get('endpoint_confirmation_required'):
            confirmations = getattr(self, '_endpoint_confirmation_turns', {})
            positions = getattr(self, '_endpoint_confirmation_positions', {})
            count = confirmations.get(key, 0)
            position = np.asarray(self.mapper.current_position, dtype=float)
            previous_position = positions.get(key)
            # A real approach to another candidate position gets a fresh pair
            # of looks. Rotation alone cannot replenish this allowance; all
            # calls still consume the original per-subtask verifier budget.
            if (previous_position is not None and
                    np.linalg.norm(position - np.asarray(previous_position)) >= .35):
                count = 0
            if count == 0:
                positions[key] = position.tolist()
            self._endpoint_confirmation_positions = positions
            maximum = min(2, max(0, int(self.navigation_cfg.get('endpoint_confirmation_max_turns', 2))))
            self.current_path, self.action_sequence = [], []
            self._waypoint_execution = None
            if count < maximum:
                confirmations[key] = count + 1
                self._endpoint_confirmation_turns = confirmations
                # Separate controller scheduling from the grounding module's
                # semantic-refresh marker (_pending_final_stop_confirmation).
                self._pending_endpoint_confirmation = {'key': key, 'is_probe': is_probe}
                self.navigation_trace.append({'role': 'controller', 'event': 'endpoint_confirmation_turn',
                                              'subtask_id': key, 'attempt': count + 1,
                                              'step': getattr(self, '_last_step_index', 0),
                                              'trigger': 'terminal_probe' if is_probe else 'completion_proposal'})
                return {'type': 'turn', 'angle': math.radians(12 if count == 0 else -24),
                        'is_verifier_correction': True, 'endpoint_confirmation': True}
            self._pending_endpoint_confirmation = None
            self._pending_final_stop_confirmation = None
            verification = {**verification, 'endpoint_confirmation_exhausted': True}
            return self._direct_verifier_correction(verification, actions, repeated_without_motion=unchanged)
        had_pending_confirmation = bool(getattr(self, '_pending_endpoint_confirmation', None))
        self._pending_endpoint_confirmation = None
        self._pending_final_stop_confirmation = None
        if had_pending_confirmation and verification.get('available') is False:
            raise NavigationAborted('verification_unavailable')
        if (had_pending_confirmation and verification.get('available') is not False
                and verification.get('failure_type') != 'wrong_path' and not verification.get('rollback_required')):
            # The new view withdrew the endpoint claim. Do not resurrect the
            # paused route or turn again without a new planner decision.
            verification = {**verification, 'endpoint_confirmation_exhausted': True}
            return self._direct_verifier_correction(verification, actions, repeated_without_motion=unchanged)
        if is_probe and verification.get('failure_type') != 'wrong_path' and not verification.get('rollback_required'):
            return {'type': 'continue_path'}
        if verification.get('available') is False:
            raise NavigationAborted('verification_unavailable')
        return self._direct_verifier_correction(verification, actions, repeated_without_motion=unchanged)

    def _build_verifier_feedback(self, key, verification, repeated_without_motion=False):
        """Compact, planner-facing account of why a completion proposal was rejected."""
        unmet = []
        for chk in (verification.get('condition_checks') or []):
            if isinstance(chk, dict) and chk.get('status') != 'satisfied':
                unmet.append({'id': chk.get('id'), 'clause': str(chk.get('text') or '')[:120],
                              'status': chk.get('status'),
                              'why': str(chk.get('evidence') or '')[:180]})
        hint = verification.get('correction_action')
        return {
            'subtask': key,
            'pose': np.asarray(self.mapper.current_position).round(2).tolist(),
            'yaw': float(checkpoint_yaw(self.mapper.current_rotation)),
            'reason': str(verification.get('reason') or '')[:400],
            'evidence_repair_required': verification.get('evidence_repair_required') is True,
            'endpoint_relocalization_exhausted': verification.get('endpoint_relocalization_exhausted') is True,
            'endpoint_confirmation_exhausted': verification.get('endpoint_confirmation_exhausted') is True,
            'endpoint_grounding': copy.deepcopy(verification.get('endpoint_grounding')),
            'endpoint_reference_grounding': copy.deepcopy(verification.get('endpoint_reference_grounding')),
            'unmet_conditions': unmet[:4],
            'correction_reason': str(verification.get('correction_reason') or '')[:300],
            'suggested_action': hint if isinstance(hint, dict) else None,
            'repeated_without_motion': repeated_without_motion,
        }

    def _direct_verifier_correction(self, verification, actions, repeated_without_motion=False):
        """Rejected completion: trigger wrong-path rollback if off-route, or execute one bounded correction."""
        key, _ = self.instruction_obj.get_current_subtask()
        if not hasattr(self, '_verifier_corrections'):
            self._verifier_corrections = {}
        count = self._verifier_corrections.get(key, 0)
        max_corrections = max(
            int(self.navigation_cfg.get('max_verifier_corrections_per_subtask',
                self.navigation_cfg.get('max_verifier_checks_per_subtask',
                    self.navigation_cfg.get('max_rollbacks_per_subtask', 3)))),
            0
        )
        per_task = max(0, max_corrections)
        total = max(0, int(self.navigation_cfg.get('max_verifier_corrections_total', per_task * 5)))
        if count >= per_task or sum(self._verifier_corrections.values()) >= total:
            raise NavigationAborted('verifier_correction_budget_exhausted')

        self._verifier_feedback = (
            self._build_verifier_feedback(key, verification, repeated_without_motion)
            if hasattr(self, '_build_verifier_feedback')
            else {'subtask': key, 'reason': str(verification.get('reason') or ''), 'repeated_without_motion': repeated_without_motion}
        )

        if (verification.get('evidence_repair_required') or verification.get('endpoint_relocalization_exhausted')
                or verification.get('endpoint_confirmation_exhausted')):
            # More forced turns cannot repair an exhausted observation attempt
            # or a historical citation. Let the planner reconsider the measured
            # diagnostics under the existing finite correction budget.
            self._verifier_corrections[key] = count + 1
            self.navigation_trace.append({
                'role': 'verifier_correction', 'subtask_id': key,
                'action': 'handback_to_planner', 'executed_type': 'noop',
                'reason': str(verification.get('reason') or ''),
                'attempt': count + 1, 'source': ('endpoint_confirmation_exhausted'
                    if verification.get('endpoint_confirmation_exhausted') else 'endpoint_relocalization_exhausted'
                    if verification.get('endpoint_relocalization_exhausted') else 'evidence_repair_exhausted')})
            return {'type': 'noop'}

        # A measured far target permits a bounded approach through an existing
        # certified waypoint, instead of treating every geometry veto as a scan.
        approach = (self._endpoint_approach_action(verification, actions)
                    if hasattr(self, '_endpoint_approach_action') else None)
        if approach is not None and not verification.get('rollback_required'):
            self._verifier_corrections[key] = count + 1
            self.navigation_trace.append({'role': 'verifier_correction', 'subtask_id': key,
                'action': approach['action_id'], 'executed_type': 'waypoint',
                'reason': 'Approach the localized target using a certified waypoint.',
                'attempt': count + 1, 'source': 'measured_target_depth'})
            return approach

        # Wrong-path rollback trigger: if the Verifier diagnosed a wrong path/room (or repeated scene contradiction),
        # roll back to the last decision junction checkpoint rather than wandering deeper into the wrong area.
        if self.navigation_cfg.get('enable_wrong_path_rollback', False) and hasattr(self, 'plan_rollback_path'):
            is_wrong_path = bool(
                verification.get('failure_type') == 'wrong_path'
                or verification.get('rollback_required') is True
            )
            if is_wrong_path:
                if not hasattr(self, '_rollback_counts'):
                    self._rollback_counts = {}
                rb_count = self._rollback_counts.get(key, 0)
                max_rb = int(self.navigation_cfg.get('max_rollbacks_per_subtask', 2))
                if rb_count < max_rb:
                    # Only the measured route gate can authorize an evidence
                    # cutoff. A raw wrong_path label is not a rollback plan.
                    witness = verification.get('route_contradiction') or {}
                    constraint = witness.get('recovery_constraint')
                    accepted = (verification.get('route_contradiction_gate') or {}).get('accepted') is True
                    if accepted and isinstance(constraint, dict):
                        rb_path, rb_rot = self.plan_rollback_path(subtask_key=key, recovery_constraint=constraint)
                    else:
                        rb_path, rb_rot = None, None
                        self.navigation_trace.append({'role': 'controller',
                            'event': 'wrong_path_rollback_unavailable', 'status': 'unavailable',
                            'step': int(getattr(self, '_last_step_index', 0)), 'subtask_id': key,
                            'reason': 'No validated contradiction checkpoint boundary.', 'recovery_source': 'wrong_path'})
                    if rb_path and len(rb_path) > 1:
                        self._verifier_corrections[key] = count + 1
                        rb_reason = str(verification.get('reason') or verification.get('correction_reason') or 'Verifier detected wrong path/room.')
                        self.navigation_trace.append({
                            'role': 'verifier_correction', 'subtask_id': key,
                            'action': 'rollback', 'executed_type': 'rollback',
                            'reason': rb_reason, 'attempt': count + 1
                        })
                        logging.info(f"[Verifier Rollback] Triggering wrong-path rollback for '{key}' ({rb_count + 1}/{max_rb}): {rb_reason}")
                        return {
                            'type': 'rollback',
                            'subtask_key': key,
                            'reason': rb_reason,
                            'recovery_source': 'wrong_path',
                            'recovery_constraint': copy.deepcopy(constraint),
                            'preplanned_path': rb_path,
                            'preplanned_rot': rb_rot,
                        }
                    if isinstance(self._verifier_feedback, dict):
                        self._verifier_feedback['rollback_unavailable'] = True
                        self._verifier_feedback['recovery_constraint'] = copy.deepcopy(constraint)
                        self._verifier_feedback['rollback_reason'] = str(self.navigation_trace[-1].get('reason', ''))

        if self.navigation_cfg.get('verifier_return_control_to_planner', False) and not (repeated_without_motion and count >= 1):
            self._verifier_corrections[key] = count + 1
            self.navigation_trace.append({'role': 'verifier_correction', 'subtask_id': key,
                                          'action': 'handback_to_planner', 'executed_type': 'noop',
                                          'reason': str(verification.get('reason') or ''),
                                          'attempt': count + 1})
            logging.info(f"[Verifier] Rejected completion of '{key}' ({count + 1}/{per_task}); returning control to planner.")
            return {'type': 'noop'}

        legal = {}
        for name, action in (actions or {}).items():
            if action.get('rejected') or str(name).upper() in ('STOP', '-1'):
                continue
            if action.get('type') == 'turn':
                angle = action.get('angle', 0)
                if not np.isfinite(angle) or abs(angle) < 1e-5:
                    continue
            elif action.get('type') == 'waypoint':
                target = np.asarray(action.get('target_point_world', []), dtype=float)
                if target.shape != (3,) or not np.isfinite(target).all():
                    continue
                if np.linalg.norm(target[:2] - np.asarray(self.mapper.current_position)[:2]) < 0.05:
                    continue
            else:
                continue
            legal[str(name).upper()] = action
        choice, correction = None, {}
        try:
            raw_corr = verification.get('correction_action')
            norm_fn = globals().get('normalize_action_object')
            if callable(norm_fn) and raw_corr is not None:
                raw_corr = norm_fn({'action': raw_corr}).get('action')
            correction = decode_action(raw_corr, legal, completion=False)
            choice = correction['action_id']
        except ActionProtocolError:
            pass
        reliable_rejection = (
            verification.get('available') is True
            and verification.get('outcome') is False
            and not verification.get('contract_conflict')
            and verification.get('evidence_state') == 'contradicted'
            and verification.get('confidence') in ('medium', 'high')
            and bool(str(verification.get('evidence') or '').strip())
        )
        if not reliable_rejection or choice not in legal:
            # Unknown evidence does not justify translation; one existing turn
            # can expose the scene. Finite budgets also cover blocked movement.
            choice = next((k for k, v in legal.items() if v['type'] == 'turn'), None)
        if choice is None:
            raise NavigationAborted('verifier_rejection_no_legal_correction')
        self._verifier_corrections[key] = count + 1
        selected = dict(legal[choice], action_id=choice, is_verifier_correction=True)
        selected['requested_action'] = (dict(verification['correction_action']) if reliable_rejection
            and isinstance(verification.get('correction_action'), dict)
            and choice == correction.get('action_id') else
            {'type': 'turn', 'direction': (choice[5:] if choice.startswith('TURN_') else choice)})
        self.navigation_trace.append({'role': 'verifier_correction', 'subtask_id': key,
                                      'action': choice, 'executed_type': selected['type'],
                                      'reason': str(verification.get('reason') or ''),
                                      'attempt': count + 1})
        return selected


    def _retrieve_navigation_memories(self):
        key, subtask = self.instruction_obj.get_current_subtask()
        if not subtask or not getattr(self, 'navigation_memory', None):
            self._last_retrieved_memories = []
            return []

        # Check retrieval cache: same subtask hits cache directly without searching
        cached = getattr(self, '_retrieval_cache', {}).get(key)
        if cached is None and hasattr(self.instruction_obj, 'get_cached_retrieval'):
            cached = self.instruction_obj.get_cached_retrieval(key)
        elif hasattr(self, '_retrieval_cache'):
            cached = self._retrieval_cache.get(key)

        if cached is not None:
            self._last_retrieved_memories = cached
            return self._last_retrieved_memories

        query = (
            f"task: {self.instruction}; current subtask: {subtask.get('instruction', '')}; "
            f"objects: {' '.join(subtask.get('recognized_objects', []))}"
        )
        # Cache even an unavailable search: at most one retrieval per subtask.
        self._retrieval_cache = getattr(self, '_retrieval_cache', {})
        self._retrieval_cache[key] = []
        try:
            memories = self.navigation_memory.search(query, limit=1, scene_id=self._scene_scope())
        except Exception as exc:
            logging.getLogger(__name__).warning('Memory retrieval unavailable: %s', exc)
            memories = []
        self._last_retrieved_memories = (memories or [])[:1]
        if hasattr(self.instruction_obj, 'set_cached_retrieval'):
            self.instruction_obj.set_cached_retrieval(key, self._last_retrieved_memories)
        if not hasattr(self, '_retrieval_cache'):
            self._retrieval_cache = {}
        self._retrieval_cache[key] = self._last_retrieved_memories
        self.instruction_obj.add_planner_memory(self._last_retrieved_memories)
        return self._last_retrieved_memories

    def _format_in_episode_working_memory(self) -> str:
        lines = []
        for key, st in self.instruction_obj.sub_instruction_dict.items():
            if st.get('completed'):
                inst = st.get('instruction', '')
                objs = st.get('recognized_objects', [])
                coord = st.get('final_coord')
                elev_str = ""
                if coord and len(coord) == 3:
                    elev_str = f" [ended at z={coord[2]:.2f}m]"
                obj_str = f"; landmarks: [{', '.join(objs[:3])}]" if objs else ""
                lines.append(f"- Subtask {key} (COMPLETED): \"{inst}\"{elev_str}{obj_str}")
        try:
            mapper = getattr(self, 'mapper', None)
            if mapper is not None:
                curr_p = np.asarray(mapper.current_position, dtype=float)
                start_p = self._episode_start_position() if hasattr(self, '_episode_start_position') else np.zeros(3, dtype=float)
                traveled_m = float(getattr(self, 'traveled_distance', 0.0) or 0.0)
                net_disp_m = float(np.linalg.norm(curr_p[:2] - start_p[:2]))
                dz_m = float(curr_p[2] - start_p[2])
                traj = getattr(mapper, 'trajectory_position', None)
                milestones_str = ""
                if isinstance(traj, list) and len(traj) >= 2:
                    n_t = len(traj)
                    m_idxs = sorted(set([0, n_t // 3, (2 * n_t) // 3, n_t - 1]))
                    m_pts = []
                    for idx in m_idxs:
                        p = np.asarray(traj[idx], dtype=float)
                        m_pts.append(f"({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})")
                    milestones_str = f"; route breadcrumbs: {' -> '.join(m_pts)}"
                lines.append(
                    f"- Cumulative Episode Progress: total_walked={traveled_m:.2f}m, "
                    f"net_horizontal_displacement_from_start={net_disp_m:.2f}m, "
                    f"elevation_change={dz_m:+.2f}m{milestones_str}"
                )
        except Exception:
            pass
        return "\n".join(lines) if lines else ""

    def _memory_prompt_context(self) -> str:
        if not getattr(self, 'navigation_memory', None):
            return ""
        return self.navigation_memory.prompt_context(self._last_retrieved_memories, limit=1)

    def _record_completed_subtask_memory(self, key, subtask):
        if not subtask or not getattr(self, 'navigation_memory', None):
            return
        inst = subtask.get('instruction', '')
        objs = subtask.get('recognized_objects', [])
        coord = subtask.get('final_coord')
        verifier_data = subtask.get('verifier') or {}

        is_verified = bool(
            verifier_data.get('available') is True
            and verifier_data.get('outcome') is True
            and not verifier_data.get('contract_conflict')
            and verifier_data.get('relation_satisfied') is True
            and verifier_data.get('completion_evidence_reliable') is True
            and subtask.get('completed') is True
        )

        if not is_verified:
            return

        record = {
            'memory_schema_version': 2,
            'subtask_id': key,
            'subtask': inst,
            'instruction': inst,
            'task_instruction': self.instruction,
            'scene_id': self._scene_scope(),
            'episode_id': str(getattr(self, '_current_episode_id', getattr(self, 'episode_id', ''))),
            'status': 'completed',
            'outcome': True,
            'completion_evidence_reliable': True,
            'verified': is_verified,
            'subtask_verified': is_verified,
            'retrieval_eligible': is_verified,
            'evidence_status': 'subtask_verified' if is_verified else 'audit_only',
            'verifier': verifier_data,
            'verifier_history': [verifier_data] if verifier_data else [],
            'recognized_objects': objs[:5],
            'final_coord': coord,
            'waypoint_reason': subtask.get('waypoint_reason', ''),
            'outcome_reason': subtask.get('outcome_reason', ''),
        }
        role = "verified_subtask" if is_verified else "unverified_subtask"
        try:
            saved_entry = self.navigation_memory.add(record, role=role)
            print(f"[MEMORY RECORDED] Successfully recorded subtask memory for {key}: role={role}, status={record.get('evidence_status')}, backend={saved_entry.get('backend')}, audit_id={saved_entry.get('id')}")
        except Exception as e:
            logging.getLogger(__name__).warning("Failed to record subtask memory: %s", e)

    def finalize_navigation_memory(self):
        if getattr(self, '_memory_finalized', False):
            return
        self._memory_finalized = True

    def _record_branch_checkpoint(self, actions: dict):
        """Record decision junction / branch checkpoints along the trajectory for wrong-path rollback."""
        if getattr(self, 'rolling_back', False) or not hasattr(self, 'mapper') or self.mapper is None:
            return
        try:
            key = self.instruction_obj.get_current_subtask_key() if hasattr(self, 'instruction_obj') and self.instruction_obj else 'P1'
            curr_mapper = np.asarray(self.mapper.current_position, dtype=float)
            curr_ground = curr_mapper + np.array([0.0, 0.0, float(getattr(self.mapper, 'initial_floor_height', -1.2))], dtype=float)
            unvisited_views = set()
            for v in (actions or {}).values():
                if isinstance(v, dict) and v.get('type') == 'waypoint' and not v.get('visited'):
                    unvisited_views.add(str(v.get('view', 'front')).lower())
            is_junction = len(unvisited_views) >= 2
            if not hasattr(self, '_branch_checkpoints') or not isinstance(self._branch_checkpoints, list):
                self._branch_checkpoints = []
            if self._branch_checkpoints:
                last_cp = self._branch_checkpoints[-1]
                last_pos = np.asarray(last_cp.get('mapper_pose', curr_mapper), dtype=float)
                dist_from_last = float(np.linalg.norm(curr_mapper[:2] - last_pos[:2]))
                dz_from_last = abs(float(curr_mapper[2] - last_pos[2])) if len(curr_mapper) >= 3 and len(last_pos) >= 3 else 0.0
                if dist_from_last < 0.50:
                    if is_junction and not last_cp.get('is_junction'):
                        last_cp['is_junction'] = True
                        last_cp['unvisited_views'] = sorted(unvisited_views)
                    return
                if not is_junction and dist_from_last < 0.85 and dz_from_last < 0.25:
                    return
            self._branch_checkpoints.append({
                'step': int(getattr(self, '_last_step_index', 0)),
                'ledger_len': len(getattr(self, '_execution_ledger', [])),
                'history_len': len(getattr(self, 'history_msgs', [])),
                'subtask_key': key,
                'ground_pose': curr_ground.round(3).tolist(),
                'mapper_pose': curr_mapper.round(3).tolist(),
                'rotation': np.asarray(self.mapper.current_rotation, dtype=float).tolist(),
                'yaw': float(self._get_agent_yaw()) if hasattr(self, '_get_agent_yaw') else 0.0,
                'floor_level': int(getattr(self.mapper, 'floor_level', 0)),
                'is_junction': bool(is_junction),
                'unvisited_views': sorted(unvisited_views),
                'chosen_target': None,
                'chosen_view': None,
                'chosen_action_id': None,
                'rollback_visits': 0,
            })
            self._branch_checkpoints = self._branch_checkpoints[-16:]
        except Exception:
            pass

    def _update_last_checkpoint_choice(self, selected: dict):
        """Record which branch direction / waypoint target was taken from the latest checkpoint."""
        if not isinstance(selected, dict) or selected.get('type') != 'waypoint':
            return
        try:
            cps = getattr(self, '_branch_checkpoints', None)
            if not cps:
                return
            curr_mapper = np.asarray(self.mapper.current_position, dtype=float)
            for cp in reversed(cps):
                cp_pos = np.asarray(cp.get('mapper_pose', curr_mapper), dtype=float)
                if float(np.linalg.norm(curr_mapper[:2] - cp_pos[:2])) <= 0.85:
                    tw = selected.get('target_point_world')
                    if tw is not None:
                        cp['chosen_target'] = np.asarray(tw, dtype=float).round(3).tolist()
                    cp['chosen_view'] = str(selected.get('view', 'front')).upper()
                    cp['chosen_action_id'] = str(selected.get('action_id', ''))
                    break
        except Exception:
            pass

    def decide_waypoint(self, is_stuck=False, is_first_step=False, is_reverify=False, is_terminal_probe=False):
        """
        Decide the current target waypoint or action.
        This function analyzes the current state (map + 4 views) via VLM, selects a waypoint or action,
        then returns the action details for subsequent planning and execution.
        :return: A decoded current candidate or a boundary-verification decision.
                 A completion proposal must pass the verifier before STOP.
                 Returns None if all schema-repair attempts fail.
        """
        obs = self.curr_obs

        self._navigation_decision_id = getattr(self, '_navigation_decision_id', 0) + 1
        labeled_image_np, actions = self._preprocessing_module(obs)
        actions = self._constrain_branch_actions(actions)
        if hasattr(self, '_filter_turn_departure_actions'):
            actions = self._filter_turn_departure_actions(actions)
        self.last_actions = actions

        self.panorama_360_image = None
        pil_panorama_image = None
        if str(self.navigation_cfg.get('planner_view_mode', 'separate')).lower() == 'panorama':
            try:
                self._create_panoramic_360_image(getattr(self, 'labeled_views', {}))
                pil_panorama_image = self.panorama_360_image
            except Exception as exc:
                self.panorama_360_image = None
                logging.getLogger(__name__).warning(
                    "Failed to create 360-degree panorama; using independent views: %s", exc)

        depth_stats = self._get_multiview_depth_stats(obs) if hasattr(self, '_get_multiview_depth_stats') else {}
        img = labeled_image_np.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.85
        thickness = 2
        f_st = depth_stats.get('front')
        label = f"Front: center={f_st['center_m']:.1f}m, p85={f_st['far_m']:.1f}m" if f_st else 'Front View'
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

        text_x = max(10, img.shape[1] - text_size[0] - 20)
        text_y = 50

        overlay = img.copy()
        cv2.rectangle(overlay, (text_x - 10, text_y - text_size[1] - 10),
                      (text_x + text_size[0] + 10, text_y + 10),
                      (255, 255, 255), -1)
        cv2.putText(overlay, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

        alpha = 0.7
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        pil_labeled_rgb_image = Image.fromarray(img[:, :, :3], 'RGB')

        view_images = []
        for direction, base_label in [('left', 'Left View'), ('right', 'Right View'), ('back', 'Back View')]:
            if getattr(self, 'labeled_views', {}).get(direction) is not None:
                img = self.labeled_views[direction].copy()
            elif obs.get(direction, {}).get('color_sensor') is not None:
                img = obs[direction]['color_sensor'].copy()
            else:
                view_images.append(None)
                continue

            v_st = depth_stats.get(direction)
            label = f"{base_label}: center={v_st['center_m']:.1f}m, p85={v_st['far_m']:.1f}m" if v_st else base_label
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = max(10, img.shape[1] - text_size[0] - 20)
            text_y = 50

            overlay = img.copy()
            cv2.rectangle(overlay, (text_x - 10, text_y - text_size[1] - 10),
                          (text_x + text_size[0] + 10, text_y + 10),
                          (255, 255, 255), -1)
            cv2.putText(overlay, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

            view_images.append(Image.fromarray(img[:, :, :3], 'RGB'))

        top_down_map_np = create_top_down_map_centered(self.mapper, self.config['camera']['fov'],
                                                       self.target_world_position, action_candidates=self.last_actions, subtasks=self.instruction_obj.get_subtasks_key_coord())
        top_down_map_rgb = cv2.cvtColor(top_down_map_np, cv2.COLOR_BGR2RGB)
        pil_map_image = Image.fromarray(top_down_map_rgb)

        if is_reverify:
            ep_ctrl = getattr(self, '_endpoint_controller', None)
            if hasattr(self, '_record_branch_checkpoint') and (ep_ctrl is None or not getattr(ep_ctrl, 'active', False)):
                self._record_branch_checkpoint(actions)
            proposal = {'action': {'type': 'propose_completion'}, '_controller_probe': is_terminal_probe}
            decision = self._boundary_stop_decision(
                proposal, pil_labeled_rgb_image, view_images,
                pil_map_image, obs, actions)
            vlm_response_text = f"(Verifier Re-verification) Proposal evaluated at adjusted pose: decision={decision.get('type')}"
            return decision, vlm_response_text, pil_labeled_rgb_image

        key, st = self.instruction_obj.get_current_subtask()
        if hasattr(self, '_record_branch_checkpoint'):
            self._record_branch_checkpoint(actions)
        if hasattr(self, 'navigation_memory') and self.navigation_memory is not None:
            if key != getattr(self, '_last_memory_subtask_key', None):
                self._retrieve_navigation_memories()
                self._last_memory_subtask_key = key

        active_instr = self.instruction
        prompt = self.generate_prompt(active_instr, is_stuck, is_first_step)
        prompt += '\nBEV height tint: blue below, amber above the current floor (0.25m bands); color is not walkability or proof of stair completion. Confirm the landing and requested doorway in current views.'
        depth_and_wp_str = self._format_multiview_depth_and_waypoints(obs, actions) if hasattr(self, '_format_multiview_depth_and_waypoints') else ""
        if depth_and_wp_str:
            prompt += '\n\n' + depth_and_wp_str

        recovery_context = getattr(self, '_last_recovery_context', None)
        if (isinstance(recovery_context, dict) and recovery_context.get('status') == 'completed'
                and recovery_context.get('recovery_source') == 'physical'):
            prompt += ('\n[PHYSICAL RECOVERY COMPLETED]: Returned along a certified path after '
                       'blocked movement or an unresolved local observation. This does not prove a '
                       'wrong room or invalidate witnessed instruction progress. Reassess the current views.')
        if isinstance(recovery_context, dict) and recovery_context.get('status') == 'failed':
            prompt += '\n[ROLLBACK FAILED]: ' + json.dumps(recovery_context) + '\nUse current observed pose; do not assume the checkpoint was reached.'
        rb_info = getattr(self, '_last_rollback_info', None)
        if isinstance(rb_info, dict) and rb_info.get('subtask') == key:
            rb_shown = int(rb_info.get('shown_count', 0)) + 1
            rb_info['shown_count'] = rb_shown
            if rb_shown > 2:
                self._last_rollback_info = None
            else:
                blocked_v = f" (previously went {rb_info['blocked_view']})" if rb_info.get('blocked_view') else ""
                prompt += (
                    f"\n\n[WRONG-PATH ROLLBACK COMPLETED]:\n"
                    f"- You previously took a wrong branch from this junction toward {rb_info.get('from_pose')}{blocked_v} "
                    f"and were rolled back to {rb_info.get('to_pose')}. Reason: {rb_info.get('reason', '')}\n"
                    f"- The failed branch direction has been blocked. Do NOT walk back into that wrong room/corridor!\n"
                    f"- Re-examine the instruction and all 4 views (FRONT, LEFT, RIGHT, BACK) from this junction and select a different unvisited waypoint branch."
                )

        previous_check = (st or {}).get('verifier') or {}
        feedback = getattr(self, '_verifier_feedback', None)
        if feedback and feedback.get('subtask') == key:
            shown_count = int(feedback.get('shown_count', 0)) + 1
            feedback['shown_count'] = shown_count
            fb_pose = feedback.get('pose')
            moved_from_fb = (
                float(np.linalg.norm(np.asarray(self.mapper.current_position, dtype=float)[:2] - np.asarray(fb_pose, dtype=float)[:2]))
                if isinstance(fb_pose, (list, tuple)) and len(fb_pose) >= 2 else 0.0
            )
            yaw_changed = abs(self._normalize_angle(
                checkpoint_yaw(self.mapper.current_rotation) - float(feedback.get('yaw', checkpoint_yaw(self.mapper.current_rotation)))))
            if moved_from_fb >= 0.20 or yaw_changed >= math.radians(20):
                self._verifier_feedback = None
            else:
                prompt += (
                    '\nINDEPENDENT VERIFIER REJECTED YOUR LAST COMPLETION PROPOSAL (at pose '
                    + json.dumps(feedback.get('pose')) + '). Reason: ' + feedback.get('reason', '')
                    + '\nUnmet conditions: ' + json.dumps(feedback.get('unmet_conditions', []), ensure_ascii=False)
                    + ('\nEndpoint evidence remains unresolved after bounded observations. '
                       'The previous route is paused; refine the current target/jamb regions or choose a '
                       'new justified approach. No STOP is allowed without measured endpoint evidence. '
                       'Measurements: ' + json.dumps({name: feedback.get(name) for name in
                           ('endpoint_grounding', 'endpoint_reference_grounding')}, ensure_ascii=False)
                       if (feedback.get('endpoint_relocalization_exhausted')
                           or feedback.get('endpoint_confirmation_exhausted')) else '')
                    + ('\nVerifier note: ' + feedback['correction_reason'] if feedback.get('correction_reason') else '')
                    + ('\nVerifier hint (waypoint IDs are regenerated every step; match by location, not by ID): '
                       + json.dumps(feedback['suggested_action']) if feedback.get('suggested_action') else '')
                    + ('\nThe changed view did not corroborate the endpoint claim. This is not proof that '
                       'more forward movement is needed. Reidentify the same target and its stopping region '
                       'using the current views, or choose a justified new approach. Do not repeat an '
                       'unresolved completion claim from the same position.'
                       if feedback.get('endpoint_confirmation_exhausted') else
                       '\nThe route evidence passed, but the selected target regions did not yield reliable '
                       'endpoint depth. This is not proof that more forward movement is needed. '
                       'Use the current images and measurement diagnostics to localize the same endpoint '
                       'again or choose a justified new approach; never claim completion without valid depth.'
                       if feedback.get('endpoint_relocalization_exhausted') else
                       '\nThe controller could not validate the cited execution evidence. This alone does not '
                       'prove the current position is wrong. Review the unmet diagnostics and actual ledger. '
                       'Propose completion again only if current evidence supports corrected citations; '
                       'otherwise choose an observation or route action that addresses a real unmet condition.'
                       if feedback.get('evidence_repair_required') else
                       '\nYou are NOT at the destination yet. Move or turn so the unmet conditions become visibly true '
                       'in the current views, and only then propose completion. Proposing again from the same '
                       'place will be rejected again.')
                    + ('\nYour last proposal was repeated without moving; choose a navigate or turn action now.'
                       if feedback.get('repeated_without_motion') and not (feedback.get('evidence_repair_required')
                           or feedback.get('endpoint_relocalization_exhausted')
                           or feedback.get('endpoint_confirmation_exhausted')) else '')
                )
        elif previous_check and not self._verification_accepts_stop(previous_check):
            prompt += '\nPrevious completion was not verified. Reassess current observations: ' + str(previous_check.get('reason', 'unknown'))[:300]

        working_memory = self._format_in_episode_working_memory()
        if working_memory:
            prompt += f"\n\nCURRENT EPISODE WORKING MEMORY (progress in this run):\n{working_memory}"
        action_history = self._format_recent_action_history() if hasattr(self, '_format_recent_action_history') else ""
        if action_history:
            prompt += ("\n\nRECENT ACTION HISTORY (oldest to newest; measured by the simulator, not your intentions). "
                       "Use it to avoid repeating moves that were blocked or went the wrong way:\n" + action_history)
        historical = self._memory_prompt_context()
        if historical:
            prompt += f"\n\nHISTORICAL EPISODIC MEMORY (advisory only; never override current observations or verifier conclusions):\n{historical[:1200]}"


        prompt += '\n' + action_prompt(actions)
        prompt += '\nMEASURED EXECUTION FEEDBACK: ' + json.dumps(self._execution_feedback(), ensure_ascii=False)
        if hasattr(self, '_turn_execution_prompt'):
            prompt += self._turn_execution_prompt()

        wp_count = sum(1 for v in actions.values() if v.get('type') == 'waypoint')
        if wp_count == 0:
            turn_options = [k for k in actions.keys() if actions[k].get('type') == 'turn']
            prompt += f"\nNo available waypoints detected in current view. Please choose an available turn from {turn_options} to explore more."

        max_retry = 3
        vlm_response_text = ""
        retry_note = ""
        for attempt in range(max_retry):
            try:
                self._pending_planner_history = None
                vlm_response_text = self._get_vlm_response_multiview(
                    pil_labeled_rgb_image, view_images[0] if len(view_images) > 0 else None,
                    view_images[1] if len(view_images) > 1 else None,
                    view_images[2] if len(view_images) > 2 else None,
                    pil_map_image, prompt + retry_note, self.img_buffer,
                    pil_panorama_image=pil_panorama_image
                )

                parsed_json = robust_json_parse(vlm_response_text)
                norm_fn = globals().get('normalize_action_object')
                if callable(norm_fn) and isinstance(parsed_json, dict):
                    parsed_json = norm_fn(parsed_json)

                if not isinstance(parsed_json, dict) or 'action' not in parsed_json:
                    raise ActionProtocolError('missing top-level action object')
                _omit = object()

                def _json_safe(val):
                    if val is None or isinstance(val, (bool, str, int)):
                        return val
                    if isinstance(val, np.bool_):
                        return bool(val)
                    if isinstance(val, np.integer):
                        return int(val)
                    if isinstance(val, (float, np.floating)):
                        fval = float(val)
                        return fval if (fval == fval and fval not in (float('inf'), float('-inf'))) else None
                    if isinstance(val, dict):
                        out = {}
                        for k, v in val.items():
                            if isinstance(k, str) and not k.startswith('_'):
                                sv = _json_safe(v)
                                if sv is not _omit:
                                    out[k] = sv
                        return out
                    if isinstance(val, (list, tuple)):
                        out = []
                        for item in val:
                            sv = _json_safe(item)
                            if sv is not _omit:
                                out.append(sv)
                        return out
                    return _omit

                proposed_action = _json_safe(parsed_json['action'])
                selected = decode_action(parsed_json['action'], actions)
                if hasattr(self, '_validate_turn_selection'):
                    selected = self._validate_turn_selection(parsed_json, selected)

                req_action = selected.get('requested_action') if isinstance(selected, dict) else None
                is_override = False
                if isinstance(selected, dict) and isinstance(proposed_action, dict):
                    prop_type = proposed_action.get('type')
                    if bool(selected.get('controller_override') or selected.get('turn_override')):
                        is_override = True
                    elif isinstance(req_action, dict):
                        req_type = req_action.get('type')
                        if bool(req_action.get('controller_override') or req_action.get('turn_override')):
                            is_override = True
                        elif req_action.get('purpose') == 'landmark_reorientation':
                            is_override = True
                        elif prop_type == 'navigate':
                            if req_type not in ('navigate', 'waypoint', None):
                                is_override = True
                            elif (req_action.get('waypoint_id') is not None
                                  and proposed_action.get('waypoint_id') is not None
                                  and req_action.get('waypoint_id') != proposed_action.get('waypoint_id')):
                                is_override = True
                        elif prop_type == 'turn':
                            if req_type not in ('turn', None):
                                is_override = True
                            elif (req_action.get('direction') is not None
                                  and proposed_action.get('direction') is not None
                                  and str(req_action.get('direction')).upper() != str(proposed_action.get('direction')).upper()):
                                is_override = True
                        elif req_type is not None and req_type != prop_type:
                            is_override = True
                    else:
                        if prop_type == 'turn':
                            if selected.get('type') not in ('turn', None):
                                is_override = True
                            elif (selected.get('direction') is not None
                                  and proposed_action.get('direction') is not None
                                  and str(selected.get('direction')).upper() != str(proposed_action.get('direction')).upper()):
                                is_override = True
                        elif prop_type == 'navigate':
                            if selected.get('type') not in ('waypoint', 'navigate', None):
                                is_override = True
                            elif (selected.get('action_id') is not None
                                  and proposed_action.get('waypoint_id') is not None
                                  and selected.get('action_id') != proposed_action.get('waypoint_id')):
                                is_override = True

                if is_override:
                    if isinstance(req_action, dict) and req_action:
                        pending_action = _json_safe(req_action)
                    else:
                        pending_action = _json_safe({
                            k: selected[k]
                            for k in ('type', 'direction', 'action_id', 'waypoint_id', 'theta', 'purpose', 'reason')
                            if isinstance(selected, dict) and k in selected
                        })
                    history_payload = _json_safe(parsed_json)
                    history_payload.pop('completed', None)
                    history_payload['action'] = _json_safe(pending_action)
                    history_payload['selected_pending_action'] = _json_safe(pending_action)
                    history_payload['original_action'] = _json_safe(proposed_action)
                    history_payload['action_status'] = 'pending'
                    history_payload['controller_override'] = True
                    trace_entry = {
                        'role': 'planner',
                        'event': 'action_selected',
                        'action': _json_safe(pending_action),
                        'selected_pending_action': _json_safe(pending_action),
                        'original_action': _json_safe(proposed_action),
                        'action_status': 'pending',
                        'controller_override': True,
                        'decision_step': getattr(self, '_last_step_index', 0),
                    }
                else:
                    history_payload = parsed_json
                    trace_entry = {
                        'role': 'planner',
                        'event': 'action_selected',
                        'action': parsed_json['action'],
                        'decision_step': getattr(self, '_last_step_index', 0),
                    }

                if hasattr(self, '_commit_planner_history'):
                    self._commit_planner_history(history_payload)
                # Generated prose never becomes execution evidence, and an
                # unrelated completed=true field cannot override this action.
                if isinstance(parsed_json.get('plan'), str):
                    self.instruction_obj.update_plan_for_current_subtask(parsed_json['plan'])
                self.navigation_trace.append(trace_entry)
                if selected['type'] == 'completion_proposal':
                    decision = self._boundary_stop_decision(
                        parsed_json, pil_labeled_rgb_image, view_images,
                        pil_map_image, obs, actions)
                    return decision, vlm_response_text, pil_labeled_rgb_image
                if selected['type'] == 'waypoint' and hasattr(self, '_update_last_checkpoint_choice'):
                    self._update_last_checkpoint_choice(selected)
                return selected, vlm_response_text, pil_labeled_rgb_image

            except ActionProtocolError as exc:
                reason = str(exc)[:240]
                retry_note = ('\nPrevious response rejected: ' + reason + '. '
                              'Return one complete JSON object with action at the top level, '
                              'using the current legal action objects. No action was executed.')
                self.navigation_trace.append({'role': 'controller', 'event': 'action_protocol_rejected',
                                              'attempt': attempt+1, 'reason': reason})
            except Exception as e:
                retry_note = ('\nPrevious response could not be processed (' + type(e).__name__ + '). '
                              'Return one complete JSON object using a current legal action. No action was executed.')
                print(f"[WARN in decide_waypoint] Attempt {attempt + 1}/{max_retry} failed with exception: {type(e).__name__}: {e}")
            finally:
                self._pending_planner_history = None

        return None, vlm_response_text, pil_labeled_rgb_image

    def plan_rollback_path(self, subtask_key: str = None, recovery_constraint=None):
        """
        Plan a rollback path to the most recent reachable decision junction checkpoint
        (or previous subtask start point if no intermediate branch checkpoint exists).
        Evidenced wrong-path returns must precede the contradicted motion while
        preserving prior verified clauses. Stalled/manual returns have no such
        constraint; an unavailable evidenced return never falls back past it.
        Returns the planned waypoint sequence and target rotation.
        """
        from route_progress import checkpoint_precedes_contradiction
        self._active_rollback_checkpoint = None
        if subtask_key is None:
            subtask_key = self.instruction_obj.get_current_subtask_key()

        if not subtask_key:
            return None, None

        def eligible(cp):
            return (recovery_constraint is None or
                    checkpoint_precedes_contradiction(cp, recovery_constraint, subtask_key))

        def unavailable(reason):
            if recovery_constraint is not None:
                self.navigation_trace.append({'role': 'controller',
                    'event': 'wrong_path_rollback_unavailable', 'status': 'unavailable',
                    'step': int(getattr(self, '_last_step_index', 0)), 'subtask_id': subtask_key,
                    'reason': reason, 'recovery_source': 'wrong_path',
                    'recovery_constraint': copy.deepcopy(recovery_constraint)})
            return None, None

        # 1. Try rolling back to the best intermediate decision junction checkpoint first
        cps = getattr(self, '_branch_checkpoints', None)
        if cps and isinstance(cps, list) and getattr(self, 'navigation_cfg', {}).get('enable_wrong_path_rollback', True):
            try:
                curr_ground = np.asarray(self.mapper.current_position, dtype=float) + np.array(
                    [0.0, 0.0, float(getattr(self.mapper, 'initial_floor_height', -1.2))], dtype=float
                )
                # Prefer junction checkpoints 1.2m - 5.0m away; otherwise any checkpoint 1.0m - 6.0m away
                for require_junction in (True, False):
                    for cp in reversed(cps):
                        if cp.get('subtask_key') != subtask_key or cp.get('floor_level') != getattr(self.mapper, 'floor_level', 0):
                            continue
                        if not eligible(cp):
                            continue
                        if cp.get('rollback_visits', 0) >= 2:
                            continue
                        if require_junction and not cp.get('is_junction'):
                            continue
                        cp_ground = np.asarray(cp.get('ground_pose', []), dtype=float)
                        if cp_ground.shape != (3,):
                            continue
                        dist_xy = float(np.linalg.norm(curr_ground[:2] - cp_ground[:2]))
                        min_d = 1.2 if require_junction else 1.0
                        max_d = 5.0 if require_junction else 6.0
                        if not (min_d <= dist_xy <= max_d):
                            continue
                        route = self._validated_waypoint_route(cp_ground, recovery=True)
                        if route is not None and len(route.get('path', [])) > 1:
                            route_length = sum(float(np.linalg.norm(np.asarray(b)-np.asarray(a)))
                                               for a, b in zip(route['path'], route['path'][1:]))
                            if route_length > float(self.navigation_cfg.get('rollback_max_path_m', 8.0)):
                                continue
                            self._active_rollback_checkpoint = cp
                            return route['path'], np.asarray(cp.get('rotation', self.init_rot), dtype=float)
            except Exception:
                pass

        # 2. Fall back to subtask start position
        try:
            target_idx = self.instruction_obj.sub_instruction_keys.index(subtask_key.upper())
        except ValueError:
            return unavailable('No recorded entry for the affected subtask.')

        if target_idx == 0:
            target_pos = self.init_pos
            target_rot = self.init_rot
        else:
            prev_subtask_key = self.instruction_obj.sub_instruction_keys[target_idx - 1]
            target_pos, target_rot = self.instruction_obj.get_subtask_pos_rot_by_key(prev_subtask_key)

        if target_pos is None or target_rot is None:
            return unavailable('Subtask entry pose is unavailable.')

        # The first task's initial pose is an actual pre-execution checkpoint.
        # Later subtask targets do not record a floor or execution-before index;
        # do not invent that evidence for a constrained wrong-path recovery.
        entry_cp = None
        if target_idx == 0:
            entry_cp = {'subtask_key': subtask_key, 'step': 0, 'ledger_len': 0,
                        'ground_pose': np.asarray(target_pos, dtype=float).tolist(),
                        'rotation': np.asarray(target_rot, dtype=float).tolist(), 'floor_level': 0,
                        'is_subtask_entry': True}
            if isinstance(recovery_constraint, dict):
                affected = recovery_constraint.get('earliest_affected_step')
                record = next((r for r in getattr(self, '_execution_ledger', [])
                    if r.get('step') == affected and r.get('subtask_id') == subtask_key
                    and not r.get('recovery') and not r.get('rejected_by_rollback')), {})
                requested = record.get('requested_action') or {}
                packet = record.get('waypoint_execution') or {}
                try:
                    if not isinstance(requested, dict) or not isinstance(packet, dict):
                        raise ValueError('missing recorded waypoint action')
                    before = np.asarray(record.get('pose_before'), dtype=float)
                    target = np.asarray(packet.get('target'), dtype=float)
                    if (requested.get('type') == 'navigate' and requested.get('waypoint_id')
                            and requested.get('waypoint_id') == packet.get('action_id')
                            and packet.get('phase') in ('aligning', 'moving', 'arrived')
                            and record.get('action_type') in ('turn', 'move_forward')
                            and isinstance(record.get('executed_action'), str) and record['executed_action']
                            and (abs(float(record.get('yaw_delta', 0.))) > 1e-5
                                 or float(record.get('motion_distance', 0.)) >= .02)
                            and before.shape == target.shape == (3,)
                            and np.isfinite(before).all() and np.isfinite(target).all()
                            and np.linalg.norm(before - np.asarray(target_pos)) <= .1
                            and np.linalg.norm(target[:2] - before[:2]) >= .15):
                        entry_cp.update(chosen_target=target.tolist(), chosen_action_id=packet['action_id'],
                            chosen_target_evidence={'step': affected, 'condition_id': recovery_constraint['condition_id'],
                                'waypoint_phase': packet['phase'], 'source': 'executed_waypoint_target'})
                except (TypeError, ValueError):
                    pass  # Missing recorded branch identity is not authority to invent a direction.
        if recovery_constraint is not None and (
                entry_cp is None or not eligible(entry_cp)
                or getattr(self.mapper, 'floor_level', 0) != entry_cp['floor_level']):
            return unavailable('No same-floor reachable checkpoint before the contradicted motion preserves the verified prefix.')

        route = self._validated_waypoint_route(target_pos, recovery=True)
        path = route['path'] if route is not None else None

        if not path or len(path) < 2:
            return unavailable('No certified route to a checkpoint before the contradicted motion.')
        route_length = sum(float(np.linalg.norm(np.asarray(b)-np.asarray(a))) for a, b in zip(path, path[1:]))
        if route_length > float(self.navigation_cfg.get('rollback_max_path_m', 8.0)):
            return unavailable('Return before the contradicted motion exceeds rollback_max_path_m.')
        self._active_rollback_checkpoint = entry_cp
        return path, target_rot

    def _get_agent_yaw(self):
        """
        Extract yaw angle (around z) from self.mapper.current_rotation (3x3 rotation matrix).
        """
        R = np.asarray(self.mapper.current_rotation)
        return checkpoint_yaw(R)

    def _rot_to_yaw(self, rotation):
        """
        Convert mapper.rotation to yaw angle.
        :param rotation:
        :return:
        """
        R = np.asarray(rotation)
        return checkpoint_yaw(R)

    def _normalize_angle(self, a):
        return float((a + np.pi) % (2 * np.pi) - np.pi)

    def step(self, step, rollback_step_threshold=70):
        """
        Execute one step based on prior agent decision; returns obs, top-down map, VLM response, labeled PIL image.
        """
        try:
            action_to_execute = None
            pending_decision = None
            pil_labeled_img = None
            self._last_step_index = step
            self._controller_step = step
            recovering = getattr(self, '_pending_recovery', None) is not None
            habitat_control = bool(self.navigation_cfg.get('habitat_control_enabled', False))
            if not recovering and habitat_control:
                recovering = self._supervise_physical_progress(step)
            if not recovering and not habitat_control and step - getattr(self, '_physical_progress_step', 0) >= int(self.navigation_cfg.get('max_no_progress_steps', 24)):
                raise NavigationAborted('no_physical_progress')
            if not recovering and not habitat_control and getattr(self, '_observation_turns_without_motion', 0) >= int(self.navigation_cfg.get('max_observation_turns_without_motion', 8)):
                raise NavigationAborted('repeated_observation_without_motion')
            if not hasattr(self, '_decision_fail_count'):
                self._decision_fail_count = getattr(self, 'decision_none_fail_count', 0)
            self.decision_none_fail_count = self._decision_fail_count
            if not hasattr(self, 'is_parse_fail'):
                self.is_parse_fail = False

            if not hasattr(self, '_position_history'):
                self._position_history = []
            if not hasattr(self, '_stuck_threshold'):
                self._stuck_threshold = 0.2
            if not hasattr(self, '_stuck_check_window'):
                self._stuck_check_window = 8
            if not hasattr(self, 'action_sequence'):
                self.action_sequence = []
            if not hasattr(self, 'current_path'):
                self.current_path = []

            if not hasattr(self, '_heading_before_nav'):
                self._heading_before_nav = 0.0
            if not hasattr(self, '_need_restore_heading'):
                self._need_restore_heading = False
            if not hasattr(self, '_no_move_steps'):
                self._no_move_steps = 0
            if not hasattr(self, '_global_stuck_limit'):
                self._global_stuck_limit = 50
            if not hasattr(self, '_no_move_epsilon'):
                self._no_move_epsilon = 1e-1

            current_pos = np.array(self.mapper.current_position[:2])
            self._position_history.append(current_pos)
            agent_state = self.curr_obs['agent_state']

            history_window = max(self._stuck_check_window + 1, int(self.navigation_cfg.get('dead_end_window_steps', 10)))
            self._position_history = self._position_history[-history_window:]

            if self.prev_agent_position is not None:
                prev3d = np.array(self.prev_agent_position)
                curr3d = np.array(agent_state.position)
                dist_moved = np.linalg.norm(curr3d - prev3d)
                last_act_type = getattr(getattr(self, 'last_action', None), 'type', '')
                if dist_moved < self._no_move_epsilon and last_act_type not in ('turn', 'L', 'R', 'B'):
                    self._no_move_steps += 1
                elif dist_moved >= self._no_move_epsilon:
                    self._no_move_steps = 0
            else:
                self._no_move_steps = 0

            if not recovering and not habitat_control and self._no_move_steps >= self._global_stuck_limit and action_to_execute is None:
                raise NavigationAborted('global_stuck_without_verified_completion')

            active = getattr(self, '_waypoint_execution', None)
            is_stuck = bool((active is not None and active.phase == 'blocked')
                            or getattr(self, '_observation_turns_without_motion', 0) >= 4)
            # Slow valid motion through a door is progress. Never discard a
            # certified route merely because eight movements were each <0.2m.

            if recovering:
                action_to_execute = self._recovery_action(step)
            elif (getattr(self, '_pending_endpoint_relocalization', None)
                  or getattr(self, '_pending_endpoint_confirmation', None)
                  or (getattr(getattr(self, '_endpoint_controller', None), 'active', False)
                      and not self.current_path and not self.action_sequence)
                  or self._terminal_probe_due(step, stalled=is_stuck, at_route_boundary=not self.current_path)):
                pending_localization = (getattr(self, '_pending_endpoint_relocalization', None)
                                        or getattr(self, '_pending_endpoint_confirmation', None))
                probe_trigger = pending_localization.get('is_probe', True) if pending_localization else True
                if pending_localization:
                    self._executing_verifier_correction = False
                if probe_trigger and not habitat_control:
                    self._claim_terminal_probe(step)
                probe, probe_response, probe_image = self.decide_waypoint(
                    is_stuck=is_stuck, is_reverify=True, is_terminal_probe=probe_trigger)
                self.vlm_responses.append(probe_response)
                self.last_vlm_response = probe_response
                pil_labeled_img = probe_image
                if probe is not None and probe.get('type') != 'continue_path':
                    pending_decision = (probe, probe_response, probe_image)
                    self.current_path, self.action_sequence = [], []
                    self._waypoint_execution = None

            if action_to_execute is None and pending_decision is None and self.current_path and not self.action_sequence:
                action_to_execute = self._get_action_for_next_waypoint()
                if (action_to_execute is PolarAction.null and not self.current_path
                        and not getattr(getattr(self, '_endpoint_controller', None), 'active', False)):
                    action_to_execute = None

            if action_to_execute is None and not self.current_path and not self.action_sequence:
                is_reverification = False
                if pending_decision is None and getattr(self, '_executing_verifier_correction', False):
                    self._executing_verifier_correction = False
                    cur_k = self.instruction_obj.get_current_subtask_key()
                    if not hasattr(self, '_verifier_correction_steps'):
                        self._verifier_correction_steps = {}
                    self._verifier_correction_steps[cur_k] = self._verifier_correction_steps.get(cur_k, 0) + 1
                    correction_count = self._verifier_correction_steps[cur_k]
                    if bool(self.navigation_cfg.get('verifier_auto_reverify', False)):
                        print(f"[Verifier Hand-off] Micro-adjustment step {correction_count} completed for subtask '{cur_k}'. Triggering closed-loop re-verification at new pose.")
                        is_reverification = True
                    else:
                        print(f"[Verifier Hand-off] Adjustment step {correction_count} completed for subtask '{cur_k}'. Returning control to planner with verifier feedback.")

                cur_sub_k = self.instruction_obj.get_current_subtask_key()
                eff_rb_threshold = int(self.navigation_cfg.get('rollback_step_threshold', rollback_step_threshold))
                rb_enabled = bool(self.navigation_cfg.get('enable_wrong_path_rollback', False)) and not habitat_control
                rb_used = getattr(self, '_rollback_counts', {}).get(cur_sub_k, 0)
                max_rb = int(self.navigation_cfg.get('max_rollbacks_per_subtask', 2))
                can_step_rollback = len(getattr(self.instruction_obj, 'sub_instruction_keys', [])) > 1
                dead_end_stalled = False
                if rb_enabled and rb_used < max_rb and len(getattr(self, '_branch_checkpoints', [])) >= 1 and len(self._position_history) >= 10:
                    recent_pts = np.asarray(self._position_history[-10:], dtype=float)
                    span_m = float(np.linalg.norm(recent_pts.max(axis=0) - recent_pts.min(axis=0)))
                    last_cp_pose = np.asarray(self._branch_checkpoints[-1].get('mapper_pose', [0.0, 0.0, 0.0]), dtype=float)
                    dist_from_cp = float(np.linalg.norm(np.asarray(self.mapper.current_position, dtype=float)[:2] - last_cp_pose[:2]))
                    if (span_m < 0.55 and getattr(self, '_observation_turns_without_motion', 0) >= 4
                            and float(getattr(self, 'traveled_distance', 0.0)) >= 3.0 and dist_from_cp >= 1.8):
                        dead_end_stalled = True

                if pending_decision is not None:
                    decision, vlm_response, pil_labeled_img = pending_decision
                elif is_reverification:
                    decision, vlm_response, pil_labeled_img = self.decide_waypoint(
                        is_stuck=is_stuck, is_first_step=False, is_reverify=True
                    )
                elif not habitat_control and (dead_end_stalled or (can_step_rollback and self.instruction_obj.current_step_count > eff_rb_threshold)):
                    rb_reason = (
                        "Agent stalled in dead-end/wrong branch without progress; rolling back to last decision junction."
                        if dead_end_stalled
                        else f"Exceeded step budget ({self.instruction_obj.current_step_count} > {eff_rb_threshold}) on current branch; rolling back to decision junction."
                    )
                    decision, vlm_response, pil_labeled_img = (
                        {"type": "rollback", "subtask_key": cur_sub_k, "reason": rb_reason,
                         "recovery_source": "stalled"},
                        f"(Auto Rollback) {rb_reason}",
                        None,
                    )
                else:
                    decision, vlm_response, pil_labeled_img = self.decide_waypoint(
                        is_stuck=is_stuck, is_first_step=step == 2
                    )

                self.vlm_responses.append(vlm_response)
                self.last_vlm_response = vlm_response

                if decision is not None:
                    self._requested_action = decision.get('requested_action')
                    self.decision_none_fail_count = self._decision_fail_count = 0
                    self.is_parse_fail = False
                    self.error = False

                if decision is None:
                    self.decision_none_fail_count = self._decision_fail_count = self._decision_fail_count + 1
                    if self._decision_fail_count >= 3:
                        print(f"[ERROR in step] Consecutive VLM decision failures reached limit ({self._decision_fail_count}/3), terminating episode with error.")
                        self._stop()
                        self.second_stop = True
                        action_to_execute = PolarAction.stop
                        self.error = True
                        self.is_parse_fail = True
                    else:
                        print(f"[WARN in step] VLM decision failed, robot will idle-scan and retry next step (fail count: {self._decision_fail_count}/3)")
                        self.current_path = []
                        self.action_sequence = []
                        self._need_restore_heading = False
                        action_to_execute = PolarAction.null
                        self.error = False
                        self.is_parse_fail = False

                elif decision['type'] == 'stop':
                    self._last_turn_dir = None
                    cur_k = self.instruction_obj.get_current_subtask_key()
                    if self.instruction_obj.is_last_subtask():
                        action_to_execute = PolarAction.stop
                        if not self.instruction_obj.get_subtask_by_key(cur_k).get('completed'):
                            self.instruction_obj.mark_current_subtask_completed(self.mapper.current_position - [0, 0, 1.2], self.mapper.current_rotation)
                            if cur_k:
                                self._record_completed_subtask_memory(cur_k, self.instruction_obj.get_subtask_by_key(cur_k))
                        self._stop()
                        self.second_stop = True
                    else:
                        action_to_execute = PolarAction.pause()
                        self.instruction_obj.mark_current_subtask_completed(self.mapper.current_position - [0, 0, 1.2], self.mapper.current_rotation)
                        if cur_k:
                            self._record_completed_subtask_memory(cur_k, self.instruction_obj.get_subtask_by_key(cur_k))

                elif decision['type'] == 'noop':
                    action_to_execute = PolarAction.null
                    self.current_path = []
                    self.action_sequence = []
                    self._need_restore_heading = False
                    self._executing_verifier_correction = decision.get('is_verifier_correction', False)

                elif decision['type'] == 'turn':
                    self._instruction_turn_command = decision.get('instruction_turn')
                    self._waypoint_execution = None
                    action_to_execute = PolarAction(r=0, theta=decision['angle'], type='turn')
                    if decision['angle'] > 0:
                        self._last_turn_dir = 'L'
                    elif decision['angle'] < 0:
                        self._last_turn_dir = 'R'
                    self.current_path = []
                    self._need_restore_heading = False
                    self._executing_verifier_correction = decision.get('is_verifier_correction', False)

                elif decision['type'] == 'look_around':
                    self._last_turn_dir = None
                    pass
                    turn_right_action = PolarAction(r=0, theta=-self.turn_angle_rad*2, type='turn')
                    self.action_sequence = [turn_right_action] * 2
                    action_to_execute = self.action_sequence.pop(0)
                    self._need_restore_heading = False
                    self._executing_verifier_correction = decision.get('is_verifier_correction', False)

                elif decision['type'] == 'waypoint':
                    self._last_turn_dir = None
                    # Revalidate after model latency; candidate and executor share
                    # the same backend. An empty route never means walk straight.
                    route = self._validated_waypoint_route(decision['target_point_world'])
                    self.current_path = [] if route is None else route['path'][1:]
                    self._need_restore_heading = False
                    self._executing_verifier_correction = decision.get('is_verifier_correction', False)
                    if route is not None:
                        self._waypoint_execution = WaypointExecution(
                            route['target'], decision.get('action_id', ''),
                            int(self.navigation_cfg.get('waypoint_max_execution_steps', 48)),
                            int(self.navigation_cfg.get('waypoint_max_alignment_turns', 8)))
                        self.target_world_position = route['target']
                        action_to_execute = self._get_action_for_next_waypoint()
                    else:
                        self._executing_verifier_correction = False
                        self.navigation_trace.append({'role': 'controller', 'event': 'waypoint_route_invalidated',
                                                      'step': step})
                        action_to_execute = PolarAction.null
                elif decision['type'] == 'rollback':
                    subtask_key = decision.get('subtask_key') or self.instruction_obj.get_current_subtask_key()
                    if decision.get('preplanned_path') is not None and decision.get('preplanned_rot') is not None:
                        path, target_rot = decision['preplanned_path'], decision['preplanned_rot']
                    else:
                        path, target_rot = self.plan_rollback_path(subtask_key=subtask_key,
                            recovery_constraint=decision.get('recovery_constraint'))
                    started = self._begin_checkpoint_recovery(
                        path, target_rot, checkpoint=getattr(self, '_active_rollback_checkpoint', None),
                        reason=decision.get('reason'), subtask_key=subtask_key,
                        recovery_constraint=decision.get('recovery_constraint'),
                        recovery_source=decision.get('recovery_source', 'manual'))
                    action_to_execute = self._recovery_action(step) if started else PolarAction.null

            elif action_to_execute is None and self.action_sequence:
                action_to_execute = self.action_sequence.pop(0)

            elif action_to_execute is None:
                action_to_execute = self._get_action_for_next_waypoint()

            self.instruction_obj.current_step_count += 1

            if action_to_execute is None:
                action_to_execute = PolarAction.null
                self.current_path = []

            try:
                self.execute_action(action_to_execute)
            except BaseException:
                self._cancel_checkpoint_recovery('execution_interrupted')
                raise

            self.prev_agent_position = agent_state.position

            top_down_map = create_top_down_map_centered(
                self.mapper,
                self.config['camera']['fov'],
                self.target_world_position,
                action_candidates=self.last_actions,
                subtasks=self.instruction_obj.get_subtasks_key_coord()
            )

            if isinstance(self.curr_obs, dict) and 'color_sensor' in self.curr_obs:
                current_image = self.curr_obs['color_sensor'][:, :, :3].copy()
                if action_to_execute is not None:
                    action_text = self._action_to_text(action_to_execute)
                    if getattr(self, '_executing_verifier_correction', False):
                        action_text = f"[Verifier Adjustment] {action_text}"
                    pos_3d = np.asarray(self.mapper.current_position, dtype=float)
                    action_text = f"[step {len(self._execution_ledger)} pos=({pos_3d[0]:.1f},{pos_3d[1]:.1f},{pos_3d[2]:.1f})] {action_text}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    font_thickness = 2
                    text_color = (255, 255, 255)
                    bg_color = (0, 0, 0)
                    padding = 10
                    (text_width, text_height), baseline = cv2.getTextSize(
                        action_text, font, font_scale, font_thickness
                    )
                    overlay = current_image.copy()
                    cv2.rectangle(
                        overlay,
                        (0, 0),
                        (text_width + 2 * padding, text_height + 2 * padding + baseline),
                        bg_color,
                        -1
                    )
                    cv2.addWeighted(overlay, 0.6, current_image, 0.4, 0, current_image)
                    cv2.putText(
                        current_image,
                        action_text,
                        (padding, text_height + padding),
                        font,
                        font_scale,
                        text_color,
                        font_thickness,
                        cv2.LINE_AA
                    )

                keyframe_pil_img = Image.fromarray(current_image, 'RGB')
                if pil_labeled_img is None or getattr(self, '_executing_verifier_correction', False):
                    pil_labeled_img = keyframe_pil_img

                # Episode-wide keyframe temporal-spatial sampling: preserve visual history from start to present.
                curr_pos_2d = np.asarray(self.mapper.current_position[:2], dtype=float)
                curr_yaw_val = float(self._get_agent_yaw()) if hasattr(self, '_get_agent_yaw') else 0.0
                is_keyframe = False
                last_pos = getattr(self, '_last_keyframe_pos', None)
                last_yaw = getattr(self, '_last_keyframe_yaw', None)
                if last_pos is None or last_yaw is None:
                    is_keyframe = True
                else:
                    dist_moved = float(np.linalg.norm(curr_pos_2d - last_pos))
                    norm_ang = getattr(self, '_normalize_angle', lambda a: float((a + math.pi) % (2 * math.pi) - math.pi))
                    yaw_diff = abs(norm_ang(curr_yaw_val - last_yaw))
                    if dist_moved >= 0.8 or yaw_diff >= math.radians(45):
                        is_keyframe = True

                if is_keyframe:
                    if not isinstance(getattr(self, 'img_buffer', None), list):
                        self.img_buffer = []
                    self.img_buffer.append(keyframe_pil_img)
                    if len(self.img_buffer) > 20:
                        # Keep earliest frame + most recent 19 frames
                        self.img_buffer = [self.img_buffer[0]] + self.img_buffer[-19:]
                    self._last_keyframe_pos = curr_pos_2d.copy()
                    self._last_keyframe_yaw = curr_yaw_val

            return self.curr_obs, top_down_map, self.last_vlm_response, pil_labeled_img
        except BaseException:
            if getattr(self, '_pending_recovery', None) is not None or getattr(self, 'rolling_back', False):
                self._cancel_checkpoint_recovery('navigation_step_interrupted')
            raise

    def _action_to_text(self, action: PolarAction) -> str:
        """
        Convert PolarAction to English description.
        :param action:
        :return: English description of the action.
        """
        if action.type == 'stop':
            return "STOP"
        elif action.type == 'pause':
            return "PAUSE"
        elif action.type == 'null':
            return ""

        if abs(action.r) < 0.01:  # essentially no forward movement, pure rotation
            angle_deg = np.rad2deg(action.theta)
            if angle_deg > 0:
                return f"Turn Left {abs(angle_deg):.1f} degrees"
            elif angle_deg < 0:
                return f"Turn Right {abs(angle_deg):.1f} degrees"
            else:
                return "NO ROTATION"

        text = f"Move Forward {action.r:.2f}m"
        if abs(action.theta) > 0.01:
            angle_deg = np.rad2deg(action.theta)
            if angle_deg > 0:
                text += f" + Turn Left {abs(angle_deg):.1f} degrees"
            else:
                text += f" + Turn Right {abs(angle_deg):.1f} degrees"

        return text

    def _validated_waypoint_route(self, target, recovery=False):
        """One physical backend for candidates, execution, and rollback."""
        target = np.asarray(target, dtype=float)
        now = int(getattr(self, '_last_step_index', 0))
        self._blocked_targets = [r for r in getattr(self, '_blocked_targets', []) if r['expires'] >= now]
        pose = np.asarray(self.mapper.current_position)
        if not recovery:
            for row in self._blocked_targets:
                rad = float(row.get('radius', .35))
                if (np.linalg.norm(pose[:2] - np.asarray(row['origin'])[:2]) < 1.2
                        and np.linalg.norm(target[:2] - np.asarray(row['target'])[:2]) < rad):
                    return None
        start = self.sim_wrapper.sim.get_agent(0).get_state().position
        route = self.waypoint_routes.plan(start, target)
        if route is None:
            return None
        snapped_target = np.asarray(route['target'], dtype=float)
        if not recovery:
            for row in self._blocked_targets:
                rad = float(row.get('radius', .35))
                if (np.linalg.norm(pose[:2] - np.asarray(row['origin'])[:2]) < 1.2
                        and np.linalg.norm(snapped_target[:2] - np.asarray(row['target'])[:2]) < rad):
                    return None
        stairs = self._ensure_stair_progress()
        if not (recovery or self.rolling_back) and not stairs.route_allowed(route['path']):
            return None
        if (not (recovery or self.rolling_back) and hasattr(self, '_turn_route_allowed')
                and not self._turn_route_allowed(route['path'])):
            return None
        return route

    def _ensure_stair_progress(self):
        """Replay measured stair phases independently of semantic completion.

        Reaching a supported landing may select the next instructed stair
        direction. It never satisfies a route/doorway condition or completes
        the subtask. Each new flight starts at the previous flight's landing.
        """
        key, st = self.instruction_obj.get_current_subtask()
        conditions = (st or {}).setdefault('completion_conditions', completion_conditions((st or {}).get('instruction', ''), key or 'P1'))
        stair_clauses = stair_condition_intents(conditions)
        records = [r for r in (st or {}).get('record', []) if 'height_delta' in r
                   and len(r.get('pose', [])) == 3 and len(r.get('pose_before', [])) == 3
                   and not r.get('recovery') and not r.get('rejected_by_rollback')]
        ground = np.asarray(self.mapper.current_position) + [0, 0, self.mapper.initial_floor_height]
        # This signature changes after an actual action or rollback, not on each
        # candidate-route query. Repeated queries cannot invent stable samples.
        signature = (key, tuple((c['id'], c.get('text', '')) for c, _ in stair_clauses),
                     tuple((id(r), r.get('step')) for r in records))
        state = getattr(self, '_stair_progress', None)
        if (state is not None and state.subtask_id == key
                and getattr(state, '_replay_signature', None) == signature):
            if st is not None:
                st['stair_progress'] = state.packet()
            return state

        # A verified previous segment can physically end on this segment's
        # stair landing. Seed only its short, contiguous measured handoff tail.
        handoff = []
        keys = getattr(self.instruction_obj, 'sub_instruction_keys', [])
        index = getattr(self.instruction_obj, 'current_subtask_index', 0)
        if (stair_clauses and index > 0 and index < len(keys)
                and ((st or {}).get('position_anchor_check') or {}).get('status') == 'aligned'):
            previous = self.instruction_obj.get_subtask_by_key(keys[index-1]) or {}
            if previous.get('completed') and previous.get('completion_evidence_reliable'):
                distance = 0.
                for record in reversed(previous.get('record', [])):
                    if ('height_delta' not in record or len(record.get('pose_before', [])) != 3
                            or len(record.get('pose', [])) != 3 or record.get('recovery')
                            or record.get('rejected_by_rollback')):
                        continue
                    distance += float(record.get('motion_distance', 0.))
                    if len(handoff) >= 6 or distance > 1.5:
                        break
                    handoff.append(record)
                handoff.reverse()
                first_pose = np.asarray(records[0]['pose_before']) if records else ground
                if handoff and np.linalg.norm(np.asarray(handoff[-1]['pose']) - first_pose) > .15:
                    handoff = []
        replay = handoff + records
        start_height = float(replay[0]['pose_before'][2]) if replay else float(ground[2])
        clause_index = 0
        condition_id, intent = ((stair_clauses[0][0]['id'], stair_clauses[0][1]) if stair_clauses else
                                (None, {'direction': 0, 'ambiguous': False, 'landing_required': False}))
        state = StairProgress(key, condition_id, intent, start_height, self.navigation_cfg)
        state.handoff_steps = [r['step'] for r in handoff]
        milestones = []
        for idx_rec, record in enumerate(replay):
            state.observe(record, bool(record.get('landing_supported')))
            # The relaxed landing configuration must not switch the controller
            # onto a later flight after just one vertical movement.
            landed = bool(state.direction and not state.violation and not state.ambiguous
                          and (state.phase == 'landing' if state.landing_required else state.completion_ready()))
            if (not landed and state.direction and not state.violation and not state.ambiguous
                    and clause_index + 1 < len(stair_clauses)
                    and stair_clauses[clause_index + 1][1].get('direction') == state.direction
                    and state.direction * (float(record['pose'][2]) - state.start_height) >= 0.35
                    and abs(float(record['height_delta'])) <= 0.04):
                rem_signed = sum(
                    state.direction * float(r.get('height_delta', 0.0))
                    for r in replay[idx_rec + 1:]
                    if r.get('action_type') == 'move_forward'
                )
                if rem_signed >= float(self.navigation_cfg.get('stair_min_progress_m', 0.25)):
                    landed = True
            if landed and not any(m['condition_id'] == state.condition_id for m in milestones):
                milestones.append({'condition_id': state.condition_id, 'step': record['step'],
                                   'ground_height': float(record['pose'][2])})
                if clause_index + 1 < len(stair_clauses):
                    clause_index += 1
                    condition, intent = stair_clauses[clause_index]
                    state = StairProgress(key, condition['id'], intent,
                                          float(record['pose'][2]), self.navigation_cfg)
        state._replay_signature = signature
        self._stair_progress = state
        if st is not None:
            st['stair_progress'] = state.packet()
            st['stair_execution_milestones'] = milestones
        return state

    def _landing_support_observed(self):
        if self.navigation_cfg.get('habitat_control_enabled', False):
            return self._habitat_measurements()['platform']['valid'] is True
        cloud = getattr(self.mapper, 'navigable_pcd', None)
        if cloud is None or cloud.is_empty():
            return False
        points = cloud.point.positions.cpu().numpy()
        ground = self.mapper.current_position + [0, 0, self.mapper.initial_floor_height]
        return landing_support(points, ground, minimum_area=float(self.navigation_cfg.get('stair_landing_area_m2', .30)))

    def _format_recent_action_history(self, max_groups=None):
        """Grouped log of recent decisions: what was requested and what physically happened."""
        if max_groups is None:
            max_groups = int(self.navigation_cfg.get('planner_history_actions', 8))
        ledger = [r for r in getattr(self, '_execution_ledger', []) if not r.get('recovery')]
        if max_groups <= 0 or not ledger:
            return ''
        groups = []
        for r in ledger:
            req = r.get('requested_action')
            sig = (json.dumps(req, sort_keys=True, default=str), r.get('subtask_id'))
            if groups and groups[-1]['sig'] == sig and r.get('action_type') != 'null':
                g = groups[-1]
            elif groups and r.get('action_type') == 'null' and groups[-1]['sig'] == sig:
                g = groups[-1]
            else:
                g = {'sig': sig, 'req': req, 'sub': r.get('subtask_id'), 'first': r.get('step'),
                     'dist': 0.0, 'yaw': 0.0, 'dz': 0.0, 'blocked': 0}
                groups.append(g)
            g['last'] = r.get('step')
            g['dist'] += float(r.get('motion_distance') or 0.0)
            g['yaw'] += float(r.get('yaw_delta') or 0.0)
            g['dz'] += float(r.get('height_delta') or 0.0)
            g['blocked'] += int(bool(r.get('blocked')))
            g['pose'] = r.get('pose')
        lines = []
        for g in groups[-max_groups:]:
            req = g['req'] if g['req'] else 'unspecified'
            if isinstance(req, dict):
                req = ' '.join(str(v) for v in req.values())
            pose = g.get('pose') or [0, 0, 0]
            lines.append(
                f"- [{g['sub']}] steps {g['first']}-{g['last']}: asked '{req}' -> moved {g['dist']:.1f} m, "
                f"turned {math.degrees(g['yaw']):+.0f} deg, height {g['dz']:+.2f} m"
                + (f", BLOCKED x{g['blocked']}" if g['blocked'] else '')
                + f"; ended at ({pose[0]:.1f}, {pose[1]:.1f}, {pose[2]:.1f}) relative to start")
        return '\n'.join(lines)

    def _execution_feedback(self):
        records = getattr(self, '_execution_ledger', [])[-6:]
        keys = ('step', 'action_type', 'requested_action', 'requested_distance', 'requested_turn_radians',
                'motion_distance', 'height_delta', 'yaw_delta', 'pose_before', 'pose', 'blocked')
        active = getattr(self, '_waypoint_execution', None)
        return {'records': [{k: r[k] for k in keys if k in r} for r in records],
                'waypoint_execution': active.packet() if active else None,
                'stair_progress': self._ensure_stair_progress().packet(),
                'observation_turns_without_motion': getattr(self, '_observation_turns_without_motion', 0)}

    def _invalidate_blocked_path(self, reason):
        if getattr(self, '_pending_recovery', None) is not None:
            self._cancel_checkpoint_recovery(reason)
        target = getattr(self, 'target_world_position', None)
        if target is not None:
            self._blocked_targets = (getattr(self, '_blocked_targets', []) + [{
                'origin': np.asarray(self.mapper.current_position).tolist(),
                'target': np.asarray(target).tolist(),
                'expires': int(getattr(self, '_last_step_index', 0)) + 12}])[-24:]
        self.current_path, self.action_sequence = [], []
        self._need_restore_heading = False
        self._executing_verifier_correction = False
        active = getattr(self, '_waypoint_execution', None)
        if active is not None:
            active.phase, active.reason = 'blocked', reason
        self.navigation_trace.append({'role': 'controller', 'event': 'blocked_path_invalidated',
                                      'reason': reason, 'step': getattr(self, '_last_step_index', 0)})

    def _sim_forward_executable(self, start_hab, rotation, length, expected_hab):
        """Simulate the exact 10-substep try_step_no_sliding executed by SimWrapper._move_forward."""
        try:
            local_pt = np.array([0.0, 0.0, -float(length)], dtype=float)
            global_pt = local_to_global(start_hab, rotation, local_pt)
            step_delta = (global_pt - start_hab) / 10.0
            pos = np.array(start_hab, dtype=float, copy=True)
            pf = self.sim_wrapper.sim.pathfinder
            allow_slide = bool(getattr(self.sim_wrapper, 'allow_slide', False))
            is_slope = abs(float(expected_hab[1]) - float(start_hab[1])) > 0.08 or abs(float(pos[1]) - float(start_hab[1])) > 0.08
            for _ in range(10):
                target_sub = pos + step_delta
                if allow_slide:
                    pos = np.asarray(pf.try_step(pos, target_sub), dtype=float)
                else:
                    step_res = np.asarray(pf.try_step_no_sliding(pos, target_sub), dtype=float)
                    sub_len = float(np.linalg.norm(step_delta[[0, 2]]))
                    moved_no_slide = float(np.linalg.norm((step_res - pos)[[0, 2]]))
                    if hasattr(pf, 'try_step') and (
                        (is_slope and np.linalg.norm((step_res - target_sub)[[0, 2]]) > 0.08)
                        or (sub_len > 1e-4 and moved_no_slide < 0.35 * sub_len)
                    ):
                        slide_res = np.asarray(pf.try_step(pos, target_sub), dtype=float)
                        moved_slide = float(np.linalg.norm((slide_res - pos)[[0, 2]]))
                        lat_err = float(np.linalg.norm((slide_res - target_sub)[[0, 2]]))
                        if is_slope or (moved_slide >= 0.5 * sub_len and lat_err <= max(0.06, 0.8 * sub_len)):
                            step_res = slide_res
                    pos = step_res
            moved_xz = float(np.linalg.norm((pos - start_hab)[[0, 2]]))
            height_tol = max(float(self.navigation_cfg.get('route_height_tolerance_m', .25)), 0.45 if is_slope else 0.25)
            min_ratio = 0.45 if is_slope else 0.72
            return bool(moved_xz >= max(.003, min_ratio * float(length))
                        and abs(pos[1] - float(expected_hab[1])) <= height_tol)
        except Exception:
            return False

    def _get_action_for_next_waypoint(self) -> Optional[PolarAction]:
        """Follow certified corners, checking the actual forward command again."""
        agent_pos = np.asarray(self.mapper.current_position)
        ground = agent_pos + [0, 0, self.mapper.initial_floor_height]
        arrival = float(self.navigation_cfg.get('path_arrival_tolerance_m', .08))
        agent_state = self.sim_wrapper.sim.get_agent(0).get_state()
        start = np.asarray(agent_state.position, dtype=float)
        while self.current_path:
            point = np.asarray(self.current_path[0], dtype=float)
            dist = float(np.linalg.norm(point[:2]-ground[:2]))
            height_tol = float(self.navigation_cfg.get('route_height_tolerance_m', .25))
            if dist <= arrival and abs(point[2]-ground[2]) <= height_tol:
                if len(self.current_path) > 1 and dist > .005:
                    next_hab = self.waypoint_routes.to_habitat(self.current_path[1])
                    if not self.waypoint_routes.segment(start, next_hab):
                        break
                self.current_path.pop(0)
            else:
                break
        if not self.current_path:
            active = getattr(self, '_waypoint_execution', None)
            if active is not None and active.phase != 'blocked':
                active.phase = 'arrived'
            return None
        point = np.asarray(self.current_path[0], dtype=float)
        if (not getattr(self, 'rolling_back', False) and hasattr(self, '_turn_route_allowed')
                and not self._turn_route_allowed(self.current_path)):
            self._invalidate_blocked_path('first_segment_opposes_instructed_turn')
            return PolarAction.null
        delta = point[:2] - ground[:2]
        distance = float(np.linalg.norm(delta))
        if distance < .003:
            self._invalidate_blocked_path('height_mismatch_at_path_corner')
            return PolarAction.null
        forward = -np.asarray(self.mapper.current_rotation)[:2, 2]
        forward /= max(float(np.linalg.norm(forward)), 1e-8)
        angle = self._normalize_angle(math.atan2(delta[1], delta[0])-math.atan2(forward[1], forward[0]))
        if abs(angle) > math.radians(3):
            active = getattr(self, '_waypoint_execution', None)
            if active is not None:
                active.phase = 'aligning'
            return PolarAction(r=0, theta=-angle, type='turn')
        max_step = float(self.navigation_cfg.get('forward_step_m', .5))
        # Conservative fallback if clearance query is unavailable.
        clearance = 0.0
        try:
            clearance = float(self.sim_wrapper.sim.pathfinder.distance_to_closest_obstacle(start, .6))
        except Exception:
            pass
        on_slope = abs(point[2]-ground[2]) > .08
        if on_slope or clearance < float(self.sim_wrapper.agent_radius) + .10:
            max_step = min(max_step, float(self.navigation_cfg.get('precision_step_m', .35)))
        if (self.navigation_cfg.get('arrival_check_enabled', False)
                and not getattr(self, 'rolling_back', False)
                and self._is_near_endpoint_or_threshold(at_route_boundary=False)):
            max_step = min(max_step, float(self.navigation_cfg.get('arrival_precision_step_m', .2)))
        length = min(distance, max_step)
        # Test the heading actually executed, not an idealized segment to the
        # waypoint. Try shorter steps before giving up a narrow passage.
        for _ in range(3):
            expected = ground.copy()
            expected[:2] += forward*length
            expected[2] += (point[2]-ground[2])*min(1.0, length/distance)
            end = self.waypoint_routes.to_habitat(expected)
            if on_slope:
                try:
                    snapped_end = np.asarray(self.sim_wrapper.sim.pathfinder.snap_point(end), dtype=float)
                    if snapped_end.shape == (3,) and np.isfinite(snapped_end).all():
                        if np.linalg.norm((snapped_end - end)[[0, 2]]) <= 0.15:
                            end[1] = snapped_end[1]
                except Exception:
                    pass
            if (self.waypoint_routes.segment(start, end)
                    and (not hasattr(self, '_sim_forward_executable')
                         or self._sim_forward_executable(start, agent_state.rotation, length, end))):
                active = getattr(self, '_waypoint_execution', None)
                if active is not None:
                    active.phase = 'moving'
                return PolarAction(r=length, theta=0, type='move_forward')
            length *= .5
        # If slight heading residual (0.1..3 deg) caused the ray to clip a nearby edge, align exactly first.
        if abs(angle) > math.radians(0.1):
            active = getattr(self, '_waypoint_execution', None)
            if active is not None:
                active.phase = 'aligning'
            return PolarAction(r=0, theta=-angle, type='turn')
        self._invalidate_blocked_path('forward_segment_not_executable')
        return PolarAction.null

    def execute_action(self, action: PolarAction):
        before = np.asarray(self.sim_wrapper.sim.get_agent(0).get_state().position, dtype=float).copy()
        ground_before = (self.mapper.current_position + [0, 0, self.mapper.initial_floor_height]).copy()
        stairs = self._ensure_stair_progress()
        yaw_before = checkpoint_yaw(self.mapper.current_rotation)
        key = self.instruction_obj.get_current_subtask_key()
        raw_obs_id_before = getattr(self.mapper, 'observation_id', None)
        obs_id_before = (
            int(raw_obs_id_before)
            if isinstance(raw_obs_id_before, (int, np.integer))
            and not isinstance(raw_obs_id_before, bool)
            and int(raw_obs_id_before) >= 0
            else None
        )
        pending_recovery = getattr(self, '_pending_recovery', None)
        recovery_source = (
            pending_recovery.get('recovery_source')
            if isinstance(pending_recovery, dict)
            else None
        )
        if isinstance(recovery_source, str):
            recovery_source = recovery_source.strip() or None
        else:
            recovery_source = None
        before_snapshot = (
            self._snapshot_verification_views(self.curr_obs)
            if hasattr(self, '_snapshot_verification_views')
            else None
        )
        obs = self.sim_wrapper.step(action)
        after = np.asarray(obs['agent_state'].position, dtype=float).copy()
        instances = self.update_map(obs)
        raw_obs_id_after = getattr(self.mapper, 'observation_id', None)
        obs_id_after = (
            int(raw_obs_id_after)
            if isinstance(raw_obs_id_after, (int, np.integer))
            and not isinstance(raw_obs_id_after, bool)
            and int(raw_obs_id_after) >= 0
            else None
        )
        self.curr_obs, self.instances, self.last_action = obs, instances, action
        self._verification_obs = None
        distance = float(np.linalg.norm(after-before))
        self.traveled_distance += distance
        yaw_delta = self._normalize_angle(checkpoint_yaw(self.mapper.current_rotation)-yaw_before)
        record = {'step': len(self._execution_ledger), 'controller_step': getattr(self, '_last_step_index', 0),
                  'subtask_id': key, 'executed_action': self._action_to_text(action),
                  'action_type': action.type, 'motion_distance': distance, 'yaw_delta': yaw_delta,
                  'yaw_before': float(yaw_before),
                  'yaw_after': float(checkpoint_yaw(self.mapper.current_rotation)),
                  'requested_distance': float(action.r), 'requested_turn_radians': float(action.theta),
                  'requested_action': getattr(self, '_requested_action', None),
                  'observation_id_before': obs_id_before,
                  'observation_id_after': obs_id_after,
                  'pose_before': ground_before.tolist(), 'height_delta': float(after[1]-before[1]),
                  'blocked': bool(action.type == 'move_forward' and action.r >= .08 and distance < .02),
                  'pose': (self.mapper.current_position + [0, 0, self.mapper.initial_floor_height]).tolist(),
                  'recovery': bool(self.rolling_back),
                  'recovery_source': recovery_source}
        self._execution_ledger.append(record)
        self.instruction_obj.add_record_to_current_subtask(record)
        active = getattr(self, '_waypoint_execution', None)
        if active is not None and active.phase in ('aligning', 'moving', 'arrived'):
            record['requested_action'] = {'type': 'navigate', 'waypoint_id': active.action_id}
            issue = active.observe(record)
            record['waypoint_execution'] = active.packet()
            if issue:
                self._invalidate_blocked_path(issue)
        if hasattr(self, '_observe_turn_execution'):
            self._observe_turn_execution(record)
        if self.current_path:
            ground_now = np.asarray(self.mapper.current_position) + [0, 0, self.mapper.initial_floor_height]
            arrival = float(self.navigation_cfg.get('path_arrival_tolerance_m', .08))
            height_tol = float(self.navigation_cfg.get('route_height_tolerance_m', .25))
            while self.current_path:
                pt = np.asarray(self.current_path[0], dtype=float)
                dist_pt = float(np.linalg.norm(pt[:2] - ground_now[:2]))
                if dist_pt <= arrival and abs(pt[2] - ground_now[2]) <= height_tol:
                    if len(self.current_path) > 1 and dist_pt > .005:
                        next_hab = self.waypoint_routes.to_habitat(self.current_path[1])
                        if not self.waypoint_routes.segment(after, next_hab):
                            break
                    self.current_path.pop(0)
                else:
                    break
            if not self.current_path and active is not None and active.phase != 'blocked':
                active.phase = 'arrived'
        if not record['recovery']:
            record['landing_supported'] = bool(stairs.direction and abs(record['height_delta']) <= .04
                                               and self._landing_support_observed())
            issue = stairs.observe(record, record['landing_supported'])
            record['stair_progress'] = stairs.packet()
            if issue:
                self._invalidate_blocked_path(issue)
        if hasattr(self, '_cache_verification_frames'):
            self._cache_verification_frames(record, before_snapshot, obs)
        if distance >= .02:
            self._physical_progress_step = getattr(self, '_last_step_index', 0)
            self._observation_turns_without_motion = 0
        elif action.type == 'turn' and (active is None or active.phase in ('arrived', 'blocked')):
            self._observation_turns_without_motion = getattr(self, '_observation_turns_without_motion', 0) + 1
        if active is None and action.type == 'move_forward' and float(action.r) >= .08 and distance < .02:
            self._blocked_move_count += 1
            if self._blocked_move_count >= 2:
                self._invalidate_blocked_path('two_forward_commands_without_motion')
        else:
            self._blocked_move_count = 0
        if getattr(self, '_pending_recovery', None) is not None:
            self._observe_recovery_action(action)

    def create_action(self, world_point_1, world_point_2):
        """
        Create a PolarAction from world_point_1 to world_point_2.
        :param world_point_1: starting world coordinate.
        :param world_point_2: target world coordinate.
        :return: a PolarAction object.
        """

        delta = np.array(world_point_2) - np.array(world_point_1)
        r = np.linalg.norm([delta[0], delta[2]])
        theta = np.arctan2(delta[0], -delta[2])
        return PolarAction(r, theta)

    @abstractmethod
    def _get_vlm_response(self, front_rgb_image: PIL.Image.Image, map_image: PIL.Image.Image, prompt: str, img_buffer: list) -> str:
        """
        Call VLM API to get response for image and prompt.
        Subclasses must implement to call a specific VLM (e.g., GPT-4V).

        :param front_rgb_image:
        :param map_image:
        :param prompt:
        :param img_buffer:
        :return: VLM text response.
        """
        raise NotImplementedError("Subclasses must implement _get_vlm_response.")

    @abstractmethod
    def _get_vlm_response_multiview(self, front_rgb_image: PIL.Image.Image, left_rgb_image: PIL.Image.Image, right_rgb_image: PIL.Image.Image, back_rgb_image: PIL.Image.Image, map_image: PIL.Image.Image, prompt: str, img_buffer: list, pil_panorama_image: Optional[PIL.Image.Image] = None) -> str:
        """
        Call VLM API with multi-view images and prompt.
        Subclasses must implement to call a specific VLM (e.g., GPT-4V).

        :param front_rgb_image:
        :param left_rgb_image:
        :param right_rgb_image:
        :param back_rgb_image:
        :param map_image:
        :param prompt:
        :param img_buffer:
        :param pil_panorama_image: Optional labeled panorama replacing independent views.
        :return: VLM text response.
        """
        raise NotImplementedError("Subclasses must implement _get_vlm_response_multiview.")

    @abstractmethod
    def _get_llm_response(self, prompt: str) -> str:
        """
        Call text LLM API for prompt response.
        Subclasses must implement to call a specific LLM.

        :param prompt:
        :return: LLM text response.
        """
        raise NotImplementedError("Subclasses must implement _get_llm_response.")

    def generate_prompt(self, instruction: str = None, is_stuck: bool = False, is_first_step=False) -> str:
        key, st = self.instruction_obj.get_current_subtask()
        clean_full_instr = re.sub(r'\s*[\r\n]+\s*', '. ', (instruction or self.instruction or '').strip())
        clean_sub_instr = re.sub(r'\s*[\r\n]+\s*', '. ', str((st or {}).get('instruction', '')).strip())
        prompt = f"""Navigate according to the source instruction: {clean_full_instr}
Current subtask {key} (complete ALL clauses in sequence before proposing completion): {clean_sub_instr}
Current image panels: FRONT (0 deg), LEFT (-90 deg), RIGHT (+90 deg), BACK (180 deg). Map is heading-up.
WP_01-style labels identify movement destinations in all 4 views; TURN_* labels identify rotations in place.
IMPORTANT: Because you already see all 4 directions (FRONT, LEFT, RIGHT, BACK) and candidate waypoints (WP_XX) are projected in all 4 views, selecting a WP_XX in the LEFT, RIGHT, or BACK view will AUTOMATICALLY turn you toward that waypoint and walk there. When an instruction says "turn left/right and walk...", directly select the appropriate WP_XX in that view instead of wasting a step on TURN_* first.
EGOCENTRIC DIRECTION RULE: Directional terms ("on the right", "on the left", "turn left/right") in sequential clauses ALWAYS refer to your ego frame AFTER completing the preceding turn or movement. In particular, when an instruction says "Turn around and go through the door on the right", ONCE you have executed the 180° turn (TURN_BACK), "the door on the right" IS in your current RIGHT view (+90 deg) — do NOT invert it to LEFT thinking about your pre-turn orientation!
Map direction markers FRONT/BACK/LEFT/RIGHT and object labels are not selectable actions. Do NOT mistake furniture/rugs/windows near the START label (inside the starting room) for the final destination when the instruction requires exiting the starting room or passing intermediate landmarks (e.g. passing a couch or walking down a hall) first.
Use the current legal action objects, never invent a forward action or reuse an old waypoint ID.
The controller finishes alignment and route following before asking you for another destination.
Follow all clauses of the instruction in order, including doorway sides, ordinals and stair direction. Never call propose_completion after only the first clause if later clauses remain!
A nearby waypoint, furniture, stair label or a 1m range arc does not establish completion.
For stairs confirm the requested flight/landing and actual height trend; a flat intermediate landing is not automatically the endpoint.
For doorways distinguish approach, crossing, and the requested stopping side using executed before/after poses and current RGB.
DEPTH-GROUNDED STOPPING RULE: Before selecting propose_completion, check the "4-View Depth & Waypoint Distances" section below!
- For SOLID landmarks (window, table, desk, bed, chair, rug, or far end of a counter/bar), do NOT propose_completion when the final target landmark is still >= 1.5m away in camera depth with unvisited waypoints WP_XX leading closer to it; walk right up to the final target landmark (within ~0.5m-1.2m).
- For OPENINGS ("stop in the doorway", "stop in the archway", "wait at the door"), once you have crossed the specified hall/room and stepped into that doorway/archway opening, STOP IMMEDIATELY in that threshold! Remember that when standing *inside* an open doorway or archway, your FRONT view looks *through* the opening into the next corridor/room (> 1.5m depth, which may show another archway/doorway further ahead), while the doorway/archway frame you are standing in flanks your LEFT/RIGHT views and the room you just crossed is in your BACK view — do NOT keep walking through the target archway into subsequent rooms/archways beyond it!
- For room-entry endpoints (e.g. "enter the dining room", "go into the hallway" without a specific furniture stop target inside), propose_completion as soon as you step 1.0m-1.5m across the room entrance threshold rather than walking to the far wall.
DEAD-END / NARROW AISLE RULE: Never rotate in place (TURN_*) more than once at the same position! If your forward aisle between furniture is too narrow to spawn unvisited waypoints and only visited (red) waypoints exist, IMMEDIATELY select a visited (red) waypoint to step back and go around the wider side of the furniture.
Historical memories and your own earlier prose are advisory. Measured execution feedback below is authoritative.
Return JSON with a short observation, plan and action object. Do not generate a movement history; the executor records it.
Subtask state: {self.instruction_obj.get_all_subtasks_str()}
Current mapper camera position (meters, Z up): {np.asarray(self.mapper.current_position).round(2).tolist()}
""".strip()
        if is_stuck:
            prompt += '\nRecent actions did not produce progress. Do NOT select TURN_* again; select a waypoint (including a visited/red waypoint if needed) to move out and bypass the obstacle.'
        return prompt

    def decompose_instruction(self, instruction: str) -> List[str]:
        """
        Call LLM to decompose a complex instruction into subgoals. Parses JSON {"subgoals": [...]} first;
        falls back to heuristic splitting on failure. Returns list of subgoal strings.
        """
        prompt = """
        Here is a visual language navigation instruction. The agent only needs to move to the designated position according to the instruction and does not operate any object.
        Decompose the following navigation instruction. Break down a complex instruction into multiple subtasks in sequence.
        The decomposition principle requires ensuring each subtask has a clear completion condition(end position); otherwise, it cannot be an independent subtask and you can combine multiple subtasks into one subtask if necessary.
        Each subtask must contain at least 3 objects or places to ensure clarity. Under the above condition, each sub-task should be as small as possible.
        **You can only split and merge based on the original instructions, and must not alter the original expression.** Do not add any additional explanations.
        For example, only "walk through the hallway" lacks a clear end position and should therefore be merged with subsequent subtasks until it's clear. But a subtask cannot contain more than three actions.
        Directly output in JSON format without markdown code blocks and without preamble. Return JSON in the exact form: {\"analysis\": \"\", \"subtasks\": [\"...\", \"...\", ...]}\n\n
        SELF-CHECK BEFORE OUTPUT:
        - For every subtask, verify it is an exact substring of the original instruction.
        - Count objects/places (must be >= 2).
        - Count actions (must be <= 3).
        - If a subtask violates rules, MERGE it with adjacent text until it satisfies all rules.

        Keep thinking concise (under 100 words) inside <think>. Directly output the JSON without markdown code blocks.

        Instruction: {instruction}\n\n
        """

        prompt = prompt.replace("{instruction}", instruction)

        try:
            raw = self._get_llm_response(prompt)
        except Exception as e:
            raw = ""
        subs: List[str] = []

        try:
            m = re.search(r'\{.*\}', raw, flags=re.S)
            candidate = m.group(0) if m else raw
            data = json.loads(candidate)
            if isinstance(data, dict) and "subtasks" in data and isinstance(data["subtasks"], list):
                for s in data["subtasks"]:
                    if not isinstance(s, str):
                        continue

                    s_clean = re.sub(r'^\s*\d+\s*[\)\.\-:]*\s*', '', s).strip()
                    if s_clean:
                        subs.append(s_clean)
        except Exception:
            subs = []

        if not subs:
            parts = re.split(r'(?:then|and then|after that|;|\n|,|\.)+', instruction, flags=re.I)
            candidates = [p.strip() for p in parts if p and p.strip()]

            merged: List[str] = []
            for p in candidates:
                if len(p) < 6 and merged:
                    merged[-1] = (merged[-1] + ' ' + p).strip()
                else:
                    merged.append(p)
            subs = merged if merged else [instruction.strip()]

        if not subs:
            subs = [instruction.strip()]

        subs = partition_instruction(instruction, subs)
        print(subs)
        return subs

    def close(self):
        """Close and release GPU memory/large object refs; safe to call repeatedly."""
        if getattr(self, "_ended", False):
            return
        if hasattr(self, 'finalize_navigation_memory'):
            self.finalize_navigation_memory()

        try:
            if hasattr(self, "mapper") and self.mapper is not None:

                try:
                    release = getattr(self.mapper, "release_all_vram", None)
                    if callable(release):
                        release()
                except Exception:
                    pass
                finally:

                    self.mapper = None
        except Exception:
            pass

        for attr in ("curr_obs", "instances", "last_actions", "img_buffer"):
            try:
                setattr(self, attr, None)
            except Exception:
                pass

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            import gc
            gc.collect()
        except Exception:
            pass

        self._ended = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

import base64
from io import BytesIO
from openai import AzureOpenAI, OpenAI

class GPTAgent(PathPlannerAgent):
    """
    VLM agent implementation using OpenAI / Azure OpenAI compatible API.
    Configured via environment variables:
      - OPENAI_API_KEY   API key
      - OPENAI_BASE_URL  (optional) custom base_url
      - AZURE_OPENAI_ENDPOINT   Azure endpoint
      - AZURE_OPENAI_API_KEY   Azure API key
      - AZURE_OPENAI_API_VERSION  Azure API version
    """

    def __init__(self, sim_wrapper, config, instruction, initial_position=None, initial_rotation=None, model_name=None, mode='vln'):
        self.model_name = model_name or os.environ.get("OPENAI_MODEL", "gpt-5")

        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

        if azure_endpoint and azure_api_key:
            self.client = AzureOpenAI(api_version=azure_api_version, azure_endpoint=azure_endpoint, api_key=azure_api_key, max_retries=0)
        else:
            import httpx
            timeout_sec = float(os.environ.get("HSGM_HTTPX_TIMEOUT", 360))
            # Direct local network connection: isolate from proxy environment
            direct_http_client = httpx.Client(trust_env=False, timeout=httpx.Timeout(timeout_sec))
            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                base_url=os.environ.get("OPENAI_BASE_URL"),
                max_retries=0,
                http_client=direct_http_client,
            )
        sys_prompt = """

        """
        self.history_msgs = [
            {"role": "system", "content": "You are an agent good at navigating in an indoor environment."}
        ]
        super().__init__(sim_wrapper, config, instruction, initial_position, initial_rotation, mode)

    def _pil_to_base64(self, pil_img):
        if not isinstance(pil_img, Image.Image):
            pil_img = Image.fromarray(np.asarray(pil_img, dtype=np.uint8))
        max_w_env = os.environ.get("HSGM_VIEW_MAX_WIDTH")
        max_w = int(max_w_env) if max_w_env else 672
        if max_w > 0 and pil_img.width > max_w:
            h = max(1, int(round(pil_img.height * max_w / pil_img.width)))
            pil_img = pil_img.convert('RGB').resize((max_w, h), Image.BILINEAR)
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def _pil_to_base64_small(self, pil_img, max_width=448):
        """JPEG base64 downscaled to max_width (history frames only)."""
        if not isinstance(pil_img, Image.Image):
            pil_img = Image.fromarray(np.asarray(pil_img, dtype=np.uint8))
        if pil_img.width > max_width:
            h = max(1, int(round(pil_img.height * max_width / pil_img.width)))
            pil_img = pil_img.convert('RGB').resize((max_width, h), Image.BILINEAR)
        buffered = BytesIO()
        pil_img.convert('RGB').save(buffered, format="JPEG", quality=80)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _select_history_keyframes(self, img_buffer, max_frames=4):
        """Select up to max_frames evenly spaced keyframes across the episode (oldest to newest), quantizing intermediate indices for cache stability."""
        if not img_buffer or max_frames <= 0:
            return []
        valid = [im for im in img_buffer if im is not None]
        if len(valid) <= max_frames:
            return valid
        if max_frames == 1:
            return [valid[0]]
        n = len(valid)
        anchor_last = max(max_frames - 1, ((n - 1) // 2) * 2)
        indices = [0]
        for i in range(1, max_frames - 1):
            idx = min(n - 2, max(1, int(round(i * anchor_last / (max_frames - 1)))))
            indices.append(idx)
        indices.append(n - 1)
        indices = sorted(set(indices))
        return [valid[idx] for idx in indices]

    def _commit_planner_history(self, parsed_json):
        """Commit only after the caller decoded an action from current candidates.

        Requests stage the current observation, never the unvalidated assistant
        response. Canonical JSON also excludes any surrounding invalid prose.
        """
        pending = getattr(self, '_pending_planner_history', None)
        if pending is not None:
            self.history_msgs.extend([pending, {
                'role': 'assistant', 'content': json.dumps(parsed_json, ensure_ascii=False)}])
        self._pending_planner_history = None

    def _get_vlm_response(self, rgb_image: Image.Image, map_image: Image.Image, prompt: str, img_buffer=None) -> str:
        self._pending_planner_history = None
        if img_buffer is None:
            img_buffer = []
        try:
            rgb_img_b64 = self._pil_to_base64(rgb_image)
            map_img_b64 = self._pil_to_base64(map_image)

            content_for_request = [{"type": "text", "text": prompt}]
            for img in img_buffer[-5:]:
                img_b64 = self._pil_to_base64(img)
                content_for_request.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "auto"}
                })
            content_for_request.extend([
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{rgb_img_b64}", "detail": "high"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{map_img_b64}", "detail": "high"}}
            ])
            current_msg_for_api = {"role": "user", "content": content_for_request}

            content_for_history = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{rgb_img_b64}", "detail": "auto"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{map_img_b64}", "detail": "auto"}},
            ]
            current_msg_for_history = {"role": "user", "content": content_for_history}

            if not hasattr(self, "history_msgs"):
                self.history_msgs = [
                    {"role": "system", "content": "You are an agent good at navigating in an indoor environment."}
                ]

            messages_for_api = self.history_msgs + [current_msg_for_api]

            user_msgs = [msg for msg in messages_for_api if msg["role"] == "user"]
            if len(user_msgs) > 7:
                user_count = 0
                for msg in messages_for_api:
                    if msg["role"] == "user":
                        user_count += 1
                        if user_count <= len(user_msgs) - 7:
                            if isinstance(msg["content"], list):
                                msg["content"] = [c for c in msg["content"] if c["type"] == "text"]

            max_turn = 30
            if len(messages_for_api) > max_turn:
                messages_for_api = messages_for_api[-max_turn:]

                if messages_for_api[0]["role"] != "system":
                    messages_for_api.insert(0, {"role": "system",
                                                "content": "You are an agent good at navigating in an indoor environment."})

            extra_kwargs = {}
            max_out = os.environ.get("HSGM_MAX_OUTPUT_TOKENS")
            extra_kwargs["max_tokens"] = int(max_out) if max_out else 4096

            response = self._create_chat_completion_with_backoff(
                operation="planner_request", model=self.model_name,
                messages=messages_for_api, **extra_kwargs)

            result = response.choices[0].message.content
            print(result)

            clean_result = result
            if clean_result and isinstance(clean_result, str):
                clean_result = re.sub(r'<think>.*?</think>', '', clean_result, flags=re.DOTALL)
                if '</think>' in clean_result:
                    clean_result = clean_result.split('</think>', 1)[-1]
                clean_result = clean_result.strip()
                if not clean_result:
                    clean_result = result

            self._pending_planner_history = current_msg_for_history
            return result
        except Exception as e:
            print(f"[VLM Exception in _get_vlm_response] {e}")
            logging.warning(f"Error in _get_vlm_response: {e}")
            return "stop"

    def _get_vlm_response_multiview(self, front_rgb_image: PIL.Image.Image, left_rgb_image: PIL.Image.Image,
                                    right_rgb_image: PIL.Image.Image, back_rgb_image: PIL.Image.Image,
                                    map_image: PIL.Image.Image, prompt: str, img_buffer: list,
                                    pil_panorama_image: Optional[PIL.Image.Image] = None) -> str:
        self._pending_planner_history = None
        if img_buffer is None:
            img_buffer = []
        try:

            map_img_b64 = self._pil_to_base64(map_image)
            cfg = getattr(self, 'navigation_cfg', {}) or {}
            content_for_request = [{"type": "text", "text": prompt}]

            # Past first-person keyframes (v4-style visual history), downscaled to bound token cost.
            default_hist_n = int(cfg.get('planner_history_frames', 1))
            default_hist_w = int(cfg.get('planner_history_frame_width', 280))
            hist_n = max(0, int(os.environ.get('HSGM_PLANNER_HISTORY_FRAMES', default_hist_n)))
            hist_w = max(160, int(os.environ.get('HSGM_PLANNER_HISTORY_FRAME_WIDTH', default_hist_w)))
            hist_frames = self._select_history_keyframes(img_buffer, hist_n)
            if hist_frames:
                content_for_request.append({"type": "text", "text": (
                    f"PAST FIRST-PERSON KEYFRAMES ({len(hist_frames)}, oldest to newest across the episode): front-camera frames "
                    "captured while executing your earlier moves; the step, pose, and action taken are printed at top-left.")})
                for img in hist_frames:
                    content_for_request.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{self._pil_to_base64_small(img, hist_w)}",
                                      "detail": "auto"}})

            if pil_panorama_image is not None:
                panorama_img_b64 = self._pil_to_base64(pil_panorama_image)
                current_views = [("360-DEGREE PANORAMA [Left | Front | Right | Back]", panorama_img_b64)]
                history_view_b64 = panorama_img_b64
            else:
                front_img_b64 = self._pil_to_base64(front_rgb_image) if front_rgb_image is not None else None
                left_img_b64 = self._pil_to_base64(left_rgb_image) if left_rgb_image is not None else None
                right_img_b64 = self._pil_to_base64(right_rgb_image) if right_rgb_image is not None else None
                back_img_b64 = self._pil_to_base64(back_rgb_image) if back_rgb_image is not None else None
                current_views = [("FRONT VIEW (0 deg, straight ahead)", front_img_b64),
                                 ("LEFT VIEW (-90 deg)", left_img_b64),
                                 ("RIGHT VIEW (+90 deg)", right_img_b64),
                                 ("BACK VIEW (180 deg)", back_img_b64)]
                history_view_b64 = front_img_b64

            content_for_request.append({"type": "text", "text": "CURRENT OBSERVATION:"})
            for tag, img_b64 in current_views + [("TOP-DOWN MAP (you face upward)", map_img_b64)]:
                if img_b64 is not None:
                    content_for_request.append({"type": "text", "text": f"[{tag}]"})
                    content_for_request.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "high"}
                    })
            if hasattr(self, "_visual_candidate_content") and callable(getattr(self, "_visual_candidate_content")):
                planner_crop_blocks = list(self._visual_candidate_content(for_verifier=False) or [])
                if planner_crop_blocks:
                    kept_crop_blocks = []
                    pending_crop_texts = []
                    crop_img_count = 0
                    emitted_crop_hdr = False
                    for blk in planner_crop_blocks:
                        if not isinstance(blk, dict):
                            continue
                        if blk.get("type") == "text":
                            pending_crop_texts.append(blk)
                        elif blk.get("type") == "image_url":
                            if crop_img_count < 2:
                                if len(pending_crop_texts) > 1 and not emitted_crop_hdr:
                                    kept_crop_blocks.extend(pending_crop_texts[:-1])
                                    emitted_crop_hdr = True
                                if pending_crop_texts:
                                    kept_crop_blocks.append(pending_crop_texts[-1])
                                kept_crop_blocks.append(blk)
                                crop_img_count += 1
                            pending_crop_texts = []
                    content_for_request.extend(kept_crop_blocks)
            current_msg_for_api = {"role": "user", "content": content_for_request}

            keep_img_turns = max(1, int((getattr(self, 'navigation_cfg', {}) or {}).get('planner_history_turns_with_images', os.environ.get('HSGM_PLANNER_HISTORY_IMG_TURNS', 1))))
            if os.environ.get("HSGM_COMPACT_HISTORY", "1") == "1":
                step_idx = getattr(self, "_last_step_index", 0)
                cur_pos = np.asarray(self.mapper.current_position).round(2).tolist() if getattr(self, "mapper", None) is not None else []
                cur_sub_k, cur_sub_st = self.instruction_obj.get_current_subtask() if getattr(self, "instruction_obj", None) is not None else ("", {})
                hist_prompt_text = (
                    f"[Earlier decision turn at step {step_idx}, pos={cur_pos}, subtask {cur_sub_k}: "
                    f"{(cur_sub_st or {}).get('instruction', '')}] "
                    "(Full cumulative action history, working memory, and current legal actions are in the latest prompt.)"
                )
            else:
                hist_prompt_text = prompt
            content_for_history = [{"type": "text", "text": hist_prompt_text}]
            if keep_img_turns > 1:
                for img_b64 in (history_view_b64, map_img_b64):
                    if img_b64 is not None:
                        content_for_history.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "auto"}
                        })
            current_msg_for_history = {"role": "user", "content": content_for_history}

            sys_prompt = """You are an embodied 3D indoor Vision-Language Navigation (VLN-CE) planning agent. Your objective is to follow every sequential clause of the natural-language navigation instruction from the starting pose to the exact instructed stopping location using multi-view RGB observations, metric depth measurements, a top-down Bird's-Eye-View (BEV) semantic map, and certified navigable waypoints.

**1. Multi-Modal Observation & Coordinate Conventions**:
- **Top-Down BEV Map (`[TOP-DOWN MAP (you face upward)]`)**:
  - Ego-centric heading-up orientation: the `FRONT` camera (0 deg) always points straight UPWARD toward the top edge of the map (`FRONT` triangle marker); `BACK` (180 deg) is at the bottom edge; `LEFT` (-90 deg) is at the left edge; `RIGHT` (+90 deg) is at the right edge.
  - `Gray`: Explored navigable floor surface on your current storey level.
  - `Black / Dark Gray`: Impassable walls, closed doors, railings, and physical obstacles.
  - `Blue / Amber Elevation Tint`: Observed surfaces below (`Blue`) or above (`Amber`) your current floor height in 0.25m bands. Elevation tint indicates observed height difference (such as stairs or lower/upper floors), not automatic walkability or proof that a stair traversal is complete.
  - `Red Triangle Arrow`: Your current 3D position and forward heading (pointing upward).
  - `Red Polyline Trail`: Your executed trajectory across the episode.
  - `Blue Circle labeled START`: Your initial episode starting coordinate `(0.0, 0.0)`.
  - `Text Box Labels`: Automatically segmented object categories projected onto the map. Detector labels can be noisy or incomplete; always verify landmarks in the four first-person RGB views.
  - `Yellow Dots`: Frontier boundaries between explored floor and unexplored space.
  - `Orange-Bordered White Circles (WP_01, WP_02, ...)`: Certified reachable waypoint candidates projected on the map.
- **Four First-Person RGB Views (`[FRONT VIEW (0 deg)]`, `[LEFT VIEW (-90 deg)]`, `[RIGHT VIEW (+90 deg)]`, `[BACK VIEW (180 deg)]`)**:
  - Provide complete 360-degree visual coverage around your current position at every step. If a stitched 360-degree panorama is supplied instead, its panels read left-to-right: `Left | Front | Right | Back`.
  - Each view header prints central-patch depth (`center=X.Xm`) and middle-band 85th-percentile depth (`p85=Y.Ym`); neither is a named target distance or a traversable range.
  - `White Numbered Circles (WP_XX)`: Unvisited certified navigable waypoints projected into the view where they are best centered.
  - `Red Numbered Circles (WP_XX)`: Previously visited certified navigable waypoints (within 0.8m of your prior trajectory).
  - `TURN_LEFT`, `TURN_RIGHT`, `TURN_BACK`: In-place rotation actions (+90 deg left, -90 deg right, 180 deg turn-around).
- **Past First-Person Keyframes (when provided)**:
  - Downscaled historical front-camera snapshots showing rooms, doorways, and landmarks you already traversed earlier in the episode.

**2. Direct 360-Degree Waypoint Navigation & Egocentric Turn Rules**:
- **Multi-View Waypoint Selection vs. Explicit Named Landmark Turns**: Because you simultaneously see all 4 cardinal views (`FRONT`, `LEFT`, `RIGHT`, `BACK`) and `WP_XX` candidates are projected across all 4 views, selecting `{"type": "navigate", "waypoint_id": "WP_XX"}` in the `LEFT`, `RIGHT`, or `BACK` view will **automatically rotate the robot toward that waypoint and walk along the certified NavMesh path** for ordinary waypoint alignment. However, ordinary `navigate` alignment is **not** proof of an explicit instruction turn: when an active clause requires a named instruction turn at a reached landmark (or the controller's explicit owned-turn contract is active), execute the explicit `TURN_*` action first to satisfy the owned-turn contract before advancing along the post-turn path.
- **Egocentric Direction Rule After Turning**: Directional phrases (`"on the left"`, `"on the right"`, `"turn left"`, `"turn right"`) in sequential clauses ALWAYS refer to your ego-centric body frame **after** completing the preceding movement or rotation. Specifically, when an instruction begins with *"Turn around and walk through the doorway on the right"*, once you have executed `TURN_BACK` (180 deg), `"on the right"` is in your current `RIGHT VIEW (+90 deg)` — never invert it to `LEFT` by thinking in your pre-turn coordinate frame.
- **Exit-Before-Turn Rule**: When an instruction says *"Walk out of the bedroom/room and turn left/right"*, you must physically step **out through the doorway threshold into the hallway/corridor** before executing the left/right branch. Do not turn inside the starting room before crossing the exit doorway.

**3. Sequential Clause Execution & Anti-Premature-Stop Protocol**:
- **Complete All Clauses in Order**: Decompose the instruction mentally into ordered clauses (Room/Area Exit -> Intermediate Hallway/Landmark/Stair Traversal -> Final Target Approach -> Stopping Relation). Never propose completion after only the first clause if subsequent clauses remain.
- **Starting-Room Disambiguation**: Do not mistake furniture, rugs, beds, chairs, or windows near the `START` marker inside the starting room for the final destination when the instruction requires exiting the starting room or passing intermediate landmarks first. Check your `Cumulative Episode Progress` (`total_walked` and `net_horizontal_displacement_from_start`) and `BACK VIEW` before proposing completion.
- **Depth-Grounded Stopping Rule (`propose_completion`)**:
  - **Solid Surface Landmarks** (`window`, `table`, `desk`, `bed`, `chair`, `couch`, `rug`, `cabinet`, `painting`, `counter/bar end`, `stairs`): Do NOT select `propose_completion` when the target landmark is still `>= 1.5m` away in camera depth with unvisited `WP_XX` waypoints leading closer to it. Navigate until you are within `~0.5m - 1.2m` of the target landmark.
  - **Open Doorways & Archways** (`"stop in the doorway"`, `"stop in the archway"`, `"wait at the door"`): Once you have crossed the specified room/hallway and stepped directly into the target open doorway or archway threshold, **propose completion immediately in that threshold**. When standing inside an open doorway or archway, your `FRONT VIEW` looks *through* the opening into the next room or corridor (`d > 1.5m`, which may even show another doorway further ahead), while the doorway/archway frame flanks your `LEFT/RIGHT` views and the room you just crossed is in your `BACK VIEW`. Do NOT keep walking through the target archway into subsequent rooms beyond it!
  - **Room-Entry Endpoints** (`"enter the dining room"`, `"go into the hallway"` without a specific furniture stop target inside): Propose completion as soon as you step `1.0m - 1.5m` past the entrance threshold inside the target room rather than walking all the way to the far wall.
- **Stair & Doorway Verification Evidence**:
  - For stairs, confirm the instructed vertical direction (`up` vs `down`), actual elevation change (`height_delta`), and arrival on a flat supported landing.
  - For doorway crossings, distinguish approaching a door from physically crossing through the threshold into the next space.

**4. Dead-End / Narrow Aisle Recovery & Output Format Contract**:
- **Anti-Oscillation & Narrow Aisle Escape**: Do not repeat an already-satisfied owned instruction turn or oscillate with uncontrolled in-place `TURN_*` actions at the same position; a second in-place turn is permitted only when the controller explicitly requests a bounded fresh observation (e.g., landmark/endpoint confirmation look) or heading restoration. If a narrow furniture aisle blocks forward waypoint generation and only visited (`red`) waypoints are available, immediately select a visited (`red`) `WP_XX` to step back and route around the wider side of the obstacle.
- **Strict JSON Output Schema**: Keep your reasoning inside `<think>...</think>` concise (under 200 words), then output one valid top-level JSON object with keys `"observation"`, `"thought"`, `"plan"`, and `"action"` (where `"action"` is copied verbatim from the current turn's `Legal action objects`). Never invent waypoint IDs, raw coordinates, or movement history."""

            use_cache_ctrl = os.environ.get("HSGM_ENABLE_CACHE_CONTROL", "0") == "1"
            sys_content = (
                [{"type": "text", "text": sys_prompt, "cache_control": {"type": "ephemeral"}}]
                if use_cache_ctrl else sys_prompt
            )

            if not hasattr(self, "history_msgs") or not self.history_msgs:
                self.history_msgs = [
                    {"role": "system", "content": sys_content}
                ]
            elif self.history_msgs[0].get("role") == "system":
                self.history_msgs[0]["content"] = sys_content

            # Chunked history compaction: keep self.history_msgs append-only up to max_turn user-assistant pairs,
            # then evict in chunks of evict_chunk_turns (default 6 turns = 12 msgs) while preserving Turn 1 as anchor.
            # This avoids 1-turn sliding-window shifts and preserves 100% KV prefix cache hits across 6-turn windows.
            max_turn = max(6, int(os.environ.get("HSGM_MAX_HISTORY_TURNS", 12)))
            evict_chunk_turns = max(2, int(os.environ.get("HSGM_HISTORY_EVICT_CHUNK_TURNS", 6)))
            max_hist_msgs = 1 + max_turn * 2
            if len(self.history_msgs) > max_hist_msgs:
                evict_msgs = min(evict_chunk_turns * 2, len(self.history_msgs) - 3)
                if evict_msgs > 0:
                    self.history_msgs = self.history_msgs[:3] + self.history_msgs[3 + evict_msgs:]

            if keep_img_turns > 1:
                hist_user_msgs = [m for m in self.history_msgs if m.get("role") == "user"]
                keep_in_hist = max(0, keep_img_turns - 1)
                if len(hist_user_msgs) > keep_in_hist:
                    u_idx = 0
                    for m in self.history_msgs:
                        if m.get("role") == "user":
                            u_idx += 1
                            if u_idx <= len(hist_user_msgs) - keep_in_hist and isinstance(m.get("content"), list):
                                m["content"] = [c for c in m["content"] if c.get("type") == "text"]

            messages_for_api = list(self.history_msgs) + [current_msg_for_api]

            extra_kwargs = {}
            max_out = os.environ.get("HSGM_MAX_OUTPUT_TOKENS")
            extra_kwargs["max_tokens"] = int(max_out) if max_out else 4096

            response = self._create_chat_completion_with_backoff(
                operation="planner_request", model=self.model_name,
                messages=messages_for_api, **extra_kwargs)
            result = response.choices[0].message.content
            print(result)

            clean_result = result
            if clean_result and isinstance(clean_result, str):
                clean_result = re.sub(r'<think>.*?</think>', '', clean_result, flags=re.DOTALL)
                if '</think>' in clean_result:
                    after_think = clean_result.split('</think>', 1)[-1].strip()
                    if '{' in after_think:
                        clean_result = after_think
                clean_result = clean_result.strip()
                if not clean_result or '{' not in clean_result:
                    clean_result = result

            self._pending_planner_history = current_msg_for_history
            return clean_result
        except Exception as e:
            print(f"[VLM Exception in _get_vlm_response_multiview] {e}")
            logging.warning(f"Error in _get_vlm_response_multiview: {e}")
            return "stop"


    def _next_request_seed(self, operation: str) -> int:
        import hashlib
        if not hasattr(self, '_base_seed'):
            self._base_seed = int(getattr(self, 'config', {}).get('seed', 42))
        if not hasattr(self, '_request_seed_counter'):
            self._request_seed_counter = 0
        episode_id = getattr(self, '_current_episode_id', 'ep')
        material = f"{self._base_seed}:{episode_id}:{operation}:{self._request_seed_counter}".encode('utf-8')
        self._request_seed_counter += 1
        return int.from_bytes(hashlib.blake2s(material, digest_size=4).digest(), 'big') & 0x7FFFFFFF

    def _get_verifier_client(self):
        client = getattr(self, "client", None)
        if client is not None and hasattr(client, "with_options"):
            return client.with_options(max_retries=0)
        return client

    def _create_chat_completion_with_backoff(self, operation: str = "llm_request", max_retries: int = 2, client_override=None, **kwargs):
        """Execute chat.completions.create with robust exponential backoff for HTTP 429 and transient errors."""
        import random
        client = client_override if client_override is not None else getattr(self, "client", None)
        if "client_override" in kwargs:
            kwargs.pop("client_override")
        kwargs.setdefault("temperature", float(os.environ.get("HSGM_TEMPERATURE", "1.0")))
        kwargs.setdefault("top_p", float(os.environ.get("HSGM_TOP_P", "0.95")))
        for name in ("presence_penalty", "frequency_penalty"):
            configured = os.environ.get("HSGM_" + name.upper())
            if configured is not None:
                kwargs.setdefault(name, float(configured))
        if (os.environ.get("HSGM_JSON_OBJECT_RESPONSES") == "1"
                and operation in ("planner_request", "verifier_request")):
            kwargs.setdefault("response_format", {"type": "json_object"})
        extra_body = dict(kwargs.get("extra_body") or {})
        extra_body.setdefault("top_k", int(os.environ.get("HSGM_TOP_K", "20")))
        is_verifier_structured = (
            operation == "verifier_request"
            and os.environ.get("HSGM_VERIFIER_STRUCTURED_THINKING", "1") == "1"
        )
        if (os.environ.get("HSGM_DISABLE_THINKING") == "1"
                or os.environ.get("HSGM_ENABLE_THINKING") == "0"
                or is_verifier_structured):
            extra_body["enable_thinking"] = False
            extra_body.pop("reasoning_effort", None)
        else:
            if os.environ.get("HSGM_REASONING_EFFORT"):
                extra_body.setdefault("reasoning_effort", os.environ.get("HSGM_REASONING_EFFORT"))
            if os.environ.get("HSGM_ENABLE_THINKING") == "1":
                extra_body.setdefault("enable_thinking", True)
        kwargs["extra_body"] = extra_body
        kwargs.setdefault("seed", self._next_request_seed(operation))
        cfg = getattr(self, "navigation_cfg", {})
        max_retries = min(max_retries, max(0, int(cfg.get("model_max_retries", 2))))
        request_timeout = max(1.0, float(cfg.get("model_request_timeout_s", os.environ.get("HSGM_MODEL_REQUEST_TIMEOUT_S", 180))))
        budget = getattr(self, "runtime_budget", None)
        base_delay = 3.0
        max_delay = 10.0
        request_started = time.monotonic()
        for attempt in range(max_retries + 1):
            if budget is not None:
                budget.claim_request()
            kwargs["timeout"] = min(request_timeout, budget.remaining()) if budget else request_timeout
            try:
                response = client.chat.completions.create(**kwargs)
                if budget is not None:
                    budget.check()
                try:
                    usage = getattr(response, "usage", None)
                    if usage is not None:
                        p_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
                        c_tok = int(getattr(usage, "completion_tokens", 0) or 0)
                        details = getattr(usage, "prompt_tokens_details", None)
                        cached_tok = int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
                        hit_pct = (100.0 * cached_tok / p_tok) if p_tok > 0 else 0.0
                        logging.info(f"[TokenUsage] {operation}: prompt={p_tok} (cached={cached_tok}, hit={hit_pct:.1f}%), completion={c_tok}")
                except Exception:
                    pass
                return response
            except Exception as e:
                is_rate_limit = type(e).__name__ == 'RateLimitError' or getattr(e, 'status_code', None) == 429
                is_transient = type(e).__name__ in ('APITimeoutError', 'InternalServerError') or getattr(e, 'status_code', None) in (500, 502, 503, 504)
                if (not is_rate_limit and not is_transient) or attempt >= max_retries:
                    raise e
                delay = min(max_delay, base_delay * (2 ** attempt) + random.uniform(0.1, 0.5))
                logging.warning(f"[{type(e).__name__}] {operation} encountered error. Waiting {delay:.1f}s before retry ({attempt + 1}/{max_retries})...")
                time.sleep(delay)

    def _get_verifier_response(self, front_rgb_image, left_rgb_image,
                                right_rgb_image, map_image, prompt: str,
                                back_rgb_image=None, img_buffer=None) -> str:
        """
        Independent, stateless verification call.
        Sends past keyframes + current 4 views (front, left, right, back) + BEV map + prompt.
        Does not mutate planner conversation history (self.history_msgs).
        """
        try:
            cfg = getattr(self, 'navigation_cfg', {}) or {}
            evidence_enabled = (
                bool(self._evidence_enabled())
                if hasattr(self, '_evidence_enabled') and callable(getattr(self, '_evidence_enabled'))
                else bool(cfg.get('verification_evidence_enabled', False))
            )
            fixed_bev_enabled = (
                bool(self._fixed_bev_enabled())
                if hasattr(self, '_fixed_bev_enabled') and callable(getattr(self, '_fixed_bev_enabled'))
                else bool(cfg.get('verifier_fixed_bev', False))
            )
            crops_enabled = bool(cfg.get('task_visual_enabled', False))
            evidence_blocks = (
                list(self._verifier_evidence_content() or [])
                if hasattr(self, '_verifier_evidence_content') and callable(getattr(self, '_verifier_evidence_content'))
                else []
            )
            crop_blocks = (
                list(self._visual_candidate_content(for_verifier=True) or [])
                if hasattr(self, '_visual_candidate_content') and callable(getattr(self, '_visual_candidate_content'))
                else []
            )
            new_inputs_active = bool(
                evidence_enabled or fixed_bev_enabled or crops_enabled or evidence_blocks or crop_blocks
            )
            front_img_b64 = self._pil_to_base64(front_rgb_image) if front_rgb_image is not None else None
            left_img_b64 = self._pil_to_base64(left_rgb_image) if left_rgb_image is not None else None
            right_img_b64 = self._pil_to_base64(right_rgb_image) if right_rgb_image is not None else None
            back_img_b64 = self._pil_to_base64(back_rgb_image) if back_rgb_image is not None else None
            map_img_b64 = (
                self._pil_to_base64(map_image)
                if (map_image is not None and cfg.get('verifier_use_bev', False) and not fixed_bev_enabled)
                else None
            )
            bev_sys_desc = (
                "3. **Optional Top-Down Bird's-Eye-View (BEV) Map (only when provided)**: Fixed-axis metric verifier map (`+X` right, `+Y` up in world meters; read the on-image legend for `START`, prior vs. current subtask trajectory segments, `TURN` markers, height profile, and `CURRENT AGENT POSE` arrow; historical/display context only, never standalone endpoint or route proof)."
                if fixed_bev_enabled else
                "3. **Optional Top-Down Bird's-Eye-View (BEV) Map (only when provided)**: Heading-up metric map (`Red Arrow` = current agent pose facing upward; `Red Trail` = trajectory; `START` = episode origin `(0, 0)`; `Blue/Amber tint` = observed height bands in 0.25m steps below/above current floor)."
            )

            sys_prompt = f"""You are an independent, rigorous embodied Vision-Language Navigation (VLN-CE) route and endpoint verifier. Your task is to audit whether the navigation agent has physically fulfilled all sequential clauses of its current subtask and reached the exact instructed stopping location, based strictly on:
1. **Past First-Person Keyframes**: Earlier front-camera frames captured along the trajectory (oldest to newest), showing rooms, doorways, hallways, and landmarks already traversed.
2. **Four Current First-Person RGB Views**: `[FRONT VIEW (0 deg, straight ahead)]`, `[LEFT VIEW (-90 deg)]`, `[RIGHT VIEW (+90 deg)]`, and `[BACK VIEW (180 deg, behind you)]`, including central-patch depth (`center=X.Xm`), middle-band 85th-percentile depth (`p85=Y.Ym`), and projected candidate waypoints (`WP_XX`).
{bev_sys_desc}
4. **Measured Execution Context**: Ground-truth simulator displacement from start, per-view depth & waypoint distances, mandatory execution conditions, and recorded step ledger.

**Core Verification Protocol & Decision Criteria**:
- The endpoint must be identified in current RGB views by visible physical features and its requested relation. Map labels, waypoint labels, a room adjacent to the target, or merely reaching a map region are never endpoint evidence. A staircase requires visible steps/risers or stair rails; furniture near a STAIRS label is insufficient. If the named target is not visible, report unknown/not_reached, not wrong_path solely for absence.
- **Sequential Clause & Route Verification (`relation_satisfied`)**:
  - Verify that EVERY sequential clause in the Current Subtask has been physically executed in the required order (e.g., exiting the starting room, turning into the corridor, passing intermediate objects, traversing stairs, and arriving at the endpoint).
  - Cross-check the `BACK VIEW`, `PAST FIRST-PERSON KEYFRAMES`, and net displacement from episode start (`START`). Never accept a premature stop near the starting area when the instruction requires traversing across a room, down a hallway, or into another room first.
- **Localized Endpoint Verification**: View-center/p85 depths are NOT target distances. Supply the exact current `target_view` and normalized `target_bbox`, or a currently visible matching entity ID. The controller measures only that localized target region. Missing or uncertain localization cannot authorize STOP. A second changed view is required for a target that only you localized.
- **Depth-Grounded Endpoint Verification (Solid Landmarks vs. Open Doorways/Archways)**:
  - **Solid Surface Landmarks** (`window`, `table`, `desk`, `bed`, `chair`, `couch`, `rug`, `counter/bar end`, `cabinet`): Sighting a solid landmark in the distance (`>= 1.5m - 7.0m` away in camera depth with candidate waypoints `WP_XX` still leading closer to it) does NOT prove arrival. Accept arrival (`relation_satisfied: true`, `target_proximity: "satisfied"`) when the agent is within `~0.5m - 1.2m` of the target landmark.
  - **Open Doorways & Archways** (`"stop in the doorway"`, `"stop in the archway"`, `"wait at the door"`): When the agent stands **inside** an open doorway or archway threshold, the doorway/archway jambs flank the `LEFT/RIGHT` views (or were just stepped into) and the room/hallway just crossed is visible in the `BACK VIEW`, while the `FRONT VIEW` looks *through* the open doorway into the next corridor or room (`d > 1.5m`, which may even show another doorway further ahead). Accept arrival when the agent has crossed the specified room/hallway and is standing in that target doorway/archway threshold — do NOT reject merely because `FRONT VIEW` center depth looks through the opening into the space beyond!
  - **Room-Entry Endpoints** (`"enter the bedroom"`, `"walk into the dining room"` without a specific furniture stop target): Accept arrival once the agent has stepped `1.0m - 1.5m` past the doorway threshold inside the target room.
- **Wrong-route evidence**: An unexecuted future clause, a target not yet visible, or uncertainty is not a route contradiction. Use `route_contradiction` only for positive evidence against an executed current/past condition, with its exact ID and measured step references. A scene_consistent=true verdict must not request rollback.
- **Stair & Doorway Witness Auditing**:
  - For stair subtasks, verify that the measured height trend matches the instructed direction (`up` vs `down`) and that the agent has reached a supported landing.
  - For doorway crossing conditions, populate `"doorway_checks"` citing the actual executed step numbers, pre-crossing `entry_side`, post-crossing `exit_side`, and RGB/BEV visual evidence.
- **Handoff & Outcome Gate**:
  - On the final subtask, set `"handoff_consistent": null` and set `"outcome": true` if and only if `"relation_satisfied": true` and `"scene_consistent": true`.
  - On intermediate subtasks, set `"outcome": true` if and only if `"relation_satisfied": true`, `"handoff_consistent": true`, and `"scene_consistent": true`.
  - When `"outcome": false`, set `"failure_type"` to `"not_reached"` or `"wrong_path"`, explain the exact remaining distance/heading gap in `"correction_reason"`, and select one exact legal action object from the available candidates in `"correction_action"`.
- **Structured Deliberation Contract**:
  - Do NOT output any <think> tags. Write all verification analysis concisely inside the "thought" and "reason" fields of the JSON.
  - Keep condition check evidence under 20 words, and output strictly valid JSON without markdown code blocks."""

            def _cap_multimodal_units(blocks, max_images, preserve_tail_bev=False):
                if max_images <= 0 or not blocks:
                    return []
                units = []
                pending_texts = []
                for blk in blocks:
                    if not isinstance(blk, dict):
                        continue
                    btype = blk.get("type")
                    if btype == "text":
                        pending_texts.append(blk)
                    elif btype == "image_url":
                        if len(pending_texts) > 1:
                            hdrs = pending_texts[:-1]
                            caps = pending_texts[-1:]
                        else:
                            hdrs = []
                            caps = list(pending_texts)
                        units.append((hdrs, caps, blk))
                        pending_texts = []
                if not units:
                    return []
                if len(units) > max_images:
                    if preserve_tail_bev:
                        last_caps = " ".join(str(c.get("text", "")) for c in units[-1][1]).upper()
                        if "BEV" in last_caps or "MAP" in last_caps:
                            kept_units = units[:max(0, max_images - 1)] + [units[-1]]
                        else:
                            kept_units = units[:max_images]
                    else:
                        kept_units = units[:max_images]
                else:
                    kept_units = units
                out_blocks = []
                emitted_hdr = False
                for idx, (hdrs, caps, img_blk) in enumerate(kept_units):
                    if hdrs and not emitted_hdr:
                        out_blocks.extend(hdrs)
                        emitted_hdr = True
                    elif idx == 0 and units[0][0] and not emitted_hdr:
                        out_blocks.extend(units[0][0])
                        emitted_hdr = True
                    out_blocks.extend(caps)
                    out_blocks.append(img_blk)
                return out_blocks

            content = [{"type": "text", "text": prompt}]
            default_hist_n = int(cfg.get('verifier_history_frames', 4))
            default_hist_w = 280 if os.environ.get("HSGM_DISABLE_THINKING") == "1" else cfg.get('planner_history_frame_width', 448)
            hist_n = max(0, int(os.environ.get('HSGM_VERIFIER_HISTORY_FRAMES', default_hist_n)))
            if new_inputs_active:
                hist_n = min(2, hist_n)
            hist_w = max(160, int(os.environ.get('HSGM_PLANNER_HISTORY_FRAME_WIDTH', default_hist_w)))
            hist_frames = self._select_history_keyframes(img_buffer, hist_n) if hasattr(self, '_select_history_keyframes') else []
            if new_inputs_active and hist_frames:
                hist_frames = list(hist_frames)[:2]
            if hist_frames:
                content.append({"type": "text", "text": (
                    f"PAST FIRST-PERSON KEYFRAMES ({len(hist_frames)}, oldest to newest across the episode): "
                    "front-camera frames showing what the agent saw and passed earlier.")})
                for img in hist_frames:
                    b64_small = (
                        self._pil_to_base64_small(img, hist_w)
                        if hasattr(self, '_pil_to_base64_small')
                        else self._pil_to_base64(img)
                    )
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_small}",
                                      "detail": "auto"}})

            content.append({"type": "text", "text": "CURRENT OBSERVATION AT PROPOSED STOP:"})
            for tag, b64 in [
                ("FRONT VIEW (0 deg, straight ahead)", front_img_b64),
                ("LEFT VIEW (-90 deg)", left_img_b64),
                ("RIGHT VIEW (+90 deg)", right_img_b64),
                ("BACK VIEW (180 deg, behind you)", back_img_b64),
                ("TOP-DOWN MAP (you face upward)", map_img_b64),
            ]:
                if b64 is not None:
                    content.append({"type": "text", "text": f"[{tag}]"})
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}})

            if evidence_blocks:
                ev_max_imgs = 5 if fixed_bev_enabled else 4
                content.extend(_cap_multimodal_units(evidence_blocks, ev_max_imgs, preserve_tail_bev=True))
            if crop_blocks:
                used_imgs = sum(1 for b in content if isinstance(b, dict) and b.get("type") == "image_url")
                crop_budget = min(4, max(0, 15 - used_imgs))
                content.extend(_cap_multimodal_units(crop_blocks, crop_budget, preserve_tail_bev=False))

            use_cache_ctrl = os.environ.get("HSGM_ENABLE_CACHE_CONTROL", "0") == "1"
            sys_content = (
                [{"type": "text", "text": sys_prompt, "cache_control": {"type": "ephemeral"}}]
                if use_cache_ctrl else sys_prompt
            )
            messages = [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": content},
            ]

            verifier_client = (
                self._get_verifier_client()
                if hasattr(self, '_get_verifier_client')
                else (self.client.with_options(max_retries=0) if hasattr(getattr(self, 'client', None), 'with_options') else getattr(self, 'client', None))
            )

            max_v_out = os.environ.get("HSGM_VERIFIER_MAX_TOKENS", str(cfg.get('verifier_max_output_tokens', 1536)))
            extra_kwargs = {
                "max_tokens": int(max_v_out),
            }

            if hasattr(self, '_create_chat_completion_with_backoff'):
                v_retries = int(getattr(self, 'navigation_cfg', {}).get('verifier_max_retries', 0))
                response = self._create_chat_completion_with_backoff(
                    operation="verifier_request",
                    model=self.model_name,
                    messages=messages,
                    max_retries=v_retries,
                    client_override=verifier_client,
                    **extra_kwargs
                )
            else:
                target_client = verifier_client if verifier_client is not None else self.client
                response = target_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    timeout=180,
                    **extra_kwargs
                )
            result = response.choices[0].message.content
            print("Verifier response:", result)
            clean_result = result
            if clean_result and isinstance(clean_result, str):
                if '</think>' in clean_result:
                    clean_result = clean_result.split('</think>', 1)[-1].strip()
                elif '<think>' in clean_result:
                    clean_result = re.sub(r'<think>.*?</think>', '', clean_result, flags=re.DOTALL).strip()
            return clean_result if clean_result else result
        except Exception as e:
            logging.getLogger(__name__).warning(
                "Verifier request failed (%s): %s", type(e).__name__, str(e)[:400]
            )
            return json.dumps({'error': f'{type(e).__name__}: {str(e)[:400]}'})

    def _get_llm_response(self, prompt: str) -> str:
        """
        Call the text LLM (no image, no chat history) for instruction decomposition.
        Returns the model's plain-text response, or "stop" on failure.
        """
        try:
            extra_kwargs = {}
            max_out = os.environ.get("HSGM_MAX_OUTPUT_TOKENS")
            extra_kwargs["max_tokens"] = int(max_out) if max_out else 2048

            response = self._create_chat_completion_with_backoff(
                operation="instruction_decomposition", model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a concise navigation assistant. Keep thinking brief (under 150 words) within <think>, then directly output the required format."},
                    {"role": "user", "content": prompt}
                ],
                **extra_kwargs
            )
            if hasattr(response, "choices") and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    return choice.message.content
                if hasattr(choice, "text"):
                    return choice.text
                return str(choice)
            if hasattr(response, "output_text"):
                return response.output_text
            return str(response)
        except Exception as e:
            print(f"[LLM Exception in _get_llm_response] {e}")
            logging.warning(f"Error in _get_llm_response: {e}")
            return "stop"


# src/wasd_walk.py
from __future__ import annotations
import datetime
import gzip
import json
import tempfile
import time

import cv2
import torch
import open3d
import habitat_sim
import numpy as np
import yaml
import logging
import os
import select
import sys
import termios
import time
import tty

from tqdm import tqdm

from simWrapper import SimWrapper, PolarAction
from mapper import Instruct_Mapper
from segmentation.instance_segmentation import visualize_instance_segmentation
from agent.agent import GPTAgent
import imageio.v2 as imageio
try:
    from runtime_budget import EpisodeBudget, EpisodeBudgetExceeded
    from progress_watchdog import NavigationAborted
    from navigation_policy import ModelServiceError
except ImportError:
    class EpisodeBudgetExceeded(BaseException):
        def __init__(self, reason=''): self.reason = reason
    class NavigationAborted(EpisodeBudgetExceeded):
        def __init__(self, reason=''): self.reason = reason
    class EpisodeBudget:
        def __init__(self, seconds=900, max_requests=48): pass
        def check(self): pass
        def remaining(self): return 900.0
        def summary(self): return {}
        def guard(self):
            from contextlib import nullcontext
            return nullcontext()
    class ModelServiceError(Exception): pass


class EpisodeVideoRecorder:
    """Stream episode frames to MP4 while retaining only the latest frame."""

    def __init__(self, video_path: str, fps: int = 2):
        self.video_path = video_path
        self.partial_path = f"{video_path}.partial.mp4"
        self.fps = max(1, int(fps))
        self.writer = None
        self.frame_size = None
        self.last_frame = None
        self.frame_count = 0
        self._closed = False
        self._writer_failed = False

    def __bool__(self):
        return self.frame_count > 0

    def __len__(self):
        return self.frame_count

    def append(self, frame, repeats=None):
        if frame is None or self._closed:
            return

        frame = np.asarray(frame)
        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating) and frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            logging.warning("Skipping invalid episode video frame with shape %s", frame.shape)
            return

        height, width = frame.shape[:2]
        if self.writer is None and not self._writer_failed:
            os.makedirs(os.path.dirname(self.video_path) or ".", exist_ok=True)
            self.frame_size = (width, height)
            self.writer = cv2.VideoWriter(
                self.partial_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                self.frame_size,
            )
            if not self.writer.isOpened():
                self.writer.release()
                self.writer = None
                self._writer_failed = True
                logging.warning("Could not open episode video writer for %s", self.video_path)

        if self.writer is not None:
            if frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            target_width, target_height = self.frame_size
            if frame_bgr.shape[1] != target_width or frame_bgr.shape[0] != target_height:
                frame_bgr = cv2.resize(
                    frame_bgr,
                    (target_width, target_height),
                    interpolation=cv2.INTER_LINEAR,
                )
            for _ in range(max(1, int(repeats if repeats is not None else self.fps))):
                self.writer.write(frame_bgr)

        self.last_frame = frame.copy()
        self.frame_count += 1

    def close(self):
        if self._closed:
            return
        self._closed = True

        if self.writer is not None:
            self.writer.release()
            self.writer = None

        if self.frame_count and os.path.isfile(self.partial_path):
            try:
                os.replace(self.partial_path, self.video_path)
            except OSError:
                logging.exception("Could not finalize episode video %s", self.video_path)

        self.last_frame = None


import datetime
from fastdtw import fastdtw
from typing import Iterable, Optional

class QuitKeyMonitor:
    """Non-blocking single-key monitor for quitting long experiment runs."""

    def __init__(self, quit_key: str = "q"):
        self.quit_key = quit_key.lower()
        self.requested = False
        self._fd = None
        self._old_settings = None
        self._enabled = False

    def __enter__(self):
        if sys.stdin is not None and sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self._enabled = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._enabled and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def poll(self) -> bool:
        if self.requested or not self._enabled:
            return self.requested

        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch.lower() == self.quit_key:
                self.requested = True
                logging.info("Detected 'q' key; will quit after saving current results.")
                break
        return self.requested

def create_top_down_map(mapper: Instruct_Mapper, fov_angle_deg: float, map_scale=0.025, padding_px=20, fov_range=2.5, show_all_labels=False, label_volume_threshold=100):
    t_start = time.time()
    timings = {}

    t = time.time()
    all_points = []
    if mapper.navigable_pcd.point.positions.shape[0] > 0:
        all_points.append(mapper.navigable_pcd.point.positions.cpu().numpy()[:, :2])
    if mapper.obstacle_pcd.point.positions.shape[0] > 0:
        all_points.append(mapper.obstacle_pcd.point.positions.cpu().numpy()[:, :2])
    if len(mapper.trajectory_position) > 0:
        all_points.append(np.array(mapper.trajectory_position)[:, :2])
    for entity in mapper.object_entities:
        if entity['pcd'].point.positions.shape[0] > 0:
            all_points.append(entity['pcd'].point.positions.cpu().numpy()[:, :2])
    timings['collect_points'] = time.time() - t

    t = time.time()
    if not all_points:
        map_size_px = 1024
        top_down_map = np.full((map_size_px, map_size_px, 3), 255, dtype=np.uint8)
    else:
        points_concat = np.vstack(all_points)
        min_coords = points_concat.min(axis=0)
        max_coords = points_concat.max(axis=0)
        map_width_px = int((max_coords[0] - min_coords[0]) / map_scale) + 2 * padding_px
        map_height_px = int((max_coords[1] - min_coords[1]) / map_scale) + 2 * padding_px
        map_width_px = max(map_width_px, 256)
        map_height_px = max(map_height_px, 256)
        top_down_map = np.full((map_height_px, map_width_px, 3), 255, dtype=np.uint8)
        offset = min_coords - padding_px * map_scale
    timings['calc_map_size'] = time.time() - t

    t = time.time()

    def world_to_pixel(world_coords):
        if world_coords.ndim == 1:
            world_coords = world_coords.reshape(1, -1)
        return ((world_coords[:, :2] - offset) / map_scale).astype(int)

    timings['coord_transform'] = time.time() - t

    map_h, map_w = top_down_map.shape[:2]

    pcd_resolution = 0.025  # voxel size of mapper point cloud
    dilation_radius = max(1, int(pcd_resolution / map_scale / 2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius * 2 + 1, dilation_radius * 2 + 1))

    t = time.time()
    if mapper.navigable_pcd.point.positions.shape[0] > 0:
        nav_points = mapper.navigable_pcd.point.positions.cpu().numpy()
        nav_pixels = world_to_pixel(nav_points)
        valid_mask = (nav_pixels[:, 0] >= 0) & (nav_pixels[:, 0] < map_w) & \
                     (nav_pixels[:, 1] >= 0) & (nav_pixels[:, 1] < map_h)
        valid_nav_pixels = nav_pixels[valid_mask]

        nav_mask = np.zeros((map_h, map_w), dtype=np.uint8)
        nav_mask[valid_nav_pixels[:, 1], valid_nav_pixels[:, 0]] = 255
        dilated_nav_mask = cv2.dilate(nav_mask, kernel)
        top_down_map[dilated_nav_mask == 255] = (200, 200, 200)
    timings['draw_navigable'] = time.time() - t

    t = time.time()
    obs_pixels = np.array([[]]).reshape(0, 2).astype(int)
    if mapper.obstacle_pcd.point.positions.shape[0] > 0:
        obs_points = mapper.obstacle_pcd.point.positions.cpu().numpy()
        obs_pixels = world_to_pixel(obs_points)
        valid_mask = (obs_pixels[:, 0] >= 0) & (obs_pixels[:, 0] < map_w) & \
                     (obs_pixels[:, 1] >= 0) & (obs_pixels[:, 1] < map_h)
        valid_obs_pixels = obs_pixels[valid_mask]

        obs_mask = np.zeros((map_h, map_w), dtype=np.uint8)
        obs_mask[valid_obs_pixels[:, 1], valid_obs_pixels[:, 0]] = 255
        dilated_obs_mask = cv2.dilate(obs_mask, kernel)
        top_down_map[dilated_obs_mask == 255] = (50, 50, 50)
    timings['draw_obstacle'] = time.time() - t

    t = time.time()
    if mapper.obstacle_pcd.point.positions.shape[0] > 0 and len(mapper.object_entities) > 0:

        obstacle_mask_bool = (dilated_obs_mask == 255) if 'dilated_obs_mask' in locals() else np.zeros((map_h, map_w),
                                                                                                       dtype=bool)

        for entity in mapper.object_entities:
            if entity['pcd'].point.positions.shape[0] > 0:
                obj_points = entity['pcd'].point.positions.cpu().numpy()
                obj_colors_rgb = entity['pcd'].point.colors.cpu().numpy()
                obj_colors_bgr = (obj_colors_rgb * 255)[:, ::-1].astype(np.uint8)
                obj_pixels = world_to_pixel(obj_points)

                valid_pixels_mask = (obj_pixels[:, 0] >= 0) & (obj_pixels[:, 0] < map_w) & \
                                    (obj_pixels[:, 1] >= 0) & (obj_pixels[:, 1] < map_h)
                valid_obj_pixels = obj_pixels[valid_pixels_mask]
                valid_obj_colors = obj_colors_bgr[valid_pixels_mask]

                if len(valid_obj_pixels) == 0: continue

                intersection_mask = obstacle_mask_bool[valid_obj_pixels[:, 1], valid_obj_pixels[:, 0]]

                if np.any(intersection_mask):
                    final_pixels = valid_obj_pixels[intersection_mask]
                    if len(final_pixels) == 0: continue

                    object_mask_img = np.zeros((map_h, map_w), dtype=np.uint8)
                    object_mask_img[final_pixels[:, 1], final_pixels[:, 0]] = 255

                    fill_kernel = np.ones((5, 5), np.uint8)
                    filled_mask = cv2.morphologyEx(object_mask_img, cv2.MORPH_CLOSE, fill_kernel, iterations=2)

                    object_color = valid_obj_colors[intersection_mask][0]
                    top_down_map[filled_mask == 255] = object_color
    timings['draw_objects'] = time.time() - t

    t = time.time()
    drawn_labels = []
    label_merge_distance_px = 50
    for entity in sorted(mapper.object_entities, key=lambda e: e['class']):

        if not show_all_labels and entity['pcd'].point.positions.shape[0] < label_volume_threshold:
            continue
        if entity['pcd'].point.positions.shape[0] == 0:
            continue

        class_name = entity['class_name']
        center_world = entity['center']
        center_pixel = world_to_pixel(center_world)[0]
        should_merge = False
        for i, label_info in enumerate(drawn_labels):
            if label_info['class'] == class_name:
                dist = np.linalg.norm(center_pixel - label_info['center_pixel'])
                if dist < label_merge_distance_px:
                    drawn_labels[i]['center_pixel'] = (label_info['center_pixel'] + center_pixel) / 2
                    should_merge = True
                    break
        if not should_merge:
            drawn_labels.append({'class': class_name, 'center_pixel': center_pixel})

    for label_info in drawn_labels:
        class_name = label_info['class']
        center_pixel = label_info['center_pixel'].astype(int)
        font_scale = 0.3
        font_thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                                              font_thickness)
        box_tl = (center_pixel[0] - text_width // 2 - 3, center_pixel[1] - text_height - 3)
        box_br = (center_pixel[0] + text_width // 2 + 3, center_pixel[1] + baseline)
        cv2.rectangle(top_down_map, box_tl, box_br, (255, 255, 255), -1)
        cv2.rectangle(top_down_map, box_tl, box_br, (0, 0, 0), 1)
        text_org = (center_pixel[0] - text_width // 2, center_pixel[1])
        cv2.putText(top_down_map, class_name, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness,
                    cv2.LINE_AA)
    timings['draw_labels'] = time.time() - t

    t = time.time()
    max_traj_length = 30
    dash_length = 8
    gap_length = 6
    if len(mapper.trajectory_position) > 1:
        traj_points = np.array(mapper.trajectory_position)[-max_traj_length:]
        traj_pixels = world_to_pixel(traj_points)
        valid_pixels = (traj_pixels[:, 0] >= 0) & (traj_pixels[:, 0] < map_w) & \
                       (traj_pixels[:, 1] >= 0) & (traj_pixels[:, 1] < map_h)
        valid_traj_pixels = traj_pixels[valid_pixels]
        for i in range(len(valid_traj_pixels) - 1):
            pt1 = tuple(valid_traj_pixels[i])
            pt2 = tuple(valid_traj_pixels[i + 1])
            vec = np.array(pt2) - np.array(pt1)
            length = np.linalg.norm(vec)
            if length == 0:
                continue
            direction = vec / length
            pos = np.array(pt1, dtype=float)
            drawn = 0
            while drawn < length:
                start = pos.astype(int)
                end = (pos + direction * min(dash_length, length - drawn)).astype(int)
                cv2.line(top_down_map, tuple(start), tuple(end), (255, 0, 0), 1)
                pos += direction * (dash_length + gap_length)
                drawn += dash_length + gap_length
    timings['draw_trajectory'] = time.time() - t

    t = time.time()
    agent_pos = mapper.current_position
    agent_rot = mapper.current_rotation
    if 'offset' in locals():
        agent_pixel = world_to_pixel(agent_pos)[0]
    else:
        agent_pixel = (agent_pos[:2] / map_scale + map_size_px // 2).astype(int)

    if 0 <= agent_pixel[0] < map_w and 0 <= agent_pixel[1] < map_h:
        overlay = top_down_map.copy()
        fov_rad = np.deg2rad(fov_angle_deg)
        forward_vec_3d = agent_rot @ np.array([0, 0, -1])
        forward_vec_2d = forward_vec_3d[:2]
        if np.linalg.norm(forward_vec_2d) > 1e-6:
            forward_vec_2d /= np.linalg.norm(forward_vec_2d)
        agent_angle_rad = np.arctan2(forward_vec_2d[1], forward_vec_2d[0])
        num_fov_points = 30
        angles = np.linspace(agent_angle_rad - fov_rad / 2, agent_angle_rad + fov_rad / 2, num_fov_points)
        sector_points = [world_to_pixel(agent_pos[:2] + np.array([np.cos(a), np.sin(a)]) * fov_range)[0] for a in
                         angles]
        sector_pts = np.array([agent_pixel] + sector_points, np.int32)
        cv2.fillPoly(overlay, [sector_pts], (255, 255, 0), lineType=cv2.LINE_AA)
        alpha = 0.4
        top_down_map = cv2.addWeighted(overlay, alpha, top_down_map, 1 - alpha, 0)
        cv2.circle(top_down_map, tuple(agent_pixel), 5, (0, 0, 255), -1)
        arrow_end_world = agent_pos[:2] + forward_vec_2d * 0.5
        arrow_end_pixel = world_to_pixel(arrow_end_world)[0]
        cv2.arrowedLine(top_down_map, tuple(agent_pixel), tuple(arrow_end_pixel), (0, 0, 255), thickness=2, tipLength=0.4)
    timings['draw_agent'] = time.time() - t

    t_end = time.time()
    pass
    for k, v in timings.items():
        print(f'  {k}: {v:.3f}s')
    pass
    return top_down_map

def wrap_text(text, font, font_scale, thickness, max_width):
    words = text.split(' ')
    lines = []
    current_line = ''
    for word in words:
        test_line = current_line + (' ' if current_line else '') + word
        (w, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if w > max_width and current_line:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    if current_line:
        lines.append(current_line)
    return lines

def _populate_error_metrics(existing_metrics: dict, error_info: dict, env=None) -> dict:
    """Preserve distance diagnostics without crediting an unsuccessful termination."""
    default_metrics = {
        'distance_to_goal': getattr(env, 'oracle_best_distance', None) if env else None,
        'spl': 0.0,
        'goal_reached': 0,
        'finish_status': 'error',
        'traveled_distance': getattr(getattr(env, 'agent', None), 'traveled_distance', 0.0) if env else 0.0,
        'done': 1,
        'oracle_navigation_error': getattr(env, 'oracle_best_distance', None) if env else None,
        'oracle_success': 1 if getattr(env, 'oracle_success', False) else 0,
        'oracle_spl': float(getattr(env, 'oracle_best_spl', 0.0)) if env else 0.0,
        'ndtw': None,
        'sdtw': None,
    }
    merged = {**default_metrics, **(existing_metrics or {})}
    merged.update(finish_status='error', goal_reached=False, spl=0.0, sdtw=0.0)
    merged['error_info'] = error_info
    return merged


def _budget_termination_metrics(metrics, reason, aborted=False):
    """A resource limit never manufactures the agent's required STOP action."""
    return {**metrics,
            'finish_status': 'navigation_aborted' if aborted else 'budget_exhausted',
            'termination_reason': reason, 'budget_reason': None if aborted else reason,
            'goal_reached': False, 'spl': 0.0, 'sdtw': 0.0}

class Env:

    def __init__(self, sim_cfg: dict, env_cfg: dict):
        self.env_cfg = env_cfg
        self.sim_cfg = sim_cfg
        self.split = 'val'

        now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.env_cfg['name'] = f'{self.env_cfg["name"]}_{now_str}'

        self.path_calculator = habitat_sim.MultiGoalShortestPath()
        self.simWrapper: SimWrapper = None
        self.num_episodes = 0
        self.agent = None
        self._quit_monitor = None
        self._quit_requested = False
        self._initialize_experiment()

    def _initialize_experiment(self):
        raise NotImplementedError

    def run_experiment(
        self,
        begin_idx: int = 0,
        end_idx: int = -1,
        show_visualization: bool = False,
        episode_ids: Optional[list[int]] = None,
    ):
        """
        Run episodes and aggregate statistics.
        \- If episode_ids given, run those ids (deduplicated, in order)
        \- Otherwise, run by begin_idx/end_idx range
        """
        aggregated_results = []
        metric_aggregates = {}
        summary = {}

        if episode_ids is not None:

            selected = list(dict.fromkeys(episode_ids))
            episode_list = []
            for ep_id in selected:
                if not isinstance(ep_id, int):
                    pass
                    continue
                if ep_id < begin_idx or ep_id >= min(self.num_episodes, end_idx if end_idx > 0 else self.num_episodes):
                    pass
                    continue
                episode_list.append(ep_id)
        else:
            episode_range = range(begin_idx, end_idx if end_idx > 0 else self.num_episodes)
            episode_list = list(episode_range)

        num_run_episodes = 0

        self._quit_requested = False
        with QuitKeyMonitor() as quit_monitor:
            self._quit_monitor = quit_monitor
            try:
                for episode_ndx in tqdm(episode_list):
                    try:
                        if self._user_requested_quit():
                            pass
                            break

                        # metrics = self._run_episode(episode_ndx, show_visualization=show_visualization)
                        metrics = self._run_episode(episode_ndx, show_visualization=show_visualization)

                    except Exception as e:
                        import traceback
                        metrics = _populate_error_metrics(
                            {}, {"exc": str(e), "traceback": traceback.format_exc()}, env=self
                        )

                    aggregated_results.append({"episode": episode_ndx, **metrics})
                    num_run_episodes += 1

                    for k, v in metrics.items():
                        if isinstance(v, (int, float)) and k not in ("episode",):
                            if k not in metric_aggregates:
                                metric_aggregates[k] = {"sum": 0.0, "count": 0}
                            metric_aggregates[k]["sum"] += float(v)
                            metric_aggregates[k]["count"] += 1

                    summary = {"averages": {}, "num_episodes": num_run_episodes, "per_episode": aggregated_results}

                    for key, data in metric_aggregates.items():
                        if data["count"] > 0:
                            summary["averages"][key] = data["sum"] / data["count"]

                    results_dir = os.path.join(self.env_cfg.get("output_root", "../results"), f"results_{self.env_cfg.get('name', 'env')}")
                    os.makedirs(results_dir, exist_ok=True)
                    summary_path = os.path.join(results_dir, "aggregate_results.json")
                    try:
                        with open(summary_path, "w", encoding="utf-8") as f:
                            json.dump(summary, f, ensure_ascii=False, indent=2)
                        pass
                    except Exception as e:
                        pass

                    if metrics.get("finish_status") in ("user_interrupt", "user_abort"):
                        self._quit_requested = True
                        pass
                        break
            finally:
                self._quit_monitor = None

        pass
        for key, value in summary.get("averages", {}).items():
            logging.info(f"{key}: {value:.4f}")
        pass
        return summary

    def _request_user_quit(self) -> None:
        self._quit_requested = True
        if self._quit_monitor is not None:
            self._quit_monitor.requested = True

    def _user_requested_quit(self) -> bool:
        if self._quit_requested:
            return True
        if self._quit_monitor is not None and self._quit_monitor.poll():
            self._quit_requested = True
        return self._quit_requested

    def _run_episode(self, episode_ndx: int, show_visualization: bool = True):
        raise NotImplementedError

    def _initialize_episode(self, episode_ndx: int):
        self._pre_init_episode()

    def _pre_init_episode(self):
        self.step = 0
        self.init_pos = None
        self.prev_agent_position = None
        self.agent_trajectory = []

        self.oracle_best_distance = float("inf")
        self.oracle_success = False
        self.oracle_best_spl = 0.0

    def _step_env(self):
        pass

    def _post_episode(self):
        pass

    def _calculate_metrics(
            self,
            agent_state: habitat_sim.AgentState,
            agent_action: PolarAction,
            geodesic_path: int,
            reference_path: list = None
    ):
        """
        Calculates the navigation metrics at a given step.
        """
        metrics = {}

        self.path_calculator.requested_start = agent_state.position
        distance_to_goal = self.simWrapper.get_path(self.path_calculator)

        success_threshold = self.env_cfg.get('success_threshold', 3.0)

        metrics['success_threshold'] = success_threshold
        metrics['distance_to_goal'] = distance_to_goal
        metrics['spl'] = 0.0
        metrics['goal_reached'] = False
        metrics['finish_status'] = 'running'
        metrics['traveled_distance'] = self.agent.traveled_distance
        metrics['done'] = True
        metrics['last_action_type'] = agent_action.type
        metrics['stop_executed'] = agent_action.type == 'stop'

        curr_pos = np.array(agent_state.position, dtype=np.float32)
        dedupe_eps = float(self.env_cfg.get('dtw_dedupe_eps', 1e-1))
        if len(self.agent_trajectory) == 0 or np.linalg.norm(curr_pos - self.agent_trajectory[-1]) > dedupe_eps:
            self.agent_trajectory.append(curr_pos)

        self.oracle_best_distance = min(self.oracle_best_distance, float(distance_to_goal))
        success_now = distance_to_goal < success_threshold
        self.oracle_success = self.oracle_success or success_now
        if success_now:
            spl_now = float(geodesic_path) / max(float(geodesic_path), float(self.agent.traveled_distance))
            self.oracle_best_spl = max(self.oracle_best_spl, spl_now)

        if (success_now and agent_action.type == 'stop'
                and not getattr(self.agent, 'error', False)
                and not getattr(self.agent, 'is_parse_fail', False)):
            metrics['finish_status'] = 'success'
            metrics['goal_reached'] = True
            metrics['spl'] = float(geodesic_path) / max(float(geodesic_path), float(self.agent.traveled_distance))
        else:
            if agent_action.type == 'stop':
                metrics['finish_status'] = 'fp'
            else:
                metrics['finish_status'] = 'max_steps'

        metrics['oracle_navigation_error'] = self.oracle_best_distance
        metrics['oracle_success'] = 1 if self.oracle_success else 0
        metrics['oracle_spl'] = float(self.oracle_best_spl)
        metrics['waypoint_backend'] = 'navmesh_no_sliding'

        ndtw = None
        sdtw = None
        if reference_path is not None and isinstance(reference_path, (list, tuple, np.ndarray)) and len(
                reference_path) > 0:
            if fastdtw is None:

                ndtw, sdtw = None, None
            else:

                P = np.asarray(self.agent_trajectory, dtype=np.float32)

                G = np.asarray(reference_path, dtype=np.float32)
                if G.ndim == 2 and G.shape[1] == 2:
                    G = np.concatenate([G, np.zeros((G.shape[0], 1), dtype=G.dtype)], axis=1)
                elif G.ndim == 2 and G.shape[1] >= 3:
                    G = G[:, :3]

                if len(P) > 0 and len(G) > 0:
                    def _euclidean(a, b):
                        a = np.asarray(a, dtype=np.float32)
                        b = np.asarray(b, dtype=np.float32)
                        return float(np.linalg.norm(b - a, ord=2))

                    dtw_distance, _ = fastdtw(P.tolist(), G.tolist(), dist=_euclidean)

                    eta = float(success_threshold)
                    ndtw_val = np.exp(-float(dtw_distance) / (len(G) * eta))
                    ndtw = float(ndtw_val)

                    sdtw = float(ndtw) if metrics['goal_reached'] else 0.0

        metrics['ndtw'] = ndtw
        metrics['sdtw'] = sdtw

        return metrics

    def _save_episode_results(self, episode_ndx: int, frames, metrics: dict, vlm_responses: list,
                              subtasks: dict = None):
        """
        Save per-episode evaluation results to JSON and frames as MP4 video.
        """
        results_dir = os.path.join(self.env_cfg.get("output_root", "../results"), f"results_{self.env_cfg['name']}")
        os.makedirs(results_dir, exist_ok=True)

        result_path = os.path.join(results_dir, f"episode_{episode_ndx}_results.json")
        video_path = os.path.join(results_dir, f"episode_{episode_ndx}.mp4")

        if isinstance(frames, EpisodeVideoRecorder) or (hasattr(frames, 'last_frame') and hasattr(frames, 'append')):
            if hasattr(frames, 'close'):
                frames.close()
        elif frames and len(frames) > 0:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 1
            writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

            for fr in frames:
                frame = fr
                if frame.dtype != np.uint8:
                    if np.issubdtype(frame.dtype, np.floating) and frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                if frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                writer.write(frame_bgr)
            writer.release()
            pass
        else:
            pass

        def _sanitize_for_json(data, round_floats=True):
            """Recursively convert NumPy types to JSON-native types and format floats."""
            if isinstance(data, (bool, np.bool_)):
                # Python bool subclasses int; preserve JSON true/false rather than 1/0.
                return bool(data)
            elif isinstance(data, dict):
                return {k: _sanitize_for_json(v, round_floats) for k, v in data.items()}
            elif isinstance(data, (list, tuple)):
                return [_sanitize_for_json(item, round_floats) for item in data]
            elif isinstance(data, np.ndarray):
                return _sanitize_for_json(data.tolist(), round_floats)
            elif isinstance(data, (np.floating, float)):
                return round(float(data), 3) if round_floats else float(data)
            elif isinstance(data, (np.integer, int)):
                return int(data)
            return data

        if hasattr(getattr(self, 'agent', None), '_has_teleport_correction'):
            metrics['has_teleport_correction'] = bool(self.agent._has_teleport_correction)

        result_data = {
            "metrics": _sanitize_for_json(metrics),
            "vlm_responses": vlm_responses,
            "agent_trajectory": _sanitize_for_json(getattr(self, 'agent_trajectory', []))
        }
        # Keep evaluation precision separate from rounded display/debug fields.
        result_data['evaluation'] = {
            name: float(metrics[name]) if metrics.get(name) is not None else None
            for name in ('distance_to_goal', 'oracle_navigation_error',
                         'traveled_distance', 'success_threshold')
        }
        ledger = getattr(getattr(self, 'agent', None), '_execution_ledger', [])
        terminal = ledger[-1] if ledger else {}
        result_data['execution_summary'] = {
            'last_action_type': terminal.get('action_type'),
            'last_action_step': int(terminal['step']) if terminal.get('step') is not None else None,
            'stop_executed': terminal.get('action_type') == 'stop',
            'stop_action_count': sum(r.get('action_type') == 'stop' for r in ledger),
        }
        if subtasks is not None:
            result_data["subtask"] = _sanitize_for_json(subtasks)
        if hasattr(getattr(self, 'agent', None), 'navigation_trace') and self.agent.navigation_trace:
            result_data["trace"] = _sanitize_for_json(self.agent.navigation_trace)

        diag_segment = None
        if isinstance(getattr(self, 'current_episode', None), dict):
            diag_segment = self.current_episode.get('diagnostic_segment')
        if diag_segment is None and isinstance(getattr(self, 'all_episodes', None), (list, tuple)):
            if 0 <= episode_ndx < len(self.all_episodes) and isinstance(self.all_episodes[episode_ndx], dict):
                diag_segment = self.all_episodes[episode_ndx].get('diagnostic_segment')
        if diag_segment is not None:
            agent_obj = getattr(self, 'agent', None)
            inst_obj = getattr(agent_obj, 'instruction_obj', None)
            result_data["diagnostic_segment"] = _sanitize_for_json(diag_segment, round_floats=False)
            enabled = getattr(agent_obj, '_instruction_decomposition_enabled', None)
            single_instruction = (
                getattr(inst_obj, 'sub_instruction_keys', None) == ['P1']
                and getattr(inst_obj, 'num_subtasks', None) == 1
                and isinstance(subtasks, dict) and list(subtasks) == ['P1']
            )
            # Preserve initialization-error results without inventing evidence
            # that decomposition was disabled on an agent that never existed.
            result_data["instruction_decomposition_enabled"] = enabled if type(enabled) is bool else None
            result_data["diagnostic_contract_satisfied"] = enabled is False and single_instruction
            result_data["evaluation_scope"] = 'local_diagnostic_only'

        temporary_path = None
        try:
            # Validate the complete payload before touching an existing result. The
            # final traversal also handles nested NumPy values in raw VLM responses,
            # while retaining their precision and the unrounded evaluation fields.
            serialized = json.dumps(_sanitize_for_json(result_data, round_floats=False),
                                    ensure_ascii=False, indent=2)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=results_dir,
                                             prefix=f".episode_{episode_ndx}_", suffix=".tmp",
                                             delete=False) as f:
                temporary_path = f.name
                f.write(serialized)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary_path, result_path)
            temporary_path = None
        except Exception:
            logging.exception("Failed to save episode %s results atomically to %s",
                              episode_ndx, result_path)
            raise
        finally:
            if temporary_path is not None:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass

    def _create_visual_feedback(self, obs, top_down_map, vlm_response, pil_labeled_img, step_count):
        """
        Composite layout: left/center/right views, Depth, Seg, TopDown + info panel + VLM output.
        """
        mapper = self.agent.mapper
        instances = self.agent.instances

        rgb_image = obs.get('color_sensor', np.zeros((480, 640, 3), dtype=np.uint8))
        depth_image = obs.get('depth_sensor', np.zeros((480, 640), dtype=np.float32))
        
        left_rgb = obs.get('left', {}).get('color_sensor', np.zeros_like(rgb_image))
        right_rgb = obs.get('right', {}).get('color_sensor', np.zeros_like(rgb_image))
        
        if left_rgb.shape[2] == 4: left_rgb = left_rgb[:, :, :3]
        if right_rgb.shape[2] == 4: right_rgb = right_rgb[:, :, :3]
        left_bgr = left_rgb[:, :, ::-1] if left_rgb.max() > 1 else (left_rgb * 255).astype(np.uint8)[:, :, ::-1]
        right_bgr = right_rgb[:, :, ::-1] if right_rgb.max() > 1 else (right_rgb * 255).astype(np.uint8)[:, :, ::-1]

        seg_vis = visualize_instance_segmentation(rgb_image, instances, alpha=0.5)
        seg_vis_bgr = seg_vis if seg_vis.shape[2] == 4 else seg_vis

        if pil_labeled_img is not None:
            center_bgr = cv2.cvtColor(np.array(pil_labeled_img), cv2.COLOR_RGB2BGR)
        else:
            center_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR) if rgb_image.shape[2]==3 else cv2.cvtColor(rgb_image[:,:,:3], cv2.COLOR_RGB2BGR)

        depth_display = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_vis = cv2.cvtColor(depth_display, cv2.COLOR_GRAY2BGR)

        map_raw = top_down_map

        def ensure_bgr_uint8(img):
            if img.dtype != np.uint8:
                if np.issubdtype(img.dtype, np.floating) and img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            if img.ndim == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            return img

        candidates = [ensure_bgr_uint8(img) for img in [left_bgr, center_bgr, right_bgr, depth_vis, seg_vis_bgr, map_raw]]
        max_h = max(img.shape[0] for img in candidates)
        max_w = max(img.shape[1] for img in candidates)
        max_cell_w = int(self.env_cfg.get('visual_feedback_cell_max_w', 1024))
        max_cell_h = int(self.env_cfg.get('visual_feedback_cell_max_h', 1024))

        def resize_with_padding(img, target_w, target_h, image_type='photo'):
            """Center-paste preserving aspect ratio; choose interpolation by content type."""
            img = ensure_bgr_uint8(img)
            h, w = img.shape[:2]
            scale = min(target_w / w, target_h / h)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))

            if new_w == w and new_h == h:
                resized = img
            else:
                is_downscale = new_w < w or new_h < h
                if image_type == 'map':
                    interpolation = cv2.INTER_AREA if is_downscale else cv2.INTER_NEAREST
                elif image_type == 'mask':
                    interpolation = cv2.INTER_NEAREST if not is_downscale else cv2.INTER_AREA
                else:
                    interpolation = cv2.INTER_AREA if is_downscale else cv2.INTER_CUBIC
                resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            y0 = (target_h - new_h) // 2
            x0 = (target_w - new_w) // 2
            canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
            return canvas

        grid_w, grid_h = min(max_w, max_cell_w), min(max_h, max_cell_h)

        left_cell = resize_with_padding(left_bgr, grid_w, grid_h, 'photo')
        center_cell = resize_with_padding(center_bgr, grid_w, grid_h, 'photo')
        right_cell = resize_with_padding(right_bgr, grid_w, grid_h, 'photo')
        depth_cell = resize_with_padding(depth_vis, grid_w, grid_h, 'photo')
        seg_cell = resize_with_padding(seg_vis_bgr, grid_w, grid_h, 'mask')
        map_cell = resize_with_padding(map_raw, grid_w, grid_h, 'map')

        def add_label(img, text):
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return img

        left_cell = add_label(left_cell, "Left View")
        center_cell = add_label(center_cell, "Front View (Labeled)")
        right_cell = add_label(right_cell, "Right View")
        depth_cell = add_label(depth_cell, "Depth")
        seg_cell = add_label(seg_cell, "Segmentation")
        map_cell = add_label(map_cell, "Top-Down Map")

        grid_img = np.zeros((grid_h * 2, grid_w * 3, 3), dtype=np.uint8)
        grid_img[0:grid_h, 0:grid_w] = left_cell
        grid_img[0:grid_h, grid_w:grid_w * 2] = center_cell
        grid_img[0:grid_h, grid_w * 2:grid_w * 3] = right_cell
        grid_img[grid_h:grid_h * 2, 0:grid_w] = map_cell
        grid_img[grid_h:grid_h * 2, grid_w:grid_w * 2] = seg_cell
        grid_img[grid_h:grid_h * 2, grid_w * 2:grid_w * 3] = depth_cell

        max_text_width = grid_img.shape[1] - 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_info = 0.65
        thickness_info = 2

        agent_pos = mapper.current_position
        floor_level = mapper.floor_level
        info_text = f"Position: ({agent_pos[0]:.2f}, {agent_pos[1]:.2f}, {agent_pos[2]:.2f}) | Floor: {floor_level} | Step: {step_count + 1}"

        instruction_raw = f"Instruction: {self.agent.instruction}"
        instruction_lines = wrap_text(instruction_raw, font, font_scale_info, thickness_info, max_text_width)
        max_instruction_lines = 3
        if len(instruction_lines) > max_instruction_lines:
            instruction_lines = instruction_lines[:max_instruction_lines]
            last = instruction_lines[-1]
            instruction_lines[-1] = (last[:-3] + "...") if len(last) > 3 else "..."

        (_, text_h), _ = cv2.getTextSize("A", font, font_scale_info, thickness_info)
        line_height = text_h + 8
        info_panel_height = 28 + line_height * len(instruction_lines)
        info_panel = np.full((info_panel_height, grid_img.shape[1], 3), 200, dtype=np.uint8)

        cv2.putText(info_panel, info_text, (10, 25), font, font_scale_info, (0, 0, 0), thickness_info)
        for i, line in enumerate(instruction_lines):
            y = 25 + (i + 1) * line_height
            cv2.putText(info_panel, line, (10, y), font, font_scale_info, (0, 0, 0), thickness_info)

        vlm_prefix = f"VLM: {vlm_response}"
        vlm_lines = wrap_text(vlm_prefix, font, font_scale_info, thickness_info, max_text_width)
        max_vlm_lines = 9
        if len(vlm_lines) > max_vlm_lines:
            vlm_lines = vlm_lines[:max_vlm_lines]
            vlm_lines[-1] = vlm_lines[-1][:max(0, len(vlm_lines[-1]) - 3)] + "..."

        vlm_panel_height = line_height * max_vlm_lines + 12
        vlm_panel = np.full((vlm_panel_height, grid_img.shape[1], 3), 230, dtype=np.uint8)
        for i, line in enumerate(vlm_lines):
            y = 10 + (i + 1) * line_height - 8
            cv2.putText(vlm_panel, line, (10, y), font, font_scale_info, (50, 50, 50), thickness_info)

        final_view = np.vstack((grid_img, info_panel, vlm_panel))
        
        cv2.waitKey(1)
        
        return final_view

class ObjectNavEnv(Env):
    def _initialize_experiment(self):
        """
        Initializes the experiment by setting up the dataset split, scene configuration, and goals.
        """
        self.all_episodes = []
        self.sim_cfg['scene_config'] = "../data/scene_datasets/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json"
        self.goals = {}

        for f in sorted(os.listdir(f'../data/datasets/objectnav_hm3d_v2/{self.split}/content')):
            if not f.endswith('.gz'):
                continue
            with gzip.open(f'../data/datasets/objectnav_hm3d_v2/{self.split}/content/{f}', 'rt') as gz:
                js = json.load(gz)
                hsh = f.split('.')[0]
                self.goals[hsh] = js['goals_by_category']
                self.all_episodes += js['episodes']

        self.num_episodes = len(self.all_episodes)
        logging.info(f"Number of episodes: {self.num_episodes}")

    def _initialize_episode(self, episode_ndx: int):
        """
        Initializes the episode for the ObjectNav task.

        Args:
            episode_ndx (int): The index of the episode to initialize.
        """
        super()._initialize_episode(episode_ndx)
        episode = self.all_episodes[episode_ndx]
        f = episode['scene_id'].split('/')[1:]
        self.sim_cfg['scene_id'] = f[1][2:5]
        self.sim_cfg['scene_path'] = f'../data/scene_datasets/hm3d_v0.2/{self.split}/{f[1]}/{f[2]}'
        self.simWrapper = SimWrapper(self.sim_cfg)

        goals = self.goals[f[1][6:]]
        all_objects = goals[f'{f[-1]}_{episode["object_category"]}']
        view_positions = []
        for obj in all_objects:
            for vp in obj['view_points']:
                view_positions.append(vp['agent_state']['position'])
        self.path_calculator.requested_ends = np.array(view_positions, dtype=np.float32)
        logging.info(f'RUNNING EPISODE {episode_ndx} with {episode["object_category"]} and {len(all_objects)} instances. GEODESIC DISTANCE: {episode["info"]["geodesic_distance"]}')
        if episode['object_category'] == 'tv_monitor':
            episode['object_category'] = 'tv screen'
        self.current_episode = {
            'object': episode['object_category'],
            'shortest_path': episode['info']['geodesic_distance'],
            'object_positions': [a['position'] for a in all_objects],
            'view_positions': view_positions
        }
        self.init_pos = np.array(episode['start_position'])
        self.init_rot = np.array(episode['start_rotation'])
        instruction = f"Navigate to the target object '{episode['object_category']}' and get as close to it as possible. The distance should be close enough to reach the target object."
        self.agent = GPTAgent(self.simWrapper, self.sim_cfg, instruction, initial_position=self.init_pos,
                         initial_rotation=self.init_rot, mode='objnav',
                         model_name=self.sim_cfg.get("model_name"))

        self.curr_run_name = f"{episode_ndx}_{self.simWrapper.scene_id}"

        obs = self.simWrapper.step(PolarAction.null)
        return obs

    def _run_episode(self, episode_ndx: int, show_visualization: bool = True):
        """
        Run an ObjectNav episode with exception protection and unified cleanup.
        """
        import traceback

        obs = self._initialize_episode(episode_ndx)

        win_name = "Agent Control View"
        if show_visualization:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        frames = []
        max_steps = self.env_cfg['max_steps']
        metrics = {}
        error_info = None

        try:
            for step_count in range(max_steps):
                pass
                self.step = step_count
                visual_key = -1

                obs, top_down_map, vlm_response, pil_labeled_img = self.agent.step(step_count)

                final_view = self._create_visual_feedback(
                    obs, top_down_map, vlm_response, pil_labeled_img, step_count
                )
                if show_visualization:
                    cv2.imshow(win_name, final_view)
                    visual_key = cv2.waitKey(1) & 0xFF
                    print("action:", self.agent.last_action.type)

                rgb_frame = cv2.cvtColor(final_view, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)

                agent_state = obs['agent_state']
                metrics = self._calculate_metrics(
                    agent_state,
                    self.agent.last_action,
                    self.current_episode['shortest_path'],
                    max_steps,
                )

                user_quit = self._user_requested_quit()
                if show_visualization and visual_key == ord('q'):
                    self._request_user_quit()
                    user_quit = True

                if user_quit:
                    pass
                    metrics['finish_status'] = 'user_interrupt'
                    self._save_episode_results(
                        episode_ndx, frames, metrics, getattr(self.agent, "vlm_responses", [])
                    )
                    if getattr(self, "agent", None):
                        self.agent.error = False
                        self.agent.is_parse_fail = False
                    break

                should_stop = False
                if getattr(self.agent, "second_stop", False):
                    pass
                    if show_visualization:
                        cv2.waitKey(500)
                    should_stop = True
                elif step_count == max_steps - 1:
                    pass
                    if metrics.get('finish_status') != 'success' and not bool(metrics.get('goal_reached', False)):
                        metrics['finish_status'] = 'max_steps'
                    should_stop = True

                if should_stop:
                    if metrics.get('finish_status') == 'success' or bool(metrics.get('goal_reached', False)):
                        metrics['finish_status'] = 'success'
                        metrics['goal_reached'] = True
                    elif getattr(self.agent, "is_parse_fail", False) or getattr(self.agent, "error", False):
                        metrics['finish_status'] = 'error'
                        if not metrics.get('error_info'):
                            metrics['error_info'] = {
                                'reason': 'agent_parse_fail',
                                'step': step_count,
                                'fail_count': getattr(self.agent, '_decision_fail_count', None)
                            }
                    self._save_episode_results(
                        episode_ndx, frames, metrics, self.agent.vlm_responses
                    )
                    if getattr(self, "agent", None):
                        self.agent.error = False
                        self.agent.is_parse_fail = False
                    break

        except Exception as e:

            tb = traceback.format_exc()
            pass
            error_info = {"exc": str(e), "traceback": tb}

            metrics = _populate_error_metrics(metrics, error_info, env=self)
            try:
                self._save_episode_results(
                    episode_ndx, frames, metrics, getattr(self.agent, "vlm_responses", [])
                )
            except Exception as save_e:
                pass

        finally:

            if show_visualization:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

            try:
                if getattr(self, "agent", None):
                    self.agent.error = False
                    self.agent.is_parse_fail = False
                    if getattr(self.agent, "sim_wrapper", None):
                        self.agent.sim_wrapper.close()
                    if hasattr(self.agent, "close"):
                        self.agent.close()
            except Exception:
                pass

        return metrics

class R2REnv(Env):
    def _initialize_experiment(self):
        """
        Initialize experiment: load R2R-CE dataset, set scene config, init episodes.
        """
        dataset_path = self.env_cfg.get('dataset_path') or os.environ.get('HSGM_DATASET_PATH') or '../data/datasets/r2r_vlnce/val_unseen.json'

        self.all_episodes = []
        try:
            with open(dataset_path, 'r') as f:
                data = json.load(f)

                self.all_episodes = data['episodes']
        except FileNotFoundError:
            pass
            raise
        except json.JSONDecodeError:
            pass
            raise

        self.num_episodes = len(self.all_episodes)
        pass

    def _initialize_episode(self, episode_ndx: int):
        """
        Initialize a single episode by index for the R2R task.
        Parse scene path, initial pose, goal, and instruction from the dataset.
        """
        super()._initialize_episode(episode_ndx)
        episode = self.all_episodes[episode_ndx]
        diag_segment = episode.get('diagnostic_segment')
        if diag_segment is not None:
            if not isinstance(diag_segment, dict):
                raise ValueError('Invalid diagnostic_segment metadata')
            if (diag_segment.get('schema') != 1
                    or diag_segment.get('restore_mode') != 'rounded_pose_cold_start'):
                raise ValueError('Unsupported diagnostic segment schema or restore mode')
            diag_max_steps = diag_segment.get('max_steps', 24)
            if isinstance(diag_max_steps, bool) or not isinstance(diag_max_steps, int) or diag_max_steps <= 0 or diag_max_steps > 24:
                raise ValueError(f'Invalid diagnostic_segment max_steps: {diag_max_steps}')
            env_cfg_dict = getattr(self, 'env_cfg', None)
            cfg_max_steps = env_cfg_dict.get('max_steps') if isinstance(env_cfg_dict, dict) else None
            attr_max_steps = getattr(self, 'max_steps', None)
            step_limits = [v for v in (cfg_max_steps, attr_max_steps) if v is not None]
            if not step_limits:
                raise ValueError('Diagnostic segment requires explicit max_steps configuration')
            for env_max_steps in step_limits:
                if isinstance(env_max_steps, bool) or not isinstance(env_max_steps, int) or env_max_steps <= 0 or env_max_steps > diag_max_steps:
                    raise ValueError(f'Diagnostic segment requires 0 < max_steps <= {diag_max_steps}, got {env_max_steps}')
            if diag_segment.get('decomposition_enabled', False):
                raise ValueError('Diagnostic segment cannot enable subtask decomposition')

        scene_name = episode['scene_id'].split('/')[1]

        self.sim_cfg['scene_path'] = f'../data/scene_datasets/mp3d/{scene_name}/{scene_name}.glb'
        self.sim_cfg['scene_id'] = scene_name

        if self.simWrapper:
            self.simWrapper.close()
        self.simWrapper = SimWrapper(self.sim_cfg)

        goal_position = np.array(episode['goals'][0]['position'], dtype=np.float32)
        self.path_calculator.requested_ends = np.array([goal_position])

        instruction = episode['instruction']['instruction_text']
        # instruction = "go upstairs"
        pass
        pass

        self.current_episode = {
            'instruction': instruction,
            'shortest_path': episode['info']['geodesic_distance'],
            'goal_position': goal_position,
            'reference_path': episode['reference_path']
        }
        if diag_segment is not None:
            self.current_episode['diagnostic_segment'] = dict(diag_segment)

        if episode.get('info', {}).get('geodesic_distance_requires_recalculation'):
            self.path_calculator.requested_start = np.array(episode['start_position'], dtype=np.float32)
            self.current_episode['shortest_path'] = self.simWrapper.get_path(self.path_calculator)
            if not np.isfinite(self.current_episode['shortest_path']) or self.current_episode['shortest_path'] <= 0 or self.current_episode['shortest_path'] >= 1000:
                raise ValueError('No valid geodesic path for sliced episode')

        self.init_pos = np.array(episode['start_position'])
        self.init_rot = np.array(episode['start_rotation'])

        episode_sim_cfg = dict(self.sim_cfg)
        if diag_segment is not None:
            nav_cfg = dict(self.sim_cfg.get('navigation') or {})
            nav_cfg['disable_subtask_decomposition'] = True
            nav_cfg['enable_subtask_decomposition'] = False
            episode_sim_cfg['navigation'] = nav_cfg

        self.agent = GPTAgent(self.simWrapper, episode_sim_cfg, instruction,
                              initial_position=self.init_pos,
                              initial_rotation=self.init_rot,
                              mode='vln',
                              model_name=episode_sim_cfg.get("model_name"))
        if diag_segment is not None:
            if (self.agent._instruction_decomposition_enabled is not False
                    or self.agent.instruction_obj.sub_instruction_keys != ['P1']
                    or self.agent.instruction_obj.num_subtasks != 1):
                raise ValueError('Diagnostic segment must initialize a single instruction without decomposition')
        self.agent._current_episode_id = episode_ndx
        if hasattr(self.agent, "bind_episode_targets"):
            self.agent.bind_episode_targets(goal_position, episode.get('reference_path'))

        obs = self.simWrapper.step(PolarAction.null)
        return obs

    def _run_episode(self, episode_ndx: int, show_visualization: bool = True):
        """
        Run an R2R episode. Saves results and traceback even on exception during step/processing.
        """
        import traceback

        diagnostic = self.all_episodes[episode_ndx].get('diagnostic_segment')
        if diagnostic is not None:
            # A rejected slice must not inherit the preceding episode's STOP,
            # trajectory or decomposition state in its error result.
            self.agent = None
            self.agent_trajectory = []
            self.current_episode = {'diagnostic_segment': diagnostic}
        try:
            obs = self._initialize_episode(episode_ndx)
        except Exception as exc:
            if diagnostic is None:
                raise
            metrics = {'finish_status': 'error', 'goal_reached': False,
                       'done': True, 'stop_executed': False,
                       'termination_reason': 'diagnostic_initialization_failed',
                       'error_info': {'exc': str(exc), 'traceback': traceback.format_exc()}}
            self._save_episode_results(episode_ndx, [], metrics, [],
                                       getattr(getattr(self.agent, 'instruction_obj', None),
                                               'sub_instruction_dict', None))
            return metrics

        win_name = "R2R Agent Control View"
        if show_visualization:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

        results_dir = os.path.join(self.env_cfg.get("output_root", "../results"), f"results_{self.env_cfg['name']}")
        os.makedirs(results_dir, exist_ok=True)
        video_path = os.path.join(results_dir, f"episode_{episode_ndx}.mp4")
        frames = EpisodeVideoRecorder(video_path, self.env_cfg.get('video_fps', 2))

        max_steps = self.env_cfg.get('max_steps', 100)
        unlimited_steps = int(max_steps) <= 0
        metrics = {}
        error_info = None

        max_ep_sec = float(os.environ.get('HSGM_MAX_EPISODE_SECONDS', self.env_cfg.get('max_episode_seconds', 900)))
        max_mod_req = int(os.environ.get('HSGM_MAX_MODEL_REQUESTS', self.env_cfg.get('max_model_requests', 48)))
        soft_margin_s = float(os.environ.get('HSGM_SOFT_BUDGET_MARGIN_S', '60'))
        budget = EpisodeBudget(max_ep_sec, max_mod_req)
        self.agent.runtime_budget = budget

        try:
            step_count = 0
            while unlimited_steps or step_count < max_steps:
                has_pending_local_step = bool(
                    getattr(self.agent, 'current_path', None)
                    or getattr(self.agent, 'action_sequence', None)
                )
                if step_count > 0 and budget.remaining() <= soft_margin_s and (not has_pending_local_step or budget.remaining() <= 5.0):
                    logging.info(f"[EpisodeBudget] Episode {episode_ndx} reached soft wall-clock margin ({budget.remaining():.1f}s remaining <= {soft_margin_s:.1f}s). Saving results cleanly.")
                    if obs is not None and 'agent_state' in obs:
                        try:
                            metrics = self._calculate_metrics(
                                obs['agent_state'],
                                getattr(self.agent, 'last_action', PolarAction.null),
                                self.current_episode['shortest_path'],
                                self.current_episode.get('reference_path')
                            )
                        except Exception:
                            pass
                    metrics = _budget_termination_metrics(metrics, 'episode_wall_clock_soft_limit')
                    self._save_episode_results(
                        episode_ndx,
                        frames,
                        metrics,
                        getattr(self.agent, "vlm_responses", []),
                        getattr(getattr(self.agent, "instruction_obj", None), "sub_instruction_dict", None)
                    )
                    break
                logging.info(f"--- Episode {episode_ndx}, Step {step_count + 1}/{max_steps} ---")
                self.step = step_count
                visual_key = -1

                try:
                    with budget.guard():
                        obs, top_down_map, vlm_response, pil_labeled_img = self.agent.step(step_count)
                except ModelServiceError:
                    raise
                except Exception as e:
                    tb = traceback.format_exc()
                    error_info = {"exc": str(e), "traceback": tb}
                    metrics = _populate_error_metrics(metrics, error_info, env=self)
                    break

                # Always render and collect frame for video (independent of display)
                try:
                    final_view = self._create_visual_feedback(obs, top_down_map, vlm_response, pil_labeled_img,
                                                              step_count)
                    if show_visualization:
                        cv2.imshow(win_name, final_view)
                        visual_key = cv2.waitKey(1) & 0xFF
                    print("Action:", self.agent.last_action.type)
                    rgb_frame = cv2.cvtColor(final_view, cv2.COLOR_BGR2RGB)
                    frames.append(rgb_frame)
                except Exception as e:
                    tb = traceback.format_exc()
                    error_info = {"exc": str(e), "traceback": tb}
                    metrics = _populate_error_metrics(metrics, error_info, env=self)
                    break

                try:
                    agent_state = obs['agent_state']
                    metrics = self._calculate_metrics(
                        agent_state,
                        self.agent.last_action,
                        self.current_episode['shortest_path'],
                        self.current_episode.get('reference_path')
                    )
                except Exception as e:
                    tb = traceback.format_exc()
                    error_info = {"exc": str(e), "traceback": tb}
                    metrics = _populate_error_metrics(metrics, error_info, env=self)
                    break

                user_quit = self._user_requested_quit()
                if show_visualization and visual_key == ord('q'):
                    self._request_user_quit()
                    user_quit = True

                if user_quit:
                    metrics['finish_status'] = 'user_interrupt'
                    self._save_episode_results(
                        episode_ndx,
                        frames,
                        metrics,
                        getattr(self.agent, "vlm_responses", []),
                        getattr(getattr(self.agent, "instruction_obj", None), "sub_instruction_dict", None)
                    )
                    if getattr(self, "agent", None):
                        self.agent.error = False
                        self.agent.is_parse_fail = False
                    break

                should_stop = False
                if getattr(self.agent, "second_stop", False):
                    if show_visualization:
                        cv2.waitKey(500)
                    should_stop = True
                elif step_count == max_steps - 1:
                    if metrics.get('finish_status') != 'success' and not bool(metrics.get('goal_reached', False)):
                        metrics['finish_status'] = 'max_steps'
                    should_stop = True

                if should_stop:
                    if metrics.get('finish_status') == 'success' or bool(metrics.get('goal_reached', False)):
                        metrics['finish_status'] = 'success'
                        metrics['goal_reached'] = True
                    elif getattr(self.agent, "is_parse_fail", False) or getattr(self.agent, "error", False):
                        metrics['finish_status'] = 'error'
                        if not metrics.get('error_info'):
                            metrics['error_info'] = {
                                'reason': 'agent_parse_fail',
                                'step': step_count,
                                'fail_count': getattr(self.agent, '_decision_fail_count', None)
                            }
                    self._save_episode_results(
                        episode_ndx,
                        frames,
                        metrics,
                        getattr(self.agent, "vlm_responses", []),
                        getattr(getattr(self.agent, "instruction_obj", None), "sub_instruction_dict", None)
                    )
                    if getattr(self, "agent", None):
                        self.agent.error = False
                        self.agent.is_parse_fail = False
                    break

                step_count += 1

        except EpisodeBudgetExceeded as e:
            is_aborted = isinstance(e, NavigationAborted)
            if obs is not None and 'agent_state' in obs:
                try:
                    metrics = self._calculate_metrics(
                        obs['agent_state'],
                        getattr(self.agent, 'last_action', PolarAction.null),
                        self.current_episode['shortest_path'],
                        self.current_episode.get('reference_path')
                    )
                except Exception:
                    pass
            metrics = _budget_termination_metrics(metrics, getattr(e, 'reason', str(e)), is_aborted)
            if hasattr(self.agent, 'current_path'):
                self.agent.current_path = []
            if hasattr(self.agent, 'action_sequence'):
                self.agent.action_sequence = []
            if hasattr(self.agent, 'navigation_trace') and isinstance(self.agent.navigation_trace, list):
                self.agent.navigation_trace.append({'role': 'controller', 'event': metrics['finish_status'],
                                                    'reason': getattr(e, 'reason', str(e)),
                                                    **getattr(budget, 'summary', lambda: {})()})
            self._save_episode_results(
                episode_ndx,
                frames,
                metrics,
                getattr(self.agent, 'vlm_responses', []),
                getattr(getattr(self.agent, "instruction_obj", None), "sub_instruction_dict", None)
            )

        except Exception as e:
            tb = traceback.format_exc()
            error_info = {"exc": str(e), "traceback": tb}
            metrics = _populate_error_metrics(metrics, error_info, env=self)

        finally:
            if error_info is not None:
                try:
                    self._save_episode_results(
                        episode_ndx,
                        frames,
                        metrics,
                        getattr(self.agent, "vlm_responses", []),
                        getattr(getattr(self.agent, "instruction_obj", None), "sub_instruction_dict", None)
                    )
                except Exception as save_e:
                    pass
            if hasattr(frames, 'close'):
                frames.close()

            if show_visualization:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

            try:
                if self.agent:
                    self.agent.error = False
                    self.agent.is_parse_fail = False
                    if getattr(self.agent, "sim_wrapper", None):
                        self.agent.sim_wrapper.close()
                    if hasattr(self.agent, "close"):
                        self.agent.close()
            except Exception:
                pass

        return metrics

class RxREnv(R2REnv):
    def _initialize_experiment(self):
        """
        Load RxR dataset and GT file, filtering to target language (default: 'en-IN') episodes.
        GT file format example:
        {
          "4705": {
            "locations": [[x,y,z], ...],
            "actions": [...],
            "forward_steps": int
          },
          ...
        }
        """
        dataset_path = self.env_cfg.get('rxr_dataset', '../data/datasets/RxR_VLNCE_v0/val_unseen/val_unseen_guide.json')
        dataset_gt_path = self.env_cfg.get('rxr_gt_dataset', '../data/datasets/RxR_VLNCE_v0/val_unseen/val_unseen_guide_gt.json')
        target_lang = self.env_cfg.get('rxr_language', 'en-IN')

        self.all_episodes = []
        try:
            with open(dataset_path, 'r') as f:
                data = json.load(f)
                episodes = data.get('episodes', [])
        except FileNotFoundError:
            pass
            raise
        except json.JSONDecodeError:
            pass
            raise

        filtered = []
        for ep in episodes:
            instr = ep.get('instruction', {})

            if isinstance(instr, dict):
                if instr.get('language') == target_lang:
                    filtered.append(ep)

            # elif isinstance(instr, list):
            #     for item in instr:
            #         if isinstance(item, dict) and item.get('language') == target_lang:
            #             ep_copy = dict(ep)
            #             ep_copy['instruction'] = item
            #             filtered.append(ep_copy)
            #             break

        self.all_episodes = filtered
        self.num_episodes = len(self.all_episodes)
        pass

        try:
            with open(dataset_gt_path, 'r') as f:
                self.rxr_gt = json.load(f)
            pass
        except FileNotFoundError:
            pass
            self.rxr_gt = {}
        except json.JSONDecodeError:
            pass
            self.rxr_gt = {}

        try:
            if getattr(self, "rxr_gt", None):

                keep_ids = set()
                for ep in self.all_episodes:
                    ep_id = ep.get('episode_id') or ep.get('instruction', {}).get('instruction_id') or ep.get('id')
                    if ep_id is not None:
                        keep_ids.add(str(ep_id))

                filtered_gt = {str(k): v for k, v in self.rxr_gt.items() if str(k) in keep_ids}

                filtered_gt_path = self.env_cfg.get(
                    'rxr_gt_filtered_output',
                    os.path.splitext(dataset_gt_path)[0] + f'_{target_lang}_filtered.json'
                )
                out_dir = os.path.dirname(filtered_gt_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)

                with open(filtered_gt_path, 'w', encoding='utf-8') as f:
                    json.dump(filtered_gt, f, ensure_ascii=False, indent=2)

                missing = len(keep_ids) - len(filtered_gt)
                pass
        except Exception as e:
            pass

    def _initialize_episode(self, episode_ndx: int):
        """
        Initialize an RxR episode: parse scene, initial pose, goal, instruction; use GT locations as reference_path.
        """
        super()._pre_init_episode()
        episode = self.all_episodes[episode_ndx]

        scene_name = episode['scene_id'].split('/')[1]
        self.sim_cfg['scene_path'] = f'../data/scene_datasets/mp3d/{scene_name}/{scene_name}.glb'
        self.sim_cfg['scene_id'] = scene_name

        if self.simWrapper:
            self.simWrapper.close()
        self.simWrapper = SimWrapper(self.sim_cfg)

        goals = episode.get('goals', [])
        goal_positions = []
        for g in goals:
            if 'position' in g:
                goal_positions.append(np.array(g['position'], dtype=np.float32))
        if not goal_positions:
            pass
            raise ValueError("RxR episode goals missing position.")
        self.path_calculator.requested_ends = np.asarray(goal_positions, dtype=np.float32)

        instruction = episode['instruction']['instruction_text']
        pass

        self.init_pos = np.array(episode['start_position'], dtype=np.float32)
        self.init_rot = np.array(episode['start_rotation'], dtype=np.float32)

        self.path_calculator.requested_start = self.init_pos
        shortest_geo = self.simWrapper.get_path(self.path_calculator)

        ep_id = str(
            episode.get('episode_id')
            or episode.get('episodeId')
            or episode.get('id')
        )
        gt_entry = self.rxr_gt.get(ep_id, None)
        if gt_entry is None:
            pass
            reference_path = None
            gt_actions = None
            gt_forward_steps = None
        else:
            locations = gt_entry.get('locations', [])
            reference_path = np.asarray(locations, dtype=np.float32) if len(locations) > 0 else None
            gt_actions = gt_entry.get('actions', None)
            gt_forward_steps = gt_entry.get('forward_steps', None)
            if reference_path is None:
                pass

        self.current_episode = {
            'instruction': instruction,
            'shortest_path': float(shortest_geo),
            'goal_position': goal_positions[0],
            'reference_path': reference_path,
            'gt_actions': gt_actions,
            'gt_forward_steps': gt_forward_steps
        }

        self.agent = GPTAgent(
            self.simWrapper,
            self.sim_cfg,
            instruction,
            initial_position=self.init_pos,
            initial_rotation=self.init_rot,
            mode='vln',
            model_name=self.sim_cfg.get("model_name")
        )
        if hasattr(self.agent, "bind_episode_targets"):
            self.agent.bind_episode_targets(goal_positions[0], reference_path)

        obs = self.simWrapper.step(PolarAction.null)
        return obs

def _parse_episode_ids(value: str) -> list[int]:
    """
    Supports two input formats:
    1\) comma-separated: 13,18,42
    2\) file path: one id per line (blank lines and \
    """
    p = Path(value)
    if p.exists() and p.is_file():
        ids: list[int] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ids.append(int(s))
        return ids

    parts = [x.strip() for x in value.split(",") if x.strip()]
    return [int(x) for x in parts]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run navigation experiments.")
    parser.add_argument("--task", type=str, default="r2r", choices=["objectnav", "r2r", "rxr"],
                        help="Task to run: 'objectnav', 'r2r', or 'rxr'.")
    parser.add_argument("--comment", type=str, default="exp", help="Optional comment for the experiment.")
    parser.add_argument("--config", type=str, default="../config/vlnce_test.yaml",
                        help="Path to the configuration file.")
    parser.add_argument("--begin_idx", type=int, default=0, help="Start episode index (inclusive).")
    parser.add_argument("--end_idx", type=int, default=-1, help="End episode index (exclusive). Use -1 for all.")
    parser.add_argument("--show", action="store_true", help="Show visualization during the run (requires a display).")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum number of turns for the agent.")
    parser.add_argument("--model_name", type=str, default=None,
                        help="VLM model name (default: $OPENAI_MODEL or gpt-5).")
    parser.add_argument(
        "--episode_ids",
        type=_parse_episode_ids,
        default=None,
        help="Only run specific episode ids. Accepts `1,2,3` or a file path with one id per line."
    )
    parser.add_argument("--dataset_path", default=None, help="Optional R2R dataset; also HSGM_DATASET_PATH")
    parser.add_argument("--memory_storage_path", default=None, help="Optional isolated memory storage with config memory settings")
    parser.add_argument("--output_root", default=None, help="Root for all result JSON, aggregates and videos")
    parser.add_argument("--success_threshold", type=float, default=None, help="Success radius in meters (default unchanged)")
    parser.add_argument("--no_subtask_decomposition", "--no-subtask-decomposition", dest="no_subtask_decomposition",
                        action="store_true", help="Hard-disable LLM instruction subtask decomposition.")
    args = parser.parse_args()
    if args.success_threshold is not None and args.success_threshold <= 0:
        parser.error("success_threshold must be positive")

    config_path = args.config
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        pass
        exit()

    res_factor = config.get("camera", {}).get("res_factor", 2)
    width = 1920 // res_factor
    height = 1080 // res_factor
    hfov = config.get("camera", {}).get("fov", 90.0)
    fx = width / (2 * np.tan(np.deg2rad(hfov / 2)))
    fy = fx

    sim_config = {
        "allow_slide": config.get("sim", {}).get("allow_slide", False),
        "use_goal_image_agent": False,
        "agent": config.get("agent", {}),
        "agent_radius": config.get("agent", {}).get("radius", 0.15),
        "agent_height": config.get("agent", {}).get("height", 1.5),
        "sensor_cfg": {
            "pitch": config.get("camera", {}).get("pitch", 0.0),
            "fov": hfov,
            "height": config.get("camera", {}).get("height", 1.2),
            "res_factor": res_factor
        },
        "camera": {
            "min_depth": config.get("camera", {}).get("min_depth", 0.5),
            "max_depth": config.get("camera", {}).get("max_depth", 5.0),
            "fov": hfov, "fx": fx, "fy": fy, "width": width, "height": height
        },
        "model_name": args.model_name,
        "navigation": dict(config.get("navigation", {})),
        "memory": dict(config.get("memory", {})),
        "map": dict(config.get("map", {})),
    }

    if args.no_subtask_decomposition:
        sim_config["navigation"]["disable_subtask_decomposition"] = True
        sim_config["navigation"]["enable_subtask_decomposition"] = False

    if args.memory_storage_path:
        sim_config["memory"] = dict(config.get("memory", {}))
        sim_config["memory"]["storage_path"] = str(Path(args.memory_storage_path).resolve())

    env_config = config.get("env", {})
    env_config["name"] = f"{args.task}_exp_{args.comment}_{args.begin_idx}_{args.end_idx if args.end_idx != -1 else 'end'}"
    env_config["max_steps"] = args.max_steps
    if args.dataset_path:
        env_config["dataset_path"] = str(Path(args.dataset_path).resolve())
    if args.output_root:
        env_config["output_root"] = str(Path(args.output_root).resolve())
    if args.success_threshold is not None:
        env_config["success_threshold"] = args.success_threshold

    if args.task == "objectnav":
        pass
        sim_config["scene_id"] = config.get("sim", {}).get("scene_id", "default")
        sim_config["scene_path"] = config.get("sim", {}).get("scene_path")
        sim_config["scene_config"] = config.get("sim", {}).get("scene_config")
        env = ObjectNavEnv(sim_cfg=sim_config, env_cfg=env_config)
        pass

    elif args.task == "r2r":
        pass
        env = R2REnv(sim_cfg=sim_config, env_cfg=env_config)
        pass

    elif args.task == "rxr":
        pass
        env = RxREnv(sim_cfg=sim_config, env_cfg=env_config)
        pass

    else:
        pass
        exit()

    env.run_experiment(
        begin_idx=args.begin_idx,
        end_idx=args.end_idx,
        show_visualization=args.show,
        episode_ids=args.episode_ids,
    )
    pass


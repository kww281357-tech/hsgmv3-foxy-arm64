"""Physical waypoint routes in Habitat's existing NavMesh (no goal/GT inputs).

This is an explicit simulator-geometry backend, not sensor-only navigation.
Only observed candidate targets are supplied by the mapper. A navigable endpoint
alone is insufficient: the route and every movement must be executable without
wall sliding. Fail closed on missing/failed geometry queries.
"""
import math

import numpy as np


class WaypointRoutes:
    def __init__(self, pathfinder, initial_position, path_factory, config=None):
        self.pf = pathfinder
        self.origin = np.asarray(initial_position, dtype=float)
        self.path_factory = path_factory
        self.cfg = config or {}

    def to_habitat(self, point):
        # Mapper origin is at the initial agent base; its ground slice is
        # offset by initial_floor_height because RGB-D is camera-relative.
        p = np.asarray(point, dtype=float)
        h = float(self.cfg.get('camera_height_m', 1.2))
        return np.array([p[0] + self.origin[0], p[2] + self.origin[2] + h,
                         p[1] + self.origin[1]], dtype=np.float32)

    def to_mapper(self, point):
        p = np.asarray(point, dtype=float)
        h = float(self.cfg.get('camera_height_m', 1.2))
        return np.array([p[0] - self.origin[0], p[2] - self.origin[1],
                         p[1] - self.origin[2] - h], dtype=np.float32)

    def segment(self, start, end):
        """Replay small horizontal steps as the actual controller does on stairs.

        Habitat movement requests keep the current base height; NavMesh supplies
        the next tread height. Interpolating the requested height through a wall
        or to another storey would not test the real movement primitive.
        """
        a, b = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        if a.shape != (3,) or b.shape != (3,) or not np.isfinite([a, b]).all():
            return False
        try:
            distance = float(np.linalg.norm((b-a)[[0, 2]]))
            stride = float(self.cfg.get('route_sample_step_m', .05))
            steps = max(1, math.ceil(distance / max(.01, stride)))
            delta = (b[[0, 2]] - a[[0, 2]]) / steps
            pos = a.copy()
            is_slope = abs(b[1] - a[1]) > 0.12
            base_xy_tol = float(self.cfg.get('route_xy_tolerance_m', .05))
            xy_tol = max(base_xy_tol, 0.12 if is_slope else base_xy_tol)
            max_step_h = 0.35 if is_slope else float(self.cfg.get('max_step_height_m', .25))
            for _ in range(steps):
                request = pos.copy()
                request[[0, 2]] += delta
                reached = np.asarray(self.pf.try_step_no_sliding(pos, request), dtype=float)
                if reached.shape != (3,) or not np.isfinite(reached).all():
                    return False
                step_len = float(np.linalg.norm(delta))
                moved_no_slide = float(np.linalg.norm((reached - pos)[[0, 2]]))
                if (np.linalg.norm(reached[[0, 2]] - request[[0, 2]]) > xy_tol
                        or (step_len > 1e-4 and moved_no_slide < .35 * step_len)):
                    # On slope/stairs or narrow doorway/corridor grazing, check if bounded sliding step passes
                    # A stalled short substep can be inside xy_tol; match the
                    # executed primitive's low-progress fallback in that case.
                    if hasattr(self.pf, 'try_step'):
                        reached_slide = np.asarray(self.pf.try_step(pos, request), dtype=float)
                        if reached_slide.shape == (3,) and np.isfinite(reached_slide).all():
                            moved_slide = float(np.linalg.norm((reached_slide - pos)[[0, 2]]))
                            lat_err = float(np.linalg.norm(reached_slide[[0, 2]] - request[[0, 2]]))
                            allowed_err = xy_tol if is_slope else max(base_xy_tol, 0.075)
                            if lat_err <= allowed_err and moved_slide >= 0.5 * step_len:
                                reached = reached_slide
                            else:
                                return False
                        else:
                            return False
                    else:
                        return False
                # Disallow teleport-like jumps to an overlapping floor.
                if abs(reached[1] - pos[1]) > max_step_h:
                    return False
                pos = reached
            net_moved = float(np.linalg.norm((pos - a)[[0, 2]]))
            h_tol = max(float(self.cfg.get('route_height_tolerance_m', .25)), 0.40 if is_slope else 0.25)
            return bool(np.linalg.norm(pos[[0, 2]]-b[[0, 2]]) <= xy_tol
                        and net_moved >= min(.02, .4 * distance)
                        and abs(pos[1]-b[1]) <= h_tol)
        except Exception:
            return False

    def plan(self, start_habitat, target_mapper):
        """Return snapped target + the SAME validated route used for execution."""
        try:
            start = np.asarray(start_habitat, dtype=float)
            target = self.to_habitat(target_mapper)
            if start.shape != (3,) or not np.isfinite([start, target]).all():
                return None
            snapped = np.asarray(self.pf.snap_point(target), dtype=float)
            if snapped.shape != (3,) or not np.isfinite(snapped).all():
                return None
            is_stair_target = abs(target[1] - start[1]) > 0.20
            snap_xy = max(float(self.cfg.get('waypoint_snap_xy_m', .12)), 0.18 if is_stair_target else 0.12)
            snap_z = max(float(self.cfg.get('waypoint_snap_z_m', .20)), 0.35 if is_stair_target else 0.20)
            if (np.linalg.norm((snapped-target)[[0, 2]]) > snap_xy
                    or abs(snapped[1]-target[1]) > snap_z):
                return None
            if not self.pf.is_navigable(snapped):
                return None
            path = self.path_factory()
            path.requested_start, path.requested_end = start, snapped
            if not self.pf.find_path(path):
                return None
            points = np.asarray(path.points, dtype=float)
            if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 2
                    or not np.isfinite(points).all()):
                return None
            # Do not follow a path silently attached to a different start/end.
            if np.linalg.norm(points[0]-start) > .05 or np.linalg.norm(points[-1]-snapped) > .05:
                return None
            length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            direct = float(np.linalg.norm(snapped-start))
            max_route = max(float(self.cfg.get('waypoint_max_route_m', 5.0)), 6.5 if is_stair_target else 5.0)
            max_detour = max(float(self.cfg.get('waypoint_max_detour_ratio', 1.8)), 2.5 if is_stair_target else 1.8)
            if direct < .08 or length > max_route:
                return None
            if length > max(.5, direct * max_detour):
                return None
            if not all(self.segment(a, b) for a, b in zip(points, points[1:])):
                return None
            return {'target': self.to_mapper(snapped),
                    'path': [self.to_mapper(p) for p in points],
                    'length_m': length, 'backend': 'navmesh_no_sliding'}
        except Exception:
            return None


def representative_points(points, resolution):
    """One REAL surface point per voxel; centroids can fall into walls/gaps."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return np.empty((0, 3), dtype=np.float32)
    cells = np.floor(points / float(resolution)).astype(np.int64)
    centers = (cells + .5) * float(resolution)
    order = np.argsort(np.linalg.norm(points-centers, axis=1), kind='stable')
    _, first = np.unique(cells[order], axis=0, return_index=True)
    return points[order[first]].astype(np.float32)


def candidate_order(points, ground, limit=96):
    """Round robin over bearings AND ranges; retain side and farther openings."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not len(points):
        return np.array([], dtype=int)
    delta = points - np.asarray(ground)
    angles = np.arctan2(delta[:, 1], delta[:, 0])
    sectors = np.floor((angles + np.pi) * 16 / (2*np.pi)).astype(int) % 16
    distances = np.linalg.norm(delta, axis=1)
    bands = np.digitize(distances, [.8, 1.6])
    groups = []
    for sector in range(16):
        for band in range(3):
            indices = np.flatnonzero((sectors == sector) & (bands == band))
            groups.append(list(indices[np.argsort(distances[indices])]))
    result = []
    while any(groups) and len(result) < limit:
        for group in groups:
            if group and len(result) < limit:
                result.append(group.pop(0))
    return np.asarray(result, dtype=int)


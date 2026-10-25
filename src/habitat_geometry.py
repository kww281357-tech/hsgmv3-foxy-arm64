"""Bounded local Habitat geometry queries, without goals, images or agent motion.

Walkability alone is not a landing: Habitat can represent stairs as an inclined
NavMesh. This samples only the near-horizontal component reached from the
current foot position. It is an explicit simulator-geometry backend, not a
sensor-only claim or a semantic proof of the intended floor.
"""
from collections import deque
from numbers import Real

import numpy as np


_DEFAULTS = dict(radius_m=.6, grid_step_m=.1, micro_step_m=.05,
                 height_tolerance_m=.04, minimum_area_m2=.30,
                 minimum_short_span_m=.40, maximum_slope_degrees=5.,
                 xy_tolerance_m=.01, anchor_height_tolerance_m=.02,
                 max_queries=2048)


class _QueryLimit(Exception):
    pass


def measured_platform(routes, ground_mapper, observation_id, *, policy=None):
    """Return current foot-connected, wide and level NavMesh support.

    ``routes`` supplies ``pf`` and the existing ``to_habitat`` conversion.
    ``ground_mapper`` is the current mapper foot position, not camera height.
    No proposed waypoint, simulator goal, historical cloud or route is read.
    The caller must compare the returned observation ID/pose with its current
    controller frame before reuse; this function deliberately keeps no cache.

    Area is occupied local grid samples times grid cell area, never a convex
    hull across missing samples. It is a resolution-limited support estimate.
    Every accepted sample has an actual no-sliding micro-step chain to the
    anchor, with *cumulative* height deviation bounded by 4 cm. Failures and
    exhausted query budgets return invalid even if partial evidence looks good.
    """
    packet = dict(valid=False, current=False, backend='habitat_navmesh_local',
                  observation_id=None, checked_ground=None, reason='unmeasured',
                  supported_area_m2=0., short_span_m=0., slope_deg=None,
                  query_count=0, support_sample_count=0)

    def reject(reason):
        packet['reason'] = reason
        return packet

    cfg = dict(_DEFAULTS)
    if policy is not None:
        if not isinstance(policy, dict) or set(policy) - set(cfg):
            return reject('invalid_platform_policy')
        cfg.update(policy)
    if (type(cfg['max_queries']) is not int or not 1 <= cfg['max_queries'] <= 4096
            or any(not isinstance(v, Real) or isinstance(v, (bool, np.bool_))
                   or not np.isfinite(v) or v <= 0 for k, v in cfg.items() if k != 'max_queries')
            or not .05 <= cfg['grid_step_m'] <= .2
            or not cfg['grid_step_m'] <= cfg['radius_m'] <= 1.
            or not .01 <= cfg['micro_step_m'] <= .05
            or cfg['height_tolerance_m'] > .04
            or cfg['anchor_height_tolerance_m'] > .04
            or cfg['xy_tolerance_m'] >= min(cfg['micro_step_m'], cfg['grid_step_m']) / 2
            or cfg['maximum_slope_degrees'] > 5.):
        return reject('invalid_platform_policy')
    try:
        ground = np.asarray(ground_mapper)
        if ground.dtype.kind not in 'iuf' or ground.shape != (3,) or not np.isfinite(ground).all():
            return reject('invalid_ground_pose')
        ground = ground.astype(float)
    except (TypeError, ValueError, OverflowError):
        return reject('invalid_ground_pose')
    if type(observation_id) is not int or observation_id < 0:
        return reject('invalid_observation_id')
    packet.update(observation_id=observation_id, checked_ground=ground.tolist())
    pf = getattr(routes, 'pf', None)
    if pf is None or not getattr(pf, 'is_loaded', False):
        return reject('navmesh_unavailable')

    def query(name, *args):
        if packet['query_count'] >= cfg['max_queries']:
            raise _QueryLimit
        packet['query_count'] += 1
        return getattr(pf, name)(*args)

    def point(value):
        value = np.asarray(value, dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError('invalid NavMesh point')
        return value

    try:
        actual = point(routes.to_habitat(ground))
        anchor = point(query('snap_point', actual))
        error = anchor - actual
        packet['anchor_snap_error_m'] = float(np.linalg.norm(error))
        if (np.linalg.norm(error[[0, 2]]) > cfg['xy_tolerance_m']
                or abs(error[1]) > cfg['anchor_height_tolerance_m']):
            return reject('current_pose_not_on_navmesh')
        if not query('is_navigable', anchor):
            return reject('current_pose_not_navigable')
        island = query('get_island', anchor)
        if not isinstance(island, (int, np.integer)) or island < 0:
            return reject('current_island_unavailable')
        packet.update(current=True, island_id=int(island), anchor_habitat=anchor.tolist())
        spacing, radius = cfg['grid_step_m'], cfg['radius_m']
        found = {(0, 0): anchor}
        pending = deque([(0, 0)])
        tried_edges = set()
        height_rejections = 0

        while pending:
            key = pending.popleft()
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (key[0] + dx, key[1] + dz)
                if neighbor in found or np.hypot(*neighbor) * spacing > radius + 1e-8:
                    continue
                edge = tuple(sorted((key, neighbor)))
                if edge in tried_edges:
                    continue
                tried_edges.add(edge)
                target_xz = anchor[[0, 2]] + np.asarray(neighbor) * spacing
                pos = found[key].copy()
                count = max(1, int(np.ceil(np.linalg.norm(target_xz - pos[[0, 2]]) / cfg['micro_step_m'])))
                delta = (target_xz - pos[[0, 2]]) / count
                reached = True
                for _ in range(count):
                    request = pos.copy()
                    request[[0, 2]] += delta
                    nxt = point(query('try_step_no_sliding', pos, request))
                    if np.linalg.norm(nxt[[0, 2]] - request[[0, 2]]) > cfg['xy_tolerance_m']:
                        reached = False
                        break
                    # Refer every substep to the original foot plane. A ramp
                    # cannot accumulate many individually small vertical moves.
                    if abs(nxt[1] - anchor[1]) > cfg['height_tolerance_m']:
                        height_rejections += 1
                        reached = False
                        break
                    pos = nxt
                if not reached or query('get_island', pos) != island:
                    continue
                found[neighbor] = pos
                pending.append(neighbor)

        points = np.array(list(found.values()))
        xy = points[:, [0, 2]] - anchor[[0, 2]]
        area = len(points) * spacing ** 2
        packet.update(support_sample_count=len(points), supported_area_m2=round(area, 6),
                      height_rejection_count=height_rejections,
                      height_range_m=float(np.ptp(points[:, 1])), radius_m=radius,
                      grid_step_m=spacing)
        if len(points) < 3:
            return reject('insufficient_connected_area')
        design = np.column_stack((xy, np.ones(len(points))))
        plane, _, rank, _ = np.linalg.lstsq(design, points[:, 1] - anchor[1], rcond=None)
        _, axes = np.linalg.eigh(np.cov(xy.T))
        short_span = float(np.min(np.ptp(xy @ axes, axis=0)))
        slope = float(np.degrees(np.arctan(np.linalg.norm(plane[:2])))) if rank == 3 else None
        packet.update(short_span_m=round(short_span, 6), slope_deg=slope)
        if rank < 3 or short_span < cfg['minimum_short_span_m']:
            return reject('narrow_support')
        if area + 1e-8 < cfg['minimum_area_m2']:
            return reject('insufficient_connected_area')
        if slope > cfg['maximum_slope_degrees']:
            return reject('nonflat_surface')
        packet.update(valid=True, reason='supported_platform')
        return packet
    except _QueryLimit:
        return reject('query_budget_exhausted')
    except (AttributeError, TypeError, ValueError, RuntimeError, OverflowError, np.linalg.LinAlgError):
        return reject('navmesh_query_failed')


def stair_profile(routes, ground_mapper, observation_id, *, radius_m=1.2,
                  ray_count=16, max_queries=2048):
    """Measure local accessible rise/fall, not semantic top/bottom completion.

    Sixteen fixed-world rays use 5 cm no-sliding steps, with no individual
    height jump above 25 cm. A measured 20 cm excursion supplies local rise or
    fall evidence. A ray stopped by a real boundary is known blocked. Missing
    queries, NaNs, island changes or height jumps invalidate the whole packet.
    Absence on these finite rays never proves global absence of another flight.
    """
    packet = dict(valid=False, current=False, backend='habitat_navmesh_local',
                  observation_id=None, checked_ground=None, reason='unmeasured',
                  ascending=False, descending=False, ascending_rays=0,
                  descending_rays=0, blocked_rays=0, completed_rays=0,
                  query_count=0, radius_m=radius_m,
                  ascending_points=[], descending_points=[])

    def reject(reason):
        packet['reason'] = reason
        return packet

    if (not isinstance(radius_m, Real) or isinstance(radius_m, (bool, np.bool_))
            or not np.isfinite(radius_m) or not .2 <= radius_m <= 1.2
            or type(ray_count) is not int or not 8 <= ray_count <= 32
            or type(max_queries) is not int or not 1 <= max_queries <= 4096):
        return reject('invalid_profile_policy')
    try:
        ground = np.asarray(ground_mapper)
        if ground.dtype.kind not in 'iuf' or ground.shape != (3,) or not np.isfinite(ground).all():
            return reject('invalid_ground_pose')
        ground = ground.astype(float)
    except (TypeError, ValueError, OverflowError):
        return reject('invalid_ground_pose')
    if type(observation_id) is not int or observation_id < 0:
        return reject('invalid_observation_id')
    packet.update(observation_id=observation_id, checked_ground=ground.tolist())
    pf = getattr(routes, 'pf', None)
    if pf is None or not getattr(pf, 'is_loaded', False):
        return reject('navmesh_unavailable')

    def query(name, *args):
        if packet['query_count'] >= max_queries:
            raise _QueryLimit
        packet['query_count'] += 1
        return getattr(pf, name)(*args)

    def point(value):
        value = np.asarray(value, dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError('invalid NavMesh point')
        return value

    try:
        actual = point(routes.to_habitat(ground))
        anchor = point(query('snap_point', actual))
        error = anchor - actual
        if np.linalg.norm(error[[0, 2]]) > .01 or abs(error[1]) > .02:
            return reject('current_pose_not_on_navmesh')
        if not query('is_navigable', anchor):
            return reject('current_pose_not_navigable')
        island = query('get_island', anchor)
        if not isinstance(island, (int, np.integer)) or island < 0:
            return reject('current_island_unavailable')
        packet.update(current=True, island_id=int(island), anchor_habitat=anchor.tolist())
        rays = []
        count = int(np.ceil(radius_m / .05))
        stride = radius_m / count
        for index in range(ray_count):
            angle = 2 * np.pi * index / ray_count
            delta = np.array([np.cos(angle), 0., np.sin(angle)]) * stride
            pos = anchor.copy()
            low, high, traveled, blocked = 0., 0., 0., False
            rising_points, falling_points = [], []
            for _ in range(count):
                requested = pos + delta
                nxt = point(query('try_step_no_sliding', pos, requested))
                if np.linalg.norm(nxt[[0, 2]] - requested[[0, 2]]) > .01:
                    blocked = True
                    break
                if abs(nxt[1] - pos[1]) > .25:
                    return reject('unsafe_height_jump')
                if query('get_island', nxt) != island:
                    return reject('unexpected_island_change')
                pos = nxt
                traveled += stride
                height = float(pos[1] - anchor[1])
                low, high = min(low, height), max(high, height)
                # Keep small measured surface sets for independent current
                # RGB-D/ROI binding. These are virtual NavMesh queries, never
                # fabricated executed steps or semantic staircase labels.
                if height >= .20 and len(rising_points) < 3:
                    rising_points.append(point(routes.to_mapper(pos)).tolist())
                if height <= -.20 and len(falling_points) < 3:
                    falling_points.append(point(routes.to_mapper(pos)).tolist())
            ascending, descending = high >= .20, low <= -.20
            packet['ascending_rays'] += int(ascending)
            packet['descending_rays'] += int(descending)
            packet['blocked_rays'] += int(blocked)
            packet['completed_rays'] += 1
            packet['ascending_points'].extend(rising_points)
            packet['descending_points'].extend(falling_points)
            rays.append(dict(bearing_rad=float(angle), distance_m=float(traveled),
                             min_height_m=low, max_height_m=high, blocked=blocked))
        packet.update(valid=True, reason='local_profile_measured', rays=rays,
                      ascending=packet['ascending_rays'] > 0,
                      descending=packet['descending_rays'] > 0,
                      coverage='finite_radial_samples_only')
        return packet
    except _QueryLimit:
        return reject('query_budget_exhausted')
    except (AttributeError, TypeError, ValueError, RuntimeError, OverflowError):
        return reject('navmesh_query_failed')


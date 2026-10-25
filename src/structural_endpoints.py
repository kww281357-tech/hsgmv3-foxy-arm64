"""Fresh RGB-D support for a physical stair landing, without evaluator inputs.

The named stair is a visual identity cue. Its whole bounding rectangle is not
the distance to the floor on which the agent must stand. This module measures
the current foot-level floor independently; NavMesh supplies the separate
connected-platform and local-stair witnesses.
"""
import re

import numpy as np


def stair_endpoint_side(text):
    text = str(text).lower()
    if not re.search(r'\b(stairs?|staircase|steps?)\b', text):
        return None
    bottom = bool(re.search(r'\b(bottom|foot|base)\b', text))
    top = bool(re.search(r'\btop\b', text))
    if bottom and not top:
        return 'bottom'
    if top and not bottom:
        return 'top'
    if not bottom and not top:
        det = r'(?:(?:the|a|an|one|two|three|first|second|third|next|another|single|set|flight|of)\s+){0,6}'
        if re.search(r'\b(?:up|ascend(?:ing)?|climb(?:ing)?)\s+' + det + r'(?:stairs?|staircase|steps?)\s+(?:and|then)\s+(?:stop|wait|halt|stand)\b[.!?\s]*$', text):
            return 'top'
        if re.search(r'\b(?:down|descend(?:ing)?)\s+' + det + r'(?:stairs?|staircase|steps?)\s+(?:and|then)\s+(?:stop|wait|halt|stand)\b[.!?\s]*$', text):
            return 'bottom'
    return None


def current_floor_depth(depth_views, camera_views, intrinsic, ground, observation_id,
                        *, radius=.65, height_tolerance=.06, minimum_pixels=32,
                        minimum_area=.06):
    """Measure visible floor adjacent to the current feet in current RGB-D.

    Camera views contain mapper-frame position/rotation. No history cloud,
    detector mask, image-centre depth or bounding-box background is substituted.
    This is only a floor observation, never proof of a named task endpoint.
    """
    packet = {'valid': False, 'current': False, 'reason': 'current_floor_depth_unavailable',
              'observation_id': observation_id, 'source': 'current_rgbd_floor',
              'support_views': [], 'pixel_count': 0, 'observed_area_m2': 0.}
    try:
        k, ground = np.asarray(intrinsic, float), np.asarray(ground, float)
        if (type(observation_id) is not int or observation_id < 0
                or k.shape != (3, 3) or ground.shape != (3,)
                or not np.isfinite(k).all() or not np.isfinite(ground).all()
                or k[0, 0] <= 0 or k[1, 1] <= 0
                or not np.allclose(k[2], [0, 0, 1])
                or not 0 < radius <= 1. or not 0 < height_tolerance <= .10):
            return packet
        points = []
        for view in ('front', 'left', 'right', 'back'):
            if view not in depth_views or view not in camera_views:
                continue
            camera = camera_views[view]
            rotation, position = np.asarray(camera['rotation'], float), np.asarray(camera['position'], float)
            if (rotation.shape != (3, 3) or position.shape != (3,)
                    or not np.isfinite(rotation).all() or not np.isfinite(position).all()
                    or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3)
                    or np.linalg.norm(position[:2]-ground[:2]) > .05):
                continue
            depth = np.asarray(depth_views[view], float)
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[:, :, 0]
            if depth.ndim != 2:
                continue
            # Deterministic subsampling bounds work at high image resolution.
            stride = max(1, int(np.ceil(max(depth.shape)/320)))
            rows, cols = np.indices(depth.shape)
            selected = (rows % stride == 0) & (cols % stride == 0)
            selected &= np.isfinite(depth) & (depth > .15) & (depth < 50.)
            rows, cols = np.nonzero(selected)
            z = depth[rows, cols]
            camera_points = np.column_stack(((cols-k[0, 2])*z/k[0, 0],
                (depth.shape[0]-1-rows-k[1, 2])*z/k[1, 1], -z))
            world = camera_points @ rotation.T + position
            delta = world-ground
            floor = world[(np.abs(delta[:, 2]) <= height_tolerance)
                          & (np.linalg.norm(delta[:, :2], axis=1) <= radius)]
            if len(floor):
                points.append(floor)
                packet['support_views'].append(view)
        if not points:
            return packet
        points = np.concatenate(points)
        cells, inverse, counts = np.unique(
            np.floor((points[:, :2]-ground[:2])/.1+1e-6).astype(int),
            axis=0, return_inverse=True, return_counts=True)
        # Isolated depth pixels must not fill a whole 10cm cell with support.
        occupied = {tuple(c) for c, count in zip(cells, counts) if count >= 3}
        largest = set()
        while occupied:
            seed = occupied.pop()
            component, pending = {seed}, [seed]
            while pending:
                x, y = pending.pop()
                for cell in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                    if cell in occupied:
                        occupied.remove(cell)
                        component.add(cell)
                        pending.append(cell)
            if len(component) > len(largest):
                largest = component
        area = len(largest)*.01
        keep = np.array([tuple(c) in largest for c in cells], dtype=bool)
        points = points[keep[inverse]]
        packet.update(pixel_count=len(points), observed_area_m2=round(area, 4))
        if len(points) < minimum_pixels or area+1e-9 < minimum_area:
            packet['reason'] = 'insufficient_current_floor_depth'
            return packet
        design = np.column_stack((points[:, :2]-ground[:2], np.ones(len(points))))
        plane, _, rank, _ = np.linalg.lstsq(design, points[:, 2]-ground[2], rcond=None)
        slope = float(np.degrees(np.arctan(np.linalg.norm(plane[:2]))))
        packet['slope_deg'] = slope
        if rank != 3 or slope > 5.:
            packet['reason'] = 'current_floor_depth_not_flat'
            return packet
        packet.update(valid=True, current=True, reason='ok', world_center=ground.tolist(),
                      distance_m=0., checked_ground=ground.tolist())
        return packet
    except (KeyError, TypeError, ValueError, IndexError, OverflowError, np.linalg.LinAlgError):
        return packet


def bind_visible_stair_surface(depth, camera, intrinsic, bbox, stair_points):
    """Bind queried stair surfaces to a named current image region and depth.

    A nearby wall/floor rectangle cannot corroborate a NavMesh stair elsewhere.
    Query points are projected into the actual RGB-D camera and must agree
    with its measured depth; occluded or out-of-region stairs do not count.
    """
    result = {'valid': False, 'reason': 'stair_surface_not_in_current_region', 'matched_points': 0}
    try:
        depth = np.asarray(depth, float)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[:, :, 0]
        box = np.asarray(bbox)
        k = np.asarray(intrinsic, float)
        rotation, position = np.asarray(camera['rotation'], float), np.asarray(camera['position'], float)
        points = np.asarray(stair_points, float)
        if (depth.ndim != 2 or box.dtype.kind not in 'iuf' or box.shape != (4,)
                or not np.isfinite(box).all() or np.any(box < 0) or np.any(box > 1)
                or np.any(box[2:] <= box[:2]) or np.prod(box[2:]-box[:2]) > .75
                or k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
                or not np.allclose(k[2], [0., 0., 1.]) or abs(k[0, 1])+abs(k[1, 0]) > 1e-6
                or rotation.shape != (3, 3) or position.shape != (3,)
                or not np.isfinite(rotation).all() or not np.isfinite(position).all()
                or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-3)
                or points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()):
            return result
        h, w = depth.shape
        if (box[2]-box[0])*w < 3 or (box[3]-box[1])*h < 3:
            return result
        camera_points = (points-position) @ rotation
        matches = []
        pixels = set()
        for world, point in zip(points, camera_points):
            z = -point[2]
            if not .15 < z < 50.:
                continue
            col = point[0]*k[0, 0]/z+k[0, 2]
            row = h-1-(point[1]*k[1, 1]/z+k[1, 2])
            if not (box[0]*w <= col < box[2]*w and box[1]*h <= row < box[3]*h):
                continue
            x, y = int(round(col)), int(round(row))
            if not (0 <= x < w and 0 <= y < h):
                continue
            center = depth[y, x]
            tolerance = max(.18, .12*z)
            # Neighbourhood background must not erase a real foreground
            # occluder at the projected stair point, or fill a missing pixel.
            if not np.isfinite(center) or not .15 < center < 50. or abs(center-z) > tolerance:
                continue
            samples = depth[max(0, y-1):min(h, y+2), max(0, x-1):min(w, x+2)]
            samples = samples[np.isfinite(samples) & (samples > .15) & (samples < 50.)]
            if len(samples) >= 3 and abs(float(np.median(samples))-z) <= tolerance:
                matches.append(world)
                pixels.add((x, y))
        unique = np.unique(np.floor(np.asarray(matches).reshape(-1, 3)/.03), axis=0)
        result['matched_points'] = len(unique)
        result['matched_pixels'] = len(pixels)
        if len(unique) >= 2 and len(pixels) >= 2:
            result.update(valid=True, reason='current_stair_surface_bound')
        return result
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        return result


def structural_stair_verdict(text, platform, profile, floor, *, observation_id, ground):
    """Combine independent current geometry; no model Boolean can set it true.

    A local rising stair is consistent with a bottom landing and a falling
    stair with a top landing. Both directions at a landing remain ambiguous.
    A ray profile is a local consistency witness, not a global floor label.
    Visual identity, route execution and independent confirmation remain in
    the caller's contract.
    """
    side = stair_endpoint_side(text)
    result = {'valid': False, 'accepted': False, 'geometry_verified': False,
              'source': 'habitat_structural_landing', 'endpoint_kind': 'stair_landing',
              'observation_id': observation_id, 'side': side,
              'reason': 'stair_endpoint_side_unknown', 'platform': platform,
              'stair_profile': profile, 'floor_depth': floor}
    if side is None:
        return result
    try:
        ground = np.asarray(ground, float)
        if type(observation_id) is not int or observation_id < 0 or ground.shape != (3,) or not np.isfinite(ground).all():
            return result
        for name, packet in (('platform', platform), ('stair_profile', profile), ('floor_depth', floor)):
            point = np.asarray(packet.get('checked_ground'), float) if isinstance(packet, dict) else np.array([])
            if (not isinstance(packet, dict) or packet.get('valid') is not True
                    or packet.get('current') is not True
                    or type(packet.get('observation_id')) is not int
                    or packet['observation_id'] != observation_id
                    or point.shape != (3,) or not np.isfinite(point).all()
                    or np.linalg.norm(point-ground) > .01):
                result['reason'] = name+'_not_current_or_supported'
                return result
        ascending, descending = profile.get('ascending'), profile.get('descending')
        supported_side = (ascending is True and descending is False if side == 'bottom'
                          else descending is True and ascending is False)
        if not supported_side:
            result['reason'] = 'stair_side_not_observed_or_intermediate_landing'
            return result
        result.update(valid=True, geometry_verified=True, reason='ok',
                      world_center=ground.tolist(), candidate_point=ground.tolist(), distance_m=0.)
        return result
    except (TypeError, ValueError, OverflowError):
        return result


"""Offline debug-only overlays. Never call these before policy RGB is captured."""
from __future__ import annotations

import math
import numpy as np
from PIL import Image, ImageDraw

from .collision import vec3
from .contracts import DummySpec, Pose


def rotation_matrix_xyzw(rotation):
    q = np.asarray(rotation, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(float(q @ q) - 1) > 1e-4:
        raise ValueError("Expected unit XYZW camera quaternion")
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def draw_interaction_circle(rgb, *, robot_ground_position, camera_position,
                            camera_rotation_xyzw, hfov_degrees=90.0,
                            radius_m=1.0, samples=128):
    """Project the true horizontal interaction disk outline onto a COPY of RGB.

    World +Y is up; the camera looks along local -Z with local +X right. No
    geometry is added to Habitat, and input RGB/depth/semantic arrays are untouched.
    This disk is the interaction query, not the person's physical collision proxy.
    """
    raw = np.asarray(rgb)
    if raw.ndim != 3 or raw.shape[2] not in (3, 4) or raw.dtype != np.uint8:
        raise ValueError("Debug RGB must be uint8 HxWx3/4")
    if not 1 < hfov_degrees < 179 or not math.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("Invalid camera FOV/radius")
    if type(samples) is not int or not 16 <= samples <= 4096:
        raise ValueError("Invalid circle sample count")
    base, camera = np.asarray(vec3(robot_ground_position)), np.asarray(vec3(camera_position))
    world_to_camera = rotation_matrix_xyzw(camera_rotation_xyzw).T
    h, w = raw.shape[:2]
    focal = w / (2 * math.tan(math.radians(hfov_degrees) / 2))
    projected = []
    for theta in np.linspace(0, 2 * math.pi, samples + 1):
        world = base + np.array([radius_m * math.cos(theta), 0.015, radius_m * math.sin(theta)])
        local = world_to_camera @ (world-camera)
        if local[2] >= -0.05:
            projected.append(None)
            continue
        u, v = w/2 + focal*local[0]/-local[2], h/2 - focal*local[1]/-local[2]
        projected.append((float(u), float(v)) if abs(u) < 4*w and abs(v) < 4*h else None)
    picture = Image.fromarray(np.array(raw[..., :3], copy=True))
    draw = ImageDraw.Draw(picture)
    for a, b in zip(projected, projected[1:]):
        if a is not None and b is not None:
            draw.line([a, b], fill=(255, 215, 0), width=2)
    draw.rectangle((0, 0, min(w, 370), 17), fill=(0, 0, 0))
    draw.text((3, 2), f"DEBUG ONLY: {radius_m:g} m interaction disk; not collision", fill=(255, 215, 0))
    return np.array(picture, copy=True)


def draw_topdown_inset(rgb, *, robot_pose: Pose, dummies: list[DummySpec],
                      interaction_radius_m=1.0, robot_radius_m=0.18,
                      size_px=150, view_radius_m=1.6, floor_tolerance_m=0.25):
    """Debug-only egocentric range diagram on a separate RGB copy.

    Uses measured base pose and explicitly privileged actor foot locations. It is
    NOT a map: no floor shape, obstacles, free space, visibility or connectivity
    are invented. Up is robot-forward, right is robot-right. Person labels are
    anonymous D1/D2, never room metadata. Gray outlines denote another height.
    This diagram must never be sent to the navigation model as raw observation.
    """
    raw = np.asarray(rgb)
    if raw.ndim != 3 or raw.shape[2] not in (3,4) or raw.dtype != np.uint8:
        raise ValueError("Debug RGB must be uint8 HxWx3/4")
    if (type(size_px) is not int or size_px < 96
            or any(not math.isfinite(x) or x <= 0 for x in
                   (interaction_radius_m,robot_radius_m,view_radius_m,floor_tolerance_m))
            or view_radius_m <= interaction_radius_m or not math.isfinite(robot_pose.yaw_rad)):
        raise ValueError("Invalid inset configuration")
    h,w = raw.shape[:2]
    size = min(size_px,w-16,h-16)
    if size < 96:
        raise ValueError("Image is too small for a labelled topdown inset")
    base = np.asarray(vec3(robot_pose.position))
    yaw = robot_pose.yaw_rad
    right = np.array([math.cos(yaw),0,-math.sin(yaw)])
    forward = np.array([-math.sin(yaw),0,-math.cos(yaw)])
    tile = Image.new("RGB",(size,size),(16,22,29))
    draw = ImageDraw.Draw(tile)
    cx,cy = size/2,(size+25)/2
    scale = (size-38)/(2*view_radius_m)
    radius = interaction_radius_m*scale
    draw.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),outline=(255,215,0),width=2)
    draw.text((3,size-13),f"disk {interaction_radius_m:g}m; body != disk",fill=(255,215,0))
    for index,dummy in enumerate(dummies,1):
        if not math.isfinite(dummy.radius_m) or dummy.radius_m <= 0:
            raise ValueError("Invalid dummy body radius")
        delta = np.asarray(vec3(dummy.ground_position))-base
        u,v = cx+float(delta@right)*scale,cy-float(delta@forward)*scale
        body_r = dummy.radius_m*scale
        if u+body_r < 0 or u-body_r > size or v+body_r < 27 or v-body_r > size-15:
            continue
        same_height = abs(delta[1]) <= floor_tolerance_m
        color = (65,160,245) if same_height else (125,125,125)
        draw.ellipse((u-body_r,v-body_r,u+body_r,v+body_r),outline=color,
                     fill=color if same_height else None,width=2)
        draw.line((u-2,v,u+2,v),fill=(255,255,255),width=1)
        draw.line((u,v-2,u,v+2),fill=(255,255,255),width=1)
        label = f"D{index}" if same_height else f"D{index} dy{delta[1]:+.1f}"
        draw.text((u+body_r+1,v-5),label,fill=(230,235,240))
    rr = max(2,robot_radius_m*scale)
    draw.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),fill=(255,135,60))
    draw.line((cx,cy,cx,cy-rr-10),fill=(255,255,255),width=2)
    draw.polygon([(cx,cy-rr-13),(cx-3,cy-rr-7),(cx+3,cy-rr-7)],fill=(255,255,255))
    # Caption is drawn last so clipped markers can never hide the limitation.
    draw.rectangle((0,0,size,26),fill=(0,0,0))
    draw.text((3,1),"DEBUG: range only",fill=(255,255,255))
    draw.text((3,13),"NOT an occupancy map",fill=(255,215,0))
    picture = Image.fromarray(np.array(raw[...,:3],copy=True))
    picture.paste(tile,(8,h-size-8))
    return np.array(picture,copy=True)

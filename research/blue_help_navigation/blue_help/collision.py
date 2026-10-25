"""Goal-free, continuous upright-cylinder occupancy checks (Habitat Y-up).

These proxies constrain the ordinary scene-node Agent, which is not a Bullet
rigid body. The 1 m question disk is deliberately NOT a collision primitive.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


def vec3(value) -> tuple[float, float, float]:
    try:
        result = tuple(float(x) for x in value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Expected finite XYZ") from exc
    if len(result) != 3 or not all(math.isfinite(x) and abs(x) < 1e7 for x in result):
        raise ValueError("Expected finite XYZ")
    return result


@dataclass(frozen=True)
class Cylinder:
    identity: str
    foot: tuple[float, float, float]
    radius_m: float
    height_m: float

    def __post_init__(self):
        object.__setattr__(self, "foot", vec3(self.foot))
        if not isinstance(self.identity, str) or not self.identity:
            raise ValueError("Cylinder identity required")
        if any(not math.isfinite(x) or x <= 0 for x in (self.radius_m, self.height_m)):
            raise ValueError("Cylinder dimensions must be positive and finite")


def swept_cylinder_hits(start, end, obstacle: Cylinder, *, agent_radius_m: float,
                        agent_height_m: float, margin_m: float = 0.03) -> bool:
    """Analytic segment-vs-inflated-cylinder test, including vertical overlap.

    Both the robot and obstacle are upright. Linear interpolation is checked
    continuously, so a step with clear endpoints cannot tunnel through a person.
    Contact counts as collision. Vertical disjointness permits different floors.
    """
    p, q = vec3(start), vec3(end)
    if (any(not math.isfinite(x) or x <= 0 for x in (agent_radius_m, agent_height_m))
            or not math.isfinite(margin_m) or margin_m < 0):
        raise ValueError("Invalid robot dimensions/margin")
    lo, hi = 0.0, 1.0
    # y(t) must lie in [obstacle bottom - robot height, obstacle top].
    low_y = obstacle.foot[1] - agent_height_m
    high_y = obstacle.foot[1] + obstacle.height_m
    dy = q[1] - p[1]
    if abs(dy) < 1e-12:
        if p[1] < low_y or p[1] > high_y:
            return False
    else:
        a, b = (low_y - p[1]) / dy, (high_y - p[1]) / dy
        lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
        if lo > hi:
            return False
    x, z = p[0] - obstacle.foot[0], p[2] - obstacle.foot[2]
    dx, dz = q[0] - p[0], q[2] - p[2]
    length2 = dx * dx + dz * dz
    t = lo if length2 < 1e-24 else max(lo, min(hi, -(x * dx + z * dz) / length2))
    distance2 = (x + t * dx) ** 2 + (z + t * dz) ** 2
    radius = agent_radius_m + obstacle.radius_m + margin_m
    return distance2 <= radius * radius + 1e-12


def first_blocker(start, end, obstacles: Iterable[Cylinder], *, agent_radius_m: float,
                  agent_height_m: float, margin_m: float = 0.03) -> str | None:
    """Return a blocking ID (not a time-of-impact ordering), or None."""
    for obstacle in obstacles:
        if swept_cylinder_hits(start, end, obstacle, agent_radius_m=agent_radius_m,
                               agent_height_m=agent_height_m, margin_m=margin_m):
            return obstacle.identity
    return None

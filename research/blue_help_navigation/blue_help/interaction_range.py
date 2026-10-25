"""New standalone geometry/eligibility contract, NOT integrated into blocked v1.1.

The 1 m horizontal disk is an interaction query, never a physical collider.
Positions are internal simulator evidence, not fields exposed to the policy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


def _position(value):
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError("A three-component internal position is required")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in value):
        raise ValueError("Position must be finite")
    return tuple(float(x) for x in value)


@dataclass(frozen=True)
class RangeConfig:
    radius_m: float = 1.0
    max_floor_height_delta_m: float = 0.5

    def __post_init__(self):
        for value in (self.radius_m, self.max_floor_height_delta_m):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("Range settings must be positive finite numbers")


def evaluate_interaction(*, robot_ground_position, dummy_foot_position,
                         dummy_id: str, visible: Optional[bool],
                         line_of_sight_clear: Optional[bool], same_floor: Optional[bool],
                         navmesh_reachable: Optional[bool], remaining_questions: int,
                         cooldown_ready: bool, config: RangeConfig = RangeConfig()) -> dict:
    """Fail closed on missing evidence; recompute at ask execution, not just prompt time.

    Foot-center inclusion: dx^2 + dz^2 <= radius^2. A body touching the edge while
    its foot center is outside does not qualify. Floor/LOS checks remain mandatory.
    Eligibility is not an automatic question and never decrements the budget.
    """
    robot = _position(robot_ground_position)
    dummy = _position(dummy_foot_position)
    if not isinstance(dummy_id, str) or not dummy_id.strip():
        raise ValueError("dummy_id required")
    if type(remaining_questions) is not int or not 0 <= remaining_questions <= 3:
        raise ValueError("Remaining delivered-answer budget must be 0..3")
    flags = (visible, line_of_sight_clear, same_floor, navmesh_reachable)
    if any(value is not None and type(value) is not bool for value in flags):
        raise ValueError("Evidence flags must be bool or unknown (None)")
    if type(cooldown_ready) is not bool:
        raise ValueError("cooldown_ready must be bool")
    horizontal = math.hypot(dummy[0] - robot[0], dummy[2] - robot[2])
    reasons = []
    if horizontal > config.radius_m:
        reasons.append("outside_interaction_disk")
    if same_floor is not True or abs(dummy[1] - robot[1]) > config.max_floor_height_delta_m:
        reasons.append("different_or_unverified_floor")
    if visible is not True:
        reasons.append("not_currently_visible")
    if line_of_sight_clear is not True:
        reasons.append("occluded_or_unverified_line_of_sight")
    if navmesh_reachable is not True:
        reasons.append("unreachable_or_unverified_navmesh")
    spatial_eligible = not reasons
    if remaining_questions == 0:
        reasons.append("delivered_answer_budget_exhausted")
    if not cooldown_ready:
        reasons.append("cooldown")
    # Only local interaction facts, no world coordinates or goal/path information.
    return {"dummy_id": dummy_id, "eligible": not reasons, "spatial_eligible": spatial_eligible,
            "reason_codes": reasons,
            "horizontal_distance_m": horizontal, "interaction_radius_m": config.radius_m,
            "boundary_rule": "dummy_foot_center_inside_or_on_horizontal_disk",
            "remaining_delivered_questions": remaining_questions,
            "automatic_question": False, "circle_is_collision_body": False}


def required_passage_width(robot_radius_m: float, safety_margin_m: float) -> float:
    """Measured free corridor width must include clearance on both sides."""
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0
           for x in (robot_radius_m, safety_margin_m)) or robot_radius_m == 0:
        raise ValueError("Measured radius and nonnegative safety margin required")
    return 2 * (robot_radius_m + safety_margin_m)


def audit_placement_evidence(*, robot_radius_m: float, safety_margin_m: float,
                            measured_free_width_m: Optional[float],
                            foot_on_navmesh: Optional[bool],
                            body_clear_of_static_geometry: Optional[bool],
                            collision_or_occupancy_proxy_active: Optional[bool],
                            no_agent_human_overlap: Optional[bool],
                            start_goal_connectivity_preserved: Optional[bool],
                            swept_path_avoids_human: Optional[bool]) -> dict:
    """Audit supplied measurements; this is NOT a geometry engine or a real test.

    Unknown facts remain pending. A collision-disabled actor alone cannot pass.
    A navigation occupancy proxy must actually constrain motion as well as plans.
    """
    required = required_passage_width(robot_radius_m, safety_margin_m)
    facts = {"foot_on_navmesh": foot_on_navmesh,
             "body_clear_of_static_geometry": body_clear_of_static_geometry,
             "collision_or_occupancy_proxy_active": collision_or_occupancy_proxy_active,
             "no_agent_human_overlap": no_agent_human_overlap,
             "start_goal_connectivity_preserved": start_goal_connectivity_preserved,
             "swept_path_avoids_human": swept_path_avoids_human}
    if any(v is not None and type(v) is not bool for v in facts.values()):
        raise ValueError("Measurement flags must be bool or unknown")
    if measured_free_width_m is not None:
        if (type(measured_free_width_m) not in (int, float)
                or not math.isfinite(measured_free_width_m) or measured_free_width_m < 0):
            raise ValueError("Invalid width measurement")
    facts["passage_width_sufficient"] = (None if measured_free_width_m is None
                                          else measured_free_width_m >= required)
    failed = [key for key, value in facts.items() if value is False]
    pending = [key for key, value in facts.items() if value is None]
    return {"status": "fail" if failed else "pending" if pending else "pass",
            "failed_checks": failed, "pending_checks": pending,
            "required_free_width_m": required,
            "measurement_source": "caller_supplied_not_simulator_verified_by_this_module"}

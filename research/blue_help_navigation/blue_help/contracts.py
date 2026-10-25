"""Shared internal contracts for the independent rebuild; not policy payloads."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Protocol

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class Pose:
    position: Vec3
    yaw_rad: float


@dataclass(frozen=True)
class DummySpec:
    dummy_id: str
    ground_position: Vec3
    yaw_rad: float
    radius_m: float = 0.32
    height_m: float = 1.6
    floor_offset_m: float = 0.8


@dataclass(frozen=True)
class Episode:
    episode_id: str
    scene_id: str
    instruction: str
    start_position: Vec3
    start_rotation_xyzw: tuple[float, float, float, float]
    goal_position: Vec3
    trajectory_id: str = ""


@dataclass
class SensorFrame:
    """Internal sensor frame. build_policy_payload must remove internal world pose."""
    rgb: Any
    depth: Any
    pose: Pose
    step: int
    collision: bool = False
    dummy_candidates: list[dict] = field(default_factory=list)
    legal_actions: tuple[str, ...] = ("move_forward", "turn_left", "turn_right", "stop", "wait")
    action_specs: dict = field(default_factory=dict)


class NavigationEnvironment(Protocol):
    def reset(self, episode: Episode, dummies: list[DummySpec], seed: int = 42) -> SensorFrame: ...
    def observe(self) -> SensorFrame: ...
    def step(self, action: str) -> SensorFrame: ...
    def find_safe_path(self, start: Vec3, end: Vec3) -> list[Vec3] | None: ...
    def close(self) -> None: ...

"""Observed-evidence lost heuristic, independent of the oracle/evaluator.

This module accepts measured ego poses, collisions, explicit previous primitives
and a public heuristic uncertainty score. It has no environment, goal, reference
route, shortest-path metric, reward or oracle input. It reports uncertainty and
observable symptoms, not correctness of the travelled route.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from numbers import Real

from .contracts import Pose, SensorFrame


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f'{name} must be finite')
    return float(value)


def _pose(value):
    if not isinstance(value, Pose):
        raise ValueError('pose must be Pose')
    try:
        position = tuple(value.position)
    except TypeError as exc:
        raise ValueError('pose requires XYZ') from exc
    if len(position) != 3:
        raise ValueError('pose requires XYZ')
    return Pose(tuple(_finite(v, 'position') for v in position), _finite(value.yaw_rad, 'yaw'))


@dataclass(frozen=True)
class LostConfig:
    uncertainty_threshold: float = .65
    uncertainty_patience: int = 3
    collision_patience: int = 3
    stagnation_window: int = 6
    min_translation_m: float = .10
    revisit_window: int = 24
    revisit_radius_m: float = .35
    min_loop_distance_m: float = 1.5
    min_revisit_gap_steps: int = 6
    recovery_patience: int = 3

    def __post_init__(self):
        threshold = _finite(self.uncertainty_threshold, 'uncertainty_threshold')
        if not 0 <= threshold <= 1:
            raise ValueError('uncertainty_threshold must lie in [0, 1]')
        object.__setattr__(self, 'uncertainty_threshold', threshold)
        for name in ('uncertainty_patience', 'collision_patience', 'min_revisit_gap_steps', 'recovery_patience'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('stagnation_window', 'revisit_window'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 3:
                raise ValueError(f'{name} must be an integer >=3')
        if self.min_revisit_gap_steps >= self.revisit_window:
            raise ValueError('revisit gap must be smaller than its observation window')
        for name in ('min_translation_m', 'revisit_radius_m', 'min_loop_distance_m'):
            value = _finite(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f'{name} must be positive')
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class LostStatus:
    is_lost: bool
    reasons: tuple[str, ...]
    newly_lost: bool = False


class LostDetector:
    def __init__(self, config=None):
        self.config = LostConfig(**config) if isinstance(config, dict) else (config or LostConfig())
        if not isinstance(self.config, LostConfig):
            raise ValueError('config must be LostConfig or dictionary')
        self.reset()

    def reset(self):
        self._history = deque(maxlen=max(self.config.revisit_window, self.config.stagnation_window))
        self._uncertain = self._collisions = self._clean = 0
        self._lost = False
        self._last_reasons = ()
        self._last_step = None
        self._last_input = None
        self._last_status = LostStatus(False, ())

    def update_frame(self, frame: SensorFrame, *, uncertainty=0.0, previous_action=None):
        """Convenience wrapper; RGB, dummy registry and all other frame fields ignored."""
        if not isinstance(frame, SensorFrame):
            raise ValueError('frame must be SensorFrame')
        return self.update(frame.pose, step=frame.step, collision=frame.collision,
                           uncertainty=uncertainty, previous_action=previous_action)

    def update(self, pose: Pose, *, step: int, collision=False,
               uncertainty=0.0, previous_action=None) -> LostStatus:
        pose = _pose(pose)
        if type(step) is not int or step < 0 or type(collision) is not bool:
            raise ValueError('step/collision types are invalid')
        uncertainty = _finite(uncertainty, 'uncertainty')
        if not 0 <= uncertainty <= 1:
            raise ValueError('uncertainty must lie in [0, 1]')
        if previous_action not in (None, 'move_forward', 'turn_left', 'turn_right', 'stop', 'wait'):
            raise ValueError('unsupported previous primitive')
        current = (pose, step, collision, uncertainty, previous_action)
        if self._last_step is not None:
            if step < self._last_step:
                raise ValueError('stale observation step')
            if step == self._last_step:
                if current != self._last_input:
                    raise ValueError('conflicting repeated observation step')
                return self._last_status
            if step > self._last_step + 1:
                # Unknown intervening observations do not constitute consecutive
                # collisions/stagnation. Rebuild evidence after a sensor gap.
                self.reset()
        self._uncertain = self._uncertain + 1 if uncertainty >= self.config.uncertainty_threshold else 0
        self._collisions = self._collisions + 1 if (collision and previous_action in (None, 'move_forward')) else 0
        self._history.append((step, pose.position, previous_action))
        history = list(self._history)
        reasons = []
        if self._uncertain >= self.config.uncertainty_patience:
            reasons.append('persistent_public_uncertainty')
        if self._collisions >= self.config.collision_patience:
            reasons.append('repeated_observed_collisions')
        stationary = history[-self.config.stagnation_window:]
        if len(stationary) == self.config.stagnation_window:
            forward_attempts = sum(action == 'move_forward' for _, _, action in stationary[1:])
            extent = max(math.dist(stationary[0][1], position) for _, position, _ in stationary)
            if (forward_attempts >= max(2, self.config.stagnation_window // 2)
                    and extent < self.config.min_translation_m):
                reasons.append('stalled_forward_motion')
        revisit = history[-self.config.revisit_window:]
        for i, (old_step, old_position, _) in enumerate(revisit[:-1]):
            if step - old_step < self.config.min_revisit_gap_steps:
                continue
            if math.dist(old_position, pose.position) > self.config.revisit_radius_m:
                continue
            walked = sum(math.dist(a[1], b[1]) for a, b in zip(revisit[i:], revisit[i+1:]))
            if walked >= self.config.min_loop_distance_m:
                reasons.append('revisited_observed_location')
                break
        previous_lost = self._lost
        if reasons:
            self._lost, self._clean, self._last_reasons = True, 0, tuple(reasons)
        else:
            self._clean += 1
            if self._clean >= self.config.recovery_patience:
                self._lost, self._last_reasons = False, ()
        status = LostStatus(self._lost, self._last_reasons if self._lost else (),
                            self._lost and not previous_lost)
        self._last_step, self._last_input, self._last_status = step, current, status
        return status

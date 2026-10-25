"""Isolated simulator oracle that delivers *text*, never hidden paths to policy.

The only origin used for a route query is the vehicle's measured question pose.
A dummy's foot position is neither accepted nor used. This module alone owns the
private goal and path callable. DirectionAnswer.reference_pose is INTERNAL
freshness metadata: a policy payload must include the answer text, not serialize
that pose. No returned answer contains a goal, route, or waypoint list.

At most 8 metres and 3 non-collinear movement segments are described. Collinear
segments may merge, but corners are never shortcut. Text is approximate verbal
help, not a guarantee that an open-loop vehicle can execute a NavMesh polyline;
normal RGB-D navigation/collision handling must still control the vehicle.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
import math
from numbers import Real
from typing import Callable, Iterable, Mapping

from .contracts import Pose


# Canonical COCO 2017 80-class display names. No names are inferred from a path.
COCO80_NAMES = frozenset(('person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,'
    'traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,'
    'elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,'
    'sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,'
    'wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,'
    'pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,'
    'keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,'
    'teddy bear,hair drier,toothbrush').split(','))


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f'{name} must be finite')
    return float(value)


def _point(value):
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError('expected XYZ point') from exc
    if len(values) != 3:
        raise ValueError('expected XYZ point')
    return tuple(_finite(x, 'coordinate') for x in values)


def _pose(value):
    if not isinstance(value, Pose):
        raise ValueError('expected a vehicle Pose')
    return Pose(_point(value.position), _finite(value.yaw_rad, 'yaw_rad'))


def _step(value):
    if type(value) is not int or value < 0:
        raise ValueError('step must be a nonnegative integer')
    return value


def _wrap(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class OracleConfig:
    max_distance_m: float = 8.0
    max_segments: int = 3
    goal_radius_m: float = 0.20
    path_endpoint_tolerance_m: float = 0.10
    max_path_points: int = 4096
    landmark_radius_m: float = 0.75

    def __post_init__(self):
        for name in ('max_distance_m', 'goal_radius_m', 'path_endpoint_tolerance_m', 'landmark_radius_m'):
            value = _finite(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f'{name} must be positive')
            object.__setattr__(self, name, value)
        if self.max_distance_m > 8:
            raise ValueError('oracle answers may describe at most 8 metres')
        if type(self.max_segments) is not int or not 1 <= self.max_segments <= 3:
            raise ValueError('oracle answers may describe one to three segments')
        if type(self.max_path_points) is not int or not 2 <= self.max_path_points <= 100000:
            raise ValueError('max_path_points must be an integer in [2, 100000]')


@dataclass(frozen=True)
class VerifiedLandmark:
    """Caller-supplied scene evidence, not an automatically verified detection.

    Create only from an actual scene annotation/observation. The nonempty evidence
    string should identify that observation or semantic instance. This schema
    checks provenance presence and scene identity; it cannot establish truth of
    invented evidence. Wrong-scene or unregistered objects are never mentioned.
    Position is oracle-internal and is never included in DirectionAnswer.
    """
    landmark_id: str
    class_name: str
    position: tuple[float, float, float]
    scene_id: str
    verification_evidence: str

    def __post_init__(self):
        for name in ('landmark_id', 'scene_id', 'verification_evidence'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f'{name} requires actual nonempty provenance')
        if self.class_name not in COCO80_NAMES:
            raise ValueError('landmark class must be an exact canonical COCO80 name')
        object.__setattr__(self, 'position', _point(self.position))


@dataclass(frozen=True)
class DirectionAnswer:
    """Delivery object. reference_pose is internal, not part of model payloads."""
    text: str
    reference_pose: Pose
    question_step: int
    terminal: bool
    landmark_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError('direction text must be nonempty')
        object.__setattr__(self, 'reference_pose', _pose(self.reference_pose))
        _step(self.question_step)
        if type(self.terminal) is not bool:
            raise ValueError('terminal must be a boolean')
        if not isinstance(self.landmark_ids, (tuple, list)) or any(
                not isinstance(x, str) or not x for x in self.landmark_ids):
            raise ValueError('landmark_ids must contain nonempty IDs')
        object.__setattr__(self, 'landmark_ids', tuple(self.landmark_ids))


def answer_is_stale(answer: DirectionAnswer, current_pose: Pose, current_step: int,
                    *, max_age_steps: int = 8, max_position_change_m: float = 0.35,
                    max_yaw_change_rad: float = math.radians(20)) -> bool:
    """Check BEFORE delivery if the vehicle moved/turned since asking.

    Do not repeatedly apply this to invalidate already-delivered instructions
    merely because the vehicle is now following them. Direction text explicitly
    states its question-time frame; newly requested answers use the new frame.
    """
    if not isinstance(answer, DirectionAnswer):
        raise ValueError('expected DirectionAnswer')
    pose, step = _pose(current_pose), _step(current_step)
    if type(max_age_steps) is not int or max_age_steps < 0:
        raise ValueError('max_age_steps must be a nonnegative integer')
    distance_limit = _finite(max_position_change_m, 'max_position_change_m')
    yaw_limit = _finite(max_yaw_change_rad, 'max_yaw_change_rad')
    if distance_limit < 0 or not 0 <= yaw_limit <= math.pi:
        raise ValueError('invalid freshness thresholds')
    return (step < answer.question_step or step - answer.question_step > max_age_steps
            or math.dist(pose.position, answer.reference_pose.position) > distance_limit
            or abs(_wrap(pose.yaw_rad - answer.reference_pose.yaw_rad)) > yaw_limit)


def _cause_text(public_history, question_step):
    """Discuss observed symptoms only; never infer a wrong turn from oracle GT."""
    recent = []
    try:
        history = list(islice(iter(public_history), 1024))
    except TypeError:
        history = []
    for item in history:
        if not isinstance(item, Mapping) or type(item.get('step')) is not int:
            continue
        if max(0, question_step - 8) <= item['step'] <= question_step:
            recent.append(item)
    # Distinct observations only: duplicates do not establish repeated failures.
    by_step = {item['step']: item for item in recent}
    recent = list(by_step.values())
    collisions = sum(item.get('collision') is True and item.get('action') in (None, 'move_forward')
                     for item in recent)
    if collisions >= 2:
        return ('Recent observations show repeated collisions, which may have blocked progress. '
                'Whether you took a wrong turn is unknown.')
    stalled = 0
    for item in recent:
        moved = item.get('displacement_m')
        if (item.get('action') == 'move_forward' and isinstance(moved, Real)
                and not isinstance(moved, bool) and math.isfinite(float(moved)) and 0 <= moved < .02):
            stalled += 1
    if stalled >= 2:
        return ('Recent forward actions produced very little movement, which may explain the lack of progress. '
                'Whether you took a wrong turn is unknown.')
    return 'The cause of getting lost is unknown from the available public observations.'


def _collinear_merge(points):
    """Only remove straight-line interior vertices; never shortcut a bend."""
    out = []
    for p in points:
        if out and math.dist(out[-1], p) < 1e-8:
            continue
        if len(out) >= 2:
            a, b = out[-2], out[-1]
            ab = tuple(y-x for x, y in zip(a, b))
            bc = tuple(y-x for x, y in zip(b, p))
            lengths = math.dist(a, b) * math.dist(b, p)
            dot = sum(x*y for x, y in zip(ab, bc))
            cross = (ab[1]*bc[2]-ab[2]*bc[1], ab[2]*bc[0]-ab[0]*bc[2], ab[0]*bc[1]-ab[1]*bc[0])
            if lengths > 0 and dot > 0 and math.sqrt(sum(x*x for x in cross)) / lengths <= 1e-7:
                out[-1] = p
                continue
        out.append(p)
    return out


class RuleDirectionOracle:
    def __init__(self, goal_position, find_safe_path: Callable, config=None,
                 verified_landmarks: Iterable[VerifiedLandmark] = (), *, scene_id: str | None = None):
        self._goal_position = _point(goal_position)
        if not callable(find_safe_path):
            raise ValueError('find_safe_path must be callable')
        self._find_safe_path = find_safe_path
        self.config = OracleConfig(**config) if isinstance(config, dict) else (config or OracleConfig())
        if not isinstance(self.config, OracleConfig):
            raise ValueError('config must be OracleConfig or a dictionary')
        if scene_id is not None and (not isinstance(scene_id, str) or not scene_id):
            raise ValueError('scene_id must be a nonempty string when supplied')
        self._scene_id = scene_id
        self._landmarks = tuple(verified_landmarks)
        if any(not isinstance(record, VerifiedLandmark) for record in self._landmarks):
            raise ValueError('landmarks require explicit VerifiedLandmark provenance')

    def answer(self, vehicle_pose: Pose, question_step: int, public_history=()) -> DirectionAnswer | None:
        pose, step = _pose(vehicle_pose), _step(question_step)
        # Never query from a dummy. Never return a partial or unsafe path supplied
        # by the backend as if it reached the evaluator goal.
        try:
            raw = self._find_safe_path(pose.position, self._goal_position)
            if raw is None:
                return None
            points = list(islice(iter(raw), self.config.max_path_points + 1))
            if not points or len(points) > self.config.max_path_points:
                return None
            points = [_point(p) for p in points]
        except (ValueError, TypeError, RuntimeError, ArithmeticError, AttributeError):
            return None
        if (math.dist(points[0], pose.position) > self.config.path_endpoint_tolerance_m
                or math.dist(points[-1], self._goal_position) > self.config.path_endpoint_tolerance_m):
            return None
        # A slightly snapped first point is checked, not silently treated as an
        # executed vehicle teleport. The verbal route is anchored at measured pose.
        points[0] = pose.position
        points = _collinear_merge(points)
        total = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
        cause = _cause_text(public_history, step)
        if total <= self.config.goal_radius_m and math.dist(pose.position, self._goal_position) <= self.config.goal_radius_m:
            return DirectionAnswer('You are already at the destination. Stop here. ' + cause,
                                   pose, step, True)
        if len(points) < 2:
            return None
        route = [pose.position]
        remaining = self.config.max_distance_m
        for target in points[1:]:
            if len(route) - 1 >= self.config.max_segments or remaining <= 1e-8:
                break
            start = route[-1]
            segment = math.dist(start, target)
            if segment <= 1e-8:
                continue
            if segment > remaining:
                ratio = remaining / segment
                route.append(tuple(a + ratio*(b-a) for a, b in zip(start, target)))
                remaining = 0
                break
            route.append(target)
            remaining -= segment
        if len(route) < 2:
            return None
        terminal = math.dist(route[-1], points[-1]) <= 1e-7
        heading = pose.yaw_rad
        instructions = []
        for start, target in zip(route, route[1:]):
            dx, dy, dz = (b-a for a, b in zip(start, target))
            horizontal = math.hypot(dx, dz)
            # An isolated vertical segment has no forward route to describe;
            # never invent stairs/elevator access from two coordinates.
            if horizontal < 1e-6:
                return None
            bearing = math.atan2(-dx, -dz)
            delta = _wrap(bearing - heading)
            degrees = abs(math.degrees(delta))
            if degrees <= 1:
                turn = ''
            elif degrees >= 175:
                turn = 'turn around, then '
            else:
                turn = f'turn {"left" if delta > 0 else "right"} about {degrees:.0f} degrees, then '
            distance = math.dist(start, target)
            grade = ' along the rising route' if dy > .15 else ' along the descending route' if dy < -.15 else ''
            instructions.append(f'{turn}go forward about {distance:.1f} meters{grade}')
            heading = bearing
        text = ('From your vehicle heading when you asked: ' + '; then '.join(instructions) + '. '
                + ('After completing these steps, stop at the destination. ' if terminal else
                   'These are only the next local directions, not confirmation of arrival. Continue normal navigation afterward. '))
        landmark_ids = ()
        eligible = [record for record in self._landmarks if self._scene_id is not None
                    and record.scene_id == self._scene_id
                    and math.dist(record.position, route[-1]) <= self.config.landmark_radius_m]
        if eligible:
            landmark = min(eligible, key=lambda record: (math.dist(record.position, route[-1]), record.landmark_id))
            text += (f'The last local leg ends near the scene-verified {landmark.class_name} '
                     f'(landmark ID: {landmark.landmark_id}). ')
            landmark_ids = (landmark.landmark_id,)
        return DirectionAnswer(text + cause, pose, step, terminal, landmark_ids)

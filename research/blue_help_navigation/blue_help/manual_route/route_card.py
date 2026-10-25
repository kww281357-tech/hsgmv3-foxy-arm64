"""Private offline manual-route records and a goal-free runtime local connector.

This schema validates provenance/geometry consistency, NOT the truth of image
annotations or correspondence of prose to a route. A human must inspect cited
frames and author/review the full route text. No full-route prose is generated.
Frame/evidence/card IDs are content-derived and objects are recursively frozen.
Never serialize these private records into a navigation-policy request. Only
LocalConnector.answer_text()'s explicitly delivered string is public.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from numbers import Real
from typing import Callable

COCO80_NAMES = frozenset(('person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,'
    'traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,'
    'elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,'
    'sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,'
    'wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,'
    'pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,'
    'keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,'
    'teddy bear,hair drier,toothbrush').split(','))


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(name + ' must be nonempty text')
    return value


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError('Finite numeric value required')
    return float(value)


def _point(value):
    result = tuple(_finite(v) for v in value)
    if len(result) != 3:
        raise ValueError('XYZ point required')
    return result


def _identity(prefix, value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return prefix + ':' + hashlib.sha256(encoded).hexdigest()


def _seal(obj, field_name, prefix):
    values = {name: value for name, value in obj.__dict__.items() if name != field_name}
    # dataclasses and normalized tuples become canonical JSON content.
    values = json.loads(json.dumps(values, default=lambda value: asdict(value), allow_nan=False))
    object.__setattr__(obj, field_name, _identity(prefix, values))


@dataclass(frozen=True)
class Pose:
    position: tuple[float, float, float]
    yaw_rad: float

    def __post_init__(self):
        object.__setattr__(self, 'position', _point(self.position))
        object.__setattr__(self, 'yaw_rad', _finite(self.yaw_rad))


@dataclass(frozen=True)
class DummyGeometry:
    dummy_id: str
    foot: tuple[float, float, float]
    radius_m: float = .32
    height_m: float = 1.6

    def __post_init__(self):
        _text(self.dummy_id, 'dummy_id')
        object.__setattr__(self, 'foot', _point(self.foot))
        for name in ('radius_m', 'height_m'):
            value = _finite(getattr(self, name))
            if value <= 0:
                raise ValueError('Positive body dimensions required')
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class FrameEvidence:
    scene_id: str
    image_sha256: str
    width: int
    height: int
    source_frame_label: str
    frame_id: str = field(init=False)

    def __post_init__(self):
        _text(self.scene_id, 'scene_id')
        _text(self.source_frame_label, 'source_frame_label')
        if (not isinstance(self.image_sha256, str) or len(self.image_sha256) != 64
                or any(c not in '0123456789abcdef' for c in self.image_sha256)):
            raise ValueError('Lowercase SHA-256 of the original evidence image bytes required')
        if any(type(v) is not int or v <= 0 for v in (self.width, self.height)):
            raise ValueError('Positive integer image dimensions required')
        _seal(self, 'frame_id', 'frame')

    def matches_image_bytes(self, data: bytes) -> bool:
        """Checks byte identity only; does NOT classify objects or verify scene facts."""
        if not isinstance(data, bytes):
            raise ValueError('Original evidence image bytes required')
        return hashlib.sha256(data).hexdigest() == self.image_sha256


@dataclass(frozen=True)
class LandmarkEvidence:
    frame_id: str
    coco_label: str
    bbox_xyxy: tuple[int, int, int, int]
    label_source: str  # manual_annotation OR detector_output, always manually reviewed
    review_evidence: str
    detector_id: str | None = None
    evidence_id: str = field(init=False)

    def __post_init__(self):
        _text(self.frame_id, 'frame_id')
        if self.coco_label not in COCO80_NAMES:
            raise ValueError('Exact canonical COCO80 label required; aliases/architecture are not labels')
        box = tuple(self.bbox_xyxy)
        if (len(box) != 4 or any(type(v) is not int or v < 0 for v in box)
                or box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError('Integer xyxy evidence bounding box required')
        object.__setattr__(self, 'bbox_xyxy', box)
        _text(self.review_evidence, 'manual scene-review evidence')
        if self.label_source not in ('manual_annotation', 'detector_output'):
            raise ValueError('Explicit manual_annotation or detector_output provenance required')
        if self.label_source == 'detector_output':
            _text(self.detector_id, 'detector identity/version')
        elif self.detector_id is not None:
            raise ValueError('Manual labels must not claim a detector produced them')
        _seal(self, 'evidence_id', 'evidence')


@dataclass(frozen=True)
class GoalAnnotation:
    goal_id: str
    position: tuple[float, float, float]
    frame_id: str
    review_evidence: str

    def __post_init__(self):
        for name in ('goal_id', 'frame_id', 'review_evidence'):
            _text(getattr(self, name), name)
        object.__setattr__(self, 'position', _point(self.position))


@dataclass(frozen=True)
class AnchorSpec:
    anchor_id: str
    pose: Pose
    review_evidence: str

    def __post_init__(self):
        _text(self.anchor_id, 'anchor_id')
        _text(self.review_evidence, 'exterior anchor review evidence')
        if not isinstance(self.pose, Pose):
            raise ValueError('Anchor requires a private position and explicit route-reference yaw')


@dataclass(frozen=True)
class GeometryConfig:
    robot_radius_m: float = .10
    robot_height_m: float = 1.5
    body_margin_m: float = .03
    interaction_radius_m: float = 1.0
    max_connector_length_m: float = 1.5
    floor_tolerance_m: float = .25
    endpoint_tolerance_m: float = .01

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = _finite(getattr(self, name))
            invalid = value < 0 if name == 'body_margin_m' else value <= 0
            if invalid:
                raise ValueError('Invalid geometry threshold')
            object.__setattr__(self, name, value)
        if (self.interaction_radius_m > 1.0 or self.max_connector_length_m > 1.5
                or self.floor_tolerance_m > .25 or self.endpoint_tolerance_m > .01):
            raise ValueError('Local interaction/connector bounds may only be tightened')


def _body_hit(a, b, dummy, config):
    """Continuous inflated upright-cylinder sweep; touching is a collision."""
    low, high = 0., 1.
    dy = b[1]-a[1]
    lower, upper = dummy.foot[1]-config.robot_height_m, dummy.foot[1]+dummy.height_m
    if abs(dy) < 1e-12:
        if a[1] < lower or a[1] > upper:
            return False
    else:
        p, q = (lower-a[1])/dy, (upper-a[1])/dy
        low, high = max(low, min(p, q)), min(high, max(p, q))
        if low > high:
            return False
    x, z = a[0]-dummy.foot[0], a[2]-dummy.foot[2]
    dx, dz = b[0]-a[0], b[2]-a[2]
    length2 = dx*dx+dz*dz
    t = low if length2 < 1e-24 else max(low, min(high, -(x*dx+z*dz)/length2))
    radius = dummy.radius_m+config.robot_radius_m+config.body_margin_m
    return (x+t*dx)**2+(z+t*dz)**2 <= radius*radius+1e-12


def _near_exterior(point, dummy, config):
    return (math.hypot(point[0]-dummy.foot[0], point[2]-dummy.foot[2])
            <= config.interaction_radius_m+1e-12
            and abs(point[1]-dummy.foot[1]) <= config.floor_tolerance_m
            and not _body_hit(point, point, dummy, config))


def _validated_path(raw, start, end, dummy, config):
    if not isinstance(raw, (list, tuple)) or not raw or len(raw) > 100000:
        raise ValueError('Nonempty collision-safe path required (no automatic fallback)')
    path = tuple(_point(p) for p in raw)
    if (math.dist(path[0], start) > 1e-7
            or math.dist(path[-1], end) > config.endpoint_tolerance_m):
        raise ValueError('Safe path has wrong endpoints')
    # Keep the ACTUAL safe-service polyline. A small terminal tolerance is
    # allowed (<=1 cm); never append an unqueried wall-crossing/teleport gap.
    # Origin must match to numerical precision: no shifted start is invented.
    if _body_hit(path[0], path[0], dummy, config) or any(
            _body_hit(a, b, dummy, config) for a, b in zip(path, path[1:])):
        raise ValueError('Path intersects the dummy body occupancy')
    return tuple(p for i, p in enumerate(path) if i == 0 or math.dist(path[i-1], p) > 1e-9)


@dataclass(frozen=True)
class RouteCard:
    """OFFLINE PRIVATE artifact. Never pass it, its goal, or path to the policy."""
    scene_id: str
    geometry_revision: str
    dummy: DummyGeometry
    anchor: AnchorSpec
    goal: GoalAnnotation
    route_text: str
    route_review_evidence: str
    frames: tuple[FrameEvidence, ...]
    landmarks: tuple[LandmarkEvidence, ...]
    mentioned_evidence_ids: tuple[str, ...]
    private_full_path: tuple[tuple[float, float, float], ...]
    config: GeometryConfig = field(default_factory=GeometryConfig)
    route_card_id: str = field(init=False)

    def __post_init__(self):
        for name in ('scene_id', 'geometry_revision', 'route_text', 'route_review_evidence'):
            _text(getattr(self, name), name)
        for name, kind in (('dummy', DummyGeometry), ('anchor', AnchorSpec),
                           ('goal', GoalAnnotation), ('config', GeometryConfig)):
            if not isinstance(getattr(self, name), kind):
                raise ValueError(name + ' has the wrong contract type')
        for name in ('frames', 'landmarks', 'mentioned_evidence_ids'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not self.frames or any(not isinstance(f, FrameEvidence) for f in self.frames):
            raise ValueError('Original scene evidence frames required')
        frame_map = {f.frame_id: f for f in self.frames}
        if len(frame_map) != len(self.frames) or any(f.scene_id != self.scene_id for f in self.frames):
            raise ValueError('Frames must have unique immutable IDs in the same scene')
        if self.goal.frame_id not in frame_map:
            raise ValueError('Goal annotation must reference a provided original frame')
        if any(not isinstance(e, LandmarkEvidence) for e in self.landmarks):
            raise ValueError('Typed, reviewed landmark evidence required')
        evidence_map = {e.evidence_id: e for e in self.landmarks}
        if len(evidence_map) != len(self.landmarks):
            raise ValueError('Duplicate evidence IDs')
        for evidence in self.landmarks:
            frame = frame_map.get(evidence.frame_id)
            if frame is None or evidence.bbox_xyxy[2] > frame.width or evidence.bbox_xyxy[3] > frame.height:
                raise ValueError('Landmark evidence must lie inside its referenced frame')
        if (len(set(self.mentioned_evidence_ids)) != len(self.mentioned_evidence_ids)
                or any(identity not in evidence_map for identity in self.mentioned_evidence_ids)):
            raise ValueError('Route landmark references must resolve to unique provided evidence')
        if not _near_exterior(self.anchor.pose.position, self.dummy, self.config):
            raise ValueError('Route anchor must be outside body clearance and within the 1 m interaction disk')
        path = _validated_path(self.private_full_path, self.anchor.pose.position,
                               self.goal.position, self.dummy, self.config)
        object.__setattr__(self, 'private_full_path', path)
        _seal(self, 'route_card_id', 'route-card')


def build_route_card(*, find_safe_path: Callable, **manual_fields) -> RouteCard:
    """OFFLINE ONLY: query anchor→annotated goal once; preserve authored text.

    ``manual_fields`` are RouteCard fields except private_full_path/route_card_id.
    No current vehicle pose is accepted. No route-length or turn-count cap exists.
    The safe-path backend must be the same actor-aware service as vehicle motion.
    This function never invents labels, fills scene prose, or chooses a new goal.
    """
    if 'private_full_path' in manual_fields or 'route_card_id' in manual_fields:
        raise ValueError('Offline builder obtains the full route from the safe path service')
    if not callable(find_safe_path):
        raise ValueError('Actor-aware find_safe_path service required')
    anchor, goal = manual_fields.get('anchor'), manual_fields.get('goal')
    if not isinstance(anchor, AnchorSpec) or not isinstance(goal, GoalAnnotation):
        raise ValueError('An explicit exterior anchor and manually annotated goal are required')
    dummy = manual_fields.get('dummy')
    config = manual_fields.get('config', GeometryConfig())
    if (not isinstance(dummy, DummyGeometry) or not isinstance(config, GeometryConfig)
            or not _near_exterior(anchor.pose.position, dummy, config)):
        raise ValueError('Validate a nearby exterior anchor before querying the full offline route')
    raw = find_safe_path(anchor.pose.position, goal.position)
    if raw is None:
        raise ValueError('Offline anchor-to-goal route unavailable; card not created')
    return RouteCard(private_full_path=raw, **manual_fields)


def _wrap(angle):
    return (angle+math.pi) % (2*math.pi)-math.pi


def _turn_text(delta):
    degrees = abs(math.degrees(_wrap(delta)))
    if degrees < .05:
        return ''
    return 'turn ' + ('left' if _wrap(delta) > 0 else 'right') + f' about {degrees:.1f} degrees'


def _connector_text(path, yaw, anchor_yaw):
    parts = []
    for a, b in zip(path, path[1:]):
        horizontal = math.hypot(b[0]-a[0], b[2]-a[2])
        if horizontal < 1e-9:
            if math.dist(a, b) > 1e-9:
                raise ValueError('Pure vertical connector cannot be expressed as a ground move')
            continue
        heading = math.atan2(-(b[0]-a[0]), -(b[2]-a[2]))
        turn = _turn_text(heading-yaw)
        parts.append((turn+' and ' if turn else '')+f'go forward about {horizontal:.3f} meters')
        yaw = heading
    turn = _turn_text(anchor_yaw-yaw)
    if turn:
        parts.append(turn+' to face the manually reviewed route-reference heading')
    if not parts:
        return 'You are at the exterior anchor and aligned with the route-reference heading.'
    return ('From your current heading, '+ '; then '.join(parts)+
            '. This brings you to the exterior anchor in its route-reference heading.')


class LocalConnector:
    """Runtime retains anchor/body/text ONLY—not the goal or full offline path.

    All geometry and IDs are internal. answer_text returns a string or None, never
    coordinates/waypoints, card objects, goal distance, or execution commands.
    The caller must separately enforce explicit consent, cooldown and delivery
    budget. A generated connector is guidance, not evidence of executed motion.
    """
    def __init__(self, card: RouteCard, find_safe_path: Callable):
        if not isinstance(card, RouteCard) or not callable(find_safe_path):
            raise ValueError('Validated private RouteCard and safe local path service required')
        self._anchor, self._dummy, self._config = card.anchor, card.dummy, card.config
        self._scene_id, self._revision = card.scene_id, card.geometry_revision
        self._find_safe_path, self._route_text = find_safe_path, card.route_text
        mentioned = {e.evidence_id: e for e in card.landmarks}
        sources = {mentioned[key].label_source for key in card.mentioned_evidence_ids}
        self._evidence_disclosure = (
            'Landmark labels are manually annotated and scene-reviewed; they are not detector outputs.'
            if sources == {'manual_annotation'} else
            'Landmark labels include detector outputs that were manually scene-reviewed; no detector is run now.'
            if sources == {'detector_output'} else
            'Landmark labels mix manual annotations and manually reviewed detector outputs; no detector is run now.'
            if sources else 'No COCO landmark labels are asserted by this card.')

    def answer_text(self, vehicle_pose, *, scene_id: str, geometry_revision: str,
                    dummy_id: str, observation_step: int, request_step: int,
                    visible: bool, line_of_sight_clear: bool, same_floor: bool) -> str | None:
        """Current ≤1 m request → fixed anchor via ≤1.5 m safe local connector.

        Steps must match exactly; stale requests, unverified visibility, changed
        geometry, blocked/malformed paths and body crossings fail closed. Only
        the local connector is recomputed. The complete authored route is intact.
        """
        if (scene_id != self._scene_id or geometry_revision != self._revision
                or dummy_id != self._dummy.dummy_id
                or type(observation_step) is not int or observation_step < 0
                or type(request_step) is not int or request_step != observation_step
                or visible is not True or line_of_sight_clear is not True or same_floor is not True):
            return None
        try:
            pose = Pose(vehicle_pose.position, vehicle_pose.yaw_rad)
            if not _near_exterior(pose.position, self._dummy, self._config):
                return None
            raw = self._find_safe_path(pose.position, self._anchor.pose.position)
            path = _validated_path(raw, pose.position, self._anchor.pose.position,
                                   self._dummy, self._config)
            if (sum(math.dist(a, b) for a, b in zip(path, path[1:]))
                    > self._config.max_connector_length_m+1e-12
                    or any(abs(p[1]-self._dummy.foot[1]) > self._config.floor_tolerance_m for p in path)):
                return None
            connector = _connector_text(path, pose.yaw_rad, self._anchor.pose.yaw_rad)
            if math.dist(path[-1], self._anchor.pose.position) > 1e-7:
                connector += (' The checked endpoint is within '
                    + f'{self._config.endpoint_tolerance_m:.3f} meters of the stored anchor; '
                    + 'no extra unverified motion to an exact coordinate is prescribed.')
        except (ValueError, TypeError, AttributeError, RuntimeError, ArithmeticError):
            return None
        return ('This is a manually authored, scene-reviewed route card, not a detector-generated route. '
                + self._evidence_disclosure + '\nLOCAL CONNECTOR: '+connector
                + '\nFollow the connector using current RGB-D and collision checks; do not teleport '
                'or treat guidance as executed motion. If blocked, do not force the move. '
                'Only after reaching the anchor and aligning, follow the FULL MANUAL ROUTE below. '
                'Each subsequent left/right instruction is relative to your heading at that point.\n'
                'FULL MANUAL ROUTE:\n'+self._route_text)

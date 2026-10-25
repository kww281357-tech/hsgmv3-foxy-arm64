"""Bounded endpoint control, independent of models, maps, and the simulator.

The caller supplies measured poses, guard verdicts and already certified local
actions. This module never executes an action or authorizes STOP by itself.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


def _point(value):
    try:
        result = tuple(float(x) for x in value)
    except (TypeError, ValueError):
        return None
    return result if len(result) == 3 and all(math.isfinite(x) for x in result) else None


def _distance(a, b, dimensions=3):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(dimensions)))


def _number(value, default=math.inf):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


@dataclass(frozen=True)
class EndpointConfig:
    max_calls: int = 6
    max_looks: int = 2
    max_approaches: int = 3
    max_protocol_repairs: int = 1
    arm_distance_m: float = 2.0
    max_target_shift_m: float = .65
    max_height_change_m: float = .35
    max_approach_path_m: float = .75
    max_geometry_approach_path_m: float = .90
    max_geometry_height_change_m: float = .50
    min_approach_gain_m: float = .10
    min_standoff_m: float = .35
    min_view_translation_m: float = .10
    min_view_rotation_rad: float = math.radians(8)
    max_observation_gap: int = 12


@dataclass(frozen=True)
class EndpointDecision:
    kind: str
    reason: str
    candidate_id: int | None = None
    action_id: str | None = None
    target: tuple | None = None

    def as_dict(self):
        return asdict(self)


class EndpointController:
    """Hold a local hypothesis until it is measured, abandoned, or recovered.

    ``pose`` and candidate/action points share one physical xyz coordinate
    system, with z vertical. ``scope_key`` identifies the task/condition, not a
    mapper floor label. The caller must reset after a completed physical escape
    or an explicitly abandoned hypothesis, never merely on a rejected verdict.

    ``verification_id`` identifies an actual model request. Reusing it is
    idempotent for the call budget; without it, observation_id is the request
    key. New requests on the same observation still cannot corroborate a view.
    """

    def __init__(self, config=None):
        self.config = config or EndpointConfig()
        self._serial = 0
        self.reset()

    def reset(self):
        self.active = False
        self.scope_key = None
        self.candidate_id = None
        self.origin_pose = None
        self.candidate_point = None
        self.target_id = None
        self._geometry_approach_active = False
        self.calls = self.looks = self.approaches = self.protocol_repairs = 0
        self._request_ids = set()
        self._action_ids = set()
        self._yaw = 0.0
        self._witness = None
        self._last_observation_id = None
        self._last_decision = EndpointDecision('continue', 'no_endpoint_candidate')
        self._recovery_reason = None
        self._protocol_failure_reason = None

    def _decide(self, kind, reason, action_id=None, target=None):
        self._last_decision = EndpointDecision(
            kind, reason, self.candidate_id, action_id, target)
        return self._last_decision

    def _recover(self, reason):
        self._recovery_reason = reason
        return self._decide('recover', reason)

    def _inspect(self, reason):
        if self.looks >= self.config.max_looks or self.calls >= self.config.max_calls:
            return self._recover('physical_escape_required:' + reason)
        return self._decide('inspect', reason)

    def _observe_action(self, row):
        if not isinstance(row, dict):
            return
        # A real ledger step is required; repeated API consumption of a row
        # must not invent motion, observations, or inspection attempts.
        step = row.get('controller_step', row.get('step'))
        if type(step) is not int or step in self._action_ids:
            return
        self._action_ids.add(step)
        yaw = _number(row.get('yaw_delta'), 0.)
        distance = _number(row.get('motion_distance'), 0.)
        self._yaw += yaw
        if (self._last_decision.kind == 'inspect' and row.get('action_type') == 'turn'
                and abs(yaw) >= .01 and distance < .02):
            self.looks += 1
        if (self._last_decision.kind == 'approach' and row.get('action_type') == 'move_forward'
                and distance >= .02):
            self.approaches += 1

    def consume(self, *, verification, observation_id, pose, scope_key,
                executed_action=None, certified_actions=(), verification_id=None):
        """Return a bounded observation/motion/repair decision, never STOP.

        A certified action has action_id, certified=True, target_point_world,
        and measured/planned path_length_m. The caller revalidates it immediately
        before execution. An invalid ROI never supplies an approach target.
        """
        verification = verification if isinstance(verification, dict) else {}
        packet = verification.get('endpoint_grounding') or {}
        packet = packet if isinstance(packet, dict) else {}
        position = _point(pose)
        if self.active and scope_key != self.scope_key:
            self.reset()
        arm = (verification.get('candidate_claimed') is True
               or verification.get('physical_arrival_candidate') is True
               or any(verification.get(k) is True for k in (
                   'endpoint_relocalization_required', 'endpoint_confirmation_required',
                   'landing_observation_pending'))
               or (packet.get('valid') is True
                   and _number(packet.get('distance_m')) <= self.config.arm_distance_m))
        if not self.active:
            if not arm:
                return self._decide('continue', 'no_endpoint_candidate')
            if position is None or type(observation_id) is not int or observation_id < 0:
                return self._decide('recover', 'invalid_candidate_observation_or_pose')
            self.active = True
            self.scope_key = scope_key
            self._serial += 1
            self.candidate_id = self._serial
            self.origin_pose = position
        self._observe_action(executed_action)
        if self._recovery_reason:
            return self._decide('recover', self._recovery_reason)
        if self._protocol_failure_reason:
            return self._decide('protocol_failure', self._protocol_failure_reason)
        if position is None or type(observation_id) is not int or observation_id < 0:
            return self._recover('invalid_current_observation_or_pose')
        if (self._last_observation_id is not None
                and observation_id < self._last_observation_id):
            return self._recover('observation_epoch_changed')
        self._last_observation_id = observation_id
        request = str(verification_id) if verification_id is not None else ('obs', observation_id)
        if request not in self._request_ids:
            self._request_ids.add(request)
            self.calls += 1
        if self.calls > self.config.max_calls:
            return self._recover('verification_budget_exhausted')
        # A rejected citation is not evidence that the physical target became
        # unobservable. Repair it against this exact frame/ledger, without
        # spending another physical look or manufacturing a fresh witness.
        if verification.get('evidence_repair_required') is True:
            if self.protocol_repairs >= self.config.max_protocol_repairs:
                self._protocol_failure_reason = 'endpoint_protocol_repair_exhausted'
            elif self.calls >= self.config.max_calls:
                self._protocol_failure_reason = 'endpoint_protocol_repair_budget_exhausted'
            if self._protocol_failure_reason:
                return self._decide('protocol_failure', self._protocol_failure_reason)
            self.protocol_repairs += 1
            return self._decide('repair', 'current_observation_reference_repair_required')
        height_limit = (self.config.max_geometry_height_change_m if self._geometry_approach_active
                        else self.config.max_height_change_m)
        if abs(position[2] - self.origin_pose[2]) > height_limit:
            return self._recover('left_candidate_physical_height')

        # A caller-measured flat observation point can move us off an actual
        # stair tread, even when the semantic ROI is unbound. It never becomes
        # target evidence, and is subject to the same finite approach allowance.
        current_guard_accept = (verification.get('outcome') is True
                                and verification.get('completion_evidence_reliable') is True
                                and packet.get('accepted') is True
                                and packet.get('geometry_verified') is True)
        if verification.get('local_geometry_candidate') is not None and not current_guard_accept:
            choice = self._geometry_approach(verification, certified_actions, position)
            if choice is not None:
                if self.approaches >= self.config.max_approaches:
                    return self._recover('local_approach_budget_exhausted')
                self._geometry_approach_active = True
                action_id, target = choice
                return self._decide('approach', 'certified_stair_landing_observation', action_id, target)

        # Invalid/mixed regions are uncertainty, never a navigable 3D anchor.
        if packet.get('valid') is not True:
            return self._inspect('target_region_unbound')
        if packet.get('observation_id') != observation_id:
            return self._inspect('target_region_not_current')
        point = _point(packet.get('candidate_point', packet.get('world_center')))
        if point is None:
            return self._inspect('target_region_has_no_stable_point')
        target_id = packet.get('target_id')
        if self.candidate_point is not None:
            if (self.target_id is not None and target_id is not None and target_id != self.target_id):
                return self._inspect('target_identity_changed')
            shift_limit = max(self.config.max_target_shift_m, .90) if (
                packet.get('endpoint_kind') == 'doorway' or packet.get('expected_object_class') == 'door'
            ) else self.config.max_target_shift_m
            if _distance(point, self.candidate_point) > shift_limit:
                return self._inspect('target_world_region_changed')
        else:
            self.candidate_point, self.target_id = point, target_id

        changed_view = False
        if (self._witness is not None and observation_id - self._witness['observation_id']
                > self.config.max_observation_gap):
            self._witness = None
        if self._witness is not None:
            old = self._witness
            delta_yaw = (self._yaw - old['yaw'] + math.pi) % (2 * math.pi) - math.pi
            changed_view = (
                0 < observation_id - old['observation_id'] <= self.config.max_observation_gap
                and _distance(point, old['point']) <= self.config.max_target_shift_m
                and (_distance(position, old['pose']) >= self.config.min_view_translation_m
                     or abs(delta_yaw) >= self.config.min_view_rotation_rad))
        else:
            self._witness = {'observation_id': observation_id, 'pose': position,
                             'point': point, 'yaw': self._yaw}

        accepted = current_guard_accept
        if accepted and changed_view:
            return self._decide('ready', 'current_guard_accept_and_independent_fresh_view')
        if accepted:
            return self._inspect('independent_changed_view_required')

        if packet.get('gate_reason') == 'target_surface_too_far':
            if self.approaches >= self.config.max_approaches:
                return self._recover('local_approach_budget_exhausted')
            choice = self._approach(certified_actions, position)
            if choice is not None:
                action_id, target = choice
                return self._decide('approach', 'certified_local_progress', action_id, target)
            return self._recover('no_certified_local_approach')
        return self._inspect('endpoint_relation_or_confirmation_unresolved')

    def _geometry_approach(self, verification, actions, pose):
        hint = verification.get('local_geometry_candidate')
        progress = verification.get('stair_progress') or {}
        change = _number(progress.get('measured_height_change_m'), 0.)
        direction = progress.get('direction')
        measured_traversal = ((direction == 'DOWN' and change <= -.15)
                              or (direction == 'UP' and change >= .15))
        if (not isinstance(hint, dict) or verification.get('candidate_claimed') is not True
                or hint.get('kind') != 'stair_landing_observation'
                or hint.get('physically_supported') is not True
                or progress.get('direction_violation') is True or not measured_traversal):
            return None
        target = _point(hint.get('target_point_world'))
        action_id = hint.get('action_id')
        if (target is None or not isinstance(action_id, str) or not action_id
                or abs(target[2] - self.origin_pose[2]) > self.config.max_geometry_height_change_m):
            return None
        # The observation point must lie in the already measured stair direction.
        if (direction == 'DOWN' and target[2] > pose[2] + .04
                or direction == 'UP' and target[2] < pose[2] - .04):
            return None
        for action in actions:
            if (not isinstance(action, dict) or action.get('certified') is not True
                    or action.get('action_id') != action_id):
                continue
            certified_target = _point(action.get('target_point_world'))
            length = _number(action.get('path_length_m'))
            if (certified_target is not None and _distance(target, certified_target) <= .02
                    and 0 < length <= self.config.max_geometry_approach_path_m
                    and length + .001 >= _distance(pose, certified_target)):
                return action_id, certified_target
        return None

    def _approach(self, actions, pose):
        candidates = []
        current_distance = _distance(pose, self.candidate_point, 2)
        for action in actions:
            if not isinstance(action, dict) or action.get('certified') is not True:
                continue
            target = _point(action.get('target_point_world'))
            action_id = action.get('action_id')
            if target is None or not isinstance(action_id, str) or not action_id:
                continue
            length = _number(action.get('path_length_m'))
            distance = _distance(target, self.candidate_point, 2)
            if (length <= 0 or length > self.config.max_approach_path_m
                    or length + .001 < _distance(pose, target)
                    or abs(target[2] - self.origin_pose[2]) > self.config.max_height_change_m
                    or distance > current_distance - self.config.min_approach_gain_m
                    or distance < self.config.min_standoff_m):
                continue
            candidates.append((distance, action_id, target))
        if not candidates:
            return None
        _, action_id, target = min(candidates)
        return action_id, target


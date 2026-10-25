"""Pinned external review-data loader and single-helper bridge for blue_help.runner.

The caller MUST explicitly supply the external frozen review root. Every listed
file, frame and content ID is checked, but Python from that bundle is NEVER
executed or imported. Only this installed package's route_card implementation is
used; its exact source bytes must match the pinned bundle's reviewed contract.
No private scene card, images or coordinates are distributed with this package.
Consent, attempts, cooldown and delivered-question budget belong to the runner.
In-process information isolation is not an adversarial OS security sandbox.
"""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re
import struct
import weakref

from . import route_card as route_contract

FROZEN_CONTRACT_SHA256 = 'f260abd43f46cec74ae97dea0e9bde40935181105383ddc10069d1dd9a088a15'
FROZEN_MANIFEST_SHA256 = '06727cfe9f4f579f93793d9dc332beb84fba789ab3909efeaa047b511634bb02'
FROZEN_CARD_SHA256 = '701636d75593af576131b5634b45d2e01510fd53a8109978015f1a8a23c34341'
FROZEN_CARD_ID = 'route-card:4c38ea15abfca804e5740c6fa1ff7781645da8b704fb6749384e48722151b331'

# Weak references authenticate objects returned by the loader without keeping
# private full routes alive after their trusted caller releases them.
_VERIFIED_CARDS = weakref.WeakValueDictionary()


class CardBridgeError(ValueError):
    """Card/runtime binding failed; never silently substitute another oracle."""


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json(data):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise CardBridgeError('Duplicate JSON key')
            value[key] = item
        return value
    def nonfinite(value):
        raise CardBridgeError('Nonfinite JSON value')
    return json.loads(data, object_pairs_hook=unique, parse_constant=nonfinite)


def _file(root, relative):
    if not isinstance(relative, str) or not relative or '\\' in relative:
        raise CardBridgeError('Invalid evidence relative path')
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts:
        raise CardBridgeError('Evidence must remain inside frozen card root')
    result = (root/p).resolve()
    if not result.is_relative_to(root):
        raise CardBridgeError('Evidence symlink escapes frozen card root')
    return result


def _verified_contract(expected_external_hash):
    """External source is checked as data only; use the locally packaged module."""
    packaged_hash = _sha(Path(route_contract.__file__).read_bytes())
    if (packaged_hash != FROZEN_CONTRACT_SHA256
            or expected_external_hash != packaged_hash):
        raise CardBridgeError('Packaged contract bytes do not match pinned external contract source')
    return route_contract


def _reconstruct(module, raw):
    """Recompute every content-addressed ID; compare, never trust stored IDs."""
    def restore(kind, values, id_name):
        values = dict(values)
        wanted = values.pop(id_name)
        result = kind(**values)
        if getattr(result, id_name) != wanted:
            raise CardBridgeError('Immutable '+id_name+' mismatch')
        return result
    values = dict(raw)
    expected = values.pop('route_card_id')
    values['dummy'] = module.DummyGeometry(**values['dummy'])
    anchor = dict(values['anchor'])
    anchor['pose'] = module.Pose(**anchor['pose'])
    values['anchor'] = module.AnchorSpec(**anchor)
    values['goal'] = module.GoalAnnotation(**values['goal'])
    values['config'] = module.GeometryConfig(**values['config'])
    values['frames'] = tuple(restore(module.FrameEvidence, item, 'frame_id') for item in values['frames'])
    values['landmarks'] = tuple(restore(module.LandmarkEvidence, item, 'evidence_id') for item in values['landmarks'])
    card = module.RouteCard(**values)
    if card.route_card_id != expected:
        raise CardBridgeError('Immutable route_card_id mismatch')
    return card


def load_frozen_card(review_root, *, expected_manifest_sha256=FROZEN_MANIFEST_SHA256,
                     expected_card_sha256=FROZEN_CARD_SHA256, expected_card_id=FROZEN_CARD_ID):
    """Load private card after pinned manifest, all listed files, IDs and text checks.

    Explicit non-default pins are for controlled fixtures/newly approved snapshots,
    never discovered automatically after a mismatch. Frame hashes authenticate
    bytes, not object-class truth. Existing manual labels remain manual labels.
    No Habitat, model, detector, API, upload, or image inference is run here.
    """
    root = Path(review_root).resolve()
    try:
        manifest_bytes = _file(root, 'MANIFEST.json').read_bytes()
        if _sha(manifest_bytes) != expected_manifest_sha256:
            raise CardBridgeError('Frozen manifest hash mismatch')
        manifest = _json(manifest_bytes)
        entries = {}
        for item in manifest['files']:
            relative = item['path']
            if relative in entries:
                raise CardBridgeError('Duplicate manifest path')
            content = _file(root, relative).read_bytes()
            if (type(item['bytes']) is not int or len(content) != item['bytes']
                    or _sha(content) != item['sha256']):
                raise CardBridgeError('Frozen manifest file checksum mismatch: '+relative)
            entries[relative] = item
        required = ('route_card.private.json', 'episode40_route_body_en.txt', 'contracts/route_card.py')
        if any(name not in entries for name in required):
            raise CardBridgeError('Required card/source/text missing from pinned manifest')
        card_bytes = _file(root, required[0]).read_bytes()
        if _sha(card_bytes) != expected_card_sha256:
            raise CardBridgeError('Frozen card file hash mismatch')
        raw = _json(card_bytes)
        if raw['route_card_id'] != expected_card_id:
            raise CardBridgeError('Unexpected approved route-card identity')
        module = _verified_contract(entries['contracts/route_card.py']['sha256'])
        card = _reconstruct(module, raw)
        text = _file(root, required[1]).read_bytes().decode('utf-8')
        if card.route_text != text:
            raise CardBridgeError('Full final manual route text changed or was truncated')
        for frame in card.frames:
            if not re.fullmatch(r'[A-Za-z0-9_-]+', frame.source_frame_label):
                raise CardBridgeError('Unsafe frame file label')
            relative = 'evidence_episode40_attempt01/'+frame.source_frame_label+'.png'
            if relative not in entries:
                raise CardBridgeError('Referenced frame missing from pinned manifest')
            image = _file(root, relative).read_bytes()
            if not frame.matches_image_bytes(image):
                raise CardBridgeError('Evidence image does not match immutable frame')
            if (len(image) < 24 or image[:8] != b'\x89PNG\r\n\x1a\n'
                    or image[12:16] != b'IHDR'
                    or struct.unpack('>II', image[16:24]) != (frame.width, frame.height)):
                raise CardBridgeError('Referenced PNG dimensions differ from reviewed frame')
        _VERIFIED_CARDS[id(card)] = card
        return card
    except CardBridgeError:
        raise
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as exc:
        raise CardBridgeError('Frozen card failed strict read/structure verification') from exc


def _point(value):
    values = tuple(value)
    if len(values) != 3 or any(isinstance(v, bool) or not isinstance(v, Real)
                               or not math.isfinite(v) for v in values):
        raise CardBridgeError('Invalid private XYZ')
    return tuple(float(v) for v in values)


def _scene(value):
    if not isinstance(value, str) or not value:
        raise CardBridgeError('Scene identity required')
    value = value.replace('\\', '/')
    return Path(value).stem if value.endswith('.glb') else value


def _same_pose(a, b):
    return (math.dist(_point(a.position), _point(b.position)) <= 1e-6
            and all(isinstance(p.yaw_rad, Real) and not isinstance(p.yaw_rad, bool)
                    and math.isfinite(p.yaw_rad) for p in (a, b))
            and abs((a.yaw_rad-b.yaw_rad+math.pi) % (2*math.pi)-math.pi) <= 1e-6)


def _spec_matches(spec, dummy):
    return (spec.dummy_id == dummy.dummy_id
            and math.dist(_point(spec.ground_position), dummy.foot) <= 1e-6
            and abs(spec.radius_m-dummy.radius_m) <= 1e-9
            and abs(spec.height_m-dummy.height_m) <= 1e-9)


def _live_binding(env, dummy, config, episode_id):
    if str(env.episode_id) != episode_id:
        return False
    actors = env._actors
    if not isinstance(actors, (tuple, list)) or len(actors) != 1 or not _spec_matches(actors[0]['spec'], dummy):
        return False
    audit = env.placement_audit
    if (not isinstance(audit, list) or len(audit) != 1
            or audit[0].get('dummy_id') != dummy.dummy_id
            or audit[0].get('status') != 'pass_coarse_checks_only'
            or audit[0].get('bullet_contact') is not False
            or audit[0].get('radial_rays_clear') is not True):
        return False
    names = (('agent_radius_m', 'robot_radius_m'), ('agent_height_m', 'robot_height_m'),
             ('collision_margin_m', 'body_margin_m'), ('interaction_radius_m', 'interaction_radius_m'),
             ('floor_tolerance_m', 'floor_tolerance_m'))
    for actual, expected in names:
        value = getattr(env.config, actual)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            return False
        if abs(value-getattr(config, expected)) > 1e-9:
            return False
    # The approved card's geometry revision explicitly uses sliding motion.
    return env.config.allow_sliding is True


class CardOracle:
    """Runtime card answer bridge: no goal/full-path/card retained or returned."""
    def __init__(self, connector, env, dummy, config, scene_id, revision, episode_id):
        self._connector, self._env, self._dummy, self._config = connector, env, dummy, config
        self._scene_id, self._revision, self._episode_id = scene_id, revision, episode_id

    def _current(self, pose, step):
        if not _live_binding(self._env, self._dummy, self._config, self._episode_id):
            return None
        frame = self._env.observe()
        if frame.step != step or type(frame.step) is not int or not _same_pose(frame.pose, pose):
            return None
        candidates = frame.dummy_candidates
        if not isinstance(candidates, (tuple, list)) or len(candidates) != 1:
            return None
        item = candidates[0]
        if (item.get('dummy_id') != self._dummy.dummy_id or item.get('visible') is not True
                or item.get('spatial_eligible') is not True or item.get('reason_codes') != []
                or item.get('interaction_radius_m') != self._config.interaction_radius_m):
            return None
        measured = math.hypot(frame.pose.position[0]-self._dummy.foot[0],
                              frame.pose.position[2]-self._dummy.foot[2])
        distance = item.get('horizontal_distance_m')
        if (isinstance(distance, bool) or not isinstance(distance, Real) or not math.isfinite(distance)
                or distance < 0 or abs(distance-measured) > 1e-6
                or measured > self._config.interaction_radius_m
                or abs(frame.pose.position[1]-self._dummy.foot[1]) > self._config.floor_tolerance_m):
            return None
        return frame

    def answer(self, vehicle_pose, question_step, public_history=()):
        # public_history is deliberately unused: this frozen reviewed card does
        # not invent a cause of error or revise its route from private metrics.
        if type(question_step) is not int or question_step < 0:
            return None
        try:
            fresh = self._current(vehicle_pose, question_step)
            if fresh is None:
                return None
            text = self._connector.answer_text(fresh.pose, scene_id=self._scene_id,
                geometry_revision=self._revision, dummy_id=self._dummy.dummy_id,
                observation_step=fresh.step, request_step=question_step,
                visible=True, line_of_sight_clear=True, same_floor=True)
            # Validate again after the local path service, before returning an
            # answer. Runner independently rechecks delivery and counts budget.
            if text is None or self._current(vehicle_pose, question_step) is None:
                return None
            from blue_help.oracle import DirectionAnswer
            from blue_help.contracts import Pose
            return DirectionAnswer(text, Pose(_point(vehicle_pose.position), float(vehicle_pose.yaw_rad)),
                                   question_step, False)
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError, ArithmeticError):
            return None


class _Factory:
    def __init__(self, env, card, dummies, revision, episode_id):
        module = route_contract
        if (not isinstance(card, module.RouteCard)
                or _VERIFIED_CARDS.get(id(card)) is not card):
            raise CardBridgeError('Use a card reconstructed by the verified frozen loader')
        if revision != card.geometry_revision or str(episode_id) != '40':
            raise CardBridgeError('Only the approved geometry revision and episode 40 are allowed')
        selected = tuple(dummies)
        if len(selected) != 1 or not _spec_matches(selected[0], card.dummy):
            raise CardBridgeError('Runner lacks chosen-dummy argument: exactly the verified helper is required')
        self._goal_to_validate = card.goal.position  # discarded after one factory call
        self._connector = module.LocalConnector(card, env.find_safe_path)
        self._env, self._dummy, self._config = env, card.dummy, card.config
        self._scene_id, self._revision, self._episode_id = card.scene_id, revision, '40'
        self._used = False
        self.route_card_id = card.route_card_id  # non-geometric provenance only

    def __call__(self, goal_position, find_safe_path, *, scene_id):
        if self._used:
            raise CardBridgeError('Single-episode oracle factory cannot be reused')
        if (math.dist(_point(goal_position), self._goal_to_validate) > 1e-6
                or _scene(scene_id) != self._scene_id):
            raise CardBridgeError('Episode goal/scene does not match the approved manual card')
        if (find_safe_path != self._env.find_safe_path
                or not _live_binding(self._env, self._dummy, self._config, self._episode_id)):
            raise CardBridgeError('Live single-helper environment/geometry does not match the card')
        self._used, self._goal_to_validate = True, None
        return CardOracle(self._connector, self._env, self._dummy, self._config,
                          self._scene_id, self._revision, self._episode_id)


def make_card_oracle_factory(env, card, dummies, *, geometry_revision=None, episode_id='40'):
    """Return run_episode-compatible factory(goal, find_safe_path, scene_id=...).

    Build before runner reset if needed; actual actors/config are checked when
    runner invokes it after reset and again at each answer. Exactly one supplied
    and one live helper is required, even if another helper is currently hidden.
    Scene/NavMesh files must separately be pinned by the trusted run entrypoint.
    """
    return _Factory(env, card, dummies, card.geometry_revision if geometry_revision is None
                    else geometry_revision, episode_id)

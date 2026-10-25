"""Synthetic unit fixtures; no Habitat/API/model, key access or secret retrieval."""
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace as NS
import unittest
import zlib

from blue_help.contracts import DummySpec, Pose, SensorFrame
from blue_help.oracle import DirectionAnswer
from blue_help.manual_route import bridge, route_card as route_contract
from unittest.mock import patch


def digest(value):
    return hashlib.sha256(value).hexdigest()


def png():
    def chunk(name, data):
        return struct.pack('>I', len(data))+name+data+struct.pack('>I', zlib.crc32(name+data)&0xffffffff)
    return (b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', struct.pack('>IIBBBBB', 2, 2, 8, 2, 0, 0, 0))
            +chunk(b'IDAT', zlib.compress(b'\x00'+b'\x00\x00\xff'*2+b'\x00'+b'\x00\x00\xff'*2))
            +chunk(b'IEND', b''))


class Fixture:
    def __init__(self, root):
        self.root = root
        self.paths = []
        source = Path(route_contract.__file__)
        self.module = route_contract
        m = self.module
        image = png()
        frame = m.FrameEvidence('fixture-scene', digest(image), 2, 2, 'fixture_frame')
        evidence = m.LandmarkEvidence(frame.frame_id, 'chair', (0, 0, 1, 1),
            'manual_annotation', 'Synthetic review record; not actual scene evidence')
        self.text = ('SYNTHETIC FULL MANUAL ROUTE. Go forward along this fixture route.\n'
                     'Do not truncate this final reviewed stage. END OF FULL TEXT.\n')
        self.card = m.RouteCard('fixture-scene', 'fixture/sliding/fixed-v1',
            m.DummyGeometry('fixture-helper', (0, 0, 0)),
            m.AnchorSpec('fixture-anchor', m.Pose((0, 0, -.8), 0), 'Synthetic anchor check'),
            m.GoalAnnotation('fixture/40/goals/0', (0, 0, -10), frame.frame_id, 'Synthetic goal review'),
            self.text, 'Synthetic complete prose review', (frame,), (evidence,), (evidence.evidence_id,),
            ((0, 0, -.8), (0, 0, -4), (0, 0, -10)))
        self.write('contracts/route_card.py', source.read_bytes())
        self.write('route_card.private.json', json.dumps(asdict(self.card), indent=2).encode())
        self.write('episode40_route_body_en.txt', self.text.encode())
        self.write('evidence_episode40_attempt01/fixture_frame.png', image)
        self.refresh_pins()

    def write(self, relative, data):
        path = self.root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        if relative not in self.paths:
            self.paths.append(relative)

    def refresh_pins(self, *, card_id=None):
        manifest = {'files': [{'path': name, 'sha256': digest((self.root/name).read_bytes()),
            'bytes': (self.root/name).stat().st_size} for name in self.paths]}
        raw = json.dumps(manifest, indent=2).encode()
        (self.root/'MANIFEST.json').write_bytes(raw)
        self.pins = dict(expected_manifest_sha256=digest(raw),
            expected_card_sha256=digest((self.root/'route_card.private.json').read_bytes()),
            expected_card_id=card_id or self.card.route_card_id)

    def load(self):
        return bridge.load_frozen_card(self.root, **self.pins)


class FakeEnvironment:
    def __init__(self, card):
        self.episode_id = '40'
        self.config = NS(agent_radius_m=.1, agent_height_m=1.5, collision_margin_m=.03,
            interaction_radius_m=1., floor_tolerance_m=.25, allow_sliding=True)
        self.dummy = DummySpec(card.dummy.dummy_id, card.dummy.foot, 0,
                              card.dummy.radius_m, card.dummy.height_m)
        self._actors = [{'spec': self.dummy}]
        self.placement_audit = [dict(dummy_id=self.dummy.dummy_id, status='pass_coarse_checks_only',
            bullet_contact=False, radial_rays_clear=True)]
        self.pose = Pose((.4, 0, -.8), 0.)
        self.step_count = 17
        self.path_calls = []
        self.observe_calls = 0
        self.extra_candidate = False
        self.candidate_override = {}
        self.after_path = None
        self.path_value = 'normal'

    def observe(self):
        self.observe_calls += 1
        candidate = dict(dummy_id=self.dummy.dummy_id, visible=True, spatial_eligible=True,
            reason_codes=[], interaction_radius_m=1.,
            horizontal_distance_m=math.hypot(self.pose.position[0], self.pose.position[2]))
        candidate.update(self.candidate_override)
        rows = [candidate]
        if self.extra_candidate:
            rows.append(dict(candidate, dummy_id='other-invisible-helper', visible=False))
        return SensorFrame(None, None, self.pose, self.step_count, dummy_candidates=rows)

    def find_safe_path(self, start, end):
        self.path_calls.append((start, end))
        if self.after_path:
            self.after_path()
        return [start, end] if self.path_value == 'normal' else self.path_value

    def step(self, action):
        raise AssertionError('Bridge test must not execute navigation')


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='bridge_fixture_', dir=Path(__file__).parent)
        self.fixture = Fixture(Path(self.tmp.name))
        self.card = self.fixture.load()
        self.env = FakeEnvironment(self.card)

    def tearDown(self):
        self.tmp.cleanup()

    def factory(self, **kwargs):
        return bridge.make_card_oracle_factory(self.env, self.card, [self.env.dummy], **kwargs)

    def oracle(self):
        return self.factory()(self.card.goal.position, self.env.find_safe_path,
                              scene_id='mp3d/fixture-scene/fixture-scene.glb')

    def test_loader_verifies_full_final_text_and_recomputed_ids(self):
        self.assertEqual(self.card.route_text, self.fixture.text)
        self.assertTrue(self.card.route_text.endswith('END OF FULL TEXT.\n'))
        self.assertEqual(self.card.route_card_id, self.fixture.card.route_card_id)
        self.assertEqual(self.card.frames[0].frame_id, self.fixture.card.frames[0].frame_id)
        self.assertFalse((self.fixture.root/'contracts/__pycache__').exists())

    def test_external_root_is_required_and_default_pins_are_not_replaced_by_fixture(self):
        with self.assertRaises(TypeError):
            bridge.load_frozen_card()
        with self.assertRaises(bridge.CardBridgeError):
            bridge.load_frozen_card(self.fixture.root)
        self.assertEqual(digest(Path(route_contract.__file__).read_bytes()), bridge.FROZEN_CONTRACT_SHA256)
        self.assertEqual(self.fixture.load().route_text, self.fixture.text)

    def test_loader_never_executes_code_from_external_review_bundle(self):
        with patch('builtins.exec', side_effect=AssertionError('No dynamic execution allowed')):
            loaded = self.fixture.load()
        self.assertIs(type(loaded), route_contract.RouteCard)
        self.assertFalse((self.fixture.root/'contracts/__pycache__').exists())

    def test_re_pinned_external_python_cannot_replace_packaged_contract(self):
        marker = self.fixture.root/'must_not_be_created'
        changed = (self.fixture.root/'contracts/route_card.py').read_bytes()
        changed += ('\n__import__("pathlib").Path('+repr(str(marker))+').write_text("executed")\n').encode()
        self.fixture.write('contracts/route_card.py', changed)
        self.fixture.refresh_pins()
        with self.assertRaisesRegex(bridge.CardBridgeError, 'Packaged contract bytes'):
            self.fixture.load()
        self.assertFalse(marker.exists())

    def test_manifest_path_escape_and_duplicate_keys_fail_closed(self):
        manifest = json.loads((self.fixture.root/'MANIFEST.json').read_bytes())
        manifest['files'][0]['path'] = '../outside-bundle'
        raw = json.dumps(manifest).encode()
        (self.fixture.root/'MANIFEST.json').write_bytes(raw)
        with self.assertRaisesRegex(bridge.CardBridgeError, 'inside frozen card root'):
            bridge.load_frozen_card(self.fixture.root,
                **dict(self.fixture.pins, expected_manifest_sha256=digest(raw)))
        raw = b'{"files": [], "files": []}'
        (self.fixture.root/'MANIFEST.json').write_bytes(raw)
        with self.assertRaisesRegex(bridge.CardBridgeError, 'Duplicate JSON key'):
            bridge.load_frozen_card(self.fixture.root,
                **dict(self.fixture.pins, expected_manifest_sha256=digest(raw)))

    def test_factory_requires_loader_verified_object_not_merely_matching_card_type(self):
        same_content_unverified = replace(self.card)
        self.assertEqual(same_content_unverified.route_card_id, self.card.route_card_id)
        with self.assertRaisesRegex(bridge.CardBridgeError, 'verified frozen loader'):
            bridge.make_card_oracle_factory(self.env, same_content_unverified, [self.env.dummy])

    def test_wrong_manifest_card_file_or_evidence_hash_fail_closed(self):
        with self.assertRaises(bridge.CardBridgeError):
            bridge.load_frozen_card(self.fixture.root, **dict(self.fixture.pins, expected_manifest_sha256='0'*64))
        with self.assertRaises(bridge.CardBridgeError):
            bridge.load_frozen_card(self.fixture.root, **dict(self.fixture.pins, expected_card_sha256='0'*64))
        frame = self.fixture.root/'evidence_episode40_attempt01/fixture_frame.png'
        frame.write_bytes(frame.read_bytes()+b'changed')
        with self.assertRaises(bridge.CardBridgeError):
            self.fixture.load()
        self.fixture.refresh_pins()  # even a newly pinned manifest cannot falsify the card's image hash
        with self.assertRaises(bridge.CardBridgeError):
            self.fixture.load()

    def test_changed_card_id_frame_id_evidence_id_rejected_after_file_pins_updated(self):
        original = (self.fixture.root/'route_card.private.json').read_bytes()
        for location in ('card', 'frame', 'evidence'):
            raw = json.loads(original)
            if location == 'card':
                raw['route_card_id'] = 'route-card:'+'0'*64
            elif location == 'frame':
                raw['frames'][0]['frame_id'] = 'frame:'+'0'*64
            else:
                raw['landmarks'][0]['evidence_id'] = 'evidence:'+'0'*64
            self.fixture.write('route_card.private.json', json.dumps(raw).encode())
            self.fixture.refresh_pins(card_id=raw['route_card_id'])
            with self.subTest(location=location), self.assertRaises(bridge.CardBridgeError):
                self.fixture.load()

    def test_truncated_final_text_rejected_even_with_updated_manifest(self):
        self.fixture.write('episode40_route_body_en.txt', b'SYNTHETIC FULL MANUAL ROUTE.')
        self.fixture.refresh_pins()
        with self.assertRaises(bridge.CardBridgeError):
            self.fixture.load()

    def test_factory_can_be_created_before_reset_but_checks_live_state_when_called(self):
        self.env._actors = []
        self.env.placement_audit = []
        del self.env.episode_id
        factory = self.factory()  # no actual live-state dependency at construction
        self.env.episode_id = '40'
        self.env._actors = [{'spec': self.env.dummy}]
        self.env.placement_audit = [dict(dummy_id=self.env.dummy.dummy_id,
            status='pass_coarse_checks_only', bullet_contact=False, radial_rays_clear=True)]
        oracle = factory(self.card.goal.position, self.env.find_safe_path, scene_id='fixture-scene')
        self.assertIsInstance(oracle.answer(self.env.pose, 17), DirectionAnswer)

    def test_factory_goal_scene_revision_episode_and_single_use_binding(self):
        for changes in ({'geometry_revision': 'different'}, {'episode_id': '41'}):
            with self.assertRaises(bridge.CardBridgeError):
                self.factory(**changes)
        for goal, scene in (((1, 0, -10), 'fixture-scene'), (self.card.goal.position, 'different-scene')):
            with self.assertRaises(bridge.CardBridgeError):
                self.factory()(goal, self.env.find_safe_path, scene_id=scene)
        factory = self.factory()
        oracle = factory(self.card.goal.position, self.env.find_safe_path, scene_id='fixture-scene')
        self.assertIsNone(factory._goal_to_validate)
        self.assertFalse(any(isinstance(v, type(self.card)) for v in vars(oracle).values()))
        self.assertNotIn('_goal', vars(oracle))
        with self.assertRaises(bridge.CardBridgeError):
            factory(self.card.goal.position, self.env.find_safe_path, scene_id='fixture-scene')

    def test_single_selected_and_live_helper_are_both_required(self):
        for dummies in ([], [self.env.dummy]*2, [replace(self.env.dummy, dummy_id='wrong')],
                        [replace(self.env.dummy, ground_position=(1, 0, 0))]):
            with self.subTest(dummies=dummies), self.assertRaises(bridge.CardBridgeError):
                bridge.make_card_oracle_factory(self.env, self.card, dummies)
        factory = self.factory()
        self.env._actors.append({'spec': replace(self.env.dummy, dummy_id='other-hidden')})
        with self.assertRaises(bridge.CardBridgeError):
            factory(self.card.goal.position, self.env.find_safe_path, scene_id='fixture-scene')

    def test_runtime_returns_original_directionanswer_and_only_queries_local_anchor(self):
        oracle = self.oracle()
        result = oracle.answer(self.env.pose, 17, public_history=[{'hidden_goal': 'ignored'}])
        self.assertIsInstance(result, DirectionAnswer)
        self.assertEqual(result.reference_pose, self.env.pose)
        self.assertEqual(result.question_step, 17)
        self.assertFalse(result.terminal)
        self.assertTrue(result.text.endswith(self.fixture.text))
        self.assertNotIn('fixture/40/goals/0', result.text)
        self.assertEqual(self.env.path_calls, [(self.env.pose.position, self.card.anchor.pose.position)])
        self.assertEqual(self.env.observe_calls, 2)

    def test_mismatched_pose_yaw_or_step_refuses_before_query(self):
        oracle = self.oracle()
        for pose, step in ((Pose((.5, 0, -.8), 0), 17), (Pose((.4, 0, -.8), .1), 17),
                           (self.env.pose, 16), (self.env.pose, True)):
            self.assertIsNone(oracle.answer(pose, step))
        self.assertEqual(self.env.path_calls, [])

    def test_fresh_radius_visibility_eligibility_and_ambiguity_are_enforced(self):
        oracle = self.oracle()
        for overrides in ({'visible': False}, {'spatial_eligible': False},
                          {'interaction_radius_m': 1.2}, {'horizontal_distance_m': .1},
                          {'reason_codes': ['occluded_or_unverified_line_of_sight']},
                          {'dummy_id': 'wrong-selected-helper'}):
            self.env.candidate_override = overrides
            self.assertIsNone(oracle.answer(self.env.pose, 17))
        self.env.candidate_override = {}
        self.env.extra_candidate = True
        self.assertIsNone(oracle.answer(self.env.pose, 17))
        self.env.extra_candidate = False
        self.env.pose = Pose((0, 0, -1.01), 0)
        self.assertIsNone(oracle.answer(self.env.pose, 17))
        self.assertEqual(self.env.path_calls, [])

    def test_environment_geometry_radius_and_placement_changes_refuse(self):
        oracle = self.oracle()
        self.env.config.interaction_radius_m = 1.2
        self.assertIsNone(oracle.answer(self.env.pose, 17))
        self.env.config.interaction_radius_m = 1.
        self.env.placement_audit[0]['status'] = 'pending'
        self.assertIsNone(oracle.answer(self.env.pose, 17))
        self.assertEqual(self.env.path_calls, [])

    def test_post_query_pose_step_or_eligibility_changes_prevent_delivery(self):
        oracle = self.oracle()
        original = self.env.pose
        self.env.after_path = lambda: setattr(self.env, 'pose', Pose((.5, 0, -.8), 0))
        self.assertIsNone(oracle.answer(original, 17))
        self.env.pose = original
        self.env.after_path = lambda: setattr(self.env, 'step_count', 18)
        self.assertIsNone(oracle.answer(original, 17))
        self.env.step_count = 17
        self.env.after_path = lambda: self.env.candidate_override.update(visible=False)
        self.assertIsNone(oracle.answer(original, 17))
        self.assertTrue(all(end == self.card.anchor.pose.position for _, end in self.env.path_calls))

    def test_local_route_unavailable_does_not_fall_back_to_goal_or_old_answer(self):
        oracle = self.oracle()
        self.env.path_value = None
        self.assertIsNone(oracle.answer(self.env.pose, 17))
        self.assertEqual(self.env.path_calls, [(self.env.pose.position, self.card.anchor.pose.position)])

    def test_factory_rejects_nonmatching_path_service_and_non_sliding_profile(self):
        with self.assertRaises(bridge.CardBridgeError):
            self.factory()(self.card.goal.position, lambda a, b: [a, b], scene_id='fixture-scene')
        self.env.config.allow_sliding = False
        with self.assertRaises(bridge.CardBridgeError):
            self.factory()(self.card.goal.position, self.env.find_safe_path, scene_id='fixture-scene')


if __name__ == '__main__':
    unittest.main()

"""Synthetic GLB hash fixtures only; no actual game model/Habitat/API invocation."""
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from blue_help.contracts import DummySpec
from blue_help import helper_assets as helpers


def synthetic_glb():
    payload = b'{"asset":{"version":"2.0"},"scene":0,"scenes":[{}]}'
    payload += b' '*((-len(payload)) % 4)
    chunk = struct.pack('<II', len(payload), 0x4e4f534a)+payload
    return struct.pack('<4sII', b'glTF', 2, 12+len(chunk))+chunk


class HelperAssetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='synthetic_helper_')
        self.root = Path(self.tmp.name)
        self.asset = self.root/'existing_game.glb'
        self.data = synthetic_glb()
        self.asset.write_bytes(self.data)
        self.source_config = self.root/'source.object_config.json'
        self.source_config.write_text('{"original":true}\n')
        self.output = self.root/'derived'

    def tearDown(self):
        self.tmp.cleanup()

    def pinned_fixture(self):
        # Test-only substitution; production has NO argument or environment
        # variable permitting a weaker/replaced game asset pin.
        return patch.object(helpers, 'GAME_V7_SHA256', hashlib.sha256(self.data).hexdigest())

    def test_default_game_requires_explicit_local_asset_without_any_output(self):
        self.assertEqual(helpers.DEFAULT_HELPER_PROFILE, 'game-v7')
        self.assertEqual(helpers.GAME_V7_SHA256,
                         'd51934f336e4b9b91b64de4b84f4f2296f8ec01ccfbce108520322758c240558')
        with self.assertRaisesRegex(ValueError, '--helper-asset'):
            helpers.prepare_helper(output_dir=self.output)
        self.assertFalse(self.output.exists())

    def test_default_game_emits_only_derived_config_with_unmodified_source(self):
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in (self.asset, self.source_config)}
        with self.pinned_fixture():
            prepared = helpers.prepare_helper(asset_path=self.asset, output_dir=self.output)
        self.assertEqual(prepared.profile, 'game-v7')
        self.assertEqual(prepared.visibility_mode, 'semantic_instance')
        self.assertEqual((prepared.floor_offset_m, prepared.radius_m, prepared.height_m), (0, .32, 1.6))
        self.assertEqual(prepared.config_path, self.output/'runtime_helper.object_config.json')
        self.assertEqual(prepared.object_config_path, prepared.config_path)
        self.assertEqual([p.name for p in self.output.iterdir()], ['runtime_helper.object_config.json'])
        config = json.loads(prepared.config_path.read_text())
        self.assertEqual(config, {'render_asset': str(self.asset.resolve()), 'collision_asset': str(self.asset.resolve()),
            'mass': 1., 'up': [0., 1., 0.], 'front': [0., 0., -1.], 'COM': [0., 0., 0.],
            'scale': [1., 1., 1.], 'units_to_meters': 1., 'is_collidable': True})
        for p in (self.asset, self.source_config):
            self.assertEqual((p.read_bytes(), p.stat().st_mtime_ns), before[p.name])
        self.assertFalse(prepared.provenance['new_model_created'])
        self.assertFalse(prepared.provenance['source_asset_modified'])
        self.assertFalse(prepared.provenance['geometry_actual_validated'])
        self.assertNotIn(str(self.root), json.dumps(prepared.provenance))
        self.assertEqual(prepared.provenance['derived_config_sha256'],
                         hashlib.sha256(prepared.config_path.read_bytes()).hexdigest())

    def test_missing_or_wrong_hash_never_creates_output_or_falls_back(self):
        with self.assertRaises(FileNotFoundError):
            helpers.prepare_helper(asset_path=self.root/'missing.glb', output_dir=self.output)
        self.assertFalse(self.output.exists())
        with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
            helpers.prepare_helper(asset_path=self.asset, output_dir=self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.asset.read_bytes(), self.data)

    def test_unknown_profile_rejected_before_read_or_write(self):
        for profile in ('blue', 'auto', '', None):
            with self.subTest(profile=profile), self.assertRaisesRegex(ValueError, 'Unknown helper profile'):
                helpers.prepare_helper(profile, self.root/'not-present.glb', self.output)
        self.assertFalse(self.output.exists())

    def test_invalid_glb_header_truncation_and_extension_fail_even_with_test_pin(self):
        cases = [b'not a GLB', struct.pack('<4sII', b'glTF', 1, 20)+b'12345678',
                 struct.pack('<4sII', b'glTF', 2, 100)+b'12345678']
        for data in cases:
            self.asset.write_bytes(data)
            with patch.object(helpers, 'GAME_V7_SHA256', hashlib.sha256(data).hexdigest()):
                with self.subTest(data=data), self.assertRaises(ValueError):
                    helpers.prepare_helper(asset_path=self.asset, output_dir=self.output)
            self.assertFalse(self.output.exists())
        other = self.root/'wrong.obj'
        other.write_bytes(self.data)
        with self.pinned_fixture(), self.assertRaisesRegex(ValueError, '.glb'):
            helpers.prepare_helper(asset_path=other, output_dir=self.output)

    def test_explicit_fresh_output_required_and_existing_evidence_unchanged(self):
        with self.pinned_fixture(), self.assertRaisesRegex(ValueError, 'fresh output_dir'):
            helpers.prepare_helper(asset_path=self.asset)
        self.output.mkdir()
        evidence = self.output/'existing.txt'
        evidence.write_text('preserve existing evidence')
        with self.pinned_fixture(), self.assertRaises(FileExistsError):
            helpers.prepare_helper(asset_path=self.asset, output_dir=self.output)
        self.assertEqual(evidence.read_text(), 'preserve existing evidence')
        self.assertEqual([p.name for p in self.output.iterdir()], ['existing.txt'])

    def test_output_symlink_is_not_used_to_overwrite_or_retarget(self):
        link = self.root/'derived_link'
        link.symlink_to(self.root/'nonexistent_target', target_is_directory=True)
        with self.pinned_fixture(), self.assertRaises(FileExistsError):
            helpers.prepare_helper(asset_path=self.asset, output_dir=link)
        self.assertFalse((self.root/'nonexistent_target').exists())
        self.assertTrue(link.is_symlink())

    def test_blue_reference_is_explicit_packaged_and_creates_no_output(self):
        source = helpers._BLUE_CONFIG_PATH
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in source.parent.iterdir() if p.is_file()}
        prepared = helpers.prepare_helper(profile='blue-reference', output_dir=self.output)
        self.assertEqual(prepared.config_path, source.resolve())
        self.assertEqual(prepared.visibility_mode, 'semantic_blue')
        self.assertEqual((prepared.floor_offset_m, prepared.radius_m, prepared.height_m), (.8, .32, 1.6))
        self.assertFalse(prepared.provenance['derived_config_created'])
        self.assertFalse(self.output.exists())
        for p in source.parent.iterdir():
            if p.is_file():
                self.assertEqual((p.read_bytes(), p.stat().st_mtime_ns), before[p.name])
        with self.assertRaisesRegex(ValueError, 'packaged asset'):
            helpers.prepare_helper(profile='blue-reference', asset_path=self.asset, output_dir=self.output)

    def test_missing_packaged_blue_asset_is_error_not_game_fallback(self):
        with patch.object(helpers, '_BLUE_CONFIG_PATH', self.root/'absent.object_config.json'):
            with self.assertRaises(FileNotFoundError):
                helpers.prepare_helper(profile='blue-reference', output_dir=self.output)
        self.assertFalse(self.output.exists())

    def test_dummy_replacement_changes_only_offset_and_proxy_fields(self):
        dummy = DummySpec('keep-this-id', (1., 2., 3.), .7, radius_m=.4, height_m=1.8, floor_offset_m=.8)
        with self.pinned_fixture():
            prepared = helpers.prepare_helper(asset_path=self.asset, output_dir=self.output)
        changed = helpers.replace_dummy_for_helper(dummy, prepared)
        self.assertIsNot(changed, dummy)
        self.assertEqual((changed.dummy_id, changed.ground_position, changed.yaw_rad),
                         (dummy.dummy_id, dummy.ground_position, dummy.yaw_rad))
        self.assertEqual((changed.floor_offset_m, changed.radius_m, changed.height_m), (0., .32, 1.6))
        self.assertEqual((dummy.floor_offset_m, dummy.radius_m, dummy.height_m), (.8, .4, 1.8))
        with self.assertRaises(FrozenInstanceError):
            prepared.floor_offset_m = .8
        with self.assertRaises(ValueError):
            helpers.replace_dummy_for_helper({}, prepared)
        with self.assertRaises(ValueError):
            helpers.replace_dummy_for_helper(dummy, {})

    def test_blue_replacement_restores_only_explicit_reference_offset(self):
        dummy = DummySpec('same', (0, 0, 0), 0, floor_offset_m=0)
        prepared = helpers.prepare_helper('blue-reference')
        self.assertEqual(helpers.replace_dummy_for_helper(dummy, prepared).floor_offset_m, .8)
        self.assertEqual(dummy.floor_offset_m, 0)


if __name__ == '__main__':
    unittest.main()

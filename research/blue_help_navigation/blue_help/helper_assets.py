"""Explicit existing-character profiles; no download, model creation or source edits.

The default game-v7 helper MUST be supplied as a local, pinned GLB. It is never
silently replaced by the historical blue figure. Only a derived Habitat object
configuration is created, inside a fresh caller-selected output directory.
Dimensions remain conservative proxies, NOT measured game-asset geometry: the
adapter must reject an actual AABB that does not fit rather than resize it.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import struct

from .contracts import DummySpec

DEFAULT_HELPER_PROFILE = 'game-v7'
GAME_V7_SHA256 = 'd51934f336e4b9b91b64de4b84f4f2296f8ec01ccfbce108520322758c240558'
_BLUE_CONFIG_PATH = Path(__file__).resolve().parent/'assets/blue_dummy/light_blue_human.object_config.json'


@dataclass(frozen=True)
class PreparedHelper:
    profile: str
    config_path: Path
    floor_offset_m: float
    radius_m: float
    height_m: float
    visibility_mode: str
    provenance: dict

    @property
    def object_config_path(self) -> Path:
        """Explicit alias for the environment factory's parameter name."""
        return self.config_path


def _file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_game_asset(asset_path):
    if asset_path is None:
        raise ValueError('game-v7 requires an explicit local --helper-asset GLB path; no blue fallback')
    requested = Path(asset_path)
    if requested.suffix.lower() != '.glb':
        raise ValueError('--helper-asset must be a local .glb file')
    asset = requested.resolve(strict=True)
    if not asset.is_file():
        raise ValueError('--helper-asset must refer to a regular GLB file')
    byte_count = asset.stat().st_size
    with asset.open('rb') as stream:
        header = stream.read(12)
    if len(header) != 12:
        raise ValueError('GLB header is incomplete')
    magic, version, declared_length = struct.unpack('<4sII', header)
    if magic != b'glTF' or version != 2 or declared_length != byte_count or byte_count < 20:
        raise ValueError('Expected a complete binary glTF 2.0 asset, not a renamed or truncated file')
    actual_sha = _file_sha(asset)
    if actual_sha != GAME_V7_SHA256:
        raise ValueError('Game-v7 asset SHA-256 mismatch; use the approved existing asset, never a fallback')
    return asset, byte_count, actual_sha


def prepare_helper(profile=DEFAULT_HELPER_PROFILE, asset_path=None, output_dir=None) -> PreparedHelper:
    """Prepare an explicitly selected helper, without changing any source file.

    game-v7: asset_path and a NOT-YET-EXISTING output_dir are required. The only
    generated file is runtime_helper.object_config.json, with absolute local
    render/collision references. Validation finishes before any directory write.
    blue-reference: use the packaged historical asset explicitly; asset_path
    must be None, output_dir is unused, and no file/directory is created.

    provenance is JSON-safe metadata, not model-facing information. It contains
    hashes and verification limits, never asset bytes, credentials or user paths.
    """
    if profile not in ('game-v7', 'blue-reference'):
        raise ValueError('Unknown helper profile; choose game-v7 or explicit blue-reference')
    if profile == 'blue-reference':
        if asset_path is not None:
            raise ValueError('blue-reference uses its packaged asset; do not supply --helper-asset')
        config = _BLUE_CONFIG_PATH.resolve(strict=True)
        asset = (config.parent/'light_blue_human.obj').resolve(strict=True)
        material = (config.parent/'light_blue_human.mtl').resolve(strict=True)
        if any(not p.is_file() for p in (config, asset, material)):
            raise ValueError('Packaged blue-reference asset/configuration is incomplete')
        return PreparedHelper(profile, config, .8, .32, 1.6, 'semantic_blue', {
            'profile': profile, 'asset_source': 'packaged_historical_blue_reference',
            'asset_sha256': _file_sha(asset), 'config_sha256': _file_sha(config),
            'material_sha256': _file_sha(material), 'derived_config_created': False,
            'new_model_created': False, 'source_asset_modified': False,
            'geometry_actual_validated': False,
            'proxy_status': 'conservative_previous_proxy_requires_actual_adapter_AABB_check',
        })
    # Fail closed before creating output when the model is missing or changed.
    asset, byte_count, asset_sha = _validate_game_asset(asset_path)
    if output_dir is None:
        raise ValueError('game-v7 requires a fresh output_dir for the derived runtime configuration')
    requested_output = Path(output_dir)
    if requested_output.exists() or requested_output.is_symlink():
        raise FileExistsError('Fresh helper output directory required; do not overwrite existing evidence')
    output = requested_output.resolve()
    config = output/'runtime_helper.object_config.json'
    settings = {
        'render_asset': str(asset), 'collision_asset': str(asset),
        'mass': 1.0, 'up': [0.0, 1.0, 0.0], 'front': [0.0, 0.0, -1.0],
        'COM': [0.0, 0.0, 0.0], 'scale': [1.0, 1.0, 1.0],
        'units_to_meters': 1.0, 'is_collidable': True,
    }
    encoded = json.dumps(settings, indent=2, allow_nan=False)+'\n'
    output.mkdir(parents=True, exist_ok=False)
    with config.open('x', encoding='utf-8') as stream:
        stream.write(encoded)
    return PreparedHelper(profile, config, 0.0, .32, 1.6, 'semantic_instance', {
        'profile': profile, 'asset_source': 'explicit_local_existing_game_character',
        'asset_sha256': asset_sha, 'asset_bytes': byte_count,
        'derived_config_sha256': hashlib.sha256(encoded.encode('utf-8')).hexdigest(),
        'derived_config_created': True, 'new_model_created': False,
        'source_asset_modified': False, 'source_configuration_modified': False,
        'geometry_actual_validated': False,
        'proxy_status': 'conservative_previous_proxy_requires_actual_adapter_AABB_check',
        'render_and_collision_share_source_asset': True,
    })


def replace_dummy_for_helper(dummy: DummySpec, prepared: PreparedHelper) -> DummySpec:
    """Preserve identity, physical foot and yaw; change only offset/body proxy."""
    if not isinstance(dummy, DummySpec) or not isinstance(prepared, PreparedHelper):
        raise ValueError('DummySpec and prepared helper are required')
    return replace(dummy, floor_offset_m=prepared.floor_offset_m,
                   radius_m=prepared.radius_m, height_m=prepared.height_m)


def orient_game_helper_to_anchor(dummy: DummySpec, anchor) -> DummySpec:
    """Face the fixed safe interaction anchor, never an episode goal/current user.

    The imported v7 character's visible front is local +Z. This changes yaw only,
    not placement, proxy or the separately validated source import-axis settings.
    """
    values = tuple(anchor)
    if len(values) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError('Finite fixed anchor XYZ required')
    dx, dz = values[0]-dummy.ground_position[0], values[2]-dummy.ground_position[2]
    if math.hypot(dx, dz) < 1e-6:
        raise ValueError('Safe exterior anchor cannot coincide with helper foot')
    return replace(dummy, yaw_rad=math.atan2(dx, dz))


__all__ = ['PreparedHelper', 'DEFAULT_HELPER_PROFILE', 'GAME_V7_SHA256',
           'prepare_helper', 'replace_dummy_for_helper', 'orient_game_helper_to_anchor']

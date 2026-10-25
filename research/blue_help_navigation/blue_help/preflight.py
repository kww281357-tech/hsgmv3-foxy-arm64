"""PRIVATE fixed-layout geometry preflight; never a policy input or navigation score.

All 20 frozen episodes are checked in their original order and at their original
starts. The SAME supplied layout is used throughout, irrespective of results.
A pass establishes necessary coarse placement/connectivity conditions only: the
reference NavMesh omits added bodies, and the conservative human-safe validator
can reject a shortest route without proving that no alternative route exists.
No model, navigation policy, action execution, layout search, or success/SPL
calculation belongs here. Simulator reset may render the initial observation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path

from .contracts import DummySpec, Episode
from .data import (DUMMIES_REL, EPISODES_REL, FROZEN_IDS, SCENE_REL, finite_vector,
                   load_dummies, load_episodes, validate_frozen_data)

PASS = 'passes_necessary_conditions_only'
FAIL = 'failed_or_unverified_necessary_conditions'


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _error(exc):
    return {'type': type(exc).__name__, 'message': str(exc)[:1000]}


def _layout(dummies):
    """Snapshot one valid, nonempty list; no per-episode layout callback is allowed."""
    result = tuple(dummies)
    if not result or any(not isinstance(d, DummySpec) for d in result):
        raise ValueError('A nonempty, fixed list of DummySpec objects is required')
    seen = set()
    for dummy in result:
        if (not isinstance(dummy.dummy_id, str) or not dummy.dummy_id.strip()
                or dummy.dummy_id in seen):
            raise ValueError('Dummy IDs must be nonempty and unique')
        seen.add(dummy.dummy_id)
        finite_vector(dummy.ground_position)
        if (not _number(dummy.yaw_rad) or not _number(dummy.radius_m) or dummy.radius_m <= 0
                or not _number(dummy.height_m) or dummy.height_m <= 0
                or not _number(dummy.floor_offset_m)
                or not 0 <= dummy.floor_offset_m <= dummy.height_m):
            raise ValueError('Invalid fixed dummy dimensions/pose')
    # Normalize vector containers, so a caller cannot change a mutable position.
    return tuple(replace(d, ground_position=finite_vector(d.ground_position)) for d in result)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _audit_snapshot(value):
    # Adapter-generated measurements only; never include arbitrary object state.
    keys = ('dummy_id', 'status', 'full_topology_validated', 'actual_aabb_radius_m',
            'actual_aabb_foot_relative_y', 'bullet_contact', 'radial_rays_clear', 'reason')
    if not isinstance(value, list):
        return []
    rows = [{k: row[k] for k in keys if k in row} for row in value if isinstance(row, dict)]
    try:
        return json.loads(json.dumps(rows, allow_nan=False))
    except (TypeError, ValueError):
        return []


def _placement_passed(rows, fixed):
    return (len(rows) == len(fixed)
            and all(isinstance(r.get('dummy_id'), str) for r in rows)
            and {r.get('dummy_id') for r in rows} == {d.dummy_id for d in fixed}
            and all(r.get('status') == 'pass_coarse_checks_only'
                    and r.get('bullet_contact') is False
                    and r.get('radial_rays_clear') is True for r in rows))


def _navmesh_info(env):
    result = {'loaded': None, 'agent_radius_m': None, 'agent_height_m': None}
    try:
        pf = env.sim.pathfinder
        result['loaded'] = bool(pf.is_loaded)
        settings = pf.nav_mesh_settings
        for key, attr in (('agent_radius_m', 'agent_radius'), ('agent_height_m', 'agent_height')):
            value = float(getattr(settings, attr))
            if math.isfinite(value) and value > 0:
                result[key] = value
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        result['error'] = _error(exc)
    return result


def _path_summary(path, start, goal, tolerance):
    if path is None:
        return {'status': 'unavailable_or_conservatively_rejected', 'exists': False,
                'length_m': None, 'point_count': 0}
    if not isinstance(path, (tuple, list)) or not 1 <= len(path) <= 100000:
        return {'status': 'invalid_path_result', 'exists': False, 'length_m': None,
                'reason': 'nonempty_bounded_point_sequence_required'}
    try:
        points = [finite_vector(p) for p in path]
        if math.dist(points[0], start) > tolerance or math.dist(points[-1], goal) > tolerance:
            raise ValueError('Path endpoints do not match unchanged episode endpoints')
        length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
        if not math.isfinite(length):
            raise ValueError('Nonfinite path length')
        return {'status': 'validated_by_environment', 'exists': True,
                'length_m': length, 'point_count': len(points)}
    except (TypeError, ValueError, OverflowError) as exc:
        return {'status': 'invalid_path_result', 'exists': False, 'length_m': None,
                'error': _error(exc)}


def preflight_episodes(env, episodes, dummies, seed=42, *, geodesic_factory=None):
    """Check all frozen episodes without selecting, moving, or dropping any.

    ``geodesic_factory(env) -> callable(start, goal) -> float | None`` is an
    explicit unit-test injection point. The default is evaluator-only
    ``habitat_metrics.make_geodesic_evaluator``. Results/geometry are PRIVATE:
    neither this report nor goal/path services may reach policy or lost detector.
    The caller owns/ closes ``env``. No ``env.step`` or policy method is called.
    """
    episodes = tuple(episodes)
    if type(seed) is not int or seed != 42:
        raise ValueError('The frozen preflight uses seed 42')
    if (any(not isinstance(e, Episode) for e in episodes)
            or [str(e.episode_id) for e in episodes] != [str(i) for i in FROZEN_IDS]):
        raise ValueError('Exactly the frozen 20 episodes in their original order are required')
    if len({e.scene_id for e in episodes}) != 1:
        raise ValueError('A fixed scene and layout are required for this preflight')
    fixed = _layout(dummies)
    layout_rows = [asdict(d) for d in fixed]
    fingerprint = _digest(layout_rows)
    if geodesic_factory is None:
        from .habitat_metrics import make_geodesic_evaluator
        geodesic_factory = make_geodesic_evaluator
    if not callable(geodesic_factory):
        raise ValueError('geodesic_factory must be callable')
    config = getattr(env, 'config', None)
    tolerance = getattr(config, 'endpoint_error_m', .05)
    if not _number(tolerance) or not 0 < tolerance <= .10:
        raise ValueError('A finite endpoint validation tolerance <= 0.10 m is required')
    configuration = {}
    for key in ('agent_radius_m', 'agent_height_m', 'collision_margin_m',
                'forward_step_m', 'turn_degrees', 'floor_tolerance_m'):
        value = getattr(config, key, None)
        configuration[key] = value if _number(value) else None
    records = []
    for episode in episodes:
        row = {'episode_id': str(episode.episode_id), 'scene_id': episode.scene_id,
               'seed': seed, 'fixed_layout_sha256': fingerprint,
               'status': FAIL, 'reset': {'status': 'not_attempted'},
               'original_start_pose_preserved': False,
               'placement_audit': [], 'coarse_placement_passed': False,
               'start_on_navmesh': None, 'goal_on_navmesh': None,
               'raw_geodesic': {'status': 'not_attempted', 'distance_m': None},
               'human_safe_path': {'status': 'not_attempted', 'exists': None, 'length_m': None},
               'failed_or_unverified_checks': [], 'errors': []}
        before_audit = getattr(env, 'placement_audit', None)
        try:
            # New list prevents simulator list mutation affecting later episodes;
            # normalized frozen DummySpecs retain the identical layout values.
            frame = env.reset(episode, list(fixed), seed=seed)
            row['reset']['status'] = 'passed'
            x, y, z, w = finite_vector(episode.start_rotation_xyzw, 4)
            expected_yaw = math.atan2(2*(x*z+y*w), 1-2*(x*x+y*y))
            heading_error = (float(frame.pose.yaw_rad)-expected_yaw+math.pi) % (2*math.pi)-math.pi
            row['original_start_pose_preserved'] = (frame.step == 0
                and math.dist(finite_vector(frame.pose.position), episode.start_position) <= 1e-4
                and math.isfinite(heading_error) and abs(heading_error) <= 1e-4)
        except Exception as exc:
            row['reset'] = {'status': 'failed', 'error': _error(exc)}
        current_audit = getattr(env, 'placement_audit', None)
        # If reset failed before clearing its previous audit, do not attribute
        # the prior episode's placement measurements to this episode.
        if row['reset']['status'] == 'passed' or current_audit is not before_audit:
            row['placement_audit'] = _audit_snapshot(current_audit)
        row['coarse_placement_passed'] = _placement_passed(row['placement_audit'], fixed)
        row['navmesh'] = _navmesh_info(env)
        dimensions = [(row['navmesh'][name], configuration[name])
                      for name in ('agent_radius_m', 'agent_height_m')]
        row['robot_navmesh_dimensions_compatible'] = (None
            if any(mesh is None or robot is None or robot <= 0 for mesh, robot in dimensions)
            else all(mesh+1e-6 >= robot for mesh, robot in dimensions))
        for name, point in (('start_on_navmesh', episode.start_position),
                            ('goal_on_navmesh', episode.goal_position)):
            try:
                row[name] = bool(env._navigable(finite_vector(point)))
            except Exception as exc:
                row['errors'].append({'check': name, **_error(exc)})
        if row['reset']['status'] == 'passed':
            try:
                distance = geodesic_factory(env)(episode.start_position, episode.goal_position)
                valid = _number(distance) and distance >= 0
                row['raw_geodesic'] = {'status': 'available' if valid else 'unavailable_or_invalid',
                                      'distance_m': float(distance) if valid else None,
                                      'source': 'reference_scene_navmesh_not_actor_aware'}
            except Exception as exc:
                row['raw_geodesic'] = {'status': 'query_error', 'distance_m': None, 'error': _error(exc)}
            try:
                path = env.find_safe_path(episode.start_position, episode.goal_position)
                row['human_safe_path'] = _path_summary(path, episode.start_position,
                                                       episode.goal_position, tolerance)
                check = getattr(env, 'last_path_check', {})
                if isinstance(check, dict):
                    diagnostic = {k: check[k] for k in ('status', 'reason', 'segment_index', 'substep',
                        'horizontal_error_m', 'height_delta_m', 'error_m', 'exception_type') if k in check}
                    row['human_safe_path']['environment_diagnostic'] = json.loads(
                        json.dumps(diagnostic, allow_nan=False))
                # Deliberately discard route points: only private scalar evidence
                # is retained; no executable privileged route is exported.
            except Exception as exc:
                row['human_safe_path'] = {'status': 'query_error', 'exists': None,
                                         'length_m': None, 'error': _error(exc)}
        requirements = {'reset': row['reset']['status'] == 'passed',
                        'unchanged_episode_start': row['original_start_pose_preserved'],
                        'coarse_placement': row['coarse_placement_passed'],
                        'loaded_navmesh': row['navmesh']['loaded'] is True,
                        'robot_navmesh_dimensions_compatible': row['robot_navmesh_dimensions_compatible'] is True,
                        'start_on_navmesh': row['start_on_navmesh'] is True,
                        'goal_on_navmesh': row['goal_on_navmesh'] is True,
                        'raw_geodesic_available': row['raw_geodesic']['status'] == 'available',
                        'human_safe_path_available': row['human_safe_path']['exists'] is True}
        row['failed_or_unverified_checks'] = [k for k, v in requirements.items() if not v]
        if not row['failed_or_unverified_checks']:
            row['status'] = PASS
        records.append(row)
    passed = sum(row['status'] == PASS for row in records)
    return {'schema_version': 1, 'report_kind': 'private_geometry_preflight_not_navigation_evaluation',
            'private_evaluator_only': True, 'status': PASS if passed == len(records) else FAIL,
            'seed': seed, 'frozen_episode_ids': list(FROZEN_IDS), 'configuration': configuration,
            'fixed_layout_sha256': fingerprint, 'fixed_layout': layout_rows,
            'fixed_layout_for_all_episodes': True, 'episode_start_override': False,
            'selection_rule': 'all_frozen_20_in_original_order_no_outcome_filtering_or_relocation',
            'episode_count': len(records), 'necessary_checks_passed_count': passed,
            'failed_or_unverified_count': len(records)-passed, 'skipped_count': 0,
            'full_topology_validated': False, 'navigation_executed': False,
            'model_calls': 0, 'benchmark_metrics_reported': False,
            'limitations': ['Coarse placement checks and one conservative route are not full topology proof.',
                'Reference geodesic does not include newly added dummy bodies.',
                'Rejected safe route does not prove all alternative routes are unavailable.',
                'Goal, path checks and this report are private validator evidence, never policy inputs.'],
            'records': records}


def _staged_layout(data_root, episodes, staging):
    """Apply one preselected, explicit relocation, never choose by episode result."""
    view = staging.get('staged_view') if isinstance(staging, dict) else None
    if not isinstance(view, dict) or view.get('placement_kind') != 'explicit_goal_independent_demo_relocation':
        raise ValueError('Staging report must explicitly identify goal-independent demo relocation')
    original = load_dummies(Path(data_root)/DUMMIES_REL, episodes[0])
    matches = [d for d in original if d.dummy_id == view.get('dummy_id')]
    if len(matches) != 1:
        raise ValueError('Staging report must identify exactly one original dummy')
    selected = replace(matches[0], ground_position=finite_vector(view.get('dummy_ground_position')))
    return [selected], {'kind': 'single_preselected_staged_dummy_shared_by_all_20',
                       'placement_kind': view['placement_kind'], 'dummy_id': selected.dummy_id,
                       'original_ground_position': matches[0].ground_position,
                       'relocated_ground_position': selected.ground_position,
                       'robot_staging_camera_pose_ignored': True,
                       'coarse_geometry_rechecked_at_each_original_episode_reset': True,
                       'other_original_dummies_not_instantiated': True}


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--navmesh', required=True, type=Path)
    parser.add_argument('--staging-report', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='Fresh output directory')
    parser.add_argument('--allow-sliding',action='store_true',help='Explicit alternative motion profile; do not mix with strict runs')
    parser.add_argument('--agent-radius', type=float, default=.10)
    parser.add_argument('--turn-degrees', type=float, default=15.0)
    args = parser.parse_args(argv)
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError('Use a fresh output directory; existing evidence is immutable')
    source_validation = validate_frozen_data(args.data_root)
    episodes = load_episodes(args.data_root/EPISODES_REL)
    staging = json.loads(args.staging_report.read_text())
    dummies, layout_provenance = _staged_layout(args.data_root, episodes, staging)
    from .habitat_adapter import AdapterConfig, create_environment
    config = AdapterConfig(agent_radius_m=args.agent_radius, turn_degrees=args.turn_degrees, allow_sliding=args.allow_sliding)
    provenance = {'allow_sliding':config.allow_sliding,'frozen_data_validation': source_validation, 'layout': layout_provenance,
                  'navmesh_sha256': _file_hash(args.navmesh),
                  'staging_report_sha256': _file_hash(args.staging_report)}
    project = Path(__file__).resolve().parent
    env = create_environment(args.data_root/SCENE_REL, args.navmesh,
        project/'assets/blue_dummy/light_blue_human.object_config.json', config=config)
    try:
        report = preflight_episodes(env, episodes, dummies)
        report['provenance'] = provenance
        report['runtime_backend'] = 'real_habitat_cpu_geometry_preflight'
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output/'geometry_preflight.private.json'
        with path.open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print(json.dumps({'status': report['status'], 'checked_episodes': report['episode_count'],
                          'failed_or_unverified_count': report['failed_or_unverified_count'],
                          'navigation_executed': False, 'private_report': str(path)}))
        return 0 if report['status'] == PASS else 2
    finally:
        env.close()


if __name__ == '__main__':
    raise SystemExit(main())

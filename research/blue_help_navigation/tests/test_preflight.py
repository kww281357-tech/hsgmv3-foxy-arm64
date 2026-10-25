"""Injected geometry facades only; these tests do not run Habitat or a model."""
from dataclasses import replace
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from blue_help.contracts import DummySpec, Episode, Pose, SensorFrame
from blue_help.data import FROZEN_IDS
from blue_help.preflight import FAIL, PASS, _staged_layout, main, preflight_episodes


def episodes():
    return [Episode(str(identity), 'fixture.glb', 'Walk ahead.',
                    (float(i), 0., 0.), (0., 0., 0., 1.), (float(i), 0., -3.))
            for i, identity in enumerate(FROZEN_IDS)]


DUMMIES = [DummySpec('fixed-blue', (50., 0., -1.), 0.)]


class FakeEnvironment:
    def __init__(self):
        self.config = NS(agent_radius_m=.10, agent_height_m=1.5, collision_margin_m=.03,
                         forward_step_m=.25, turn_degrees=15., floor_tolerance_m=.25,
                         endpoint_error_m=.05)
        self.sim = NS(pathfinder=NS(is_loaded=True,
            nav_mesh_settings=NS(agent_radius=.10, agent_height=1.5)))
        self.calls = []
        self.path_calls = []
        self.placement_audit = []
        self.reset_fail = set()
        self.path_unavailable = set()
        self.coarse_fail = set()
        self.not_navigable = set()
        self.path_override = None
        self.path_error = set()
        self.mutate_reset_list = False
        self.override_start = False
        self.closed = False
        self.last_path_check = {}

    def reset(self, episode, dummies, seed):
        self.current = episode.episode_id
        self.calls.append((episode, tuple(dummies), seed))
        if self.current in self.reset_fail:
            raise RuntimeError('Injected reset failure before clearing the last audit')
        self.placement_audit = [dict(dummy_id=d.dummy_id,
            status='rejected_or_unverified' if self.current in self.coarse_fail else 'pass_coarse_checks_only',
            full_topology_validated=False, bullet_contact=False, radial_rays_clear=True,
            actual_aabb_radius_m=.30, actual_aabb_foot_relative_y=[0., 1.6]) for d in dummies]
        if self.mutate_reset_list:
            dummies.clear()
        p = (100., 0., 100.) if self.override_start else episode.start_position
        return SensorFrame(None, None, Pose(p, 0.), 0)

    def _navigable(self, point):
        return tuple(point) not in self.not_navigable

    def find_safe_path(self, start, goal):
        self.path_calls.append((self.current, start, goal))
        if self.current in self.path_error:
            raise RuntimeError('Injected path service error')
        if self.current in self.path_unavailable:
            self.last_path_check = {'status': 'rejected', 'reason': 'no_sliding_step_mismatch'}
            return None
        self.last_path_check = {'status': 'passed', 'segment_count': 1}
        return self.path_override if self.path_override is not None else [start, goal]

    def step(self, action):
        raise AssertionError('Preflight must never execute navigation actions')

    def close(self):
        self.closed = True


class Factory:
    def __init__(self):
        self.calls = []
        self.invalid = {}
        self.fail = set()

    def __call__(self, env):
        identity = env.current
        def distance(start, goal):
            self.calls.append((identity, start, goal))
            if identity in self.fail:
                raise RuntimeError('Injected raw geodesic query error')
            return self.invalid.get(identity, math.dist(start, goal))
        return distance


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.env = FakeEnvironment()
        self.factory = Factory()

    def run_check(self, **kwargs):
        return preflight_episodes(self.env, episodes(), DUMMIES,
                                 geodesic_factory=self.factory, **kwargs)

    def test_all20_original_episodes_same_seed_layout_and_no_action(self):
        report = self.run_check()
        self.assertEqual(report['status'], PASS)
        self.assertEqual(report['necessary_checks_passed_count'], 20)
        self.assertEqual(report['skipped_count'], 0)
        self.assertEqual([r['episode_id'] for r in report['records']], list(map(str, FROZEN_IDS)))
        self.assertEqual([e for e, _, _ in self.env.calls], episodes())
        self.assertTrue(all(ds == tuple(DUMMIES) and seed == 42 for _, ds, seed in self.env.calls))
        self.assertEqual(len(self.factory.calls), 20)
        self.assertEqual(len(self.env.path_calls), 20)
        self.assertEqual(len({r['fixed_layout_sha256'] for r in report['records']}), 1)
        for record in report['records']:
            self.assertTrue(record['original_start_pose_preserved'])
            self.assertTrue(record['start_on_navmesh'])
            self.assertTrue(record['goal_on_navmesh'])
            self.assertEqual(record['raw_geodesic']['distance_m'], 3.)
            self.assertTrue(record['human_safe_path']['exists'])
        self.assertFalse(self.env.closed)  # caller owns environment lifetime

    def test_reset_failure_not_dropped_and_no_stale_prior_audit(self):
        self.env.reset_fail = {'41', '1755'}
        report = self.run_check()
        self.assertEqual(report['status'], FAIL)
        self.assertEqual(report['episode_count'], 20)
        self.assertEqual(report['failed_or_unverified_count'], 2)
        self.assertEqual(len(self.env.calls), 20)
        record = report['records'][1]
        self.assertEqual(record['placement_audit'], [])
        self.assertEqual(record['reset']['status'], 'failed')
        self.assertEqual(record['human_safe_path']['status'], 'not_attempted')
        self.assertEqual(record['raw_geodesic']['status'], 'not_attempted')
        self.assertEqual(report['records'][2]['status'], PASS)

    def test_failed_checks_do_not_relocate_remove_or_select_other_episode(self):
        self.env.coarse_fail = {'40'}
        self.env.path_unavailable = {'41'}
        self.env.not_navigable = {episodes()[2].goal_position}
        report = self.run_check()
        self.assertEqual(report['failed_or_unverified_count'], 3)
        self.assertFalse(report['records'][0]['coarse_placement_passed'])
        blocked = report['records'][1]['human_safe_path']
        self.assertFalse(blocked['exists'])
        self.assertEqual(blocked['environment_diagnostic']['reason'], 'no_sliding_step_mismatch')
        self.assertEqual(len(self.env.calls), 20)
        self.assertTrue(all(ds == tuple(DUMMIES) for _, ds, _ in self.env.calls))
        self.assertIn('all alternative routes', ' '.join(report['limitations']))

    def test_no_raw_paths_goals_or_instruction_in_report_and_no_scores(self):
        report = self.run_check()
        self.assertTrue(report['private_evaluator_only'])
        self.assertFalse(report['full_topology_validated'])
        self.assertFalse(report['navigation_executed'])
        self.assertEqual(report['model_calls'], 0)
        self.assertFalse(report['benchmark_metrics_reported'])
        for forbidden in ('goal_position', 'start_position', 'instruction', 'waypoints', 'success', 'spl'):
            self.assertNotIn('"'+forbidden+'":', json.dumps(report, allow_nan=False))

    def test_reference_geodesic_and_safe_path_are_distinct_checks(self):
        self.factory.invalid = {'40': None}
        self.env.path_unavailable = {'41'}
        rows = self.run_check()['records']
        self.assertEqual(rows[0]['raw_geodesic']['status'], 'unavailable_or_invalid')
        self.assertTrue(rows[0]['human_safe_path']['exists'])
        self.assertEqual(rows[1]['raw_geodesic']['status'], 'available')
        self.assertFalse(rows[1]['human_safe_path']['exists'])
        self.assertEqual([r['status'] for r in rows[:2]], [FAIL, FAIL])

    def test_nonfinite_invalid_geodesics_and_service_exceptions_preserved(self):
        self.factory.invalid = dict(zip(map(str, FROZEN_IDS), [math.nan, math.inf, -1, True, None]))
        self.factory.fail = {'249'}
        self.env.path_error = {'254'}
        report = self.run_check()
        self.assertEqual(report['failed_or_unverified_count'], 7)
        self.assertEqual(report['records'][5]['raw_geodesic']['status'], 'query_error')
        self.assertEqual(report['records'][6]['human_safe_path']['status'], 'query_error')
        json.dumps(report, allow_nan=False)

    def test_empty_nonfinite_or_wrong_endpoint_safe_path_never_passes(self):
        for path in ([], [(0., 0., 0.)], [(0., 0., 0.), (0., math.nan, -3.)],
                     [(100., 0., 100.), (100., 0., 99.)]):
            with self.subTest(path=path):
                self.env.path_override = path
                report = self.run_check()
                self.assertEqual(report['records'][0]['human_safe_path']['status'], 'invalid_path_result')
                self.assertEqual(report['necessary_checks_passed_count'], 0)

    def test_mutating_reset_list_cannot_change_subsequent_fixed_layout(self):
        self.env.mutate_reset_list = True
        report = self.run_check()
        self.assertEqual(report['status'], PASS)
        self.assertTrue(all(ds == tuple(DUMMIES) for _, ds, _ in self.env.calls))
        self.assertEqual(len(DUMMIES), 1)

    def test_changed_start_cannot_pass_as_original_episode_geometry(self):
        self.env.override_start = True
        report = self.run_check()
        self.assertEqual(report['necessary_checks_passed_count'], 0)
        self.assertIn('unchanged_episode_start', report['records'][0]['failed_or_unverified_checks'])

    def test_missing_or_undersized_navmesh_cannot_pass(self):
        self.env.sim.pathfinder.nav_mesh_settings.agent_radius = .09
        report = self.run_check()
        self.assertEqual(report['necessary_checks_passed_count'], 0)
        self.assertFalse(report['records'][0]['robot_navmesh_dimensions_compatible'])
        self.env.sim.pathfinder.is_loaded = False
        self.assertIn('loaded_navmesh', self.run_check()['records'][0]['failed_or_unverified_checks'])

    def test_missing_navmesh_helper_or_coarse_evidence_fail_closed(self):
        self.env._navigable = None
        report = self.run_check()
        self.assertEqual(report['necessary_checks_passed_count'], 0)
        self.assertEqual(len(report['records'][0]['errors']), 2)
        self.env = FakeEnvironment()
        original = self.env.reset
        def reset(*args, **kwargs):
            frame = original(*args, **kwargs)
            self.env.placement_audit = []
            return frame
        self.env.reset = reset
        self.assertEqual(self.run_check()['necessary_checks_passed_count'], 0)

    def test_fixed20_order_scene_seed_and_nonempty_layout_strict(self):
        for values in (episodes()[:19], episodes()[::-1], [None]*20,
                       [replace(e, scene_id='different.glb') if i == 0 else e for i, e in enumerate(episodes())]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                preflight_episodes(self.env, values, DUMMIES, geodesic_factory=self.factory)
        for seed in (0, True, 42., 7):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                self.run_check(seed=seed)
        for layout in ([], DUMMIES+DUMMIES, [replace(DUMMIES[0], radius_m=0)]):
            with self.assertRaises(ValueError):
                preflight_episodes(self.env, episodes(), layout, geodesic_factory=self.factory)
        self.assertEqual(self.env.calls, [])


class StagingAndCliTests(unittest.TestCase):
    def staging(self):
        return {'staged_view': {'dummy_id': 'fixed-blue', 'dummy_ground_position': [50., 0., -1.2],
            'camera_ground_position': [999., 99., 999.],
            'placement_kind': 'explicit_goal_independent_demo_relocation'}}

    def test_explicit_single_goal_independent_layout_ignores_staged_camera(self):
        with patch('blue_help.preflight.load_dummies', return_value=DUMMIES):
            layout, evidence = _staged_layout(Path('/unused'), episodes(), self.staging())
        self.assertEqual(layout, [replace(DUMMIES[0], ground_position=(50., 0., -1.2))])
        self.assertTrue(evidence['robot_staging_camera_pose_ignored'])
        self.assertTrue(evidence['other_original_dummies_not_instantiated'])
        self.assertEqual(DUMMIES[0].ground_position, (50., 0., -1.))

    def test_unknown_id_or_implicit_or_bad_relocation_is_rejected(self):
        bad = [None, {}, {'staged_view': {}}, self.staging(), self.staging()]
        bad[3]['staged_view']['dummy_id'] = 'other'
        bad[4]['staged_view']['dummy_ground_position'] = [0, math.nan, 0]
        with patch('blue_help.preflight.load_dummies', return_value=DUMMIES):
            for staging in bad:
                with self.subTest(staging=staging), self.assertRaises(ValueError):
                    _staged_layout(Path('/unused'), episodes(), staging)

    def test_cli_defaults_output_privacy_and_close_without_actual_simulator(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            staging_path = root/'staging.json'
            staging_path.write_text(json.dumps(self.staging()))
            navmesh = root/'mesh.navmesh'
            navmesh.write_bytes(b'fake mesh, not Habitat evidence')
            out = root/'out'
            env = FakeEnvironment()
            report = preflight_episodes(env, episodes(), DUMMIES, geodesic_factory=Factory())
            with patch('blue_help.preflight.validate_frozen_data', return_value={'status': 'fixture'}), \
                 patch('blue_help.preflight.load_episodes', return_value=episodes()), \
                 patch('blue_help.preflight.load_dummies', return_value=DUMMIES), \
                 patch('blue_help.preflight.preflight_episodes', return_value=report) as check, \
                 patch('blue_help.habitat_adapter.create_environment', return_value=env) as create, \
                 patch('builtins.print'):
                code = main(['--data-root', str(root), '--navmesh', str(navmesh),
                             '--staging-report', str(staging_path), '--output', str(out)])
            self.assertEqual(code, 0)
            self.assertEqual(create.call_args.kwargs['config'].agent_radius_m, .10)
            self.assertEqual(create.call_args.kwargs['config'].turn_degrees, 15.)
            self.assertEqual(check.call_args.args[1], episodes())
            self.assertEqual(check.call_args.args[2][0].ground_position, (50., 0., -1.2))
            self.assertTrue(env.closed)
            self.assertTrue((out/'geometry_preflight.private.json').is_file())
            with self.assertRaises(ValueError):
                main(['--data-root', str(root), '--navmesh', str(navmesh),
                      '--staging-report', str(staging_path), '--output', str(out)])


if __name__ == '__main__':
    unittest.main()

"""CPU-only runner/production timeout regressions: no simulator, API, or GPU."""
import ast
from contextlib import contextmanager
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('failed_retest', ROOT / 'scripts/run_failed_case_retest.py')
retest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retest)


def production_functions(names, namespace):
    tree = ast.parse((ROOT / 'src/run_experiments.py').read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name in {'R2REnv', 'Env'}:
            nodes.extend(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names
                         and (node.name != '_run_episode' or cls.name == 'R2REnv'))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'production_run_experiments', 'exec'), namespace)
    return {name: namespace[name] for name in names}


def result(ne=1.0, status='success', action='stop'):
    return {'metrics': {'distance_to_goal': ne, 'oracle_navigation_error': 0.0,
                        'traveled_distance': 4.0, 'finish_status': status, 'goal_reached': status == 'success'},
            'subtask': {'P1': {'record': [{'step': 3, 'action_type': action, 'motion_distance': 0.0}]}},
            'trace': []}


class ResultMetricsTests(unittest.TestCase):
    def setUp(self):
        self.meta = {'episode_id': 722, 'episode_ndx': 37, 'scene_id': 'scene', 'geodesic': 2.0}

    def test_zero_distance_is_success_not_missing(self):
        row = retest.summarize_result(result(0.0), self.meta)
        self.assertEqual((row['ne'], row['oracle_ne'], row['sr_2m'], row['spl_2m']), (0.0, 0.0, 1, 0.5))

    def test_near_goal_motion_is_only_oracle_success(self):
        row = retest.summarize_result(result(0.1, action='move_forward'), self.meta)
        self.assertEqual((row['sr_2m'], row['osr_2m'], row['distance_only_2m']), (0, 1, 1))

    def test_error_budget_abort_and_max_steps_never_receive_success_credit(self):
        for status in ('error', 'budget_exhausted', 'navigation_aborted', 'max_steps', 'user_interrupt'):
            with self.subTest(status=status):
                row = retest.summarize_result(result(0.1, status=status), self.meta)
                self.assertFalse(row['normal_stop'])
                self.assertEqual(row['sr_2m'], 0)
        data = result(0.1)
        data['metrics']['error_info'] = {'reason': 'artifact_error'}
        self.assertEqual(retest.summarize_result(data, self.meta)['sr_2m'], 0)

    def test_no_ledger_never_infers_stop_from_reported_success(self):
        data = result()
        del data['subtask']
        row = retest.summarize_result(data, self.meta)
        self.assertEqual(row['stop_evidence_source'], 'unavailable')
        self.assertEqual(row['sr_2m'], 0)

    def test_unrounded_evaluation_defines_boundary(self):
        data = result(2.0)
        data['evaluation'] = {'distance_to_goal': 1.9998}
        self.assertEqual(retest.summarize_result(data, self.meta)['sr_2m'], 1)
        data['evaluation']['distance_to_goal'] = 2.0
        self.assertEqual(retest.summarize_result(data, self.meta)['sr_2m'], 0)

    def test_real_execution_summary_overrides_old_subtask_stop(self):
        data = result()
        data['execution_summary'] = {'last_action_type': 'move_forward', 'stop_executed': False,
                                     'stop_action_count': 1}
        self.assertEqual(retest.summarize_result(data, self.meta)['sr_2m'], 0)

    def test_lifecycle_events_are_not_double_counted_as_verification(self):
        data = result()
        data['trace'] = [{'role': 'verifier', 'event': 'verification_started'},
                         {'role': 'verifier', 'event': 'verification_result'},
                         {'role': 'verifier_correction', 'executed_type': 'rollback'},
                         {'role': 'controller', 'event': 'wrong_path_rollback_started'},
                         {'role': 'controller', 'event': 'wrong_path_rollback_completed'},
                         {'role': 'controller', 'event': 'rollback_failed'}]
        row = retest.summarize_result(data, self.meta)
        self.assertEqual([row[key] for key in ('verifier_trace_events', 'verifier_results',
                         'verifier_corrections', 'rollback_requests', 'rollback_started',
                         'rollback_completed', 'rollback_failed')], [2, 1, 1, 1, 1, 1, 1])


class ProvenanceAndRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        repo = root / 'repo'
        (repo / 'src').mkdir(parents=True)
        (repo / 'config').mkdir()
        (repo / 'src/run_experiments.py').write_text('# offline fixture\n')
        config = repo / 'config/vlnce_test.yaml'
        config.write_text('navigation: {}\n')
        dataset = root / 'dataset.json'
        dataset.write_text(json.dumps({'episodes': [
            {'episode_id': '10', 'scene_id': 's/one', 'info': {'geodesic_distance': 2}},
            {'episode_id': '20', 'scene_id': 's/two', 'info': {'geodesic_distance': 3}}]}))
        baseline = root / 'baseline'
        baseline.mkdir()
        for eid in (10, 20):
            (baseline / f'episode_{eid}_results.json').write_text(json.dumps(result()))
        self.args = NS(repo_root=repo, config=config, dataset=dataset, baseline_root=baseline,
                       output_root=root / 'output', python=Path(sys.executable).resolve(),
                       model='qwen3.8-27b-q4', max_steps=150, max_seconds=3200, max_requests=96,
                       resume=False, wait_for_pid=[], env_file=None)
        self.env = {'OPENAI_BASE_URL': 'http://127.0.0.1:8000/v1', 'OPENAI_API_KEY': 'PRIVATE_TEST_TOKEN'}
        self.metas = retest.dataset_metadata(dataset, [20, 10])
        self.git = patch.object(retest.subprocess, 'run', return_value=NS(stdout='testcommit\n'))
        self.git.start()
        self.addCleanup(self.git.stop)

    def manifest(self):
        return retest.build_manifest(self.args, self.metas, self.env)

    def launch(self, command, **kwargs):
        index = int(command[command.index('--episode_ids') + 1])
        output = Path(command[command.index('--output_root') + 1]) / 'results_fixture'
        output.mkdir()
        (output / f'episode_{index}_results.json').write_text(json.dumps(result()))
        return NS(returncode=0)

    def test_ids_map_through_dataset_not_id_minus_one(self):
        self.assertEqual([m['episode_ndx'] for m in self.metas], [1, 0])
        with self.assertRaises(ValueError):
            retest.dataset_metadata(self.args.dataset, [30])
        with self.assertRaises(ValueError):
            retest.parse_ids('10,10')
        self.assertEqual(retest.parse_ids('20,10'), [20, 10])

    def test_duplicate_dataset_identity_is_rejected(self):
        data = json.loads(self.args.dataset.read_text())
        data['episodes'].append(data['episodes'][0])
        self.args.dataset.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            retest.dataset_metadata(self.args.dataset, [10])

    def test_manifest_is_stable_hashed_and_contains_no_credentials(self):
        first, second = self.manifest(), self.manifest()
        self.assertEqual(first['fingerprint'], second['fingerprint'])
        self.assertNotIn('PRIVATE_TEST_TOKEN', json.dumps(first))
        self.assertNotIn(self.env['OPENAI_BASE_URL'], json.dumps(first))
        self.args.config.write_text('navigation: {arrival_check_enabled: true}\n')
        self.assertNotEqual(first['fingerprint'], self.manifest()['fingerprint'])
        self.assertEqual(len(first['dataset']['sha256']), 64)
        self.assertEqual(len(first['baseline'][0]['sha256']), 64)

    def test_local_environment_enforces_limits_without_azure_override(self):
        with patch.dict(os.environ, {**self.env, 'AZURE_OPENAI_ENDPOINT': 'https://unused',
                                    'AZURE_OPENAI_API_KEY': 'unused',
                                    'HTTPS_PROXY': 'http://proxy.invalid:8080'}, clear=True):
            env = retest.configured_environment(self.args)
        self.assertNotIn('AZURE_OPENAI_ENDPOINT', env)
        self.assertNotIn('HTTPS_PROXY', env)
        self.assertEqual(env['NO_PROXY'], 'localhost,127.0.0.1,::1')
        self.assertEqual(self.manifest()['inference_mode'], 'local')
        self.assertEqual(env['HSGM_MAX_EPISODE_SECONDS'], '3200')
        self.assertEqual(env['HSGM_MAX_MODEL_REQUESTS'], '96')
        self.assertEqual(env['HSGM_ALLOW_SOFT_LIMIT_GOAL_CREDIT'], '0')

    def test_remote_api_requires_opt_in_and_keeps_endpoint_out_of_errors(self):
        with patch.dict(os.environ, {**self.env, 'OPENAI_BASE_URL': 'https://private.invalid/v1'}, clear=True):
            with self.assertRaisesRegex(ValueError, '--allow-remote-api') as caught:
                retest.configured_environment(self.args)
        self.assertNotIn('private.invalid', str(caught.exception))

    def test_remote_env_file_preserves_proxy_routing_and_removes_azure(self):
        self.args.allow_remote_api = True
        self.args.model = 'qwen3.8-flash'
        self.args.env_file = self.args.dataset.parent / 'api.env'
        self.args.env_file.write_text('OPENAI_BASE_URL=https://provider.invalid/compatible-mode/v1\n'
                                     'OPENAI_API_KEY="PRIVATE_REMOTE_TOKEN"\n'
                                     'AZURE_OPENAI_ENDPOINT=https://unused.invalid\n')
        proxies = {'https_proxy': 'http://proxy.invalid:8080', 'HTTP_PROXY': 'http://proxy.invalid:8081',
                   'ALL_PROXY': 'socks5://proxy.invalid:8082', 'NO_PROXY': 'internal.invalid',
                   'no_proxy': 'other.invalid'}
        with patch.dict(os.environ, {**self.env, **proxies, 'AZURE_OPENAI_API_KEY': 'unused'}, clear=True):
            env = retest.configured_environment(self.args)
        self.assertFalse(any(key.startswith('AZURE_OPENAI_') for key in env))
        self.assertEqual({key: env[key] for key in proxies}, proxies)
        self.assertEqual(env['OPENAI_API_KEY'], 'PRIVATE_REMOTE_TOKEN')
        self.assertEqual(env['OPENAI_MODEL'], 'qwen3.8-flash')
        self.assertEqual(env['HSGM_MAX_MODEL_REQUESTS'], '96')
        self.assertEqual(env['HSGM_MAX_EPISODE_SECONDS'], '3200')
        self.assertEqual(env['HSGM_ALLOW_SOFT_LIMIT_GOAL_CREDIT'], '0')
        manifest = retest.build_manifest(self.args, self.metas, env)
        self.assertEqual(manifest['inference_mode'], 'remote_https')
        self.assertEqual(manifest['model'], 'qwen3.8-flash')
        self.assertEqual(len(manifest['endpoint_sha256']), 64)
        for hidden in ('PRIVATE_REMOTE_TOKEN', 'provider.invalid', 'proxy.invalid', 'AZURE_OPENAI', str(self.args.env_file)):
            self.assertNotIn(hidden, json.dumps(manifest))
        launch = Mock(side_effect=self.launch)
        retest.run_sequential(self.args, self.metas, manifest, env, launch, Mock())
        self.assertEqual(launch.call_args.kwargs['env'], env)

    def test_remote_preflight_rejects_insecure_or_secret_bearing_urls(self):
        self.args.allow_remote_api = True
        for endpoint in ('http://provider.invalid/v1', 'https://' + 'private:token@provider.invalid/v1',
                         'https://@provider.invalid/v1', 'https://provider.invalid/v1?key=PRIVATE_TOKEN',
                         'https://provider.invalid/v1#PRIVATE_TOKEN', 'https://provider.invalid:bad/v1',
                         'https://provider.invalid:0/v1', 'https://bad host/v1', 'https://[bad/v1', ''):
            with self.subTest(endpoint=endpoint), patch.dict(os.environ, {**self.env, 'OPENAI_BASE_URL': endpoint}, clear=True):
                with self.assertRaises(ValueError) as caught:
                    retest.configured_environment(self.args)
                self.assertNotIn('PRIVATE_TOKEN', str(caught.exception))
                self.assertNotIn('private:token', str(caught.exception))
        with patch.dict(os.environ, {'OPENAI_BASE_URL': 'https://provider.invalid/v1', 'OPENAI_API_KEY': '  '}, clear=True):
            with self.assertRaisesRegex(ValueError, 'OPENAI_API_KEY'):
                retest.configured_environment(self.args)

    def test_remote_dry_run_exposes_mode_hash_model_without_launch_or_secrets(self):
        argv = ['--repo-root', str(self.args.repo_root), '--dataset', str(self.args.dataset),
                '--baseline-root', str(self.args.baseline_root), '--output-root', str(self.args.output_root),
                '--python', str(self.args.python), '--episode-ids', '20,10', '--allow-remote-api',
                '--model', 'qwen3.8-flash', '--dry-run']
        for flag in ([], ['--json-object-responses']):
            with self.subTest(flag=flag):
                output = io.StringIO()
                inherited = {**self.env, 'OPENAI_BASE_URL': 'https://provider.invalid/v1',
                             'HSGM_JSON_OBJECT_RESPONSES': '0' if flag else '1'}
                with patch.dict(os.environ, inherited, clear=True), patch.object(sys, 'stdout', output), \
                        patch.object(retest, 'run_sequential') as launch:
                    self.assertEqual(retest.main(argv + flag), 0)
                manifest = json.loads(output.getvalue())
                self.assertEqual((manifest['inference_mode'], manifest['model']), ('remote_https', 'qwen3.8-flash'))
                self.assertIs(manifest['json_object_responses'], bool(flag))
                self.assertNotIn('PRIVATE_TEST_TOKEN', output.getvalue())
                self.assertNotIn('provider.invalid', output.getvalue())
                launch.assert_not_called()
                self.assertFalse(self.args.output_root.exists())

    def test_json_object_option_overrides_env_file_and_reaches_worker(self):
        self.args.env_file = self.args.dataset.parent / 'format.env'
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                self.args.json_object_responses = enabled
                opposite = '0' if enabled else '1'
                self.args.env_file.write_text('HSGM_JSON_OBJECT_RESPONSES=' + opposite + '\n')
                with patch.dict(os.environ, {**self.env, 'HSGM_JSON_OBJECT_RESPONSES': opposite}, clear=True):
                    env = retest.configured_environment(self.args)
                self.assertEqual(env['HSGM_JSON_OBJECT_RESPONSES'], '1' if enabled else '0')
                self.args.output_root = self.args.dataset.parent / ('json-output-' + str(enabled))
                manifest = retest.build_manifest(self.args, self.metas, env)
                self.assertIs(manifest['json_object_responses'], enabled)
                launch = Mock(side_effect=self.launch)
                retest.run_sequential(self.args, self.metas, manifest, env, launch, Mock())
                self.assertEqual(launch.call_args.kwargs['env']['HSGM_JSON_OBJECT_RESPONSES'], '1' if enabled else '0')

    def test_json_object_mode_change_invalidates_resume(self):
        disabled = self.manifest()
        enabled = retest.build_manifest(self.args, self.metas, {**self.env, 'HSGM_JSON_OBJECT_RESPONSES': '1'})
        self.assertIs(disabled['json_object_responses'], False)
        self.assertIs(enabled['json_object_responses'], True)
        self.assertNotEqual(disabled['fingerprint'], enabled['fingerprint'])
        retest.prepare_output(self.args, disabled)
        self.args.resume = True
        with self.assertRaisesRegex(ValueError, 'provenance differs'):
            retest.prepare_output(self.args, enabled)

    def test_remote_mode_and_endpoint_change_invalidate_resume(self):
        local = self.manifest()
        self.args.allow_remote_api = True
        remote_env = {**self.env, 'OPENAI_BASE_URL': 'https://provider.invalid/v1'}
        remote = retest.build_manifest(self.args, self.metas, remote_env)
        self.assertNotEqual(local['fingerprint'], remote['fingerprint'])
        other = retest.build_manifest(self.args, self.metas, {**remote_env, 'OPENAI_BASE_URL': 'https://other.invalid/v1'})
        self.assertNotEqual(remote['fingerprint'], other['fingerprint'])
        retest.prepare_output(self.args, local)
        self.args.resume = True
        with self.assertRaisesRegex(ValueError, 'provenance differs'):
            retest.prepare_output(self.args, remote)

    def test_sequential_launch_and_resume_do_not_rerun_finished_episodes(self):
        launch = Mock(side_effect=self.launch)
        wait = Mock()
        manifest = self.manifest()
        state = retest.run_sequential(self.args, self.metas, manifest, self.env, launch, wait)
        self.assertEqual(list(state['episodes']), ['20', '10'])
        self.assertEqual([call.args[0][call.args[0].index('--episode_ids') + 1]
                          for call in launch.call_args_list], ['1', '0'])
        self.assertEqual(wait.call_count, 2)
        self.args.resume = True
        retest.run_sequential(self.args, self.metas, manifest, self.env, launch, wait)
        self.assertEqual(launch.call_count, 2)
        row = json.loads((self.args.output_root / 'summary.json').read_text())
        self.assertEqual(row['completed_count'], 2)

    def test_existing_output_or_interrupted_episode_cannot_be_overwritten(self):
        manifest = self.manifest()
        state = retest.prepare_output(self.args, manifest)
        with self.assertRaises(FileExistsError):
            retest.prepare_output(self.args, manifest)
        state['episodes']['20'] = {'status': 'started'}
        retest.atomic_json(self.args.output_root / 'state.json', state)
        self.args.resume = True
        with self.assertRaises(ValueError):
            retest.prepare_output(self.args, manifest)

    def test_resume_rejects_changed_code_and_tampered_results(self):
        manifest = self.manifest()
        retest.run_sequential(self.args, self.metas, manifest, self.env, self.launch, Mock())
        self.args.resume = True
        path = self.args.output_root / 'results_by_ep_id/episode_20_results.json'
        path.write_text('{}')
        with self.assertRaises(ValueError):
            retest.prepare_output(self.args, manifest)
        with self.assertRaises(ValueError):
            retest.prepare_output(self.args, {**manifest, 'fingerprint': 'changed'})

    def test_subprocess_failure_is_retained_without_success_or_retry(self):
        manifest = self.manifest()
        with self.assertRaises(ValueError):
            retest.run_sequential(self.args, self.metas, manifest, self.env,
                                  Mock(return_value=NS(returncode=1)), Mock())
        state = json.loads((self.args.output_root / 'state.json').read_text())
        self.assertEqual(state['episodes']['20']['status'], 'started')
        self.assertFalse((self.args.output_root / 'results_by_ep_id').exists())

    def test_worker_with_wrong_result_index_is_rejected(self):
        def bad_launch(command, **kwargs):
            raw = Path(command[command.index('--output_root') + 1])
            (raw / 'episode_999_results.json').write_text(json.dumps(result()))
            return NS(returncode=0)
        with self.assertRaises(ValueError):
            retest.run_sequential(self.args, self.metas, self.manifest(), self.env, bad_launch, Mock())

    def test_paused_exemption_identity_is_retained_across_resume_and_episodes(self):
        manifest = self.manifest()
        state = retest.prepare_output(self.args, manifest)
        state['allowed_paused_processes'] = {'100004': 'original-start'}
        retest.atomic_json(self.args.output_root / 'state.json', state)
        self.args.resume = True
        self.args.allow_paused_pid = [100004]
        wait = Mock()
        with patch.object(retest, 'read_processes', return_value={
                100004: {'start': 'recycled-start', 'state': 'T', 'argv': ['python', 'run_experiments.py']}}):
            retest.run_sequential(self.args, self.metas, manifest, self.env, self.launch, wait)
        self.assertEqual(wait.call_count, 2)
        for call in wait.call_args_list:
            self.assertEqual(call.kwargs['allow_paused'], {100004: 'original-start'})

    def test_parallel_mode_allows_concurrent_runners_on_different_roots(self):
        root1 = self.args.output_root / 'root1'
        root2 = self.args.output_root / 'root2'
        with retest.exclusive_runner(root1, allow_parallel=True):
            with retest.exclusive_runner(root2, allow_parallel=True):
                pass
            with self.assertRaises(ValueError):
                with retest.exclusive_runner(root1, allow_parallel=True):
                    pass

    def test_run_sequential_skips_wait_when_allow_parallel(self):
        manifest = self.manifest()
        self.args.allow_parallel = True
        wait = Mock()
        retest.run_sequential(self.args, self.metas, manifest, self.env, self.launch, wait)
        wait.assert_not_called()


class WaitTests(unittest.TestCase):
    def test_proc_reader_keeps_stop_state_and_start_identity(self):
        with tempfile.TemporaryDirectory() as root:
            process = Path(root) / '100004'
            process.mkdir()
            fields = ['T'] + ['0'] * 18 + ['123456']
            (process / 'stat').write_text('100004 (worker with spaces) ' + ' '.join(fields))
            (process / 'cmdline').write_bytes(b'python\0run_experiments.py\0')
            info = retest.read_processes(Path(root))[100004]
        self.assertEqual((info['state'], info['start']), ('T', '123456'))

    def test_explicit_matching_paused_pid_can_be_ignored(self):
        for stopped_state in ('T', 't'):
            reader = Mock(return_value={100004: {'start': 'original', 'state': stopped_state,
                                                'argv': ['python', 'run_experiments.py']}})
            sleeper = Mock()
            retest.wait_until_idle([100004], quiet_seconds=0, process_reader=reader,
                                   sleep=sleeper, allow_paused={100004: 'original'})
            sleeper.assert_not_called()

    def test_authorized_pid_running_again_blocks_until_it_finishes(self):
        elapsed = [0]
        def reader():
            if elapsed[0] >= 3:
                return {}
            return {100004: {'start': 'original', 'state': 'T' if elapsed[0] == 0 else 'R',
                            'argv': ['python', 'run_experiments.py']}}
        retest.wait_until_idle([], poll_seconds=1, quiet_seconds=2, process_reader=reader,
                               sleep=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
                               clock=lambda: elapsed[0], allow_paused={100004: 'original'})
        self.assertEqual(elapsed[0], 5)

    def test_recycled_paused_pid_and_unlisted_paused_pid_are_not_exempt(self):
        for pid, token in ((100004, 'recycled'), (100005, 'original')):
            elapsed = [0]
            def reader():
                return {pid: {'start': token, 'state': 'T', 'argv': ['python', 'run_experiments.py']}} if elapsed[0] < 2 else {}
            retest.wait_until_idle([], poll_seconds=1, quiet_seconds=0, process_reader=reader,
                                   sleep=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
                                   clock=lambda: elapsed[0], allow_paused={100004: 'original'})
            self.assertEqual(elapsed[0], 2)

    def test_waits_until_coordinator_and_workers_end_without_killing(self):
        elapsed = [0.0]
        def processes():
            result = {100000: {'start': '1', 'argv': ['python', 'llama_server.py']}}
            if elapsed[0] < 2:
                result[100001] = {'start': '2', 'argv': ['python', '/tmp/run_v3_important_120.py']}
            if elapsed[0] < 4:
                result[100002] = {'start': '3', 'argv': ['python', 'run_experiments.py']}
            return result
        retest.wait_until_idle([], 1, 2, processes, lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
                               lambda: elapsed[0])
        self.assertEqual(elapsed[0], 6)

    def test_explicit_pid_reuse_is_not_waited_forever(self):
        calls = [0]
        def processes():
            calls[0] += 1
            return {100003: {'start': 'old' if calls[0] == 1 else 'new', 'argv': ['unrelated']}}
        retest.wait_until_idle([100003], quiet_seconds=0, process_reader=processes, sleep=Mock())
        self.assertEqual(calls[0], 2)

    def test_shell_text_containing_script_name_is_not_a_worker(self):
        self.assertFalse(retest.experiment_process({'argv': ['bash', '-c', 'echo run_experiments.py']}))
        self.assertTrue(retest.experiment_process({'argv': ['python', '/repo/src/run_experiments.py']}))


class ProductionTerminationTests(unittest.TestCase):
    def harness(self, hard=False):
        class BudgetExceeded(BaseException):
            def __init__(self, reason):
                self.reason = reason
        class Aborted(BudgetExceeded):
            pass
        class Budget:
            def __init__(self, *args):
                pass
            def remaining(self):
                return 0
            @contextmanager
            def guard(self):
                if hard:
                    raise BudgetExceeded('model_requests')
                yield
        class Recorder(list):
            def __init__(self, *args):
                pass
            def close(self):
                pass
        ns = dict(os=os, logging=logging, EpisodeBudget=Budget, EpisodeBudgetExceeded=BudgetExceeded,
                  NavigationAborted=Aborted, EpisodeVideoRecorder=Recorder,
                  PolarAction=NS(stop=NS(type='stop'), null=NS(type='null')),
                  ModelServiceError=RuntimeError,
                  cv2=NS(cvtColor=lambda x, y: x, COLOR_BGR2RGB=1))
        functions = production_functions(['_budget_termination_metrics', '_populate_error_metrics', '_run_episode'], ns)
        state = NS(position=[0, 0, 0])
        agent = NS(last_action=NS(type='move_forward'), current_path=[], action_sequence=[],
                   second_stop=False, error=False, is_parse_fail=False, navigation_trace=[])
        agent.step = lambda step: ({'agent_state': state}, None, '', None)
        observed_actions, saved = [], []
        def metrics(agent_state, action, *args):
            observed_actions.append(action.type)
            return {'goal_reached': action.type == 'stop', 'finish_status': 'success' if action.type == 'stop' else 'max_steps',
                    'distance_to_goal': 0.2, 'oracle_navigation_error': 0.2, 'oracle_success': 1,
                    'spl': 1.0 if action.type == 'stop' else 0.0, 'stop_executed': action.type == 'stop'}
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        env = NS(env_cfg={'output_root': temp.name, 'name': 'test', 'max_steps': 3}, agent=agent,
                 all_episodes=[{}],
                 _initialize_episode=lambda idx: {'agent_state': state},
                 current_episode={'shortest_path': 1}, _create_visual_feedback=lambda *args: 'frame',
                 _calculate_metrics=metrics, _user_requested_quit=lambda: False,
                 _save_episode_results=lambda idx, frames, row, *args: saved.append(dict(row)))
        with patch.dict(os.environ, {'HSGM_ALLOW_SOFT_LIMIT_GOAL_CREDIT': '1'}):
            functions['_run_episode'](env, 0, False)
        return observed_actions, saved, functions

    def test_soft_wall_clock_at_goal_does_not_manufacture_stop(self):
        actions, saved, _ = self.harness()
        self.assertEqual(actions, ['move_forward', 'move_forward'])
        self.assertEqual(saved[-1]['finish_status'], 'budget_exhausted')
        self.assertFalse(saved[-1]['goal_reached'])
        self.assertFalse(saved[-1]['stop_executed'])
        self.assertEqual(saved[-1]['oracle_success'], 1)

    def test_request_budget_at_goal_does_not_manufacture_stop(self):
        actions, saved, _ = self.harness(hard=True)
        self.assertEqual(actions, ['move_forward'])
        self.assertEqual(saved[-1]['finish_status'], 'budget_exhausted')
        self.assertFalse(saved[-1]['goal_reached'])

    def test_error_cannot_preserve_stale_success(self):
        functions = production_functions(['_populate_error_metrics'], {})
        data = functions['_populate_error_metrics']({'goal_reached': True, 'finish_status': 'success', 'spl': 1}, {'reason': 'failed'})
        self.assertEqual(data['finish_status'], 'error')
        self.assertFalse(data['goal_reached'])
        self.assertEqual(data['spl'], 0)

    def test_saved_evaluation_keeps_precision_and_real_stop(self):
        functions = production_functions(['_save_episode_results'], dict(os=os, json=json, np=np,
                                                                        tempfile=tempfile, logging=logging,
                                                                        EpisodeVideoRecorder=type('Recorder', (), {})))
        with tempfile.TemporaryDirectory() as root:
            env = NS(env_cfg={'output_root': root, 'name': 'test'}, agent=NS(_execution_ledger=[
                {'step': 1, 'action_type': 'move_forward'}, {'step': 2, 'action_type': 'stop'}]))
            functions['_save_episode_results'](env, 0, [], {'distance_to_goal': 1.9998}, [], {})
            data = json.loads((Path(root) / 'results_test/episode_0_results.json').read_text())
        self.assertEqual(data['metrics']['distance_to_goal'], 2.0)
        self.assertEqual(data['evaluation']['distance_to_goal'], 1.9998)
        self.assertTrue(data['execution_summary']['stop_executed'])
        self.assertEqual(data['execution_summary']['stop_action_count'], 1)


class ProductionResultPersistenceTests(unittest.TestCase):
    """Exercise the production saver without importing Habitat or creating a simulator."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = NS(env_cfg={'output_root': self.temp.name, 'name': 'test'},
                      agent=NS(_execution_ledger=[]))
        self.path = Path(self.temp.name) / 'results_test/episode_0_results.json'
        self.save = production_functions(['_save_episode_results'], dict(
            os=os, json=json, np=np, tempfile=tempfile, logging=logging,
            EpisodeVideoRecorder=type('Recorder', (), {})))['_save_episode_results']

    def save_result(self, metrics=None, responses=None, subtasks=None):
        self.save(self.env, 0, [], metrics or {'distance_to_goal': 1.9998},
                  responses or [], subtasks)

    def test_nested_numpy_bools_remain_json_bools_in_all_result_sections(self):
        self.env.agent.navigation_trace = [{
            'endpoint_grounding': {'confirmed_changed_view': np.bool_(True),
                                   'geometry_ok': np.bool_(False)},
            'nested': (True, [False, np.array([True, False])]),
            'distance': np.float64(1.23456), 'count': np.int64(4),
        }]
        self.env.agent._has_teleport_correction = np.bool_(False)
        self.env.agent_trajectory = [np.array([1.23456, 2.34567]), np.bool_(True)]
        self.save_result(metrics={'distance_to_goal': np.float64(1.9998),
                                  'goal_reached': np.bool_(True), 'done': False},
                         responses=[{'valid': np.bool_(True), 'raw_score': np.float64(0.123456)}],
                         subtasks={'P1': {'completed': np.bool_(False)}})
        data = json.loads(self.path.read_text())
        self.assertIs(data['metrics']['goal_reached'], True)
        self.assertIs(data['metrics']['done'], False)
        self.assertIs(data['metrics']['has_teleport_correction'], False)
        self.assertIs(data['trace'][0]['endpoint_grounding']['confirmed_changed_view'], True)
        self.assertIs(data['trace'][0]['endpoint_grounding']['geometry_ok'], False)
        self.assertIs(data['trace'][0]['nested'][0], True)
        self.assertIs(data['trace'][0]['nested'][1][0], False)
        self.assertIs(data['trace'][0]['nested'][1][1][0], True)
        self.assertIs(data['trace'][0]['nested'][1][1][1], False)
        self.assertEqual(data['trace'][0]['distance'], 1.235)
        self.assertEqual(data['trace'][0]['count'], 4)
        self.assertEqual(data['agent_trajectory'][0], [1.235, 2.346])
        self.assertIs(data['agent_trajectory'][1], True)
        self.assertIs(data['vlm_responses'][0]['valid'], True)
        self.assertEqual(data['vlm_responses'][0]['raw_score'], 0.123456)
        self.assertIs(data['subtask']['P1']['completed'], False)
        self.assertEqual(data['evaluation']['distance_to_goal'], 1.9998)

    def test_execution_summary_preserves_formal_boolean_and_integer_types(self):
        for action in ('stop', 'move_forward'):
            with self.subTest(action=action):
                self.env.agent._execution_ledger = [
                    {'step': np.int64(1), 'action_type': 'stop'},
                    {'step': np.int64(2), 'action_type': action}]
                self.save_result()
                summary = json.loads(self.path.read_text())['execution_summary']
                self.assertIs(summary['stop_executed'], action == 'stop')
                self.assertIs(type(summary['last_action_step']), int)
                self.assertIs(type(summary['stop_action_count']), int)
                self.assertEqual(summary['stop_action_count'], 2 if action == 'stop' else 1)

    def test_unserializable_payload_preserves_existing_complete_result(self):
        self.save_result()
        original = self.path.read_bytes()
        with self.assertLogs(level='ERROR'), self.assertRaises(TypeError):
            self.save_result(responses=[{'unsupported': object()}])
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_first_serialization_failure_leaves_no_result(self):
        with self.assertLogs(level='ERROR'), self.assertRaises(TypeError):
            self.save_result(subtasks={'P1': {'unsupported': object()}})
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_replace_failure_preserves_destination_and_removes_temporary_file(self):
        for existing in (False, True):
            with self.subTest(existing=existing):
                if existing:
                    self.save_result()
                original = self.path.read_bytes() if existing else None
                with patch.object(os, 'replace', side_effect=OSError('replace denied')), \
                        self.assertLogs(level='ERROR'), self.assertRaisesRegex(OSError, 'replace denied'):
                    self.save_result(metrics={'distance_to_goal': 5.0})
                self.assertEqual(self.path.read_bytes() if self.path.exists() else None, original)
                self.assertEqual(list(self.path.parent.iterdir()), [self.path] if existing else [])

    def test_partial_temporary_write_failure_never_changes_destination(self):
        named_temporary_file = tempfile.NamedTemporaryFile

        class PartialWrite:
            def __init__(self, *args, **kwargs):
                self.file = named_temporary_file(*args, **kwargs)

            def __enter__(self):
                self.file.__enter__()
                return self

            def __exit__(self, *args):
                return self.file.__exit__(*args)

            def __getattr__(self, name):
                return getattr(self.file, name)

            def write(self, content):
                self.file.write(content[:20])
                raise OSError('disk full')

        for existing in (False, True):
            with self.subTest(existing=existing):
                if existing:
                    self.save_result()
                original = self.path.read_bytes() if existing else None
                with patch.object(tempfile, 'NamedTemporaryFile', side_effect=PartialWrite), \
                        self.assertLogs(level='ERROR'), self.assertRaisesRegex(OSError, 'disk full'):
                    self.save_result(metrics={'distance_to_goal': 5.0})
                self.assertEqual(self.path.read_bytes() if self.path.exists() else None, original)
                self.assertEqual(list(self.path.parent.iterdir()), [self.path] if existing else [])

    def test_success_replaces_only_after_complete_serialization_in_same_directory(self):
        self.save_result()
        original = self.path.read_bytes()
        replace = os.replace

        def verify_and_replace(source, destination):
            self.assertEqual(Path(source).parent, self.path.parent)
            self.assertEqual(Path(destination), self.path)
            self.assertEqual(self.path.read_bytes(), original)
            self.assertEqual(json.loads(Path(source).read_text())['evaluation']['distance_to_goal'], 5.1234)
            replace(source, destination)

        with patch.object(os, 'replace', side_effect=verify_and_replace) as replace_mock:
            self.save_result(metrics={'distance_to_goal': np.float64(5.1234)})
        replace_mock.assert_called_once()
        self.assertEqual(json.loads(self.path.read_text())['evaluation']['distance_to_goal'], 5.1234)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])


if __name__ == '__main__':
    unittest.main()


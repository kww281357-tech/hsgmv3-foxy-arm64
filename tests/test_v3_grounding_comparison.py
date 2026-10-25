"""CPU fixtures for pairing, explicit STOP semantics and mechanism observability."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from compare_grounding_retest import compare, failure_tags, mechanisms, main
from run_failed_case_retest import summarize_result


META = {'episode_id': 722, 'episode_ndx': 17, 'scene_id': 'fixture/scene', 'geodesic': 4.0}


def result(ne=5.0, oracle=1.0, status='fp', stop=True, modern=True):
    data = {'metrics': {'distance_to_goal': round(ne, 3), 'oracle_navigation_error': round(oracle, 3),
                        'traveled_distance': 6., 'finish_status': status, 'goal_reached': int(ne < 2)},
            'subtask': {'P1': {'record': [{'step': 3, 'action_type': 'stop' if stop else 'move_forward'}]}},
            'trace': []}
    if modern:
        data['evaluation'] = {'distance_to_goal': ne, 'oracle_navigation_error': oracle, 'traveled_distance': 6.}
        data['execution_summary'] = {'stop_executed': stop, 'last_action_type': 'stop' if stop else 'move_forward',
                                     'stop_action_count': int(stop)}
    return data


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(data) + '\n').encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


class ComparisonTests(unittest.TestCase):
    def test_timeout_at_goal_is_not_success_without_executed_stop(self):
        data = result(ne=.7, status='max_steps', stop=False)
        summary = summarize_result(data, META)
        self.assertEqual(summary['sr_2m'], 0)
        tags = failure_tags(summary)
        self.assertIn('max_steps_without_stop', tags)
        self.assertIn('oracle_hit_without_success_2m', tags)
        self.assertNotIn('visited_then_left_2m', tags)

    def test_left_goal_and_abnormal_stop_remain_separate_flags(self):
        summary = summarize_result(result(status='error'), META)
        tags = failure_tags(summary)
        self.assertIn('explicit_stop_failure', tags)
        self.assertIn('abnormal_termination_with_stop', tags)
        self.assertIn('visited_then_left_2m', tags)
        self.assertNotIn('false_stop_2m', tags)

    def test_unknown_stop_is_not_reported_as_proven_no_stop(self):
        data = result(modern=False)
        data['subtask'] = {}
        tags = failure_tags(summarize_result(data, META))
        self.assertIn('stop_evidence_unavailable', tags)
        self.assertNotIn('no_executed_terminal_stop', tags)

    def test_old_missing_instrumentation_is_unknown_not_zero(self):
        data = result(modern=False)
        stats = mechanisms(data, summarize_result(data, META), {'invocations': 2})
        self.assertIsNone(stats['verifier_results'])
        self.assertEqual(stats['legacy_logged_verifier_invocations']['invocations'], 2)
        self.assertIsNone(stats['rollback_completed'])
        self.assertIsNone(stats['grounding_vetoes'])

    def test_new_mechanism_counts_are_separate_and_do_not_invent_success(self):
        data = result()
        data['trace'] = [
            {'role': 'verifier_correction', 'executed_type': 'rollback'},
            {'role': 'controller', 'event': 'wrong_path_rollback_started'},
            {'role': 'controller', 'event': 'wrong_path_rollback_failed'},
            {'role': 'controller', 'event': 'endpoint_confirmation_turn'},
            {'role': 'verifier', 'event': 'verification_result', 'trigger': 'terminal_probe',
             'outcome': False, 'failure_type': 'not_reached',
             'endpoint_grounding': {'valid': True, 'accepted': False},
             'endpoint_confirmation_required': True, 'route_contradiction_gate': {'requested': True, 'accepted': False}},
            {'role': 'verifier', 'event': 'verification_result', 'outcome': False,
             'endpoint_grounding': {'valid': False, 'reason': 'stale'}, 'route_contradiction_gate': None},
            {'role': 'verifier', 'event': 'verification_result', 'outcome': False,
             'endpoint_grounding': {'valid': True, 'accepted': True, 'confirmed_changed_view': True},
             'route_contradiction_gate': None},
            {'role': 'verifier', 'event': 'verification_result', 'outcome': False, 'failure_type': 'not_reached',
             'endpoint_grounding': None, 'route_contradiction_gate': {'requested': False, 'accepted': False}},
        ]
        stats = mechanisms(data, summarize_result(data, META))
        self.assertEqual([stats[k] for k in ('rollback_requests', 'rollback_started', 'rollback_completed', 'rollback_failed')], [1, 1, 0, 1])
        self.assertEqual(stats['grounding_vetoes'], 2)
        self.assertEqual(stats['grounding_accepted'], 1)
        self.assertEqual(stats['confirmation_requests'], 1)
        self.assertEqual(stats['confirmation_turns'], 1)
        self.assertEqual(stats['confirmation_observations'], 1)
        self.assertEqual(stats['wrong_path_gate_requests'], 1)
        self.assertEqual(stats['wrong_path_gate_downgrades'], 1)
        self.assertEqual(stats['terminal_probe_accepted'], 0)

    def fixture(self, root):
        original, previous, current = root/'original', root/'previous', root/'current'
        original_path = original/'results_by_ep_id/episode_722_results.json'
        digest = write(original_path, result(modern=False))
        previous_hash = write(previous/'results_by_ep_id/episode_722_results.json', result(ne=1.123456, oracle=.9, status='success'))
        write(previous/'manifest.json', {'episodes': [META], 'baseline': [
            {'path': str(original_path), 'sha256': digest, 'summary': summarize_result(result(modern=False), META)}],
            'code': {'commit': 'previous-code'}, 'fingerprint': 'fixture-manifest'})
        write(previous/'state.json', {'episodes': {'722': {'status': 'finished', 'result_sha256': previous_hash}}})
        (original/'logs').mkdir()
        (original/'logs/worker.log').write_text('--- Episode 17, Step 1/150 ---\n[Verifier] Subtask verification call 1/4\n--- Episode 722, Step 1/150 ---\n[Verifier] Wrong episode verification call 1/4\n')
        return original, previous, current

    def test_saved_integer_flags_preserve_gate_and_confirmation_counts(self):
        data = result()
        data['trace'] = [
            {'role': 'verifier', 'event': 'verification_result', 'outcome': 0,
             'failure_type': 'not_reached',
             'endpoint_grounding': {'valid': 1, 'accepted': 0},
             'endpoint_confirmation_required': 1,
             'route_contradiction_gate': {'requested': 0, 'accepted': 0}},
            {'role': 'controller', 'event': 'endpoint_confirmation_turn'},
            {'role': 'verifier', 'event': 'verification_result', 'outcome': 0,
             'failure_type': 'not_reached',
             'endpoint_grounding': {'valid': 0},
             'route_contradiction_gate': {'requested': 1, 'accepted': 0}},
            {'role': 'verifier', 'event': 'verification_result', 'outcome': 1,
             'endpoint_grounding': {'valid': 1, 'accepted': 1, 'confirmed_changed_view': 1},
             'route_contradiction_gate': {'requested': 1, 'accepted': 1}},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'saved_result.json'
            write(path, data)
            saved = json.loads(path.read_text())
        stats = mechanisms(saved, summarize_result(saved, META))
        self.assertEqual(stats['grounding_vetoes'], 2)
        self.assertEqual(stats['grounding_accepted'], 1)
        self.assertEqual(stats['confirmation_requests'], 1)
        self.assertEqual(stats['confirmation_turns'], 1)
        self.assertEqual(stats['confirmation_observations'], 1)
        self.assertEqual(stats['wrong_path_gate_requests'], 2)
        self.assertEqual(stats['wrong_path_gate_accepted'], 1)
        self.assertEqual(stats['wrong_path_gate_downgrades'], 1)

    def test_missing_or_string_flags_are_not_counted_as_decisions(self):
        data = result()
        data['trace'] = [
            {'role': 'verifier', 'event': 'verification_result',
             'failure_type': 'not_reached', 'endpoint_confirmation_required': 'true',
             'endpoint_grounding': {'valid': None, 'accepted': 'false', 'confirmed_changed_view': '1'},
             'route_contradiction_gate': {'requested': '1', 'accepted': '0'}},
        ]
        stats = mechanisms(data, summarize_result(data, META))
        for key in ('grounding_vetoes', 'grounding_accepted', 'confirmation_requests',
                    'confirmation_observations', 'wrong_path_gate_requests',
                    'wrong_path_gate_accepted', 'wrong_path_gate_downgrades'):
            self.assertEqual(stats[key], 0, key)

    def test_official_id_pairing_and_pending_current_do_not_dilute_denominator(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = self.fixture(Path(temporary))
            report = compare(*roots, ids=[722])
            row = report['episodes'][0]
            self.assertEqual(row['episode_ndx'], 17)  # Deliberately not ID - 1.
            self.assertEqual(row['versions']['current']['status'], 'pending')
            self.assertEqual(report['paired']['previous_vs_current']['before']['count'], 0)
            self.assertIsNone(report['paired']['previous_vs_current']['after']['sr_2m'])
            self.assertEqual(row['versions']['previous']['summary']['ne'], 1.123456)
            calls = row['versions']['original']['mechanisms']['legacy_logged_verifier_invocations']
            self.assertEqual(calls['invocations'], 1)
            self.assertEqual(calls['source_lines'][0]['line'], 2)

    def test_result_hash_mismatch_is_not_silently_compared(self):
        with tempfile.TemporaryDirectory() as temporary:
            original, previous, current = self.fixture(Path(temporary))
            write(original/'results_by_ep_id/episode_722_results.json', result(ne=.2))
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                compare(original, previous, current, ids=[722])

    def test_current_manifest_index_mismatch_and_duplicate_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            original, previous, current = self.fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, 'unique'):
                compare(original, previous, current, ids=[722, 722])
            write(current/'manifest.json', {'episodes': [{**META, 'episode_ndx': 721}]})
            with self.assertRaisesRegex(ValueError, 'ID/index'):
                compare(original, previous, current, ids=[722])

    def test_three_versions_and_json_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, previous, current = self.fixture(root)
            digest = write(current/'results_by_ep_id/episode_722_results.json', result(status='max_steps', stop=False))
            write(current/'state.json', {'episodes': {'722': {'status': 'finished', 'result_sha256': digest}}})
            output = root/'reports/comparison.json'
            main(['--original-root', str(original), '--previous-root', str(previous), '--current-root', str(current),
                  '--episode-ids', '722', '--output', str(output)])
            report = json.loads(output.read_text())
            paired = report['paired']['previous_vs_current']
            self.assertEqual(paired['episode_ids'], [722])
            self.assertEqual(paired['delta']['sr_2m'], -1)
            self.assertEqual(report['cohorts']['success_controls']['current']['count'], 0)


if __name__ == '__main__':
    unittest.main()


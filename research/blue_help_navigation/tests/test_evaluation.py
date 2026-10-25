"""Pure synthetic evaluator fixtures; no actual episode/provider run is scheduled."""
from dataclasses import replace
import json
import math
import unittest

from blue_help.data import FROZEN_IDS
from blue_help.evaluation import (EpisodeMetrics, build_paired_manifest, evaluate_episode,
                                  make_result, summarize_paired)


def metrics(**changes):
    args = dict(stopped=True, final_geodesic_m=1., shortest_path_m=5.,
                travelled_path_m=10., question_count=0, provenance='mock')
    args.update(changes)
    return evaluate_episode(**args)


def row(identity=40, condition='no_ask', provider='test-model', **changes):
    return make_result(identity, condition, metrics(**changes), provider_id=provider)


class ManifestTests(unittest.TestCase):
    def test_exact_frozen_twenty_by_two_seed42_plan(self):
        manifest = build_paired_manifest()
        self.assertEqual(manifest['frozen_episode_ids'], FROZEN_IDS)
        self.assertEqual(manifest['total_runs'], 40)
        self.assertEqual(manifest['seed'], 42)
        self.assertEqual(manifest['concurrency'], 1)
        self.assertEqual(len({r['run_id'] for r in manifest['runs']}), 40)
        for i, identity in enumerate(FROZEN_IDS):
            pair = manifest['runs'][2*i:2*i+2]
            self.assertEqual([r['condition'] for r in pair], ['no_ask', 'max3_ask'])
            self.assertEqual([r['episode_id'] for r in pair], [str(identity)] * 2)
            self.assertEqual([r['seed'] for r in pair], [42, 42])
            self.assertEqual([r['max_questions'] for r in pair], [0, 3])
            self.assertTrue(all(r['status'] == 'pending_not_executed' for r in pair))
        self.assertEqual(manifest['status'], 'plan_only_not_executed')

    def test_batches_count_two_to_four_episodes_and_never_split_pairs(self):
        for size in (2, 3, 4):
            manifest = build_paired_manifest(size, 'selected-model')
            batches = manifest['batches']
            with self.subTest(size=size):
                self.assertTrue(all(2 <= batch['size'] <= 4 and batch['concurrency'] == 1 for batch in batches))
                self.assertEqual(manifest['batch_size_unit'], 'episodes')
                self.assertEqual(sum(batch['size'] for batch in batches), 20)
                self.assertEqual(sum(batch['run_count'] for batch in batches), 40)
                self.assertEqual([identity for batch in batches for identity in batch['episode_ids']],
                                 [str(identity) for identity in FROZEN_IDS])
                for batch in batches:
                    self.assertEqual(batch['batch_size_unit'], 'episodes')
                    self.assertEqual(batch['episode_count'], batch['size'])
                    self.assertEqual(batch['run_count'], 2 * batch['episode_count'])
                    self.assertTrue(4 <= batch['run_count'] <= 8)
                    self.assertEqual(batch['run_ids'], [f'{identity}-seed42-{condition}'
                        for identity in batch['episode_ids'] for condition in ('no_ask', 'max3_ask')])
                self.assertEqual([run for batch in batches for run in batch['run_ids']],
                                 [run['run_id'] for run in manifest['runs']])
                self.assertTrue(all(run['provider_id'] == 'selected-model' for run in manifest['runs']))
        self.assertEqual(len(build_paired_manifest()['batches']), 10)
        self.assertTrue(all(batch['run_count'] == 4 for batch in build_paired_manifest()['batches']))
        self.assertEqual([batch['size'] for batch in build_paired_manifest(3)['batches']], [3]*6 + [2])

    def test_eighty_is_two_providers_four_groups_and_not_automatically_scheduled(self):
        manifest = build_paired_manifest()
        pending = manifest['four_group_comparison']
        self.assertEqual(pending['provider_count'], 2)
        self.assertEqual(pending['condition_count'], 2)
        self.assertEqual(pending['group_count'], 4)
        self.assertEqual(pending['planned_runs'], 80)
        self.assertFalse(pending['automatic_execution'])
        self.assertEqual(pending['status'], 'pending_user_interpretation_not_scheduled')
        self.assertEqual(len(manifest['runs']), 40)

    def test_invalid_batch_and_provider_values_rejected(self):
        for value in (0, 1, 5, 2.0, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_paired_manifest(value)
        with self.assertRaises(ValueError):
            build_paired_manifest(provider_id='')


class MetricTests(unittest.TestCase):
    def test_geodesic_success_requires_explicit_stop_at_threshold(self):
        self.assertTrue(metrics(final_geodesic_m=3.).success)
        self.assertFalse(metrics(final_geodesic_m=3.00001).success)
        self.assertFalse(metrics(stopped=False, final_geodesic_m=0.).success)
        self.assertFalse(metrics(stopped=False, final_geodesic_m=0.).spl)
        self.assertFalse(metrics(final_geodesic_m=2., success_threshold_m=1.).success)

    def test_spl_standard_ratio_and_bounded_success_factor(self):
        self.assertEqual(metrics().spl, .5)
        self.assertEqual(metrics(travelled_path_m=2.).spl, 1.)
        self.assertEqual(metrics(final_geodesic_m=9.).spl, 0.)

    def test_spl_zero_length_edges(self):
        self.assertEqual(metrics(shortest_path_m=0., travelled_path_m=0., final_geodesic_m=0.).spl, 1.)
        self.assertEqual(metrics(shortest_path_m=0., travelled_path_m=2., final_geodesic_m=0.).spl, 0.)
        self.assertEqual(metrics(shortest_path_m=0., travelled_path_m=0., final_geodesic_m=0., stopped=False).spl, 0.)

    def test_missing_or_unreachable_geodesics_are_invalid_not_scored_failures(self):
        for name in ('final_geodesic_m', 'shortest_path_m'):
            for value in (None, math.inf, math.nan, -1, True):
                with self.subTest(name=name, value=value):
                    result = metrics(**{name: value})
                    self.assertEqual(result.status, 'invalid_runtime')
                    self.assertIsNone(result.success)
                    self.assertIsNone(result.spl)
                    self.assertIn('path', result.invalid_reason)
                    json.dumps(result.to_dict(), allow_nan=False)

    def test_bad_movement_question_count_and_stop_record_are_invalid(self):
        for changes in ({'travelled_path_m': -1}, {'travelled_path_m': math.nan},
                        {'question_count': True}, {'question_count': -1}, {'question_count': 1.5},
                        {'stopped': 'stop'}):
            with self.subTest(changes=changes):
                result = metrics(**changes)
                self.assertEqual(result.status, 'invalid_runtime')
                self.assertIsNone(result.success)

    def test_failed_runtime_does_not_become_normal_navigation_failure(self):
        result = metrics(runtime_valid=False, error='runtime unavailable')
        self.assertEqual(result.status, 'invalid_runtime')
        self.assertIsNone(result.success)
        self.assertIn('runtime_not_validated_or_failed', result.invalid_reason)

    def test_safe_path_proxy_never_masquerades_as_standard_geodesic(self):
        for source in ('safe_path_proxy', 'unavailable'):
            result = metrics(distance_source=source)
            self.assertEqual(result.status, 'invalid_runtime')
            self.assertIsNone(result.success)
            self.assertIsNone(result.spl)
            self.assertEqual(result.distance_source, source)

    def test_invalid_configuration_raises(self):
        for changes in ({'provenance': 'unknown'}, {'success_threshold_m': 0},
                        {'success_threshold_m': math.nan}, {'runtime_valid': 1},
                        {'distance_source': 'euclidean'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                metrics(**changes)

    def test_metric_dataclass_cannot_serialize_inconsistent_success_or_spl(self):
        good = metrics()
        with self.assertRaises(ValueError):
            replace(good, success=False)
        with self.assertRaises(ValueError):
            replace(good, spl=1)
        with self.assertRaises(ValueError):
            replace(good, status='invalid_runtime', invalid_reason='invalid but scored')

    def test_serialized_metrics_round_trip_validation(self):
        good = metrics()
        decoded = json.loads(json.dumps(good.to_dict(), allow_nan=False))
        self.assertEqual(EpisodeMetrics(**decoded), good)
        self.assertEqual(make_result(40, 'no_ask', decoded)['metrics'], good.to_dict())

    def test_budget_violation_remains_visible_as_invalid_result(self):
        no_ask = make_result(40, 'no_ask', metrics(question_count=1))
        over_three = make_result(40, 'max3_ask', metrics(question_count=4))
        for result in (no_ask, over_three):
            self.assertEqual(result['metrics']['status'], 'invalid_runtime')
            self.assertIn('question_budget_violation', result['metrics']['invalid_reason'])
            self.assertIsNone(result['metrics']['success'])
        self.assertEqual(make_result(40, 'max3_ask', metrics(question_count=3))['metrics']['status'], 'valid')

    def test_seed_and_condition_contracts(self):
        with self.assertRaises(ValueError):
            make_result(40, 'no_ask', metrics(), seed=7)
        with self.assertRaises(ValueError):
            make_result(40, 'ask_unlimited', metrics())


class AggregationTests(unittest.TestCase):
    def test_common_valid_pairs_not_different_episode_subsets(self):
        records = [row(40, final_geodesic_m=4.), row(40, 'max3_ask', question_count=2),
                   row(41), row(41, 'max3_ask', final_geodesic_m=None),
                   row(52, stopped=False)]
        group = summarize_paired(records)['groups'][0]
        self.assertEqual(group['expected_pairs'], 20)
        self.assertEqual(group['both_arms_present'], 2)
        self.assertEqual(group['common_valid_pairs'], 1)
        self.assertEqual(group['common_valid_coverage'], .05)
        self.assertEqual(group['common_valid_episode_ids'], ['40'])
        self.assertEqual(group['conditions']['no_ask']['valid'], 3)
        self.assertEqual(group['conditions']['max3_ask']['valid'], 1)
        self.assertEqual(group['conditions']['max3_ask']['invalid'], 1)
        self.assertEqual(group['conditions']['max3_ask']['missing'], 18)
        self.assertAlmostEqual(group['conditions']['no_ask']['all_valid_means']['success_rate'], 1/3)
        self.assertEqual(group['common_valid_means']['no_ask']['success_rate'], 0.)
        self.assertEqual(group['common_valid_means']['max3_ask']['success_rate'], 1.)
        self.assertEqual(group['ask_minus_no_ask']['success_rate'], 1.)
        self.assertEqual(group['ask_minus_no_ask']['mean_question_count'], 2.)

    def test_all_twenty_valid_pairs_report_full_coverage(self):
        records = []
        for identity in FROZEN_IDS:
            records.extend([row(identity), row(identity, 'max3_ask', question_count=1,
                                                travelled_path_m=5.)])
        group = summarize_paired(records)['groups'][0]
        self.assertEqual(group['common_valid_pairs'], 20)
        self.assertEqual(group['common_valid_coverage'], 1.)
        self.assertEqual(group['submitted_runs'], 40)
        self.assertEqual(group['ask_minus_no_ask']['mean_spl'], .5)
        self.assertEqual(group['ask_minus_no_ask']['mean_path_length_m'], -5.)

    def test_invalid_runtime_never_enters_failure_denominator(self):
        group = summarize_paired([row(40, final_geodesic_m=None),
                                  row(40, 'max3_ask', runtime_valid=False)])['groups'][0]
        self.assertEqual(group['common_valid_pairs'], 0)
        self.assertIsNone(group['ask_minus_no_ask']['success_rate'])
        self.assertEqual(group['conditions']['no_ask']['invalid'], 1)
        self.assertIsNone(group['conditions']['no_ask']['all_valid_means']['success_rate'])

    def test_real_replay_mock_never_pooled(self):
        records = []
        for provenance in ('real', 'replay', 'mock'):
            records += [row(40, provenance=provenance), row(40, 'max3_ask', provenance=provenance)]
        summary = summarize_paired(records)
        self.assertTrue(summary['mixed_provenance'])
        self.assertEqual(summary['provenances'], ['mock', 'real', 'replay'])
        self.assertEqual(len(summary['groups']), 3)
        self.assertIsNone(summary['pooled_metrics'])
        self.assertTrue(all(group['common_valid_pairs'] == 1 for group in summary['groups']))
        mock = next(group for group in summary['groups'] if group['provenance'] == 'mock')
        self.assertEqual(mock['result_kind'], 'synthetic_mock_not_r2rce_results')

    def test_different_providers_cannot_form_a_fake_pair(self):
        summary = summarize_paired([row(40, provider='A'), row(40, 'max3_ask', provider='B')])
        self.assertEqual(len(summary['groups']), 2)
        self.assertTrue(all(group['common_valid_pairs'] == 0 for group in summary['groups']))

    def test_duplicate_records_rejected_not_double_counted(self):
        record = row()
        with self.assertRaises(ValueError):
            summarize_paired([record, record])

    def test_unexpected_episode_and_metadata_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            summarize_paired([row(999999)])
        bad = row()
        bad['provenance'] = 'real'
        with self.assertRaises(ValueError):
            summarize_paired([bad])
        bad = row()
        bad['seed'] = 0
        with self.assertRaises(ValueError):
            summarize_paired([bad])

    def test_different_thresholds_or_start_geodesics_do_not_form_fair_pairs(self):
        with self.assertRaises(ValueError):
            summarize_paired([row(), row(40, 'max3_ask', success_threshold_m=2.)])
        with self.assertRaises(ValueError):
            summarize_paired([row(), row(40, 'max3_ask', shortest_path_m=6.)])

    def test_empty_summary_reports_no_fabricated_scores(self):
        summary = summarize_paired([])
        self.assertEqual(summary['groups'], [])
        self.assertEqual(summary['provenances'], [])
        self.assertIsNone(summary['pooled_metrics'])
        self.assertEqual(len(summary['expected_episode_ids']), 20)

    def test_output_is_json_serializable_without_nonfinite_values(self):
        summary = summarize_paired([row(40, final_geodesic_m=math.inf), row(40, 'max3_ask')])
        json.dumps(summary, allow_nan=False)
        json.dumps(build_paired_manifest(), allow_nan=False)


if __name__ == '__main__':
    unittest.main()

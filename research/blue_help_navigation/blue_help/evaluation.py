"""Pure evaluator-side planning and metrics. Never feed these values to policy.

Geodesic-to-goal, initial shortest-path distance and success are evaluation-only
privileged information. No function in this module calls a model, simulator,
network, scheduler or provider. A manifest is a plan, not evidence of execution.
Real simulator, recorded replay and synthetic mock results remain separate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from numbers import Real
from typing import Iterable

from .data import FROZEN_IDS

CONDITIONS = ('no_ask', 'max3_ask')
PROVENANCES = frozenset({'real', 'replay', 'mock'})
FIXED_SEED = 42


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def _count(value):
    return value if type(value) is int and value >= 0 else None


def _provider(value):
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError('provider_id must be a nonempty string or None')
    return value


def _identity(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError('episode_id must be a nonempty string or integer')
    return str(value)


def build_paired_manifest(batch_size: int = 2, provider_id: str | None = None) -> dict:
    """Plan frozen 20 x 2 conditions, seed 42, one serial run at a time.

    Batch size counts EPISODES, each containing both paired conditions: 2..4
    episodes means 4..8 serial arm runs per batch. Pairs are never split between
    batches. Default two episodes gives ten batches of four runs. This controls
    review/staging checkpoints, NOT concurrency. The
    separate two-provider/four-group/80-run comparison remains pending and unscheduled.
    """
    if type(batch_size) is not int or not 2 <= batch_size <= 4:
        raise ValueError('batch_size must be an integer from 2 to 4')
    provider_id = _provider(provider_id)
    ids = tuple(FROZEN_IDS)
    if len(ids) != 20 or len(set(ids)) != 20:
        raise ValueError('the frozen 20-episode manifest is malformed')
    runs = []
    for episode_id in ids:
        for condition in CONDITIONS:
            runs.append(dict(run_index=len(runs), run_id=f'{episode_id}-seed42-{condition}',
                             episode_id=str(episode_id), seed=FIXED_SEED, condition=condition,
                             max_questions=0 if condition == 'no_ask' else 3,
                             provider_id=provider_id, status='pending_not_executed'))
    batches = []
    offset = 0
    while offset < len(ids):
        size = min(batch_size, len(ids) - offset)
        if len(ids) - offset - size == 1:
            size -= 1
        batch_episodes = ids[offset:offset + size]
        batch_runs = runs[2 * offset:2 * (offset + size)]
        batches.append(dict(batch_index=len(batches),
                            episode_ids=[str(identity) for identity in batch_episodes],
                            run_ids=[r['run_id'] for r in batch_runs],
                            size=len(batch_episodes), batch_size_unit='episodes',
                            episode_count=len(batch_episodes), run_count=len(batch_runs),
                            concurrency=1, status='pending_not_executed'))
        offset += size
    return dict(schema_version=1, status='plan_only_not_executed', frozen_episode_ids=list(ids),
                seed=FIXED_SEED, conditions=list(CONDITIONS), provider_id=provider_id,
                total_runs=len(runs), concurrency=1, requested_batch_size=batch_size, batch_size_unit='episodes',
                paired_controls='Same episode, seed, scene and dummy placements in both conditions; only ask budget differs.',
                batches=batches, runs=runs,
                four_group_comparison=dict(status='pending_user_interpretation_not_scheduled', provider_count=2,
                    condition_count=2, group_count=4, episodes_per_group=20, planned_runs=80,
                    selected_providers=None, automatic_execution=False,
                    note='80 means TWO providers x TWO conditions x20 episodes, not four providers. This separate expansion is pending and is not part of the scheduled single-provider40 manifest.'))


def _spl(success, shortest, travelled):
    if not success:
        return 0.0
    denominator = max(shortest, travelled)
    # Already at the start/goal with no movement: perfect successful path. If
    # shortest is zero but movement occurred, conventional ratio is zero.
    return 1.0 if denominator == 0 else shortest / denominator


@dataclass(frozen=True)
class EpisodeMetrics:
    status: str
    provenance: str
    stopped: bool | None
    success: bool | None
    spl: float | None
    path_length_m: float | None
    shortest_path_m: float | None
    final_geodesic_m: float | None
    question_count: int | None
    success_threshold_m: float = 3.0
    invalid_reason: str | None = None
    distance_source: str = 'geodesic'

    def __post_init__(self):
        if self.status not in ('valid', 'invalid_runtime') or self.provenance not in PROVENANCES:
            raise ValueError('unsupported metric status or provenance')
        if self.distance_source not in ('geodesic', 'safe_path_proxy', 'unavailable'):
            raise ValueError('unsupported distance source')
        threshold = _number(self.success_threshold_m)
        if threshold is None or threshold <= 0:
            raise ValueError('success threshold must be positive and finite')
        if self.stopped is not None and type(self.stopped) is not bool:
            raise ValueError('stopped must be boolean or unavailable')
        for name in ('path_length_m', 'shortest_path_m', 'final_geodesic_m'):
            value = getattr(self, name)
            if value is not None and _number(value) is None:
                raise ValueError(f'{name} must be finite, nonnegative or unavailable')
        if self.question_count is not None and _count(self.question_count) is None:
            raise ValueError('question_count must be a nonnegative integer or unavailable')
        if self.status == 'invalid_runtime':
            if self.success is not None or self.spl is not None or not isinstance(self.invalid_reason, str) or not self.invalid_reason:
                raise ValueError('invalid runtime metrics need a reason and null success/SPL')
            return
        if self.distance_source != 'geodesic':
            raise ValueError('valid standard metrics require actual geodesic distances')
        if (self.stopped is None or self.path_length_m is None or self.shortest_path_m is None
                or self.final_geodesic_m is None or self.question_count is None):
            raise ValueError('valid metrics require every measured input')
        expected_success = self.stopped and self.final_geodesic_m <= threshold
        if type(self.success) is not bool or self.success != expected_success:
            raise ValueError('success must match explicit STOP and final geodesic threshold')
        value = _number(self.spl)
        if value is None or not math.isclose(value, _spl(expected_success, self.shortest_path_m, self.path_length_m),
                                              rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('SPL does not match measured inputs')
        if self.invalid_reason is not None:
            raise ValueError('valid metrics cannot include invalid_reason')

    def to_dict(self):
        return asdict(self)


def evaluate_episode(*, stopped: bool, final_geodesic_m, shortest_path_m,
                     travelled_path_m, question_count: int, provenance: str,
                     success_threshold_m: float = 3.0, runtime_valid: bool = True,
                     error: str | None = None, distance_source: str = 'geodesic') -> EpisodeMetrics:
    """Compute evaluator-only success/SPL or an explicit invalid-runtime record.

    Success requires an ACTUAL explicit STOP primitive, not timeout, a model
    completion claim or Euclidean proximity. Missing/unreachable geodesic queries
    (None/inf/NaN), bad movement totals, or runtime errors are not scored as
    ordinary failures and cannot quietly lower or inflate a success rate.
    A human-safe-path proxy is not standard geodesic: callers must label it
    distance_source='safe_path_proxy', which leaves standard metrics invalid.
    """
    if distance_source not in ('geodesic', 'safe_path_proxy', 'unavailable'):
        raise ValueError('unsupported distance_source')
    if provenance not in PROVENANCES:
        raise ValueError('provenance must be real, replay, or mock')
    threshold = _number(success_threshold_m)
    if threshold is None or threshold <= 0:
        raise ValueError('success_threshold_m must be positive and finite')
    if type(runtime_valid) is not bool:
        raise ValueError('runtime_valid must be boolean')
    if error is not None and not isinstance(error, str):
        raise ValueError('error must be text or None')
    final = _number(final_geodesic_m)
    shortest = _number(shortest_path_m)
    travelled = _number(travelled_path_m)
    questions = _count(question_count)
    stop = stopped if type(stopped) is bool else None
    reasons = []
    if distance_source != 'geodesic':
        reasons.append('nonstandard_or_unavailable_geodesic_source')
    if not runtime_valid:
        reasons.append('runtime_not_validated_or_failed')
    if error is not None:
        reasons.append('runtime_error: ' + error[:1000])
    if stop is None:
        reasons.append('missing_or_invalid_explicit_stop_record')
    if final is None:
        reasons.append('missing_or_invalid_final_geodesic_path')
    if shortest is None:
        reasons.append('missing_or_invalid_initial_shortest_path')
    if travelled is None:
        reasons.append('missing_or_invalid_measured_path_length')
    if questions is None:
        reasons.append('missing_or_invalid_delivered_question_count')
    if reasons:
        return EpisodeMetrics('invalid_runtime', provenance, stop, None, None, travelled,
                              shortest, final, questions, threshold, '; '.join(reasons), distance_source)
    success = stop and final <= threshold
    return EpisodeMetrics('valid', provenance, stop, success, _spl(success, shortest, travelled),
                          travelled, shortest, final, questions, threshold)


def _metrics(value):
    if isinstance(value, EpisodeMetrics):
        return value
    if not isinstance(value, dict):
        raise ValueError('metrics must be EpisodeMetrics or its serialized dictionary')
    return EpisodeMetrics(**value)


def make_result(episode_id, condition: str, metrics: EpisodeMetrics, seed: int = FIXED_SEED,
                provider_id: str | None = None) -> dict:
    """Bind metrics to a comparison arm; preserve invalid runs, not just successes."""
    identity = _identity(episode_id)
    if condition not in CONDITIONS:
        raise ValueError('condition must be no_ask or max3_ask')
    if type(seed) is not int or seed != FIXED_SEED:
        raise ValueError('this frozen comparison requires seed 42')
    provider_id, metrics = _provider(provider_id), _metrics(metrics)
    limit = 0 if condition == 'no_ask' else 3
    if metrics.question_count is not None and metrics.question_count > limit:
        previous = metrics.invalid_reason + '; ' if metrics.invalid_reason else ''
        metrics = replace(metrics, status='invalid_runtime', success=None, spl=None,
                          invalid_reason=previous + 'delivered_question_budget_violation')
    return dict(episode_id=identity, condition=condition, seed=seed, provider_id=provider_id,
                provenance=metrics.provenance, metrics=metrics.to_dict())


def _means(rows):
    if not rows:
        return dict(count=0, success_rate=None, mean_spl=None, mean_path_length_m=None,
                    mean_question_count=None)
    n = len(rows)
    return dict(count=n,
                success_rate=sum(int(row.success) for row in rows) / n,
                mean_spl=sum(row.spl for row in rows) / n,
                mean_path_length_m=sum(row.path_length_m for row in rows) / n,
                mean_question_count=sum(row.question_count for row in rows) / n)


def summarize_paired(results: Iterable[dict], expected_episode_ids=FROZEN_IDS) -> dict:
    """Aggregate separately by provider AND provenance, using common valid pairs.

    Each condition also exposes its all-valid (possibly unpaired) coverage.
    Comparative means/deltas use ONLY episodes valid in both arms, never two
    different episode subsets. Missing/invalid records remain visible in counts.
    Duplicate keys, unexpected episodes and mixed seeds are contract errors,
    not silently dropped rows. No real/replay/mock pooled metric is produced.
    """
    expected = tuple(_identity(value) for value in expected_episode_ids)
    if not expected or len(expected) != len(set(expected)):
        raise ValueError('expected episode IDs must be nonempty and unique')
    expected_set = set(expected)
    groups = {}
    for record in results:
        if not isinstance(record, dict):
            raise ValueError('result records must be dictionaries')
        row = make_result(record['episode_id'], record['condition'], record['metrics'],
                          record['seed'], record.get('provider_id'))
        if record.get('provenance') != row['provenance']:
            raise ValueError('record and metric provenance disagree')
        if row['episode_id'] not in expected_set:
            raise ValueError('result episode is outside the fixed comparison')
        key = (row['provider_id'], row['provenance'])
        group = groups.setdefault(key, {})
        run_key = (row['episode_id'], row['condition'])
        if run_key in group:
            raise ValueError('duplicate episode/condition/provider/provenance result')
        group[run_key] = _metrics(row['metrics'])
    summaries = []
    for (provider_id, provenance), group in sorted(groups.items(), key=lambda item: (item[0][0] or '', item[0][1])):
        thresholds = {row.success_threshold_m for row in group.values()}
        if len(thresholds) != 1:
            raise ValueError('paired comparison mixes success thresholds')
        for identity in expected:
            pair = [group.get((identity, condition)) for condition in CONDITIONS]
            if all(row is not None and row.status == 'valid' for row in pair) and not math.isclose(
                    pair[0].shortest_path_m, pair[1].shortest_path_m, rel_tol=1e-9, abs_tol=1e-6):
                raise ValueError('same-episode pair has inconsistent initial shortest-path distance')
        conditions = {}
        for condition in CONDITIONS:
            rows = [group[(identity, condition)] for identity in expected if (identity, condition) in group]
            valid = [row for row in rows if row.status == 'valid']
            conditions[condition] = dict(expected=len(expected), submitted=len(rows), valid=len(valid),
                                         invalid=len(rows)-len(valid), missing=len(expected)-len(rows),
                                         all_valid_means=_means(valid))
        both = [identity for identity in expected if all((identity, condition) in group for condition in CONDITIONS)]
        common = [identity for identity in both if all(group[(identity, condition)].status == 'valid' for condition in CONDITIONS)]
        paired_means = {condition: _means([group[(identity, condition)] for identity in common])
                        for condition in CONDITIONS}
        deltas = {name: (paired_means['max3_ask'][name] - paired_means['no_ask'][name]) if common else None
                  for name in ('success_rate', 'mean_spl', 'mean_path_length_m', 'mean_question_count')}
        summaries.append(dict(provider_id=provider_id, provenance=provenance,
                              result_kind={'real': 'real_simulator_results', 'replay': 'recorded_replay_not_live_evaluation',
                                           'mock': 'synthetic_mock_not_r2rce_results'}[provenance],
                              seed=FIXED_SEED, success_threshold_m=next(iter(thresholds)), expected_pairs=len(expected), submitted_runs=len(group),
                              both_arms_present=len(both), common_valid_pairs=len(common),
                              common_valid_coverage=len(common)/len(expected),
                              common_valid_episode_ids=common, conditions=conditions,
                              common_valid_means=paired_means, ask_minus_no_ask= deltas))
    kinds = sorted({provenance for _, provenance in groups})
    return dict(schema_version=1, expected_episode_ids=list(expected), seed=FIXED_SEED,
                groups=summaries, provenances=kinds, mixed_provenance=len(kinds)>1,
                pooled_metrics=None,
                interpretation='Comparative means use common-valid episode pairs within one provider and one provenance. No cross-provenance or cross-provider pooling.')

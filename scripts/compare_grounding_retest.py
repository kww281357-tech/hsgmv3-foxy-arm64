#!/usr/bin/env python3
"""Offline three-version diagnostic comparison; never imports an agent or model.

Pass the original results directory, the complete previous retest root (with its
manifest), and optionally the new retest root. Missing new results remain pending.
All distances come from saved evaluator results, used only for this report.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_failed_case_retest import CONTROL_IDS, FAILURE_IDS, parse_ids, summarize_result

METRICS = ('sr_2m', 'sr_3m', 'osr_2m', 'osr_3m', 'spl_2m', 'spl_3m',
           'ne', 'oracle_ne', 'traveled_distance')
VERSIONS = ('original', 'previous', 'current')


def read_snapshot(path):
    path = Path(path)
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
        raise ValueError(f'Input changed during snapshot: {path}')
    return json.loads(raw), {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
                             'mtime_ns': after.st_mtime_ns, 'bytes': len(raw)}


def results_directory(root):
    root = Path(root)
    return root / 'results_by_ep_id' if (root / 'results_by_ep_id').is_dir() else root


def failure_tags(summary):
    tags = []
    stop_known = summary['stop_evidence_source'] != 'unavailable'
    if summary['sr_2m']:
        tags.append('success_2m')
    else:
        if summary['stop_executed']:
            tags.append('explicit_stop_failure')
            tags.append('false_stop_2m' if summary['normal_stop'] else 'abnormal_termination_with_stop')
        elif stop_known:
            tags.append('no_executed_terminal_stop')
            if summary['finish_status'] == 'max_steps':
                tags.append('max_steps_without_stop')
        else:
            tags.append('stop_evidence_unavailable')
    for radius in (2, 3):
        if summary[f'osr_{radius}m'] and not summary[f'sr_{radius}m']:
            tags.append(f'oracle_hit_without_success_{radius}m')
        if summary[f'osr_{radius}m'] and not summary[f'distance_only_{radius}m']:
            tags.append(f'visited_then_left_{radius}m')
    return tags


def saved_flag(value, expected=True):
    """Result serialization stores bool flags as either JSON booleans or 0/1."""
    return value is expected or (type(value) is int and value == int(expected))


def mechanisms(data, summary, legacy_calls=None):
    trace = [e for e in data.get('trace', data.get('navigation_trace', [])) if isinstance(e, dict)]
    verifiers = [e for e in trace if e.get('role') == 'verifier' and e.get('event') in
                 {'verification_result', 'route_verification_result', 'verify_result'}]
    lifecycle = [e for e in trace if e.get('event') in {
        f'{prefix}{phase}' for prefix in ('wrong_path_rollback_', 'rollback_')
        for phase in ('started', 'completed', 'failed')}]
    instrumented = bool(verifiers or lifecycle or 'execution_summary' in data)
    grounding_instrumented = any('endpoint_grounding' in e for e in verifiers)
    grounding = [e['endpoint_grounding'] for e in verifiers if isinstance(e.get('endpoint_grounding'), dict)]
    contradiction_instrumented = any('route_contradiction_gate' in e for e in verifiers)
    def downgraded(event):
        gate = event.get('route_contradiction_gate')
        if not isinstance(gate, dict):
            return event.get('event') in {'wrong_path_gate_downgraded', 'route_contradiction_downgraded'}
        if saved_flag(gate.get('requested'), False):
            return False
        rejected = (saved_flag(gate.get('downgraded')) or saved_flag(gate.get('allowed'), False)
                    or saved_flag(gate.get('accepted'), False) or saved_flag(gate.get('rollback_allowed'), False))
        return rejected and event.get('failure_type') == 'not_reached'
    return {
        'verifier_results': len(verifiers) if instrumented else None,
        'legacy_logged_verifier_invocations': legacy_calls,
        'terminal_probe_results': sum(e.get('trigger') == 'terminal_probe' for e in verifiers) if instrumented else None,
        'terminal_probe_accepted': sum(e.get('trigger') == 'terminal_probe' and e.get('outcome') == 1
                                       and e.get('completion_evidence_reliable') == 1 for e in verifiers) if instrumented else None,
        'completion_proposals': summary['completion_proposals'],
        'verifier_corrections': summary['verifier_corrections'],
        'rollback_requests': summary['rollback_requests'],
        **{f'rollback_{p}': summary[f'rollback_{p}'] if instrumented else None for p in ('started', 'completed', 'failed')},
        'rollback_lifecycle_events': lifecycle,
        'grounding_checks': len(grounding) if grounding_instrumented else None,
        'grounding_vetoes': sum(saved_flag(g.get('valid'), False) or saved_flag(g.get('accepted'), False) for g in grounding) if grounding_instrumented else None,
        'grounding_accepted': sum(saved_flag(g.get('accepted')) for g in grounding) if grounding_instrumented else None,
        'confirmation_requests': sum(saved_flag(e.get('endpoint_confirmation_required')) for e in verifiers) if grounding_instrumented else None,
        'confirmation_turns': sum(e.get('event') == 'endpoint_confirmation_turn' for e in trace) if grounding_instrumented else None,
        'confirmation_observations': sum(saved_flag(g.get('confirmed_changed_view')) for g in grounding) if grounding_instrumented else None,
        'wrong_path_gate_requests': sum(isinstance(e.get('route_contradiction_gate'), dict)
                                        and saved_flag(e['route_contradiction_gate'].get('requested')) for e in verifiers) if contradiction_instrumented else None,
        'wrong_path_gate_accepted': sum(isinstance(e.get('route_contradiction_gate'), dict)
                                        and saved_flag(e['route_contradiction_gate'].get('accepted')) for e in verifiers) if contradiction_instrumented else None,
        'wrong_path_gate_downgrades': sum(downgraded(e) for e in trace) if contradiction_instrumented else None,
        'decision_events': [{k: e[k] for k in ('event', 'step', 'trigger', 'outcome', 'failure_type', 'reason',
                            'endpoint_grounding', 'endpoint_reference_grounding', 'endpoint_confirmation_required',
                            'route_contradiction_gate') if k in e} for e in verifiers],
    }


def legacy_verifier_calls(root, metas):
    root = Path(root)
    log_root = (root.parent if root.name == 'results_by_ep_id' else root) / 'logs'
    by_index = {m['episode_ndx']: m['episode_id'] for m in metas}
    calls = {m['episode_id']: {'step_headers': 0, 'invocations': 0, 'source_lines': []} for m in metas}
    for path in sorted(log_root.glob('*.log')):
        current = None
        for number, line in enumerate(path.read_text(errors='replace').splitlines(), 1):
            marker = re.search(r'--- Episode (\d+), Step', line)
            if marker:
                current = int(marker.group(1))
            if current not in by_index:
                continue
            row = calls[by_index[current]]
            row['step_headers'] += int(marker is not None)
            if '[Verifier]' in line and 'verification call ' in line:
                row['invocations'] += 1
                row['source_lines'].append({'path': str(path.resolve()), 'line': number})
    return {eid: row if row['step_headers'] else None for eid, row in calls.items()}


def aggregate(rows, version, ids):
    chosen = [r['versions'][version] for r in rows if r['episode_id'] in ids
              and r['versions'][version]['status'] == 'completed']
    return {'count': len(chosen), **{
        key: statistics.mean(v['summary'][key] for v in chosen)
        if chosen and all(v['summary'][key] is not None for v in chosen) else None for key in METRICS},
        'failure_tag_counts': dict(Counter(tag for v in chosen for tag in v['failure_tags']))}


def compare(original_root, previous_root, current_root=None, ids=None):
    ids = list(ids if ids is not None else FAILURE_IDS + CONTROL_IDS)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Official episode IDs must be nonempty and unique.')
    previous_root = Path(previous_root)
    manifest, manifest_source = read_snapshot(previous_root / 'manifest.json')
    metas = {m['episode_id']: m for m in manifest['episodes']}
    if any(eid not in metas for eid in ids):
        raise ValueError('Requested official ID is absent from the previous manifest.')
    frozen_baseline = {b['summary']['episode_id']: b for b in manifest['baseline']}
    roots = {'original': Path(original_root), 'previous': previous_root,
             'current': Path(current_root) if current_root is not None else None}
    states, manifests = {}, {'previous': manifest}
    sources = [manifest_source]
    for version in ('previous', 'current'):
        root = roots[version]
        if root is None:
            continue
        if version == 'current' and (root / 'manifest.json').exists():
            manifests[version], source = read_snapshot(root / 'manifest.json')
            sources.append(source)
            for item in manifests[version]['episodes']:
                if item['episode_id'] in ids and item != metas[item['episode_id']]:
                    raise ValueError('Current and previous manifests disagree on official ID/index mapping.')
        if (root / 'state.json').exists():
            states[version], source = read_snapshot(root / 'state.json')
            sources.append(source)
    log_calls = legacy_verifier_calls(roots['original'], [metas[eid] for eid in ids])
    rows = []
    for eid in ids:
        row = {'episode_id': eid, 'episode_ndx': metas[eid]['episode_ndx'],
               'cohort': 'success_control' if eid in CONTROL_IDS else 'directed_failure', 'versions': {}}
        for version, root in roots.items():
            state = states.get(version, {}).get('episodes', {}).get(str(eid), {})
            path = results_directory(root) / f'episode_{eid}_results.json' if root is not None else None
            if path is None or not path.exists():
                if version != 'current':
                    raise ValueError(f'Missing required {version} result for official episode {eid}.')
                row['versions'][version] = {'status': 'pending', 'runner_state': state.get('status', 'not_started'),
                                             'expected_path': str(path) if path else None}
                continue
            data, source = read_snapshot(path)
            expected_hash = frozen_baseline[eid]['sha256'] if version == 'original' else state.get('result_sha256')
            if expected_hash and source['sha256'] != expected_hash:
                raise ValueError(f'{version} episode {eid} hash differs from its frozen manifest/state.')
            summary = summarize_result(data, metas[eid])
            explicit_index = data.get('episode_ndx', data.get('metrics', {}).get('episode_ndx'))
            if explicit_index is not None and int(explicit_index) != metas[eid]['episode_ndx']:
                raise ValueError(f'{version} episode {eid} contains the wrong dataset index.')
            row['versions'][version] = {'status': 'completed', 'source': source,
                'hash_verified': expected_hash is not None, 'summary': summary,
                'failure_tags': failure_tags(summary),
                'mechanisms': mechanisms(data, summary, log_calls[eid] if version == 'original' else None)}
        rows.append(row)
    pairs = {}
    for left, right in (('original', 'previous'), ('original', 'current'), ('previous', 'current')):
        shared = [r['episode_id'] for r in rows if all(r['versions'][v]['status'] == 'completed' for v in (left, right))]
        before, after = aggregate(rows, left, shared), aggregate(rows, right, shared)
        pairs[f'{left}_vs_{right}'] = {'episode_ids': shared, 'before': before, 'after': after,
            'delta': {k: after[k] - before[k] if after[k] is not None and before[k] is not None else None for k in METRICS}}
    return {'schema': 1, 'snapshot_utc': datetime.now(timezone.utc).isoformat(), 'episode_ids': ids,
        'sources': sources,
        'version_provenance': {v: {'root': str(roots[v]) if roots[v] else None,
                                  'commit': manifests.get(v, {}).get('code', {}).get('commit'),
                                  'fingerprint': manifests.get(v, {}).get('fingerprint')} for v in VERSIONS},
        'episodes': rows, 'paired': pairs,
        'cohorts': {name: {v: aggregate(rows, v, selected) for v in VERSIONS}
                    for name, selected in [('all_diagnostics', ids), ('directed_failures', [i for i in ids if i not in CONTROL_IDS]),
                                            ('success_controls', [i for i in ids if i in CONTROL_IDS])]},
        'definitions': {'sr': 'distance < radius with actual terminal STOP and normal error-free termination',
                        'osr': 'minimum saved evaluator distance < radius, independent of STOP',
                        'oracle_hit_without_success': 'OSR=1 and SR=0; this alone does not prove the agent left the goal',
                        'visited_then_left': 'minimum distance < radius but final distance >= radius',
                        'unknown_events': 'null means unavailable instrumentation, not zero events',
                        'confirmation_observations': 'changed-view confirmation flags, not necessarily successful STOP',
                        'legacy_calls': 'logged verifier invocations in episode-indexed segments, not HTTP retries'},
        'limitations': ['Directed failures and two success controls are not an unbiased benchmark.',
                        'Different stochastic trajectories prevent attributing outcome changes to one fix.',
                        'Original distances are rounded; newer evaluation blocks preserve full precision.',
                        'Where raw JSON lacks identity fields, identity uses manifest mapping, filename and saved hash.',
                        'All evaluator/goal data here are used only offline; this script never runs or feeds an agent.']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root', type=Path, required=True)
    parser.add_argument('--previous-root', type=Path, required=True)
    parser.add_argument('--current-root', type=Path)
    parser.add_argument('--episode-ids', type=parse_ids, default=FAILURE_IDS + CONTROL_IDS)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare(args.original_root, args.previous_root, args.current_root, args.episode_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=args.output.parent, delete=False, encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        temporary = Path(stream.name)
    temporary.replace(args.output)
    print(f'Wrote {len(report["episodes"])} official-episode comparisons to {args.output}')


if __name__ == '__main__':
    main()


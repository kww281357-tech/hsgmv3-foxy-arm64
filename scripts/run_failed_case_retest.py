#!/usr/bin/env python3
"""Sequential, resumable diagnostics. This module imports no simulator or model.

Example (endpoint/credentials must already be configured in the environment)::
    python scripts/run_failed_case_retest.py --dataset /path/val_unseen.json \
        --baseline-root /path/previous/results_by_ep_id --output-root /path/new-run \
        --python /path/hsgm/bin/python --wait-for-pid 1234 --include-controls

Use --dry-run to validate IDs and print a provenance manifest without launching or
creating outputs. Resuming never repeats an interrupted/ambiguous episode: choose
a new output directory for an intentional rerun.

Remote OpenAI-compatible HTTPS endpoints require --allow-remote-api. Supply
credentials through the environment or --env-file, and select --model explicitly.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
FAILURE_IDS = [722, 1518, 626, 1263, 1159, 1395]
CONTROL_IDS = [1434, 556]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def digest_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def parse_ids(text):
    values = [int(part.strip()) for part in text.split(',')]
    if not values or any(value < 0 for value in values) or len(set(values)) != len(values):
        raise ValueError('Episode IDs must be unique, nonnegative integers.')
    return values


def dataset_metadata(path, ids):
    episodes = json.loads(Path(path).read_text())['episodes']
    by_id = {}
    for index, episode in enumerate(episodes):
        eid = int(episode['episode_id'])
        if eid in by_id:
            raise ValueError(f'Duplicate official episode ID in dataset: {eid}')
        by_id[eid] = {'episode_id': eid, 'episode_ndx': index,
                     'scene_id': episode['scene_id'],
                     'geodesic': float(episode['info']['geodesic_distance'])}
    missing = sorted(set(ids) - by_id.keys())
    if missing:
        raise ValueError(f'Official episode IDs absent from dataset: {missing}')
    return [by_id[eid] for eid in ids]


def finite_number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def summarize_result(data, meta):
    """Require a recorded terminal STOP and a normal, error-free termination."""
    if data.get('diagnostic_segment') is not None:
        raise ValueError('Local diagnostic segments cannot be counted as full-episode benchmark results.')
    metrics = data.get('metrics', {}) or {}
    precise = data.get('evaluation', {}) or {}
    def number(name):
        return finite_number(precise[name] if precise.get(name) is not None else metrics.get(name))
    records = [r for sub in data.get('subtask', {}).values() for r in sub.get('record', [])
               if isinstance(r, dict) and r.get('action_type')]
    records.sort(key=lambda r: r.get('step', -1))
    execution = data.get('execution_summary')
    if execution is not None:
        stop = execution.get('stop_executed') is True and execution.get('last_action_type') == 'stop'
        last_type = execution.get('last_action_type')
        stop_count = execution.get('stop_action_count', 0)
        source = 'execution_summary'
    elif records:
        stop = records[-1]['action_type'] == 'stop'
        last_type = records[-1]['action_type']
        stop_count = sum(r['action_type'] == 'stop' for r in records)
        source = 'execution_ledger'
    else:
        stop, last_type, stop_count, source = False, None, None, 'unavailable'
    status = metrics.get('finish_status', 'unknown')
    normal_stop = (stop and status in {'success', 'fp'} and not metrics.get('error_info')
                   and not metrics.get('has_teleport_correction'))
    ne, oracle, traveled = number('distance_to_goal'), number('oracle_navigation_error'), number('traveled_distance')
    geo = meta['geodesic']
    ratio = geo / max(geo, traveled) if geo > 0 and traveled is not None and traveled >= 0 else 0.0
    trace = [event for event in data.get('trace', data.get('navigation_trace', [])) if isinstance(event, dict)]
    corrections = [event for event in trace if event.get('role') == 'verifier_correction']
    result = {**meta, 'ne': ne, 'oracle_ne': oracle, 'traveled_distance': traveled,
              'finish_status': status, 'stop_executed': stop, 'normal_stop': normal_stop,
              'last_action_type': last_type, 'stop_action_count': stop_count, 'stop_evidence_source': source,
              'runner_reported_success': bool(metrics.get('goal_reached')),
              'verifier_trace_events': sum(e.get('role') == 'verifier' for e in trace),
              'verifier_results': sum(e.get('role') == 'verifier' and e.get('event') in
                                      {'verification_result', 'route_verification_result', 'verify_result'} for e in trace),
              'verifier_corrections': len(corrections),
              'completion_proposals': sum(e.get('role') == 'planner' and
                                          isinstance(e.get('action'), dict) and
                                          e['action'].get('type') == 'propose_completion' for e in trace),
              'rollback_requests': sum(e.get('executed_type') == 'rollback' for e in corrections),
              'recovery_motion_m': sum(float(r.get('motion_distance', 0)) for r in records if r.get('recovery'))}
    for phase in ('started', 'completed', 'failed'):
        result['rollback_' + phase] = sum(e.get('event') in
                                          {'wrong_path_rollback_' + phase, 'rollback_' + phase} for e in trace)
    for threshold in (2, 3):
        reached = ne is not None and ne < threshold
        success = bool(reached and normal_stop)
        result.update({f'sr_{threshold}m': int(success), f'distance_only_{threshold}m': int(reached),
                       f'osr_{threshold}m': int(oracle is not None and oracle < threshold),
                       f'spl_{threshold}m': ratio if success else 0.0})
    return result


def code_identity(root):
    files = {}
    for folder in ('src', 'scripts', 'tests', 'config'):
        for path in sorted((root / folder).rglob('*')):
            if path.is_file() and path.suffix in {'.py', '.yaml', '.yml', '.toml', '.txt'}:
                files[path.relative_to(root).as_posix()] = sha256_file(path)
    for path in sorted(root.glob('requirements*.txt')):
        files[path.name] = sha256_file(path)
    head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                          check=True, capture_output=True, text=True).stdout.strip()
    return {'commit': head, 'source_sha256': digest_object(files), 'files': files}


def endpoint_mode(env, allow_remote_api=False):
    """Validate configuration without making requests or echoing URL secrets."""
    endpoint = env.get('OPENAI_BASE_URL', '')
    try:
        parsed = urlsplit(endpoint)
        valid = (bool(parsed.hostname) and parsed.port != 0 and parsed.username is None
                 and parsed.password is None and not parsed.query and not parsed.fragment
                 and not any(character.isspace() for character in endpoint))
    except ValueError:
        valid = False
    if not valid:
        raise ValueError('OPENAI_BASE_URL must be an HTTP(S) base URL without credentials, query, or fragment.')
    if parsed.hostname in {'localhost', '127.0.0.1', '::1'} and parsed.scheme in {'http', 'https'}:
        return 'local'
    if not allow_remote_api:
        raise ValueError('Remote OPENAI_BASE_URL requires explicit --allow-remote-api; default mode is local only.')
    if parsed.scheme != 'https':
        raise ValueError('Remote OPENAI_BASE_URL must use HTTPS.')
    return 'remote_https'


def configured_environment(args):
    env = dict(os.environ)
    if args.env_file:
        for line in args.env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('export '):
                line = line[7:]
            key, separator, value = line.partition('=')
            if not separator or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key.strip()):
                raise ValueError('Invalid environment file assignment; shell expressions are unsupported.')
            env[key.strip()] = value.strip().strip('\"\'')
    mode = endpoint_mode(env, getattr(args, 'allow_remote_api', False))
    if not env.get('OPENAI_API_KEY', '').strip():
        raise ValueError('Configure OPENAI_API_KEY in the environment or --env-file.')
    for key in tuple(env):
        if key.startswith('AZURE_OPENAI_'):
            env.pop(key)
    env.update(OPENAI_MODEL=args.model, HSGM_DISABLE_THINKING='1', HSGM_ENABLE_THINKING='0',
               HSGM_JSON_OBJECT_RESPONSES='1' if getattr(args, 'json_object_responses', False) else '0',
               HSGM_MAX_EPISODE_SECONDS=str(args.max_seconds), HSGM_MAX_MODEL_REQUESTS=str(args.max_requests),
               HSGM_MODEL_REQUEST_TIMEOUT_S='180', HSGM_SOFT_BUDGET_MARGIN_S='60',
               HSGM_ALLOW_SOFT_LIMIT_GOAL_CREDIT='0', PYTHONUNBUFFERED='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               NUMEXPR_NUM_THREADS='4', TORCH_NUM_THREADS='4')
    if mode == 'local':
        # Remote mode preserves the caller's proxy routing, including NO_PROXY.
        for key in ('http_proxy', 'https_proxy', 'all_proxy', 'socks_proxy',
                    'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'SOCKS_PROXY'):
            env.pop(key, None)
        env['NO_PROXY'] = env['no_proxy'] = 'localhost,127.0.0.1,::1'
    if getattr(args, 'verifier_dino_sam3', False):
        env['HSGM_VERIFIER_DINO_SAM3'] = '1'
    elif 'HSGM_VERIFIER_DINO_SAM3' not in env:
        env['HSGM_VERIFIER_DINO_SAM3'] = '0'
    return env


def build_manifest(args, metas, env):
    baseline = []
    for meta in metas:
        path = args.baseline_root / f"episode_{meta['episode_id']}_results.json"
        if not path.is_file():
            raise ValueError(f"Baseline missing for episode {meta['episode_id']}")
        baseline.append({'path': str(path), 'sha256': sha256_file(path),
                         'summary': summarize_result(json.loads(path.read_text()), meta)})
    payload = {'schema': 1, 'repository': str(args.repo_root), 'code': code_identity(args.repo_root),
               'dataset': {'path': str(args.dataset), 'sha256': sha256_file(args.dataset)},
               'config': {'path': str(args.config), 'sha256': sha256_file(args.config)},
               'python': str(args.python), 'python_sha256': sha256_file(args.python), 'model': args.model,
               'inference_mode': endpoint_mode(env, getattr(args, 'allow_remote_api', False)),
               'json_object_responses': env.get('HSGM_JSON_OBJECT_RESPONSES') == '1',
               'endpoint_sha256': hashlib.sha256(env['OPENAI_BASE_URL'].encode()).hexdigest(),
               'runtime_environment_sha256': digest_object({key: value for key, value in env.items()
                                                            if key.startswith('HSGM_')}),
               'max_steps': args.max_steps, 'success_threshold': 2.0,
               'max_seconds': args.max_seconds, 'max_requests': args.max_requests,
               'episodes': metas, 'baseline': baseline}
    return {'created_at': now(), 'fingerprint': digest_object(payload), **payload}


def read_processes(proc_root=Path('/proc')):
    """PID start times prevent accidentally waiting for a recycled PID."""
    result = {}
    for directory in proc_root.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            stat = (directory / 'stat').read_text().rsplit(')', 1)[1].split()
            if stat[0] == 'Z':
                continue
            argv = (directory / 'cmdline').read_bytes().decode(errors='replace').rstrip('\0').split('\0')
            result[int(directory.name)] = {'start': stat[19], 'state': stat[0], 'argv': argv}
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return result


def experiment_process(info):
    # Match script arguments, never a substring inside shell/Python source text.
    for arg in info['argv']:
        name = Path(arg).name
        if '/' in name or '\n' in name or ' ' in name:
            continue
        if name == 'run_experiments.py' or re.fullmatch(r'run_(?:v[23]_.*|.*(?:benchmark|retest).*)\.py', name):
            return True
    return False


def wait_until_idle(pids, poll_seconds=10.0, quiet_seconds=20.0, process_reader=read_processes,
                    sleep=time.sleep, clock=time.monotonic, allow_paused=None):
    initial = process_reader()
    watched = {pid: initial[pid]['start'] for pid in pids if pid in initial}
    allow_paused = allow_paused or {}
    quiet_since = None
    announced = None
    while True:
        processes = process_reader()
        busy = sorted(pid for pid, info in processes.items() if pid != os.getpid() and
                      (experiment_process(info) or watched.get(pid) == info['start']) and not
                      (allow_paused.get(pid) == info['start'] and info.get('state') in {'T', 't'}))
        if busy:
            quiet_since = None
            if busy != announced:
                print(f'Waiting for existing experiment processes: {busy}', flush=True)
                announced = busy
        else:
            quiet_since = clock() if quiet_since is None else quiet_since
            if clock() - quiet_since >= quiet_seconds:
                return
        sleep(poll_seconds)


@contextmanager
def exclusive_runner(output_root=None, allow_parallel=False):
    if allow_parallel and output_root is not None:
        tag = hashlib.sha256(str(Path(output_root).resolve()).encode()).hexdigest()[:12]
        path = Path(tempfile.gettempdir()) / f'hsgm-failed-retest-{os.getuid()}-{tag}.lock'
    else:
        path = Path(tempfile.gettempdir()) / f'hsgm-failed-retest-{os.getuid()}.lock'
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f'Another failed-case retest runner is active{" on this output root" if allow_parallel else ""}.') from None
        yield


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare_output(args, manifest):
    root = args.output_root
    if args.resume:
        previous = json.loads((root / 'manifest.json').read_text())
        if previous['fingerprint'] != manifest['fingerprint']:
            raise ValueError('Resume provenance differs; use a new output root.')
        state = json.loads((root / 'state.json').read_text())
        if any(row['status'] != 'finished' for row in state['episodes'].values()):
            raise ValueError('Interrupted episode found; refusing an implicit rerun. Use a new output root.')
        for row in state['episodes'].values():
            if sha256_file(Path(row['result_path'])) != row['result_sha256']:
                raise ValueError('A completed result changed since the previous run.')
        return state
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / 'manifest.json', manifest)
    state = {'episodes': {}}
    atomic_json(root / 'state.json', state)
    return state


def episode_command(args, meta, raw):
    return [str(args.python), '-u', 'run_experiments.py', '--task', 'r2r',
            '--config', str(args.config), '--max_steps', str(args.max_steps),
            '--model_name', args.model, '--dataset_path', str(args.dataset),
            '--output_root', str(raw), '--memory_storage_path', str(raw / 'memory'),
            '--success_threshold', '2.0', '--episode_ids', str(meta['episode_ndx']),
            '--comment', f"failed_retest_{meta['episode_id']}"]


def write_summary(root, state, total):
    rows = [row['summary'] for row in state['episodes'].values() if row['status'] == 'finished']
    summary = {'completed_count': len(rows), 'total_target': total, 'per_episode': rows}
    for key in ('sr_2m', 'sr_3m', 'osr_2m', 'osr_3m', 'spl_2m', 'spl_3m'):
        summary[key] = sum(row[key] for row in rows) / len(rows) if rows else None
    atomic_json(root / 'summary.json', summary)


def run_sequential(args, metas, manifest, env, launch=subprocess.run, wait=wait_until_idle):
    state = prepare_output(args, manifest)
    # Pin exemptions once, retaining the original start time across episodes
    # and resume. Authorization covers only these stopped process identities.
    # A resumed process or a recycled PID must re-enter the normal busy check.
    previous = state.get('allowed_paused_processes', {})
    processes = read_processes()
    allowed = {pid: previous.get(str(pid), processes[pid]['start'] if pid in processes else None)
               for pid in getattr(args, 'allow_paused_pid', [])}
    allowed = {pid: start for pid, start in allowed.items() if start is not None}
    state['allowed_paused_processes'] = {str(pid): start for pid, start in allowed.items()}
    atomic_json(args.output_root / 'state.json', state)
    for meta in metas:
        eid = str(meta['episode_id'])
        if eid in state['episodes']:
            continue
        if not getattr(args, 'allow_parallel', False):
            wait(args.wait_for_pid, allow_paused=allowed)
        elif args.wait_for_pid:
            wait(args.wait_for_pid, allow_paused=allowed)
        # A long wait must not silently run code/config/data changed by another task.
        if build_manifest(args, metas, env)['fingerprint'] != manifest['fingerprint']:
            raise ValueError('Inputs changed while waiting; create a fresh manifest/output root.')
        raw = args.output_root / 'raw' / eid
        raw.mkdir(parents=True, exist_ok=False)
        logs = args.output_root / 'logs'
        logs.mkdir(exist_ok=True)
        command = episode_command(args, meta, raw)
        state['episodes'][eid] = {'status': 'started', 'started_at': now(), 'command': command}
        atomic_json(args.output_root / 'state.json', state)
        print(f"Running official episode {eid} (dataset index {meta['episode_ndx']})", flush=True)
        with (logs / f'episode_{eid}.log').open('x') as log:
            completed = launch(command, cwd=args.repo_root / 'src', env=env,
                               stdout=log, stderr=subprocess.STDOUT, check=False)
        if build_manifest(args, metas, env)['fingerprint'] != manifest['fingerprint']:
            raise ValueError('Inputs changed during execution; result retained without reproducibility credit.')
        candidates = list(raw.rglob(f"episode_{meta['episode_ndx']}_results.json"))
        all_results = list(raw.rglob('episode_*_results.json'))
        if completed.returncode or len(candidates) != 1 or len(all_results) != 1:
            raise ValueError(f'Episode {eid} subprocess failed or produced missing/ambiguous results; retained as started.')
        data = json.loads(candidates[0].read_text())
        row = summarize_result(data, meta)
        if row['stop_evidence_source'] == 'unavailable':
            raise ValueError(f'Episode {eid} lacks execution evidence; refusing to infer STOP.')
        output = args.output_root / 'results_by_ep_id'
        output.mkdir(exist_ok=True)
        result_path = output / f'episode_{eid}_results.json'
        with result_path.open('xb') as stream:
            stream.write(candidates[0].read_bytes())
        state['episodes'][eid].update(status='finished', finished_at=now(), summary=row,
                                     result_path=str(result_path), result_sha256=sha256_file(result_path))
        atomic_json(args.output_root / 'state.json', state)
        write_summary(args.output_root, state, len(metas))
    write_summary(args.output_root, state, len(metas))
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo-root', type=Path, default=ROOT)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--baseline-root', type=Path, required=True, help='Directory containing episode_ID_results.json')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--allow-remote-api', action='store_true',
                        help='Explicitly allow a remote HTTPS OPENAI_BASE_URL; preserves proxy environment')
    parser.add_argument('--json-object-responses', action='store_true',
                        help='Request JSON object responses; default off overrides inherited environment')
    parser.add_argument('--model', default='qwen3.8-27b-q4')
    parser.add_argument('--episode-ids', type=parse_ids, default=FAILURE_IDS)
    parser.add_argument('--include-controls', action='store_true')
    parser.add_argument('--max-steps', type=int, default=150)
    parser.add_argument('--max-seconds', type=int, default=3200)
    parser.add_argument('--max-requests', type=int, default=96)
    parser.add_argument('--wait-for-pid', type=int, action='append', default=[])
    parser.add_argument('--allow-paused-pid', type=int, action='append', default=[],
                        help='Authorized existing PID to ignore only while stopped (T/t); may be repeated')
    parser.add_argument('--allow-parallel', action='store_true',
                        help='Scope lock to output_root and bypass idle-process check for parallel experiments')
    parser.add_argument('--verifier-dino-sam3', action='store_true',
                        help='Enable GroundingDINO + SAM3 perception refresh during verification')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    args.repo_root = args.repo_root.resolve()
    args.config = (args.config or args.repo_root / 'config/vlnce_test.yaml').resolve()
    for name in ('dataset', 'baseline_root', 'output_root', 'python'):
        setattr(args, name, getattr(args, name).resolve())
    try:
        if (min(args.max_steps, args.max_seconds, args.max_requests) <= 0
                or any(pid <= 0 for pid in args.wait_for_pid + args.allow_paused_pid)):
            raise ValueError('Limits and process IDs must be positive.')
        ids = list(args.episode_ids)
        if args.include_controls:
            ids = [eid for eid in ids if eid not in CONTROL_IDS] + CONTROL_IDS
        metas = dataset_metadata(args.dataset, ids)
        env = configured_environment(args)
        manifest = build_manifest(args, metas, env)
        if args.dry_run:
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
            return 0
        print('API configuration preflight passed: ' + json.dumps({key: manifest[key] for key in
              ('inference_mode', 'endpoint_sha256', 'model', 'json_object_responses')}), flush=True)
        with exclusive_runner(args.output_root, args.allow_parallel):
            run_sequential(args, metas, manifest, env)
        return 0
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        parser.exit(2, f'{type(error).__name__}: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())


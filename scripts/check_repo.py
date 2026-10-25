#!/usr/bin/env python3
"""Check code syntax and the staged snapshot without importing production models."""
import argparse
import ast
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROOT_FILES = {
    '.gitignore', '.env.example', 'AGENTS.md', 'README.md', 'LICENSE',
    'requirements.txt', 'Dockerfile', '.dockerignore', 'compose.yaml',
}
ALLOWED_EXTENSIONS = {
    'src': {'.py'}, 'config': {'.yaml', '.yml'}, 'tests': {'.py', '.md'},
    'scripts': {'.py', '.sh'}, 'docs': {'.md', '.json'},
    'docker': {'.py', '.sh', '.yaml', '.xml'},
    'ros2_ws': {'.py', '.xml', '.cfg'},
}
SECRET_PATTERNS = [
    re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(rb'\bgh[pousr]_[A-Za-z0-9]{30,}\b'),
    re.compile(rb'\bgithub_pat_[A-Za-z0-9_]{30,}\b'),
    re.compile(rb'\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}\b'),
    re.compile(rb'\bAKIA[0-9A-Z]{16}\b'),
    re.compile(rb'https?://[^\s/:]+:[^\s/@]+@'),
]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def check_staged():
    count = 0
    # Inspect the entire index: ignored files may already have been tracked.
    for record in git('ls-files', '--stage', '-z').split(b'\0'):
        if not record:
            continue
        meta, raw_path = record.split(b'\t', 1)
        mode, oid, stage = meta.decode().split()
        name = raw_path.decode()
        path = PurePosixPath(name)
        allowed = name in ALLOWED_ROOT_FILES
        if len(path.parts) > 1:
            allowed = path.suffix in ALLOWED_EXTENSIONS.get(path.parts[0], set())
        research = len(path.parts) > 2 and path.parts[0] == 'research' and path.parts[1] in {'blue_help_navigation', 'character_assets'}
        if research:
            allowed = (path.suffix in {'.py', '.md', '.json', '.txt', '.toml', '.sh', '.yml', '.yaml', '.obj', '.mtl'}
                       or path.name in {'Makefile', '.gitignore', 'HSGMV3_LICENSE', 'BLUE_DUMMY_LICENSE'})
        if name in {'docker/Dockerfile.jetson-base', '.githooks/pre-commit', 'ros2_ws/src/hsgm_go2_monitor/resource/hsgm_go2_monitor'}:
            allowed = True
        if (not allowed or mode not in {'100644', '100755'} or stage != '0'
                or any(part.startswith('.') and not (research and part == '.gitignore' and part == path.name)
                       for part in path.parts[1:])
                or any(part in {'__pycache__', 'results', 'data', 'ckpt', 'logs'} for part in path.parts)
                or '.bak' in path.name or '.local.' in path.name):
            raise ValueError('Disallowed staged file: ' + name)
        data = git('cat-file', 'blob', oid)
        if len(data) > 1024 * 1024 or b'\0' in data:
            raise ValueError('Binary or oversized staged file: ' + name)
        data.decode('utf-8')
        if any(pattern.search(data) for pattern in SECRET_PATTERNS):
            raise ValueError('Possible credential in staged file: ' + name)
        if path.suffix == '.py':
            ast.parse(data, filename=name)
        count += 1
    print('Staged snapshot passed: {} text/code files.'.format(count), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged', action='store_true')
    parser.add_argument('--offline-tests', action='store_true')
    args = parser.parse_args()
    count = 0
    for folder in ('src', 'tests', 'scripts', 'docker', 'ros2_ws'):
        for path in sorted((ROOT / folder).rglob('*.py')):
            if '__pycache__' not in path.parts:
                ast.parse(path.read_bytes(), filename=str(path.relative_to(ROOT)))
                count += 1
    print('Python syntax passed: {} files.'.format(count), flush=True)
    if args.staged:
        check_staged()
    if args.offline_tests:
        subprocess.run(
            [sys.executable, '-B', str(ROOT / 'scripts' / 'run_offline_tests.py')],
            cwd=ROOT / 'tests',
            check=True,
        )


if __name__ == '__main__':
    try:
        main()
    except (ValueError, SyntaxError, UnicodeError, subprocess.CalledProcessError) as exc:
        print('CHECK FAILED: ' + str(exc), file=sys.stderr)
        sys.exit(1)


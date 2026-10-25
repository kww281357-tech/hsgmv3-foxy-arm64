"""Verify the published source manifest; no download, credentials or API access."""
from pathlib import Path
import argparse
import hashlib
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root/'SOURCE_MANIFEST.json').read_text())
    seen = set()
    for row in manifest['files']:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or relative.as_posix() in seen:
            raise SystemExit('Unsafe or duplicate source manifest path')
        seen.add(relative.as_posix())
        path = (root/relative).resolve()
        if not path.is_relative_to(root):
            raise SystemExit('Source symlink escapes root')
        data = path.read_bytes()
        if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
            raise SystemExit('Source checksum mismatch: '+relative.as_posix())
    print(json.dumps({'status': 'all_manifest_source_files_verified', 'files': len(seen),
                      'version': manifest['version'], 'network_calls': 0}))


if __name__ == '__main__':
    main()

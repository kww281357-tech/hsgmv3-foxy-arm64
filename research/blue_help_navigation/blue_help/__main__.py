"""One lazy CLI: help/plan/unit tests never start a simulator or network client."""
import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {
        'trial': ('blue_help.trial.run_trial', 'One staged full-route trial; fixture mode has no API'),
        'probe': ('blue_help.trial.probe', 'One authorized image/API compatibility probe'),
        'smoke': ('blue_help.smoke', 'Legacy scripted interaction diagnostic, not autonomous navigation'),
        'geometry': ('blue_help.preflight', 'Private geometry-only preflight, not navigation results'),
        'survey': ('blue_help.manual_route.survey', 'Privileged offline full-route survey, not policy execution'),
    }
    if argv and argv[0] in commands:
        import importlib
        return importlib.import_module(commands[argv[0]][0]).main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    val = sub.add_parser('validate-data'); val.add_argument('--data-root', type=Path, required=True)
    plan = sub.add_parser('plan'); plan.add_argument('--batch-size', type=int, default=2)
    for command, (_, description) in commands.items():
        sub.add_parser(command, help=description)
    args = parser.parse_args(argv)
    if args.command == 'validate-data':
        from .data import validate_frozen_data
        result = validate_frozen_data(args.data_root)
    else:
        from .evaluation import build_paired_manifest
        result = build_paired_manifest(batch_size=args.batch_size)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

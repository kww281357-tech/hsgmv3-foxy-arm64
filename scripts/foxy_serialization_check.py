#!/usr/bin/env python3
"""Check real Foxy-generated Unitree messages/CDR against physics trace; no DDS node."""
import argparse
import json
import math
from pathlib import Path
import sys


def require(value, message):
    if not value:
        raise RuntimeError(message)


def run(args):
    require(sys.version_info[:2] == (3, 8), 'Expected actual CPython3.8 for Foxy')
    from rclpy.serialization import serialize_message, deserialize_message
    from unitree_go.msg import SportModeState
    rows = [json.loads(s) for s in Path(args.trace).read_text().splitlines() if s.strip()]
    require(len(rows) > 0, 'Empty trace')
    sizes = set()
    previous = {}
    for row in rows:
        require(row['time'] > previous.get(row['episode'], -1), 'Nonincreasing episode timestamp')
        previous[row['episode']] = row['time']
        msg = SportModeState()
        for key in ('position', 'velocity', 'yaw_speed', 'mode', 'error_code'):
            setattr(msg, key, row[key])
        msg.stamp.sec = int(row['time'])
        msg.stamp.nanosec = int((row['time'] - int(row['time'])) * 1e9)
        encoded = serialize_message(msg)
        sizes.add(len(encoded))
        decoded = deserialize_message(encoded, SportModeState)
        for key in ('position', 'velocity'):
            require(len(getattr(decoded, key)) == 3, 'Wrong vector size')
            require(all(math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6)
                        for a, b in zip(getattr(decoded, key), row[key])), 'Float32 roundtrip mismatch')
        require(math.isclose(decoded.yaw_speed, row['yaw_speed'], rel_tol=1e-6, abs_tol=1e-6), 'Yaw mismatch')
        require(decoded.stamp == msg.stamp and decoded.mode == msg.mode and decoded.error_code == msg.error_code, 'Scalar mismatch')
    invalid = SportModeState()
    invalid.position = [float('nan'), 0., 0.]
    decoded = deserialize_message(serialize_message(invalid), SportModeState)
    require(math.isnan(decoded.position[0]), 'NaN transport expectation changed')
    # CDR preserves NaN; application must reject it. This does not test the callback.
    result = dict(result='PASS', python=sys.version, samples=len(rows),
                  episodes=len(previous), cdr_sizes=sorted(sizes), nan_preserved=True,
                  real_generated_message=True, dds_node_created=False,
                  limitation='Serialization only; not DDS transport, monitor reception, ARM64 or CUDA')
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', required=True)
    p.add_argument('--output', required=True)
    run(p.parse_args())

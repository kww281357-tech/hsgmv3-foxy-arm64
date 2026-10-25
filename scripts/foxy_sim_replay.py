#!/usr/bin/env python3
"""Isolated DDS telemetry integration test; never sends robot control commands.

Requires a sourced Foxy + unitree_go workspace. Run only in a simulation-only
container/host, NOT on a robot or on a host with a robot network attached.
Input is the JSONL emitted by go2_mujoco_smoke.py. This is replay, not physics.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_status(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def run(args):
    require(args.confirm_isolated_simulation, 'Pass --confirm-isolated-simulation only on an isolated simulator host')
    require(sys.version_info[:2] == (3, 8), 'This Foxy test requires Python 3.8')
    require(os.environ.get('ROS_DISTRO') == 'foxy', 'Source a real Foxy environment first')
    require('lo' in [name for _, name in socket.if_nameindex()], 'Loopback lo is required')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    status = out / 'monitor-status.json'
    require(not status.exists(), 'Use a fresh output directory to avoid stale evidence')
    xml = out / 'loopback.xml'
    xml.write_text('''<CycloneDDS><Domain id="211"><General>
<NetworkInterfaceAddress>lo</NetworkInterfaceAddress><AllowMulticast>false</AllowMulticast>
</General><Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>9</MaxAutoParticipantIndex>
<Peers><Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>''')
    # Overwrite inherited DDS settings, including a possibly robot-facing URI.
    os.environ.update(ROS_DOMAIN_ID='211', ROS_LOCALHOST_ONLY='1',
                      RMW_IMPLEMENTATION='rmw_cyclonedds_cpp',
                      CYCLONEDDS_URI=xml.as_uri(), HSGM_STATUS_PATH=str(status))
    rows = [json.loads(line) for line in Path(args.trace).read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row['episode'] == 0]
    require(len(rows) >= 2, 'Need episode 0 physics trace')
    require(all(b['time'] > a['time'] for a, b in zip(rows, rows[1:])), 'Simulation timestamps must increase')
    require(all(math.isfinite(float(v)) for row in rows for v in row['position'] + row['velocity'] + [row['yaw_speed']]), 'Non-finite input trace')
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from unitree_go.msg import SportModeState
    monitor_file = Path(args.monitor).resolve()
    require(monitor_file.is_file(), 'Monitor source file not found')
    child_code = 'import runpy; runpy.run_path(%r)["main"]()' % str(monitor_file)
    child = None
    node = None
    snapshots = []
    try:
        rclpy.init()
        node = rclpy.create_node('hsgm_sim_telemetry_replay', enable_rosout=False, start_parameter_services=False)
        publisher = node.create_publisher(SportModeState, '/lf/sportmodestate', qos_profile_sensor_data)
        log = (out / 'monitor.log').open('w')
        child = subprocess.Popen([sys.executable, '-c', child_code], stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 15
        while not read_status(status):
            require(child.poll() is None, 'Monitor exited; inspect monitor.log')
            require(time.monotonic() < deadline, 'Monitor startup timeout')
            time.sleep(.1)
        before = read_status(status)
        require(before.get('robot_connected') is False, 'Unexpected existing telemetry: domain may be occupied')
        snapshots.append({'phase': 'before_publisher', 'status': before})
        time.sleep(2)  # bounded DDS discovery; then replay at physics timestamps
        start = time.monotonic()
        for row in rows:
            due = start + row['time'] - rows[0]['time']
            time.sleep(max(0, due - time.monotonic()))
            message = SportModeState()
            for name in ('position', 'velocity', 'yaw_speed', 'mode', 'error_code'):
                setattr(message, name, row[name])
            seconds = row['time']
            message.stamp.sec = int(seconds)
            message.stamp.nanosec = int((seconds - int(seconds)) * 1e9)
            publisher.publish(message)
            rclpy.spin_once(node, timeout_sec=0)
        live = read_status(status)
        require(live.get('robot_connected') is True, 'No live telemetry received over DDS')
        require(live.get('read_only') is True, 'Monitor lost read-only marker')
        snapshots.append({'phase': 'live_physics_replay', 'status': live})
        invalid = SportModeState()
        invalid.position = [float('nan'), 0., 0.]
        for _ in range(10):
            publisher.publish(invalid)
            time.sleep(.05)
        time.sleep(2.2)
        invalid_status = read_status(status)
        require(invalid_status.get('invalid_messages', 0) > 0, 'NaN telemetry was not rejected')
        snapshots.append({'phase': 'invalid_nan_rejected', 'status': invalid_status})
        node.destroy_publisher(publisher)
        time.sleep(7)
        stale = read_status(status)
        require(stale.get('robot_connected') is False, 'Stopped publisher did not become stale')
        require(stale['telemetry_age_seconds'] >= 5, 'Stale age threshold not reached')
        require(stale['heartbeat_monotonic'] > live['heartbeat_monotonic'], 'Heartbeat stopped')
        require(all(math.isclose(float(a), float(b), rel_tol=1e-5, abs_tol=1e-5)
                    for a, b in zip(stale['telemetry']['position'], rows[-1]['position'])),
                'Final valid physics position did not reach monitor')
        snapshots.append({'phase': 'publisher_stopped_stale', 'status': stale})
        result = dict(result='PASS', kind='real_foxy_DDS_replay_not_physics', domain=211,
                      interface='lo', multicast=False, samples=len(rows), snapshots=snapshots)
        (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        print(json.dumps(result))
    finally:
        if child is not None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait(timeout=5)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', required=True)
    parser.add_argument('--monitor', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--confirm-isolated-simulation', action='store_true')
    run(parser.parse_args())

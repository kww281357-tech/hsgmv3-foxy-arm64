"""Pure callback/state-machine tests, not ROS or physics integration claims."""
import ast
import contextlib
import io
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]


def callbacks():
    path = ROOT / 'ros2_ws/src/hsgm_go2_monitor/hsgm_go2_monitor/monitor.py'
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'telemetry_packet')
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Monitor')
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('on_state', 'heartbeat')]
    clock = types.SimpleNamespace(value=100.)
    clock.monotonic = lambda: clock.value
    scope = {'math': math, 'json': json, 'time': clock}
    exec(compile(ast.Module(body=[fn, cls], type_ignores=[]), str(path), 'exec'), scope)
    return scope['Monitor'], clock


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        cls, self.clock = callbacks()
        self.node = cls()
        self.node.last_received = self.node.last_packet = None
        self.node.invalid_messages = 0
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.node.status_path = Path(self.directory.name) / 'state.json'
        self.message = types.SimpleNamespace(position=[0., 0., .3], velocity=[0., 0., 0.], yaw_speed=0., mode=0, error_code=0)

    def status(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.node.heartbeat()
        self.assertFalse(self.node.status_path.with_suffix('.tmp').exists())
        return json.loads(self.node.status_path.read_text())

    def test_never_connected_then_fresh_then_stale_and_recovered(self):
        self.assertFalse(self.status()['robot_connected'])
        self.node.on_state(self.message)
        self.assertTrue(self.status()['robot_connected'])
        self.clock.value += 5.
        self.assertFalse(self.status()['robot_connected'])
        self.node.on_state(self.message)
        state = self.status()
        self.assertTrue(state['robot_connected'])
        self.assertTrue(state['read_only'])
        self.assertEqual(state['telemetry_age_seconds'], 0.)

    def test_invalid_packet_does_not_refresh_telemetry(self):
        self.node.on_state(self.message)
        for invalid in (float('nan'), float('inf'), -float('inf')):
            self.clock.value += 2.
            self.message.velocity[0] = invalid
            self.node.on_state(self.message)
        state = self.status()
        self.assertEqual(state['invalid_messages'], 3)
        self.assertFalse(state['robot_connected'])
        self.assertEqual(state['telemetry']['velocity'], [0., 0., 0.])

    def test_wrong_vector_length_rejected(self):
        self.message.position = [0.]
        self.node.on_state(self.message)
        state = self.status()
        self.assertEqual(state['invalid_messages'], 1)
        self.assertIsNone(state['telemetry'])

    def test_liveness_is_not_connectivity_and_rejects_stale(self):
        environment = dict(os.environ, HSGM_STATUS_PATH=str(self.node.status_path))
        command = [sys.executable, str(ROOT / 'docker/healthcheck.py')]
        def code():
            return subprocess.run(command, env=environment, capture_output=True).returncode
        self.assertNotEqual(code(), 0)  # missing status
        self.node.status_path.write_text('{invalid json')
        self.assertNotEqual(code(), 0)
        self.node.status_path.write_text(json.dumps(dict(heartbeat_monotonic=time.monotonic(), robot_connected=False)))
        self.assertEqual(code(), 0)  # process health, never permission to move
        for age in (21., -60.):
            self.node.status_path.write_text(json.dumps(dict(heartbeat_monotonic=time.monotonic() - age)))
            self.assertNotEqual(code(), 0)


if __name__ == '__main__':
    unittest.main()

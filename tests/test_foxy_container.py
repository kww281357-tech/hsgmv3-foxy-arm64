"""Offline packaging/telemetry contracts; does not require ROS or touch a robot."""
import ast
import importlib.util
from pathlib import Path
import sys
import types
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FoxyPackagingTests(unittest.TestCase):
    def test_dds_validation(self):
        module = load("dds_config", ROOT / "docker/configure_dds.py")
        root = ET.fromstring(module.build_config("eth0", "0", {"eth0"}))
        self.assertEqual(root.find(".//NetworkInterface").get("name"), "eth0")
        for interface, domain in [("missing", "0"), ("eth0", "233"), ('eth0"', "0")]:
            with self.assertRaises(ValueError):
                module.build_config(interface, domain, {"eth0"})

    def test_no_actuation_api(self):
        path = ROOT / "ros2_ws/src/hsgm_go2_monitor/hsgm_go2_monitor/monitor.py"
        tree = ast.parse(path.read_text(), feature_version=(3, 8))
        forbidden = {"create_publisher", "create_client", "create_service", "publish", "call_async"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, forbidden)

    def test_telemetry_validation(self):
        # Extract only the pure function, leaving ROS imports completely untouched.
        path = ROOT / "ros2_ws/src/hsgm_go2_monitor/hsgm_go2_monitor/monitor.py"
        tree = ast.parse(path.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "telemetry_packet")
        import math
        scope = {"math": math}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), scope)
        msg = types.SimpleNamespace(position=[1, 2, 3], velocity=[0, 0, 0],
                                    yaw_speed=0, mode=0, error_code=0)
        self.assertEqual(scope["telemetry_packet"](msg)["position"], [1., 2., 3.])
        msg.yaw_speed = float("nan")
        with self.assertRaises(ValueError):
            scope["telemetry_packet"](msg)

    def test_architecture_and_safe_network_defaults(self):
        text = (ROOT / "Dockerfile").read_text()
        self.assertIn('test "$TARGETARCH" = arm64', text)
        compose = (ROOT / "compose.yaml").read_text()
        self.assertNotIn("network_mode: host", compose)
        self.assertNotIn("ports:", compose)
        self.assertIn('ROS_LOCALHOST_ONLY: "1"', compose)


if __name__ == "__main__":
    unittest.main()

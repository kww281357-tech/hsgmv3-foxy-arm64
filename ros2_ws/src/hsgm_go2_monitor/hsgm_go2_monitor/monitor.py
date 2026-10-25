"""Subscriber only. This package contains no robot command publishers or clients."""
import json
import math
import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from unitree_go.msg import SportModeState


def telemetry_packet(message):
    position = [float(v) for v in message.position]
    velocity = [float(v) for v in message.velocity]
    yaw_speed = float(message.yaw_speed)
    if len(position) != 3 or len(velocity) != 3:
        raise ValueError("Unexpected telemetry vector length")
    if not all(math.isfinite(v) for v in position + velocity + [yaw_speed]):
        raise ValueError("Non-finite robot telemetry")
    return dict(position=position, velocity=velocity, yaw_speed=yaw_speed,
                mode=int(message.mode), error_code=int(message.error_code))


class Monitor(Node):
    def __init__(self):
        # Disable parameter services and rosout; Foxy parameter-event traffic remains.
        # No robot-command publishers or clients are created.
        super().__init__("hsgm_go2_monitor", enable_rosout=False, start_parameter_services=False)
        self.declare_parameter("state_topic", "/lf/sportmodestate")
        topic = self.get_parameter("state_topic").value
        if topic not in ("/lf/sportmodestate", "/sportmodestate"):
            raise ValueError("state_topic must be an official Go2 sport-state topic")
        self.last_received = None
        self.last_packet = None
        self.invalid_messages = 0
        self.status_path = Path(os.environ.get("HSGM_STATUS_PATH", "/tmp/hsgm-status.json"))
        self.subscription = self.create_subscription(
            SportModeState, topic, self.on_state, qos_profile_sensor_data)
        self.timer = self.create_timer(2.0, self.heartbeat)
        self.heartbeat()

    def on_state(self, message):
        try:
            packet = telemetry_packet(message)
        except (ValueError, TypeError, OverflowError):
            self.invalid_messages += 1
            return
        self.last_packet = packet
        self.last_received = time.monotonic()

    def heartbeat(self):
        now = time.monotonic()
        age = None if self.last_received is None else now - self.last_received
        status = dict(heartbeat_monotonic=now, read_only=True,
                      robot_connected=age is not None and age < 5.0,
                      telemetry_age_seconds=age, invalid_messages=self.invalid_messages,
                      telemetry=self.last_packet)
        temporary = self.status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, allow_nan=False))
        temporary.replace(self.status_path)
        print(json.dumps({key: value for key, value in status.items() if key != "telemetry"}), flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Monitor()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

# Foxy simulation message validation (2026-10-25)

## Actual results, separate from physics

- On **Debian 13 x86_64**, workspace-only CPython **3.8.20**, the official final
  ROS 2 Foxy binary archive (2023-06-20) loaded real `rclpy`.
- Built `unitree_go` from commit `668d1ec5a05d1c38d3306bdca7d59f2ba3581a88`.
- `foxy_serialization_check.py` passed **250 real MuJoCo-generated records,
  two episodes**, actual generated `SportModeState` → CDR → message round trip.
  Each serialized payload was 236 bytes. Float32 values, timestamps, mode and
  error code passed explicit checks. CDR preserved NaN, as expected; the
  application must reject it. This script creates **no ROS node or publisher**.
- Actual DDS node creation was attempted with domain **211**, explicit `lo`,
  multicast **false**, and peer `127.0.0.1`. Cyclone's configuration log confirmed
  those settings, but node creation failed during interface enumeration:
  `interfaces:ddsi_enumerate_interfaces(udp): -1`, followed by
  `rmw_create_node: failed to create domain`.
- The cloud also denied NETLINK_ROUTE creation for a network-namespace probe and
  denied ptrace for strace. Those restrictions were not changed or bypassed.
  Therefore **live publish/subscribe, monitor reception over DDS, socket binding,
  and DDS failure cases are BLOCKED/UNRUN**, not passed.

The official archive contains CycloneDDS **0.7.0**. This experiment does not
validate the Docker recipe's separately pinned CycloneDDS **0.10.2**, ARM64,
Ubuntu 20.04 kernel/runtime, Jetson CUDA, or any physical Go2 behavior.
No ROS command topics/services or robot client calls were used.

## Rerun on a proper isolated Foxy simulator

Source a real Foxy installation and built `unitree_go` overlay first. Use actual
Python 3.8. Generate the physics JSONL with the provided MuJoCo smoke script.

```bash
python3 scripts/foxy_serialization_check.py \
  --trace /path/to/telemetry.jsonl --output /path/to/serialization-result.json
```

The following **integration script is provided but its success path has not
run here**. It requires a dedicated simulator host/container, not a robot or a
computer attached to a robot network. It overwrites inherited DDS configuration
with loopback-only/domain211/no multicast, uses sensor-data QoS, and launches
monitor and telemetry publisher in separate processes. Use a fresh output path.

```bash
python3 scripts/foxy_sim_replay.py --confirm-isolated-simulation \
  --trace /path/to/telemetry.jsonl \
  --monitor ros2_ws/src/hsgm_go2_monitor/hsgm_go2_monitor/monitor.py \
  --output /path/to/new-dds-replay-output
```

It checks initial disconnected state, live replay, NaN rejection, then stopped
publisher/stale telemetry with continuing heartbeat. It publishes only simulated
sport-state telemetry, never motion/control. **Replay is not a physics engine**;
physics is the preceding MuJoCo run. The monitor's `robot_connected` field means
recent telemetry in this test, not connection to a physical robot.

## Actual rootless bootstrap provenance

This was an experimental cloud-only compatibility environment, not a deployment
installer. All archives/packages were extracted into the workspace; no host apt
installation, driver modification or security changes occurred. Official sources:

- [Final Foxy release](https://github.com/ros2/ros2/releases/tag/release-foxy-20230620)
- Archive `ros2-foxy-20230620-linux-focal-amd64.tar.bz2`, SHA256
  `7f94c846553c56064732786931e9fbc049d65bab4c3751653594294c57fcbc73`
- [Pinned Unitree source](https://github.com/unitreerobotics/unitree_ros2/tree/668d1ec5a05d1c38d3306bdca7d59f2ba3581a88)
  GitHub archive SHA256 `3e96da843c611878e28148ce113d978e5464dec922fc06dcd8c340f8232a6883`.

Ubuntu packages were retrieved from `https://archive.ubuntu.com/ubuntu/` using
Focal main/universe indexes, extracted using `dpkg-deb -x`, and compared with
index SHA256 values:

| Package | Version | SHA256 |
|---|---|---|
| libssl1.1 | 1.1.1f-1ubuntu2 | 09ee28588a1fb5613ddc6c26a992d5a76931b3cf22c022930da413a5e580599e |
| libyaml-0-2 | 0.2.2-1 | 5a31dfa3bf44863cb13a323e3dd873bef7d8a7fe3bbda708bab697ddd620e67f |
| libyaml-dev | 0.2.2-1 | 6dec711322490e2746bacc98047fed6581e716c6d28f8e9577d8c906a110e335 |
| libconsole-bridge0.4 | 0.4.4+dfsg-1build1 | 8dcd22cf2f42aaaf017dcb0d6d7641d0a0867699e5fc25e9870be21af4e33f00 |
| libspdlog1 | 1:1.5.0-1 | e6dd3c62a2d34e362fa5cd2e0fbf97c97cd48390ec4092af270c8368dc5da479 |
| libtinyxml2-6a | 7.0.0+dfsg-1build1 | 7a433dcbd40e86cd8741a204c99466f3f9b3a43ac102790eddb4b244171a939a |

These old ABI packages are for this isolated compatibility probe only, not a
recommendation to install obsolete packages on a production host. Foxy is EOL.

Build tools in the Python3.8 workspace venv: CMake3.31.10, Ninja1.13.2,
colcon-core0.21.3/colcon-common-extensions0.3.0, empy3.3.4, catkin-pkg1.1.0,
lark-parser0.12.0, setuptools75.3.4, NumPy1.24.4.

Sequence used (paths represented by variables rather than machine-specific names):

1. Extract final Foxy archive to `$R/ros2-linux`; extract Ubuntu packages to
   `$R/deps`; source `ros2-linux/setup.bash` with the Python3.8 venv first in PATH.
2. Set LD_LIBRARY_PATH to extracted deps `usr/lib/x86_64-linux-gnu`, standalone
   Python3.8 `lib`, and Foxy `lib`; set ROS_HOME/ROS_LOG_DIR to writable workspace.
3. Copy only pinned `cyclonedds_ws/src/unitree/unitree_go` to `$R/ws/src`.
4. The binary archive's fastrtps CMake export hardcoded
   `/usr/lib/x86_64-linux-gnu/libtinyxml2.so`. Relocate that one export path to the
   extracted package's `libtinyxml2.so.6`; add a workspace-only `libtinyxml2.so`
   symlink and put the extracted library directory in LIBRARY_PATH. This changes
   only the local imported build metadata, not Unitree message source.
5. `colcon build --packages-select unitree_go --cmake-args -DBUILD_TESTING=OFF`
   plus explicit `PYTHON_EXECUTABLE`, `PYTHON_LIBRARY`, `PYTHON_INCLUDE_DIR` for
   the standalone Python3.8 runtime. First attempts failed for missing Python
   development paths and the hardcoded TinyXML path; the final build succeeded.
6. Source `$R/ws/install/setup.bash` and run the serialization check above.

Do not treat this ad-hoc rootless mix as an Ubuntu/ARM Docker build verification.
Logs, binaries, packages and experiment outputs remain outside the source repo.

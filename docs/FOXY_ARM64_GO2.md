# Ubuntu 20.04 / ROS 2 Foxy / ARM64 — Go2 EDU staging

## Scope and verified limits

Target supplied: **Go2 EDU, 16 GB**, matching the **Jetson Orin NX16GB** option
in the Unitree module documentation selected by the user. User reports Ubuntu20.04 / JetPack5.1. The selected Unitree page lists a JetPack5.1.1 factory image, so the exact
installed L4T patch, camera interfaces, firmware and actual NIC still need verification. Do not infer Jetson from RAM size.
The default image is CPU-only and does not install CUDA or use the NVIDIA
container runtime; an explicit optional Jetson profile is documented below.

This is a **read-only robot deployment foundation**, NOT completed autonomous
navigation on Go2. Original HSGM source is preserved, with Python 3.8 compatibility
fixes. Its Habitat camera/pose/action loop is not a real-robot driver. The new ROS
package only subscribes to official Go2 telemetry and writes a local status file.
It never publishes cmd_vel, sport requests or low-level motor commands, changes
sport mode, stands up, or arms the robot. DDS discovery and Foxy parameter-event traffic still occur.

No real robot, ARM64 runtime, Docker engine or hardware acceleration was
available in the cloud authoring environment. A workspace-local real
CPython3.8.20 x86_64 interpreter was later installed for offline verification. Docker build and robot
communication remain **unverified**. Do not describe the Dockerfile as a built
image. No model/API calls, data downloads, or actuation tests were performed.

Foxy reached EOL on June 20, 2023. This version is retained as requested; keep the
robot network isolated and do not expose DDS to the Internet. Ubuntu20.04 ordinary
support is also no longer a current maintenance baseline; discuss maintained
deployment options before production use.

## Build and inspect on a Docker host

Use Docker Engine with Buildx and Compose v2. A native ARM64 Linux builder is
preferred; an x86 builder needs working ARM64 emulation, configured separately.
The image build fetches only public middleware sources and apt packages.

```bash
docker buildx build --platform linux/arm64 --load -t hsgmv3:foxy-arm64 .
docker image inspect hsgmv3:foxy-arm64 --format '{{.Os}}/{{.Architecture}}'
# Must print linux/arm64.
docker run --rm hsgmv3:foxy-arm64 bash -c \
  'cat /etc/os-release; python3 --version; echo "$ROS_DISTRO"; ros2 interface show unitree_go/msg/SportModeState'
docker run --rm hsgmv3:foxy-arm64 python3 scripts/check_repo.py --offline-tests
docker run --rm hsgmv3:foxy-arm64 python3 -m unittest discover -s tests -p 'test_python38_compat.py'
docker run --rm hsgmv3:foxy-arm64 python3 -m unittest discover -s tests -p 'test_foxy_container.py'
```

Builds intentionally reject non-ARM64 TARGETARCH. Base tag is
`ros:foxy-ros-base-focal`; resolve and record a validated ARM64 digest when a build
environment is available. Apt versions/base tag are not fully locked, so this is
not claimed byte-for-byte reproducible. Public source commits are pinned:

- Unitree ROS2: `668d1ec5a05d1c38d3306bdca7d59f2ba3581a88`
- CycloneDDS 0.10.2: `9995905bce6c4cf9f740d6438bbf7fcfd1c83dfd`
- rmw_cyclonedds Foxy: `c12abc56983204f1d91f2d839d394528c7b29b42`

Unitree's documented Foxy sequence is followed: build CycloneDDS before sourcing
ROS, then build rmw_cyclonedds_cpp and unitree_go against Foxy. The container does
not build or run upstream robot motion examples. Upstream licenses are retained
in the installed middleware/source package as supplied; HSGM LICENSE is unchanged.

## Safe default: isolated liveness check

```bash
docker compose config
docker compose up -d --build
docker compose logs --tail 20 monitor
docker compose exec monitor cat /tmp/hsgm-status.json
docker compose down
```

Default is loopback, ROS_LOCALHOST_ONLY=1, no published ports, no host networking,
nonroot UID10001, dropped capabilities, read-only filesystem and bounded tmpfs.
Monitor has a 1GB container memory ceiling, not a promise for full HSGM models.
Status distinguishes `robot_connected` from process health. Healthy only means a
recent monitor heartbeat; it does not prove robot connection or safe actuation.
No received telemetry gives robot_connected=false, not a fabricated pose.
Malformed/non-finite telemetry is ignored. Monitor logs at 0.5Hz.

## Explicit trusted robot LAN mode

Only after confirming the correct robot-facing NIC and network topology:

```bash
# Example variable only: replace with the actual existing interface.
export GO2_INTERFACE=eth0
export ROS_DOMAIN_ID=0
docker compose -f compose.yaml -f docker/compose.go2-lan.yaml config
docker compose -f compose.yaml -f docker/compose.go2-lan.yaml up -d
docker compose -f compose.yaml -f docker/compose.go2-lan.yaml logs --tail 20 monitor
docker compose -f compose.yaml -f docker/compose.go2-lan.yaml down
```

This opt-in Linux host-network mode bypasses Docker bridge isolation; interface
binding restricts DDS to the named NIC but is not an authentication boundary.
It does NOT alter host addresses, routes or firewall rules. No privileged mode,
device mounts, host PID/IPC or Docker socket are used. Domain0 matches official
Go2 examples, but verify robot configuration. No runtime cloud/API connection is
made by the monitor. Full VLM API use is a separate configured operation.

Default ROS state topic is `/lf/sportmodestate`, type
`unitree_go/msg/SportModeState`; `/sportmodestate` may be selected:
`ros2 run hsgm_go2_monitor monitor --ros-args -p state_topic:=/sportmodestate`.
These positions are reported as robot telemetry, not assumed Habitat world poses.

### Necessary network ports — not a guessed web port

The audited application has outbound model/API connections, but no inbound
HTTP server. Therefore no 8000/8080/5000 TCP service is exposed.

DDS uses UDP multicast/discovery/data. CycloneDDS default base is
`B=7400+250*ROS_DOMAIN_ID`; multicast ports B/B+1, participant-index unicast
B+10+2*PI and B+11+2*PI. At domain0 and PI0..9 that is UDP7400–7401 and
UDP7410–7429. This container caps automatic participant index search at 9.
That describes indexed ports, **not a complete firewall allowlist**: CycloneDDS
can also use anonymous UDP participant sockets, and robot-side participant
indices/configuration are independent. Inspect actual traffic and the vendor
configuration before a network administrator narrows rules. Do not open a wide
UDP range to the Internet. EXPOSE or Docker -p alone does not solve DDS multicast.
No firewall changes were made.

## Dependency audit and separation

| Component | Foxy robot image decision |
|---|---|
| Python | Ubuntu20.04 system3.8, ROS apt rclpy |
| NumPy/SciPy/OpenCV/Pillow/YAML | Ubuntu apt ARM64 packages; avoid incompatible pip ABI upgrades |
| HSGM Python APIs | removed str.removeprefix; postponed modern annotations; 3.8-compatible test extraction |
| Habitat-Sim/Habitat-Lab | not installed; original simulator retained for separate development machine |
| PyTorch/YOLOE/Open3D/quaternion | not installed in lean monitor; ARM64/JetPack wheel/model availability unverified |
| Mem0/sentence-transformers | existing markers require Python>=3.9; excluded from Foxy3.8 |
| VLM APIs | original code retained, not called by monitor; secrets never baked into image |

The original requirements.txt is a simulator research dependency list, NOT a
tested Foxy lockfile. Do not run pip install -r requirements.txt inside this ROS
runtime and assume it becomes a complete robot navigator. A CPU-only 16GB board
also cannot be assumed suitable for all segmentation/VLM models.

Next integration gates: confirm board/OS/firmware; build and execute ARM64 image;
verify read-only timestamps and coordinates; implement synchronized camera,
depth/point-cloud and odometry observations; define transforms; separately
review bounded motion execution, stale-sensor fail-closed logic, physical
emergency-stop and operator enable; then authorize controlled robot tests.
The delivered monitor intentionally has no flag that enables motion.

## Upstream references

- [Unitree ROS2 instructions](https://github.com/unitreerobotics/unitree_ros2)
- [Unitree SDK2 platform list](https://github.com/unitreerobotics/unitree_sdk2)
- [CycloneDDS0.10.2 port behavior](https://cyclonedds.io/docs/cyclonedds/0.10.2/config/cyclonedds_specifics.html#controlling-port-numbers)
- [Official Foxy final release/EOL](https://discourse.ros.org/t/new-packages-for-foxy-fitzroy-2023-06-20/32039)
- [Official Foxy platform announcement](https://www.openrobotics.org/blog/2020/6/5/ros-2-foxy-fitzroy-release)

## Optional Jetson profile (user reports Ubuntu20.04 / JetPack5.1)

NVIDIA's exact **JetPack5.1** release is **L4T35.2.1**, CUDA11.4,
TensorRT8.5.2 and cuDNN8.6. “5.1.x” can mean another L4T release; confirm the
actual host before choosing the image. The selected module documentation identifies the 16GB option as Orin NX.
The page now offers a JetPack5.1.1 factory image; do not assume it matches the
user's currently installed system. The following optional profile is ONLY for
verified L4T35.2.1. It must refuse a 35.3.1/other host pending a matching profile.
The base image below is officially listed by NVIDIA for JP5.1 and contains
PyTorch2.0.0. Do not install desktop CUDA wheels on top of it.

On the intended Jetson, with Docker and NVIDIA Container Runtime already installed:

```bash
export L4T_TAG=r35.2.1 # ONLY if /etc/nv_tegra_release says R35 REVISION2.1
bash scripts/check_jetson_host.sh
# Stop if release mismatches, nvidia runtime is absent, or board identity is unclear.
docker buildx build --platform linux/arm64 --load \
  -f docker/Dockerfile.jetson-base -t hsgm-jetson-base:r35.2.1 .
docker compose -f compose.yaml -f docker/compose.jetson.yaml build
# Explicit GPU allocation, but still loopback networking and no motion:
docker compose -f compose.yaml -f docker/compose.jetson.yaml up -d
docker compose -f compose.yaml -f docker/compose.jetson.yaml exec monitor \
  python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
docker compose -f compose.yaml -f docker/compose.jetson.yaml down
```

These commands are documentation, not results. No GPU test was run. Monitor
does not import Torch or use GPU, even in this profile. Full YOLOE/Habitat
dependencies and models are still excluded; optional GPU capability is not
equivalent to a validated robot perception/navigation stack. The GPU base is
larger than the CPU base; its own upstream licenses apply. NVIDIA container
runtime shares selected host GPU/driver resources, so use this profile only
on the intended trusted Jetson. Host release preflight does not install or
change drivers, Docker runtime, network settings or firewall.

- [NVIDIA JetPack5.1 release notes](https://docs.nvidia.com/jetson/jetpack/5.1/release-notes/index.html)
- [NVIDIA L4T PyTorch image/tag matrix](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch)

## User-selected module documentation

[Unitree expansion module configuration](https://support.unitree.com/home/zh/developer/module_update#heading-1),
read 2026-10-25: compares Orin Nano8GB vs Orin NX16GB, lists expansion-module
LAN address 192.168.123.18 and current factory image
`go2_nx_Jetpack5.1.1_20250930.img.bz2`. These are reference facts, not proof of
the attached host configuration. Do not overwrite the vendor SLAM/odometry
services or management HTTP/SSH ports. This container adds no competing web
server. No firmware update, flashing, disk restore, device-tree edit, or default
credential use is part of this task.

### Alternative for verified JetPack5.1.1 / L4T35.3.1

Do NOT use a guessed r35.3.1-pth2.0-py3 tag. NVIDIA documents the SDK image
`nvcr.io/nvidia/l4t-jetpack:r35.3.1`; build the same Foxy layer on that instead:

```bash
export L4T_TAG=r35.3.1 # ONLY after reading installed host release
bash scripts/check_jetson_host.sh
docker buildx build --platform linux/arm64 --load \
  -f docker/Dockerfile.jetson-base \
  --build-arg NVIDIA_BASE=nvcr.io/nvidia/l4t-jetpack:r35.3.1 \
  -t hsgm-jetson-base:r35.3.1 .
docker compose -f compose.yaml -f docker/compose.jetson.yaml build
```

This alternative contains NVIDIA's JetPack SDK, not an asserted working Torch
installation. The telemetry monitor does not need Torch. NVIDIA's corresponding
optional Python3.8/aarch64 Torch wheel is
`torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl` in its JPv511 directory.
Model inference integration must separately pin its complete dependencies and
test this wheel; do not replace it with generic PyPI CUDA Torch.
NumPy1.26 is incompatible with Python3.8 (use at most1.24.4).
No GPU build or wheel installation is claimed.

- [JetPack5.1.1 / L4T35.3.1](https://developer.nvidia.com/embedded/jetpack-sdk-511)
- [Official JetPack SDK container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-jetpack)
- [Official JPv511 PyTorch wheel directory](https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/)

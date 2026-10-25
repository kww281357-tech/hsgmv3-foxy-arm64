# Offline Go2 physics smoke (not a real robot controller)

Tested on CPython 3.8.20, Linux x86_64, MuJoCo 3.1.6. Version3.2.7 requires Python>=3.9 and was rejected during resolution. No ROS, DDS, Unitree SDK, or physical network connection is used. A Python audit hook blocks Python socket creation/connect/bind/DNS; it is not a native network sandbox. Native syscall tracing was attempted but denied by this sandbox (`PTRACE_TRACEME: Operation not permitted`). No bypass attempted.

Use a separate Python 3.8 venv. Save the following exact tested dependency pins to a local requirements file and install with `python -m pip install -r <file>` (these are simulation-only, not robot production dependencies):

```text
absl-py==2.3.1
etils==1.3.0
glfw==2.11.0
imageio==2.35.1
imageio-ffmpeg==0.5.1
importlib-resources==6.4.5
mujoco==3.1.6
numpy==1.24.4
pillow==10.4.0
pyopengl==3.1.10
setuptools==75.3.4
typing-extensions==4.13.2
zipp==3.20.2
```
 Download the official model outside the source repository:

```sh
git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git menagerie
git -C menagerie sparse-checkout set unitree_go2
git -C menagerie checkout f3b6013629c7b94f8ee93fd977b161fa8350f9bc
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LIBGL_ALWAYS_SOFTWARE=1 MUJOCO_GL=egl \
  python scripts/go2_physics_smoke.py --model-dir menagerie/unitree_go2 \
  --monitor-source ros2_ws/src/hsgm_go2_monitor/hsgm_go2_monitor/monitor.py \
  --output /path/outside/repository/evidence
```

Model: Google DeepMind MuJoCo Menagerie `unitree_go2`, simplified description derived from Unitree publicly released URDF. BSD-3-Clause, copyright Unitree2016–2022; preserve model LICENSE. External assets, output/video/logs are not committed. Model MJCF/OBJ hashes are emitted in report.json; commit is pinned above.

Rollout: two identical resets, 2500 steps each, dt0.002sec. Each reset raises the base8cm above model home pose; gravity causes a genuine settling/contact transient. In-process PD joint torques are bounded to model limits. Assert every step finite qpos/qvel/qacc/torques, normalized quaternion, height/velocity bounds; final four ground contacts and upright base; exact repeated per-step time/qpos/qvel trajectory equality; zero engine warnings. These are smoke bounds, NOT hardware safety limits or a deployable control policy.

XML SHA256 mismatch and Python -O execution are rejected before loading/stepping the model; these failure cases were executed.

125 samples/episode at25Hz feed the actual monitor.py telemetry_packet pure function extracted with AST to avoid importing ROS. Accepted250 samples; NaN rejection tested. This validates conversion against physics values, NOT ROS transport or timestamps on a physical robot. yaw_speed is qvel[5] (body angular z for this near-level stand), not a general Euler yaw derivative. JSONL timestamps reset between episodes; replay one episode or explicitly rebase.

Output: report.json, telemetry.jsonl, go2_physics_smoke.mp4 (first5-second episode), go2_physics_final.png. The final frame shows settling posture. No locomotion, obstacle avoidance, navigation, perception, ROS transport, ARM64 image, CUDA, Jetson, or realGo2 validation is implied.

## Validation observed on 2026-10-25

- Actual cloud host: Linux x86_64, CPython 3.8.20. This is **not** Ubuntu 20.04/ARM64 hardware validation.
- MuJoCo: two 5-second episodes, 2,500 steps each; all-step deterministic reset equality; final four ground contacts and base height 0.241725m. Video is the first episode, 640x480, H.264, 125 frames/25fps/5 seconds.
- Adapted monitor conversion: 250 actual physics samples accepted; invalid NaN rejected. Synthetic mode/error fields are zero, not a simulated Unitree firmware state machine.
- Pure lifecycle tests: initial disconnected → fresh → stale at 5 seconds → recovered; invalid telemetry does not refresh the last valid state; malformed vector rejected; liveness healthcheck rejects missing/corrupt/stale/future status. This test layer does not instantiate ROS.
- Existing guarded offline suite: 650 tests passed under real Python 3.8, no skips. Compatibility tests: 4 passed; packaging contracts: 4 passed; lifecycle tests: 4 passed.
- No motion command publisher/client was added to the production monitor. The physics PD controller exists only inside MuJoCo and is not a robot controller.
- No Docker engine, ARM64 runtime, CUDA/Jetson GPU, physical robot, full Habitat/VLM navigation, locomotion or obstacle avoidance was tested.

For the pure lifecycle tests run `python tests/test_monitor_lifecycle.py`.

## Real Foxy message layer

[Detailed Foxy runtime and reproduction](FOXY_SIMULATION_RUNTIME.md): built the
pinned Unitree `SportModeState` under actual Foxy/Python3.8 and round-tripped all
250 physical simulation records through real CDR serialization (236 bytes each).
Live DDS node creation failed at cloud-restricted network interface enumeration;
live publish/subscribe is **not passed**. The archive uses CycloneDDS0.7.0, so
this also does not verify the Dockerfile's pinned0.10.2 middleware build.

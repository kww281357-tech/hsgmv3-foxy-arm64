# Port validation — 2026-10-25

Source read from private jkffj54/hsgmv3 main:
3a0a6558a5a485bca7e0793461b9822060b392eb. Original remote was not modified.
Independent local repository initialized from the 114 text/source files only.

## Passed in authoring environment (Linux x86_64 / Python3.12)

- Original guarded offline suite before changes: 650 tests run, OK, 1 skipped.
- Port guarded offline suite after changes: 650 tests run, OK, 1 skipped.
- Python3.8 grammar/API checks: 4 tests passed (static, not real Python3.8).
- Docker/network/telemetry contract tests: 4 tests passed, no ROS/robot imports.
- Source syntax, shell syntax and git diff whitespace checks.
- Staged source-content/credential-pattern checks before commit.
- Independent review of Foxy Node API, Unitree SportModeState fields, pinned
  Cyclone version, no motion commands, and default network isolation.
- Third-party license retention added after review.

Old INITIAL_VALIDATION.md describes the 2026-10-19 snapshot; the newer current
650-test baseline supersedes that old test result for this port, not its history.
Existing two invalid escape sequence warnings remain unchanged.

## Not run / not established

- Docker build (no docker/podman/buildah engine available here).
- Ubuntu20.04 OS execution, colcon build and ROS middleware loading (standalone
  CPython3.8 x86_64 checks below do not establish these).
- Actual ARM64 image architecture (there is no built image yet).
- Container healthcheck against a live ROS process or Compose validation.
- DDS discovery, robot telemetry reception or any Go2 hardware integration.
- Jetson CUDA/PyTorch/TensorRT or model inference.
- Habitat runtime, online model calls or navigation simulation.
- Real-robot navigation, actuation or safety certification.
- GitHub Actions: checked after publication; no workflows/runs configured.
  No CI pass is claimed.

The artifact is source plus a Docker build recipe, not a docker-save image archive.

## Additional real Python3.8 verification

A workspace-local CPython3.8.20 x86_64 environment was subsequently installed
without Docker, GPU or host security changes. Running the downloaded remote
snapshot exposed two test-only Python3.9 dict-union expressions. They were
changed to equivalent dict-unpacking expressions.

Final modified source on actual CPython3.8.20: **650 tests run, all OK, no skips**.
The 4 compatibility tests and 4 container-contract tests also pass on Python3.8.
Offline dependencies: NumPy1.24.4, Pillow10.4.0, OpenCV-headless4.8.1.78,
SciPy1.10.1, PyYAML6.0.2. This is NOT an ARM64/Focal/ROS or Docker build result.

Private repository published to kww281357-tech/hsgmv3-foxy-arm64 under explicit
user authorization, branch feature/foxy-arm64; main remains the initial README.
All 134 files in the initial uploaded branch were downloaded and byte-compared:
no missing, extra or changed content. Original jkffj54/hsgmv3 main rechecked
unchanged at the source SHA above. No PR was created (separate approval blocked).
Browser-based upload may not preserve executable permissions; before enabling
the Git hook run `chmod +x .githooks/pre-commit`.

#!/usr/bin/env python3
"""Offline Go2 physics smoke. No SDK, ROS, DDS, socket, or physical robot control.
Requires an external BSD-licensed Menagerie unitree_go2 model; assets not vendored.
"""
import argparse
import ast
import math
from types import SimpleNamespace
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import sys

def deny_network(event, args):
    if event in ("socket.__new__", "socket.connect", "socket.bind", "socket.getaddrinfo"):
        raise RuntimeError("Network/socket use is forbidden in offline physics smoke")

sys.addaudithook(deny_network)

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
import mujoco
import numpy as np

MODEL_COMMIT = 'f3b6013629c7b94f8ee93fd977b161fa8350f9bc'
EXPECTED_XML = {'scene.xml': 'b56123ea2bf09070bf4054f0be9aa418e49041ddddce44dbbf7ab35e8b732641', 'go2.xml': '50adb09a4365293e2acdaf2010ae35a82b0f09fea18ae51806fef91e310ed04a'}

def require_finite(*arrays):
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError('non-finite simulation state or control')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--no-render', action='store_true')
    parser.add_argument('--monitor-source', type=Path, required=True)
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError('Do not run smoke assertions with Python -O')
    for name, expected in EXPECTED_XML.items():
        if hashlib.sha256((args.model_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Model XML differs from pinned upstream: ' + name)
    args.output.mkdir(parents=True, exist_ok=True)
    tree = ast.parse(args.monitor_source.read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'telemetry_packet')
    namespace = {'math': math}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(args.monitor_source), 'exec'), namespace)
    telemetry_packet = namespace['telemetry_packet']
    model = mujoco.MjModel.from_xml_path(str(args.model_dir / 'scene.xml'))
    assert model.nu == 12 and model.nq == 19 and model.nv == 18
    model.opt.timestep = .002
    data = mujoco.MjData(model)
    joint_ids = model.actuator_trnid[:, 0]
    qadr, dadr = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'home')
    assert home >= 0
    target = model.key_qpos[home, qadr].copy()
    renderer = None
    writer = None
    if not args.no_render:
        import imageio.v2 as imageio
        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0, 0, .22]
        camera.distance, camera.azimuth, camera.elevation = 1.5, 140, -20
        writer = imageio.get_writer(str(args.output / 'go2_physics_smoke.mp4'), fps=25, codec='libx264')
    rows, finals, trajectories = [], [], []
    start = time.monotonic()
    # Repeated identical rollout proves reset and deterministic integration.
    for episode in range(2):
        mujoco.mj_resetDataKeyframe(model, data, home)
        assert data.time == 0.0
        np.testing.assert_array_equal(data.qpos, model.key_qpos[home])
        trajectory = []
        data.qpos[2] += .08  # Actual gravity/contact settling, not static pose playback.
        for step in range(2500):  # 5 simulated seconds per episode at 500Hz.
            torque = 40 * (target - data.qpos[qadr]) - 2 * data.qvel[dadr]
            require_finite(torque, data.qpos, data.qvel)
            data.ctrl[:] = np.clip(torque, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            mujoco.mj_step(model, data)
            require_finite(data.qpos, data.qvel, data.qacc, data.ctrl)
            assert .05 < data.qpos[2] < .6, 'base height outside smoke bounds'
            assert np.max(np.abs(data.qvel)) < 100, 'velocity outside smoke bounds'
            assert np.all(data.ctrl >= model.actuator_ctrlrange[:, 0])
            assert np.all(data.ctrl <= model.actuator_ctrlrange[:, 1])
            assert abs(np.linalg.norm(data.qpos[3:7]) - 1) < 1e-6
            trajectory.append(np.concatenate([[data.time], data.qpos.copy(), data.qvel.copy()]))
            if step % 20 == 0:
                rows.append({'episode': episode, 'step': step + 1, 'time': float(data.time),
                             'position': data.qpos[:3].tolist(), 'velocity': data.qvel[:3].tolist(),
                             'yaw_speed': float(data.qvel[5]), 'mode': 0, 'error_code': 0,
                             'joint_position': data.qpos[qadr].tolist(), 'contacts': int(data.ncon)})
                packet = telemetry_packet(SimpleNamespace(**rows[-1]))
                assert packet['position'] == rows[-1]['position']
                if episode == 0 and renderer:
                    renderer.update_scene(data, camera=camera)
                    frame = renderer.render()
                    writer.append_data(frame)
                    if step == 2480:
                        imageio.imwrite(args.output / 'go2_physics_final.png', frame)
        assert abs(data.time - 5.0) < 1e-9
        assert data.ncon >= 4, 'expected ground contact after settling'
        assert data.qpos[2] > .15, 'not standing after settling'
        assert sum(int(w.number) for w in data.warning) == 0, 'MuJoCo warnings'
        trajectories.append(np.asarray(trajectory))
        finals.append(np.concatenate([data.qpos.copy(), data.qvel.copy()]))
    np.testing.assert_array_equal(finals[0], finals[1])
    np.testing.assert_array_equal(trajectories[0], trajectories[1])
    # Invalid input guard is actually executed; never inject NaN into engine.
    try:
        require_finite(np.array([np.nan]))
    except ValueError:
        invalid_rejected = True
    else:
        raise AssertionError('NaN guard did not reject')
    invalid_telemetry = dict(rows[0], position=[float('nan'), 0, 0])
    try:
        telemetry_packet(SimpleNamespace(**invalid_telemetry))
    except ValueError:
        pass
    else:
        raise AssertionError('monitor did not reject NaN')
    if writer:
        writer.close()
        renderer.close()
    with (args.output / 'telemetry.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    hashes = {str(p.relative_to(args.model_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(args.model_dir.rglob('*')) if p.is_file() and p.suffix in ('.xml', '.obj')}
    report = {'status': 'passed', 'python': platform.python_version(), 'architecture': platform.machine(),
              'mujoco': mujoco.__version__, 'model_source': 'https://github.com/google-deepmind/mujoco_menagerie',
              'model_commit': MODEL_COMMIT, 'model_license': 'BSD-3-Clause (Unitree Go2)',
              'model_file_sha256': hashes, 'episodes': 2, 'steps_per_episode': 2500,
              'simulated_seconds_per_episode': 5, 'timestep': .002,
              'final_position': data.qpos[:3].tolist(), 'final_contacts': int(data.ncon),
              'monitor_source_sha256': hashlib.sha256(args.monitor_source.read_bytes()).hexdigest(),
              'yaw_speed_semantics': 'qvel[5], body-frame angular z approximation for this near-level stand; not general Euler yaw derivative',
              'monitor_accepted_physics_samples': len(rows), 'monitor_nan_rejected': True,
              'reset_identical_rollouts': True, 'invalid_input_rejected': invalid_rejected,
              'wall_seconds': time.monotonic() - start,
              'network': 'Python audit hook rejects socket creation/connect/bind/DNS; no ROS/DDS/SDK imports. Native syscall tracing unavailable (sandbox ptrace denied).' ,
              'rendering': None if args.no_render else 'EGL software requested; 640x480 25fps',
              'limitations': ['Not locomotion or navigation validation', 'Not ROS integration',
                              'Not ARM64 Docker or Jetson GPU validation', 'Simplified Go2 model, not EDU-specific sensors']}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'model_file_sha256'}, indent=2))

if __name__ == '__main__':
    main()

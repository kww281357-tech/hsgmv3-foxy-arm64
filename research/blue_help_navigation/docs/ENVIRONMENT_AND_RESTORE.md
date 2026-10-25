# Isolated environments and recovery

Keep this research interpreter separate from ROS Foxy / Go2 and from character
modelling or the YOLO detector environment. Importing this package does not start
Habitat, a model, browser, download, robot or provider request.

## Tested Habitat CPU environment

Linux x86_64, Python3.9.23, NumPy1.26.4, Pillow11.3.0; official Habitat-Sim0.3.3,
`py3.9_headless_bullet_linux_acbe6f4922e68145e401e55c30f9dfea460a3f24`.
Mesa llvmpipe: EGL1.5 / OpenGL4.5, software rendering, no GPU allocation/CUDA.
`habitat-cpu-environment.yml` is the human-readable recipe; the explicit Linux
lock includes exact upstream package URLs and SHA256. No binary environment or
package cache is included in Git. Conda/micromamba must support the lock format;
inspect/validate package hashes when restoring. Do not blindly upgrade Habitat.

Example on an authorized new Linux server with conda/micromamba already available:

```bash
micromamba create -p /private/envs/blue-help -f docs/habitat-cpu-explicit-linux-64.txt
# Activate the created environment using your shell's normal micromamba setup.
python -m pip install --no-deps --no-build-isolation -e .
source scripts/cpu_render_env.sh
python -c 'import habitat_sim; print(habitat_sim.__version__)'
python -m unittest discover -s tests -v
```

The explicit conda lock does not claim to include pip-only Pillow. If Pillow is
missing, install `Pillow==11.3.0` into this environment only. Recipe/lock restore
still requires compatible OS drivers/libraries and network access; it is not a
portable frozen binary or a promise that no configuration will ever be needed.
The shell helper only sets software-render environment options and workspace
cache paths; it does not install or modify system software.

Official dependency reference: https://github.com/facebookresearch/habitat-sim
Pinned source: https://github.com/facebookresearch/habitat-sim/tree/acbe6f4922e68145e401e55c30f9dfea460a3f24
CPU renderer: https://docs.mesa3d.org/drivers/llvmpipe.html

## Detector is separate and optional

The earlier six-frame check used Python3.12.14, Ultralytics8.4.0,
Torch2.9.0+cpu, Torchvision0.24.0+cpu. Official YOLO26n weight hash:
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
Source: https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
All80 actual classes matched COCO80; task=detect, head=Detect, end2end=true.
Six historical frames are not full-route detector coverage or live integration.
Weights and package environment are not included. Review their separate licensing
before installation/distribution. Do not install this stack into Habitat/Go2.

## What must be preserved privately

1. Git source / commit and this source manifest.
2. Licensed scene and20-episode subset + original300-record source checksums.
3. Frozen manual v2 evidence/route card + staged placement report.
4. Per-attempt results, including failures, raw RGB and private execution logs.
5. Environment recipes/locks and a short verification report on the new server.

Keep secrets out of all five. Use only an explicitly authorized private archive.
A source upload is not proof that scene data, result files or credentials were
backed up. Obtain those inputs by their lawful original route if unavailable;
do not bypass prior Library/Drive permission denials. Retest data hashes, unit
suite and one CPU fixture on the new server before any model/GPU experiment.

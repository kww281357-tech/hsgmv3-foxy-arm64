# Verified status — source consolidation v0.4.1

Snapshot date2026-10-25. This file distinguishes already observed results from
future work. It is not a navigation score sheet.

| Check | Result / evidence boundary |
|---|---|
| Consolidated unit suite, Python3.9.23 |317 tests passed, no skips; no model API |
| Same suite, Python3.12.14 |317 tests,316 passed,1 expected skip: actual Magnum is installed only in the isolated Habitat environment |
| Wheel build and fresh target install |Passed; imported from outside source directory;3 prompts, external pinned GAME asset and external pinned review card loaded |
| Organized GAME entrypoint, real Habitat CPU + fake responses |Passed;1 valid full answer,0 m movement,0 authenticated API requests, public/private checks passed |
| Full original reviewed route |Earlier v2 privileged CPU survey:18.938684981 m,192 forward steps,235 actions,47 RGB frames; no collision/proxy rejection recorded |
| Added-person collision |Actual GAME v7 CPU microcheck:0.85 m eligible,0.25 m forward to0.60 m, next forward blocked; turn-away ineligible. Not full-scene topology or general safety proof |
| Six-frame YOLO26n check |Earlier actual isolated CPU inference; actual80 classes verified, duplicate/class-conflict/weak outputs retained; not live integration |
| Real visual-policy API probe |NOT RUN; Extra Usage OFF account check pending |
| Real VLM following the answer |NOT RUN |
|20-episode paired performance / success / SPL |NOT RUN; no fabricated statistics |
|Go2 / ROS / physical robot integration |NOT IMPLEMENTED by this module; separate project |

## Changes from the frozen working versions

- One installable package and lazy CLI; explicit private input paths instead of
  sibling workspace path injection.
- One delivery controller and one runner. The validated8192-character whole-answer
  cap is integrated directly, retaining atomic counting and all offer semantics.
- Only current hand-authored full-route English prompts are loaded at runtime.
- Frozen review data is authenticated, but external Python is never executed.
  Contract code is packaged and byte-checked against the reviewed version.
- Portable offline survey command retains actual simulation movement; no world
  coordinates or images embedded in source. Its refactored command has mock/CLI
  tests; GAME full-route runtime results are recorded separately below.
- Staged starts invalidate both metric/evaluation score views, so generic
  aggregators cannot silently treat the staged trial as a valid benchmark.
- Probe content must be manually compared with its image before a live trial.
- Source excludes raw scene data, private route cards,20 scene-specific review
  cards/instruction records, images/results, weights, environments and credentials.
  Those remain in their original private artifacts, not silently discarded.

## Frozen predecessors (unchanged)

- Independent real-CPU v1 milestone ZIP SHA256:
  `c06b4ca2b702024e77a04f3f7fe4058135744ec687131cbb481e6a84e24f0a3a`.
- Complete manual route v2 ZIP SHA256:
  `8cb050bab2c51c0b97f51e9d8c161461ad2e5b013c8104655f2a27621fdd8ced`.
- Single-trial v3 preflight ZIP SHA256:
  `48223bfaeb172b7cdae9c280909c57a76602c34d62c2323ee03216b6fa1e80b0`.

The Git source publication is separate from these private evidence archives.
It does not imply raw data or result backups were newly uploaded.

## Existing GAME v7 integration (no new model creation)

Default trial/survey profile changed to game-v7 with an explicitly supplied,
SHA256-pinned existing GLB. The original model/config remain unchanged. Derived
runtime configuration and a goal-independent facing correction are separate.
Measured height1.513286 m, radius0.310802 m, feet0 fit the retained .32×1.6 proxy.
Current-instance visibility replaces the blue-color requirement, still guarded
by genuine LOS/floor/1 m/path checks. One microcheck answer consumed exactly1;
duplicate callbacks consumed0. This does not prove autonomous navigation.

### GAME full-route and final integration evidence

The organized GAME CLI actually completed the entire episode40 privileged survey:
18.938684981 m,192 forward moves,235 total actions,47 RGB frames,0 collisions,
0 human-proxy rejections, endpoint numerical error1.49e-8 m. No pose teleport
occurred after initial reset. This confirms this one route with this fixed actor
placement and collision profile, not all20 episodes or universal topology safety.
API0 and no autonomous policy were involved.

Final GAME full-answer fixture also passed in the real CPU simulator:1 delivered
answer,0 m travel,API0,goal-input isolation passed. Both success/SPL representations
are null and invalid for benchmarking. Actual AABB/clearance audit is recorded in
private outputs. These checks do not resolve the pending real VLM/billing gate.

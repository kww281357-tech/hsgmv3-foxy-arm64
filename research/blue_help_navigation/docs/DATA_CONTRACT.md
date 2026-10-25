# Private input contract

No automatic dataset download, invented world geometry, or missing-input mock fallback.
The following files must be supplied by an authorized owner outside the Git tree.

## 1. `--data-root`

```text
<root>/
  data/scene_datasets/mp3d/zsNo4HB9uLZ/
    zsNo4HB9uLZ.glb
    zsNo4HB9uLZ.navmesh
    zsNo4HB9uLZ.house          # retained provenance; not needed by this trial
    zsNo4HB9uLZ_semantic.ply   # retained provenance; not needed by this trial
  data/datasets/R2R_VLNCE_v1-3/val_unseen/
    zsNo4HB9uLZ_demo20.json
  examples/
    dummies.zsNo4HB9uLZ_demo20.json
```

Exact required hashes are in `blue_help/data.py`; `validate-data` checks all four
runtime files, episode order, normalized rotations and the identical historical
15-person layout across episodes. Validation does not establish scene rendering,
placement safety or navigation performance. Keep the 300-record source separately.
The 20 were selected before outcomes, seed 42; 18 distinct trajectories. Episode40
is the first fixed ID, not selected because it succeeds. Split is **val_unseen**.

The sidecar is historical data, not an endorsement of all 15 placements. The
current staged trial uses one explicitly reviewed goal-independent relocation.
Original 0.10 m agent-radius NavMesh is retained; the previously tried 0.18 m
alternative does not replace it. Current trial: height1.5 m, radius0.10 m,
forward0.25 m, turn15 degrees, sliding explicitly true. Cross-profile comparisons
must not be pooled as if they used identical geometry.

## 2. `--staging-report`

The previously verified private JSON report contains `staged_view` with:

- `dummy_id`, `dummy_ground_position` (XYZ metres),
- `camera_ground_position`, `camera_rotation_xyzw`,
- `placement_kind = explicit_goal_independent_demo_relocation` for the relocated helper.

Do not replace these values with invented coordinates. The trial cross-checks
helper/card geometry and the frozen episode goal; the adapter checks actual
placement, visibility, raycast, safe joining path and collision sweep. The report
hash is recorded in private run outputs. The CLI does not promote these private
coordinates into policy input.

## 3. `--review-root`

Use the frozen **BlueHelp_ManualFullRoute_Episode40_v2_20261025.zip** extracted
as `blue_help_manual_route_v2` (ZIP SHA256
`8cb050bab2c51c0b97f51e9d8c161461ad2e5b013c8104655f2a27621fdd8ced`).
It contains MANIFEST.json, contracts/route_card.py, route_card.private.json,
episode40_route_body_en.txt and the complete 47-frame private review evidence.
The bridge pins the manifest/card hashes and recomputes content-derived IDs,
then verifies every referenced image. It uses the package's own contract source;
the external Python file is read only for byte equality and never executed.

The runtime helper origin is a verified safe exterior anchor next to the person,
not the body's centre. The stored route goes anchor→annotated destination. Only
a short safe vehicle→anchor connector is queried at delivery time. A connector
mismatch, wrong scene/card, old observation step, unsafe route, missing evidence
or stale geometry returns no answer and consumes no successful-answer allowance.

Manual labels remain manual; the subsequent six-frame YOLO26 check is separate.
Nothing in a checksum proves a label or prose is factually correct. Human review
and, where available, actual detector output remain necessary.

## Output privacy

Each attempt needs a fresh output directory. Keep `*.private.json`, raw images,
model context, user instructions, trajectories and detector outputs private and
outside Git. They can be archived in the user's approved private storage, but
must not be silently uploaded or made public. The source package does not include
these files or reconstruct them from a restricted/denied artifact.

## 4. `--helper-asset` (default game-v7)

Provide the existing private GAME v7 `adult_neutral.glb`, SHA256
`d51934f336e4b9b91b64de4b84f4f2296f8ec01ccfbce108520322758c240558`.
No model download is performed and the25MB asset is not part of source Git.
The original config is left untouched; a separate runtime config is generated.
See GAME_HELPER.md for verified dimensions, recognition limits and collision
semantics. `--helper-profile blue-reference` is the only explicit blue fallback.
The external frozen route card remains the reviewed historical card: its fixed
anchor, full text and conservative collision envelope are not silently changed
when swapping the enclosed visual mesh. Current GAME frames are separate evidence.

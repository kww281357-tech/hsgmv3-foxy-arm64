# Existing GAME v7 helper — no new character generation

User selected an existing game character instead of the blue reference dummy.
This version uses the unchanged **adult_neutral.glb from GAME v7**. It does not
switch to HumanNOVA, the unrelated original cartoon, or a newly generated person.
Only neutral is selected for the navigation demo; no expression animation is
claimed. The existing smile/cry assets are not silently mixed with other versions.

## External asset contract

Supply `--helper-asset /private/game_character_v7/adult_neutral.glb` to `trial` or
`survey`. Their default profile is `game-v7`. The exact SHA256 is:

`d51934f336e4b9b91b64de4b84f4f2296f8ec01ccfbce108520322758c240558`

Missing/wrong assets fail closed, never automatically fall back to blue. For an
explicit historical comparison, select `--helper-profile blue-reference`; the
small original MIT OBJ/MTL remains packaged. No network download occurs.

The visible imported character front is local +Z. Actor yaw is explicitly set
to face the fixed safe anchor (not the hidden goal and not a moving robot); the
source import-axis setting remains front-Z. Front/back control images verified
this distinction.

The existing game config is **noncollidable**. We do not overwrite it. The
runtime writes a separate derived template beside private attempt outputs with
mass1, upY/front-Z, metres/scale1/COM0 and the validated local GLB as both render
and collision asset. The environment makes the actor collidable for genuine
Bullet raycasts and then STATIC after setting/reading back its pose. Agent
blocking is independently enforced by a swept conservative body proxy; a static
collision mesh or pretty rendering alone is not a safety guarantee.

## Measured geometry and preserved safety envelope

Actual Habitat CPU readback for v7 neutral:

- Local AABB min[-0.246157,0,-0.118605], max[0.244838,1.513286,0.189749] m.
- Maximum horizontal AABB-corner radius0.310802 m; height1.513286 m.
- Feet-origin offset0 m, rather than blue OBJ's0.8 m translation offset.
- Keep the previous conservative occupancy cylinder **radius0.32 m / height1.6 m**
  plus the unchanged0.03 m movement margin. The adapter rejects actual geometry
  outside that envelope; do not shrink it merely to make a placement pass.

These are asset-local dimensions, not hidden goal/scene coordinates. The fixed
helper foot and safe route anchor remain in external private data. Keeping the
same enclosing proxy preserves the existing conservative path-check geometry;
we still recheck live placement/contact/raycast/connector. The old frozen route
card and47 images remain untouched; a blue-helper historical image is not passed
off as a newly rendered game-helper frame.

## Recognition and eligibility

GAME clothes are not blue. The game profile uses **currently rendered simulator
instance pixels** (minimum8) rather than the old blue-color test. This is explicitly
simulator-assisted helper recognition, NOT YOLO person detection or real-robot
perception validation. Missing/hidden/wrong instance pixels fail closed. Current
true raycast LOS, same-floor,1 m foot-centre distance, safe short NavMesh approach,
consent/cooldown and atomic3-answer budget all remain separate requirements.
Semantic maps, object IDs and world coordinates do not become policy inputs;
only allowed helper aliases and eligibility records do. Raw RGB stays unpainted.

## Actual microcheck boundary

With the unchanged reviewed staging position and original0.10 m agent radius:
0.85 m starts eligible; one actual0.25 m move reaches0.60 m; the next attempted
forward step is blocked by the proxy with0 m displacement. Turning away removes
eligibility. One synthetic valid answer is counted once, duplicate delivery is
free,2 successful-answer opportunities remain. Source GLB/config hashes are
unchanged. API0, no VLM decision, no general full-scene navigability claim.

The new default package is separately tested; full-route survey/fixture status
is recorded in STATUS.md and VERIFICATION_SUMMARY.json, not inferred from this
microcheck. Original attempt01 evidence retains its end-of-report JSON failure
on STATIC effective mass=infinity; attempt02 records that value honestly as a
static-body property and completed successfully. Configured positive mass is1.

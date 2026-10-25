"""Privileged OFFLINE route survey; not a navigation policy or benchmark.

Run ``python -m blue_help survey --data-root DATA --staging-report STAGE.json
--output NEW_PRIVATE_DIRECTORY``. All three locations are explicit inputs.
The frozen episode-40 goal comes only from the checked dataset; the helper and
starting anchor come from an external staging report. No private coordinates,
scene images, workspace roots, model client or downloaded data are bundled here.

Every translation/rotation after the single reset uses actual ``env.step`` and
the installed NavMesh/human-body movement filter. Variable precise turn angles
and <= 0.10 m steps are survey controls, NOT official R2RCE action primitives.
PNG/depth/pose/action/goal evidence is written only to the fresh private output.
Media/contact-sheet assembly is intentionally separate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import time

from ..contracts import DummySpec, Episode
from ..data import (DUMMIES_REL, EPISODES_REL, NAVMESH_REL, SCENE_REL,
                    finite_vector, load_dummies, load_episodes, validate_frozen_data)

PINNED_EPISODE_ID = "40"
SURVEY_SEED = 42
MAX_FORWARD_M = 0.10
CORNER_TOLERANCE_M = 0.012
ENDPOINT_TOLERANCE_M = 0.05


@dataclass(frozen=True)
class SurveyInputs:
    data_root: Path
    staging_report: Path
    episode: Episode
    staged_episode: Episode
    dummy: DummySpec
    frozen_validation: dict
    staging_sha256: str
    placement_kind: str


def _strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate staging JSON key")
            result[key] = value
        return result
    def bad_constant(value):
        raise ValueError("Nonfinite staging JSON value")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad_constant)


def load_survey_inputs(data_root, staging_report) -> SurveyInputs:
    """Verify the pinned dataset before reading any privileged route inputs.

    There is no CLI bypass for checksum/order checks. A changed helper placement
    must be explicitly labelled as an audited goal-independent demo relocation;
    no failed point is silently snapped or moved by this loader.
    """
    root, stage_path = Path(data_root).resolve(), Path(staging_report).resolve()
    validation = validate_frozen_data(root)
    episodes = load_episodes(root / EPISODES_REL)
    if not episodes or episodes[0].episode_id != PINNED_EPISODE_ID:
        raise ValueError("Survey is pinned to the first frozen episode: 40")
    episode = episodes[0]
    contents = stage_path.read_bytes()
    document = _strict_json(contents)
    stage = document["staged_view"]
    if not isinstance(stage, dict):
        raise ValueError("External staging report must contain a staged_view object")
    if ("episode_id" in document and str(document["episode_id"]) != episode.episode_id):
        raise ValueError("Staging report is bound to a different episode")
    if "scene_id" in document:
        if Path(str(document["scene_id"])).stem != Path(episode.scene_id).stem:
            raise ValueError("Staging report is bound to a different scene")
    choices = [d for d in load_dummies(root / DUMMIES_REL, episode)
               if d.dummy_id == stage.get("dummy_id")]
    if len(choices) != 1:
        raise ValueError("Staged dummy must identify exactly one frozen-sidecar helper")
    original_dummy = choices[0]
    foot = finite_vector(stage["dummy_ground_position"])
    anchor = finite_vector(stage["camera_ground_position"])
    rotation = finite_vector(stage["camera_rotation_xyzw"], 4)
    if abs(sum(x*x for x in rotation)-1) > 1e-4:
        raise ValueError("Staging rotation must be a unit XYZW quaternion")
    if abs(rotation[0]) > 1e-5 or abs(rotation[2]) > 1e-5:
        raise ValueError("The level-camera survey requires a yaw-only anchor rotation")
    yaw = stage.get("dummy_yaw_rad", original_dummy.yaw_rad)
    if type(yaw) not in (float, int) or not math.isfinite(yaw):
        raise ValueError("Invalid staged dummy yaw")
    placement_kind = stage.get("placement_kind", "unchanged_historical_sidecar")
    if ((foot != original_dummy.ground_position or float(yaw) != original_dummy.yaw_rad)
            and placement_kind != "explicit_goal_independent_demo_relocation"):
        raise ValueError("Changed helper pose requires explicit goal-independent relocation metadata")
    dummy = replace(original_dummy, ground_position=foot, yaw_rad=float(yaw))
    staged_episode = replace(episode, start_position=anchor, start_rotation_xyzw=rotation)
    return SurveyInputs(root, stage_path, episode, staged_episode, dummy, validation,
                        hashlib.sha256(contents).hexdigest(), str(placement_kind))


def packaged_object_config() -> Path:
    """Installed package asset directory, not a sibling checkout/workspace path."""
    asset = Path(__file__).resolve().parents[1] / "assets" / "blue_dummy" / "light_blue_human.object_config.json"
    if not asset.is_file():
        raise FileNotFoundError("Packaged blue dummy object configuration is missing")
    return asset


def _write_json(path, value):
    # Checkpoints replace only this attempt's own files, never prior evidence.
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _wrap(angle):
    return (angle + math.pi) % (2*math.pi) - math.pi


def run_survey(inputs: SurveyInputs, output, *, max_actions=10000,
               environment_factory=None, shortest_path_factory=None,
               helper_profile='game-v7', helper_asset=None) -> dict:
    """Walk and record the entire episode-40 route using actual step calls.

    Dependency factories exist for tests; the CLI always uses the real lazy
    Habitat factories. There is no automatic mock fallback. Errors leave partial
    private evidence with walk_complete=false and close the owned environment.
    """
    if type(max_actions) is not int or max_actions <= 0:
        raise ValueError("max_actions must be a positive integer")
    destination = Path(output).resolve()
    if destination == inputs.data_root or inputs.data_root in destination.parents:
        raise ValueError("Private survey outputs must not modify the frozen data root")
    if destination.exists() or Path(output).is_symlink():
        raise ValueError("Evidence already exists; choose a fresh attempt directory")
    if (inputs.episode.episode_id != PINNED_EPISODE_ID
            or inputs.staged_episode.goal_position != inputs.episode.goal_position):
        raise ValueError("Pinned episode/annotated goal changed")
    from ..habitat_adapter import AdapterConfig, create_environment
    from PIL import Image
    import numpy as np

    config = AdapterConfig(agent_radius_m=.10, turn_degrees=15, allow_sliding=True,
                           width=640, height=480, forward_step_m=MAX_FORWARD_M)
    destination.mkdir(parents=True, exist_ok=False)
    from ..helper_assets import prepare_helper, replace_dummy_for_helper, orient_game_helper_to_anchor
    helper = prepare_helper(helper_profile, helper_asset, destination/'helper_runtime')
    dummy = replace_dummy_for_helper(inputs.dummy, helper)
    if helper.profile == "game-v7":
        dummy = orient_game_helper_to_anchor(dummy, inputs.staged_episode.start_position)
    inputs = replace(inputs, dummy=dummy)
    config = replace(config, helper_visibility_mode=helper.visibility_mode)
    object_config = helper.config_path
    environment_factory = environment_factory or create_environment
    rows, actions = [], []
    travelled = 0.0
    frame, env = None, None
    started = time.monotonic()
    report = {
        "kind": "privileged_offline_oracle_survey",
        "episode_id": inputs.episode.episode_id,
        "scene_id": inputs.episode.scene_id,
        "selection": "First episode in pre-frozen sample; not selected for success or landmark richness",
        "instruction": inputs.episode.instruction,
        "annotated_goal": inputs.episode.goal_position,
        "original_episode_start": inputs.episode.start_position,
        "route_origin": inputs.staged_episode.start_position,
        "dummy_ground": inputs.dummy.ground_position,
        "dummy_placement_kind": inputs.placement_kind,
        "origin_kind": "Explicit external staged exterior anchor; eligibility rechecked at runtime",
        "staging_sha256": inputs.staging_sha256,
        "frozen_data_validation": inputs.frozen_validation,
        "object_config_sha256": hashlib.sha256(object_config.read_bytes()).hexdigest(),
        "adapter_config": asdict(config),
        "motion_profile": "Variable exact turns and <=0.10m actual forward actions; NOT official R2RCE primitives",
        "camera_height_m": config.camera_height_m,
        "seed": SURVEY_SEED,
        "max_actions": max_actions,
        "walk_complete": False,
        "no_teleports_after_reset": True,
        "policy_executed": False,
        "goal_used_by_survey": True,
        "goal_used_by_policy": False,
        "is_benchmark_result": False,
        "model_api_calls": 0,
        "helper_asset": helper.provenance,
        "detector_run": False,
    }

    def act(name, amount):
        nonlocal frame, travelled
        if len(actions) >= max_actions:
            raise RuntimeError("Survey action budget exceeded")
        if name not in ("move_forward", "turn_left", "turn_right"):
            raise ValueError("Unsupported survey action")
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("Invalid survey action magnitude")
        if name == "move_forward" and amount > MAX_FORWARD_M + 1e-9:
            raise ValueError("Survey forward action exceeds bounded step")
        before = frame.pose
        # Same actual action-space adjustment as the reviewed survey source.
        # This changes primitive magnitude only, never pose/state/transforms.
        env._agent.agent_config.action_space[name].actuation.amount = float(amount)
        frame = env.step(name)
        delta = math.dist(before.position, frame.pose.position)
        travelled += delta
        proxy_rejected = bool(env._proxy_collision)
        actions.append({"step": frame.step, "action": name, "amount": float(amount),
            "before": before.position, "after": frame.pose.position,
            "yaw_after": frame.pose.yaw_rad, "displacement_m": delta,
            "collision": bool(frame.collision), "proxy_collision": proxy_rejected})
        _write_json(destination / "actions.private.json", actions)
        if proxy_rejected:
            raise RuntimeError("Human collision guard rejected movement")
        if not math.isfinite(delta):
            raise RuntimeError("Nonfinite measured movement")
        if name == "move_forward":
            if delta > amount + .04:
                raise RuntimeError("Unbounded translation")
            if delta < 1e-5:
                raise RuntimeError("Actual movement stalled")
        elif delta > 1e-4:
            raise RuntimeError("Stationary survey turn unexpectedly translated the agent")

    def face(yaw):
        delta = _wrap(yaw-frame.pose.yaw_rad)
        if abs(delta) > 1e-5:
            act("turn_left" if delta > 0 else "turn_right", abs(math.degrees(delta)))
        if abs(_wrap(frame.pose.yaw_rad-yaw)) > .001:
            raise RuntimeError("Actual heading mismatch")

    def save(label, kind="route_forward"):
        identity = f"frame_{len(rows):03d}_{label}"
        Image.fromarray(frame.rgb).save(destination / (identity + ".png"))
        np.savez_compressed(destination / (identity + "_depth.npz"), depth_m=frame.depth)
        rows.append({"frame_id": identity, "kind": kind, "step": frame.step,
            "distance_from_anchor_m": travelled,
            "pose": {"position": frame.pose.position, "yaw_rad": frame.pose.yaw_rad},
            "camera": env.debug_camera(),
            "rgb_sha256": hashlib.sha256(frame.rgb.tobytes()).hexdigest(),
            "dummy_eligibility": frame.dummy_candidates})
        _write_json(destination / "frames.private.json", rows)

    def capture_views(label):
        yaw = frame.pose.yaw_rad
        save(label + "_forward")
        face(yaw + math.pi/2)
        save(label + "_left", "stationary_left_view")
        face(yaw - math.pi/2)
        save(label + "_right", "stationary_right_view")
        face(yaw)

    try:
        _write_json(destination / "audit.private.json", report)
        env = environment_factory(inputs.data_root / SCENE_REL,
                                  inputs.data_root / NAVMESH_REL, object_config, config=config)
        frame = env.reset(inputs.staged_episode, [inputs.dummy], SURVEY_SEED)
        save("helper_at_anchor", "helper_origin")
        if not any(row.get("spatial_eligible") is True for row in frame.dummy_candidates):
            raise RuntimeError("Staged anchor is not currently eligible to interact with the helper")
        if env.find_safe_path(inputs.staged_episode.start_position, inputs.episode.goal_position) is None:
            raise RuntimeError("No audited safe route: " + str(env.last_path_check))
        if shortest_path_factory is None:
            from habitat_sim import ShortestPath
            shortest_path_factory = ShortestPath
        query = shortest_path_factory()
        query.requested_start = np.asarray(inputs.staged_episode.start_position)
        query.requested_end = np.asarray(inputs.episode.goal_position)
        if not env.sim.pathfinder.find_path(query):
            raise RuntimeError("Shortest-path query failed after safety validation")
        corners = [finite_vector(tuple(float(x) for x in point)) for point in query.points]
        if (not corners or math.dist(corners[0], inputs.staged_episode.start_position) > .05
                or math.dist(corners[-1], inputs.episode.goal_position) > .05
                or not math.isfinite(float(query.geodesic_distance)) or query.geodesic_distance < 0):
            raise RuntimeError("Invalid queried route endpoints/distance")
        report.update(query_corners=corners, queried_geodesic_m=float(query.geodesic_distance))
        _write_json(destination / "audit.private.json", report)
        next_sample, next_survey = 1.0, 2.0
        for index, point in enumerate(corners[1:]):
            target = np.asarray(point)
            distance = float(np.linalg.norm((target-np.asarray(frame.pose.position))[[0,2]]))
            limit = math.ceil(distance/MAX_FORWARD_M)*4 + 20
            for _ in range(limit):
                difference = target-np.asarray(frame.pose.position)
                remaining = float(np.linalg.norm(difference[[0,2]]))
                if remaining <= CORNER_TOLERANCE_M:
                    break
                face(math.atan2(-difference[0], -difference[2]))
                act("move_forward", min(MAX_FORWARD_M, remaining))
                if travelled >= next_survey:
                    capture_views(f"m{int(next_survey):02d}")
                    next_survey += 2.0
                    next_sample = max(next_sample, math.floor(travelled)+1)
                elif travelled >= next_sample:
                    save(f"m{int(next_sample):02d}")
                    next_sample += 1.0
            else:
                raise RuntimeError("Corner iteration budget exceeded")
            save(f"corner{index+1:02d}")
        capture_views("goal")
        error = math.dist(frame.pose.position, inputs.episode.goal_position)
        if error > ENDPOINT_TOLERANCE_M:
            raise RuntimeError(f"Goal endpoint mismatch: {error}")
        report.update(walk_complete=True, endpoint_error_m=error,
                      final_position=frame.pose.position, error=None)
    except Exception as exc:
        report.update(walk_complete=False, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report.update(actual_walked_m=travelled, action_count=len(actions),
            movement_count=sum(a["action"] == "move_forward" for a in actions),
            collided_action_count=sum(a["collision"] for a in actions),
            human_proxy_rejections=sum(a["proxy_collision"] for a in actions),
            frame_count=len(rows), elapsed_seconds=time.monotonic()-started)
        try:
            _write_json(destination / "audit.private.json", report)
            _write_json(destination / "actions.private.json", actions)
            _write_json(destination / "frames.private.json", rows)
        finally:
            if env is not None:
                env.close()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path,
                        help="External frozen R2RCE data root; hashes/order are verified")
    parser.add_argument("--staging-report", required=True, type=Path,
                        help="External audited anchor/helper placement report")
    parser.add_argument("--output", required=True, type=Path,
                        help="Fresh PRIVATE evidence directory; never bundled/published automatically")
    parser.add_argument("--max-actions", type=int, default=10000)
    parser.add_argument("--helper-profile", choices=("game-v7","blue-reference"), default="game-v7")
    parser.add_argument("--helper-asset", type=Path)
    args = parser.parse_args(argv)
    inputs = load_survey_inputs(args.data_root, args.staging_report)
    report = run_survey(inputs, args.output, max_actions=args.max_actions,
                        helper_profile=args.helper_profile, helper_asset=args.helper_asset)
    print(json.dumps({"kind": report["kind"], "episode_id": report["episode_id"],
        "walk_complete": report["walk_complete"], "actual_walked_m": report["actual_walked_m"],
        "is_benchmark_result": False, "model_api_calls": 0,
        "output": str(args.output)}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare short diagnostic navigation segments from prior episode result ledgers.

Pure Python standard library implementation. Does not import model or simulator
packages and produces isolated cold-start segment datasets (`rounded_pose_cold_start`)
without restoring mapper state, history, or route prefix.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

DIAGNOSTIC_SCHEMA_VERSION = 1
RESTORE_MODE_COLD_START = "rounded_pose_cold_start"
DIAGNOSTIC_MAX_STEPS = 24
DEFAULT_CAMERA_HEIGHT = 1.2


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _validate_vec3(value: Any, field_name: str) -> Tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Invalid {field_name}: expected a 3-element sequence, got {value!r}")
    if not all(_is_finite_number(v) for v in value):
        raise ValueError(f"Invalid {field_name}: all coordinates must be finite numbers, got {value!r}")
    return float(value[0]), float(value[1]), float(value[2])


def mapper_pose_to_habitat_position(
    start_position: Sequence[float],
    pose_before: Sequence[float],
    camera_height: float = DEFAULT_CAMERA_HEIGHT,
) -> List[float]:
    """Convert dataset origin + ledger ground-plane pose_before to Habitat start_position."""
    if not _is_finite_number(camera_height) or float(camera_height) <= 0.0:
        raise ValueError(f"Invalid camera_height: expected finite positive number, got {camera_height!r}")
    hx, hy, hz = _validate_vec3(start_position, "start_position")
    mx, my, mz = _validate_vec3(pose_before, "pose_before")
    ch = float(camera_height)
    return [float(hx + mx), float(hy + mz + ch), float(hz + my)]


def mapper_yaw_to_habitat_theta(yaw_before: float) -> float:
    """Convert saved mapper yaw alpha (rad) to Habitat heading angle theta = -alpha - pi/2."""
    if not _is_finite_number(yaw_before):
        raise ValueError(f"Invalid yaw_before: expected finite number, got {yaw_before!r}")
    return -float(yaw_before) - (math.pi / 2.0)


def mapper_yaw_to_habitat_quaternion(yaw_before: float) -> List[float]:
    """Convert saved mapper yaw alpha (rad) to Habitat xyzw quaternion [0, sin(theta/2), 0, cos(theta/2)]."""
    theta = mapper_yaw_to_habitat_theta(yaw_before)
    half_theta = theta / 2.0
    return [0.0, float(math.sin(half_theta)), 0.0, float(math.cos(half_theta))]


def extract_step_executed_action(result_data: dict, target_step: int) -> dict:
    if isinstance(target_step, bool) or not isinstance(target_step, int) or target_step < 0:
        raise ValueError("target_step must be a non-negative integer")
    if not isinstance(result_data, dict):
        raise ValueError("result_data must be a dict")
    subtasks = result_data.get("subtask")
    if not isinstance(subtasks, dict) or not subtasks:
        raise ValueError("result_data['subtask'] must be a non-empty dict")

    matches = []
    for subtask_key, subtask_val in subtasks.items():
        if not isinstance(subtask_val, dict):
            raise ValueError(f"subtask '{subtask_key}' must be a dict")
        records = subtask_val.get("record")
        if not isinstance(records, list):
            raise ValueError(f"subtask '{subtask_key}' record must be a list")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"record in subtask '{subtask_key}' must be a dict")
            step = record.get("step")
            if not isinstance(step, bool) and isinstance(step, int) and step == target_step:
                matches.append(record)

    if not matches:
        raise ValueError(f"target_step {target_step} not found in subtask records")
    if len(matches) > 1:
        raise ValueError(f"ambiguous target_step {target_step}: found {len(matches)} records")

    matched = matches[0]
    executed_action = matched.get("executed_action")
    if not isinstance(executed_action, str) or not executed_action.strip():
        raise ValueError("executed_action must be a non-empty string")
    if "pose_before" not in matched:
        raise ValueError("pose_before is missing from matched record")
    _validate_vec3(matched["pose_before"], "pose_before")
    if "yaw_before" not in matched or not _is_finite_number(matched["yaw_before"]):
        raise ValueError("yaw_before must be a finite number")

    return matched


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_source_result_path(raw_path: Any, cases_dir: Path) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"Invalid source_result path: {raw_path!r}")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        return (cases_dir / candidate).resolve()
    return candidate.resolve()


def _index_dataset_episodes(dataset: Dict[str, Any]) -> Dict[Any, Dict[str, Any]]:
    if not isinstance(dataset, dict) or not isinstance(dataset.get("episodes"), list):
        raise ValueError("Dataset JSON must be an object containing an 'episodes' list")
    by_id: Dict[Any, Dict[str, Any]] = {}
    seen_str_ids = set()
    for ep in dataset["episodes"]:
        if not isinstance(ep, dict) or "episode_id" not in ep:
            raise ValueError("Each dataset episode must be an object with 'episode_id'")
        ep_id = ep["episode_id"]
        if isinstance(ep_id, bool) or ep_id is None:
            raise ValueError(f"Invalid episode_id in dataset: {ep_id!r}")
        str_id = str(ep_id)
        if ep_id in by_id or str_id in seen_str_ids:
            raise ValueError(f"Duplicate episode_id {ep_id!r} in source dataset")
        by_id[ep_id] = ep
        seen_str_ids.add(str_id)
    return by_id


def prepare_failure_segments(
    dataset_path: Union[str, Path],
    cases_path: Union[str, Path],
    output_path: Union[str, Path],
    camera_height: float = DEFAULT_CAMERA_HEIGHT,
) -> Dict[str, Any]:
    """Validate cases, construct cold-start diagnostic segment dataset, and atomically write output."""
    if not _is_finite_number(camera_height) or float(camera_height) <= 0.0:
        raise ValueError(f"Invalid camera_height: {camera_height!r}")

    dataset_file = Path(dataset_path).resolve()
    cases_file = Path(cases_path).resolve()
    output_file = Path(output_path).resolve()

    if not dataset_file.is_file():
        raise ValueError(f"Dataset file does not exist: {dataset_file}")
    if not cases_file.is_file():
        raise ValueError(f"Cases file does not exist: {cases_file}")

    forbidden_outputs = {dataset_file, cases_file}
    if output_file in forbidden_outputs:
        raise ValueError(f"Refusing to overwrite input file: {output_file}")
    if output_file.exists():
        raise ValueError(f"Refusing to overwrite existing file: {output_file}")

    dataset_sha256 = _sha256_file(dataset_file)
    with dataset_file.open("r", encoding="utf-8") as f:
        dataset_data = json.load(f)
    with cases_file.open("r", encoding="utf-8") as f:
        cases_data = json.load(f)

    if not isinstance(cases_data, list) or not cases_data:
        raise ValueError("Cases JSON must be a non-empty list of case specifications")

    episodes_by_id = _index_dataset_episodes(dataset_data)
    seen_segment_ids = set()
    result_cache: Dict[Path, Tuple[str, Dict[str, Any]]] = {}
    diagnostic_episodes: List[Dict[str, Any]] = []

    for idx, case in enumerate(cases_data):
        if not isinstance(case, dict):
            raise ValueError(f"Case at index {idx} must be a JSON object")

        segment_id = case.get("segment_id")
        if not isinstance(segment_id, str) or not segment_id.strip():
            raise ValueError(f"Case at index {idx} has missing or empty segment_id")
        if segment_id in seen_segment_ids:
            raise ValueError(f"Duplicate segment_id {segment_id!r}")
        seen_segment_ids.add(segment_id)

        source_ep_id = case.get("episode_id")
        if isinstance(source_ep_id, bool) or source_ep_id is None:
            raise ValueError(f"Invalid episode_id in case {segment_id!r}: {source_ep_id!r}")
        if source_ep_id not in episodes_by_id:
            raise ValueError(f"episode_id {source_ep_id!r} for segment {segment_id!r} not found in dataset")
        orig_ep = episodes_by_id[source_ep_id]

        step = case.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError(f"Invalid step in case {segment_id!r}: {step!r}")

        segment_instruction = case.get("instruction")
        if not isinstance(segment_instruction, str) or not segment_instruction.strip():
            raise ValueError(f"Invalid or empty instruction in case {segment_id!r}")

        orig_inst_obj = orig_ep.get("instruction")
        orig_instruction_text = (
            orig_inst_obj.get("instruction_text") if isinstance(orig_inst_obj, dict) else None
        )
        if not isinstance(orig_instruction_text, str) or not orig_instruction_text.strip():
            raise ValueError(f"Dataset episode {source_ep_id!r} missing valid instruction_text")
        if segment_instruction not in orig_instruction_text:
            raise ValueError(
                f"Segment instruction for {segment_id!r} must be a contiguous verbatim substring "
                f"of original instruction_text"
            )

        result_file = _resolve_source_result_path(case.get("source_result"), cases_file.parent)
        if not result_file.is_file():
            raise ValueError(f"source_result file not found for {segment_id!r}: {result_file}")
        if output_file == result_file:
            raise ValueError(f"Refusing to overwrite source_result file: {result_file}")

        if result_file not in result_cache:
            with result_file.open("r", encoding="utf-8") as rf:
                result_json = json.load(rf)
            result_cache[result_file] = (_sha256_file(result_file), result_json)
        result_sha256, result_json = result_cache[result_file]

        source_subtasks = result_json.get('subtask', {})
        if (not isinstance(source_subtasks, dict) or list(source_subtasks) != ['P1']
                or not isinstance(source_subtasks['P1'], dict)
                or not isinstance(source_subtasks['P1'].get('instruction'), str)
                or source_subtasks['P1']['instruction'].strip() != orig_instruction_text.strip()):
            raise ValueError('Source result must contain the selected original single instruction')
        for identity_field in ('episode_id', 'scene_id'):
            if (identity_field in result_json
                    and str(result_json[identity_field]) != str(orig_ep.get(identity_field))):
                raise ValueError('Source result ' + identity_field + ' does not match dataset episode')

        exec_action = extract_step_executed_action(result_json, step)
        new_start_pos = mapper_pose_to_habitat_position(
            orig_ep.get("start_position"),
            exec_action["pose_before"],
            camera_height=float(camera_height),
        )
        new_start_rot = mapper_yaw_to_habitat_quaternion(float(exec_action["yaw_before"]))

        new_ep = copy.deepcopy(orig_ep)
        new_ep["episode_id"] = idx
        new_ep["start_position"] = new_start_pos
        new_ep["start_rotation"] = new_start_rot
        new_ep["instruction"]["instruction_text"] = segment_instruction
        new_ep["reference_path"] = None
        if not isinstance(new_ep.get("info"), dict):
            new_ep["info"] = {}
        new_ep["info"]["geodesic_distance_requires_recalculation"] = True
        new_ep["diagnostic_segment"] = {
            "schema": DIAGNOSTIC_SCHEMA_VERSION,
            "segment_id": segment_id,
            "source_episode_id": orig_ep["episode_id"],
            "source_step": int(step),
            "source_result_sha256": result_sha256,
            "source_dataset_sha256": dataset_sha256,
            "instruction": segment_instruction,
            "source_original_instruction": orig_instruction_text,
            "restore_mode": RESTORE_MODE_COLD_START,
            "decomposition_enabled": False,
            "max_steps": DIAGNOSTIC_MAX_STEPS,
            "source_match_basis": "saved_single_instruction_text",
            "camera_height_m": float(camera_height),
        }
        if 'source_adaptation' in case:
            new_ep['diagnostic_segment']['source_adaptation'] = copy.deepcopy(case['source_adaptation'])
        diagnostic_episodes.append(new_ep)

    output_payload = {
        k: copy.deepcopy(v) for k, v in dataset_data.items() if k != "episodes"
    }
    output_payload["episodes"] = diagnostic_episodes

    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(output_file.parent),
            prefix=f".{output_file.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            json.dump(output_payload, tmp_file, indent=2, ensure_ascii=False, allow_nan=False)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        # Publish only when the destination is still absent; a concurrent writer
        # must not have its diagnostic or original result replaced.
        os.link(str(tmp_path), str(output_file))
        tmp_path.unlink()
    except Exception:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise

    return output_payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare short diagnostic failure-segment dataset from result ledgers."
    )
    parser.add_argument("--dataset", required=True, help="Path to original R2R dataset JSON.")
    parser.add_argument("--cases", required=True, help="Path to diagnostic segment cases JSON.")
    parser.add_argument("--output", required=True, help="Path to new output segment dataset JSON.")
    parser.add_argument(
        "--camera-height",
        "--camera_height",
        dest="camera_height",
        type=float,
        default=DEFAULT_CAMERA_HEIGHT,
        help="Camera height offset in meters (default: 1.2).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    prepare_failure_segments(
        dataset_path=args.dataset,
        cases_path=args.cases,
        output_path=args.output,
        camera_height=args.camera_height,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


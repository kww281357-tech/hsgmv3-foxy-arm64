"""Strict local R2RCE inputs. Full episodes remain on the environment side."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
from .contracts import DummySpec, Episode

FROZEN_IDS = [40,41,52,53,237,249,254,301,387,630,712,715,744,822,894,1334,1354,1612,1720,1755]
EPISODES_REL = "data/datasets/R2R_VLNCE_v1-3/val_unseen/zsNo4HB9uLZ_demo20.json"
DUMMIES_REL = "examples/dummies.zsNo4HB9uLZ_demo20.json"
SCENE_REL = "data/scene_datasets/mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb"
NAVMESH_REL = "data/scene_datasets/mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.navmesh"
EXPECTED = {
    EPISODES_REL: "b0e89cb0147a3496fbeab914ce9fb14ec90a3c437060204d3580672b990c6781",
    DUMMIES_REL: "4a6ef33c83cbc3bee3185bfca0481626c08666c0e95e57b198a0e812b004c6fe",
    SCENE_REL: "3611ac3b2a2b76bb15ab2623054929a932aa40c03786b3890cb17a73eccc5689",
    NAVMESH_REL: "495495066a57c1eb662d7a3daa3c73f3889460695db148bd991044eef9b06c42",
}


def finite_vector(value, length=3):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError("Invalid vector shape")
    if any(type(x) not in (int, float) or abs(x) > 1e6 or not math.isfinite(x) for x in value):
        raise ValueError("Invalid finite coordinate")
    return tuple(float(x) for x in value)


def load_episodes(path: Path) -> list[Episode]:
    raw = json.loads(Path(path).read_text())
    episodes = []
    seen = set()
    for item in raw["episodes"]:
        identity = str(item["episode_id"])
        if identity in seen:
            raise ValueError("Duplicate episode ID")
        seen.add(identity)
        text = item["instruction"]["instruction_text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Instruction required")
        rotation = finite_vector(item["start_rotation"], 4)
        if abs(sum(x*x for x in rotation)-1) > .01:
            raise ValueError("Start quaternion is not normalized")
        episodes.append(Episode(identity, item["scene_id"], text,
                                finite_vector(item["start_position"]), rotation,
                                finite_vector(item["goals"][0]["position"]),
                                str(item.get("trajectory_id", ""))))
    return episodes


def load_dummies(path: Path, episode: Episode) -> list[DummySpec]:
    data = json.loads(Path(path).read_text())
    entries = [row for row in data["episodes"] if str(row["episode_id"]) == episode.episode_id
               and row["scene_id"] == episode.scene_id]
    if len(entries) != 1:
        raise ValueError("Missing or duplicate dummy sidecar episode")
    result = []
    seen = set()
    for item in entries[0]["dummies"]:
        identity = item["dummy_id"]
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("Invalid or duplicate dummy ID")
        seen.add(identity)
        yaw = item["yaw_rad"]
        if type(yaw) not in (int, float) or abs(yaw) > 1e6 or not math.isfinite(yaw):
            raise ValueError("Invalid dummy yaw")
        offset = item.get("floor_offset_m", .8)
        if type(offset) not in (int, float) or not 0 <= offset <= 3:
            raise ValueError("Invalid foot-origin offset")
        result.append(DummySpec(identity, finite_vector(item["position"]), float(yaw),
                                floor_offset_m=float(offset)))
    return result


def validate_frozen_data(root: Path) -> dict:
    root = Path(root)
    hashes = {}
    for relative, expected in EXPECTED.items():
        path = root / relative
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError("Frozen file checksum mismatch: " + relative)
        hashes[relative] = {"sha256": actual, "bytes": path.stat().st_size}
    episodes = load_episodes(root / EPISODES_REL)
    if [int(e.episode_id) for e in episodes] != FROZEN_IDS:
        raise ValueError("Frozen episode IDs/order changed")
    layouts = [load_dummies(root / DUMMIES_REL, e) for e in episodes]
    if any(layout != layouts[0] for layout in layouts[1:]):
        raise ValueError("Paired episodes do not share the frozen layout")
    return {"status": "validated_local_files_only", "episode_count": len(episodes),
            "dummy_count": len(layouts[0]), "unique_trajectories": len({e.trajectory_id for e in episodes}),
            "scene": "zsNo4HB9uLZ", "files": hashes,
            "rendered": False, "navmesh_runtime_validated": False}

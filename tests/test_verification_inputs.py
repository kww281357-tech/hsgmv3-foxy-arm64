from __future__ import annotations

import json
import os
import sys
import unittest

import numpy as np
from PIL import Image

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
_AGENT_DIR = os.path.join(_SRC, "agent")
for _p in (_SRC,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import navigation_policy  # noqa: E402
from agent.verification_inputs import VerificationInputs  # noqa: E402


class _DummyMapper:
    def __init__(self, obs_id: int = 1, pos=(1.0, 2.0, 0.5), floor_h: float = -0.5, yaw_deg: float = 0.0):
        self.observation_id = obs_id
        self.current_position = np.array(pos, dtype=float)
        self.initial_floor_height = floor_h
        rad = float(np.deg2rad(yaw_deg))
        c, s = float(np.cos(rad)), float(np.sin(rad))
        c_test, s_test = float(np.cos(0.5)), float(np.sin(0.5))
        candidates = [
            ((s_test, c_test), (s, c)),
            ((c_test, s_test), (c, s)),
            ((-s_test, c_test), (-s, c)),
            ((c_test, -s_test), (c, -s)),
        ]
        fx, fy = s, c
        for (tx, ty), (rx, ry) in candidates:
            r_probe = np.array([[ty, 0.0, -tx], [-tx, 0.0, -ty], [0.0, 1.0, 0.0]], dtype=float)
            y_probe = navigation_policy.checkpoint_yaw(r_probe)
            if y_probe is not None and abs(float(y_probe) - 0.5) < 1e-6:
                fx, fy = rx, ry
                break
        self.current_rotation = np.array([[fy, 0.0, -fx], [-fx, 0.0, -fy], [0.0, 1.0, 0.0]], dtype=float)
        self._pts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [1.5, 2.5, 0.2]], dtype=float)

    def get_bev_height_points(self):
        return self._pts.copy()


class _Harness(VerificationInputs):
    def __init__(self, evidence: bool = True, bev: bool = True, cache_size: int = 48, max_frames: int = 4):
        self.navigation_cfg = {
            "verification_evidence_enabled": evidence,
            "verifier_fixed_bev": bev,
            "verification_frame_cache_size": cache_size,
            "verifier_event_max_frames": max_frames,
        }
        self.mapper = _DummyMapper()
        self._execution_ledger: list[dict] = []
        self.navigation_trace: list[dict] = []
        self._route_verification_context: dict = {
            "key": "sub_1",
            "checks": [{"id": "c0", "status": "satisfied"}, {"id": "c1", "status": "unknown"}],
            "ledger": [],
        }
        self.img_buffer: list[Image.Image] = []


def _make_obs(front_rgb=(255, 0, 0), left_rgb=(0, 255, 0), right_rgb=(0, 0, 255), back_rgb=(255, 255, 0), h=40, w=60):
    def _arr(rgb):
        a = np.zeros((h, w, 3), dtype=np.uint8)
        a[:, :] = rgb
        return a
    return {
        "color_sensor": _arr(front_rgb),
        "left": {"color_sensor": _arr(left_rgb) if left_rgb else None},
        "right": {"color_sensor": _arr(right_rgb) if right_rgb else None},
        "back": {"color_sensor": _arr(back_rgb) if back_rgb else None},
    }


def _make_row(
    step: int,
    obs_b: int | None,
    obs_a: int | None,
    x0: float = 0.0,
    x1: float = 0.5,
    yaw0: float = 0.0,
    yaw1: float = 0.0,
    cond: str | None = None,
    action_type: str = "move_forward",
    direction: str = "LEFT",
    landmark_view: str | None = None,
    recovery: bool = False,
    blocked: bool = False,
) -> dict:
    is_turn = action_type == "turn"
    dist = 0.0 if is_turn else float(abs(x1 - x0))
    pose_b = [float(x0), 0.0, 0.0]
    pose_a = [float(x0 if is_turn else x1), 0.0, 0.0]
    yaw_delta = float(yaw1 - yaw0)
    norm_dir = direction.upper()
    if is_turn:
        exec_act = "turn_left" if norm_dir == "LEFT" else "turn_right"
        req_act = {"type": "turn", "direction": norm_dir}
        req_dist = 0.0
        req_turn = yaw_delta
    else:
        exec_act = "move_forward"
        req_act = {"type": "move_forward", "distance": dist}
        req_dist = dist
        req_turn = 0.0
    row: dict = {
        "step": step,
        "controller_step": step * 10,
        "subtask_id": "sub_1",
        "executed_action": exec_act,
        "action_type": action_type,
        "motion_distance": dist,
        "pose_before": pose_b,
        "pose": pose_a,
        "yaw_before": float(yaw0),
        "yaw_after": float(yaw1),
        "yaw_delta": yaw_delta,
        "requested_action": req_act,
        "requested_distance": req_dist,
        "requested_turn_radians": req_turn,
        "height_delta": 0.0,
        "blocked": bool(blocked),
        "recovery": bool(recovery),
        "observation_id_before": obs_b,
        "observation_id_after": obs_a,
    }
    if recovery:
        row["source"] = "physical"
    if is_turn and cond:
        verb = "left" if norm_dir == "LEFT" else "right"
        cmd: dict = {
            "subtask_id": "sub_1",
            "condition_id": cond,
            "text": f"Turn {verb} at the statue",
            "direction": norm_dir,
        }
        if landmark_view:
            cmd["landmark_evidence"] = {"view": landmark_view}
        row["instruction_turn"] = cmd
    return row


class TestVerificationInputs(unittest.TestCase):
    def test_flags_off_no_extra_work(self):
        h = _Harness(evidence=False, bev=False)
        obs = _make_obs()
        self.assertIsNone(h._snapshot_verification_views(obs))
        row = _make_row(1, 1, 2)
        h._execution_ledger.append(row)
        h._cache_verification_frames(row, {"observation_id": 1, "views": {}}, obs)
        self.assertFalse(hasattr(h, "_verification_event_frames"))
        self.assertEqual(h._verification_execution_prompt(h._route_verification_context), "")
        self.assertEqual(h._verifier_evidence_content(), [])
        self.assertEqual(h._last_verification_input_metadata["selected_frames"], [])

    def test_snapshot_4views_copy_isolation_and_resize(self):
        h = _Harness(evidence=True, bev=False)
        h.mapper.observation_id = 7
        obs = _make_obs(front_rgb=(200, 10, 20), left_rgb=(10, 200, 20), right_rgb=(20, 10, 200), back_rgb=None, h=600, w=1200)
        snap = h._snapshot_verification_views(obs)
        self.assertIsNotNone(snap)
        self.assertEqual(snap["observation_id"], 7)
        self.assertEqual(set(snap["views"].keys()), {"front", "left", "right"})
        self.assertEqual(snap["views"]["front"].size, (512, 256))
        obs["color_sensor"][:, :] = 0
        px = snap["views"]["front"].getpixel((10, 10))
        self.assertEqual(px, (200, 10, 20))
        self.assertEqual(snap["views"]["left"].getpixel((10, 10)), (10, 200, 20))
        self.assertEqual(snap["views"]["right"].getpixel((10, 10)), (20, 10, 200))

    def test_strict_observation_ids_and_unexecuted_ignored(self):
        h = _Harness(evidence=True, bev=False)
        obs1 = _make_obs()
        h.mapper.observation_id = -1
        self.assertIsNone(h._snapshot_verification_views(obs1))
        h.mapper.observation_id = 10
        snap_before = h._snapshot_verification_views(obs1)
        unexec = _make_row(99, 10, 11)
        h._cache_verification_frames(unexec, snap_before, obs1)
        self.assertEqual(getattr(h, "_verification_event_frames", []), [])

        row_stale = _make_row(1, 9, 12)
        h._execution_ledger.append(row_stale)
        h.mapper.observation_id = 11
        h._cache_verification_frames(row_stale, snap_before, _make_obs())
        self.assertEqual(h._verification_event_frames, [])

        row_half = _make_row(2, 10, None)
        h._execution_ledger.append(row_half)
        h._cache_verification_frames(row_half, snap_before, _make_obs())
        self.assertTrue(all(f["step"] == 2 and f["phase"] == "BEFORE" and f["observation_id"] == 10 for f in h._verification_event_frames))
        self.assertEqual(h.img_buffer, [])

    def test_bounded_cache_retains_key_landmark_and_recent_frames(self):
        h = _Harness(evidence=True, bev=False, cache_size=16)
        obs_id = 1
        cur_x = 0.0
        cur_yaw = 0.0
        left_yaw = -float(np.pi / 2.0)
        for st in range(1, 18):
            h.mapper.observation_id = obs_id
            sb = h._snapshot_verification_views(_make_obs(left_rgb=(12, 234, 56)))
            obs_id += 1
            h.mapper.observation_id = obs_id
            after_obs = _make_obs(left_rgb=(12, 234, 56))
            if st == 2:
                row = _make_row(
                    st,
                    obs_id - 1,
                    obs_id,
                    x0=cur_x,
                    x1=cur_x,
                    yaw0=cur_yaw,
                    yaw1=left_yaw,
                    cond="c1",
                    action_type="turn",
                    direction="LEFT",
                    landmark_view="left",
                )
                cur_yaw = left_yaw
            else:
                next_x = cur_x + 0.5
                row = _make_row(
                    st,
                    obs_id - 1,
                    obs_id,
                    x0=cur_x,
                    x1=next_x,
                    yaw0=cur_yaw,
                    yaw1=cur_yaw,
                    action_type="move_forward",
                    recovery=(st == 5),
                )
                cur_x = next_x
            h._execution_ledger.append(row)
            h._cache_verification_frames(row, sb, after_obs)
            obs_id += 1
        self.assertLessEqual(len(h._verification_event_frames), 16)
        keys = {(f["step"], f["phase"], f["view"]) for f in h._verification_event_frames}
        self.assertIn((2, "BEFORE", "left"), keys)
        self.assertIn((2, "AFTER", "left"), keys)
        self.assertIn((5, "BEFORE", "front"), keys)
        self.assertIn((17, "BEFORE", "front"), keys)
        self.assertIn((17, "AFTER", "front"), keys)
        self.assertNotIn((1, "BEFORE", "front"), keys)

    def test_dynamic_rollback_prompt_content_bev_and_json_metadata(self):
        h = _Harness(evidence=True, bev=True, max_frames=4)
        left_yaw = -float(np.pi / 2.0)
        h.mapper.observation_id = 100
        sb1 = h._snapshot_verification_views(_make_obs(left_rgb=(50, 150, 250)))
        h.mapper.observation_id = 101
        obs_a1 = _make_obs(left_rgb=(60, 160, 250))
        r1 = _make_row(
            1,
            100,
            101,
            x0=0.0,
            x1=0.0,
            yaw0=0.0,
            yaw1=left_yaw,
            cond="c1",
            action_type="turn",
            direction="LEFT",
            landmark_view="left",
        )
        h._execution_ledger.append(r1)
        h._cache_verification_frames(r1, sb1, obs_a1)

        h.mapper.observation_id = 102
        sb2 = h._snapshot_verification_views(_make_obs())
        h.mapper.observation_id = 103
        obs_a2 = _make_obs()
        r2 = _make_row(
            2,
            102,
            103,
            x0=0.0,
            x1=0.8,
            yaw0=left_yaw,
            yaw1=left_yaw,
            action_type="move_forward",
            recovery=True,
        )
        h._execution_ledger.append(r2)
        h._cache_verification_frames(r2, sb2, obs_a2)

        r1["rejected_by_rollback"] = True
        h.navigation_trace.append({"step": r2["controller_step"], "event": "physical_recovery_started", "reason": "stuck"})
        prompt = h._verification_execution_prompt(h._route_verification_context)
        self.assertIn("execution_step != controller_step != observation_id", prompt)
        self.assertIn("CANNOT prove completion", prompt)
        self.assertNotIn("_source_row", prompt)
        full_txt = h._execution_ledger_text(h._execution_ledger)
        self.assertIn("step=1", full_txt)
        self.assertIn("step=2", full_txt)

        blocks = h._verifier_evidence_content()
        self.assertGreaterEqual(len(blocks), 6)
        texts = [b["text"] for b in blocks if b.get("type") == "text"]
        imgs = [b for b in blocks if b.get("type") == "image_url"]
        self.assertTrue(any("[REJECTED_BY_ROLLBACK]" in t and "view=left" in t for t in texts))
        self.assertTrue(any("[RECOVERY]" in t for t in texts))
        self.assertTrue(any("FIXED MAPPER AXES VERIFICATION BEV" in t and "legend" in t.lower() for t in texts))
        self.assertTrue(all(b["image_url"]["detail"] == "high" and b["image_url"]["url"].startswith("data:image/jpeg;base64,") for b in imgs))

        meta = h._last_verification_input_metadata
        dumped = json.dumps(meta)
        self.assertNotIn("data:image", dumped)
        self.assertNotIn("_source_row", dumped)
        self.assertTrue(any(f["rejected_by_rollback"] and f["diagnostic_only"] and f["view"] == "left" for f in meta["selected_frames"]))
        self.assertIsNotNone(meta["bev"])
        self.assertEqual(meta["errors"], [])

    def test_empty_ledger_bev_and_missing_mapper_diagnostic(self):
        h = _Harness(evidence=False, bev=True)
        h.mapper = _DummyMapper(yaw_deg=0.0)
        self.assertAlmostEqual(float(navigation_policy.checkpoint_yaw(h.mapper.current_rotation)), 0.0, places=5)
        blocks = h._verifier_evidence_content()
        self.assertEqual(len(blocks), 2)
        self.assertIn("legend", blocks[0]["text"].lower())
        self.assertNotIn("red/orange", blocks[0]["text"].lower())
        self.assertEqual(h._last_verification_input_metadata["errors"], [])
        bev_0 = h._last_verification_input_metadata["bev"]
        self.assertIsNotNone(bev_0)
        self.assertAlmostEqual(float(bev_0["current_yaw"]), 0.0, places=5)

        h.mapper = _DummyMapper(yaw_deg=90.0)
        self.assertAlmostEqual(float(navigation_policy.checkpoint_yaw(h.mapper.current_rotation)), float(np.pi / 2.0), places=5)
        h._verifier_evidence_content()
        bev_90 = h._last_verification_input_metadata["bev"]
        self.assertAlmostEqual(float(bev_90["current_yaw"]), float(np.pi / 2.0), places=5)

        h.mapper = None
        blocks_err = h._verifier_evidence_content()
        self.assertEqual(len(blocks_err), 1)
        self.assertIn("DIAGNOSTIC ERROR", blocks_err[0]["text"])
        self.assertEqual(len(h._last_verification_input_metadata["errors"]), 1)


if __name__ == "__main__":
    unittest.main()


"""Unit tests for verification_evidence diagnostic event and frame selection module."""

import math
import os
import sys
import unittest
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from verification_evidence import (
    build_execution_events,
    format_execution_ledger,
    select_event_summaries,
    select_evidence_frames,
)


def _turn_row(step: int, yaw_deg: float, cond: str = "c1", text: str = "turn right at door",
              direction: str = "RIGHT", sub: str = "s1", **extra) -> dict:
    row = {
        "step": step, "controller_step": step, "subtask_id": sub, "action_type": "turn",
        "motion_distance": 0.0, "yaw_delta": math.radians(yaw_deg),
        "requested_action": {"type": "turn", "direction": direction},
        "instruction_turn": {
            "subtask_id": sub, "condition_id": cond, "text": text,
            "direction": direction, "landmark_evidence": {"view": "right"},
        },
    }
    row.update(extra)
    return row


class TestVerificationEvidence(unittest.TestCase):
    """Verify execution event building, dynamic rejection, ledger formatting, and frame matching."""

    def test_turn_validation_departure_and_unowned_navigate_rotation(self) -> None:
        """Require exact requested_action, verb direction, and waypoint_execution for turn ownership."""
        ledger = [
            _turn_row(0, 90, cond="c_turn"),
            {"step": 1, "subtask_id": "s1", "action_type": "move_forward", "motion_distance": 0.45,
             "turn_departure": {"accepted": True, "condition_id": "c_turn", "turn_step": 0}},
            {"step": 2, "subtask_id": "s1", "action_type": "turn", "yaw_delta": -math.radians(40),
             "requested_action": {"type": "navigate", "waypoint_id": 7},
             "waypoint_execution": {"phase": "aligning", "action_id": 7}},
            {"step": 3, "subtask_id": "s1", "action_type": "turn", "yaw_delta": -math.radians(30),
             "requested_action": {"type": "navigate", "waypoint_id": 8}},
            _turn_row(4, 85, cond="c_missing_req", requested_action=None),
            _turn_row(5, -85, cond="c_lm_left", text="walk past the chair on your left", direction="LEFT"),
        ]
        ev = build_execution_events(ledger)
        self.assertEqual((ev[0]["event_type"], ev[0]["owned_condition_id"], ev[0]["turn_direction"]), ("instruction_turn", "c_turn", "RIGHT"))
        self.assertTrue(ev[1]["departure_accepted"])
        self.assertEqual(ev[1]["owned_condition_id"], "c_turn")
        self.assertEqual((ev[2]["event_type"], ev[2]["waypoint_id"], ev[2]["owned_condition_id"]), ("waypoint_alignment", 7, None))
        self.assertEqual((ev[3]["event_type"], ev[3]["waypoint_id"], ev[3]["owned_condition_id"]), ("unattributed_turn", None, None))
        self.assertEqual((ev[4]["event_type"], ev[4]["stale_instruction_turn"], ev[4]["owned_condition_id"]), ("unattributed_turn", True, None))
        self.assertEqual((ev[5]["event_type"], ev[5]["stale_instruction_turn"], ev[5]["owned_condition_id"]), ("unattributed_turn", True, None))

    def test_in_place_undo_and_dynamic_rollback_reclassification(self) -> None:
        """Mark in-place undo and dynamically reclassify events/summaries/frames on rollback."""
        img = Image.new("RGB", (4, 4))
        undo_ledger = [_turn_row(0, 90), {"step": 1, "subtask_id": "s1", "action_type": "turn", "yaw_delta": -math.radians(90)}]
        u_sum = select_event_summaries(build_execution_events(undo_ledger), current_condition_ids=("c1",), limit=5)
        self.assertTrue(u_sum[0]["undone_in_place"])
        self.assertFalse(u_sum[0]["route_evidence"])
        self.assertEqual((u_sum[1]["event_type"], u_sum[1]["undo_of_step"]), ("in_place_undo", 0))

        ledger = [
            _turn_row(0, 90, observation_id_before=10, observation_id_after=11),
            {"step": 1, "subtask_id": "s1", "action_type": "move_forward", "motion_distance": 0.4,
             "turn_departure": {"accepted": True, "condition_id": "c1", "turn_step": 0}},
        ]
        events = build_execution_events(ledger)
        ledger[0]["rejected_by_rollback"] = True
        summaries = select_event_summaries(events)
        self.assertTrue(all(not k.startswith("_") for s in summaries for k in s))
        self.assertTrue(summaries[0]["rejected_by_rollback"] and summaries[0]["diagnostic_only"])
        self.assertIsNone(summaries[0]["owned_condition_id"])
        self.assertFalse(summaries[1]["departure_accepted"] or summaries[1]["route_evidence"])
        self.assertEqual(summaries[1]["status"], "rejected_departure")
        sel = select_evidence_frames([{"step": 0, "phase": "AFTER", "observation_id": 11, "view": "right", "image": img}], events)
        self.assertTrue(sel[0]["rejected_by_rollback"] and sel[0]["diagnostic_only"])

    def test_partial_alignment_preserves_right_turn_and_departure_vs_complete_reverse(self) -> None:
        """Keep partial opposite waypoint alignment as waypoint_alignment when net RIGHT turn holds."""
        partial_ledger = [
            _turn_row(0, 90, cond="c1"),
            {"step": 1, "subtask_id": "s1", "action_type": "turn", "yaw_delta": -math.radians(18),
             "requested_action": {"type": "navigate", "waypoint_id": 1},
             "waypoint_execution": {"phase": "aligning", "waypoint_id": 1}},
            {"step": 2, "subtask_id": "s1", "action_type": "move_forward", "motion_distance": 0.35,
             "turn_departure": {"accepted": True, "condition_id": "c1", "turn_step": 0}},
        ]
        ev = build_execution_events(partial_ledger)
        self.assertEqual((ev[0]["event_type"], ev[0]["owned_condition_id"], ev[0]["undone_in_place"]), ("instruction_turn", "c1", False))
        self.assertEqual(
            (ev[1]["event_type"], ev[1]["in_place_undo"], ev[1]["undo_of_step"], ev[1]["condition_id"], ev[1]["owned_condition_id"]),
            ("waypoint_alignment", False, None, None, None),
        )
        self.assertTrue(ev[2]["departure_accepted"] and ev[2]["route_evidence"])
        self.assertEqual(ev[2]["owned_condition_id"], "c1")

        reverse_ledger = [
            _turn_row(0, 90, cond="c1"),
            {"step": 1, "subtask_id": "s1", "action_type": "turn", "yaw_delta": -math.radians(90),
             "requested_action": {"type": "navigate", "waypoint_id": 1},
             "waypoint_execution": {"phase": "aligning", "waypoint_id": 1}},
            {"step": 2, "subtask_id": "s1", "action_type": "move_forward", "motion_distance": 0.35,
             "turn_departure": {"accepted": True, "condition_id": "c1", "turn_step": 0}},
        ]
        rev = build_execution_events(reverse_ledger)
        self.assertTrue(rev[0]["undone_in_place"] and not rev[0]["route_evidence"])
        self.assertIsNone(rev[0]["owned_condition_id"])
        self.assertEqual((rev[1]["event_type"], rev[1]["in_place_undo"], rev[1]["undo_of_step"], rev[1]["condition_id"]), ("in_place_undo", True, 0, "c1"))
        self.assertIsNone(rev[1]["owned_condition_id"])
        self.assertFalse(rev[2]["departure_accepted"] or rev[2]["route_evidence"])

    def test_unknown_motion_not_blocked_and_measured_stairs_platform(self) -> None:
        """Keep unknown/negative/zero-request motion unblocked; require measured height/platform."""
        ledger = [
            {"step": 0, "subtask_id": "s1", "action_type": "move_forward", "motion_distance": None,
             "requested_distance": 0.2, "requested_action": {"type": "move_forward", "distance": 0.2}},
            {"step": 1, "subtask_id": "s1", "action_type": "move_forward", "motion_distance": 0.0,
             "requested_distance": 0.2, "executed_action": "climb stairs", "thought": "climbed stairs", "height_delta": 0.01},
            {"step": 2, "subtask_id": "s1", "action_type": "move_forward", "motion_distance": 0.3,
             "height_delta": 0.18, "turn_departure": {"accepted": False, "condition_id": "c1", "turn_step": 1}},
            {"step": 3, "subtask_id": "s1", "action_type": "move_forward", "motion_distance": 0.25, "landing_supported": True},
            {"step": 4, "subtask_id": "s1", "action_type": "move_forward", "motion_distance": 0.0, "requested_distance": 0.0},
            {"step": 5, "subtask_id": "s1", "action_type": "move_forward", "motion_distance": 0.0},
            {"step": 6, "subtask_id": "s1", "action_type": "move_forward", "motion_distance": -0.1, "requested_distance": 0.2},
            {"step": 7, "subtask_id": "s1", "action_type": ["move_forward"], "motion_distance": 0.0, "requested_distance": 0.2},
            {"step": 8, "subtask_id": "s1", "action_type": {"type": "turn"}, "yaw_delta": 0.5},
        ]
        ev = build_execution_events(ledger)
        self.assertEqual((ev[0]["event_type"], ev[0]["status"], ev[0]["blocked"]), ("unknown", "unknown", False))
        self.assertEqual((ev[1]["event_type"], ev[1]["blocked"], ev[1]["height_change"]), ("blocked", True, None))
        self.assertNotIn("climb", ev[1]["summary"])
        self.assertEqual((ev[2]["event_type"], ev[2]["departure_accepted"], ev[2]["height_change"]), ("turn_departure", False, "ascent"))
        self.assertEqual((ev[3]["event_type"], ev[3]["platform_supported"]), ("platform", True))
        self.assertEqual((ev[4]["event_type"], ev[4]["blocked"]), ("noop", False))
        self.assertEqual((ev[5]["event_type"], ev[5]["blocked"]), ("noop", False))
        self.assertEqual((ev[6]["event_type"], ev[6]["status"], ev[6]["motion_distance"], ev[6]["blocked"]), ("unknown", "unknown", None, False))
        self.assertEqual((ev[7]["action_type"], ev[7]["blocked"]), (None, False))
        self.assertEqual((ev[8]["action_type"], ev[8]["blocked"]), (None, False))

    def test_recovery_overflow_malformed_and_limit_checks(self) -> None:
        """Fail malformed/overflow values to unknown, require PIL images, and validate limits."""
        ledger = [
            {"step": 0, "controller_step": 10, "subtask_id": "s1", "action_type": "move_forward",
             "motion_distance": 10**1000, "yaw_delta": True, "height_delta": float("inf"),
             "pose_before": [1.0, 10**1000, 0.0], "pose": [1.0, 2.0, 3.0], "yaw_before": 10**1000, "yaw_after": 0.5},
            {"step": "bad", "controller_step": True, "subtask_id": "s1", "action_type": "move_forward", "motion_distance": 0.3},
        ]
        rec = [{"event": "physical_recovery_started", "status": "started", "step": 10, "subtask_id": "s1", "reason": "stuck"}]
        events = build_execution_events(ledger, recovery_events=rec, scope_key="s1")
        self.assertIsNone(events[0]["motion_distance"])
        self.assertIsNone(events[0]["pose_before"])
        self.assertEqual(events[0]["pose"], [1.0, 2.0, 3.0])
        ctrl_ev = [e for e in events if e["event_type"] == "recovery"][0]
        self.assertEqual((ctrl_ev["step"], ctrl_ev["controller_step"], ctrl_ev["recovery_source"]), (None, 10, "unknown"))
        inv_ev = [e for e in events if e["step"] is None and not e["recovery"]][0]
        self.assertEqual((inv_ev["event_type"], inv_ev["status"]), ("unknown", "unknown"))

        for bad_lim in (None, True, False, -1, "6", 10**1000):
            with self.assertRaises(ValueError):
                select_event_summaries(events, limit=bad_lim)  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                select_evidence_frames([], events, max_frames=bad_lim)  # type: ignore[arg-type]

    def test_paired_frames_missing_landmark_and_full_ledger_geometry(self) -> None:
        """Prioritize BEFORE/AFTER pairs, reject non-PIL/conflicting frames, and format full geometry."""
        img1, img2 = Image.new("RGB", (4, 4)), Image.new("RGB", (4, 4))
        ledger = [
            _turn_row(0, 90, observation_id_before=50, observation_id_after=51,
                      pose_before=[0.0, 0.0, 0.0], pose=[0.0, 0.0, 0.0], yaw_before=0.0, yaw_after=1.571),
            {"step": 1, "subtask_id": "s1", "action_type": "move_forward", "motion_distance": 0.25},
            _turn_row(2, -90, cond="c2", text="turn left", direction="LEFT",
                      instruction_turn={"subtask_id": "s1", "condition_id": "c2", "text": "turn left", "direction": "LEFT"},
                      requested_action={"type": "turn", "direction": "LEFT"}, observation_id_after=61),
        ] + [{"step": i, "subtask_id": "s1", "action_type": "move_forward", "motion_distance": 0.25} for i in range(3, 16)]
        events = build_execution_events(ledger)
        frames = [
            {"step": 0, "phase": "BEFORE", "observation_id": 50, "view": "front", "image": img1},
            {"step": 0, "phase": "AFTER", "observation_id": 51, "view": "right", "image": img2, "subtask_id": "s1", "condition_id": "c1"},
            {"step": 0, "phase": "AFTER", "observation_id": 51, "view": "right", "image": "not_pil"},
            {"step": 0, "phase": "AFTER", "observation_id": 51, "view": "right", "image": img1, "subtask_id": "other"},
            {"step": 2, "phase": "AFTER", "observation_id": 61, "view": "front", "image": img1},
        ]
        sel = select_evidence_frames(frames, events, current_condition_ids=("c1", "c2"), max_frames=3)
        self.assertEqual(len(sel), 3)
        self.assertTrue(sel[0]["is_primary_landmark_view"] and sel[0]["paired_before_after"])
        self.assertTrue(sel[1]["secondary_context"] and sel[1]["primary_landmark_view_missing"] and sel[1]["paired_before_after"])
        self.assertTrue(sel[2]["secondary_context"] and sel[2]["primary_landmark_view_missing"] and not sel[2]["landmark_match"])

        text = format_execution_ledger(ledger)
        self.assertEqual(len(text.splitlines()), 16)
        self.assertIn("pose_before=[0.0,0.0,0.0] pose=[0.0,0.0,0.0] yaw_before=0.000 yaw_after=1.571", text)
        self.assertIn("obs_before=50 obs_after=51", text)


    def test_huge_yaw_delta_sanitized(self):
        events = build_execution_events([
            _turn_row(1, 90, yaw_delta=1e308),
            _turn_row(2, 90, yaw_delta=1e308),
        ])
        self.assertEqual(len(events), 2)
        for ev in events:
            self.assertIsNone(ev["yaw_delta"])
            self.assertFalse(ev["route_evidence"])
            self.assertIsNone(ev["owned_condition_id"])

        normal = build_execution_events([_turn_row(1, 90)])[0]
        self.assertAlmostEqual(normal["yaw_delta"], math.pi / 2)
        self.assertEqual(normal["owned_condition_id"], "c1")

        rounded = build_execution_events([_turn_row(1, 90, yaw_delta=3.142)])[0]
        self.assertAlmostEqual(rounded["yaw_delta"], 3.142)
        self.assertEqual(rounded["owned_condition_id"], "c1")

if __name__ == "__main__":
    unittest.main()


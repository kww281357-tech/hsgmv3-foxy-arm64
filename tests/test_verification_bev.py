import json
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verification_bev import (  # noqa: E402
    CURRENT_ARROW_COLOR,
    CURRENT_POSE_COLOR,
    OBS_COLOR,
    PANEL_BG,
    STYLE_COLORS,
    TEXT_COLOR,
    TURN_AFTER_COLOR,
    TURN_BEFORE_COLOR,
    render_verification_bev,
)


class _FakeNumpyBool:
    __module__ = "numpy"

    def __float__(self):
        return 1.0


_FakeNumpyBool.__name__ = "bool_"


class TestVerificationBEV(unittest.TestCase):
    def test_yaw_zero_right_vs_half_pi_up_and_map_fixed_across_yaw(self):
        img0, meta0 = render_verification_bev([], [1.0, 2.0, 0.5], 0.0, max_side=240)
        self.assertEqual(img0.size, (240, 240))
        self.assertIsNone(meta0["axes"]["compass_north"])
        arr0 = meta0["current_arrow"]
        self.assertEqual(arr0["direction_px"], [1.0, 0.0])
        self.assertGreater(arr0["tip_px"][0], arr0["base_px"][0])
        self.assertEqual(arr0["tip_px"][1], arr0["base_px"][1])
        mid0 = ((arr0["base_px"][0] + arr0["tip_px"][0]) // 2, arr0["base_px"][1])
        up0 = (arr0["base_px"][0], arr0["base_px"][1] - 12)
        self.assertEqual(img0.getpixel(mid0), CURRENT_ARROW_COLOR)
        self.assertEqual(img0.getpixel(up0), PANEL_BG)

        img90, meta90 = render_verification_bev(
            [], [1.0, 2.0, 0.5], math.pi / 2, max_side=240
        )
        arr90 = meta90["current_arrow"]
        self.assertEqual(arr90["direction_px"], [0.0, -1.0])
        self.assertEqual(arr90["tip_px"][0], arr90["base_px"][0])
        self.assertLess(arr90["tip_px"][1], arr90["base_px"][1])
        mid90 = (arr90["base_px"][0], (arr90["base_px"][1] + arr90["tip_px"][1]) // 2)
        right90 = (arr90["base_px"][0] + 12, arr90["base_px"][1])
        self.assertEqual(img90.getpixel(mid90), CURRENT_ARROW_COLOR)
        self.assertEqual(img90.getpixel(right90), PANEL_BG)

        ledger = [
            {
                "step": 1,
                "action_type": "move_forward",
                "pose_before": [0.0, 0.0, 0.1],
                "pose": [1.0, 0.5, 0.2],
            }
        ]
        _, m_a = render_verification_bev(ledger, [1.0, 0.5, 0.2], 0.0, max_side=260)
        _, m_b = render_verification_bev(
            ledger, [1.0, 0.5, 0.2], 0.75 * math.pi, max_side=260
        )
        self.assertEqual(m_a["transform"], m_b["transform"])
        self.assertEqual(m_a["rendered_rows"], m_b["rendered_rows"])

    def test_in_place_right_turn_then_reversal_kept_separate_no_move_arrow(self):
        ledger = [
            {
                "step": 10,
                "controller_step": 99,
                "action_type": "turn",
                "pose_before": [2.0, 3.0, 0.2],
                "pose": [2.0, 3.0, 0.2],
                "yaw_before": 0.0,
                "yaw_after": 0.5 * math.pi,
                "yaw_delta": 0.5 * math.pi,
                "instruction_turn": True,
            },
            {
                "step": 11,
                "controller_step": 100,
                "action_type": "turn",
                "pose_before": [2.0, 3.0, 0.2],
                "pose": [2.0, 3.0, 0.2],
                "yaw_before": 0.5 * math.pi,
                "yaw_after": -0.5 * math.pi,
                "yaw_delta": -math.pi,
                "instruction_turn": False,
            },
        ]
        img, meta = render_verification_bev(
            ledger, [2.0, 3.0, 0.2], -0.5 * math.pi, max_side=240
        )
        events = meta["turn_events"]
        self.assertEqual([e["step"] for e in events], [10, 11])
        self.assertEqual([e["label"] for e in events], ["instruction_turn", "turn"])
        self.assertEqual([e["turn_direction"] for e in events], ["right", "left"])
        self.assertEqual(meta["map_labels"]["move_arrow_steps"], [])
        self.assertFalse(meta["rendered_rows"][0]["move_arrow"])
        self.assertFalse(meta["rendered_rows"][1]["move_arrow"])
        self.assertNotEqual(events[0]["strip_box_px"], events[1]["strip_box_px"])
        for ev in events:
            bx0, by0, bx1, by1 = ev["strip_box_px"]
            self.assertLess(bx0, bx1)
            self.assertLess(by0, by1)
            box_colors = {
                img.getpixel((x, y))
                for x in range(bx0 + 1, bx1)
                for y in range(by0 + 1, by1)
            }
            self.assertIn(TURN_BEFORE_COLOR, box_colors)
            self.assertIn(TURN_AFTER_COLOR, box_colors)

    def test_recovery_rejected_styles_and_planned_targets_ignored(self):
        ledger_with_requests = [
            {
                "step": 4,
                "controller_step": 40,
                "action_type": "move_forward",
                "pose_before": [0.0, 0.0, 0.0],
                "pose": [0.0, 1.0, 0.0],
                "recovery": True,
                "requested_action": {"type": "navigate", "target": [0.8, 0.5, 0.0]},
                "waypoint_execution": {"target": [-0.5, 0.4, 0.0]},
                "requested_distance": 3.0,
            },
            {
                "step": 5,
                "controller_step": 41,
                "action_type": "move_forward",
                "pose_before": [0.0, 1.0, 0.0],
                "pose": [-1.0, 1.0, 0.0],
                "recovery": False,
                "rejected_by_rollback": True,
                "waypoint_execution": {"target": [0.8, 0.5, 0.0]},
            },
        ]
        ledger_stripped = [
            {
                "step": 4,
                "action_type": "move_forward",
                "pose_before": [0.0, 0.0, 0.0],
                "pose": [0.0, 1.0, 0.0],
                "recovery": True,
            },
            {
                "step": 5,
                "action_type": "move_forward",
                "pose_before": [0.0, 1.0, 0.0],
                "pose": [-1.0, 1.0, 0.0],
                "recovery": False,
                "rejected_by_rollback": True,
            },
        ]
        img, meta = render_verification_bev(
            ledger_with_requests, [-1.0, 1.0, 0.0], math.pi, max_side=320
        )
        img_clean, meta_clean = render_verification_bev(
            ledger_stripped, [-1.0, 1.0, 0.0], math.pi, max_side=320
        )
        self.assertEqual(meta["transform"], meta_clean["transform"])
        self.assertEqual(meta["rendered_rows"], meta_clean["rendered_rows"])
        self.assertEqual(list(img.getdata()), list(img_clean.getdata()))

        rows = meta["rendered_rows"]
        self.assertEqual(rows[0]["style"], "recovery")
        self.assertEqual(rows[0]["recovery_source"], "unknown")
        self.assertEqual(rows[1]["style"], "rejected")
        self.assertTrue(rows[0]["move_arrow"])
        self.assertTrue(rows[1]["move_arrow"])
        self.assertEqual(meta["map_labels"]["move_arrow_steps"], [4, 5])
        self.assertEqual(meta["map_labels"]["step_labeled_steps"], [4, 5])
        for st in ("4", "5"):
            lbox = meta["map_labels"]["label_boxes_px"][st]
            lcolors = {
                img.getpixel((x, y))
                for x in range(lbox[0], lbox[2])
                for y in range(lbox[1], lbox[3])
            }
            self.assertTrue(
                any(
                    all(abs(c - tc) <= 32 for c, tc in zip(px[:3], TEXT_COLOR[:3]))
                    and sum(abs(c - bg) for c, bg in zip(px[:3], PANEL_BG[:3])) >= 300
                    for px in lcolors
                ),
                f"Expected visible antialiased label text near {TEXT_COLOR} contrasting with {PANEL_BG}, got {sorted(lcolors)[:8]}",
            )

        m0 = (
            (rows[0]["segment_px"][0][0] + rows[0]["segment_px"][1][0]) // 2,
            (rows[0]["segment_px"][0][1] + rows[0]["segment_px"][1][1]) // 2,
        )
        m1 = (
            (rows[1]["segment_px"][0][0] + rows[1]["segment_px"][1][0]) // 2,
            (rows[1]["segment_px"][0][1] + rows[1]["segment_px"][1][1]) // 2,
        )
        self.assertEqual(img.getpixel(m0), STYLE_COLORS["recovery"])
        self.assertEqual(img.getpixel(m1), STYLE_COLORS["rejected"])

        tf = meta["transform"]
        self.assertLess(tf["x_max"], 0.8)
        span = tf["x_max"] - tf["x_min"]
        box = tf["map_box_px"]
        ps = box[2] - box[0]
        interior_wp_px = (
            int(round(box[0] + ((-0.5 - tf["x_min"]) / span) * ps)),
            int(round(box[1] + ((tf["y_max"] - 0.4) / span) * ps)),
        )
        self.assertTrue(box[0] <= interior_wp_px[0] <= box[2])
        self.assertTrue(box[1] <= interior_wp_px[1] <= box[3])
        self.assertEqual(img.getpixel(interior_wp_px), PANEL_BG)

    def test_row_gaps_not_bridged_and_z_profile_recorded(self):
        ledger = [
            {
                "step": 2,
                "action_type": "move_forward",
                "pose_before": [-1.0, 0.0, 0.0],
                "pose": [-1.0, 1.0, 0.4],
            },
            {
                "step": 4,
                "action_type": "move_forward",
                "pose_before": [0.0, -0.8],
                "pose": [0.0, -0.4],
            },
            {
                "step": 6,
                "action_type": "move_forward",
                "pose_before": [1.0, 1.0, 1.2],
                "pose": [1.0, 0.0, 1.6],
            },
        ]
        img, meta = render_verification_bev(
            ledger, [1.0, 0.0, 1.6], -0.5 * math.pi, max_side=300
        )
        self.assertEqual(meta["counts"]["missing_step_gaps"], [[2, 4], [4, 6]])
        self.assertEqual(meta["z_profile"]["gap_steps"], [[2, 4], [4, 6]])
        self.assertEqual(len(meta["z_profile"]["segments"]), 2)
        self.assertEqual([s["step"] for s in meta["z_profile"]["segments"]], [2, 6])
        r0_end = meta["rendered_rows"][0]["segment_px"][1]
        r2_start = meta["rendered_rows"][2]["segment_px"][0]
        gap_mid = ((r0_end[0] + r2_start[0]) // 2, (r0_end[1] + r2_start[1]) // 2)
        self.assertEqual(img.getpixel(gap_mid), PANEL_BG)

        z_seg0 = meta["z_profile"]["segments"][0]
        z_seg1 = meta["z_profile"]["segments"][1]
        z_mid_x = (z_seg0["px1"][0] + z_seg1["px0"][0]) // 2
        z_mid_y = (z_seg0["px1"][1] + z_seg1["px0"][1]) // 2
        self.assertEqual(img.getpixel((z_mid_x, z_mid_y)), PANEL_BG)

    def test_malformed_bool_nan_overflow_outlier_and_unknown_pose(self):
        ledger = [
            {"step": True, "pose_before": [0, 0, 0], "pose": [1, 0, 0]},
            {"step": 1, "pose_before": [float("nan"), 0, 0], "pose": [1, 0, 0]},
            {"step": 2, "pose_before": [1e100, 0, 0], "pose": [1, 0, 0]},
            {"step": 3, "pose_before": [10**1000, 0, 0], "pose": [1, 0, 0]},
            {"step": 4, "pose_before": [_FakeNumpyBool(), 0, 0], "pose": [1, 0, 0]},
            {
                "step": 5,
                "action_type": "move_forward",
                "pose_before": [0.0, 0.0, 0.0],
                "pose": [0.5, 0.0, 0.0],
            },
            {
                "step": 6,
                "action_type": "move_forward",
                "pose_before": [0.5, 0.0, 0.0],
                "pose": [0.5, 0.6, 0.0],
            },
            {
                "step": 7,
                "action_type": "move_forward",
                "pose_before": [250.0, -200.0, 0.0],
                "pose": [251.0, -200.0, 0.0],
            },
        ]
        img, meta = render_verification_bev(
            ledger, [float("nan"), 0.0, 0.0], 0.0, max_side=260, outlier_radius=10.0
        )
        self.assertFalse(meta["current_pose_valid"])
        self.assertIsNone(meta["current_arrow"])
        self.assertNotIn(CURRENT_ARROW_COLOR, set(img.getdata()))
        self.assertEqual(meta["counts"]["invalid_rows"], 5)
        self.assertEqual(meta["counts"]["retained_rows"], 3)
        self.assertEqual(meta["counts"]["clipped_steps"], [7])
        self.assertTrue(meta["coordinate_clipping"]["active"])
        bbox = meta["coordinate_clipping"]["banner_box_px"]
        self.assertIsNotNone(bbox)
        self.assertTrue(0 <= bbox[0] < bbox[2] <= 260)
        self.assertTrue(0 <= bbox[1] < bbox[3] <= 260)
        banner_colors = {
            img.getpixel((x, y))
            for x in range(bbox[0], bbox[2])
            for y in range(bbox[1], bbox[3])
        }
        self.assertTrue(
            any(
                all(abs(c - tc) <= 32 for c, tc in zip(px[:3], TURN_BEFORE_COLOR[:3]))
                and sum(abs(c - bg) for c, bg in zip(px[:3], PANEL_BG[:3])) >= 250
                for px in banner_colors
            ),
            f"Expected visible antialiased banner text near {TURN_BEFORE_COLOR} contrasting with {PANEL_BG}, got {sorted(banner_colors)[:8]}",
        )

    def test_bounded_history_profile_fit_visible_legend_and_invalid_max_side(self):
        ledger = [
            {
                "step": 50 + i,
                "action_type": "turn" if i % 2 == 0 else "move_forward",
                "pose_before": [i * 0.02, 0.0, 0.1 + (i % 5) * 0.05],
                "pose": [(i + 1) * 0.02, 0.0, 0.15 + (i % 5) * 0.05],
                "yaw_before": 0.0,
                "yaw_after": 0.3 if i % 2 == 0 else 0.0,
                "yaw_delta": 0.3 if i % 2 == 0 else 0.0,
            }
            for i in range(150)
        ]
        obs = [[(i % 20) * 0.1, (i // 20) * 0.05] for i in range(2300)] + [
            [float("inf"), 0.0],
            [10**1000, 0.0],
            [500.0, 500.0],
        ]
        ledger_copy = list(ledger)
        img, meta = render_verification_bev(
            ledger,
            [1.0, 0.0, 0.1],
            0.0,
            observed_points=obs,
            max_side=280,
            max_records=128,
            max_points=2000,
            max_turn_labels=6,
            max_step_labels=8,
        )
        self.assertEqual(ledger, ledger_copy)
        self.assertEqual(img.size, (280, 280))
        self.assertEqual(meta["counts"]["retained_rows"], 128)
        self.assertEqual(meta["counts"]["omitted_history_rows"], 22)
        self.assertEqual(meta["rendered_rows"][0]["step"], 72)
        self.assertEqual(meta["counts"]["rendered_observed_points"], 2000)
        self.assertEqual(meta["counts"]["omitted_observed_points"], 301)
        self.assertEqual(meta["counts"]["invalid_observed_points"], 2)
        self.assertGreater(meta["counts"]["omitted_turn_labels"], 0)
        self.assertGreater(meta["counts"]["omitted_step_labels"], 0)
        self.assertGreater(meta["counts"]["omitted_map_labels"], 0)

        z_panel = meta["z_profile"]["panel_box_px"]
        self.assertEqual(len(meta["z_profile"]["segments"]), 128)
        for seg in meta["z_profile"]["segments"]:
            for pt in (seg["px0"], seg["px1"]):
                self.assertTrue(z_panel[0] <= pt[0] <= z_panel[2])
                self.assertTrue(z_panel[1] <= pt[1] <= z_panel[3])
                self.assertTrue(0 <= pt[0] < 280 and 0 <= pt[1] < 280)

        expected_swatches = {
            "executed": STYLE_COLORS["executed"],
            "recovery": STYLE_COLORS["recovery"],
            "rejected": STYLE_COLORS["rejected"],
            "blocked": STYLE_COLORS["blocked"],
            "current": CURRENT_POSE_COLOR,
            "observed_samples": OBS_COLOR,
            "turn_before": TURN_BEFORE_COLOR,
            "turn_after": TURN_AFTER_COLOR,
        }
        for key, expected_rgb in expected_swatches.items():
            sw = meta["legend"][key]["swatch_box_px"]
            cx = (sw[0] + sw[2]) // 2
            cy = (sw[1] + sw[3]) // 2
            self.assertEqual(img.getpixel((cx, cy)), expected_rgb)

        encoded = json.dumps(meta, allow_nan=False)
        self.assertLess(len(encoded), 120000)

        with self.assertRaises(ValueError):
            render_verification_bev([], [0, 0, 0], 0.0, max_side=120)
        with self.assertRaises(ValueError):
            render_verification_bev([], [0, 0, 0], 0.0, max_side=64)
        with self.assertRaises(ValueError):
            render_verification_bev([], [0, 0, 0], 0.0, max_side=True)
        with self.assertRaises(ValueError):
            render_verification_bev([], [0, 0, 0], 0.0, outlier_radius=10**1000)


if __name__ == "__main__":
    unittest.main()


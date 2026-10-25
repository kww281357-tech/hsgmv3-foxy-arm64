import unittest
from blue_help.interaction_range import RangeConfig, evaluate_interaction, audit_placement_evidence


def check(**changes):
    kwargs = dict(robot_ground_position=(0, 0, 0), dummy_foot_position=(1, 0, 0),
                  dummy_id="synthetic_blue_1", visible=True, line_of_sight_clear=True,
                  same_floor=True, navmesh_reachable=True, remaining_questions=3,
                  cooldown_ready=True)
    kwargs.update(changes)
    return evaluate_interaction(**kwargs)


def placement(**changes):
    kwargs = dict(robot_radius_m=.2, safety_margin_m=.05, measured_free_width_m=.6,
                  foot_on_navmesh=True, body_clear_of_static_geometry=True,
                  collision_or_occupancy_proxy_active=True, no_agent_human_overlap=True,
                  start_goal_connectivity_preserved=True, swept_path_avoids_human=True)
    kwargs.update(changes)
    return audit_placement_evidence(**kwargs)


class InteractionTests(unittest.TestCase):
    def test_inside_not_just_boundary(self):
        self.assertTrue(check(dummy_foot_position=(0, 0, 0))["eligible"])
        self.assertTrue(check(dummy_foot_position=(1, 0, 0))["eligible"])

    def test_exact_boundary_included(self):
        self.assertTrue(check(dummy_foot_position=(1, 0, 0))["eligible"])
        self.assertTrue(check(dummy_foot_position=(.6, 0, .8))["eligible"])

    def test_body_edge_touch_is_not_foot_inclusion(self):
        # Hypothetical 0.3m body reaches inside, but foot center is still outside.
        self.assertFalse(check(dummy_foot_position=(1.1, 0, 0))["eligible"])

    def test_horizontal_disk_and_separate_floor_check(self):
        self.assertTrue(check(dummy_foot_position=(.99, .3, 0))["eligible"])
        self.assertFalse(check(dummy_foot_position=(1, 3, 0))["eligible"])
        self.assertFalse(check(same_floor=False)["eligible"])

    def test_wall_occlusion_in_range_rejected(self):
        self.assertFalse(check(line_of_sight_clear=False)["eligible"])

    def test_out_of_camera_or_unreachable_rejected(self):
        self.assertFalse(check(visible=False)["eligible"])
        self.assertFalse(check(navmesh_reachable=False)["eligible"])

    def test_missing_evidence_fails_closed(self):
        for key in ("visible", "line_of_sight_clear", "same_floor", "navmesh_reachable"):
            with self.subTest(key=key):
                self.assertFalse(check(**{key: None})["eligible"])

    def test_budget_cooldown_and_no_auto_ask(self):
        self.assertFalse(check(remaining_questions=0)["eligible"])
        self.assertFalse(check(cooldown_ready=False)["eligible"])
        info = check()
        self.assertFalse(info["automatic_question"])
        self.assertFalse(info["circle_is_collision_body"])
        self.assertEqual(info["remaining_delivered_questions"], 3)

    def test_radius_configurable(self):
        self.assertFalse(check(config=RangeConfig(radius_m=.5))["eligible"])
        self.assertTrue(check(dummy_foot_position=(2.5, 0, 0), config=RangeConfig(radius_m=3))["eligible"])

    def test_environment_must_recheck_after_model_decides(self):
        self.assertTrue(check()["eligible"])
        self.assertFalse(check(dummy_foot_position=(3, 0, 0))["eligible"])
        self.assertFalse(check(remaining_questions=0)["eligible"])

    def test_public_info_omits_world_and_goal_coordinates(self):
        result = check()
        for key in ("position", "robot_ground_position", "dummy_foot_position", "goal", "path"):
            self.assertNotIn(key, result)

    def test_invalid_config_positions_and_budget(self):
        for changes in [dict(remaining_questions=4), dict(remaining_questions=True),
                        dict(dummy_foot_position=(float("nan"), 0, 0)), dict(visible=1)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                check(**changes)
        for radius in [0, -1, float("inf")]:
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                RangeConfig(radius_m=radius)


class PlacementEvidenceTests(unittest.TestCase):
    def test_complete_synthetic_evidence_passes(self):
        self.assertEqual(placement()["status"], "pass")

    def test_noncolliding_actor_is_not_proof_of_safety(self):
        self.assertEqual(placement(collision_or_occupancy_proxy_active=False)["status"], "fail")

    def test_narrow_passage_fails(self):
        self.assertEqual(placement(measured_free_width_m=.49)["status"], "fail")
        self.assertEqual(placement(measured_free_width_m=.5)["status"], "pass")

    def test_connectivity_and_swept_body_path_required(self):
        self.assertEqual(placement(start_goal_connectivity_preserved=False)["status"], "fail")
        self.assertEqual(placement(swept_path_avoids_human=False)["status"], "fail")

    def test_no_measurements_remain_pending(self):
        report = placement(measured_free_width_m=None, foot_on_navmesh=None)
        self.assertEqual(report["status"], "pending")
        self.assertIn("foot_on_navmesh", report["pending_checks"])

    def test_overlap_or_static_geometry_intersection_fails(self):
        self.assertEqual(placement(no_agent_human_overlap=False)["status"], "fail")
        self.assertEqual(placement(body_clear_of_static_geometry=False)["status"], "fail")


if __name__ == "__main__":
    unittest.main()

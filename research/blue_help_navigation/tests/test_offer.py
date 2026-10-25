import concurrent.futures
import unittest
from blue_help.help_offer import HelpOfferMachine, OfferConfig, OFFER_TEXT
from blue_help.interaction_range import evaluate_interaction


def candidate(distance=.8, ok=True, identity="blue"):
    return {"dummy_id": identity, "horizontal_distance_m": distance, "spatial_eligible": ok,
            "interaction_radius_m": 1.0}


def deliver(machine, request_id, step, **changes):
    kwargs = dict(answer_text="Synthetic validated answer: turn left relative to your current heading.", step=step,
                  answer_validated=True, spatial_eligible_now=True, pose_reference_valid=True)
    kwargs.update(changes)
    return machine.deliver(request_id, **kwargs)


class OfferTests(unittest.TestCase):
    def setUp(self):
        self.machine = HelpOfferMachine()
        self.machine.reset("synthetic_episode")

    def accept(self, step=0):
        offer = self.machine.observe([candidate()], step=step)
        self.assertIsNotNone(offer)
        return self.machine.choose(offer["offer_id"], "accept", step=step,
                                   spatial_eligible_now=True)["request_id"]

    def leave(self, step):
        self.machine.observe([candidate(1.3, False)], step=step)

    def test_offer_contains_no_route_and_no_budget_consumption(self):
        offer = self.machine.observe([candidate()], step=0)
        self.assertEqual(offer["text"], OFFER_TEXT)
        self.assertFalse(offer["route_information_in_offer"])
        self.assertEqual(self.machine.snapshot()["delivered_answers"], 0)
        self.assertEqual(self.machine.snapshot()["question_attempts"], 0)

    def test_decline_and_ignore_do_not_spend(self):
        for choice in ("decline", "ignore"):
            self.machine.reset(choice)
            offer = self.machine.observe([candidate()], step=0)
            result = self.machine.choose(offer["offer_id"], choice, step=1, spatial_eligible_now=True)
            self.assertFalse(result["budget_consumed"])
            self.assertTrue(result["resume_navigation"])
            self.assertEqual(self.machine.snapshot()["remaining_delivered_answers"], 3)

    def test_no_choice_expires_as_ignore(self):
        self.machine.observe([candidate()], step=0)
        self.assertIsNone(self.machine.observe([candidate()], step=5))
        self.assertFalse(self.machine.snapshot()["offer_pending"])
        self.assertEqual(self.machine.snapshot()["question_attempts"], 0)
        self.assertEqual(self.machine.events[-1]["event"], "ignored_timeout")

    def test_staying_inside_does_not_repeat_offer_after_cooldown(self):
        offer = self.machine.observe([candidate()], step=0)
        self.machine.choose(offer["offer_id"], "decline", step=1, spatial_eligible_now=True)
        self.assertIsNone(self.machine.observe([candidate()], step=20))

    def test_hysteresis_and_cooldown_both_required(self):
        offer = self.machine.observe([candidate()], step=0)
        self.machine.choose(offer["offer_id"], "ignore", step=0, spatial_eligible_now=True)
        self.machine.observe([candidate(1.1, False)], step=1)
        self.assertIsNone(self.machine.observe([candidate()], step=11))
        self.leave(12)
        self.assertIsNotNone(self.machine.observe([candidate()], step=13))
        self.machine.reset("early")
        offer = self.machine.observe([candidate()], step=0)
        self.machine.choose(offer["offer_id"], "ignore", step=0, spatial_eligible_now=True)
        self.leave(1)
        self.assertIsNone(self.machine.observe([candidate()], step=2))
        self.assertIsNotNone(self.machine.observe([candidate()], step=10))

    def test_visibility_loss_not_treated_as_exit(self):
        self.machine.observe([candidate()], step=0)
        self.machine.observe([], step=1)
        self.assertIsNone(self.machine.observe([candidate()], step=20))

    def test_accept_requires_fresh_environment_check(self):
        offer = self.machine.observe([candidate()], step=0)
        result = self.machine.choose(offer["offer_id"], "accept", step=1, spatial_eligible_now=False)
        self.assertEqual(result["status"], "not_eligible")
        self.assertEqual(self.machine.snapshot()["question_attempts"], 0)

    def test_accept_is_not_delivered_answer_or_api_call(self):
        request = self.accept()
        self.assertTrue(request)
        self.assertEqual(self.machine.snapshot()["question_attempts"], 1)
        self.assertEqual(self.machine.snapshot()["delivered_answers"], 0)

    def test_success_delivered_exactly_once(self):
        request = self.accept()
        self.assertTrue(deliver(self.machine, request, 1)["budget_consumed"])
        self.assertFalse(deliver(self.machine, request, 1)["budget_consumed"])
        self.assertEqual(self.machine.snapshot()["delivered_answers"], 1)

    def test_concurrent_duplicate_callbacks_count_once(self):
        request = self.accept()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: deliver(self.machine, request, 1), range(24)))
        self.assertEqual(sum(result["budget_consumed"] for result in results), 1)
        self.assertEqual(self.machine.snapshot()["delivered_answers"], 1)

    def test_failure_stale_pose_or_out_of_range_not_counted(self):
        for changes in [dict(answer_text=None), dict(answer_text=""), dict(answer_validated=False),
                        dict(spatial_eligible_now=False), dict(pose_reference_valid=False)]:
            with self.subTest(changes=changes):
                self.machine.reset("failed")
                request = self.accept()
                self.assertFalse(deliver(self.machine, request, 1, **changes)["budget_consumed"])
                self.assertEqual(self.machine.snapshot()["delivered_answers"], 0)
                self.assertEqual(self.machine.snapshot()["question_attempts"], 1)

    def test_pending_answer_timeout(self):
        request = self.accept()
        self.machine.observe([candidate()], step=30)
        self.assertFalse(self.machine.snapshot()["answer_pending"])
        self.assertFalse(deliver(self.machine, request, 31)["budget_consumed"])
        self.assertEqual(self.machine.snapshot()["delivered_answers"], 0)

    def test_max_three_and_episode_reset(self):
        for i in range(3):
            if i:
                self.leave(i * 20 - 1)
            request = self.accept(i * 20)
            self.assertTrue(deliver(self.machine, request, i * 20 + 1)["budget_consumed"])
        self.leave(60)
        self.assertIsNone(self.machine.observe([candidate()], step=61))
        self.assertEqual(self.machine.snapshot()["remaining_delivered_answers"], 0)
        self.machine.reset("new")
        self.assertEqual(self.machine.snapshot()["remaining_delivered_answers"], 3)
        self.assertIsNotNone(self.machine.observe([candidate()], step=0))

    def test_late_previous_episode_callback_rejected(self):
        old = self.accept()
        self.machine.reset("new")
        new = self.accept()
        self.assertNotEqual(old, new)
        self.assertFalse(deliver(self.machine, old, 1)["budget_consumed"])
        self.assertTrue(deliver(self.machine, new, 1)["budget_consumed"])

    def test_stale_episode_answer_cannot_advance_new_episode_clock(self):
        old = self.accept()
        self.machine.reset("new")
        new = self.accept()
        before = self.machine.snapshot(), list(self.machine.events)
        self.assertFalse(deliver(self.machine, old, 10000)["budget_consumed"])
        self.assertEqual(before, (self.machine.snapshot(), self.machine.events))
        self.assertTrue(deliver(self.machine, new, 1)["budget_consumed"])

    def test_stale_episode_offer_cannot_expire_current_offer(self):
        old = self.machine.observe([candidate()], step=0)
        self.machine.reset("new")
        new = self.machine.observe([candidate()], step=0)
        before = self.machine.snapshot(), list(self.machine.events)
        result = self.machine.choose(old["offer_id"], "accept", step=10000, spatial_eligible_now=True)
        self.assertEqual(result["status"], "stale_or_missing_offer")
        self.assertEqual(before, (self.machine.snapshot(), self.machine.events))
        result = self.machine.choose(new["offer_id"], "accept", step=1, spatial_eligible_now=True)
        self.assertEqual(result["status"], "accepted")

    def test_geometry_and_offer_share_radius(self):
        config = OfferConfig(radius_m=1.5)
        self.assertEqual(config.range_config().radius_m, 1.5)
        machine = HelpOfferMachine(config)
        with self.assertRaisesRegex(ValueError, "configuration must match"):
            machine.observe([candidate()], step=0)
        item = candidate(distance=1.4)
        item["interaction_radius_m"] = 1.5
        self.assertIsNotNone(machine.observe([item], step=0))

    def test_no_ask_arm_never_offers(self):
        no_ask = HelpOfferMachine(OfferConfig(max_delivered_answers=0))
        self.assertIsNone(no_ask.observe([candidate()], step=0))

    def test_nearest_candidate_deterministic(self):
        offer = self.machine.observe([candidate(.9, identity="a"), candidate(.5, identity="b")], step=0)
        self.assertEqual(offer["dummy_id"], "b")

    def test_cooldown_does_not_invalidate_existing_offer(self):
        args = dict(robot_ground_position=(0, 0, 0), dummy_foot_position=(.8, 0, 0),
                    dummy_id="blue", visible=True, line_of_sight_clear=True,
                    same_floor=True, navmesh_reachable=True, remaining_questions=3)
        info = evaluate_interaction(**args, cooldown_ready=True)
        offer = self.machine.observe([info], step=0)
        info = evaluate_interaction(**args, cooldown_ready=False)
        self.assertFalse(info["eligible"])
        self.assertTrue(info["spatial_eligible"])
        self.machine.observe([info], step=1)
        result = self.machine.choose(offer["offer_id"], "accept", step=1, spatial_eligible_now=True)
        self.assertEqual(result["status"], "accepted")


if __name__ == "__main__":
    unittest.main()

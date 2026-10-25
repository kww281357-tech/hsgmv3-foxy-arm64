"""Pure synthetic state-machine tests; no API, simulator, or navigation rollout."""
import concurrent.futures
from pathlib import Path
import sys
import threading
import unittest

from blue_help.help_offer import HelpOfferMachine, MAX_ANSWER_CHARACTERS, OfferConfig


def candidate(distance=.8, eligible=True):
    return dict(dummy_id='synthetic-helper', horizontal_distance_m=distance,
                spatial_eligible=eligible, interaction_radius_m=1.)


def full_text(length=3286):
    beginning = 'SYNTHETIC FULL MANUAL ROUTE\n'
    ending = '\nFINAL STAGE MUST REMAIN COMPLETE.'
    return beginning+'x'*(length-len(beginning)-len(ending))+ending


def deliver(machine, token, step=1, **changes):
    args = dict(answer_text=full_text(), step=step, answer_validated=True,
                spatial_eligible_now=True, pose_reference_valid=True)
    args.update(changes)
    return machine.deliver(token, **args)


def accept(machine, step=0):
    offer = machine.observe([candidate()], step=step)
    if offer is None:
        raise AssertionError('Fixture expected a fresh offer')
    return machine.choose(offer['offer_id'], 'accept', step=step,
                          spatial_eligible_now=True)['request_id']


class TrialControllerTests(unittest.TestCase):
    def setUp(self):
        self.machine = HelpOfferMachine()
        self.machine.reset('synthetic-episode')

    def leave(self, step):
        self.machine.observe([candidate(1.3, False)], step=step)

    def test_3286_char_full_answer_successfully_delivers_without_truncation(self):
        text = full_text()
        self.assertEqual(len(text), 3286)
        token = accept(self.machine)
        self.assertEqual(self.machine.attempts, 1)
        self.assertEqual(self.machine.delivered, 0)
        result = deliver(self.machine, token, answer_text=text)
        self.assertEqual(result['status'], 'answer_delivered')
        self.assertEqual(result['answer_text'], text)
        self.assertTrue(result['answer_text'].endswith('FINAL STAGE MUST REMAIN COMPLETE.'))
        self.assertEqual(self.machine.delivered, 1)
        self.assertEqual(self.machine.events[-1]['delivered_answers'], 1)

    def test_exact_8192_char_boundary_accepted_and_8193_rejected_without_charge(self):
        self.assertEqual(MAX_ANSWER_CHARACTERS, 8192)
        token = accept(self.machine)
        self.assertTrue(deliver(self.machine, token, answer_text=full_text(8192))['budget_consumed'])
        self.machine.reset('overlong')
        token = accept(self.machine)
        result = deliver(self.machine, token, answer_text=full_text(8193))
        self.assertEqual(result['status'], 'answer_failed_or_invalid')
        self.assertFalse(result['budget_consumed'])
        self.assertNotIn('answer_text', result)
        self.assertEqual(self.machine.delivered, 0)
        self.assertEqual(self.machine.attempts, 1)
        self.assertFalse(self.machine.snapshot()['answer_pending'])
        self.assertFalse(deliver(self.machine, token)['budget_consumed'])

    def test_character_limit_is_not_byte_limit_and_string_is_unmodified(self):
        text = '路'*8192
        token = accept(self.machine)
        result = deliver(self.machine, token, answer_text=text)
        self.assertGreater(len(text.encode('utf8')), 8192)
        self.assertEqual(result['answer_text'], text)

    def test_atomic_concurrent_duplicate_callbacks_deliver_exactly_once(self):
        token = accept(self.machine)
        barrier = threading.Barrier(8)
        def callback(_):
            barrier.wait(timeout=5)
            return deliver(self.machine, token)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(callback, range(8)))
        self.assertEqual(sum(r['budget_consumed'] for r in results), 1)
        self.assertEqual(sum('answer_text' in r for r in results), 1)
        self.assertEqual(self.machine.delivered, 1)
        self.assertEqual(self.machine.attempts, 1)
        self.assertEqual(sum(e['event'] == 'answer_delivered' for e in self.machine.events), 1)

    def test_three_successful_answers_maximum_and_reentry_cannot_add_fourth(self):
        for i in range(3):
            if i:
                self.leave(i*20-1)
            token = accept(self.machine, i*20)
            result = deliver(self.machine, token, step=i*20+1)
            self.assertTrue(result['budget_consumed'])
            self.assertEqual(result['remaining_delivered_answers'], 2-i)
        self.leave(60)
        self.assertIsNone(self.machine.observe([candidate()], step=61))
        self.assertEqual(self.machine.delivered, 3)
        self.assertEqual(self.machine.attempts, 3)
        self.assertFalse(deliver(self.machine, token, step=62)['budget_consumed'])
        self.assertEqual(self.machine.delivered, 3)

    def test_invalid_answers_and_trusted_validation_failures_are_free(self):
        for changes in ({'answer_text': None}, {'answer_text': ''}, {'answer_text': ' \n\t'},
                        {'answer_validated': False}, {'spatial_eligible_now': False},
                        {'pose_reference_valid': False}):
            self.machine.reset('invalid-answer')
            token = accept(self.machine)
            with self.subTest(changes=changes):
                self.assertFalse(deliver(self.machine, token, **changes)['budget_consumed'])
                self.assertEqual(self.machine.delivered, 0)
                self.assertEqual(self.machine.attempts, 1)
                self.assertFalse(deliver(self.machine, token)['budget_consumed'])
        self.leave(9)
        token = accept(self.machine, 10)
        self.assertTrue(deliver(self.machine, token, step=11)['budget_consumed'])
        self.assertEqual(self.machine.attempts, 2)
        self.assertEqual(self.machine.delivered, 1)

    def test_reset_restores_budget_and_invalidates_old_token_without_advancing_clock(self):
        old = accept(self.machine)
        self.machine.reset('next-episode')
        current = accept(self.machine)
        before = self.machine.snapshot(), list(self.machine.events)
        self.assertNotEqual(old, current)
        self.assertFalse(deliver(self.machine, old, step=100000)['budget_consumed'])
        self.assertEqual(before, (self.machine.snapshot(), self.machine.events))
        self.assertEqual(self.machine.snapshot()['remaining_delivered_answers'], 3)
        self.assertTrue(deliver(self.machine, current)['budget_consumed'])

    def test_decline_ignore_and_timeout_never_charge_or_create_request(self):
        for choice in ('decline', 'ignore'):
            self.machine.reset(choice)
            offer = self.machine.observe([candidate()], step=0)
            result = self.machine.choose(offer['offer_id'], choice, step=1, spatial_eligible_now=True)
            self.assertFalse(result['budget_consumed'])
            self.assertTrue(result['resume_navigation'])
            self.assertEqual(self.machine.delivered, 0)
            self.assertEqual(self.machine.attempts, 0)
            self.assertFalse(self.machine.snapshot()['answer_pending'])
        self.machine.reset('ignored-timeout')
        self.machine.observe([candidate()], step=0)
        self.machine.observe([candidate()], step=5)
        self.assertEqual(self.machine.events[-1]['event'], 'ignored_timeout')
        self.assertEqual(self.machine.attempts, 0)
        self.assertEqual(self.machine.delivered, 0)

    def test_expired_pending_answer_does_not_charge(self):
        token = accept(self.machine)
        self.assertFalse(deliver(self.machine, token, step=30)['budget_consumed'])
        self.assertEqual(self.machine.delivered, 0)
        self.assertEqual(self.machine.events[-1]['event'], 'answer_timeout')

    def test_cooldown_and_exit_hysteresis_unchanged(self):
        token = accept(self.machine)
        deliver(self.machine, token)
        self.assertIsNone(self.machine.observe([candidate()], step=20))
        self.machine.observe([candidate(1.1, False)], step=21)
        self.assertIsNone(self.machine.observe([candidate()], step=22))
        self.leave(23)
        self.assertIsNotNone(self.machine.observe([candidate()], step=24))

    def test_invalid_validation_flags_do_not_consume_pending_answer(self):
        token = accept(self.machine)
        with self.assertRaises(ValueError):
            deliver(self.machine, token, answer_validated=1)
        self.assertTrue(self.machine.snapshot()['answer_pending'])
        self.assertTrue(deliver(self.machine, token)['budget_consumed'])

    def test_runner_uses_the_same_consolidated_controller(self):
        from blue_help.runner import HelpOfferMachine as RunnerMachine
        self.assertIs(RunnerMachine, HelpOfferMachine)

    def test_no_ask_arm_remains_disabled_and_cap_above_three_invalid(self):
        no_ask = HelpOfferMachine(OfferConfig(max_delivered_answers=0))
        self.assertIsNone(no_ask.observe([candidate()], step=0))
        with self.assertRaises(ValueError):
            HelpOfferMachine(OfferConfig(max_delivered_answers=4))


if __name__ == '__main__':
    unittest.main()

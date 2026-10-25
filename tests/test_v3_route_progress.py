"""CPU regressions for route prefix, signed turns and positive wrong-route evidence."""
import math
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from navigation_contracts import completion_conditions, validate_condition_checks
from route_progress import (check_route_contradiction, route_progress_packet,
                            straight_witness, turn_directions)
from test_v3_verification_contract import Guard, record, verdict


def row(cid, status, steps, evidence='Observed physical action at the specified route landmark.'):
    return {'id': cid, 'status': status, 'steps': steps, 'evidence': evidence}


def move(step, before, after, heading=None):
    distance = math.dist(before, after)
    result = record(step, pose=(*after, 0.), motion=distance)
    result['pose_before'] = [*before, 0.]
    if heading is not None:
        result['yaw_before'] = heading
        result['yaw_after'] = heading
    return result


def contradiction(kind='wrong_room', cid='P1.C1', steps=None,
                  expected='Enter the kitchen', observed='Entered a bedroom with a bed and nightstands.'):
    return {'scene_consistent': False, 'confidence': 'high', 'route_contradiction': {
        'kind': kind, 'condition_id': cid, 'steps': [1] if steps is None else steps,
        'expected': expected, 'observed': observed}}


class SignedMotionTests(unittest.TestCase):
    def checked_turn(self, instruction, yaw):
        checks = completion_conditions(instruction)
        return validate_condition_checks(checks, [row('P1.C1', 'satisfied', [1])],
            [record(1, 'turn', motion=0., yaw=yaw)])[0][0]

    def test_production_mapper_yaw_convention_left_negative_right_positive(self):
        # Saved Ep722: left step37=-1.107 rad, right step52=+0.929 rad.
        for text, yaw in [('Turn left.', -1.107), ('Make a right.', .929)]:
            self.assertEqual(self.checked_turn(text, yaw)['status'], 'satisfied')
            self.assertEqual(self.checked_turn(text, -yaw)['status'], 'unknown')

    def test_episode_1159_fifteen_degree_left_is_not_turn_around(self):
        self.assertEqual(self.checked_turn('Turn around.', -.267)['status'], 'unknown')
        self.assertEqual(self.checked_turn('Turn around.', -math.pi)['status'], 'satisfied')
        self.assertEqual(self.checked_turn('Turn around.', math.pi)['status'], 'satisfied')
        self.assertEqual(self.checked_turn('Turn around.', 2 * math.pi)['status'], 'unknown')

    def test_landmark_relative_position_is_not_an_egocentric_turn(self):
        self.assertEqual(turn_directions('Walk through the door to the right of the fireplace.'), [])
        self.assertEqual(turn_directions('Turn left and walk to the door on your right.'), ['left'])
        self.assertEqual(turn_directions('Make another left near the outside door.'), ['left'])

    def test_two_lefts_need_distinct_executed_turns(self):
        conditions = completion_conditions('Turn left. Then turn left.')
        rows = [row(c['id'], 'satisfied', [1]) for c in conditions]
        checked, _ = validate_condition_checks(conditions, rows, [record(1, 'turn', motion=0., yaw=-1.57)])
        self.assertEqual([c['status'] for c in checked], ['satisfied', 'unknown'])
        rows[1]['steps'] = [3]
        checked, _ = validate_condition_checks(conditions, rows, [
            record(1, 'turn', motion=0., yaw=-1.57), record(2), record(3, 'turn', motion=0., yaw=-1.57)])
        self.assertEqual([c['status'] for c in checked], ['satisfied', 'satisfied'])

    def test_merged_tail_preserves_two_turns_separated_by_movement(self):
        conditions = completion_conditions('Walk ahead. Turn left. Walk onward. Turn right.', max_conditions=2)
        ledger = [record(1), record(2, 'turn', motion=0., yaw=-1.57), record(3),
                  record(4, 'turn', motion=0., yaw=1.57)]
        rows = [row('P1.C1', 'satisfied', [1]), row('P1.C2', 'satisfied', [2, 3, 4])]
        checked, _ = validate_condition_checks(conditions, rows, ledger)
        self.assertTrue(all(c['status'] == 'satisfied' for c in checked))

    def test_straight_motion_needs_geometry_and_heading_not_forward_label(self):
        conditions = completion_conditions('Go straight along the corridor.')
        rows = [row('P1.C1', 'satisfied', [1])]
        checked, _ = validate_condition_checks(conditions, rows, [record(1)])
        self.assertEqual(checked[0]['status'], 'unknown')
        checked, _ = validate_condition_checks(conditions, rows, [move(1, (0, 0), (.5, 0), 0.)])
        self.assertEqual(checked[0]['status'], 'satisfied')
        checked, _ = validate_condition_checks(conditions, rows, [move(1, (0, 0), (0, .5), 0.)])
        self.assertEqual(checked[0]['status'], 'unknown')

    def test_straight_omitted_middle_turn_is_still_checked(self):
        ledger = [move(1, (0, 0), (.5, 0), 0.), record(2, 'turn', motion=0., yaw=1.57),
                  move(3, (.5, 0), (.5, .5), 1.57)]
        okay, violated, _ = straight_witness([ledger[0], ledger[-1]], ledger)
        self.assertFalse(okay)
        self.assertTrue(violated)

    def test_straight_cannot_hide_a_prior_turn_but_can_follow_a_separate_turn_clause(self):
        ledger = [record(0, 'turn', motion=0., yaw=1.57), move(1, (0, 0), (0, .5), math.pi/2)]
        conditions = completion_conditions('Go straight.')
        checked, _ = validate_condition_checks(conditions, [row('P1.C1', 'satisfied', [1])], ledger)
        self.assertEqual(checked[0]['status'], 'unknown')
        conditions = completion_conditions('Turn right. Go straight.')
        rows = [row('P1.C1', 'satisfied', [0]), row('P1.C2', 'satisfied', [1])]
        checked, _ = validate_condition_checks(conditions, rows, ledger)
        self.assertEqual([c['status'] for c in checked], ['satisfied', 'satisfied'])
        previous = {c['id']: c for c in checked}
        rows[1]['status'] = 'unknown'
        retained, _ = validate_condition_checks(conditions, rows, ledger, previous=previous)
        self.assertEqual(retained[1]['status'], 'satisfied')

    def test_legacy_straight_uses_real_displacements_without_fake_heading(self):
        ledger = [move(1, (0, 0), (.5, 0)), move(2, (.5, 0), (1, 0))]
        self.assertTrue(straight_witness(ledger, ledger)[0])
        self.assertFalse(straight_witness([ledger[0]], [ledger[0]])[0])
        ledger[1]['pose_before'] = [2, 2, 0]
        self.assertFalse(straight_witness(ledger, ledger)[0])


class RoutePrefixTests(unittest.TestCase):
    def test_future_or_empty_contradictions_become_unknown(self):
        checks = completion_conditions('Enter the house. Turn left. Stop by the fireplace.')
        ledger = [record(1)]
        rows = [row('P1.C1', 'unknown', []), row('P1.C2', 'contradicted', [1]),
                row('P1.C3', 'contradicted', [])]
        actual, _ = validate_condition_checks(checks, rows, ledger)
        self.assertEqual([c['status'] for c in actual], ['unknown', 'unknown', 'unknown'])
        self.assertEqual(route_progress_packet(actual)['current_condition_id'], 'P1.C1')

    def test_episode_722_future_lefts_and_endpoint_are_not_wrong_route(self):
        a = Guard('When you leave the sitting area and go to the dinning area, make a left '
                  'then make another left near the outside door. Now go into the kitchen '
                  'and make a right. You should stop between the island and counter.')
        a.st['record'] = [record(1), record(2)]
        raw = verdict(a.prepare())
        raw.update(outcome=False, relation_satisfied=False, failure_type='wrong_path', scene_consistent=True,
                   contradiction_evidence='No left turns or kitchen entry in steps. Endpoint not reached.')
        for r in raw['condition_checks'][2:]:
            r.update(status='contradicted', steps=[])
        result = a.evaluate(raw)
        self.assertFalse(result['rollback_required'])
        self.assertEqual(result['failure_type'], 'not_reached')
        self.assertTrue(result['scene_consistent'])
        self.assertTrue(result['route_contradiction_gate']['requested'])
        self.assertFalse(result['route_contradiction_gate']['accepted'])
        self.assertTrue(all(c['status'] == 'unknown' for c in result['condition_checks'][2:]))

    def test_episode_1159_unseen_front_door_is_not_a_positive_route_contradiction(self):
        checks = completion_conditions('Walk through the door to the right of the fireplace.')
        checked, _ = validate_condition_checks(checks, [row('P1.C1', 'contradicted', [1])], [record(1)])
        raw = contradiction(expected='Walk through the door to the right of the fireplace',
                            observed='Fireplace visible in LEFT view; no doorway in FRONT view.')
        self.assertIsNone(check_route_contradiction(raw, checked, [record(1)])[0])

    def test_negative_verdict_keeps_measured_prefix_separate_from_completion(self):
        a = Guard('Walk down the hall. Turn left. Stop by the chair.')
        a.st['record'] = [record(1)]
        raw = verdict(a.prepare())
        raw.update(outcome=False, relation_satisfied=False, failure_type='not_reached')
        raw['condition_checks'][1:] = [row('P1.C2', 'unknown', []), row('P1.C3', 'unknown', [])]
        result = a.evaluate(raw)
        self.assertEqual(result['route_progress']['completed_prefix'], ['P1.C1'])
        self.assertFalse(a.st['completed'])
        self.assertTrue(all(c['status'] == 'unknown' for c in a.st['completion_conditions']))
        self.assertEqual(a.prepare()['checks'][0]['status'], 'satisfied')
        a.st['record'][0]['rejected_by_rollback'] = True
        self.assertEqual(a.prepare()['checks'][0]['status'], 'unknown')


class PositiveContradictionTests(unittest.TestCase):
    def setUp(self):
        self.ledger = [record(1)]
        self.checks, _ = validate_condition_checks(completion_conditions('Enter the kitchen. Stop by the counter.'),
            [row('P1.C1', 'contradicted', [1]), row('P1.C2', 'unknown', [])], self.ledger)

    def test_current_wrong_branch_can_recover_with_future_conditions_unfinished(self):
        for kind in ('wrong_room', 'wrong_branch'):
            with self.subTest(kind=kind):
                valid, reason = check_route_contradiction(contradiction(kind), self.checks, self.ledger)
                self.assertIsNotNone(valid, reason)

    def test_completed_return_synonyms_are_positive_transition_claims(self):
        for phrase in ('traveled back to', 'travelled back to', 'walked back to',
                       'moved back to', 'went back to', 'stepped back to',
                       'returned to', 'returned back to', 'returned into',
                       'traveled to', 'travelled through', 're-entered'):
            with self.subTest(phrase=phrase):
                raw = contradiction(observed=f'Agent {phrase} the living room with the fireplace.')
                valid, reason = check_route_contradiction(raw, self.checks, self.ledger)
                self.assertIsNotNone(valid, reason)

    def test_negated_hypothetical_and_future_transitions_cannot_authorize_recovery(self):
        phrases = (
            'Agent has not returned to the living room.',
            'Agent never returned to the living room.',
            "Agent hasn't returned to the living room.",
            'Agent has not yet traveled back to the living room.',
            'No evidence shows that the agent returned to the living room.',
            'If the agent returned to the living room, the fireplace would be visible.',
            'Agent returned to the living room if the previous room was the kitchen.',
            'Agent may have returned to the living room.',
            'Agent might have walked back to the living room.',
            'Agent could have returned to the living room.',
            'Agent should have returned to the living room.',
            'Agent will have returned to the living room after the next move.',
            'Agent must have returned to the living room.',
            'Perhaps the agent returned to the living room.',
            'Agent seems to have returned to the living room.',
            'Planned: agent returned to the living room.',
            'Agent returned to the living room is not confirmed.',
            'Agent returned to the living room is unconfirmed.',
            'Agent almost returned to the living room.',
            'Agent nearly walked back to the living room.',
            'It is false that the agent traveled back to the living room.',
            'Agent returned to the living room, but this transition was not observed.',
            'Agent returned to the living room. This transition was not verified.',
            'Insufficient evidence shows the agent returned to the living room.',
            'Agent returned to the living room?',
            # Existing entry verbs must not bypass these same factuality checks.
            'Agent has never entered the bedroom.',
            'If the agent entered the bedroom, it would see a bed.',
            'Agent would have crossed into the bedroom.',
            'Agent plans to have entered the bedroom next.',
        )
        for observed in phrases:
            with self.subTest(observed=observed):
                self.assertIsNone(check_route_contradiction(
                    contradiction(observed=observed), self.checks, self.ledger)[0])

    def test_return_intention_absence_or_heading_is_not_an_executed_transition(self):
        for observed in ('Return to the living room.', 'Agent will return to the living room.',
                         'Agent plans to move back to the living room.',
                         'Agent turned back to face the living room.',
                         'Agent is looking back toward the living room.',
                         'Target not visible after agent returned to the living room.'):
            with self.subTest(observed=observed):
                self.assertIsNone(check_route_contradiction(
                    contradiction(observed=observed), self.checks, self.ledger)[0])

    def test_asserted_return_still_requires_real_translation_and_valid_current_steps(self):
        raw = contradiction(observed='Agent returned to the living room with a fireplace.')
        for ledger in ([record(1, 'turn', motion=0., yaw=math.pi)],
                       [{**self.ledger[0], 'motion_distance': .01}],
                       [{**self.ledger[0], 'recovery': True}],
                       [{**self.ledger[0], 'rejected_by_rollback': True}]):
            self.assertIsNone(check_route_contradiction(raw, self.checks, ledger)[0])
        for steps in ([], [99], [True]):
            bad = contradiction(steps=steps, observed=raw['route_contradiction']['observed'])
            self.assertIsNone(check_route_contradiction(bad, self.checks, self.ledger)[0])
        bad = contradiction(cid='P1.C2', expected='Stop by the counter',
                            observed=raw['route_contradiction']['observed'])
        self.assertIsNone(check_route_contradiction(bad, self.checks, self.ledger)[0])
        raw['scene_consistent'] = True
        self.assertIsNone(check_route_contradiction(raw, self.checks, self.ledger)[0])

    def test_production_route_guard_accepts_return_wording_without_completing_future_goal(self):
        a = Guard('Enter the kitchen. Stop by the counter.')
        a.st['record'] = [record(1)]
        raw = verdict(a.prepare())
        raw.update(contradiction(observed='Agent traveled back to the living room with the fireplace.'))
        raw.update(outcome=False, relation_satisfied=False, failure_type='wrong_path',
                   contradiction_evidence=raw['route_contradiction']['observed'])
        raw['condition_checks'] = [row('P1.C1', 'contradicted', [1]), row('P1.C2', 'unknown', [])]
        result = a.evaluate(raw)
        self.assertTrue(result['rollback_required'])
        self.assertTrue(result['route_contradiction_gate']['accepted'])
        self.assertEqual(result['failure_type'], 'wrong_path')
        self.assertFalse(result['outcome'])
        self.assertFalse(result['completion_evidence_reliable'])
        self.assertEqual(result['condition_checks'][1]['status'], 'unknown')

    def test_saved_room_return_claim_and_measured_steps_pass_language_gate_only(self):
        # Minimal production snapshot: frozen 0c19c43 episode_1434_results.json,
        # sha256 c825d50c8907f585b03c0068dd84a0209d5e002470b2604d055f7bfe7ea2bf9f.
        # These are real room-internal motions, not proof of a room crossing.
        # The model's claimed return history still needs independent visual review.
        conditions = completion_conditions(
            'Walk to the right side of the couch and stand in front of a grill/oven.  '
            'Turn right and walk through the room with the microwaves, leaving on the opposite side.  '
            'Stop after stepping through the doorway out of the microwave room.')
        for condition, steps in zip(conditions[:3],
                                    ([1, 2, 4, 5, 8, 9, 10],
                                     [1, 2, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], [23, 24])):
            condition.update(status='satisfied', execution_checked=True, steps=steps)
        conditions[3].update(status='contradicted', execution_checked=False, steps=[70, 75])
        ledger = [
            {'step': 70, 'action_type': 'move_forward', 'motion_distance': .2,
             'pose_before': [.762, 1.357, -1.2], 'pose': [.628, 1.506, -1.2], 'recovery': False},
            {'step': 75, 'action_type': 'move_forward', 'motion_distance': .2,
             'pose_before': [.094, 2.102, -1.2], 'pose': [-.039, 2.251, -1.2], 'recovery': False},
        ]
        raw = contradiction(cid='P1.C4', steps=[70, 75],
            expected='walk through the room with the microwaves',
            observed='Agent traveled back to the living room area (visible fireplace and glass doors).')
        valid, reason = check_route_contradiction(raw, conditions, ledger)
        self.assertIsNotNone(valid, reason)
        self.assertEqual({k: valid[k] for k in raw['route_contradiction']}, raw['route_contradiction'])
        self.assertEqual(valid['recovery_constraint']['max_checkpoint_ledger_len'], 70)

    def test_structured_witness_does_not_override_consistent_scene(self):
        raw = contradiction()
        raw['scene_consistent'] = True
        self.assertIsNone(check_route_contradiction(raw, self.checks, self.ledger)[0])

    def test_missing_foreign_recovery_and_future_step_references_rejected(self):
        for steps in ([], [12], [True]):
            self.assertIsNone(check_route_contradiction(contradiction(steps=steps), self.checks, self.ledger)[0])
        for flag in ('recovery', 'rejected_by_rollback'):
            ledger = [{**self.ledger[0], flag: True}]
            self.assertIsNone(check_route_contradiction(contradiction(), self.checks, ledger)[0])
        raw = contradiction(cid='P1.C2', expected='Stop by the counter')
        self.assertIsNone(check_route_contradiction(raw, self.checks, self.ledger)[0])

    def test_unrelated_expected_or_absence_only_observation_rejected(self):
        for raw in [contradiction(expected='Exit the kitchen'),
                    contradiction(observed='Target not visible'),
                    contradiction(observed='Only moved 1.93m, later clauses not executed'),
                    contradiction(observed='Enter the kitchen'),
                    contradiction(observed='Fireplace visible in LEFT view')]:
            self.assertIsNone(check_route_contradiction(raw, self.checks, self.ledger)[0])

    def test_actual_opposite_turn_not_text_alone_can_trigger_midroute_recovery(self):
        conditions = completion_conditions('Walk down the hall. Turn left. Go to the chair.')
        ledger = [record(1), record(2, 'turn', motion=0., yaw=1.57)]
        rows = [row('P1.C1', 'satisfied', [1]), row('P1.C2', 'contradicted', [2]), row('P1.C3', 'unknown', [])]
        checked, _ = validate_condition_checks(conditions, rows, ledger)
        raw = contradiction('turn_direction', 'P1.C2', [2], 'Turn left', 'Executed a right turn at the hall junction.')
        self.assertIsNotNone(check_route_contradiction(raw, checked, ledger)[0])
        ledger[1]['yaw_delta'] = -1.57
        self.assertIsNone(check_route_contradiction(raw, checked, ledger)[0])

    def test_wrong_floor_requires_opposite_measured_elevation(self):
        conditions = completion_conditions('Go up the stairs.')
        ledger = [dict(record(1), height_delta=-.4)]
        checked, _ = validate_condition_checks(conditions, [row('P1.C1', 'contradicted', [1])], ledger)
        raw = contradiction('wrong_floor', expected='Go up the stairs', observed='Moved down to the lower landing.')
        self.assertIsNotNone(check_route_contradiction(raw, checked, ledger)[0])
        ledger[0]['height_delta'] = .4
        self.assertIsNone(check_route_contradiction(raw, checked, ledger)[0])


class StructuredContradictionEntryTests(unittest.TestCase):
    def case(self):
        a = Guard('Enter the kitchen. Stop by the counter.')
        a.st['record'] = [record(1)]
        raw = verdict(a.prepare())
        raw.update(contradiction())
        raw.update(outcome=False, relation_satisfied=False)
        raw.pop('failure_type')
        raw['condition_checks'] = [row('P1.C1', 'contradicted', [1]), row('P1.C2', 'unknown', [])]
        return a, raw

    def test_missing_failure_label_requests_and_passes_existing_evidence_gate(self):
        a, raw = self.case()
        before = copy.deepcopy(raw)
        result = a.evaluate(raw)
        self.assertTrue(result['route_contradiction_gate']['requested'])
        self.assertTrue(result['route_contradiction_gate']['accepted'])
        self.assertTrue(result['route_contradiction_gate']['failure_type_inferred'])
        self.assertTrue(result['rollback_required'])
        self.assertEqual(result['failure_type'], 'wrong_path')
        self.assertIn('recovery_constraint', result['route_contradiction'])
        self.assertFalse(result['outcome'])
        self.assertEqual(raw, before)

    def test_empty_malformed_or_prose_only_claim_does_not_request_recovery(self):
        claims = (None, {}, 'Entered the wrong room.', {'kind': 'wrong_room'},
                  {**contradiction()['route_contradiction'], 'steps': []},
                  {**contradiction()['route_contradiction'], 'steps': [True]},
                  {**contradiction()['route_contradiction'], 'observed': ''},
                  {**contradiction()['route_contradiction'], 'kind': ['wrong_room']},
                  {**contradiction()['route_contradiction'], 'kind': {'name': 'wrong_room'}},
                  {**contradiction()['route_contradiction'], 'kind': 'invented_kind'})
        for claim in claims:
            with self.subTest(claim=claim):
                a, raw = self.case()
                raw['route_contradiction'] = claim
                raw['reason'] = 'Wrong path; roll back now.'
                result = a.evaluate(raw)
                self.assertFalse(result['route_contradiction_gate']['requested'])
                self.assertFalse(result['rollback_required'])

    def test_explicit_wrong_path_with_malformed_kind_is_rejected_without_crashing(self):
        for kind in (['wrong_room'], {'name': 'wrong_room'}):
            with self.subTest(kind=kind):
                a, raw = self.case()
                raw['failure_type'] = 'wrong_path'
                raw['route_contradiction']['kind'] = kind
                result = a.evaluate(raw)
                self.assertTrue(result['route_contradiction_gate']['requested'])
                self.assertFalse(result['route_contradiction_gate']['accepted'])
                self.assertFalse(result['rollback_required'])

    def test_shaped_claim_cannot_bypass_missing_future_recovery_or_rejected_steps(self):
        for change in ('missing', 'future_condition', 'recovery', 'rejected_by_rollback'):
            with self.subTest(change=change):
                a, raw = self.case()
                if change == 'missing':
                    raw['route_contradiction']['steps'] = [99]
                elif change == 'future_condition':
                    raw['route_contradiction'].update(condition_id='P1.C2', expected='Stop by the counter')
                else:
                    a.st['record'][0][change] = True
                    a.prepare()
                result = a.evaluate(raw)
                self.assertTrue(result['route_contradiction_gate']['requested'])
                self.assertFalse(result['route_contradiction_gate']['accepted'])
                self.assertFalse(result['rollback_required'])

    def test_wrong_turn_claim_still_requires_actual_opposite_turn(self):
        for yaw, expected in ((1.57, True), (-1.57, False)):
            with self.subTest(yaw=yaw):
                a = Guard('Turn left. Walk to the counter.')
                a.st['record'] = [record(1, 'turn', motion=0., yaw=yaw)]
                raw = verdict(a.prepare())
                raw.update(contradiction('turn_direction', steps=[1], expected='Turn left',
                    observed='The robot executed a right turn.'))
                raw.update(outcome=False, relation_satisfied=False)
                raw.pop('failure_type')
                raw['condition_checks'] = [row('P1.C1', 'contradicted', [1]), row('P1.C2', 'unknown', [])]
                self.assertEqual(a.evaluate(raw)['rollback_required'], expected)

    def test_scene_confidence_and_completion_conflicts_still_forbid_recovery(self):
        for updates in ({'scene_consistent': True}, {'confidence': 'low'},
                        {'outcome': True}, {'relation_satisfied': True}):
            with self.subTest(updates=updates):
                a, raw = self.case()
                raw.update(updates)
                result = a.evaluate(raw)
                self.assertFalse(result['route_contradiction_gate']['accepted'])
                self.assertFalse(result['rollback_required'])

    def test_explicit_other_failure_label_is_not_silently_overridden(self):
        for failure in ('not_reached', 'none', 'unknown', None, ''):
            with self.subTest(failure=failure):
                a, raw = self.case()
                raw['failure_type'] = failure
                result = a.evaluate(raw)
                self.assertTrue(result['route_contradiction_gate']['requested'])
                self.assertFalse(result['route_contradiction_gate']['accepted'])
                self.assertFalse(result['route_contradiction_gate']['failure_type_inferred'])
                self.assertFalse(result['rollback_required'])
                self.assertIn('Explicit failure_type conflicts', result['route_contradiction_gate']['reason'])

    def test_explicit_wrong_path_retains_its_existing_authorization(self):
        a, raw = self.case()
        raw['failure_type'] = 'wrong_path'
        result = a.evaluate(raw)
        self.assertTrue(result['rollback_required'])
        self.assertFalse(result['route_contradiction_gate']['failure_type_inferred'])

    def test_saved_1263_missing_label_is_requested_but_inconsistent_citations_are_rejected(self):
        # Frozen331 fragment1263 step19: exact condition/contradiction fields.
        # Cited step16 is a real waypoint alignment; 17/18 are real .2m moves,
        # but the model bound only16 to C2. Do not repair that claim silently.
        a = Guard('Turn right at the eye chart and stand next to the bathroom door.')
        a.st['record'] = [record(6, 'turn', motion=0., yaw=.857),
                         record(16, 'turn', motion=0., yaw=1.446),
                         record(17, motion=.2), record(18, motion=.2)]
        raw = verdict(a.prepare())
        raw.pop('failure_type')
        raw.update(outcome=False, relation_satisfied=False, scene_consistent=False, confidence='high',
            reason='The agent has drifted into a dark shelving hallway, losing proximity to the bathroom door and turning away from the valid endpoint.',
            route_contradiction={'kind': 'wrong_branch', 'condition_id': 'P1.C2', 'steps': [16, 17, 18],
                'expected': 'stand next to the bathroom door',
                'observed': 'turned right into a shelving corridor (steps 16-18), moving away from the bathroom door area'},
            condition_checks=[row('P1.C1', 'satisfied', [6], 'Right turn executed at step 6 near the eye chart.'),
                row('P1.C2', 'contradicted', [16], 'Agent entered the shelving hallway at step 16, moving away from the bathroom door.')])
        result = a.evaluate(raw)
        self.assertTrue(result['route_contradiction_gate']['requested'])
        self.assertFalse(result['route_contradiction_gate']['accepted'])
        self.assertIn('must cite measured', result['route_contradiction_gate']['reason'])
        self.assertFalse(result['rollback_required'])
        self.assertIsNone(result['route_contradiction'])


if __name__ == '__main__':
    unittest.main()


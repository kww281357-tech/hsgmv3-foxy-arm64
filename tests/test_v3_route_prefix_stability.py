"""CPU regressions for stable historical progress and current endpoint evidence."""
import copy
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from navigation_contracts import completion_conditions, validate_condition_checks
from route_progress import route_progress_packet, turn_witness
from test_v3_verification_contract import Guard, record, verdict


def row(cid, steps, status='satisfied'):
    return {'id': cid, 'steps': steps, 'status': status,
            'evidence': 'Observed the instructed route landmark and measured action.'}


def turn(step, yaw):
    return record(step, 'turn', motion=0., yaw=yaw)


class HistoricalPrefixTests(unittest.TestCase):
    def setUp(self):
        self.conditions = completion_conditions(
            'Leave the lounge. Go to the hall. Turn left. Make another left. Enter the kitchen. Stop by the sink.')
        self.ledger = [record(1), record(2), turn(3, -1.), record(4), turn(5, -.6), record(6)]
        self.rows = [row(c['id'], [s]) for c, s in zip(self.conditions, [1, 2, 3, 5, 6, 6])]
        checked, conflict = validate_condition_checks(self.conditions, self.rows, self.ledger)
        self.assertFalse(conflict)
        self.previous = {c['id']: c for c in checked}

    def check(self, rows, ledger=None):
        return validate_condition_checks(self.conditions, rows,
                                         self.ledger if ledger is None else ledger,
                                         previous=self.previous)

    def test_new_legal_turn_citation_cannot_steal_later_milestone(self):
        drift = copy.deepcopy(self.rows)
        drift[2]['steps'] = [5]
        drift[3]['steps'] = [3]
        checked, conflict = self.check(drift)
        self.assertFalse(conflict)
        self.assertEqual([c['steps'] for c in checked[:5]], [[1], [2], [3], [5], [6]])
        self.assertTrue(all(c['status'] == 'satisfied' for c in checked))

    def test_chronological_rewording_does_not_rewrite_valid_history(self):
        drift = copy.deepcopy(self.rows)
        drift[1]['steps'] = [4, 6]
        checked, conflict = self.check(drift)
        self.assertFalse(conflict)
        self.assertEqual(route_progress_packet(checked)['completed_prefix'], [c['id'] for c in self.conditions])
        self.assertEqual(checked[1]['steps'], [2])

    def test_missing_reported_rows_still_fail_coverage(self):
        _, conflict = self.check(self.rows[:-1])
        self.assertTrue(conflict)

    def test_live_rollback_or_recovery_invalidates_old_prefix(self):
        unknown = [row(c['id'], [], 'unknown') for c in self.conditions]
        for flag in ('rejected_by_rollback', 'recovery'):
            with self.subTest(flag=flag):
                ledger = copy.deepcopy(self.ledger)
                ledger[2][flag] = True
                checked, _ = self.check(unknown, ledger)
                self.assertEqual(route_progress_packet(checked)['completed_prefix'], ['P1.C1', 'P1.C2'])
                self.assertEqual([c['status'] for c in checked[2:]], ['unknown'] * 4)

    def test_live_motion_failure_is_not_rescued_by_cached_execution_flag(self):
        ledger = copy.deepcopy(self.ledger)
        ledger[0]['motion_distance'] = 0.
        checked, _ = self.check([row(c['id'], [], 'unknown') for c in self.conditions], ledger)
        self.assertEqual(route_progress_packet(checked)['completed_prefix'], [])

    def test_invalid_duplicate_turns_in_cached_rows_are_revalidated_together(self):
        self.previous['P1.C3']['steps'] = [5]
        self.previous['P1.C4']['steps'] = [5]
        checked, _ = self.check([row(c['id'], [], 'unknown') for c in self.conditions])
        self.assertEqual(route_progress_packet(checked)['completed_prefix'], ['P1.C1', 'P1.C2', 'P1.C3'])


class GuardPrefixTests(unittest.TestCase):
    def historical_guard(self):
        guard = Guard('Enter the kitchen. Turn left. Stop beside the sink.')
        guard.st['record'] = [record(1), turn(2, -.7), record(3)]
        raw = verdict(guard.prepare())
        raw['condition_checks'] = [row('P1.C1', [1]), row('P1.C2', [2]), row('P1.C3', [], 'unknown')]
        raw.update(outcome=False, relation_satisfied=False, failure_type='not_reached')
        guard.evaluate(raw)
        self.assertEqual(guard.st['route_progress']['completed_prefix'], ['P1.C1', 'P1.C2'])
        return guard

    def test_positive_contradiction_revokes_affected_and_later_history(self):
        guard = self.historical_guard()
        guard.st['record'].append(record(4))
        raw = verdict(guard.prepare())
        raw.update(outcome=False, relation_satisfied=False, failure_type='wrong_path',
                   scene_consistent=False, contradiction_evidence='Entered a bedroom with a bed.',
                   route_contradiction={'kind': 'wrong_room', 'condition_id': 'P1.C1', 'steps': [4],
                                        'expected': 'Enter the kitchen', 'observed': 'Entered a bedroom with a bed.'})
        raw['condition_checks'] = [row('P1.C1', [4], 'contradicted'), row('P1.C2', [], 'unknown'),
                                   row('P1.C3', [], 'unknown')]
        result = guard.evaluate(raw)
        self.assertTrue(result['route_contradiction_gate']['accepted'])
        self.assertTrue(result['rollback_required'])
        self.assertEqual(guard.st['route_progress']['completed_prefix'], [])

    def test_absence_only_claim_does_not_erase_existing_prefix(self):
        guard = self.historical_guard()
        raw = verdict(guard.prepare())
        raw.update(outcome=False, relation_satisfied=False, failure_type='wrong_path', scene_consistent=False,
                   contradiction_evidence='The kitchen is not visible.',
                   route_contradiction={'kind': 'wrong_room', 'condition_id': 'P1.C1', 'steps': [1],
                                        'expected': 'Enter the kitchen', 'observed': 'Target not visible.'})
        raw['condition_checks'] = [row('P1.C1', [1], 'contradicted'), row('P1.C2', [], 'unknown'),
                                   row('P1.C3', [], 'unknown')]
        result = guard.evaluate(raw)
        self.assertFalse(result['route_contradiction_gate']['accepted'])
        self.assertFalse(result['rollback_required'])
        self.assertEqual(result['route_progress']['completed_prefix'], ['P1.C1', 'P1.C2'])
        self.assertEqual(guard.prepare()['checks'][1]['status'], 'satisfied')

    def test_compound_terminal_turn_never_restores_historical_arrival(self):
        conditions = completion_conditions('Turn right and stop beside the sink.', max_conditions=1)
        ledger = [turn(1, .8)]
        old, _ = validate_condition_checks(conditions, [row('P1.C1', [1])], ledger)
        checked, _ = validate_condition_checks(conditions, [row('P1.C1', [1])], ledger,
            previous={'P1.C1': old[0]}, require_current_arrival=True, current_pose=[3., 0., 0.])
        self.assertEqual(checked[0]['status'], 'unknown')
        self.assertIn('current arrival position', checked[0]['evidence'])

    def test_terminal_arrival_excluded_from_saved_and_legacy_prefix(self):
        guard = Guard('Walk to the chair. Stop beside it.')
        guard.st['record'] = [record(1)]
        result = guard.evaluate(verdict(guard.prepare()))
        self.assertTrue(result['completion_evidence_reliable'])
        self.assertEqual(result['route_progress']['completed_prefix'], ['P1.C1', 'P1.C2'])
        self.assertEqual(guard.st['route_progress']['completed_prefix'], ['P1.C1'])
        # Old snapshots may still contain a terminal condition. Do not import
        # that satisfied endpoint into a new verification context.
        guard.st['route_progress']['conditions'] = result['condition_checks']
        context = guard.prepare()
        self.assertEqual([c['status'] for c in context['checks']], ['satisfied', 'unknown'])


class PhysicalTurnBlockTests(unittest.TestCase):
    def test_real_uncited_translation_separates_sparse_turn_citations(self):
        ledger = [turn(1, .8), record(2), turn(3, -1.6)]
        self.assertTrue(turn_witness('Turn right.', [ledger[0], ledger[2]], ledger)[0])

    def test_uncited_opposite_turn_in_same_block_cancels_selected_turn(self):
        for selected in ([1], [1, 3]):
            with self.subTest(selected=selected):
                ledger = [turn(1, .8), turn(2, -1.0), turn(3, .2)]
                self.assertFalse(turn_witness('Turn right.', [r for r in ledger if r['step'] in selected], ledger)[0])

    def test_blocked_forward_does_not_separate_rotation_blocks(self):
        for motion in (0., 1e-5, .02):
            with self.subTest(motion=motion):
                ledger = [turn(1, .8), record(2, motion=motion), turn(3, -.8)]
                self.assertFalse(turn_witness('Turn right.', [ledger[0]], ledger)[0])

    def test_consecutive_micro_moves_accumulate_to_real_translation(self):
        ledger = [turn(1, .8), record(2, motion=.015), record(3, motion=.015), turn(4, -.8)]
        self.assertTrue(turn_witness('Turn right.', [ledger[0]], ledger)[0])

    def test_two_clauses_cannot_reuse_different_citations_of_same_rotation_block(self):
        conditions = completion_conditions('Turn left. Turn left.')
        ledger = [turn(1, -.6), turn(2, -.6)]
        checked, _ = validate_condition_checks(conditions, [row('P1.C1', [1]), row('P1.C2', [2])], ledger)
        self.assertEqual([c['status'] for c in checked], ['satisfied', 'unknown'])

    def test_rollback_filtered_turn_cannot_supply_missing_reversal(self):
        ledger = [turn(1, .8), {**turn(2, -.8), 'rejected_by_rollback': True}]
        self.assertTrue(turn_witness('Turn right.', [ledger[0]], ledger)[0])

    def test_recovery_motion_separates_surviving_turns_from_different_attempts(self):
        ledger = [turn(1, .8), {**record(2), 'recovery': True}, turn(3, -.8)]
        self.assertTrue(turn_witness('Turn right.', [ledger[0], ledger[2]], ledger)[0])

    def test_unknown_step_gap_cannot_join_partial_turns(self):
        ledger = [turn(1, -.2), turn(3, -.2)]
        self.assertFalse(turn_witness('Turn left.', ledger, ledger)[0])

    def test_large_left_rotation_does_not_witness_a_right_turn(self):
        ledger = [turn(1, -math.pi/2), turn(2, -math.pi/2), turn(3, -math.pi/2)]
        self.assertFalse(turn_witness('Turn right.', ledger, ledger)[0])
        self.assertTrue(turn_witness('Turn left.', ledger, ledger)[0])

    def test_rounded_half_circle_keeps_measured_direction(self):
        for sign, direction, opposite in ((-1, 'left', 'right'), (1, 'right', 'left')):
            with self.subTest(direction=direction):
                ledger = [turn(1, sign * 1.571), turn(2, sign * 1.571)]
                self.assertTrue(turn_witness(f'Turn {direction}.', ledger, ledger)[0])
                self.assertFalse(turn_witness(f'Turn {opposite}.', ledger, ledger)[0])
                self.assertTrue(turn_witness('Turn around.', ledger, ledger)[0])

    def test_complete_revolution_supplies_no_directional_progress(self):
        for sign in (-1, 1):
            ledger = [turn(step, sign * math.pi/2) for step in range(1, 5)]
            for instruction in ('Turn left.', 'Turn right.', 'Turn around.', 'Rotate.'):
                self.assertFalse(turn_witness(instruction, ledger, ledger)[0])


if __name__ == '__main__':
    unittest.main()


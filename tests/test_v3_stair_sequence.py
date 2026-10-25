"""Measured stair phase sequencing; no simulator, model or completed-route claims."""
import ast
import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import numpy as np

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
from navigation_contracts import completion_conditions
from navigation_control import StairProgress, pending_stair_condition, stair_condition_intents, check_stair_witnesses

tree = ast.parse((SRC / 'agent/agent.py').read_text())
method = next(node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef) and node.name == '_ensure_stair_progress')
namespace = {'np': np, 'completion_conditions': completion_conditions,
             'StairProgress': StairProgress, 'pending_stair_condition': pending_stair_condition,
             'stair_condition_intents': stair_condition_intents}
exec(compile(ast.Module(body=[method], type_ignores=[]), '<production stair method>', 'exec'), namespace)


class Controller:
    _ensure_stair_progress = namespace['_ensure_stair_progress']

    def __init__(self, instruction='Go up the stairs. Cross the doorway. Go down the stairs.'):
        self.st = {'instruction': instruction, 'record': []}
        self.mapper = NS(current_position=np.array([0., 0., 1.2]), initial_floor_height=-1.2)
        self.navigation_cfg = {'stair_landing_observations': 2, 'require_strict_stair_landing': True}
        self.instruction_obj = NS(get_current_subtask=lambda: ('P1', self.st),
            sub_instruction_keys=['P1'], current_subtask_index=0)

    def append(self, height, supported=False, kind='move_forward'):
        previous = float(self.mapper.current_position[2] - 1.2)
        step = len(self.st['record'])
        before_x = float(self.mapper.current_position[0])
        after_x = before_x + (.1 if kind == 'move_forward' else 0.)
        rec = {'step': step, 'action_type': kind, 'executed_action': kind,
               'pose_before': [before_x, 0., previous], 'pose': [after_x, 0., height],
               'height_delta': height - previous, 'motion_distance': math.hypot(height - previous, after_x - before_x),
               'landing_supported': supported}
        self.st['record'].append(rec)
        self.mapper.current_position = np.array([rec['pose'][0], 0., height + 1.2])
        return self._ensure_stair_progress()

    def up(self):
        self.append(.5)
        self.append(.5, True, 'turn')
        return self.append(.5, True, 'turn')


class StairSequenceTests(unittest.TestCase):
    def test_take_stairs_to_bottom_activates_descent_and_requires_a_supported_landing(self):
        a = Controller('take the stairs then head for the kitchen. stop at the bottom of the stairs.')
        a.mapper.current_position[2] = 0.
        state = a._ensure_stair_progress()
        self.assertEqual(state.direction, -1)
        self.assertEqual(state.direction_source, 'stated_landing_endpoint')
        self.assertTrue(state.landing_required)
        self.assertFalse(state.completion_ready())
        state = a.append(-1.7)
        self.assertEqual(state.phase, 'traverse')
        self.assertFalse(state.completion_ready())
        a.append(-1.7, True, 'turn')
        self.assertTrue(a.append(-1.7, True, 'turn').completion_ready())

    def test_stair_endpoint_does_not_invent_a_flight_and_hint_does_not_cross_next_flight(self):
        self.assertEqual(stair_condition_intents(completion_conditions('Stand at the bottom of the stairs.')), [])
        conditions = completion_conditions('Take the stairs. Then go up the steps. Stop at the top of the stairs.')
        clauses = stair_condition_intents(conditions)
        self.assertEqual(len(clauses), 1)
        self.assertEqual(clauses[0][1]['direction'], 1)
        conflicting = completion_conditions('Take the stairs. Stop at the bottom of the stairs. Wait at the top of the stairs.')
        self.assertTrue(stair_condition_intents(conflicting)[0][1]['ambiguous'])

    def test_inferred_descent_still_requires_actual_downward_motion(self):
        conditions = completion_conditions('Take the stairs. Stop at the bottom of the stairs.')
        conditions[0].update(status='satisfied', steps=[0], execution_checked=True)
        row = {'step': 0, 'action_type': 'move_forward', 'height_delta': .4}
        self.assertEqual(check_stair_witnesses(conditions, [row])[0]['status'], 'unknown')
        row['height_delta'] = -.4
        self.assertEqual(check_stair_witnesses(conditions, [row])[0]['status'], 'satisfied')

    def test_supported_first_landing_switches_direction_without_semantic_completion(self):
        a = Controller()
        first = a._ensure_stair_progress()
        original = copy.deepcopy(a.st['completion_conditions'])
        self.assertEqual(first.direction, 1)
        second = a.up()
        self.assertEqual(second.condition_id, 'P1.C3')
        self.assertEqual(second.direction, -1)
        self.assertAlmostEqual(second.start_height, .5)
        self.assertEqual(a.st['completion_conditions'], original)
        self.assertNotIn('completed', a.st)
        self.assertEqual(a.st['stair_execution_milestones'][0]['condition_id'], 'P1.C1')

    def test_height_alone_or_one_supported_sample_cannot_switch(self):
        a = Controller()
        a.navigation_cfg['require_strict_stair_landing'] = False
        self.assertEqual(a.append(.5).direction, 1)
        state = a.append(.5, True, 'turn')
        self.assertEqual(state.direction, 1)
        self.assertFalse(state.route_allowed([[.2, 0, .5], [.3, 0, 0.]]))

    def test_second_descent_uses_its_own_landing_start_height(self):
        a = Controller()
        a.up()
        state = a.append(0.)
        self.assertAlmostEqual(state.start_height, .5)
        self.assertAlmostEqual(state.last_height, 0.)
        self.assertFalse(state.completion_ready())
        a.append(0., True, 'turn')
        state = a.append(0., True, 'turn')
        self.assertTrue(state.completion_ready())
        self.assertEqual(state.phase, 'landing')
        self.assertEqual(len(a.st['stair_execution_milestones']), 2)

    def test_repeated_candidate_queries_do_not_create_stable_samples(self):
        a = Controller()
        a.append(.5)
        state = a.append(.5, True, 'turn')
        for _ in range(20):
            self.assertIs(a._ensure_stair_progress(), state)
        self.assertEqual(state.stable_samples, 1)
        self.assertEqual(state.direction, 1)

    def test_rejected_rollback_records_do_not_keep_a_completed_phase(self):
        a = Controller()
        a.up()
        for rec in a.st['record']:
            rec['rejected_by_rollback'] = True
        a.mapper.current_position = np.array([0., 0., 1.2])
        state = a._ensure_stair_progress()
        self.assertEqual(state.direction, 1)
        self.assertFalse(state.completion_ready())
        self.assertEqual(a.st['stair_execution_milestones'], [])

    def test_rebuild_from_actual_ledger_preserves_second_phase(self):
        a = Controller()
        a.up()
        a.append(.2)
        a._stair_progress = None
        state = a._ensure_stair_progress()
        self.assertEqual(state.direction, -1)
        self.assertAlmostEqual(state.start_height, .5)
        self.assertAlmostEqual(state.last_height, .2)

    def test_counted_steps_can_switch_without_inventing_a_platform(self):
        a = Controller('Go up two steps. Then go down the stairs.')
        state = a.append(.12)
        self.assertEqual(state.direction, -1)
        self.assertAlmostEqual(state.start_height, .12)

    def test_completed_verified_handoff_seeds_only_contiguous_nearby_motion(self):
        a = Controller('Go up the stairs.')
        prev = {'completed': True, 'completion_evidence_reliable': True, 'record': [
            {'step': 0, 'action_type': 'move_forward', 'executed_action': 'move_forward',
             'height_delta': .5, 'motion_distance': .7, 'pose_before': [0., 0., 0.],
             'pose': [.5, 0., .5], 'landing_supported': False}]}
        a.st['position_anchor_check'] = {'status': 'aligned'}
        a.mapper.current_position = np.array([.5, 0., 1.7])
        a.instruction_obj = NS(get_current_subtask=lambda: ('P2', a.st),
            sub_instruction_keys=['P1', 'P2'], current_subtask_index=1,
            get_subtask_by_key=lambda key: prev if key == 'P1' else a.st)
        state = a._ensure_stair_progress()
        self.assertEqual(state.handoff_steps, [0])
        self.assertAlmostEqual(state.start_height, 0.)
        self.assertAlmostEqual(state.last_height, .5)


if __name__ == '__main__':
    unittest.main()


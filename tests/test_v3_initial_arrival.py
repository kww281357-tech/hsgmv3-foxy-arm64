"""Current-frame arrival evidence without invented executed actions (CPU only)."""
import unittest

from test_v3_boundary_height import agent, boundary, verdict, ACTIONS
from test_v3_endpoint_integration import grounded_guard
from test_v3_verification_contract import Guard, record
from navigation_contracts import completion_conditions, validate_condition_checks, verification_schema
from verification_feedback import initial_arrival_observation_retry_required


def arrival_row(observation_id=5, steps=None, status='satisfied'):
    return {'id': 'P1.C1', 'status': status, 'evidence': 'The current view locates the target beside the agent.',
            'steps': [] if steps is None else steps, 'observation_id': observation_id}


class CurrentObservationArrivalTests(unittest.TestCase):
    def check(self, row, instruction='Stop beside the chair.', ledger=None):
        return validate_condition_checks(
            completion_conditions(instruction), [row], ledger or [],
            require_current_arrival=True, current_pose=[0., 0., 0.], current_observation_id=5)

    def test_empty_ledger_current_observation_is_audited_separately(self):
        rows, conflict = self.check(arrival_row())
        self.assertFalse(conflict)
        self.assertEqual(rows[0]['status'], 'satisfied')
        self.assertTrue(rows[0]['observation_checked'])
        self.assertFalse(rows[0]['execution_checked'])
        self.assertEqual(rows[0]['steps'], [])
        self.assertEqual(rows[0]['observation_id'], 5)

    def test_missing_stale_boolean_future_and_malformed_refs_never_pass(self):
        for row in (arrival_row(None), arrival_row(4), arrival_row(True),
                    arrival_row(5, [1]), arrival_row(5, ['future']), arrival_row(5, '[]')):
            with self.subTest(row=row):
                rows, _ = self.check(row)
                self.assertNotEqual(rows[0]['status'], 'satisfied')
                self.assertFalse(rows[0]['observation_checked'])
        rows, _ = self.check(arrival_row(status='unknown'))
        self.assertEqual(rows[0]['status'], 'unknown')

    def test_observation_does_not_prove_motion_turn_or_prior_route(self):
        for text in ('Walk to the chair.', 'Turn left.', 'Walk straight and stop beside the chair.'):
            rows, _ = self.check(arrival_row(), instruction=text)
            self.assertNotEqual(rows[0]['status'], 'satisfied')
        a, raw = grounded_guard()
        a.st['record'] = []
        a.prepare()
        raw['condition_checks'][0].update(steps=[], observation_id=None)
        raw['condition_checks'][-1].update(steps=[], observation_id=5)
        self.assertFalse(a.evaluate(raw)['outcome'])

    def test_fresh_metadata_preserves_valid_executed_arrival_and_motion(self):
        for text, kind, yaw in (('Stop beside the chair.', 'move_forward', 0.),
                                ('Walk to the chair.', 'move_forward', 0.),
                                ('Turn right.', 'turn', .8)):
            with self.subTest(instruction=text):
                ledger = [record(1, kind, pose=(0., 0., 0.), yaw=yaw,
                                 motion=.5 if kind == 'move_forward' else 0.)]
                rows, conflict = self.check(arrival_row(5, [1], 'executed'), text, ledger)
                self.assertFalse(conflict)
                self.assertEqual(rows[0]['status'], 'satisfied')
                self.assertTrue(rows[0]['execution_checked'])
                self.assertFalse(rows[0]['observation_checked'])
                self.assertIsNone(rows[0]['observation_id'])

    def test_fresh_metadata_cannot_rescue_future_step_wrong_turn_or_blocked_motion(self):
        cases = [
            ('Stop beside the chair.', [2], [record(1, pose=(0., 0., 0.))]),
            ('Turn right.', [1], [record(1, 'turn', motion=0., yaw=-.8)]),
            ('Walk to the chair.', [1], [record(1, motion=0.)]),
            ('Walk straight.', [1], [dict(record(1, pose=(0., .5, 0.)), yaw_before=0., yaw_after=0.)]),
        ]
        for text, steps, ledger in cases:
            with self.subTest(instruction=text):
                rows, _ = self.check(arrival_row(5, steps, 'executed'), text, ledger)
                self.assertEqual(rows[0]['status'], 'unknown')
                self.assertFalse(rows[0]['execution_checked'])
                self.assertFalse(rows[0]['observation_checked'])

    def test_fresh_metadata_cannot_replace_an_explicit_past_arrival_pose(self):
        rows, conflict = self.check(arrival_row(5, [1]), ledger=[record(1, pose=(3., 0., 0.))])
        self.assertFalse(conflict)
        self.assertEqual(rows[0]['status'], 'unknown')
        self.assertIn('current arrival position', rows[0]['evidence'])
        self.assertFalse(rows[0]['observation_checked'])

    def test_invalid_metadata_rejects_even_an_otherwise_valid_executed_witness(self):
        for observation_id in (4, 6, -1, True, 5., '5'):
            with self.subTest(observation_id=observation_id):
                rows, conflict = self.check(arrival_row(observation_id, [1]),
                                            ledger=[record(1, pose=(0., 0., 0.))])
                self.assertTrue(conflict)
                self.assertEqual(rows[0]['status'], 'unknown')
                self.assertFalse(rows[0]['execution_checked'])

    def test_auxiliary_observation_is_not_cached_as_a_stale_historical_binding(self):
        conditions = completion_conditions('Walk to the chair. Stop beside the chair.')
        ledger = [record(1)]
        reported = [dict(arrival_row(5, [1], 'executed'), id=c['id']) for c in conditions]
        first, conflict = validate_condition_checks(
            conditions, reported, ledger, require_current_arrival=True,
            current_pose=[.5, 0., 0.], current_observation_id=5)
        self.assertFalse(conflict)
        self.assertTrue(all(c['observation_id'] is None for c in first))
        reported = [dict(arrival_row(None, [], 'unknown'), id='P1.C1'),
                    dict(arrival_row(6, [1]), id='P1.C2')]
        second, conflict = validate_condition_checks(
            conditions, reported, ledger, previous={c['id']: c for c in first},
            require_current_arrival=True, current_pose=[.5, 0., 0.], current_observation_id=6)
        self.assertFalse(conflict)
        self.assertTrue(all(c['status'] == 'satisfied' for c in second))
        self.assertTrue(all(c['observation_id'] is None for c in second))
        self.assertEqual(second[0]['steps'], [1])

    def test_auxiliary_metadata_does_not_enable_nonterminal_observation_substitution(self):
        conditions = completion_conditions('Stop beside the chair.')
        checked, conflict = validate_condition_checks(
            conditions, [arrival_row()], [], current_pose=[0., 0., 0.], current_observation_id=5)
        self.assertFalse(conflict)
        self.assertFalse(checked[0]['observation_checked'])
        # Production nonterminal evaluation does not provide a current frame ID.
        checked, conflict = validate_condition_checks(conditions, [arrival_row(5, [1])], [record(1)])
        self.assertTrue(conflict)
        self.assertEqual(checked[0]['status'], 'unknown')

    def test_current_mask_geometry_can_stop_without_history_even_if_config_disables_grounding(self):
        a, raw = grounded_guard('Stop beside the chair.')
        a.st['record'] = []
        a.navigation_cfg['require_endpoint_grounding'] = False
        a.prepare()
        raw['condition_checks'] = [arrival_row()]
        result = a.evaluate(raw)
        self.assertTrue(result['completion_evidence_reliable'], result['reason'])
        self.assertTrue(result['endpoint_grounding']['accepted'])
        self.assertFalse(result['condition_checks'][0]['execution_checked'])
        self.assertEqual(a.st.get('route_progress', {}).get('completed_prefix'), [])
        a.curr_obs['left']['depth_sensor'][:] = 4.
        result = a.evaluate(raw)
        self.assertFalse(result['outcome'])
        self.assertEqual(result['endpoint_grounding']['gate_reason'], 'target_surface_too_far')

    def test_context_frame_must_still_match_live_observation(self):
        a, raw = grounded_guard('Stop beside the chair.')
        a.st['record'] = []
        a.prepare()
        raw['condition_checks'] = [arrival_row()]
        a.mapper.observation_id = 6
        result = a.evaluate(raw)
        self.assertFalse(result['outcome'])
        self.assertFalse(result['condition_checks'][0]['observation_checked'])

    def test_whole_image_doorway_and_missing_second_jamb_cannot_stop(self):
        for bbox in ([0., 0., 1., 1.], [.01, .1, .12, .9]):
            a, raw = grounded_guard('Stop in the doorway to the left of the white statue.')
            a.st['record'] = []
            a.navigation_cfg['require_endpoint_grounding'] = False
            a.prepare()
            raw.update(target_entity_id=None, target_view='front', target_bbox=bbox, reference_region=None)
            raw['condition_checks'] = [arrival_row()]
            result = a.evaluate(raw)
            self.assertFalse(result['outcome'])
            self.assertIn('endpoint_grounding', result)
            self.assertFalse(result['endpoint_grounding'].get('accepted', False))

    def test_prompt_and_schema_expose_real_frame_without_fake_step_example(self):
        a = Guard('Stop in the doorway.')
        a.mapper.observation_id = 5
        prompt = a._generate_verification_prompt(a.prepare(), None)
        self.assertIn('CURRENT observation_id (controller-provided, not an execution step): 5', prompt)
        self.assertIn('Never invent step 0/1', prompt)
        self.assertNotIn('"steps": [1]', prompt)
        props = verification_schema(completion_conditions(a.instruction), [])['schema']['properties']
        row_schema = props['reasoning']['properties']['condition_checks']['items']
        self.assertIn('observation_id', row_schema['required'])


class InitialObservationHoldTests(unittest.TestCase):
    def pair(self):
        conditions = completion_conditions('Stop in the doorway.')
        first = verdict(outcome=False, relation_satisfied=False, completion_evidence_reliable=False,
                        evidence_repair_required=True, initial_arrival_observation_eligible=True,
                        current_observation_id=5, condition_checks=conditions, failure_type='not_reached')
        second = verdict(outcome=False, relation_satisfied=False, completion_evidence_reliable=False,
                         current_observation_id=5, condition_checks=conditions, failure_type='not_reached',
                         correction_action={'type': 'navigate', 'waypoint_id': 'WP_01'})
        return first, second

    def test_same_frame_flip_buys_only_one_look_then_handback_not_move(self):
        for is_probe in (False, True):
            with self.subTest(is_probe=is_probe):
                first, second = self.pair()
                a = agent()
                a.mapper.observation_id = 5
                dispatch = lambda: a._boundary_stop_decision(
                    {'_controller_probe': is_probe}, None, [], None, {}, ACTIONS)
                a.current_path, a.action_sequence, a._waypoint_execution = [[1, 0, 0]], [ACTIONS['WP_01']], {'old': True}
                a._verify_current_subtask.side_effect = [first, second, dict(second, current_observation_id=6)]
                self.assertEqual(dispatch()['type'], 'turn')
                self.assertEqual(a.current_path, [])
                self.assertIsNone(a._waypoint_execution)
                self.assertEqual(a._verify_current_subtask.call_count, 2)
                self.assertTrue(a._pending_endpoint_relocalization['observation_only'])
                a.mapper.observation_id = 6
                a._observation_revision = 1
                self.assertEqual(dispatch()['type'], 'noop')
                self.assertEqual(a._endpoint_relocalization_turns['P1'], 1)
                self.assertEqual(a._verify_current_subtask.call_count, 3)

    def test_future_frame_wrong_route_api_failure_or_measured_far_target_never_gets_this_hold(self):
        first, second = self.pair()
        for changes in ({'current_observation_id': 6}, {'rollback_required': True},
                        {'available': False}, {'route_contradiction_gate': {'accepted': True}},
                        {'endpoint_grounding': {'valid': True, 'distance_m': 4.}},
                        {'condition_checks': [{'status': 'contradicted'}]}):
            self.assertFalse(initial_arrival_observation_retry_required(first, dict(second, **changes), 5))
        self.assertFalse(initial_arrival_observation_retry_required(
            dict(first, initial_arrival_observation_eligible=False), second, 5))


if __name__ == '__main__':
    unittest.main()


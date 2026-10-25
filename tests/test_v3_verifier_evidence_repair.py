"""Exercise production dispatch and prompt code without models or a simulator."""
import copy
import json
import unittest
from unittest.mock import Mock

from test_v3_boundary_height import agent, boundary, verdict, ACTIONS, NavigationAborted
from test_v3_verification_contract import Guard, record
from verification_feedback import evidence_repair_required, remember_verification


def citation_rejection():
    return verdict(outcome=False, relation_satisfied=False, completion_evidence_reliable=False,
                   evidence_state='unknown', confidence='low', evidence_repair_required=True,
                   reason='Every route condition requires satisfied execution evidence.',
                   condition_checks=[{'id': 'P1.C1', 'text': 'Turn left.', 'status': 'unknown',
                                      'steps': [3], 'execution_checked': False,
                                      'evidence': 'Referenced measured turns do not match the instructed direction and magnitude.'}])


class EvidenceRepairTests(unittest.TestCase):
    def test_same_pose_retry_makes_a_new_call_and_can_pass_full_gate(self):
        a = agent()
        a._verify_current_subtask.side_effect = [citation_rejection(), verdict()]
        a.runtime_budget = Mock()
        self.assertEqual(boundary(a)['type'], 'stop')
        self.assertEqual(a._verify_current_subtask.call_count, 2)
        self.assertEqual(a._boundary_checks['P1']['calls'], 2)
        self.assertEqual(a.runtime_budget.check.call_count, 2)
        self.assertFalse(getattr(a, '_verifier_corrections', {}))
        self.assertEqual([r['event'] for r in a.navigation_trace], ['verification_evidence_repair'])

    def test_failed_repair_hands_back_without_turn_and_cached_replay_is_bounded(self):
        a = agent(citation_rejection())
        a.navigation_cfg['max_verifier_corrections_per_subtask'] = 2
        self.assertEqual(boundary(a)['type'], 'noop')
        self.assertEqual(a._verify_current_subtask.call_count, 2)
        self.assertEqual(boundary(a)['type'], 'noop')
        self.assertEqual(a._verify_current_subtask.call_count, 2)
        with self.assertRaisesRegex(NavigationAborted, 'verifier_correction_budget_exhausted'):
            boundary(a)
        self.assertEqual(a._verify_current_subtask.call_count, 2)
        self.assertTrue(all(r.get('executed_type') != 'turn' for r in a.navigation_trace))

    def test_repair_cannot_bypass_verifier_call_limit(self):
        a = agent(citation_rejection())
        a.navigation_cfg['max_verifier_checks_per_subtask'] = 1
        self.assertEqual(boundary(a)['type'], 'noop')
        self.assertEqual(a._verify_current_subtask.call_count, 1)
        a._observation_revision = 1
        with self.assertRaisesRegex(NavigationAborted, 'verifier_check_budget_exhausted'):
            boundary(a)
        self.assertEqual(a._verify_current_subtask.call_count, 1)

    def test_repair_cannot_bypass_runtime_budget(self):
        a = agent(citation_rejection())
        a.runtime_budget = Mock()
        a.runtime_budget.check.side_effect = [None, NavigationAborted('request_budget')]
        with self.assertRaisesRegex(NavigationAborted, 'request_budget'):
            boundary(a)
        self.assertEqual(a._verify_current_subtask.call_count, 1)

    def test_negative_probe_can_continue_route_after_bounded_repair(self):
        a = agent(citation_rejection())
        self.assertEqual(a._boundary_stop_decision(
            {'_controller_probe': True}, None, [], None, {}, ACTIONS)['type'], 'continue_path')
        self.assertEqual(a._verify_current_subtask.call_count, 2)
        self.assertFalse(getattr(a, '_verifier_corrections', {}))

    def test_real_not_reached_still_uses_measured_movement_correction(self):
        result = verdict(outcome=False, completion_evidence_reliable=False,
                         failure_type='not_reached', correction_action={'type': 'navigate', 'waypoint_id': 'WP_01'})
        a = agent(result)
        self.assertEqual(boundary(a)['type'], 'waypoint')
        self.assertEqual(a._verify_current_subtask.call_count, 1)

    def test_only_affirmative_citation_veto_enters_repair(self):
        good = {'outcome': True, 'failure_type': 'none'}
        rejection = citation_rejection()
        self.assertTrue(evidence_repair_required(good, rejection))
        for parsed in ({'outcome': False, 'failure_type': 'not_reached'},
                       {'outcome': False, 'failure_type': 'wrong_path'},
                       {'outcome': True, 'failure_type': 'wrong_path'}):
            self.assertFalse(evidence_repair_required(parsed, rejection))
        for reason in ('target_surface_too_far', 'changed_view_confirmation_required',
                       'Current endpoint identity and stopping relation are not established.'):
            self.assertFalse(evidence_repair_required(good, dict(rejection, reason=reason)))
        contradicted = copy.deepcopy(rejection)
        contradicted['condition_checks'][0]['status'] = 'contradicted'
        self.assertFalse(evidence_repair_required(good, contradicted))

    def test_prompt_exposes_normalized_diagnostics_from_both_call_types(self):
        a = Guard()
        a.st['record'] = [record(1)]
        history = {}
        for step, trigger in enumerate(('completion_proposal', 'terminal_probe', 'completion_proposal')):
            history = remember_verification(history, 'P1', citation_rejection(),
                                            trigger=trigger, step=step, pose=[.5, 0, 1.2])
        a._verification_history = history
        prompt = a._generate_verification_prompt(a.prepare(), None)
        self.assertEqual(len(history['P1']), 2)
        self.assertIn('terminal_probe', prompt)
        self.assertIn('completion_proposal', prompt)
        self.assertIn('Referenced measured turns do not match', prompt)
        self.assertIn('P1.C1', prompt)
        self.assertIn('not new route evidence', prompt)
        self.assertEqual(json.loads(json.dumps(history)), history)

    def test_production_verifier_persists_each_normalized_result(self):
        a = agent()
        del a._verify_current_subtask
        a._get_forward_clearance_and_waypoints = Mock(return_value=(None, []))
        seen_history = []
        a._verification_prompt = lambda *args, **kwargs: seen_history.append(
            copy.deepcopy(getattr(a, '_verification_history', {}))) or 'prompt'
        a._get_verifier_response = Mock(return_value=json.dumps({'outcome': True, 'failure_type': 'none'}))
        a._evaluate_route_verification = Mock(return_value=citation_rejection())
        a.instruction_obj.update_verifier = Mock()
        for proposal in ({'_controller_probe': True}, {}):
            result = a._verify_current_subtask(proposal, None, [], None, obs={}, actions=ACTIONS)
            self.assertTrue(result['evidence_repair_required'])
        self.assertEqual(seen_history[0], {})
        self.assertEqual(seen_history[1]['P1'][0]['trigger'], 'terminal_probe')
        self.assertEqual([r['trigger'] for r in a._verification_history['P1']],
                         ['terminal_probe', 'completion_proposal'])

    def test_depth_rejection_feedback_keeps_selected_regions_without_pixel_arrays(self):
        result = citation_rejection()
        result['endpoint_grounding'] = {
            'valid': False, 'reason': 'insufficient_door_frame_support', 'view': 'front',
            'bbox': [.2, .1, .8, .9], 'depth_pixels': object(),
            'frame_support': {'left': {'valid': False, 'reason': 'mixed_target_depth',
                                       'distance_m': .8, 'depth_pixels': object()}}}
        result['endpoint_reference_grounding'] = {
            'valid': True, 'source': 'vlm_bbox', 'view': 'right', 'distance_m': 2.8}
        a = Guard()
        a.st['record'] = [record(1)]
        a._verification_history = remember_verification(
            {}, 'P1', result, trigger='completion_proposal', step=2, pose=[.5, 0, 1.2])
        prompt = a._generate_verification_prompt(a.prepare(), None)
        self.assertIn('insufficient_door_frame_support', prompt)
        self.assertIn('mixed_target_depth', prompt)
        self.assertIn('"distance_m": 2.8', prompt)
        self.assertIn('each region must isolate one jamb', prompt)
        self.assertNotIn('depth_pixels', prompt)
        json.dumps(a._verification_history)  # Pixel data must never enter the prompt/history.

    def test_same_pose_dispatch_passes_first_diagnostics_to_second_real_prompt(self):
        for is_probe in (False, True):
            with self.subTest(is_probe=is_probe):
                a = agent()
                del a._verify_current_subtask
                a._get_forward_clearance_and_waypoints = Mock(return_value=(None, []))
                prompts = []
                a._verification_prompt = lambda *args, **kwargs: prompts.append(
                    copy.deepcopy(getattr(a, '_verification_history', {}))) or 'prompt'
                a._get_verifier_response = Mock(return_value=json.dumps({'outcome': True, 'failure_type': 'none'}))
                a._evaluate_route_verification = Mock(side_effect=[citation_rejection(), verdict()])
                a.instruction_obj.update_verifier = Mock()
                decision = a._boundary_stop_decision(
                    {'_controller_probe': is_probe}, None, [], None, {}, ACTIONS)
                self.assertEqual(decision['type'], 'stop')
                self.assertEqual(a._get_verifier_response.call_count, 2)
                self.assertEqual(prompts[0], {})
                saved = prompts[1]['P1'][0]
                self.assertEqual(saved['trigger'], 'terminal_probe' if is_probe else 'completion_proposal')
                self.assertIn('Referenced measured turns', saved['condition_checks'][0]['diagnostic'])


if __name__ == '__main__':
    unittest.main()


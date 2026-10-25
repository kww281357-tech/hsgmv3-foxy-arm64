"""Production candidate, parser and depth-binding tests without model imports."""
import copy
import json
import unittest
from types import MethodType
from unittest.mock import Mock

import numpy as np

from test_v3_endpoint_integration import grounded_guard
from test_v3_verification_contract import verdict
from test_v3_boundary_height import Agent, parser_ns
from agent.route_guard import RouteEvidenceGuard
from agent.endpoint_guard import endpoint_kind
from target_identity import detector_matches_endpoint, endpoint_object_roles


def detector_agent(instruction, labels):
    agent, _ = grounded_guard(instruction)
    prototype = copy.deepcopy(agent.mapper.object_entities[0])
    agent.mapper.object_entities = [dict(copy.deepcopy(prototype), class_name=label,
        center=[.5, .5, 1.2], last_seen_observation_id=5) for label in labels]
    agent._get_agent_yaw = lambda: 0.
    agent._observed_target_candidates = MethodType(RouteEvidenceGuard._observed_target_candidates, agent)
    return agent


def parsed_claim(agent, entity_id=0):
    raw = verdict(agent.prepare())
    raw.update(target_entity_id=entity_id, target_view='left', target_bbox=[.1, .2, .4, .7])
    return parser_ns['robust_json_parse'](json.dumps(raw))


class TargetNounTests(unittest.TestCase):
    def test_failure_and_control_endpoint_clauses_keep_the_target_not_transit_objects(self):
        for text, category in (
                ('Stop after stepping through the doorway out of the microwave room.', 'door'),
                ('Stop before crossing through the doorway out of the kitchen.', 'door'),
                ('Stand next to the bathroom door.', 'door'),
                ('Stop in the doorway to the left of the white statue of two people.', 'door'),
                ('Stop at the bottom of the stairs.', 'stairs')):
            with self.subTest(text=text):
                self.assertEqual(endpoint_object_roles(text)['primary'], category)

    def test_action_direction_and_room_adjectives_do_not_select_furniture(self):
        agent = detector_agent('Stand next to the bathroom door.',
                               ['desk', 'nightstand', 'bathroom cabinet', 'door', 'sofa'])
        for text in ('Stand next to the bathroom door.', 'Go right and stand by the bathroom door.',
                     'Wait next to the bathroom doorway.', 'Stop in front of the bathroom door.'):
            with self.subTest(text=text):
                self.assertEqual([row['label'] for row in agent._observed_target_candidates(text)], ['door'])

    def test_aliases_and_compound_heads_are_preserved(self):
        positives = [('couch', 'Stand beside the sofa.'), ('fridge', 'Stop by the refrigerator.'),
                     ('sculpture', 'Wait beside the statue.'), ('staircase', 'Wait beside the stairs.'),
                     ('door frame', 'Stop in the doorway.'), ('nightstand', 'Stop beside the bedside table.'),
                     ('end table', 'Stop next to the nightstands.'), ('bedside tables', 'Wait by the end table.'),
                     ('dining table', 'Stop next to the table.'), ('coffee table', 'Wait by the table.'),
                     ('pool table', 'Stop next to the table.'), ('console table', 'Wait by the table.'),
                     ('countertop', 'Stop beside the counter.'), ('counter', 'Stop at the countertop.'),
                     ('stairs', 'Stand at the bottom of the stairs.'),
                     ('writing desk', 'Stand by the desk.'), ('armchair', 'Wait next to the chair.')]
        positives.append(('chandelier', 'Wait under the light.'))
        for label, text in positives:
            with self.subTest(label=label, text=text):
                self.assertTrue(detector_matches_endpoint(label, text))
        for label in ('bathroom cabinet', 'desk', 'nightstand', 'door mat', 'door handle', 'cabinet door'):
            self.assertFalse(detector_matches_endpoint(label, 'Stand beside the bathroom door.'))
        self.assertFalse(detector_matches_endpoint('table lamp', 'Stop at the table.'))
        self.assertFalse(detector_matches_endpoint('TV stand', 'Stop beside the television.'))
        self.assertFalse(detector_matches_endpoint('chair', 'Stand right there.'))
        for text in ('Stand by the wall beside the sofa.', 'Stand at the basket near the chair.',
                     'Wait by the sign in front of the door.', 'Stand next to the table cover.'):
            with self.subTest(text=text):
                self.assertIsNone(endpoint_object_roles(text)['primary'])

    def test_main_object_and_supporting_landmark_have_different_roles(self):
        text = 'Stop by the door to the left of the statue.'
        self.assertEqual(endpoint_object_roles(text), {'primary': 'door', 'reference': 'door'})
        self.assertFalse(detector_matches_endpoint('statue', text))
        self.assertFalse(detector_matches_endpoint('statue', text, 'reference'))
        text = 'Stop between the island and the counter.'
        self.assertEqual(endpoint_object_roles(text), {'primary': 'island', 'reference': 'counter'})
        self.assertTrue(detector_matches_endpoint('island', text))
        self.assertTrue(detector_matches_endpoint('counter', text, 'reference'))
        self.assertFalse(detector_matches_endpoint('counter', text))
        self.assertFalse(detector_matches_endpoint('island', text, 'reference'))
        self.assertEqual(endpoint_object_roles('Stop between the two chairs.'),
                         {'primary': 'chair', 'reference': 'chair'})
        self.assertEqual(endpoint_object_roles('Stop near the island between the sofa and table.'),
                         {'primary': 'island', 'reference': None})
        self.assertEqual(endpoint_kind('Stop near the island between the sofa and table.'), 'near_object')
        self.assertEqual(endpoint_object_roles('Stop at the door between the two paintings.'),
                         {'primary': 'door', 'reference': 'door'})
        self.assertEqual(endpoint_kind('Stop in the doorway between the two paintings.'), 'doorway')
        agent = detector_agent('Stop near the island between the sofa and table.', ['island', 'sofa', 'table'])
        self.assertEqual([row['label'] for row in agent.prepare()['entities']], ['island'])


class TargetBindingIntegrationTests(unittest.TestCase):
    def test_production_verifier_parses_structured_response_and_records_rejected_class(self):
        agent = detector_agent('Stand next to the bathroom door.', ['desk', 'door'])
        raw = parsed_claim(agent, 0)
        output_keys = {'outcome', 'failure_type', 'reason', 'confidence', 'relation_satisfied'}
        nested = {'reasoning': {k: v for k, v in raw.items() if k not in output_keys},
                  'output': {k: v for k, v in raw.items() if k in output_keys}}
        agent._get_forward_clearance_and_waypoints = Mock(return_value=(None, []))
        def prompt(*args, **kwargs):
            agent.prepare()
            return 'Current production candidate binding prepared.'
        agent._verification_prompt = prompt
        agent._get_verifier_response = Mock(return_value=json.dumps(nested))
        agent.instruction_obj.update_verifier = Mock()
        result = Agent._verify_current_subtask(agent, {}, None, [], None, obs=agent.curr_obs, actions={})
        self.assertFalse(result['outcome'])
        self.assertEqual(result['selected_detector_class_name'], 'desk')
        self.assertEqual(agent.navigation_trace[-1]['selected_detector_class_name'], 'desk')

    def test_production_candidates_parse_and_grounding_reject_desk_for_bathroom_door(self):
        agent = detector_agent('Stand next to the bathroom door.', ['desk', 'door'])
        raw = parsed_claim(agent, 0)
        self.assertEqual([row['id'] for row in agent._route_verification_context['entities']], [1])
        result = agent.evaluate(raw)
        self.assertFalse(result['outcome'])
        self.assertEqual(result['selected_detector_class_name'], 'desk')
        # ID leniency must never authorize a different class's mask/depth.
        agent.navigation_cfg['require_strict_target_id'] = False
        result = agent.evaluate(raw)
        self.assertFalse(result['outcome'])
        self.assertEqual(result['endpoint_grounding']['reason'], 'target_detector_class_mismatch')
        self.assertEqual(result['endpoint_grounding']['selected_detector_class_name'], 'desk')
        self.assertEqual(result['endpoint_grounding']['expected_object_class'], 'door')

    def test_poisoned_candidate_list_cannot_bypass_actual_detector_class_binding(self):
        agent = detector_agent('Stand next to the bathroom door.', ['desk', 'door'])
        raw = parsed_claim(agent, 0)
        agent._route_verification_context['entities'].append({'id': 0, 'label': 'door'})
        result = agent.evaluate(raw)
        self.assertFalse(result['outcome'])
        self.assertEqual(result['endpoint_grounding']['reason'], 'target_detector_class_mismatch')

    def test_matching_door_mask_remains_usable_and_records_class_provenance(self):
        agent = detector_agent('Stand next to the bathroom door.', ['desk', 'door'])
        result = agent.evaluate(parsed_claim(agent, 1))
        self.assertTrue(result['outcome'], result.get('reason'))
        self.assertEqual(result['endpoint_grounding']['source'], 'detector_mask')
        self.assertEqual(result['endpoint_grounding']['selected_detector_class_name'], 'door')
        self.assertEqual(result['endpoint_grounding']['expected_object_class'], 'door')
        self.assertEqual(result['endpoint_grounding']['evidence_role'], 'primary')

    def test_missing_candidate_uses_bbox_and_still_requires_changed_view(self):
        agent = detector_agent('Stand next to the bathroom door.', ['desk'])
        raw = parsed_claim(agent, None)
        self.assertEqual(agent._route_verification_context['entities'], [])
        result = agent.evaluate(raw)
        self.assertFalse(result['outcome'])
        self.assertEqual(result['endpoint_grounding']['source'], 'vlm_bbox')
        self.assertEqual(result['endpoint_grounding']['gate_reason'], 'changed_view_confirmation_required')
        self.assertIsNone(result['endpoint_grounding']['selected_detector_class_name'])
        self.assertTrue(result['endpoint_confirmation_required'])

    def test_between_regions_bind_two_distinct_correct_object_classes(self):
        agent = detector_agent('Walk toward the island. Stop between the island and counter.', ['island', 'counter'])
        # Opposite observed surfaces surround the agent, with independent masks.
        agent.curr_obs['back']['agent_state'].sensor_states['color_sensor'].rotation = (
            np.diag([-1., -1., 1.]) @ agent.curr_obs['agent_state'].sensor_states['color_sensor'].rotation)
        mask = np.zeros((100, 100), bool)
        mask[40:60, 40:60] = True
        agent.mapper.current_observation_masks = {'first': mask, 'second': mask.copy()}
        for index, view in enumerate(('front', 'back')):
            agent.mapper.object_entities[index]['observations'] = [
                {'view': view, 'bbox': [.4, .4, .6, .6], 'confidence': .9, 'observation_id': 5,
                 'mask_key': ('first', 'second')[index]}]
        raw = parsed_claim(agent, 0)
        raw.update(target_view='front', target_bbox=[.4, .4, .6, .6],
                   reference_region={'view': 'back', 'bbox': [.4, .4, .6, .6], 'entity_id': 1})
        self.assertEqual([row['label'] for row in agent._route_verification_context['entities']], ['island'])
        self.assertEqual([row['label'] for row in agent._route_verification_context['reference_entities']], ['counter'])
        result = agent.evaluate(raw)
        self.assertTrue(result['outcome'], result.get('reason'))
        self.assertEqual(result['endpoint_reference_grounding']['selected_detector_class_name'], 'counter')
        self.assertEqual(result['endpoint_reference_grounding']['evidence_role'], 'reference')
        agent.mapper.object_entities[1]['class_name'] = 'island'
        result = agent.evaluate(raw)
        self.assertFalse(result['outcome'])
        self.assertEqual(result['endpoint_reference_grounding']['reason'], 'target_detector_class_mismatch')

    def test_door_reference_cannot_borrow_a_statue_landmark_mask(self):
        agent = detector_agent('Stop in the doorway to the left of the statue.', ['door', 'statue'])
        raw = parsed_claim(agent, 0)
        raw['reference_region'] = {'view': 'left', 'bbox': [.1, .2, .4, .7], 'entity_id': 1,
                                   'role': 'door_jamb'}
        result = agent.evaluate(raw)
        self.assertFalse(result['outcome'])
        self.assertEqual(result['endpoint_reference_grounding']['reason'], 'target_detector_class_mismatch')
        self.assertEqual(result['endpoint_reference_grounding']['expected_object_class'], 'door')

    def test_v3_batch_regression_contracts_and_endpoint_clauses(self):
        from navigation_contracts import completion_conditions
        from navigation_control import stair_intent
        from structural_endpoints import stair_endpoint_side
        from target_identity import object_category

        conds = completion_conditions(
            'Exit the bedroom and go straight, go past the stairs on your left and go stand in the first doorway on the right.',
            'P1',
        )
        self.assertEqual(len(conds), 4)
        self.assertEqual(conds[-1]['kind'], 'arrival')
        self.assertFalse(conds[-1]['requires_translation'])
        self.assertEqual(endpoint_kind('Wait at the bottom of the steps next to the bar.'), 'stair_landing')
        self.assertFalse(stair_intent('Turn right and descend three steps.')['landing_required'])
        self.assertEqual(object_category('bar'), object_category('workbench'))

        guard = RouteEvidenceGuard()
        clause_1096 = guard._endpoint_clause(
            'Walk straight until you get to a dresser with a vase on top of it. Turn left and wait there.'
        )
        self.assertEqual(object_category(clause_1096), 'dresser')
        clause_531 = guard._endpoint_clause(
            'Walk up the stairs. Continue past the first landing, up the second set of stairs and stop.'
        )
        self.assertEqual(stair_endpoint_side(clause_531), 'top')


if __name__ == '__main__':
    unittest.main()


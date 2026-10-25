"""Endpoint depth gates exercised through the production route verifier."""
import copy
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from agent.endpoint_guard import doorway_position, between_position, endpoint_kind
from test_v3_verification_contract import Guard, record, verdict


def grounded_guard(instruction='Walk to the chair. Stop beside it.'):
    agent = Guard(instruction)
    agent.navigation_cfg = {'require_endpoint_grounding': True}
    agent.st['record'] = [record(1)]
    mapper = agent.mapper
    mapper.observation_id = 5
    mapper.camera_intrinsic = np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]])
    mapper.initial_position = np.zeros(3)
    mapper.translation_func = lambda value: np.asarray(value)
    mapper.rotation_func = lambda value: np.asarray(value)
    rotation = np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
    mapper.current_rotation = rotation.copy()
    state = NS(position=mapper.current_position.copy(),
               sensor_states={'color_sensor': NS(rotation=rotation.copy())})
    obs = {'depth_sensor': np.full((100, 100), 1.), 'agent_state': state}
    agent.curr_obs = {**obs, 'left': copy.deepcopy(obs), 'right': copy.deepcopy(obs), 'back': copy.deepcopy(obs)}
    mask = np.zeros((100, 100), bool)
    mask[20:70, 10:40] = True
    mapper.current_observation_masks = {'m': mask}
    mapper.object_entities = [{'class_name': 'chair', 'observations': [{'view': 'left', 'bbox': [.1, .2, .4, .7],
                               'confidence': .9, 'observation_id': 5, 'mask_key': 'm'}]}]
    agent.entities = [{'id': 0, 'label': 'chair', 'horizontal_distance': 1.0}]
    ctx = agent.prepare()
    raw = verdict(ctx)
    raw.update(target_entity_id=0, target_view='left', target_bbox=[.1, .2, .4, .7])
    return agent, raw


class EndpointIntegrationTests(unittest.TestCase):
    def _split_door_frames(self, role='door_jamb'):
        agent, raw = grounded_guard('Stop in the doorway to the left of the statue.')
        box = [.4, .4, .6, .6]
        mask = np.zeros((100, 100), bool)
        mask[40:60, 40:60] = True
        agent.mapper.current_observation_masks = {'first': mask, 'second': mask.copy()}
        agent.mapper.object_entities = []
        for index, (view, angle) in enumerate((('left', -np.pi/2), ('right', np.pi/2))):
            turn = np.array([[np.cos(angle), -np.sin(angle), 0.],
                             [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
            agent.curr_obs[view]['agent_state'].sensor_states['color_sensor'].rotation = (
                turn @ agent.mapper.current_rotation)
            agent.curr_obs[view]['depth_sensor'][:] = .7
            agent.mapper.object_entities.append({'class_name': 'door', 'observations': [
                {'view': view, 'bbox': box, 'confidence': .9, 'observation_id': 5,
                 'mask_key': ('first', 'second')[index]}]})
        agent.waypoint_routes = NS(to_habitat=lambda p: p, segment=lambda a, b: True,
                                  pf=NS(is_navigable=lambda p: True))
        raw.update(target_entity_id=0, target_view='left', target_bbox=box,
                   reference_region={'view': 'right', 'bbox': box, 'entity_id': 1})
        if role is not None:
            raw['reference_region']['role'] = role
        return agent, raw

    def _between_frames(self, reference_depth=1.0):
        agent, raw = grounded_guard('Walk toward the island. Stop between the island and counter.')
        raw.update(target_entity_id=None, target_view='front', target_bbox=[.4, .4, .6, .6],
                   reference_region={'view': 'back', 'bbox': [.4, .4, .6, .6], 'entity_id': None})
        # Independent RGB-depth camera poses face the two opposite surfaces.
        front = agent.curr_obs['agent_state'].sensor_states['color_sensor']
        back = agent.curr_obs['back']['agent_state'].sensor_states['color_sensor']
        back.rotation = np.diag([-1., -1., 1.]) @ front.rotation
        first = agent.evaluate(raw)
        self.assertFalse(first['outcome'])
        self.assertIs(first['endpoint_grounding']['geometry_verified'], True)
        self.assertTrue(first['endpoint_confirmation_required'])
        theta = np.deg2rad(12)
        turn = np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        agent.mapper.current_rotation = turn @ agent.mapper.current_rotation
        front.rotation = turn @ front.rotation
        back.rotation = turn @ back.rotation
        agent.mapper.observation_id += 1
        # Keep actual world-space planes fixed while the cameras turn. Changing
        # only the reference plane tests independent second-surface consistency.
        rows, cols = np.indices((100, 100))
        rays = np.stack(((cols-50)/100, (99-rows-50)/100, -np.ones_like(cols)), axis=-1)
        agent.curr_obs['depth_sensor'] = -1. / (rays @ front.rotation.T)[..., 1]
        agent.curr_obs['back']['depth_sensor'] = reference_depth / (rays @ back.rotation.T)[..., 1]
        second = agent.evaluate(raw)
        # The next verification prompt serializes its prior measured verdicts.
        json.dumps([first, second])
        return first, second

    def test_two_changed_view_between_regions_authorize_measured_endpoint(self):
        first, result = self._between_frames()
        self.assertTrue(result['outcome'], result.get('reason'))
        self.assertIs(result['endpoint_grounding']['confirmed_changed_view'], True)
        self.assertIs(result['endpoint_grounding']['accepted'], True)
        for key in ('endpoint_grounding', 'endpoint_reference_grounding'):
            self.assertEqual(result[key]['source'], 'vlm_bbox')
            self.assertNotEqual(first[key]['observation_id'], result[key]['observation_id'])

    def test_changed_view_with_displaced_reference_still_rejects_endpoint(self):
        _, result = self._between_frames(reference_depth=1.8)
        self.assertFalse(result['outcome'])
        self.assertIs(result['endpoint_grounding']['geometry_verified'], True)
        self.assertIs(result['endpoint_grounding']['confirmed_changed_view'], False)
        self.assertEqual(result['endpoint_grounding']['gate_reason'], 'changed_view_confirmation_required')
        self.assertTrue(result['endpoint_reference_grounding']['valid'])

    def test_current_side_mask_and_depth_can_authorize_stop(self):
        agent, raw = grounded_guard()
        result = agent.evaluate(raw)
        self.assertTrue(result['outcome'], result.get('reason'))
        self.assertEqual(result['endpoint_grounding']['source'], 'detector_mask')

    def test_near_view_center_does_not_override_far_target(self):
        agent, raw = grounded_guard()
        agent.curr_obs['left']['depth_sensor'][20:70, 10:40] = 4.
        agent.curr_obs['left']['depth_sensor'][35:65, 45:65] = .4
        result = agent.evaluate(raw)
        self.assertFalse(result['outcome'])
        self.assertGreater(result['endpoint_grounding']['distance_m'], 3.)

    def test_null_entity_and_proximity_prose_without_localization_rejected(self):
        agent, raw = grounded_guard()
        raw.update(target_entity_id=None, target_view=None, target_bbox=None)
        self.assertFalse(agent.evaluate(raw)['outcome'])

    def test_stale_entity_cannot_supply_current_depth(self):
        agent, raw = grounded_guard()
        agent.mapper.observation_id = 6
        result = agent.evaluate(raw)
        self.assertFalse(result['outcome'])
        self.assertIn('current', result['reason'])

    def test_entity_region_mismatch_is_rejected(self):
        agent, raw = grounded_guard()
        raw['target_bbox'] = [.7, .2, .9, .7]
        self.assertFalse(agent.evaluate(raw)['outcome'])

    def test_visual_only_bbox_requires_changed_view(self):
        agent, raw = grounded_guard()
        raw['target_entity_id'] = None
        result = agent.evaluate(raw)
        self.assertFalse(result['outcome'])
        self.assertTrue(result['endpoint_confirmation_required'])
        agent.mapper.observation_id += 1
        self.assertFalse(agent.evaluate(raw)['outcome'])
        theta = np.deg2rad(12)
        r = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        agent.mapper.current_rotation = r @ agent.mapper.current_rotation
        agent.curr_obs['left']['agent_state'].sensor_states['color_sensor'].rotation = agent.mapper.current_rotation
        agent.mapper.observation_id += 1
        result = agent.evaluate(raw)
        self.assertTrue(result['outcome'], result.get('reason'))

    def test_target_geometry_does_not_erase_incomplete_route(self):
        agent, raw = grounded_guard()
        raw['condition_checks'][0].update(status='unknown', steps=[])
        self.assertFalse(agent.evaluate(raw)['outcome'])

    def test_far_endpoint_approach_uses_only_a_same_floor_legal_waypoint(self):
        agent, _ = grounded_guard()
        verification = {'endpoint_grounding': {'valid': True, 'world_center': [5, 0, 1.2],
                                               'gate_reason': 'target_surface_too_far'}}
        actions = {'WP_A': {'type': 'waypoint', 'target_point_world': [2, 0, 0]},
                   'WP_B': {'type': 'waypoint', 'target_point_world': [4, 0, 3]},
                   'WP_C': {'type': 'waypoint', 'target_point_world': [4, 0, 0], 'rejected': True}}
        choice = agent._endpoint_approach_action(verification, actions)
        self.assertEqual(choice['action_id'], 'WP_A')
        self.assertTrue(choice['is_verifier_correction'])
        verification['endpoint_grounding']['gate_reason'] = 'target_depth_not_grounded'
        self.assertIsNone(agent._endpoint_approach_action(verification, actions))

    def test_opening_background_is_not_a_threshold_witness(self):
        packet = {'frame_support': {'left': {'world_center': [-.5, 3, 0]},
                                    'right': {'world_center': [.5, 3, 0]}},
                  'opening_depth_contrast_m': 4}
        self.assertFalse(doorway_position(packet, [0, 0, 0]))
        self.assertTrue(doorway_position(packet, [0, 2.8, 0]))
        self.assertFalse(doorway_position(packet, [2, 2.8, 0]))

    def test_jambs_in_opposite_views_can_ground_a_threshold(self):
        agent, raw = grounded_guard('Walk to the doorway. Stop in the doorway.')
        agent.waypoint_routes = NS(to_habitat=lambda p: p, segment=lambda a, b: True,
                                  pf=NS(is_navigable=lambda p: True))
        def region(r, kind='near_object'):
            return {'valid': True, 'source': 'detector_mask', 'confidence': .9,
                    'view': r['view'], 'endpoint_kind': kind, 'observation_id': 5,
                    'target_id': 0, 'depth_m': .7, 'distance_m': .7,
                    'world_center': [-.2 if r['view'] == 'left' else 1.2, 0, 1.2], 'floor': 0}
        agent._ground_region = region
        raw['reference_region'] = {'view': 'right', 'bbox': [.1, .2, .4, .7], 'entity_id': 0,
                                   'role': 'door_jamb'}
        self.assertTrue(agent.evaluate(raw)['outcome'])
        agent.waypoint_routes.segment = lambda a, b: False
        self.assertFalse(agent.evaluate(raw)['outcome'])

    def test_explicit_jamb_role_binds_two_independently_measured_current_surfaces(self):
        agent, raw = self._split_door_frames()
        result = agent.evaluate(raw)
        self.assertTrue(result['outcome'], result.get('reason'))
        frame = result['endpoint_grounding']['frame_support']
        self.assertAlmostEqual(np.linalg.norm(np.asarray(frame['left']['world_center'])[:2]
                                             -np.asarray(frame['right']['world_center'])[:2]), 1.4, places=3)
        self.assertEqual(result['endpoint_reference_grounding']['selected_detector_class_name'], 'door')
        agent.mapper.observation_id += 1
        self.assertFalse(agent.evaluate(raw)['outcome'])

    def test_landmark_unknown_and_missing_roles_cannot_turn_a_single_jamb_into_a_doorway(self):
        for role in ('spatial_landmark', 'unknown', 'between_object', None, True):
            with self.subTest(role=role):
                agent, raw = self._split_door_frames(role)
                # Even favourable paired depths and traversability cannot make
                # an untyped reference count as the second doorway frame.
                result = agent.evaluate(raw)
                self.assertFalse(result['outcome'])
                self.assertNotIn('endpoint_reference_grounding', result)
                packet = result['endpoint_grounding']
                self.assertEqual(packet['reference_region_ignored_reason'],
                                 'doorway_reference_requires_door_jamb_role')
                self.assertEqual(packet['sampling'], 'door_frame_borders')
                self.assertFalse(packet['geometry_verified'])

    def test_ignored_landmark_keeps_strict_full_door_border_sampling(self):
        agent, raw = grounded_guard('Stop in the doorway to the left of the statue.')
        raw.update(target_entity_id=None, target_view='front', target_bbox=[.05, .1, .95, .9],
                   reference_region={'view': 'right', 'bbox': [.4, .4, .6, .6],
                                     'entity_id': None, 'role': 'spatial_landmark'})
        # Both real frame borders are 0.7m ahead; the opening sees 3m through.
        depth = agent.curr_obs['depth_sensor']
        depth[:] = 3.
        depth[:, :22] = depth[:, 78:] = .7
        first = agent.evaluate(raw)
        self.assertFalse(first['outcome'])  # Bbox still needs a changed view.
        packet = first['endpoint_grounding']
        self.assertTrue(packet['geometry_verified'])
        self.assertEqual(packet['sampling'], 'door_frame_borders')
        self.assertNotIn('endpoint_reference_grounding', first)
        agent.curr_obs['depth_sensor'][:, 78:] = np.nan
        rejected = agent.evaluate(raw)
        self.assertFalse(rejected['outcome'])
        self.assertEqual(rejected['endpoint_grounding']['reason'], 'insufficient_door_frame_support')

    def test_near_an_opening_does_not_require_standing_inside_it(self):
        self.assertEqual(endpoint_kind('Stand next to the archway.'), 'near_object')
        self.assertEqual(endpoint_kind('Stop beside the doorway.'), 'near_object')
        self.assertEqual(endpoint_kind('Stop in the doorway.'), 'doorway')

    def test_between_requires_agent_between_both_observed_surfaces(self):
        a = {'world_center': [-.7, 0, 1], 'distance_m': .7}
        b = {'world_center': [.7, 0, 1], 'distance_m': .7}
        self.assertTrue(between_position(a, b, [0, 0, 0]))
        self.assertFalse(between_position(a, b, [1.5, 0, 0]))
        a['target_id'] = b['target_id'] = 3
        self.assertFalse(between_position(a, b, [0, 0, 0]))


if __name__ == '__main__':
    unittest.main()


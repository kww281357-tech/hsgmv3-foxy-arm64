"""Synthetic sensor tests; not a detector or real rendering claim."""
import unittest
import math
import numpy as np
from blue_help.habitat_adapter import AdapterConfig, HabitatNavigationEnvironment
from test_habitat_adapter import env_and_sim, episode
from blue_help.contracts import DummySpec

class GameVisibilityTests(unittest.TestCase):
    def test_nonblue_current_instance_is_visible_but_missing_instance_is_not(self):
        env, sim = env_and_sim(AdapterConfig(helper_visibility_mode='semantic_instance'))
        try:
            sim.rgb[:] = [200,120,100,255]
            row = env.reset(episode(),[DummySpec('game',(0,0,-.8),0)]).dummy_candidates[0]
            self.assertTrue(row['visible'])
            self.assertTrue(row['spatial_eligible'])
            sim.semantic[:] = 0
            self.assertFalse(env.observe().dummy_candidates[0]['visible'])
            sim.semantic[0,0] = 10000
            self.assertFalse(env.observe().dummy_candidates[0]['visible'])
        finally: env.close()

    def test_instance_visibility_does_not_override_wall_los(self):
        env, sim = env_and_sim(AdapterConfig(helper_visibility_mode='semantic_instance'))
        try:
            sim.rgb[:] = [200,120,100,255]
            sim.ray_hits = []
            row = env.reset(episode(),[DummySpec('game',(0,0,-.8),0)]).dummy_candidates[0]
            self.assertTrue(row['visible'])
            self.assertFalse(row['spatial_eligible'])
        finally: env.close()

    def test_game_front_faces_fixed_anchor_without_moving_foot(self):
        from blue_help.helper_assets import orient_game_helper_to_anchor
        dummy=DummySpec('game',(0,0,0),0)
        placed=orient_game_helper_to_anchor(dummy,(1,0,-1))
        self.assertEqual(placed.ground_position,dummy.ground_position)
        self.assertAlmostEqual(placed.yaw_rad,3*math.pi/4)
        with self.assertRaises(ValueError): orient_game_helper_to_anchor(dummy,(0,0,0))

    def test_unknown_visibility_mode_rejected(self):
        with self.assertRaises(ValueError): AdapterConfig(helper_visibility_mode='registry_always_visible')

if __name__ == '__main__': unittest.main()

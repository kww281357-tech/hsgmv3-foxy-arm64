import unittest
import numpy as np
from blue_help.habitat_adapter import AdapterConfig
from blue_help.contracts import DummySpec
from test_habitat_adapter import env_and_sim,episode

class SlidingProfileTests(unittest.TestCase):
    def setUp(self):
        self.env,self.sim=env_and_sim(AdapterConfig(allow_sliding=True))
        self.sim.pathfinder.try_step=lambda a,b:np.asarray(b)
        self.env.reset(episode(),[])
    def tearDown(self):self.env.close()
    def test_actual_substeps_returned_and_profile_explicit(self):
        path=self.env.find_safe_path((0,0,0),(0,0,-1))
        self.assertGreater(len(path),2)
        self.assertEqual(self.env.last_path_check['motion_profile'],'sliding')
        self.assertAlmostEqual(path[-1][2],-1)
    def test_no_stall_or_unbounded_jump_allowed(self):
        for bad in (lambda a,b:np.asarray(a),lambda a,b:np.asarray(a)+10*(np.asarray(b)-a)):
            self.sim.pathfinder.try_step=bad
            self.assertIsNone(self.env.find_safe_path((0,0,0),(0,0,-1)))
    def test_same_human_sweep_rejects_path(self):
        self.env.reset(episode(),[DummySpec('blue',(0,0,-.8),0)])
        self.assertIsNone(self.env.find_safe_path((0,0,0),(0,0,-2)))
        self.assertEqual(self.env.last_path_check['reason'],'executed_substep_body_crossing')
    def test_vertical_jump_rejected(self):
        self.sim.pathfinder.try_step=lambda a,b:np.asarray(b)+[0,2,0]
        self.assertIsNone(self.env.find_safe_path((0,0,0),(0,0,-1)))
    def test_strict_profile_does_not_silently_fallback(self):
        strict,sim=env_and_sim()
        try:
            strict.reset(episode(),[]);sim.pathfinder.stuck=True
            sim.pathfinder.try_step=lambda a,b:np.asarray(b)
            self.assertIsNone(strict.find_safe_path((0,0,0),(0,0,-1)))
        finally:strict.close()
    def test_missing_matching_query_fails_closed(self):
        self.sim.pathfinder.try_step=None
        self.assertIsNone(self.env.find_safe_path((0,0,0),(0,0,-1)))
    def test_invalid_allow_sliding_type(self):
        with self.assertRaises(ValueError):AdapterConfig(allow_sliding='yes')

class SlidingZeroLengthTests(unittest.TestCase):
    def test_zero_length_query_inside_body_rejected(self):
        env,sim=env_and_sim(AdapterConfig(allow_sliding=True))
        sim.pathfinder.try_step=lambda a,b:np.asarray(b)
        try:
            env.reset(episode(),[DummySpec('blue',(0,0,-.8),0)])
            self.assertIsNone(env.find_safe_path((0,0,-.8),(0,0,-.8)))
        finally:env.close()

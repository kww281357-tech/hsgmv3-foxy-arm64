import unittest
from blue_help.policy import PolicyContext
from blue_help.smoke import ScriptedTextFollower

class SmokePolicyTests(unittest.TestCase):
    def context(self,distance=.1):
        return PolicyContext({'active_offer':None,'last_delivered_answer':f'From your vehicle heading when you asked: go forward about {distance} meters. Continue normal navigation afterward.',
            'observation_step':1,'executed_history':[],'legal_candidate_ids':['move_forward','stop','wait'],
            'action_specs':{'move_forward':{'requested_distance_m':.25},'turn_left':{'turn_degrees':15}}},'unused')
    def test_short_first_leg_does_not_overshoot(self):
        p=ScriptedTextFollower();p.reset()
        self.assertEqual(p.decide(self.context()).action,'stop')
    def test_complete_primitive_within_first_leg(self):
        p=ScriptedTextFollower();p.reset()
        self.assertEqual(p.decide(self.context(.5)).action,'move_forward')
    def test_unverified_step_size_stops(self):
        p=ScriptedTextFollower();p.reset();ctx=self.context(.5);ctx.public['action_specs']={}
        self.assertEqual(p.decide(ctx).action,'stop')

class RotationTests(unittest.TestCase):
    def test_36degree_answer_uses_two_actual_15degree_steps(self):
        p=ScriptedTextFollower();p.reset()
        context=SmokePolicyTests().context(1)
        context.public['last_delivered_answer']='From your vehicle heading when you asked: turn right about 36 degrees, then go forward about 5.2 meters; then turn left about 2 degrees.'
        context.public['legal_candidate_ids']+=['turn_right','turn_left']
        self.assertEqual(p.decide(context).action,'turn_right')
        self.assertEqual(p.decide(context).action,'turn_right')
        self.assertEqual(p.decide(context).action,'move_forward')
